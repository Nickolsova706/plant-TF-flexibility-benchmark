"""
OHE + DNA Flexibility Based TF Binding Site Classification (Run 3)
====================================================================
Target: ProLiant DL380 Gen11 — Xeon Gold 6530 (128 threads), NVIDIA L40S, 256GB RAM

Models: RandomForest (CPU) + XGBoost (GPU)
Features: OHE (400) + 5 flexibility descriptors (~493) = ~893 total
Reference: Dey, Yella & Kumar (2026) — DNA conformational flexibility descriptors

5 flexibility descriptors:
  Trinucleotide (2): DNaseI sensitivity, NPP (Nucleosome Positioning Preference)
  Dinucleotide  (3): Twist dispersion, Twist-roll-x displacement (trx), Stiffness

Same CV folds (random_state=42) as Run 1 and Run 2 for fair comparison.

=== MODIFICATION LOG ===
[2026-04-20] v1.0 — Initial script. Built from Run 1/Run 2 architecture with
                     DNAflexpy integration from final_flexpy.py sample.
                     All optimizations from Run 1 & 2 baked in from day one:
                     - Resume logic (skip TFs with existing model_comparison.csv)
                     - Load balancing (interleave TFs by FASTA size)
                     - Bulletproof SHAP (_extract_shap_values for list/3D/2D)
                     - check_additivity=False, interaction_index=None
                     - All plots in try/except with plt.close('all')
                     - Model saving (joblib, compress=3)
                     - tqdm + as_completed progress bar
                     - No lag features (matches project plan)
                     - No StandardScaler (tree-based models don't need it)
                     - No RC augmentation (FIMO strand-aware instead)
                     MAX_WORKERS=8, RF_INNER_JOBS=12, FLEX_THREADS=2
                     8×12=96 threads during RF; 8×2=16 during flex extraction.
========================
"""

import os, gc, time, logging, warnings, sys
from tqdm import tqdm
import joblib
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Bio import SeqIO
from Bio.Seq import reverse_complement
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score, confusion_matrix,
    f1_score, precision_score, recall_score, matthews_corrcoef,
    roc_curve, precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from concurrent.futures import ProcessPoolExecutor, as_completed
import xgboost as xgb
import shap

warnings.filterwarnings("ignore", category=FutureWarning)

# --- 1. SETTINGS ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

FASTA_BASE = os.environ.get("TF_FASTA_BASE")
FIMO_BASE  = os.environ.get("TF_FIMO_BASE")
OUTPUT_DIR = os.environ.get("OHE_FLEXIBILITY_CLASSIFICATION_OUTPUT", os.path.join(PROJECT_ROOT, "results", "OHE_flexibility_classification"))
SEQ_LEN    = 100

# Parallelism: 8 workers × 12 inner = 96 threads (RF training)
#              8 workers × 2 flex threads = 16 threads (flex extraction)
MAX_WORKERS   = 10
RF_INNER_JOBS = 12
FLEX_THREADS  = 2

# DNAflexpy library fix — YAML lookup table path
# Adjust this path if DNAflexpy is installed elsewhere
DNAFLEXPY_YAML = os.environ.get("DNAFLEXPY_YAML")

# Required external input paths
if not FASTA_BASE:
    raise RuntimeError("Set TF_FASTA_BASE to the location of the input FASTA directory.")
if not FIMO_BASE:
    raise RuntimeError("Set TF_FIMO_BASE to the location of the FIMO results directory.")
if not DNAFLEXPY_YAML:
    raise RuntimeError("Set DNAFLEXPY_YAML to the location of the DNAflexpy lookupNEW.yaml file.")

# 5 flexibility descriptors from Dey, Yella & Kumar (2026)
FLEX_FEATURES = ["DNaseI", "NPP", "twistDisp", "trx", "stiffness"]

RF_PARAM_GRID = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
}

XGB_BASE_PARAMS = dict(
    tree_method      = "gpu_hist",
    gpu_id           = 0,
    objective        = "binary:logistic",
    eval_metric      = "aucpr",
    subsample        = 0.8,
    colsample_bytree = 0.8,
    random_state     = 42,
    verbosity        = 0,
)

XGB_PARAM_GRID = [
    {'max_depth': d, 'n_estimators': n, 'learning_rate': lr}
    for d in [4, 6, 8]
    for n in [200, 400]
    for lr in [0.05, 0.1]
]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("OHE_flexibility_classification.log"), logging.StreamHandler()]
)

# --- 2. DNAFLEXPY LIBRARY FIX ---
# DNAflexpy has a hardcoded path issue for its lookup table.
# This fix patches the load function before any worker uses it.

import DNAflexpy.core as df_core
import DNAflexpy.utils as df_utils

def _manual_yaml_load():
    with open(DNAFLEXPY_YAML, 'r') as f:
        return yaml.safe_load(f)

df_utils.load_feature_data = _manual_yaml_load
df_core.load_feature_data = _manual_yaml_load


# --- 3. LOAD BALANCING ---

def get_tf_dataset_size(tf):
    """Get total FASTA file size in bytes for a TF (positives + negatives)."""
    pos = os.path.join(FASTA_BASE, tf, f"{tf}_positives.fasta")
    neg = os.path.join(FASTA_BASE, tf, f"{tf}_negatives.fasta")
    size = 0
    if os.path.exists(pos):
        size += os.path.getsize(pos)
    if os.path.exists(neg):
        size += os.path.getsize(neg)
    return size


def interleave_by_size(tf_list):
    """
    Sort TFs by dataset size and interleave: largest, smallest, 2nd largest,
    2nd smallest, etc. Prevents OOM from simultaneous large-TF processing.
    """
    sized = [(tf, get_tf_dataset_size(tf)) for tf in tf_list]
    sized.sort(key=lambda x: x[1], reverse=True)

    interleaved = []
    left, right = 0, len(sized) - 1
    toggle = True
    while left <= right:
        if toggle:
            interleaved.append(sized[left])
            left += 1
        else:
            interleaved.append(sized[right])
            right -= 1
        toggle = not toggle

    return interleaved


# --- 4. FLEXIBILITY FEATURE EXTRACTION ---

def extract_flex_features(sequences, seq_ids):
    """
    Extract 5 flexibility descriptors for a list of sequences using DNAflexpy.
    Writes sequences to a temp FASTA, runs DNAflexpyMP per feature, returns
    concatenated numpy array.

    Returns: (flex_matrix, flex_col_names)
        flex_matrix: np.ndarray of shape (n_sequences, n_flex_features)
        flex_col_names: list of feature names (e.g., Flex_pos3_DNaseI)
    """
    import tempfile
    from Bio.SeqRecord import SeqRecord
    from Bio.Seq import Seq

    # Write strand-corrected sequences to temp FASTA
    tmp_path = tempfile.mktemp(suffix=".fasta", prefix=f"flex_{os.getpid()}_")
    records = []
    for sid, seq in zip(seq_ids, sequences):
        records.append(SeqRecord(Seq(seq), id=sid, description=""))
    SeqIO.write(records, tmp_path, "fasta")

    try:
        flex_arrays = []
        flex_names = []

        for feat_name in FLEX_FEATURES:
            df_feat = df_core.DNAflexpyMP(
                input_file=tmp_path,
                window_size=0,
                feature=feat_name,
                threads=FLEX_THREADS,
                outfile=None,
            )
            # Drop sequence ID column (first column)
            vals = df_feat.iloc[:, 1:].values.astype(np.float32)
            n_cols = vals.shape[1]
            flex_arrays.append(vals)

            # Build feature names
            for col_idx in range(n_cols):
                flex_names.append(f"Flex_pos{col_idx+1}_{feat_name}")

        flex_matrix = np.hstack(flex_arrays)
        return flex_matrix, flex_names

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# --- 5. FEATURE VECTOR (OHE + Flexibility) ---
_OHE_MAP  = np.eye(4, dtype=np.float32)
_BASE2INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
_BASES    = ['A', 'C', 'G', 'T']

def get_ohe_features(seq, strand):
    """Convert a single sequence to OHE feature vector (400 features)."""
    if strand == "-":
        seq = str(reverse_complement(seq))
    seq = seq.upper()[:SEQ_LEN].ljust(SEQ_LEN, 'N')
    indices = np.array([_BASE2INT.get(b, -1) for b in seq], dtype=np.int8)
    ohe = np.zeros((SEQ_LEN, 4), dtype=np.float32)
    valid = indices >= 0
    ohe[valid] = _OHE_MAP[indices[valid]]
    return ohe.flatten()


def build_ohe_feature_names():
    names = []
    for pos in range(SEQ_LEN):
        for base in _BASES:
            names.append(f"OHE_pos{pos+1}_{base}")
    return names


def get_feature_group(name):
    """Classify feature into group for contribution analysis."""
    if name.startswith("OHE_"):
        return "OHE"
    elif name.startswith("Flex_"):
        parts = name.split("_", 2)
        return "Flex_" + parts[2] if len(parts) > 2 else "Flex"
    return "Unknown"


# --- 6. EVALUATION ---

def compute_fold_metrics(y_true, y_pred, y_prob, fold_idx):
    auprc = average_precision_score(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)
    f1    = f1_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    mcc   = matthews_corrcoef(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'Fold': fold_idx, 'AUPRC': auprc, 'AUROC': auroc,
        'F1': f1, 'Precision': prec, 'Recall': rec, 'MCC': mcc,
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
    }


# --- 7. MODEL TRAINING ---

def train_rf_nested_cv(X, y, outer_cv, inner_cv):
    metrics, curves = [], []
    best_model, best_score = None, -1
    for f_idx, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y)):
        grid = GridSearchCV(
            RandomForestClassifier(class_weight='balanced', random_state=42),
            RF_PARAM_GRID, cv=inner_cv, scoring='average_precision',
            n_jobs=RF_INNER_JOBS,
        )
        grid.fit(X[tr_idx], y[tr_idx])
        model = grid.best_estimator_
        probs = model.predict_proba(X[te_idx])[:, 1]
        preds = model.predict(X[te_idx])
        m = compute_fold_metrics(y[te_idx], preds, probs, f_idx + 1)
        metrics.append(m)
        fpr, tpr, _ = roc_curve(y[te_idx], probs)
        prec_c, rec_c, _ = precision_recall_curve(y[te_idx], probs)
        curves.append({'fpr': fpr, 'tpr': tpr, 'precision': prec_c, 'recall': rec_c})
        if m['AUPRC'] > best_score:
            best_score = m['AUPRC']
            best_model = model
    return metrics, best_model, best_score, curves


def train_xgb_nested_cv(X, y, outer_cv, inner_cv, spw):
    metrics, curves = [], []
    best_model, best_score = None, -1
    for f_idx, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]
        best_inner_score = -1
        best_params = XGB_PARAM_GRID[0]
        for params in XGB_PARAM_GRID:
            inner_scores = []
            for itr, ite in inner_cv.split(X_tr, y_tr):
                mdl = xgb.XGBClassifier(
                    **{**XGB_BASE_PARAMS, **params},
                    scale_pos_weight=spw, early_stopping_rounds=20,
                )
                mdl.fit(X_tr[itr], y_tr[itr],
                        eval_set=[(X_tr[ite], y_tr[ite])], verbose=False)
                p = mdl.predict_proba(X_tr[ite])[:, 1]
                inner_scores.append(average_precision_score(y_tr[ite], p))
            mean_s = np.mean(inner_scores)
            if mean_s > best_inner_score:
                best_inner_score = mean_s
                best_params = params
        mdl = xgb.XGBClassifier(
            **{**XGB_BASE_PARAMS, **best_params},
            scale_pos_weight=spw, early_stopping_rounds=20,
        )
        Xf, Xv, yf, yv = train_test_split(
            X_tr, y_tr, test_size=0.1, stratify=y_tr, random_state=42
        )
        mdl.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)
        probs = mdl.predict_proba(X_te)[:, 1]
        preds = mdl.predict(X_te)
        m = compute_fold_metrics(y_te, preds, probs, f_idx + 1)
        metrics.append(m)
        fpr, tpr, _ = roc_curve(y_te, probs)
        prec_c, rec_c, _ = precision_recall_curve(y_te, probs)
        curves.append({'fpr': fpr, 'tpr': tpr, 'precision': prec_c, 'recall': rec_c})
        if m['AUPRC'] > best_score:
            best_score = m['AUPRC']
            best_model = mdl
    return metrics, best_model, best_score, curves


# --- 8. OUTPUT FUNCTIONS ---

def plot_curves(curves, metrics, tf_out, tf, tag):
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        mean_fpr = np.linspace(0, 1, 200)
        tprs = [np.interp(mean_fpr, c['fpr'], c['tpr']) for c in curves]
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr  = np.std(tprs, axis=0)
        mean_auroc = np.mean([m['AUROC'] for m in metrics])
        axes[0].plot(mean_fpr, mean_tpr, lw=2, label=f'Mean AUROC = {mean_auroc:.3f}')
        axes[0].fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.2)
        axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title(f'{tf} \u2014 {tag} ROC Curve')
        axes[0].legend(loc='lower right')
        mean_rec = np.linspace(0, 1, 200)
        precs = []
        for c in curves:
            sort_idx = np.argsort(c['recall'])
            precs.append(np.interp(mean_rec, c['recall'][sort_idx], c['precision'][sort_idx]))
        mean_prec = np.mean(precs, axis=0)
        std_prec  = np.std(precs, axis=0)
        mean_auprc = np.mean([m['AUPRC'] for m in metrics])
        axes[1].plot(mean_rec, mean_prec, lw=2, label=f'Mean AUPRC = {mean_auprc:.3f}')
        axes[1].fill_between(mean_rec, mean_prec - std_prec, mean_prec + std_prec, alpha=0.2)
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title(f'{tf} \u2014 {tag} PR Curve')
        axes[1].legend(loc='lower left')
        plt.tight_layout()
        plt.savefig(os.path.join(tf_out, f"{tf}_{tag}_ROC_PR_curves.png"),
                    bbox_inches='tight', dpi=150)
        plt.close()
    except Exception as e:
        logging.warning(f"{tf} {tag}: ROC/PR curve plot failed ({e})")
        plt.close('all')


def _extract_shap_values(shap_vals):
    """Bulletproof SHAP value extraction: handles list, 3D, 2D formats."""
    if isinstance(shap_vals, list):
        return shap_vals[1]
    if isinstance(shap_vals, np.ndarray):
        if shap_vals.ndim == 3:
            return shap_vals[:, :, 1]
        if shap_vals.ndim == 2:
            return shap_vals
    return None


def save_model_outputs(tf_out, tf, model, X, feature_names, tag, n_ohe_features):
    """SHAP plots, top features, group contributions, position importance."""

    # --- SHAP computation ---
    sv = None
    shap_sample = X[:min(len(X), 200)]
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(shap_sample, check_additivity=False)
        sv = _extract_shap_values(shap_vals)
        if sv is not None and sv.ndim != 2:
            logging.warning(f"{tf} {tag}: SHAP shape {sv.shape} after extraction, skipping")
            sv = None
    except Exception as e:
        logging.warning(f"{tf} {tag}: SHAP computation failed ({e})")
        sv = None

    # SHAP summary plot
    if sv is not None:
        try:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(sv, shap_sample, feature_names=feature_names,
                              show=False, max_display=20)
            plt.title(f"{tf} \u2014 {tag}")
            plt.savefig(os.path.join(tf_out, f"{tf}_{tag}_shap_summary.png"),
                        bbox_inches='tight', dpi=150)
            plt.close()
        except Exception as e:
            logging.warning(f"{tf} {tag}: SHAP summary plot failed ({e})")
            plt.close('all')

    # Top 20 features
    imp = model.feature_importances_
    top_idx = np.argsort(imp)[-20:][::-1]
    top_df = pd.DataFrame({
        'Rank': range(1, 21),
        'Feature_Index': top_idx,
        'Feature_Name': [feature_names[i] for i in top_idx],
        'Importance': imp[top_idx],
    })
    top_df.to_csv(os.path.join(tf_out, f"{tf}_{tag}_top20_features.csv"), index=False)

    # Coarse group contribution (OHE vs Flex)
    group_imp = {}
    for i, importance in enumerate(imp):
        grp = get_feature_group(feature_names[i])
        base_grp = grp.split("_")[0]  # OHE or Flex
        if base_grp not in group_imp:
            group_imp[base_grp] = 0.0
        group_imp[base_grp] += importance

    # Fine group contribution (per flexibility descriptor)
    fine_group_imp = {}
    for i, importance in enumerate(imp):
        grp = get_feature_group(feature_names[i])
        if grp not in fine_group_imp:
            fine_group_imp[grp] = 0.0
        fine_group_imp[grp] += importance

    grp_df = pd.DataFrame([
        {'Group': k, 'Total_Importance': v}
        for k, v in sorted(group_imp.items(), key=lambda x: -x[1])
    ])
    grp_df['Fraction'] = grp_df['Total_Importance'] / grp_df['Total_Importance'].sum()
    grp_df.to_csv(os.path.join(tf_out, f"{tf}_{tag}_group_contribution_coarse.csv"), index=False)

    fine_df = pd.DataFrame([
        {'Group': k, 'Total_Importance': v}
        for k, v in sorted(fine_group_imp.items(), key=lambda x: -x[1])
    ])
    fine_df['Fraction'] = fine_df['Total_Importance'] / fine_df['Total_Importance'].sum()
    fine_df.to_csv(os.path.join(tf_out, f"{tf}_{tag}_group_contribution_fine.csv"), index=False)

    # Group contribution bar plot (2 panels: coarse + fine)
    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes[0].barh(grp_df['Group'], grp_df['Fraction'])
        axes[0].set_xlabel('Fraction of Total Importance')
        axes[0].set_title(f'{tf} \u2014 {tag} Coarse Groups (OHE vs Flex)')
        axes[0].invert_yaxis()
        fine_top = fine_df.head(10)
        axes[1].barh(fine_top['Group'], fine_top['Fraction'])
        axes[1].set_xlabel('Fraction of Total Importance')
        axes[1].set_title(f'{tf} \u2014 {tag} Per Descriptor')
        axes[1].invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(tf_out, f"{tf}_{tag}_group_contribution.png"),
                    bbox_inches='tight', dpi=150)
        plt.close()
    except Exception as e:
        logging.warning(f"{tf} {tag}: group contribution plot failed ({e})")
        plt.close('all')

    # SHAP dependence plots for top 5 features
    if sv is not None:
        try:
            fig, axes = plt.subplots(1, 5, figsize=(25, 5))
            for rank, feat_idx in enumerate(top_idx[:5]):
                try:
                    shap.dependence_plot(feat_idx, sv, shap_sample,
                                         feature_names=feature_names,
                                         interaction_index=None,
                                         ax=axes[rank], show=False)
                except Exception:
                    axes[rank].set_title(f"Feature {feat_idx}\n(plot failed)")
            plt.suptitle(f"{tf} \u2014 {tag} SHAP Dependence (Top 5)", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(tf_out, f"{tf}_{tag}_shap_dependence_top5.png"),
                        bbox_inches='tight', dpi=150)
            plt.close()
        except Exception as e:
            logging.warning(f"{tf} {tag}: dependence plot failed ({e})")
            plt.close('all')

    # Position importance (OHE block only — clean 100-position mapping)
    try:
        ohe_imp = imp[:n_ohe_features]
        pos_imp = ohe_imp.reshape(SEQ_LEN, 4).sum(axis=1)
        plt.figure(figsize=(14, 4))
        plt.bar(range(1, SEQ_LEN + 1), pos_imp, width=1.0, edgecolor='none')
        plt.xlabel('Position (bp)')
        plt.ylabel('Summed OHE Importance')
        plt.title(f'{tf} \u2014 {tag} Sequence Importance by Position')
        plt.tight_layout()
        plt.savefig(os.path.join(tf_out, f"{tf}_{tag}_position_importance.png"),
                    bbox_inches='tight', dpi=150)
        plt.close()
    except Exception as e:
        logging.warning(f"{tf} {tag}: position importance plot failed ({e})")
        plt.close('all')


def save_prediction_scores(tf_out, tf, model, X, y, seq_ids, tag):
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    score_df = pd.DataFrame({
        'Sequence_ID': seq_ids,
        'True_Label': y,
        'Predicted_Label': preds,
        'Binding_Probability': probs,
    })
    score_df.to_csv(os.path.join(tf_out, f"{tf}_{tag}_prediction_scores.csv"), index=False)


# --- 9. CORE TF PROCESSING ---

def process_tf(tf_args):
    tf, idx, total_tfs, global_start = tf_args
    tf_start = time.time()

    try:
        tf_out = os.path.join(OUTPUT_DIR, tf)
        os.makedirs(tf_out, exist_ok=True)

        fimo_file = os.path.join(FIMO_BASE, tf, "fimo.tsv")
        pos_fasta = os.path.join(FASTA_BASE, tf, f"{tf}_positives.fasta")
        neg_fasta = os.path.join(FASTA_BASE, tf, f"{tf}_negatives.fasta")

        if not os.path.exists(pos_fasta):
            return f"SKIP: {tf} (no positives fasta)"

        # --- Strand map from FIMO ---
        strand_map = {}
        if os.path.exists(fimo_file):
            fimo = pd.read_csv(fimo_file, sep='\t', comment='#').dropna(subset=['sequence_name'])
            strand_map = dict(zip(fimo.sequence_name, fimo.strand))

        # --- Read sequences and apply strand correction ---
        sequences, ohe_list, y_list, seq_ids = [], [], [], []
        for rec in SeqIO.parse(pos_fasta, "fasta"):
            strand = strand_map.get(rec.id, "+")
            raw_seq = str(rec.seq)
            if strand == "-":
                corrected = str(reverse_complement(raw_seq))
            else:
                corrected = raw_seq
            corrected = corrected.upper()[:SEQ_LEN].ljust(SEQ_LEN, 'N')
            sequences.append(corrected)
            ohe_list.append(get_ohe_features(raw_seq, strand))
            y_list.append(1)
            seq_ids.append(rec.id)

        pos_n = len(y_list)
        if pos_n == 0:
            return f"SKIP: {tf} (0 positives)"

        if os.path.exists(neg_fasta):
            for rec in SeqIO.parse(neg_fasta, "fasta"):
                if len(y_list) >= pos_n * 3:
                    break
                raw_seq = str(rec.seq)
                corrected = raw_seq.upper()[:SEQ_LEN].ljust(SEQ_LEN, 'N')
                sequences.append(corrected)
                ohe_list.append(get_ohe_features(raw_seq, "+"))
                y_list.append(0)
                seq_ids.append(rec.id)

        # --- OHE features (400) ---
        X_ohe = np.array(ohe_list, dtype=np.float32)
        n_ohe = X_ohe.shape[1]

        # --- Flexibility features (~493) via DNAflexpy ---
        X_flex, flex_names = extract_flex_features(sequences, seq_ids)

        # --- Combine: [OHE(400) | Flex(~493)] = ~893 ---
        X = np.hstack([X_ohe, X_flex]).astype(np.float32)
        y = np.array(y_list, dtype=np.int8)

        # Build feature names
        ohe_names = build_ohe_feature_names()
        feature_names = ohe_names + flex_names

        del ohe_list, sequences, X_ohe, X_flex; gc.collect()

        neg_n = int(np.sum(y == 0))
        spw = neg_n / max(pos_n, 1)

        # --- Same CV folds as Run 1 & Run 2 ---
        outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # --- Train both models ---
        rf_metrics, rf_model, rf_best, rf_curves = train_rf_nested_cv(X, y, outer_cv, inner_cv)
        xgb_metrics, xgb_model, xgb_best, xgb_curves = train_xgb_nested_cv(X, y, outer_cv, inner_cv, spw)

        # --- Save fold metrics ---
        metric_cols = ['Fold', 'AUPRC', 'AUROC', 'F1', 'Precision', 'Recall', 'MCC',
                       'TP', 'FP', 'TN', 'FN']
        pd.DataFrame(rf_metrics)[metric_cols].to_csv(
            os.path.join(tf_out, f"{tf}_RF_metrics.csv"), index=False)
        pd.DataFrame(xgb_metrics)[metric_cols].to_csv(
            os.path.join(tf_out, f"{tf}_XGB_metrics.csv"), index=False)

        # --- Model comparison ---
        rf_mean  = np.mean([m['AUPRC'] for m in rf_metrics])
        xgb_mean = np.mean([m['AUPRC'] for m in xgb_metrics])
        comp = pd.DataFrame({
            'Model':       ['RandomForest', 'XGBoost'],
            'Mean_AUPRC':  [rf_mean, xgb_mean],
            'Best_AUPRC':  [rf_best, xgb_best],
            'Mean_AUROC':  [np.mean([m['AUROC'] for m in rf_metrics]),
                            np.mean([m['AUROC'] for m in xgb_metrics])],
            'Mean_F1':     [np.mean([m['F1'] for m in rf_metrics]),
                            np.mean([m['F1'] for m in xgb_metrics])],
            'Mean_MCC':    [np.mean([m['MCC'] for m in rf_metrics]),
                            np.mean([m['MCC'] for m in xgb_metrics])],
        })
        comp.to_csv(os.path.join(tf_out, f"{tf}_model_comparison.csv"), index=False)

        # --- Save trained models ---
        joblib.dump(rf_model, os.path.join(tf_out, f"{tf}_RF_model.joblib"), compress=3)
        joblib.dump(xgb_model, os.path.join(tf_out, f"{tf}_XGB_model.joblib"), compress=3)

        # --- ROC & PR curves ---
        plot_curves(rf_curves, rf_metrics, tf_out, tf, "RF")
        plot_curves(xgb_curves, xgb_metrics, tf_out, tf, "XGB")

        # --- SHAP, top features, group contribution, dependence, position profile ---
        save_model_outputs(tf_out, tf, rf_model, X, feature_names, "RF", n_ohe)
        save_model_outputs(tf_out, tf, xgb_model, X, feature_names, "XGB", n_ohe)

        # --- Per-sequence prediction scores ---
        save_prediction_scores(tf_out, tf, rf_model, X, y, seq_ids, "RF")
        save_prediction_scores(tf_out, tf, xgb_model, X, y, seq_ids, "XGB")

        tf_min = (time.time() - tf_start) / 60
        tot_min = (time.time() - global_start) / 60
        return (f"OK: {tf} [{idx}/{total_tfs}] "
                f"RF={rf_mean:.3f} XGB={xgb_mean:.3f} | "
                f"{tf_min:.1f}m | total {tot_min:.1f}m")

    except Exception as e:
        import traceback
        return f"ERR: {tf} | {traceback.format_exc()}"
    finally:
        gc.collect()


# --- 10. MAIN ---
if __name__ == "__main__":
    # Verify DNAflexpy YAML exists
    if not os.path.exists(DNAFLEXPY_YAML):
        logging.error(f"DNAflexpy lookup table not found: {DNAFLEXPY_YAML}")
        logging.error("Update DNAFLEXPY_YAML path and restart.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_tfs = sorted(d for d in os.listdir(FASTA_BASE)
                     if os.path.isdir(os.path.join(FASTA_BASE, d)))

    # --- RESUME LOGIC: skip TFs that already completed ---
    pending_tfs = []
    skipped = 0
    for tf in all_tfs:
        comp_file = os.path.join(OUTPUT_DIR, tf, f"{tf}_model_comparison.csv")
        if os.path.exists(comp_file):
            skipped += 1
        else:
            pending_tfs.append(tf)

    # --- LOAD BALANCING: interleave by dataset size ---
    interleaved = interleave_by_size(pending_tfs)
    balanced_tfs = [tf for tf, _ in interleaved]

    sizes_mb = [s / (1024 * 1024) for _, s in interleaved]
    if sizes_mb:
        logging.info(f"Load balancing: largest={max(sizes_mb):.1f}MB, "
                     f"smallest={min(sizes_mb):.1f}MB, "
                     f"median={sorted(sizes_mb)[len(sizes_mb)//2]:.1f}MB")
        first_batch = [(tf, s / (1024 * 1024)) for tf, s in interleaved[:MAX_WORKERS]]
        logging.info(f"First batch: {', '.join(f'{tf}({sz:.0f}MB)' for tf, sz in first_batch)}")

    logging.info(f"PIPELINE RESUME (OHE + FLEX): {len(all_tfs)} total TFs | "
                 f"{skipped} already done | {len(pending_tfs)} remaining | "
                 f"{MAX_WORKERS} workers | load-balanced")
    logging.info(f"Features: OHE (400) + Flex (~493) = ~893 | SEQ_LEN: {SEQ_LEN}")
    t0 = time.time()

    tasks = [(tf, i + 1, len(balanced_tfs), t0) for i, tf in enumerate(balanced_tfs)]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(process_tf, t): t[0] for t in tasks}
        with tqdm(total=len(balanced_tfs), desc="TFs Processed",
                  bar_format='{l_bar}{bar:30}{r_bar}',
                  ncols=100, unit="TF") as pbar:
            for future in as_completed(futures):
                res = future.result()
                logging.info(res)
                if res.startswith("OK"):
                    tf_name = res.split("[")[0].replace("OK: ", "").strip()
                    scores = res.split("]")[1].split("|")[0].strip()
                    pbar.set_postfix_str(f"{tf_name} {scores}")
                elif res.startswith("ERR"):
                    tf_name = res.split("|")[0].replace("ERR: ", "").strip()
                    pbar.set_postfix_str(f"{tf_name} FAILED")
                else:
                    tf_name = res.split(":")[1].strip().split(" ")[0]
                    pbar.set_postfix_str(f"{tf_name} skipped")
                pbar.update(1)

    # Global summary (includes all TFs, not just pending)
    summary_rows = []
    for tf in all_tfs:
        comp_file = os.path.join(OUTPUT_DIR, tf, f"{tf}_model_comparison.csv")
        if os.path.exists(comp_file):
            c = pd.read_csv(comp_file)
            summary_rows.append({
                'TF': tf,
                'RF_Mean_AUPRC':  c.loc[c.Model == 'RandomForest', 'Mean_AUPRC'].values[0],
                'XGB_Mean_AUPRC': c.loc[c.Model == 'XGBoost', 'Mean_AUPRC'].values[0],
                'RF_Mean_AUROC':  c.loc[c.Model == 'RandomForest', 'Mean_AUROC'].values[0],
                'XGB_Mean_AUROC': c.loc[c.Model == 'XGBoost', 'Mean_AUROC'].values[0],
                'RF_Mean_MCC':    c.loc[c.Model == 'RandomForest', 'Mean_MCC'].values[0],
                'XGB_Mean_MCC':   c.loc[c.Model == 'XGBoost', 'Mean_MCC'].values[0],
            })
    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary['Winner_AUPRC'] = np.where(
            summary.XGB_Mean_AUPRC > summary.RF_Mean_AUPRC, 'XGB', 'RF')
        summary.to_csv(os.path.join(OUTPUT_DIR, "all_TFs_model_comparison.csv"), index=False)
        logging.info(f"Winner counts: {summary.Winner_AUPRC.value_counts().to_dict()}")

    logging.info(f"DONE \u2014 {(time.time() - t0) / 60:.1f} min total")
