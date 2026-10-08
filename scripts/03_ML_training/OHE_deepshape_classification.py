"""
OHE + DeepDNAshape Based TF Binding Site Classification (Run 5)
================================================================
Target: ProLiant DL380 Gen11 — Xeon Gold 6530 (128 threads), NVIDIA L40S, 256GB RAM
Environment: deepDNAshape (Python 3.10, tensorflow 2.15.1)

Models: RandomForest (CPU) + XGBoost (GPU)
Features: OHE (400) + DeepDNAshape (1,294) = 1,694 total
Reference: Li, Chiu & Rohs (2024) Nat Commun 15:1243

DeepDNAshape predicts the same 13 shape parameters as the pentamer lookup table
(Run 1) but uses a deep learning model that captures extended flanking region
effects beyond the 5-mer window. This enables comparison of pentamer-based vs
DL-based shape prediction for TF binding site classification.

13 shape parameters (excluding EP):
  Intrabase (7): MGW, ProT, Opening, Buckle, Shear, Stretch, Stagger  → 100 values each
  Interbase (6): Roll, HelT, Tilt, Rise, Shift, Slide                 → 99 values each
  Total per sequence: 7×100 + 6×99 = 1,294 features

Same CV folds (random_state=42) as all other runs for fair comparison.

=== MODIFICATION LOG ===
[2026-05-07] v1.0 — Initial script. Built from Run 1/Run 3 architecture with
                     DeepDNAshape integration (Li et al. 2024).
                     All optimizations from Runs 1-3 baked in from day one:
                     - Resume logic, load balancing, bulletproof SHAP
                     - check_additivity=False, interaction_index=None
                     - All plots in try/except with plt.close('all')
                     - Model saving (joblib, compress=3), tqdm + as_completed
                     DeepDNAshape runs on CPU (mode='cpu') to keep GPU free
                     for XGBoost. Layer=4 (default, moderate flanking influence).
                     predictBatch for efficient shape inference.
                     MAX_WORKERS=6, RF_INNER_JOBS=12, total 72 threads.
                     Conservative workers because each loads 13 TF models (~650 MB).
[2026-05-08] v1.1 — XGB_BASE_PARAMS: replaced gpu_id=0 + tree_method='gpu_hist'
                     with device='cuda:0' (XGBoost >= 3.1 API change).
                     SHAP fix: use model.get_booster() for XGBoost >= 3.1 to
                     avoid string-to-float conversion error. Fallback to
                     model_output='raw' if primary method fails.
[2026-05-08] v1.2 — SHAP fix v2: get_booster() still failed with SHAP library.
                     Replaced with XGBoost native pred_contribs=True which
                     bypasses SHAP library entirely. Uses booster.predict()
                     with DMatrix. Works with all XGBoost versions.
========================
"""

import os, gc, time, logging, warnings, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings before import

from tqdm import tqdm
import joblib
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
FASTA_BASE = "/NASpool/anwesha2026/input_TF_seq_fasta"
FIMO_BASE  = "/NASpool/anwesha2026/input_fimo_results"
OUTPUT_DIR = "/NASpool/anwesha2026/OHE_deepshape_classification_output"
SEQ_LEN    = 100

# Parallelism: 6 workers × 12 inner = 72 threads (RF training)
# Conservative workers: each loads 13 TF shape models (~650 MB RAM)
MAX_WORKERS   = 10
RF_INNER_JOBS = 12

# DeepDNAshape settings
DEEPSHAPE_LAYER = 4          # flanking influence depth (0-7, default=4)
DEEPSHAPE_BATCH = 2048       # batch size for predictBatch
DEEPSHAPE_MODE  = "cpu"      # keep GPU free for XGBoost

# 13 shape parameters (excluding EP, matching Run 1)
INTRABASE_FEATURES = ["MGW", "ProT", "Opening", "Buckle", "Shear", "Stretch", "Stagger"]
INTERBASE_FEATURES = ["Roll", "HelT", "Tilt", "Rise", "Shift", "Slide"]
ALL_SHAPE_FEATURES = INTRABASE_FEATURES + INTERBASE_FEATURES

RF_PARAM_GRID = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
}

XGB_BASE_PARAMS = dict(
    device           = "cuda:0",
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
    handlers=[logging.FileHandler("OHE_deepshape_classification.log"), logging.StreamHandler()]
)


# --- 2. LOAD BALANCING ---

def get_tf_dataset_size(tf):
    pos = os.path.join(FASTA_BASE, tf, f"{tf}_positives.fasta")
    neg = os.path.join(FASTA_BASE, tf, f"{tf}_negatives.fasta")
    size = 0
    if os.path.exists(pos):
        size += os.path.getsize(pos)
    if os.path.exists(neg):
        size += os.path.getsize(neg)
    return size


def interleave_by_size(tf_list):
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


# --- 3. DEEPDNASHAPE FEATURE EXTRACTION ---

_worker_predictor = None

def _init_deepshape():
    """Lazy initialization of DeepDNAshape predictor (once per worker)."""
    global _worker_predictor
    if _worker_predictor is None:
        from deepDNAshape.predictor import predictor
        _worker_predictor = predictor(mode=DEEPSHAPE_MODE)
        # Pre-load all 13 models to avoid lazy loading during batch processing
        _worker_predictor.loadAll()


def extract_deepshape_features(sequences):
    """
    Extract 13 DeepDNAshape features for a list of sequences.

    Returns: (shape_matrix, feature_names)
        shape_matrix: np.ndarray (n_sequences, 1294)
        feature_names: list of names (e.g., DeepShape_pos1_MGW)
    """
    _init_deepshape()
    pred = _worker_predictor

    shape_arrays = []
    shape_names = []

    for feat_name in ALL_SHAPE_FEATURES:
        # Batch prediction for efficiency
        all_preds = []
        for batch_start in range(0, len(sequences), DEEPSHAPE_BATCH):
            batch = sequences[batch_start:batch_start + DEEPSHAPE_BATCH]
            batch_preds = pred.predictBatch(feat_name, batch, layer=DEEPSHAPE_LAYER)
            all_preds.append(batch_preds)

        feat_vals = np.vstack(all_preds).astype(np.float32)
        n_cols = feat_vals.shape[1]
        shape_arrays.append(feat_vals)

        for col_idx in range(n_cols):
            shape_names.append(f"DeepShape_pos{col_idx+1}_{feat_name}")

    shape_matrix = np.hstack(shape_arrays)
    return shape_matrix, shape_names


# --- 4. OHE FEATURES ---
_OHE_MAP  = np.eye(4, dtype=np.float32)
_BASE2INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
_BASES    = ['A', 'C', 'G', 'T']


def get_ohe_features(seq, strand):
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
    elif name.startswith("DeepShape_"):
        parts = name.rsplit("_", 1)
        return "DeepShape_" + parts[1] if len(parts) > 1 else "DeepShape"
    return "Unknown"


# --- 5. EVALUATION ---

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


# --- 6. MODEL TRAINING ---

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


# --- 7. OUTPUT FUNCTIONS ---

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
    if isinstance(shap_vals, list):
        return shap_vals[1]
    if isinstance(shap_vals, np.ndarray):
        if shap_vals.ndim == 3:
            return shap_vals[:, :, 1]
        if shap_vals.ndim == 2:
            return shap_vals
    return None


def save_model_outputs(tf_out, tf, model, X, feature_names, tag, n_ohe):
    # --- SHAP computation ---
    sv = None
    shap_sample = X[:min(len(X), 200)]
    try:
        if tag == "XGB" and hasattr(model, 'get_booster'):
            # XGBoost native SHAP — bypasses SHAP library entirely
            # Works with all XGBoost versions including 3.2
            booster = model.get_booster()
            dmatrix = xgb.DMatrix(shap_sample)
            shap_contribs = booster.predict(dmatrix, pred_contribs=True)
            sv = shap_contribs[:, :-1].astype(np.float32)
        else:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(shap_sample, check_additivity=False)
            sv = _extract_shap_values(shap_vals)
            if sv is not None and sv.ndim != 2:
                sv = None
    except Exception as e:
        logging.warning(f"{tf} {tag}: SHAP computation failed ({e})")

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

    # Coarse group contribution (OHE vs DeepShape)
    group_imp = {}
    for i, importance in enumerate(imp):
        grp = get_feature_group(feature_names[i])
        base_grp = grp.split("_")[0]
        if base_grp == "DeepShape":
            base_grp = "DeepShape"
        if base_grp not in group_imp:
            group_imp[base_grp] = 0.0
        group_imp[base_grp] += importance

    # Fine group contribution (per shape parameter)
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

    # Group contribution bar plot (2 panels)
    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes[0].barh(grp_df['Group'], grp_df['Fraction'])
        axes[0].set_xlabel('Fraction of Total Importance')
        axes[0].set_title(f'{tf} \u2014 {tag} Coarse (OHE vs DeepShape)')
        axes[0].invert_yaxis()
        fine_top = fine_df.head(15)
        axes[1].barh(fine_top['Group'], fine_top['Fraction'])
        axes[1].set_xlabel('Fraction of Total Importance')
        axes[1].set_title(f'{tf} \u2014 {tag} Per Parameter')
        axes[1].invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(tf_out, f"{tf}_{tag}_group_contribution.png"),
                    bbox_inches='tight', dpi=150)
        plt.close()
    except Exception as e:
        logging.warning(f"{tf} {tag}: group contribution plot failed ({e})")
        plt.close('all')

    # SHAP dependence (top 5)
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

    # Position importance (OHE block: clean 100-position mapping)
    try:
        ohe_imp = imp[:n_ohe]
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


# --- 8. CORE TF PROCESSING ---

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

        # --- Read sequences with strand correction ---
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

        # --- DeepDNAshape features (1,294) ---
        X_shape, shape_names = extract_deepshape_features(sequences)

        # --- Combine: [OHE(400) | DeepShape(1294)] = 1,694 ---
        X = np.hstack([X_ohe, X_shape]).astype(np.float32)
        y = np.array(y_list, dtype=np.int8)

        ohe_names = build_ohe_feature_names()
        feature_names = ohe_names + shape_names

        del ohe_list, sequences, X_ohe, X_shape; gc.collect()

        neg_n = int(np.sum(y == 0))
        spw = neg_n / max(pos_n, 1)

        # --- Same CV folds as all other runs ---
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


# --- 9. MAIN ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_tfs = sorted(d for d in os.listdir(FASTA_BASE)
                     if os.path.isdir(os.path.join(FASTA_BASE, d)))

    # --- RESUME LOGIC ---
    pending_tfs = []
    skipped = 0
    for tf in all_tfs:
        comp_file = os.path.join(OUTPUT_DIR, tf, f"{tf}_model_comparison.csv")
        if os.path.exists(comp_file):
            skipped += 1
        else:
            pending_tfs.append(tf)

    # --- LOAD BALANCING ---
    interleaved = interleave_by_size(pending_tfs)
    balanced_tfs = [tf for tf, _ in interleaved]

    sizes_mb = [s / (1024 * 1024) for _, s in interleaved]
    if sizes_mb:
        logging.info(f"Load balancing: largest={max(sizes_mb):.1f}MB, "
                     f"smallest={min(sizes_mb):.1f}MB, "
                     f"median={sorted(sizes_mb)[len(sizes_mb)//2]:.1f}MB")
        first_batch = [(tf, s / (1024 * 1024)) for tf, s in interleaved[:MAX_WORKERS]]
        logging.info(f"First batch: {', '.join(f'{tf}({sz:.0f}MB)' for tf, sz in first_batch)}")

    logging.info(f"PIPELINE RESUME (OHE + DEEPSHAPE): {len(all_tfs)} total TFs | "
                 f"{skipped} already done | {len(pending_tfs)} remaining | "
                 f"{MAX_WORKERS} workers | load-balanced")
    logging.info(f"Features: OHE (400) + DeepDNAshape (1294) = 1694 | "
                 f"Layer: {DEEPSHAPE_LAYER} | SEQ_LEN: {SEQ_LEN}")
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

    # Global summary
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
