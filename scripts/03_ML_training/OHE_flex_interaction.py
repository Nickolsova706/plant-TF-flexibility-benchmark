"""
================================================================================
OHE + Flexibility + Neighbor Product Interaction Classification
================================================================================
TARGET:  ProLiant DL380 Gen11 — Xeon Gold 6530 (128 threads), L40S GPU, 256GB
ENV:     translate_env (Python 3.9, XGBoost 2.1.4)

PURPOSE:
    Test whether *non-linear* nearest-neighbor interaction features improve
    TF binding site classification beyond raw flexibility descriptors.

    Earlier experiments showed:
    - Gradients [Flex(i+1) - Flex(i)]: USELESS (+0.0004). Linear — tree models
      reconstruct it implicitly through Flex(i) and Flex(i+1) splits.
    - Neighbor products [Flex(i) × Flex(i+1)]: small average effect (+0.003),
      with strong family-specific benefit (SPL +0.011, some AP2/ERF).
      Genuinely non-linear — trees cannot reconstruct it from raw features.
    - Cross-descriptor products [DescA(i) × DescB(i)]: bulky (>shape), skipped.

    This script tests the ONE engineering choice that showed promise:
    OHE + Flex + Neighbor products. Gradients excluded as dead weight.

FEATURE SET:
    OHE                : 400 features (100 positions × 4 bases)
    Flex (5 descriptors): 493 features (DNaseI:98 + NPP:98 + twistDisp:99
                                        + trx:99 + stiffness:99)
    Neighbor products  : 488 features (Flex(i) × Flex(i+1) per descriptor)
                                       (97+97+98+98+98 = 488)
    -------------------------------------------------------------
    TOTAL              : 1,381 features
                         (vs Shape 2,300 — 60% — efficient if performance matches)

COMPARISON TARGETS (XGB mean AUPRC, full 216):
    OHE only:           0.780
    Flex only:          0.798
    DeepShape:          0.800
    Shape:              0.803
    *This run:          target ~0.800 (match shape with 60% of features)

WORKER STRATEGY (manual two-phase):
    Phase 1: MAX_WORKERS=6, RF_INNER_JOBS=12 (heavy load per worker)
             → processes the 6 largest TFs first via load-balanced scheduling
             → keeps RAM headroom for REM19 (119 MB FASTA, ~900K sequences)
    Phase 2: After the ~10 largest TFs finish, stop the script, change to
             MAX_WORKERS=10, RF_INNER_JOBS=10, restart.
             Resume logic auto-skips completed TFs via their model_comparison.csv.

OUTPUTS (per TF in OHE_flex_nbrprod_output/<TF>/):
    23 files per TF (same as all previous runs):
      RF & XGB: metrics, model.joblib, ROC_PR_curves, top20_features,
                group_contribution_coarse + fine + plot, shap_summary,
                shap_dependence_top5, position_importance, prediction_scores
      model_comparison.csv

    Global:
      all_TFs_model_comparison.csv  (merged across completed TFs)

LOG FILE: OHE_flex_nbrprod.log
================================================================================
"""

import os, gc, time, logging, warnings
import numpy as np
import pandas as pd
import yaml
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
from tqdm import tqdm
import xgboost as xgb
import shap
import joblib

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================================
# CONFIGURATION
# ============================================================================
# Phase 1 settings: heavy per-worker resources for the largest TFs.
# After the big ones finish, change to MAX_WORKERS=10, RF_INNER_JOBS=10
# and restart — resume logic handles continuity automatically.

MAX_WORKERS   = 10     # Phase 1: 6 workers (bump to 10 after big TFs done)
RF_INNER_JOBS = 10    # RF GridSearchCV inner parallelism (6 × 12 = 72 threads)

# --- Paths ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

FASTA_BASE = os.environ.get("TF_FASTA_BASE")
FIMO_BASE  = os.environ.get("TF_FIMO_BASE")
OUTPUT_DIR = os.environ.get("OHE_FLEX_INTERACTION_OUTPUT", os.path.join(PROJECT_ROOT, "results", "OHE_flex_nbrprod"))
SEQ_LEN    = 100

# --- DNAflexpy YAML lookup ---
# Override DNAflexpy's default YAML loader with absolute path to avoid
# importlib-resource errors when forked by multiprocessing workers.
YAML_PATH = os.environ.get("DNAFLEXPY_YAML")

# Required external input paths
if not FASTA_BASE:
    raise RuntimeError("Set TF_FASTA_BASE to the location of the input FASTA directory.")
if not FIMO_BASE:
    raise RuntimeError("Set TF_FIMO_BASE to the location of the FIMO results directory.")
if not YAML_PATH:
    raise RuntimeError("Set DNAFLEXPY_YAML to the location of the DNAflexpy lookupNEW.yaml file.")

import DNAflexpy.core as df_core
import DNAflexpy.utils as df_utils

def _manual_yaml_load():
    """Hard-coded YAML loader to bypass DNAflexpy's importlib resolution."""
    with open(YAML_PATH, 'r') as f:
        return yaml.safe_load(f)

df_utils.load_feature_data = _manual_yaml_load
df_core.load_feature_data  = _manual_yaml_load

# The five flexibility descriptors used.
# Trinucleotide resolution (98 values per 100bp): DNaseI, NPP
# Dinucleotide resolution (99 values per 100bp): twistDisp, trx, stiffness
FLEX_DESCRIPTORS = ["DNaseI", "NPP", "twistDisp", "trx", "stiffness"]

# --- Model hyperparameter grids (identical to all previous runs) ---
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

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("OHE_flex_nbrprod.log"),
        logging.StreamHandler()
    ]
)


# ============================================================================
# SECTION 1: OHE FEATURES (400)
# ============================================================================

_OHE_MAP  = np.eye(4, dtype=np.float32)
_BASE2INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
_BASES    = ['A', 'C', 'G', 'T']


def get_ohe_features(seq, strand):
    """
    One-hot encode a sequence to 400 features (100 positions × 4 bases).
    Reverse-complements if FIMO assigned the minus strand.
    """
    if strand == "-":
        seq = str(reverse_complement(seq))
    seq = seq.upper()[:SEQ_LEN].ljust(SEQ_LEN, 'N')
    indices = np.array([_BASE2INT.get(b, -1) for b in seq], dtype=np.int8)
    ohe = np.zeros((SEQ_LEN, 4), dtype=np.float32)
    valid = indices >= 0
    ohe[valid] = _OHE_MAP[indices[valid]]
    return ohe.flatten()


def build_ohe_feature_names():
    """Generate 400 OHE feature names like 'OHE_pos1_A'."""
    return [f"OHE_pos{pos+1}_{base}" for pos in range(SEQ_LEN) for base in _BASES]


# ============================================================================
# SECTION 2: FLEXIBILITY FEATURES (493) — via DNAflexpy
# ============================================================================

def extract_flex_features(sequences, tf_name):
    """
    Extract 5 flexibility descriptors for all sequences via DNAflexpy.
    Writes sequences to a PID-unique temp FASTA to avoid worker collisions.

    Returns:
        flex_matrix: np.ndarray (n_seqs, 493)
        flex_names:  list of str
        flex_dict:   dict of {descriptor: np.ndarray} for interaction engineering
    """
    pid = os.getpid()
    tmp_fasta = f"/tmp/flex_tmp_{tf_name}_{pid}.fasta"

    try:
        # Write sequences to temp FASTA
        with open(tmp_fasta, 'w') as f:
            for i, seq in enumerate(sequences):
                f.write(f">seq_{i}\n{seq}\n")

        flex_arrays = []
        flex_names  = []
        flex_dict   = {}

        for desc in FLEX_DESCRIPTORS:
            try:
                # DNAflexpyMP returns DataFrame: first col is seq ID, rest are values
                df_feat = df_core.DNAflexpyMP(
                    input_file=tmp_fasta,
                    window_size=0,
                    feature=desc,
                    threads=1,       # single thread per worker; outer parallelism by ProcessPoolExecutor
                    outfile=None,
                )
                vals = df_feat.iloc[:, 1:].values.astype(np.float32)
                flex_arrays.append(vals)
                flex_dict[desc] = vals
                for col_idx in range(vals.shape[1]):
                    flex_names.append(f"Flex_{desc}_pos{col_idx+1}")
            except Exception as e:
                logging.warning(f"DNAflexpy failed for {desc}: {e}")
                expected_len = 98 if desc in ["DNaseI", "NPP"] else 99
                zeros = np.zeros((len(sequences), expected_len), dtype=np.float32)
                flex_arrays.append(zeros)
                flex_dict[desc] = zeros
                for col_idx in range(expected_len):
                    flex_names.append(f"Flex_{desc}_pos{col_idx+1}")

        flex_matrix = np.hstack(flex_arrays)
        return flex_matrix, flex_names, flex_dict

    finally:
        if os.path.exists(tmp_fasta):
            os.remove(tmp_fasta)


# ============================================================================
# SECTION 3: NEIGHBOR PRODUCT INTERACTION FEATURES (488)
# ============================================================================
# For each of the 5 flexibility descriptors, compute element-wise products
# of consecutive positions: Flex(i) × Flex(i+1).
#
# BIOLOGICAL RATIONALE:
#   High product = both adjacent positions extreme (both stiff or both flexible),
#                  signaling cooperative DNA mechanical behavior over a 2bp window.
#   Low product  = one position high, one low (a local discontinuity).
#
# WHY NON-REDUNDANT (unlike gradients):
#   Tree models split on individual features. A product Flex(i)*Flex(i+1) is a
#   non-linear combination that a single tree split cannot compute. Trees would
#   need multiple sequential splits to approximate it. Adding the product
#   explicitly gives the model direct access to this cooperativity signal.
#
# FEATURE COUNT:
#   DNaseI    (98 vals) → 97 products
#   NPP       (98 vals) → 97 products
#   twistDisp (99 vals) → 98 products
#   trx       (99 vals) → 98 products
#   stiffness (99 vals) → 98 products
#   Total: 488 features
# ============================================================================

def engineer_neighbor_products(flex_dict):
    """
    Generate non-linear nearest-neighbor product features.
    For each descriptor and each consecutive position pair (i, i+1),
    compute the element-wise product.

    Returns:
        nbr_matrix: np.ndarray (n_seqs, 488)
        nbr_names:  list of str
    """
    arrays = []
    names  = []

    for desc in FLEX_DESCRIPTORS:
        vals = flex_dict[desc]                 # (n_seqs, 98 or 99)
        product = vals[:, 1:] * vals[:, :-1]   # (n_seqs, 97 or 98)
        arrays.append(product.astype(np.float32))
        for pos in range(product.shape[1]):
            names.append(f"NbrProd_{desc}_pos{pos+1}x{pos+2}")

    nbr_matrix = np.hstack(arrays)
    return nbr_matrix, names


# ============================================================================
# SECTION 4: FEATURE GROUP CLASSIFICATION
# ============================================================================
# Used for the group_contribution_coarse/fine CSVs that decompose feature
# importance by feature class. Three coarse groups:
#   - OHE             : sequence-only
#   - Flex_Original   : the 493 raw flexibility features
#   - Flex_NbrProduct : the 488 neighbor-product interaction features

def get_feature_group(name):
    """Classify a feature name into coarse and fine groups."""
    if name.startswith("OHE_"):
        return "OHE", "OHE"
    elif name.startswith("Flex_"):
        desc = name.split("_")[1]
        return "Flex_Original", f"Flex_{desc}"
    elif name.startswith("NbrProd_"):
        desc = name.split("_")[1]
        return "Flex_NbrProduct", f"NbrProd_{desc}"
    return "Unknown", "Unknown"


# ============================================================================
# SECTION 5: LOAD BALANCING
# ============================================================================
# Process TFs in interleaved size order (largest, smallest, 2nd-largest, ...)
# so that workers stay roughly balanced and the heaviest jobs get a head start.

def get_tf_dataset_size(tf):
    """Total bytes of positive + negative FASTAs for sizing."""
    pos = os.path.join(FASTA_BASE, tf, f"{tf}_positives.fasta")
    neg = os.path.join(FASTA_BASE, tf, f"{tf}_negatives.fasta")
    size = 0
    if os.path.exists(pos): size += os.path.getsize(pos)
    if os.path.exists(neg): size += os.path.getsize(neg)
    return size


def interleave_by_size(tf_list):
    """
    Largest-first interleaving for balanced parallel processing.
    Returns list of (tf, size_bytes) pairs.

    The largest TF is item 0, second-smallest is item 1, second-largest is
    item 2, etc. This keeps wall time balanced because the longest tasks
    get dispatched immediately when workers start up.
    """
    sized = [(tf, get_tf_dataset_size(tf)) for tf in tf_list]
    sized.sort(key=lambda x: x[1], reverse=True)
    interleaved = []
    left, right = 0, len(sized) - 1
    toggle = True
    while left <= right:
        if toggle:
            interleaved.append(sized[left]); left += 1
        else:
            interleaved.append(sized[right]); right -= 1
        toggle = not toggle
    return interleaved


# ============================================================================
# SECTION 6: EVALUATION METRICS
# ============================================================================

def compute_fold_metrics(y_true, y_pred, y_prob, fold_idx):
    """Per-fold classification metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'Fold': fold_idx,
        'AUPRC': average_precision_score(y_true, y_prob),
        'AUROC': roc_auc_score(y_true, y_prob),
        'F1': f1_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
    }


# ============================================================================
# SECTION 7: MODEL TRAINING (RF + XGBoost, nested CV)
# ============================================================================

def train_rf_nested_cv(X, y, outer_cv, inner_cv):
    """RandomForest with 10-fold outer × 5-fold inner CV + grid search."""
    metrics, curves, best_model, best_score = [], [], None, -1
    for f_idx, (tr, te) in enumerate(outer_cv.split(X, y)):
        grid = GridSearchCV(
            RandomForestClassifier(class_weight='balanced', random_state=42),
            RF_PARAM_GRID, cv=inner_cv, scoring='average_precision',
            n_jobs=RF_INNER_JOBS)
        grid.fit(X[tr], y[tr])
        model = grid.best_estimator_
        probs = model.predict_proba(X[te])[:, 1]
        preds = model.predict(X[te])
        m = compute_fold_metrics(y[te], preds, probs, f_idx + 1)
        metrics.append(m)
        fpr, tpr, _ = roc_curve(y[te], probs)
        prec_c, rec_c, _ = precision_recall_curve(y[te], probs)
        curves.append({'fpr': fpr, 'tpr': tpr, 'precision': prec_c, 'recall': rec_c})
        if m['AUPRC'] > best_score:
            best_score, best_model = m['AUPRC'], model
    return metrics, best_model, best_score, curves


def train_xgb_nested_cv(X, y, outer_cv, inner_cv, spw):
    """XGBoost with 10-fold outer × 5-fold inner CV + manual grid search (GPU)."""
    metrics, curves, best_model, best_score = [], [], None, -1
    for f_idx, (tr, te) in enumerate(outer_cv.split(X, y)):
        X_tr, y_tr, X_te, y_te = X[tr], y[tr], X[te], y[te]

        # Inner CV: pick best hyperparams by mean AUPRC across inner folds
        best_inner, best_p = -1, XGB_PARAM_GRID[0]
        for params in XGB_PARAM_GRID:
            iscores = []
            for itr, ite in inner_cv.split(X_tr, y_tr):
                mdl = xgb.XGBClassifier(
                    **{**XGB_BASE_PARAMS, **params},
                    scale_pos_weight=spw, early_stopping_rounds=20)
                mdl.fit(X_tr[itr], y_tr[itr],
                        eval_set=[(X_tr[ite], y_tr[ite])], verbose=False)
                iscores.append(average_precision_score(
                    y_tr[ite], mdl.predict_proba(X_tr[ite])[:, 1]))
            ms = np.mean(iscores)
            if ms > best_inner: best_inner, best_p = ms, params

        # Refit on the full outer train with the best inner params
        mdl = xgb.XGBClassifier(
            **{**XGB_BASE_PARAMS, **best_p},
            scale_pos_weight=spw, early_stopping_rounds=20)
        Xf, Xv, yf, yv = train_test_split(
            X_tr, y_tr, test_size=0.1, stratify=y_tr, random_state=42)
        mdl.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)
        probs = mdl.predict_proba(X_te)[:, 1]
        preds = mdl.predict(X_te)
        m = compute_fold_metrics(y_te, preds, probs, f_idx + 1)
        metrics.append(m)
        fpr, tpr, _ = roc_curve(y_te, probs)
        prec_c, rec_c, _ = precision_recall_curve(y_te, probs)
        curves.append({'fpr': fpr, 'tpr': tpr, 'precision': prec_c, 'recall': rec_c})
        if m['AUPRC'] > best_score:
            best_score, best_model = m['AUPRC'], mdl
    return metrics, best_model, best_score, curves


# ============================================================================
# SECTION 8: PLOT + OUTPUT HELPERS
# ============================================================================

def _extract_shap_values(shap_vals):
    """Normalize SHAP output across sklearn/xgb formats to (n_samples, n_features)."""
    if isinstance(shap_vals, list): return shap_vals[1]
    if isinstance(shap_vals, np.ndarray):
        if shap_vals.ndim == 3: return shap_vals[:, :, 1]
        if shap_vals.ndim == 2: return shap_vals
    return None


def plot_curves(curves, metrics, tf_out, tf, tag):
    """Mean ROC and PR curves with std band across CV folds."""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        mean_fpr = np.linspace(0, 1, 200)
        tprs = [np.interp(mean_fpr, c['fpr'], c['tpr']) for c in curves]
        mean_tpr, std_tpr = np.mean(tprs, axis=0), np.std(tprs, axis=0)
        mean_auroc = np.mean([m['AUROC'] for m in metrics])
        axes[0].plot(mean_fpr, mean_tpr, lw=2, label=f'Mean AUROC = {mean_auroc:.3f}')
        axes[0].fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.2)
        axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
        axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
        axes[0].set_title(f'{tf} {tag} ROC'); axes[0].legend(loc='lower right')

        mean_rec = np.linspace(0, 1, 200)
        precs = [np.interp(mean_rec, np.sort(c['recall']),
                           c['precision'][np.argsort(c['recall'])]) for c in curves]
        mean_prec, std_prec = np.mean(precs, axis=0), np.std(precs, axis=0)
        mean_auprc = np.mean([m['AUPRC'] for m in metrics])
        axes[1].plot(mean_rec, mean_prec, lw=2, label=f'Mean AUPRC = {mean_auprc:.3f}')
        axes[1].fill_between(mean_rec, mean_prec - std_prec, mean_prec + std_prec, alpha=0.2)
        axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
        axes[1].set_title(f'{tf} {tag} PR'); axes[1].legend(loc='lower left')
        plt.tight_layout()
        plt.savefig(os.path.join(tf_out, f"{tf}_{tag}_ROC_PR_curves.png"),
                    bbox_inches='tight', dpi=150)
        plt.close()
    except Exception as e:
        logging.warning(f"{tf} {tag}: curve plot failed ({e})")
        plt.close('all')


def save_model_outputs(tf_out, tf, model, X, feature_names, tag, n_ohe):
    """
    Save the model interpretability outputs:
      - SHAP summary + dependence (top 5)
      - top20 features CSV
      - coarse + fine group contribution CSVs + bar plot
      - position importance (OHE block)
    """
    sv = None
    shap_sample = X[:min(len(X), 200)]
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(shap_sample, check_additivity=False)
        sv = _extract_shap_values(shap_vals)
        if sv is not None and sv.ndim != 2: sv = None
    except Exception as e:
        logging.warning(f"{tf} {tag}: SHAP failed ({e})")

    # SHAP summary plot
    if sv is not None:
        try:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(sv, shap_sample, feature_names=feature_names,
                              show=False, max_display=20)
            plt.title(f"{tf} {tag}")
            plt.savefig(os.path.join(tf_out, f"{tf}_{tag}_shap_summary.png"),
                        bbox_inches='tight', dpi=150)
            plt.close()
        except: plt.close('all')

    imp = model.feature_importances_
    top_idx = np.argsort(imp)[-20:][::-1]

    # Top 20 features CSV
    pd.DataFrame({
        'Rank': range(1, 21), 'Feature_Index': top_idx,
        'Feature_Name': [feature_names[i] for i in top_idx],
        'Importance': imp[top_idx],
    }).to_csv(os.path.join(tf_out, f"{tf}_{tag}_top20_features.csv"), index=False)

    # Coarse + fine group contribution
    coarse_imp, fine_imp = {}, {}
    for i, importance in enumerate(imp):
        coarse_grp, fine_grp = get_feature_group(feature_names[i])
        coarse_imp[coarse_grp] = coarse_imp.get(coarse_grp, 0) + importance
        fine_imp[fine_grp] = fine_imp.get(fine_grp, 0) + importance

    for label, grp_dict, suffix in [
        ("coarse", coarse_imp, "coarse"), ("fine", fine_imp, "fine")
    ]:
        df = pd.DataFrame([
            {'Group': k, 'Total_Importance': v}
            for k, v in sorted(grp_dict.items(), key=lambda x: -x[1])
        ])
        df['Fraction'] = df['Total_Importance'] / df['Total_Importance'].sum()
        df.to_csv(os.path.join(tf_out, f"{tf}_{tag}_group_contribution_{suffix}.csv"),
                  index=False)

    # Group contribution plot (coarse + fine top 15)
    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        coarse_df = pd.DataFrame([{'Group': k, 'Fraction': v / sum(coarse_imp.values())}
                                   for k, v in sorted(coarse_imp.items(), key=lambda x: -x[1])])
        axes[0].barh(coarse_df['Group'], coarse_df['Fraction'])
        axes[0].set_title(f'{tf} {tag} Coarse'); axes[0].invert_yaxis()
        fine_df = pd.DataFrame([{'Group': k, 'Fraction': v / sum(fine_imp.values())}
                                 for k, v in sorted(fine_imp.items(), key=lambda x: -x[1])[:15]])
        axes[1].barh(fine_df['Group'], fine_df['Fraction'])
        axes[1].set_title(f'{tf} {tag} Fine (top 15)'); axes[1].invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(tf_out, f"{tf}_{tag}_group_contribution.png"),
                    bbox_inches='tight', dpi=150)
        plt.close()
    except: plt.close('all')

    # SHAP dependence (top 5 features)
    if sv is not None:
        try:
            fig, axes = plt.subplots(1, 5, figsize=(25, 5))
            for r, fi in enumerate(top_idx[:5]):
                try:
                    shap.dependence_plot(fi, sv, shap_sample,
                        feature_names=feature_names, interaction_index=None,
                        ax=axes[r], show=False)
                except:
                    axes[r].set_title(f"Feature {fi}\n(failed)")
            plt.suptitle(f"{tf} {tag} SHAP Dependence (Top 5)")
            plt.tight_layout()
            plt.savefig(os.path.join(tf_out, f"{tf}_{tag}_shap_dependence_top5.png"),
                        bbox_inches='tight', dpi=150)
            plt.close()
        except: plt.close('all')

    # Position importance (sum of 4 OHE channels per position)
    try:
        ohe_imp = imp[:n_ohe].reshape(SEQ_LEN, 4).sum(axis=1)
        plt.figure(figsize=(14, 4))
        plt.bar(range(1, SEQ_LEN + 1), ohe_imp, width=1.0, edgecolor='none')
        plt.xlabel('Position (bp)'); plt.ylabel('Summed OHE Importance')
        plt.title(f'{tf} {tag} Position Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(tf_out, f"{tf}_{tag}_position_importance.png"),
                    bbox_inches='tight', dpi=150)
        plt.close()
    except: plt.close('all')


def save_prediction_scores(tf_out, tf, model, X, y, seq_ids, tag):
    """Per-sequence predictions: probability + binary label."""
    pd.DataFrame({
        'Sequence_ID': seq_ids, 'True_Label': y,
        'Predicted_Label': model.predict(X),
        'Binding_Probability': model.predict_proba(X)[:, 1],
    }).to_csv(os.path.join(tf_out, f"{tf}_{tag}_prediction_scores.csv"), index=False)


# ============================================================================
# SECTION 9: CORE PER-TF PROCESSING FUNCTION
# ============================================================================

def process_tf(tf_args):
    """
    Full pipeline for one TF:
      1. Load sequences (positives + negatives, FIMO strand-corrected)
      2. Extract features (OHE + Flex + NbrProduct)
      3. Train RF and XGBoost (10-fold outer × 5-fold inner CV)
      4. Save 23 output files

    Returns a status string for the main process log.
    """
    tf, idx, total_tfs, global_start = tf_args
    tf_start = time.time()

    try:
        # --- Setup output dir for this TF ---
        tf_out = os.path.join(OUTPUT_DIR, tf)
        os.makedirs(tf_out, exist_ok=True)

        # --- Locate input files ---
        fimo_file = os.path.join(FIMO_BASE, tf, "fimo.tsv")
        pos_fasta = os.path.join(FASTA_BASE, tf, f"{tf}_positives.fasta")
        neg_fasta = os.path.join(FASTA_BASE, tf, f"{tf}_negatives.fasta")

        if not os.path.exists(pos_fasta):
            return f"SKIP: {tf} (no positives fasta)"

        # --- Build FIMO strand map for reverse-complementing minus-strand hits ---
        strand_map = {}
        if os.path.exists(fimo_file):
            fimo = pd.read_csv(fimo_file, sep='\t', comment='#').dropna(subset=['sequence_name'])
            strand_map = dict(zip(fimo.sequence_name, fimo.strand))

        # --- Read sequences: positives first, then negatives (capped at 2x positives) ---
        sequences, ohe_list, y_list, seq_ids = [], [], [], []
        for rec in SeqIO.parse(pos_fasta, "fasta"):
            strand = strand_map.get(rec.id, "+")
            raw_seq = str(rec.seq)
            corrected = str(reverse_complement(raw_seq)) if strand == "-" else raw_seq
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
                if len(y_list) >= pos_n * 3: break
                raw_seq = str(rec.seq)
                corrected = raw_seq.upper()[:SEQ_LEN].ljust(SEQ_LEN, 'N')
                sequences.append(corrected)
                ohe_list.append(get_ohe_features(raw_seq, "+"))
                y_list.append(0)
                seq_ids.append(rec.id)

        # --- Feature extraction ---
        # OHE (400)
        X_ohe = np.array(ohe_list, dtype=np.float32)
        n_ohe = X_ohe.shape[1]

        # Flex (493)
        X_flex, flex_names, flex_dict = extract_flex_features(sequences, tf)

        # Neighbor products (488)
        X_nbr, nbr_names = engineer_neighbor_products(flex_dict)

        # --- Combine: [OHE(400) | Flex(493) | NbrProducts(488)] = 1,381 ---
        X = np.hstack([X_ohe, X_flex, X_nbr]).astype(np.float32)
        y = np.array(y_list, dtype=np.int8)
        feature_names = build_ohe_feature_names() + flex_names + nbr_names

        total_features = X.shape[1]
        logging.info(f"{tf}: {len(sequences)} seqs, {total_features} features "
                     f"(OHE={n_ohe}, Flex={X_flex.shape[1]}, NbrProd={X_nbr.shape[1]})")

        # Free intermediate arrays
        del ohe_list, sequences, X_ohe, X_flex, X_nbr, flex_dict
        gc.collect()

        # --- Class imbalance handling ---
        neg_n = int(np.sum(y == 0))
        spw = neg_n / max(pos_n, 1)

        # --- CV folds: same random_state across all runs for fair comparison ---
        outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # --- Train both models ---
        rf_metrics, rf_model, rf_best, rf_curves = train_rf_nested_cv(X, y, outer_cv, inner_cv)
        xgb_metrics, xgb_model, xgb_best, xgb_curves = train_xgb_nested_cv(X, y, outer_cv, inner_cv, spw)

        # --- Save per-fold metrics ---
        metric_cols = ['Fold', 'AUPRC', 'AUROC', 'F1', 'Precision', 'Recall', 'MCC',
                       'TP', 'FP', 'TN', 'FN']
        pd.DataFrame(rf_metrics)[metric_cols].to_csv(
            os.path.join(tf_out, f"{tf}_RF_metrics.csv"), index=False)
        pd.DataFrame(xgb_metrics)[metric_cols].to_csv(
            os.path.join(tf_out, f"{tf}_XGB_metrics.csv"), index=False)

        # --- Model comparison summary ---
        rf_mean  = np.mean([m['AUPRC'] for m in rf_metrics])
        xgb_mean = np.mean([m['AUPRC'] for m in xgb_metrics])

        pd.DataFrame({
            'Model': ['RandomForest', 'XGBoost'],
            'Mean_AUPRC': [rf_mean, xgb_mean],
            'Best_AUPRC': [rf_best, xgb_best],
            'Mean_AUROC': [np.mean([m['AUROC'] for m in rf_metrics]),
                           np.mean([m['AUROC'] for m in xgb_metrics])],
            'Mean_F1':    [np.mean([m['F1'] for m in rf_metrics]),
                           np.mean([m['F1'] for m in xgb_metrics])],
            'Mean_MCC':   [np.mean([m['MCC'] for m in rf_metrics]),
                           np.mean([m['MCC'] for m in xgb_metrics])],
            'Total_Features': [total_features, total_features],
        }).to_csv(os.path.join(tf_out, f"{tf}_model_comparison.csv"), index=False)

        # --- Save models and all derived outputs ---
        joblib.dump(rf_model,  os.path.join(tf_out, f"{tf}_RF_model.joblib"),  compress=3)
        joblib.dump(xgb_model, os.path.join(tf_out, f"{tf}_XGB_model.joblib"), compress=3)

        plot_curves(rf_curves,  rf_metrics,  tf_out, tf, "RF")
        plot_curves(xgb_curves, xgb_metrics, tf_out, tf, "XGB")

        save_model_outputs(tf_out, tf, rf_model,  X, feature_names, "RF",  n_ohe)
        save_model_outputs(tf_out, tf, xgb_model, X, feature_names, "XGB", n_ohe)

        save_prediction_scores(tf_out, tf, rf_model,  X, y, seq_ids, "RF")
        save_prediction_scores(tf_out, tf, xgb_model, X, y, seq_ids, "XGB")

        tf_min = (time.time() - tf_start) / 60
        tot_min = (time.time() - global_start) / 60
        return (f"OK: {tf} [{idx}/{total_tfs}] "
                f"RF={rf_mean:.3f} XGB={xgb_mean:.3f} | "
                f"Feat={total_features} | {tf_min:.1f}m | total {tot_min:.1f}m")

    except Exception as e:
        import traceback
        return f"ERR: {tf} | {traceback.format_exc()}"
    finally:
        gc.collect()


# ============================================================================
# SECTION 10: MAIN EXECUTION (RESUME + LOAD BALANCING)
# ============================================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Discover all TFs ---
    all_tfs = sorted(d for d in os.listdir(FASTA_BASE)
                     if os.path.isdir(os.path.join(FASTA_BASE, d)))

    # --- Resume logic: skip TFs that already have a completed model_comparison.csv ---
    pending_tfs = []
    skipped = 0
    for tf in all_tfs:
        comp = os.path.join(OUTPUT_DIR, tf, f"{tf}_model_comparison.csv")
        if os.path.exists(comp):
            skipped += 1
        else:
            pending_tfs.append(tf)

    # --- Load balancing: largest-first interleaving ---
    # Ensures the largest TFs (REM19, BPC1, GTL1...) start immediately
    # under Phase 1 settings (6 workers × 12 inner jobs).
    interleaved = interleave_by_size(pending_tfs)
    balanced_tfs = [tf for tf, _ in interleaved]

    # --- Startup banner ---
    logging.info(f"{'='*70}")
    logging.info(f"OHE + FLEX + NEIGHBOR PRODUCT INTERACTION RUN")
    logging.info(f"{'='*70}")
    logging.info(f"Features: OHE(400) + Flex(493) + NbrProducts(488) = 1,381 total")
    logging.info(f"Workers: {MAX_WORKERS} | Inner jobs: {RF_INNER_JOBS}")
    logging.info(f"  (Phase 1 settings — bump to MAX_WORKERS=10, RF_INNER_JOBS=10")
    logging.info(f"   after the largest TFs finish, then restart this script.)")
    logging.info(f"Output: {OUTPUT_DIR}")
    logging.info(f"TFs: {len(all_tfs)} total | {skipped} done | {len(pending_tfs)} remaining")
    if interleaved:
        logging.info(f"Largest pending TFs (head of queue):")
        for tf, sz in interleaved[:5]:
            logging.info(f"  {tf}: {sz / (1024*1024):.1f} MB")
    logging.info(f"Comparison targets (XGB mean AUPRC, 216 TFs):")
    logging.info(f"  OHE only:        0.780")
    logging.info(f"  Flex only:       0.798")
    logging.info(f"  DeepShape:       0.800")
    logging.info(f"  Shape:           0.803")
    logging.info(f"{'='*70}")

    # --- Parallel execution ---
    t0 = time.time()
    tasks = [(tf, i + 1, len(balanced_tfs), t0) for i, tf in enumerate(balanced_tfs)]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(process_tf, t): t[0] for t in tasks}
        with tqdm(total=len(balanced_tfs), desc="TFs",
                  bar_format='{l_bar}{bar:30}{r_bar}', ncols=100, unit="TF") as pbar:
            for future in as_completed(futures):
                res = future.result()
                logging.info(res)
                if res.startswith("OK"):
                    tf_name = res.split("[")[0].replace("OK: ", "").strip()
                    pbar.set_postfix_str(f"{tf_name}")
                pbar.update(1)

    # --- Global summary: merge all completed TFs into one CSV ---
    summary_rows = []
    for tf in all_tfs:
        comp = os.path.join(OUTPUT_DIR, tf, f"{tf}_model_comparison.csv")
        if os.path.exists(comp):
            c = pd.read_csv(comp)
            summary_rows.append({
                'TF': tf,
                'RF_Mean_AUPRC':  c.loc[c.Model == 'RandomForest', 'Mean_AUPRC'].values[0],
                'XGB_Mean_AUPRC': c.loc[c.Model == 'XGBoost',      'Mean_AUPRC'].values[0],
                'Total_Features': c['Total_Features'].values[0],
            })

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary['Winner'] = np.where(
            summary.XGB_Mean_AUPRC > summary.RF_Mean_AUPRC, 'XGB', 'RF')
        summary.to_csv(os.path.join(OUTPUT_DIR, "all_TFs_model_comparison.csv"), index=False)

        mean_xgb = summary['XGB_Mean_AUPRC'].mean()
        logging.info(f"\n{'='*70}")
        logging.info(f"RUN RESULTS ({len(summary)} TFs)")
        logging.info(f"{'='*70}")
        logging.info(f"Mean XGB AUPRC: {mean_xgb:.4f}")
        logging.info(f"Mean RF AUPRC:  {summary['RF_Mean_AUPRC'].mean():.4f}")
        logging.info(f"  vs Flex only (0.798):  {mean_xgb - 0.798:+.4f}")
        logging.info(f"  vs DeepShape (0.800):  {mean_xgb - 0.800:+.4f}")
        logging.info(f"  vs Shape (0.803):      {mean_xgb - 0.803:+.4f}")
        logging.info(f"Winner counts: {summary.Winner.value_counts().to_dict()}")
        logging.info(f"{'='*70}")

    logging.info(f"DONE — {(time.time() - t0) / 60:.1f} min total")
