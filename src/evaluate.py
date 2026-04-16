"""
src/evaluate.py
Evaluation utilities shared across all model conditions.
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import EXPERIMENT_LOG, METRICS_DIR, N_BOOTSTRAP, RANDOM_SEED


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = 'auprc',
    n_iterations: int = N_BOOTSTRAP,
    ci: float = 0.95,
    seed: int = RANDOM_SEED,
) -> tuple:
    """
    Bootstrap confidence interval for AUC-ROC or AUPRC.

    Resamples rows with replacement and skips any resample that ends up
    with only one class (metric undefined). Returns (mean, lower, upper).
    """
    rng    = np.random.default_rng(seed)
    n      = len(y_true)
    scores = []

    for _ in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        y_t = y_true[idx]
        y_p = y_prob[idx]

        # Skip resamples with only one class — metric undefined
        if len(np.unique(y_t)) < 2:
            continue

        if metric == 'auprc':
            scores.append(average_precision_score(y_t, y_p))
        elif metric == 'auc_roc':
            scores.append(roc_auc_score(y_t, y_p))
        else:
            raise ValueError(f"metric must be 'auprc' or 'auc_roc', got '{metric}'")

    scores = np.array(scores)
    alpha  = (1 - ci) / 2
    lower  = float(np.percentile(scores, 100 * alpha))
    upper  = float(np.percentile(scores, 100 * (1 - alpha)))
    mean   = float(scores.mean())

    return round(mean, 4), round(lower, 4), round(upper, 4)


def select_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return the threshold in [0.01, 0.99] that maximises F1."""
    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1, best_thresh = 0.0, 0.5
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return best_thresh


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Return auc_roc, auprc, f1, precision, recall, and threshold in a dict."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        'auc_roc'  : round(float(roc_auc_score(y_true, y_prob)), 4),
        'auprc'    : round(float(average_precision_score(y_true, y_prob)), 4),
        'f1'       : round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        'precision': round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        'recall'   : round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        'threshold': round(float(threshold), 4),
    }


def log_results(
    condition: str,
    model: str,
    strategy: str,
    val_metrics: dict,
    test_metrics: dict,
    hyperparams: dict = None,
) -> None:
    """Append one row to results/experiment_log.csv. Creates the file if absent."""
    row = {
        'condition'        : condition,
        'model'            : model,
        'strategy'         : strategy,
        'hyperparams'      : str(hyperparams or {}),
        'val_auc_roc'      : val_metrics.get('auc_roc'),
        'val_auprc'        : val_metrics.get('auprc'),
        'val_f1'           : val_metrics.get('f1'),
        'test_auc_roc'     : test_metrics.get('auc_roc'),
        'test_auprc'       : test_metrics.get('auprc'),
        'test_f1'          : test_metrics.get('f1'),
        'test_precision'   : test_metrics.get('precision'),
        'test_recall'      : test_metrics.get('recall'),
        'threshold'        : test_metrics.get('threshold'),
    }

    os.makedirs(METRICS_DIR, exist_ok=True)
    df_row = pd.DataFrame([row])

    if os.path.exists(EXPERIMENT_LOG):
        df_row.to_csv(EXPERIMENT_LOG, mode='a', header=False, index=False)
    else:
        df_row.to_csv(EXPERIMENT_LOG, index=False)

    print(f'Logged: {condition} | {model} | {strategy} | '
          f'val_AUPRC={val_metrics.get("auprc"):.4f} | '
          f'test_AUPRC={test_metrics.get("auprc"):.4f}')
