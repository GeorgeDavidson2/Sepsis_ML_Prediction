"""
src/evaluate.py
──────────────────────────────────────────────────────────────────────────────
Evaluation utilities shared across all model conditions.

Usage:
    from src.evaluate import compute_all_metrics, select_threshold, log_results
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

from src.config import EXPERIMENT_LOG, METRICS_DIR


def select_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Select the decision threshold that maximises F1 on the validation set.

    Parameters
    ----------
    y_true : np.ndarray — ground truth binary labels
    y_prob : np.ndarray — predicted probabilities

    Returns
    -------
    float — optimal threshold in [0, 1]
    """
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
    """
    Compute the full set of evaluation metrics for one condition.

    Parameters
    ----------
    y_true    : ground truth binary labels
    y_prob    : predicted probabilities (before thresholding)
    threshold : decision threshold for precision/recall/F1

    Returns
    -------
    dict with keys: auc_roc, auprc, f1, precision, recall, threshold
    """
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
    """
    Append one row to results/experiment_log.csv.

    Parameters
    ----------
    condition   : e.g. 'A', 'B', 'C', 'D'
    model       : e.g. 'LR', 'XGBoost', 'LSTM'
    strategy    : e.g. 'Strategy_A', 'Strategy_B'
    val_metrics : dict from compute_all_metrics() on validation set
    test_metrics: dict from compute_all_metrics() on test set
    hyperparams : optional dict of key hyperparameters used
    """
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
