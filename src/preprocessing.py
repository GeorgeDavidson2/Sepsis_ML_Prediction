"""
src/preprocessing.py
──────────────────────────────────────────────────────────────────────────────
Label engineering and preprocessing utilities for the Sepsis ML pipeline.

Usage:
    from src.preprocessing import engineer_labels
    df_shifted, excluded_ids = engineer_labels(df)
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.config import ALL_FEATURES, LABEL_SHIFT_HOURS, MODELS_DIR, OUTLIER_BOUNDS
from src.utils import validate_no_nans


def engineer_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Apply a 6-hour early-warning label shift to create the prediction target.

    For each patient:
      - Finds the first hour where SepsisLabel = 1 (T_onset)
      - Sets EarlyLabel = 1 at T_onset - LABEL_SHIFT_HOURS and all later timesteps
      - Patients with T_onset <= LABEL_SHIFT_HOURS are excluded (no valid window)
      - Non-sepsis patients keep EarlyLabel = 0 throughout

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_all_patients(). Must contain columns:
        patient_id, ICULOS, SepsisLabel.

    Returns
    -------
    df_shifted : pd.DataFrame
        Full DataFrame with new 'EarlyLabel' column. Excluded patients removed.
    excluded_ids : list
        Patient IDs excluded due to sepsis onset within first LABEL_SHIFT_HOURS hours.

    Notes
    -----
    Vectorized implementation — uses groupby aggregation rather than a per-patient
    loop, so it runs in seconds on the full 40k-patient dataset.
    """
    df = df.copy()
    df['EarlyLabel'] = 0

    # ── Find T_onset for each sepsis patient ──────────────────────────────────
    sepsis_rows = df[df['SepsisLabel'] == 1]
    onset = (
        sepsis_rows
        .groupby('patient_id')['ICULOS']
        .min()
        .rename('t_onset')
    )

    # ── Identify patients to exclude (onset too early) ────────────────────────
    excluded_mask = onset <= LABEL_SHIFT_HOURS
    excluded_ids  = onset[excluded_mask].index.tolist()
    valid_onset   = onset[~excluded_mask]  # patients with a valid 6h window

    # ── Vectorized label shift ────────────────────────────────────────────────
    # Merge t_onset onto every row for valid sepsis patients, then threshold
    df = df.merge(valid_onset, on='patient_id', how='left')
    positive_mask = df['t_onset'].notna() & (df['ICULOS'] >= df['t_onset'] - LABEL_SHIFT_HOURS)
    df.loc[positive_mask, 'EarlyLabel'] = 1
    df = df.drop(columns=['t_onset'])

    # ── Remove excluded patients ──────────────────────────────────────────────
    df_shifted = df[~df['patient_id'].isin(excluded_ids)].reset_index(drop=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_remaining  = df_shifted['patient_id'].nunique()
    prevalence   = df_shifted['EarlyLabel'].mean() * 100
    n_sep_remain = df_shifted.groupby('patient_id')['EarlyLabel'].max().sum()

    print(f'Label shift      : {LABEL_SHIFT_HOURS} hours')
    print(f'Excluded patients: {len(excluded_ids):,}  (onset within first {LABEL_SHIFT_HOURS}h)')
    print(f'Remaining        : {n_remaining:,} patients')
    print(f'Sepsis patients  : {int(n_sep_remain):,}')
    print(f'Row-level prevalence (EarlyLabel=1): {prevalence:.2f}%')

    return df_shifted, excluded_ids


def clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set physiologically implausible values to NaN.

    Uses bounds defined in src/config.py OUTLIER_BOUNDS. Runs BEFORE imputation
    so both Strategy A and Strategy B receive clean starting data.

    Parameters
    ----------
    df : pd.DataFrame
        Output of engineer_labels(). Must contain the feature columns.

    Returns
    -------
    pd.DataFrame with implausible values replaced by NaN. All other values
    (including already-NaN entries) are unchanged.
    """
    df = df.copy()
    clip_counts = {}

    for feature, (low, high) in OUTLIER_BOUNDS.items():
        if feature not in df.columns:
            continue
        mask = df[feature].notna() & ((df[feature] < low) | (df[feature] > high))
        clip_counts[feature] = int(mask.sum())
        df.loc[mask, feature] = np.nan

    print('Outlier clipping summary:')
    total = 0
    for feat, count in clip_counts.items():
        pct = 100 * count / df[feat].notna().sum() if df[feat].notna().sum() > 0 else 0
        print(f'  {feat:<12}: {count:>5} values clipped  ({pct:.3f}% of observed)')
        total += count
    print(f'  {"TOTAL":<12}: {total:>5} values clipped')

    return df


def apply_strategy_A(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """
    Strategy A: median imputation + standard scaling.

    Fits imputer and scaler on training data only, then transforms all three
    sets. This prevents data leakage from val/test into the imputation step.

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
        Row-level DataFrames filtered to the corresponding patient split.
        Must contain ALL_FEATURES columns and 'EarlyLabel'.

    Returns
    -------
    X_train, X_val, X_test : np.ndarray  — imputed and scaled feature arrays
    y_train, y_val, y_test : np.ndarray  — EarlyLabel arrays
    feature_cols           : list[str]   — ordered feature names
    """
    feature_cols = [c for c in ALL_FEATURES if c in train_df.columns]

    X_train_raw = train_df[feature_cols].values
    X_val_raw   = val_df[feature_cols].values
    X_test_raw  = test_df[feature_cols].values

    y_train = train_df['EarlyLabel'].values
    y_val   = val_df['EarlyLabel'].values
    y_test  = test_df['EarlyLabel'].values

    # Step 1: Median imputation — fit on train only
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_val_imp   = imputer.transform(X_val_raw)
    X_test_imp  = imputer.transform(X_test_raw)

    # Step 2: Standard scaling — fit on train only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_imp)
    X_val   = scaler.transform(X_val_imp)
    X_test  = scaler.transform(X_test_imp)

    # Step 3: Save fitted objects for reproducibility
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(imputer, f'{MODELS_DIR}strategy_A_imputer.pkl')
    joblib.dump(scaler,  f'{MODELS_DIR}strategy_A_scaler.pkl')

    # Step 4: Validate no NaN remains
    print('Strategy A — NaN validation:')
    for arr, name in [(X_train, 'train_A'), (X_val, 'val_A'), (X_test, 'test_A')]:
        validate_no_nans(arr, name, feature_cols)

    print(f'Strategy A complete | features: {X_train.shape[1]} | '
          f'train rows: {X_train.shape[0]:,} | '
          f'train mean ≈ {X_train.mean():.4f} (expect ~0)')

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols
