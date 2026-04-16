"""
src/preprocessing.py
Label engineering and preprocessing utilities for the Sepsis ML pipeline.
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
    Shift SepsisLabel back by LABEL_SHIFT_HOURS to create a 6-hour early-warning target.

    Patients whose sepsis onset falls within the first LABEL_SHIFT_HOURS are excluded
    because there is no valid prediction window before onset. Returns the filtered
    DataFrame and the list of excluded patient IDs.
    """
    df = df.copy()
    df['EarlyLabel'] = 0

    sepsis_rows = df[df['SepsisLabel'] == 1]
    onset = (
        sepsis_rows
        .groupby('patient_id')['ICULOS']
        .min()
        .rename('t_onset')
    )

    excluded_mask = onset <= LABEL_SHIFT_HOURS
    excluded_ids  = onset[excluded_mask].index.tolist()
    valid_onset   = onset[~excluded_mask]  # patients with a valid 6h window

    df = df.merge(valid_onset, on='patient_id', how='left')
    positive_mask = df['t_onset'].notna() & (df['ICULOS'] >= df['t_onset'] - LABEL_SHIFT_HOURS)
    df.loc[positive_mask, 'EarlyLabel'] = 1
    df = df.drop(columns=['t_onset'])

    df_shifted = df[~df['patient_id'].isin(excluded_ids)].reset_index(drop=True)

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
    Set values outside OUTLIER_BOUNDS to NaN before imputation.

    Only affects observed values; already-missing entries are left unchanged.
    Bounds are defined per feature in config.py.
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
    Strategy A: median imputation then standard scaling.

    Imputer and scaler are fit on training data only to prevent leakage.
    Returns X_train, X_val, X_test, y_train, y_val, y_test, feature_cols.
    """
    feature_cols = [c for c in ALL_FEATURES if c in train_df.columns]

    X_train_raw = train_df[feature_cols].values
    X_val_raw   = val_df[feature_cols].values
    X_test_raw  = test_df[feature_cols].values

    y_train = train_df['EarlyLabel'].values
    y_val   = val_df['EarlyLabel'].values
    y_test  = test_df['EarlyLabel'].values

    # fit on train only to prevent leakage
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_val_imp   = imputer.transform(X_val_raw)
    X_test_imp  = imputer.transform(X_test_raw)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_imp)
    X_val   = scaler.transform(X_val_imp)
    X_test  = scaler.transform(X_test_imp)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(imputer, f'{MODELS_DIR}strategy_A_imputer.pkl')
    joblib.dump(scaler,  f'{MODELS_DIR}strategy_A_scaler.pkl')

    print('Strategy A — NaN validation:')
    for arr, name in [(X_train, 'train_A'), (X_val, 'val_A'), (X_test, 'test_A')]:
        validate_no_nans(arr, name, feature_cols)

    print(f'Strategy A complete | features: {X_train.shape[1]} | '
          f'train rows: {X_train.shape[0]:,} | '
          f'train mean ≈ {X_train.mean():.4f} (expect ~0)')

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


def apply_strategy_B(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """
    Strategy B: forward-fill within patient + binary missingness indicators.

    Indicators are added before filling so observed-vs-missing is preserved as a
    feature. First-timestep NaN (no prior value to fill from) falls back to
    training-set median. Scaler is fit on training data only.
    Returns X_train, X_val, X_test, y_train, y_val, y_test, feature_cols_B.
    """
    META_COLS = {'patient_id', 'hospital_id', 'timestep', 'SepsisLabel', 'EarlyLabel', 'ICULOS'}
    feature_cols = [c for c in ALL_FEATURES if c in train_df.columns]

    def add_indicators_and_ffill(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # indicators must be created before any filling
        for feat in feature_cols:
            if df[feat].isna().any():
                df[f'{feat}_missing'] = df[feat].isna().astype(np.int8)
        df[feature_cols] = df.groupby('patient_id')[feature_cols].ffill()
        return df

    train_proc = add_indicators_and_ffill(train_df)
    val_proc   = add_indicators_and_ffill(val_df)
    test_proc  = add_indicators_and_ffill(test_df)

    # val/test may be missing indicator columns if that feature had no NaN in that split
    feature_cols_B = [c for c in train_proc.columns if c not in META_COLS]
    for proc_df in (val_proc, test_proc):
        for col in feature_cols_B:
            if col not in proc_df.columns:
                proc_df[col] = 0

    X_train_raw = train_proc[feature_cols_B].values
    X_val_raw   = val_proc[feature_cols_B].values
    X_test_raw  = test_proc[feature_cols_B].values

    y_train = train_df['EarlyLabel'].values
    y_val   = val_df['EarlyLabel'].values
    y_test  = test_df['EarlyLabel'].values

    # first-timestep fallback — no prior value to forward-fill from
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_val_imp   = imputer.transform(X_val_raw)
    X_test_imp  = imputer.transform(X_test_raw)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_imp)
    X_val   = scaler.transform(X_val_imp)
    X_test  = scaler.transform(X_test_imp)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(imputer,       f'{MODELS_DIR}strategy_B_imputer.pkl')
    joblib.dump(scaler,        f'{MODELS_DIR}strategy_B_scaler.pkl')
    joblib.dump(feature_cols_B, f'{MODELS_DIR}strategy_B_feature_names.pkl')

    print('Strategy B — NaN validation:')
    for arr, name in [(X_train, 'train_B'), (X_val, 'val_B'), (X_test, 'test_B')]:
        validate_no_nans(arr, name, feature_cols_B)

    n_indicators = X_train.shape[1] - len(feature_cols)
    print(f'Strategy B complete | features: {X_train.shape[1]} '
          f'(40 original + {n_indicators} indicators) | '
          f'train rows: {X_train.shape[0]:,} | '
          f'train mean ≈ {X_train.mean():.4f} (expect ~0)')

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols_B
