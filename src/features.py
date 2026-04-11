"""
src/features.py
──────────────────────────────────────────────────────────────────────────────
Temporal lag feature engineering for XGBoost.

Usage:
    from src.features import add_lag_features
    train_df = add_lag_features(train_df)

Must be called AFTER the patient-level split and BEFORE imputation.
"""

import pandas as pd

from src.config import VITAL_SIGNS

# Number of lag features added per vital sign
_LAG_FEATURES_PER_VITAL = 6  # lag1, lag2, lag4, roll4_mean, roll4_std, delta1


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal lag features for each of the 8 vital signs.

    For each vital sign adds:
      {feat}_lag1       — value 1 hour ago
      {feat}_lag2       — value 2 hours ago
      {feat}_lag4       — value 4 hours ago
      {feat}_roll4_mean — rolling mean over last 4 hours (trend)
      {feat}_roll4_std  — rolling std over last 4 hours (volatility)
      {feat}_delta1     — current minus 1-hour-ago (rate of change)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'patient_id', 'ICULOS', and VITAL_SIGNS columns.
        Should be a single split (train, val, or test) — NOT the full dataset.

    Returns
    -------
    pd.DataFrame with 48 additional lag feature columns (8 vitals × 6 features).

    Notes
    -----
    - All lags are computed within each patient via groupby — no cross-patient bleed.
    - First 1–4 rows of each patient will have NaN for lag features (no prior history).
      These NaN values are handled downstream by Strategy A or B imputation.
    - roll4_mean and roll4_std use shift(1) before rolling so they exclude the current
      timestep (predicting from past values only, no look-ahead).
    """
    df = df.copy().sort_values(['patient_id', 'ICULOS']).reset_index(drop=True)

    for feat in VITAL_SIGNS:
        if feat not in df.columns:
            continue

        grp = df.groupby('patient_id')[feat]

        df[f'{feat}_lag1']       = grp.shift(1)
        df[f'{feat}_lag2']       = grp.shift(2)
        df[f'{feat}_lag4']       = grp.shift(4)
        df[f'{feat}_roll4_mean'] = grp.transform(
            lambda x: x.shift(1).rolling(4, min_periods=2).mean()
        )
        df[f'{feat}_roll4_std']  = grp.transform(
            lambda x: x.shift(1).rolling(4, min_periods=2).std()
        )
        df[f'{feat}_delta1']     = df[feat] - df[f'{feat}_lag1']

    n_new = len(VITAL_SIGNS) * _LAG_FEATURES_PER_VITAL
    print(f'Added {n_new} lag features ({len(VITAL_SIGNS)} vitals × {_LAG_FEATURES_PER_VITAL}). '
          f'Total columns: {len(df.columns)}')

    return df
