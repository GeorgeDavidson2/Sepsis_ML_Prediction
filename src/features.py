"""
src/features.py
Temporal lag feature engineering for XGBoost.
"""

import pandas as pd

from src.config import VITAL_SIGNS

# Number of lag features added per vital sign
_LAG_FEATURES_PER_VITAL = 6  # lag1, lag2, lag4, roll4_mean, roll4_std, delta1


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag1/lag2/lag4, rolling mean/std, and delta features for each vital sign.

    All operations are grouped by patient so there is no cross-patient leakage.
    NaN values at the start of each sequence are handled by the downstream imputer.
    roll4_mean and roll4_std shift(1) before rolling to avoid look-ahead.

    Returns a DataFrame with 48 additional columns (8 vitals × 6 features).
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
