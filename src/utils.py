"""
src/utils.py
──────────────────────────────────────────────────────────────────────────────
Shared utilities for the Sepsis ML pipeline.

Usage:
    from src.utils import create_patient_splits
    train_ids, val_ids, test_ids = create_patient_splits(df)
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_SEED, SPLIT_RATIOS, SPLITS_DIR


def create_patient_splits(df: pd.DataFrame) -> tuple[list, list, list]:
    """
    Create and save a stratified patient-level 70/15/15 train/val/test split.

    Stratification ensures the sepsis prevalence is preserved proportionally
    across all three sets. Split is performed at the patient level — no patient's
    rows appear in more than one set (prevents data leakage).

    Parameters
    ----------
    df : pd.DataFrame
        Output of engineer_labels(). Must contain 'patient_id' and 'EarlyLabel'.

    Returns
    -------
    train_ids, val_ids, test_ids : lists of patient ID strings

    Saves
    -----
    data/splits/train_ids.csv
    data/splits/val_ids.csv
    data/splits/test_ids.csv
    """
    # One row per patient: 1 if they ever develop sepsis, 0 otherwise
    patient_labels = (
        df.groupby('patient_id')['EarlyLabel']
        .max()
        .reset_index()
        .rename(columns={'EarlyLabel': 'is_sepsis'})
    )

    patient_ids = patient_labels['patient_id'].values
    y_patients  = patient_labels['is_sepsis'].values

    train_size, val_size, test_size = SPLIT_RATIOS  # (0.70, 0.15, 0.15)

    # First split: 70% train, 30% temp (val + test combined)
    train_ids, temp_ids, _, temp_y = train_test_split(
        patient_ids, y_patients,
        test_size=(val_size + test_size),
        stratify=y_patients,
        random_state=RANDOM_SEED,
    )

    # Second split: split temp evenly into val and test
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.5,
        stratify=temp_y,
        random_state=RANDOM_SEED,
    )

    # Save ID lists to CSV
    os.makedirs(SPLITS_DIR, exist_ok=True)
    pd.Series(train_ids).to_csv(f'{SPLITS_DIR}train_ids.csv', index=False, header=['patient_id'])
    pd.Series(val_ids).to_csv(  f'{SPLITS_DIR}val_ids.csv',   index=False, header=['patient_id'])
    pd.Series(test_ids).to_csv( f'{SPLITS_DIR}test_ids.csv',  index=False, header=['patient_id'])

    # Print class balance across all three sets
    print(f'Split complete  (seed={RANDOM_SEED}, ratios={train_size}/{val_size}/{test_size})')
    print(f'{"Set":<8} {"Patients":>10} {"Sepsis":>10} {"Prevalence":>12}')
    print('-' * 44)
    for name, ids in [('Train', train_ids), ('Val', val_ids), ('Test', test_ids)]:
        subset = patient_labels[patient_labels['patient_id'].isin(ids)]
        n_sep  = subset['is_sepsis'].sum()
        pct    = subset['is_sepsis'].mean() * 100
        print(f'{name:<8} {len(ids):>10,} {int(n_sep):>10,} {pct:>11.2f}%')
    print(f'{"TOTAL":<8} {len(patient_ids):>10,}')

    return list(train_ids), list(val_ids), list(test_ids)
