"""
src/data_loader.py
Loads the raw PhysioNet/CinC 2019 dataset from individual .psv files into a
single unified DataFrame.
"""

import glob
import os

import pandas as pd
from tqdm import tqdm

from src.config import DATA_DIR, ALL_FEATURES

# Expected columns in the output DataFrame (43 total)
OUTPUT_COLUMNS = ['patient_id', 'hospital_id', 'timestep'] + ALL_FEATURES + ['SepsisLabel']


def load_all_patients(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Read all .psv patient files from Set A and Set B.
    """
    records = []

    for hospital, folder in [('A', 'training_setA'), ('B', 'training_setB')]:
        pattern = os.path.join(data_dir, folder, '*.psv')
        files   = sorted(glob.glob(pattern))

        if not files:
            print(f'WARNING: No .psv files found in {os.path.join(data_dir, folder)}')
            print('         Run: python src/download_data.py')
            continue

        print(f'Loading Set {hospital}: {len(files):,} files...')

        for filepath in tqdm(files, desc=f'Set {hospital}', unit='patient'):
            patient_id = os.path.splitext(os.path.basename(filepath))[0]

            try:
                pat_df = pd.read_csv(filepath, sep='|')
            except Exception as exc:
                print(f'WARNING: Could not read {filepath}: {exc}')
                continue

            pat_df['patient_id']  = patient_id
            pat_df['hospital_id'] = hospital
            pat_df['timestep']    = range(len(pat_df))
            records.append(pat_df)

    if not records:
        raise RuntimeError(
            f'No patient records loaded from {data_dir}. '
            'Check that the data has been downloaded.'
        )

    full_df = pd.concat(records, ignore_index=True)

    # OUTPUT_COLUMNS defines the canonical order; drop any extras silently
    present = [c for c in OUTPUT_COLUMNS if c in full_df.columns]
    full_df = full_df[present]

    print(
        f'\nLoaded {full_df["patient_id"].nunique():,} patients | '
        f'{len(full_df):,} total rows | '
        f'{full_df.shape[1]} columns'
    )
    print(f'Hospital A rows: {(full_df["hospital_id"]=="A").sum():,}')
    print(f'Hospital B rows: {(full_df["hospital_id"]=="B").sum():,}')

    return full_df
