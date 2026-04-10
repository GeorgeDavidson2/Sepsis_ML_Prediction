# src/config.py
# ─────────────────────────────────────────────────────────────────────────────
# SINGLE SOURCE OF TRUTH for all project settings.
# Every other file in src/ imports from here — nothing hardcoded elsewhere.
# ─────────────────────────────────────────────────────────────────────────────

import os

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42  # Applied everywhere: numpy, torch, sklearn, XGBoost

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR       = "data/raw/"
PROCESSED_DIR  = "data/processed/"
SPLITS_DIR     = "data/splits/"
RESULTS_DIR    = "results/"
MODELS_DIR     = "results/models/"
FIGURES_DIR    = "results/figures/"
METRICS_DIR    = "results/metrics/"
EXPERIMENT_LOG = "results/experiment_log.csv"

# ── Feature Groups (must match column names in the .psv files exactly) ────────
VITAL_SIGNS = [
    'HR',      # Heart rate
    'O2Sat',   # Pulse oximetry
    'Temp',    # Temperature
    'SBP',     # Systolic blood pressure
    'MAP',     # Mean arterial pressure
    'DBP',     # Diastolic blood pressure
    'Resp',    # Respiratory rate
    'EtCO2',   # End-tidal CO2
]

LAB_VALUES = [
    'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2',
    'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride',
    'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate',
    'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total',
    'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets',
]

DEMOGRAPHICS = [
    'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS',
]

ALL_FEATURES = VITAL_SIGNS + LAB_VALUES + DEMOGRAPHICS  # Exactly 40 features

# ── Outlier Clipping Bounds ───────────────────────────────────────────────────
# Values outside these ranges are physiologically impossible (data entry errors).
# They will be set to NaN before imputation — not dropped.
OUTLIER_BOUNDS = {
    'HR':      (5, 300),
    'O2Sat':   (50, 100),
    'Temp':    (25, 45),
    'SBP':     (40, 300),
    'DBP':     (20, 200),
    'MAP':     (30, 250),
    'Resp':    (4, 60),
    'Glucose': (20, 1000),
    'Lactate': (0.1, 30),
}

# ── Experiment Settings ───────────────────────────────────────────────────────
LABEL_SHIFT_HOURS = 6               # Predict sepsis this many hours before onset
MAX_SEQ_LEN       = 72              # Cap LSTM input sequences at 72 hours (3 days)
SPLIT_RATIOS      = (0.70, 0.15, 0.15)  # Train / val / test proportions

# ── Evaluation Settings ───────────────────────────────────────────────────────
N_BOOTSTRAP = 1000                  # Iterations for bootstrap confidence intervals
