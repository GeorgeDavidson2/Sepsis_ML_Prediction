# CS6140 — Comprehensive Project Implementation Plan
## Temporal Modeling vs. Structured Preprocessing: A Controlled Comparison for Early Sepsis Prediction
### George Arthur & Promise Owa

---

## Master Checklist (High-Level)

- [ ] Phase 0 — Environment & Project Structure
- [ ] Phase 1 — Data Acquisition & Integrity Check
- [ ] Phase 2 — Exploratory Data Analysis (EDA)
- [ ] Phase 3 — Data Pipeline (Label Engineering + Both Preprocessing Strategies)
- [ ] Phase 4 — Feature Engineering (Lag Features for XGBoost)
- [ ] Phase 5 — Model Implementation (All 4 Conditions)
- [ ] Phase 6 — Training Protocol & Hyperparameter Decisions
- [ ] Phase 7 — Evaluation Framework (All Metrics + 2×2 Comparison)
- [ ] Phase 8 — Interpretability (SHAP + Calibration)
- [ ] Phase 9 — Results, Figures & Write-up

---

## Phase 0 — Environment & Project Structure

### Goal
Set up a clean, reproducible environment before touching any data or code.

### Folder Structure
```
Sepsis_ML_Prediction/
│
├── data/
│   ├── raw/                  # Original .psv files — never modified
│   │   ├── training_setA/
│   │   └── training_setB/
│   ├── processed/            # Preprocessed outputs saved here
│   │   ├── strategy_A_simple/
│   │   └── strategy_B_missingness_aware/
│   └── splits/               # Patient-level train/val/test index files
│
├── notebooks/
│   ├── 00_EDA.ipynb
│   ├── 01_preprocessing.ipynb
│   ├── 02_condition_A_baseline.ipynb
│   ├── 03_condition_B.ipynb
│   ├── 04_condition_C.ipynb
│   ├── 05_condition_D.ipynb
│   └── 06_evaluation_comparison.ipynb
│
├── src/
│   ├── data_loader.py        # Load and merge all .psv files
│   ├── preprocessing.py      # Both imputation strategies as functions
│   ├── features.py           # Lag feature engineering for XGBoost
│   ├── models.py             # Model definitions (LR, XGBoost, LSTM)
│   ├── train.py              # Training loops
│   ├── evaluate.py           # All metric computations
│   └── utils.py              # Shared helpers (seeds, splits, etc.)
│
├── results/
│   ├── metrics/              # CSV files with per-condition metrics
│   ├── figures/              # All plots (ROC curves, SHAP, calibration)
│   └── models/               # Saved model checkpoints
│
├── Project_Implementation_Plan.md
├── Lecturer_Feedback_Response.md
└── requirements.txt
```

### Libraries to Install
```
pandas
numpy
scikit-learn
xgboost
torch              # For LSTM (PyTorch)
shap
matplotlib
seaborn
scipy
tqdm
joblib
```

### Reproducibility Rules
- Set a **global random seed = 42** everywhere (numpy, torch, sklearn, XGBoost)
- Save the exact train/val/test patient ID splits to `data/splits/` as CSV files
- Never overwrite raw data files
- Save all trained models to `results/models/` with descriptive names
  (e.g., `xgboost_condition_B.pkl`, `lstm_condition_C.pt`)

---

## Phase 1 — Data Acquisition & Integrity Check

### Goal
Download the dataset and verify it is complete and uncorrupted before any processing.

### Steps

**1.1 — Download**
- Primary source: https://physionet.org/content/challenge-2019/1.0.0/
- Kaggle mirror (easier): https://kaggle.com/datasets/salikhussaini49/prediction-of-sepsis
- Expected: ~20,336 `.psv` files in Set A, ~20,000 in Set B

**1.2 — Integrity Checks**
Run these checks and log the results before proceeding:
- Total file count: should be ~40,336
- Column names: every file should have the same 41 columns (40 features + SepsisLabel)
- Confirm the 40 columns: 8 vitals, 26 labs, 6 demographics + SepsisLabel
- Check that SepsisLabel is binary (0 or 1 only)
- Check that ICULOS (ICU length of stay) is monotonically increasing per patient

**1.3 — Quick Stats**
- How many unique patients total?
- How many have at least one SepsisLabel = 1?
- What is the overall sepsis prevalence (target: ~5.6%)?
- What is the distribution of patient stay lengths (min, max, median hours)?

**Key Risk:** Some files may be malformed or have inconsistent columns. Flag these and exclude them with logging rather than silently dropping.

---

## Phase 2 — Exploratory Data Analysis (EDA)

### Goal
Understand the data deeply before modeling. EDA findings directly inform preprocessing decisions and feature engineering choices.

### Steps

**2.1 — Missingness Analysis**
- Compute % missing per feature across all patients
- Expected: lab values (lactate, creatinine, bilirubin, etc.) will have >90% missing
- Vital signs will have <20% missing
- Plot a missingness heatmap (features × patient timeline)
- **Key question:** Is missingness random (MCAR) or informative (MNAR)?
  - Compute: are sicker patients (SepsisLabel=1) missing *more* or *fewer* lab values?
  - If yes → missingness is MNAR → missingness indicator columns will carry signal

**2.2 — Class Imbalance**
- Count SepsisLabel distribution: target ~5.6% positive
- Plot distribution of time-to-sepsis (how many hours before onset do we have data for each patient?)

**2.3 — Feature Distributions**
- Plot histograms for each vital sign: heart rate, temperature, BP, respiratory rate
- Look for extreme outliers (e.g., heart rate = 0, temperature = 0) that indicate data entry errors
- These outliers should be clipped or treated as missing before imputation

**2.4 — Temporal Patterns**
- Plot average vital sign trajectories for sepsis patients vs. non-sepsis patients over time
- This is the visual hypothesis check: do sepsis patients show measurably different trends?
- If trends diverge hours before sepsis onset → validates the temporal modeling hypothesis

**2.5 — Correlation Analysis**
- Compute feature correlation matrix
- Identify redundant features (high pairwise correlation > 0.95)
- Note: do NOT drop features based on this — just be aware for interpreting SHAP values later

**EDA Outputs (save these):**
- `figures/missingness_heatmap.png`
- `figures/class_distribution.png`
- `figures/vital_trajectories_sepsis_vs_not.png`
- `figures/feature_correlation_matrix.png`
- A written summary paragraph of key EDA findings to include in the final report

---

## Phase 3 — Data Pipeline

### Goal
Build the complete data pipeline: loading → label engineering → preprocessing → splitting.
This is the foundation of the entire project. Get it right before touching any model.

### 3.1 — Data Loading

```python
# Pseudocode — implement in src/data_loader.py

def load_all_patients(data_dir):
    """
    Reads all .psv files from Set A and Set B.
    Returns a single DataFrame with columns:
    [patient_id, timestep, feature_1, ..., feature_40, SepsisLabel]
    """
    records = []
    for filepath in glob(data_dir + '/**/*.psv'):
        patient_id = extract_id_from_filename(filepath)
        df = pd.read_csv(filepath, sep='|')
        df['patient_id'] = patient_id
        df['timestep'] = range(len(df))
        records.append(df)
    return pd.concat(records, ignore_index=True)
```

### 3.2 — Label Engineering (Critical)

The task is to predict sepsis **6 hours before onset**.

```
For each patient:
  - Find the first timestep where SepsisLabel = 1 (call it T_onset)
  - Set a positive label at timestep T_onset - 6
  - All timesteps before T_onset - 6 are labeled 0
  - All timesteps AT or AFTER T_onset - 6 are labeled 1
  - For patients who never develop sepsis: all labels = 0
```

**Edge Case — Short ICU Stays:**
- If a patient's first SepsisLabel=1 occurs at timestep 3 (i.e., T_onset < 6), there is no
  valid 6-hour prediction window. These patients should be EXCLUDED from the dataset and
  their exclusion should be logged and reported.

**Verify after label engineering:**
- New sepsis prevalence (will be slightly different from raw 5.6%)
- How many patients were excluded due to short stay?

### 3.3 — Patient-Level Train/Val/Test Split (Critical)

**Rule:** Split at the patient level — NEVER at the timestep level.

```
Train:      70% of patients  → used for model training
Validation: 15% of patients  → used for threshold tuning, early stopping
Test:       15% of patients  → used ONLY for final reported metrics
```

**Steps:**
1. Get list of all unique patient IDs
2. Shuffle with random seed = 42
3. Split 70/15/15
4. Save the three lists of patient IDs to `data/splits/train_ids.csv`, `val_ids.csv`, `test_ids.csv`
5. These splits are FIXED for all 4 experimental conditions — this is what makes the 2×2 comparison valid

**Why this matters:** If a patient's hourly rows appear in both train and test, the model
has effectively seen that patient before, artificially inflating all performance metrics.

### 3.4 — Preprocessing Strategy A: Simple Median Imputation

```
For each feature:
  1. Compute the median value across ALL training patients only (never use test data)
  2. Fill missing values in train, val, and test with that training median
  3. Apply standard scaling (zero mean, unit variance) using training statistics only
```

**Implementation notes:**
- Fit the median imputer and scaler on TRAIN set only
- Transform val and test using the fitted imputer/scaler (no data leakage)
- Save fitted transformers to `results/models/strategy_A_imputer.pkl`

### 3.5 — Preprocessing Strategy B: Missingness-Aware (Forward-Fill + Indicator Columns)

```
Step 1 — Create binary missingness indicator columns:
  For each feature f with missing values:
    Create a new column: f_missing (1 if f was missing at that timestep, 0 otherwise)
  This doubles the feature count for the missing features.

Step 2 — Forward-fill within each patient's time series:
  For each patient, fill missing values with the last known value (forward fill)
  For timesteps before the first observation (no prior value exists):
    Fill with the training median as a fallback

Step 3 — Standard scaling:
  Same as Strategy A — fit on training set, transform all sets
```

**Why this matters:** Forward-fill preserves the last known clinical state (clinically
meaningful) while the indicator column tells the model "this value wasn't freshly measured
here" (also clinically meaningful). Median imputation destroys both signals.

**Save fitted transformers to** `results/models/strategy_B_imputer.pkl`

---

## Phase 4 — Feature Engineering (XGBoost Lag Features)

### Goal
Give XGBoost limited temporal awareness via manually engineered lag features,
without using a sequence architecture.

### Features to Engineer (per vital sign)

For the 8 vital signs (heart rate, pulse ox, temperature, systolic BP,
diastolic BP, mean BP, respiratory rate, EtCO2):

```
- value_1h_ago        (lag-1)
- value_2h_ago        (lag-2)
- value_4h_ago        (lag-4)
- rolling_mean_4h     (mean of last 4 timesteps)
- rolling_std_4h      (std of last 4 timesteps — captures volatility)
- delta_1h            (current value minus 1h ago — rate of change)
```

**Implementation rules:**
- Compute lags WITHIN each patient's time series only (never bleed across patients)
- Timesteps without enough prior history (e.g., first 4 rows of a patient's stay)
  should have NaN for lag features — these get imputed with strategy A or B accordingly
- Apply lag feature engineering AFTER the patient-level split and BEFORE imputation

**Note:** Logistic Regression will NOT use these lag features. It operates on the
current timestep's flat feature vector only. XGBoost uses them in both Condition A and B.

---

## Phase 5 — Model Implementation

### The 2×2 Matrix

|                          | Strategy A (Simple)      | Strategy B (Missingness-Aware) |
|--------------------------|--------------------------|-------------------------------|
| **LR + XGBoost**         | Condition A (baseline)   | Condition B                   |
| **LSTM**                 | Condition C              | Condition D                   |

### 5.1 — Condition A & B: Logistic Regression

**Purpose:** Interpretable floor baseline. No temporal information whatsoever.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    class_weight='balanced',   # Handles 5.6% class imbalance
    max_iter=1000,
    C=1.0,                     # Tune via validation set
    solver='lbfgs'
)
```

- Input: flat feature vector at each timestep independently
- No lag features
- No sequence structure
- Train on all timestep rows from training patients

### 5.2 — Condition A & B: XGBoost

**Purpose:** Strong tabular baseline with limited temporal awareness via lag features.

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    scale_pos_weight=17,       # ~(100-5.6)/5.6 — handles class imbalance
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='aucpr',       # AUPRC — better than AUC under imbalance
    early_stopping_rounds=50,
    random_state=42
)
```

- Input: flat feature vector + lag features at each timestep
- `scale_pos_weight`: ratio of negative to positive samples — critical for imbalance
- Use validation set for early stopping

### 5.3 — Condition C & D: LSTM

**Purpose:** Genuine sequence model that reads the full hourly patient timeline.

**Architecture:**
```
Input:  (batch_size, max_seq_len, n_features)
        → LSTM layer 1: hidden_size=64, dropout=0.3
        → LSTM layer 2: hidden_size=32, dropout=0.3
        → Linear layer: 32 → 1
        → Sigmoid activation
Output: Risk score at each timestep (many-to-many)
```

**Key implementation details:**

*Variable-length sequences:*
- Pad all patient sequences to the length of the longest patient in the batch
- Use `torch.nn.utils.rnn.pack_padded_sequence` and `pad_packed_sequence`
  to mask padded timesteps — they must not contribute to the loss

*Class imbalance:*
```python
# Positive weight for BCEWithLogitsLoss
pos_weight = torch.tensor([n_negative / n_positive])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

*Training loop:*
```
Optimizer: Adam, lr=1e-3
Scheduler: ReduceLROnPlateau (monitor validation AUPRC)
Epochs: up to 50 with early stopping (patience=10)
Batch size: 32 patients per batch
```

*Sequence construction:*
- Each training example is ONE FULL PATIENT SEQUENCE (not individual timesteps)
- Sort batch by sequence length (descending) for efficient packing

---

## Phase 6 — Training Protocol

### Preventing Data Leakage — Checklist

Before training any model, verify all of the following:
- [ ] Imputer/scaler fitted on TRAIN set only
- [ ] Patient IDs in train, val, test are disjoint (no overlap)
- [ ] Lag features computed within-patient only
- [ ] Label shift applied correctly (T_onset - 6)
- [ ] Short-stay patients excluded before splitting
- [ ] Same train/val/test split used for all 4 conditions

### Training Order

1. **Condition A first** — get the full pipeline working end-to-end
2. Verify metrics on Condition A are reasonable (AUC > 0.70 on validation)
3. Then build Condition B (same models, swap preprocessing)
4. Then build Condition C (LSTM with Strategy A preprocessing)
5. Then build Condition D (LSTM with Strategy B preprocessing)

### Hyperparameter Decisions

| Model | Parameter | Search Range | Method |
|---|---|---|---|
| Logistic Regression | C (regularization) | [0.01, 0.1, 1.0, 10.0] | Grid search on val set |
| XGBoost | max_depth | [4, 6, 8] | Grid search on val set |
| XGBoost | learning_rate | [0.01, 0.05, 0.1] | Grid search on val set |
| LSTM | hidden_size | [32, 64, 128] | Manual tuning on val AUPRC |
| LSTM | dropout | [0.2, 0.3, 0.5] | Manual tuning on val AUPRC |
| LSTM | learning_rate | [1e-4, 1e-3, 5e-3] | Manual tuning |

**Rule:** All hyperparameter decisions are made using the VALIDATION set.
The TEST set is touched exactly ONCE — for final reported results.

---

## Phase 7 — Evaluation Framework

### Primary Metrics (report for all 4 conditions)

| Metric | Why It Matters | Tool |
|---|---|---|
| **AUC-ROC** | Threshold-independent discrimination | `sklearn.metrics.roc_auc_score` |
| **AUPRC** | Better than AUC under severe class imbalance | `sklearn.metrics.average_precision_score` |
| **Sensitivity (Recall)** | Missing a sepsis patient is catastrophic | `sklearn.metrics.recall_score` |
| **Specificity** | Too many false alarms → alert fatigue | Compute from confusion matrix |
| **Clinical Utility Score** | PhysioNet challenge metric — penalizes late predictions | Implement from challenge paper |

### Threshold Selection

Do NOT use the default 0.5 threshold. Instead:
- On the validation set, sweep thresholds from 0.01 to 0.99
- Select the threshold that maximizes: **F-beta score (beta=2)** — weights recall higher
  than precision, appropriate for a medical alarm system
- Apply that selected threshold when reporting sensitivity/specificity on the TEST set

### The 2×2 Comparison Table

Report all metrics in this format:

|  | Strategy A (Simple) | Strategy B (Missingness-Aware) | Effect of Preprocessing |
|---|---|---|---|
| **LR + XGBoost** | AUC, AUPRC, Recall, Specificity | AUC, AUPRC, Recall, Specificity | B − A |
| **LSTM** | AUC, AUPRC, Recall, Specificity | AUC, AUPRC, Recall, Specificity | D − C |
| **Effect of Temporal** | C − A | D − B | — |

This table is the core result of the paper. Every number in it answers a specific question.

### Clinical Utility Score Implementation

From Reyna et al. (2020) — the scoring function rewards early predictions and
penalizes missed or late ones:

```python
def compute_utility_score(labels, predictions, dt_early=12, dt_optimal=6, dt_late=3):
    """
    labels: array of true labels (0/1) per timestep
    predictions: array of predicted labels (0/1) per timestep (after thresholding)
    dt_early: hours before onset where early prediction is rewarded
    dt_optimal: optimal prediction window (6h before onset)
    dt_late: hours after onset where late prediction is partially credited
    Returns: utility score (higher is better)
    """
    # Implementation follows the official PhysioNet challenge scoring function
    # Reference: https://github.com/physionetchallenges/python-challenge-2019
    pass
```

Use the official scoring code from:
https://github.com/physionetchallenges/python-challenge-2019

### Plots to Generate

- [ ] ROC curves for all 4 conditions on one plot (with AUC in legend)
- [ ] Precision-Recall curves for all 4 conditions on one plot
- [ ] Bar chart of sensitivity and specificity across 4 conditions
- [ ] Calibration plot (reliability diagram) for all 4 conditions
- [ ] Confusion matrix for each condition (at selected threshold)

---

## Phase 8 — Interpretability

### 8.1 — SHAP Analysis (XGBoost)

Run SHAP on XGBoost under BOTH preprocessing strategies (Condition A and B):

```python
import shap

explainer = shap.TreeExplainer(xgboost_model)
shap_values = explainer.shap_values(X_test)

# Plot 1: Global feature importance (bar chart of mean |SHAP|)
shap.summary_plot(shap_values, X_test, plot_type='bar')

# Plot 2: SHAP beeswarm — shows direction and magnitude
shap.summary_plot(shap_values, X_test)
```

**Key questions to answer with SHAP:**
- Which clinical features matter most (e.g., lactate, heart rate)?
- Under Strategy B, do the missingness indicator columns appear in the top features?
  If yes → strong evidence that missingness is genuinely informative
- Do the top features align with known clinical sepsis markers?

### 8.2 — Calibration Analysis

```python
from sklearn.calibration import calibration_curve

for condition, (y_true, y_prob) in results.items():
    fraction_positives, mean_predicted = calibration_curve(y_true, y_prob, n_bins=10)
    plt.plot(mean_predicted, fraction_positives, label=condition)

plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration Plot — All 4 Conditions')
```

A well-calibrated model means: when it says 80% probability of sepsis, ~80% of
those cases actually develop sepsis. This matters for clinical trust.

---

## Phase 9 — Results & Write-up

### Results Section Structure

1. **Dataset summary** — final patient counts after exclusions, class balance
2. **EDA highlights** — key missingness patterns, temporal divergence findings
3. **Main 2×2 results table** — all metrics across all 4 conditions
4. **Pairwise comparisons** — explicit sentences interpreting each comparison:
   - A vs B: "Missingness-aware preprocessing improved AUC-ROC by X..."
   - A vs C: "Temporal modeling improved AUC-ROC by Y..."
   - B vs D: "After optimal preprocessing, LSTM adds/does not add Z..."
   - A vs D: "Combined effect is W..."
5. **SHAP analysis** — which features drive predictions, do indicators show up?
6. **Calibration** — which model is best calibrated?
7. **Clinical utility score** — how do results compare to published challenge results?

### Limitations to Acknowledge

- Vanilla LSTM may not fully represent the potential of temporal modeling
- Results are on a single dataset from two hospital systems (generalizability unknown)
- Hyperparameter search was limited by compute budget
- No subgroup analysis by age or demographics

### Comparison to Published Results

Reyna et al. (2020) reported best challenge AUC-ROC of ~0.83 on the hidden test set.
Your XGBoost with Strategy B should be in the 0.78–0.83 range on the public test set.
If it isn't — diagnose before reporting (likely a data leakage or label issue).

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| LSTM fails to converge | Medium | High | Use GRU as backup; try smaller model |
| Data leakage discovered late | Low | Critical | Verify split integrity in Phase 3 before training |
| Class imbalance tanking recall | Medium | High | Tune threshold on val set, not 0.5 default |
| Short ICU stay edge cases | Medium | Medium | Log and exclude, report count |
| Compute time too long | Medium | Medium | Train on Set A only first; use Kaggle GPU |
| Missing utility score implementation | Low | Medium | Use official PhysioNet Python scoring code |

---

## Timeline Suggestion

| Phase | Estimated Effort | Notes |
|---|---|---|
| Phase 0 — Setup | 0.5 days | Do this together, once |
| Phase 1 — Data | 0.5 days | Mostly download + verify |
| Phase 2 — EDA | 1–2 days | Most important for understanding |
| Phase 3 — Pipeline | 2–3 days | Hardest phase — do not rush |
| Phase 4 — Features | 0.5 days | Lag features for XGBoost |
| Phase 5 — Models | 2–3 days | Build one condition at a time |
| Phase 6 — Training | 1–2 days | Hyperparameter tuning |
| Phase 7 — Evaluation | 1 day | Run after all 4 conditions done |
| Phase 8 — Interpret | 1 day | SHAP + calibration |
| Phase 9 — Write-up | 2–3 days | Leave enough time here |

---

## Key Design Decisions Summary

| Decision | Choice | Reason |
|---|---|---|
| Random seed | 42 (everywhere) | Reproducibility |
| Train/val/test | 70/15/15 patient-level | Strict leakage prevention |
| Imbalance — LR | class_weight='balanced' | Avoids accuracy paradox |
| Imbalance — XGBoost | scale_pos_weight | Native XGBoost parameter |
| Imbalance — LSTM | pos_weight in BCEWithLogitsLoss | PyTorch native |
| Threshold | Val-set optimized (F-beta=2) | Prioritizes recall for clinical safety |
| Primary metric | AUPRC | More informative than AUC under imbalance |
| Temporal model | LSTM (vanilla → upgrade to GRU if unstable) | Fits 2×2 design |
| SHAP | XGBoost only | LSTM interpretability is a separate research area |
