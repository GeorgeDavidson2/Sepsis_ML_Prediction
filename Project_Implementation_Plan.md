# Project Implementation Plan
## Temporal Modeling vs. Structured Preprocessing: A Controlled Comparison for Early Sepsis Prediction
### George Arthur & Promise Owa | CS6140

---

## Experiment Design

We designed a controlled 2x2 experiment crossing two preprocessing strategies with two model families to isolate the contribution of each factor independently.

|  | Strategy A | Strategy B |
|---|---|---|
| **XGBoost** | Condition A | Condition B |
| **LSTM** | Condition C | Condition D |

---

## Dataset

PhysioNet/CinC 2019 Challenge dataset: 40,336 ICU patients across two hospital systems (Set A: 20,336; Set B: 20,000). Each patient is represented as an hourly time series of 40 clinical features (8 vital signs, 26 laboratory values, 6 demographic variables).

The SepsisLabel was shifted 6 hours before clinical onset to create the EarlyLabel early-warning target. Patients with onset within the first 6 hours (706 patients) were excluded. Final cohort: 39,630 patients, 5.62% sepsis prevalence.

Split: 70/15/15 stratified patient-level train/validation/test. The same split was used across all four conditions to ensure valid pairwise comparisons.

---

## Preprocessing Strategies

**Strategy A: Median Imputation**
Missing values replaced with per-feature training medians, followed by standard scaling. The imputer and scaler were fitted on the training set only and applied to validation and test sets.

**Strategy B: Forward-Fill with Missingness Indicators**
Per-patient forward-fill of the last observed value, with training median as fallback for the first timestep. Binary missingness indicator columns (one per original feature) were appended before scaling, encoding when each value was absent as an explicit model input. This preserves the MNAR signal identified in EDA: sepsis patients have consistently lower lab missingness rates than non-sepsis patients, meaning absent measurements are clinically informative.

---

## Feature Engineering

For XGBoost (Conditions A and B), 48 lag features were added after the patient-level split and before imputation: lag-1, lag-2, lag-4, 4-hour rolling mean, 4-hour rolling standard deviation, and 1-hour delta for each of the 8 vital signs. Final feature counts: Condition A = 88, Condition B = 125.

The LSTM (Conditions C and D) received raw padded sequences with no manual lag features, learning temporal dependencies directly from the data. Feature counts: Condition C = 40, Condition D = 76.

---

## Models

**XGBoost**
Gradient-boosted tree ensemble trained on per-timestep flat feature vectors. Class imbalance handled via `scale_pos_weight = 43.3`. Hyperparameters selected by grid search over depth {3, 4, 6} and learning rate {0.05, 0.1} with 500 estimators and early stopping (patience = 20) on validation AUPRC.

**LSTM**
Two-layer LSTM with a per-timestep linear output head, implemented in PyTorch using packed padded sequences. Sequences capped at 72 timesteps. BCEWithLogitsLoss with `pos_weight = 43.3`, Adam optimiser, gradient clipping at `max_norm = 1.0`, and early stopping (patience = 10) on validation AUPRC. Grid search over hidden size {64, 128}, dropout {0.2, 0.3}, and learning rate {0.001, 0.0005}.

---

## Evaluation

Primary metric: AUPRC, chosen for robustness to the 43:1 class imbalance. AUC-ROC reported as secondary metric. Decision thresholds selected by maximising F1 on the validation set. All metrics accompanied by 95% bootstrap confidence intervals (1,000 iterations, patient-level resampling).

SHAP TreeExplainer applied to both XGBoost conditions. Hospital generalizability assessed by stratifying the test set by hospital system (Set A vs Set B).

---

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Random seed | 42 everywhere | Reproducibility |
| Split | 70/15/15 patient-level stratified | Prevents temporal leakage; preserves class prevalence |
| Class imbalance | scale_pos_weight / pos_weight = 43.3 | Ratio of negative to positive training timesteps |
| Primary metric | AUPRC | More informative than AUC-ROC under severe class imbalance |
| Threshold | Validation-set F1 maximisation | Avoids defaulting to 0.5 on an imbalanced task |
| Sequence cap | 72 timesteps | Covers the clinically relevant window; prevents GPU memory issues |
| SHAP | XGBoost only | Applied where TreeExplainer is exact; LSTM interpretability is a separate problem |
