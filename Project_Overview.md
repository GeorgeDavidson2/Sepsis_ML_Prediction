# Project Overview
## Sepsis Early Prediction Using Machine Learning
### George Arthur & Promise Owa | CS6140

---

## Background

Sepsis is a life-threatening condition in which the body's response to infection begins damaging its own organs. It affects 1.7 million Americans annually, with at least 350,000 deaths during hospitalization (CDC, 2024). Every hour of delayed treatment is associated with a 4 to 8 percent increase in mortality (Kumar et al., 2006).

ICU patients are already monitored continuously, generating hourly vital signs and laboratory measurements. This project uses that data to predict sepsis onset 6 hours before clinical diagnosis, giving clinicians an actionable early warning window.

---

## Research Question

In the studies we reviewed, preprocessing strategy and model architecture are typically varied together, making it difficult to attribute performance gains to either factor independently (Moor et al., 2021; Bomrah et al., 2024). This project addresses that directly.

We designed a controlled 2x2 experiment crossing two preprocessing strategies with two model families to answer: does better data handling matter more than model sophistication for early sepsis prediction?

---

## Experiment Design

Four conditions are defined as the Cartesian product of two preprocessing strategies and two model families:

|  | Strategy A (Median Imputation) | Strategy B (Forward-fill + Missingness Indicators) |
|---|---|---|
| **XGBoost** | Condition A | Condition B |
| **LSTM** | Condition C | Condition D |

**Strategy A** fills missing values with per-feature training medians and applies standard scaling. It treats missingness as uninformative noise.

**Strategy B** forward-fills each patient's last observed value and appends 40 binary missingness indicator columns, one per original feature. This encodes when values were absent as an explicit model input, based on the observation that lab orders are clinically driven rather than random.

**XGBoost** operates on a flat per-timestep feature vector with 48 manually engineered lag features across 8 vital signs, giving the model explicit access to recent temporal trends.

**LSTM** processes the full patient time series as a padded sequence up to 72 timesteps, learning temporal dependencies directly from the raw data without manual lag features.

---

## Dataset

PhysioNet/CinC 2019 Challenge dataset: 40,336 ICU patients across two hospital systems (Set A and Set B), each represented as an hourly time series of 40 clinical features. The SepsisLabel is shifted 6 hours before clinical onset to create the early-warning target. Patients with onset within the first 6 hours (706 patients) are excluded. Final cohort: 39,630 patients, 5.62% sepsis prevalence, split 70/15/15 by stratified patient-level sampling.

---

## Evaluation

Primary metric is AUPRC, which is robust to the 43:1 class imbalance and directly reflects the precision-recall trade-off relevant to clinical screening. AUC-ROC is reported as a secondary metric. All results are accompanied by 95% bootstrap confidence intervals (1,000 iterations, patient-level resampling). SHAP TreeExplainer is applied to both XGBoost conditions for feature attribution. Generalizability is assessed by stratifying the test set by hospital system.

---

## Hypothesis

We expect preprocessing strategy to be the stronger lever. Lab missingness in this dataset is not random: sepsis patients have consistently lower missingness rates than non-sepsis patients across all 26 lab values, meaning the pattern of absent measurements carries clinical signal. Strategy B encodes this signal explicitly. XGBoost with lag features already captures most of the temporal structure available in the data, so the marginal gain from switching to an LSTM may be smaller than the gain from switching preprocessing strategies.

Either outcome is informative. If the LSTM outperforms XGBoost regardless of preprocessing, that identifies model architecture as the dominant factor. If preprocessing drives the gap, that has direct implications for how future clinical prediction systems should be built.
