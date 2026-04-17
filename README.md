# Sepsis Early Prediction — CS6140 Machine Learning

Early prediction of sepsis in ICU patients using a controlled 2×2 experiment across two preprocessing strategies and two model architectures, trained on the PhysioNet/CinC 2019 dataset.

## Research Question

Does preprocessing strategy (how missing data is handled) or model architecture (XGBoost vs LSTM) have a greater impact on sepsis prediction performance?

## Repository Structure

```
Sepsis_ML_Prediction/
├── data/
│   ├── raw/                  # Raw .psv patient files (not tracked)
│   └── splits/               # Train/val/test patient ID lists
├── notebooks/
│   ├── 00_EDA.ipynb
│   ├── 01_preprocessing.ipynb
│   ├── 02_condition_A_baseline.ipynb
│   ├── 03_condition_B.ipynb
│   ├── 04_condition_C.ipynb
│   ├── 05_condition_D.ipynb
│   ├── 06_results_comparison.ipynb
│   ├── 07_shap_analysis.ipynb
│   └── 08_hospital_generalizability.ipynb
├── paper/
│   ├── main.tex
│   └── references.bib
├── results/
│   ├── figures/
│   ├── metrics/
│   ├── models/
│   └── experiment_log.csv
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── utils.py
│   ├── download_data.py
│   └── integrity_check.py
├── requirements.txt
└── Project_Overview.md
```

## Setup

**Python 3.13**

```bash
pip install -r requirements.txt
```

To download the dataset:

```bash
python src/download_data.py
```

This requires a Kaggle API token (`~/.kaggle/kaggle.json`). The raw files will be saved to `data/raw/`.

## Reproducing the Experiment

Run notebooks in order:

| Notebook | Description |
|---|---|
| `00_EDA.ipynb` | Exploratory data analysis |
| `01_preprocessing.ipynb` | Label engineering, outlier clipping, train/val/test split |
| `02_condition_A_baseline.ipynb` | XGBoost — Strategy A |
| `03_condition_B.ipynb` | XGBoost — Strategy B |
| `04_condition_C.ipynb` | LSTM — Strategy A |
| `05_condition_D.ipynb` | LSTM — Strategy B |
| `06_results_comparison.ipynb` | Cross-condition evaluation and plots |
| `07_shap_analysis.ipynb` | SHAP feature importance for XGBoost |
| `08_hospital_generalizability.ipynb` | Set A vs Set B performance |

Each notebook reads from `data/` and writes outputs to `results/`.

## Authors

George Arthur — arthur.ge@northeastern.edu  
Promise Owa — owa.p@northeastern.edu
