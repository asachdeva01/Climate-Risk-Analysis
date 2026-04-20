# Student Exam Performance — Regression Analysis

Course project for **IEE 578: Regression Analysis** (Spring 2026, Dr. Douglas C. Montgomery, ASU).

**Team:** Abhi Sachdeva, Amanda Hightower, Abishek Balasubramanian.

## Problem
Model and predict student `Exam_Score` from academic, behavioral, and demographic factors using multiple linear regression. Identify the most statistically significant predictors via EDA and stepwise selection, and compare MLR against Ridge / Lasso.

## Data
[Kaggle — Student Exam Performance Factors](https://www.kaggle.com/datasets/grandmaster07/student-exam-performance-dataset-analysis). ~6600 student records, 20 columns (6 numeric, 13 categorical, 1 continuous target).

## Pipeline

```
data/                  # input CSV
data_cleaning/         # type inference, missing values, duplicates, outliers
feature_engineering/   # one-hot encoding, transforms, interactions
exploratory_analysis/  # eda.ipynb — justifies predictor selection
modeling/              # MLR / Ridge / Lasso + variable selection + diagnostics
reports/               # per-model metrics, coefficients, statsmodels summaries
```

## Setup

```bash
python3 -m venv climate
source climate/bin/activate
pip install -r requirements.txt
```

## Usage

**EDA (pick predictors):** open `exploratory_analysis/eda.ipynb`.

**Fit the pipeline (all three models, ranked by adjusted R²):**
```bash
python -m modeling.fit_model --data data/StudentPerformanceFactors.csv --target Exam_Score
```

The pipeline auto-infers numeric vs. categorical columns, one-hot encodes the categoricals, splits train/test, fits each model, runs diagnostics, and saves artifacts to `reports/`.

**Fit a specific model only:**
```bash
python -m modeling.fit_model --data data/StudentPerformanceFactors.csv --target Exam_Score --model mlr
```

## Dataset-agnostic
To run on a different CSV, just pass new `--data` and `--target` flags. The pipeline infers column types, encodes categoricals, and chooses the best model by adjusted R² automatically.
