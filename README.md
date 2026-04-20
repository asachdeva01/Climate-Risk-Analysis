# Climate Risk Analysis — IEE 578 Regression Analysis Project

## Project Overview
Graduate-level regression analysis project for **IEE 578: Regression Analysis** (Spring 2026, Dr. Douglas C. Montgomery, ASU).

**Team:** Abhi Sachdeva, Amanda Hightower, Abishek Balasubramanian.

**Research objective:** Model and predict the `climate_risk_index` (composite 0–100) from country-level climate, environmental, and human-activity indicators using regression analysis.

**Data source:** Kaggle — [Climate Change and Global Warming](https://www.kaggle.com/datasets/algozee/climate-cahnge/data) (~1200 observations, 20 features, panel structure: country × year, early 1980s–2024).

**Response variable:** `climate_risk_index`

**Predictor variables (per project proposal):**
- `heatwave_days`, `drought_index`, `flood_events_count`
- `deforestation_rate`, `fossil_fuel_consumption`, `co2_concentration_ppm`
- `renewable_energy_share`, `forest_cover_percent`, `air_quality_index`

Additional climate variables may be considered. Variable selection techniques determine the final subset.

---

## Methodology (per proposal, Section 3.4)
1. **EDA** — distributions, trends, outliers, predictor–response relationships
2. **Multiple Linear Regression** — primary framework (continuous response)
3. **Variable Selection** — stepwise, adjusted R², significance testing
4. **Model Diagnostics** — linearity, homoscedasticity, normality, multicollinearity
5. **Model Comparison** — alternative approaches (Ridge, Lasso) if MLR underperforms
6. **Prediction & Validation** — train/test split, MSE / MAE / R²

**Primary library:** `statsmodels` (inference: p-values, CIs, diagnostics).
**Secondary:** `scikit-learn` (train/test split, regularized models, metrics).

---

## Repository Structure
```
Climate-Risk-Analysis/
├── data/                    # Kaggle CSV lives here (public data, tracked in git)
├── data_cleaning/           # Preprocessing pipeline (types, missing, dupes, outliers)
├── feature_engineering/     # Derived features (transforms, encodings, interactions)
├── exploratory_analysis/    # EDA notebook + stats/visualization helpers
├── modeling/                # Regression pipeline (one model per helper file)
│   └── helpers/
│       ├── mlr.py           # Multiple linear regression (statsmodels OLS)
│       ├── ridge.py         # Ridge regression (comparison model)
│       ├── lasso.py         # Lasso regression (comparison model)
│       ├── selection.py     # Variable selection (stepwise, AIC/BIC)
│       ├── diagnostics.py   # Residual plots, VIF, normality, homoscedasticity
│       ├── metrics.py       # MSE, MAE, R², adj R²
│       ├── validation.py    # Train/test split
│       └── save_outputs.py  # Persist summaries/metrics/coefficients to reports/
└── reports/
    ├── model_outputs/       # {model}_metrics.json, {model}_coefficients.csv
    └── summary/             # {model}_summary.txt (statsmodels pretty printout)
```

Each top-level directory has an **entry point script** that composes helpers. Helpers are split by core task so no single file grows too large (matches the style of the IDX-Exchange portfolio).

---

## Coding Conventions
- Python package structure (underscores in directory names, `__init__.py` in each module).
- One responsibility per helper file.
- Entry points (`preprocess.py`, `add_new_features.py`, `fit_model.py`) only orchestrate — logic lives in `helpers/`.
- Docstrings on every public function describing inputs, outputs, and the statistical/data step it performs.
- `statsmodels` is preferred for regression fits that require inference; `sklearn` for regularized models and validation splits.
