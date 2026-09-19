# Insurance Claims Risk Analytics

End-to-end data analytics and predictive modeling pipeline for the French Motor Third-Party Liability (MTPL) insurance dataset. This project demonstrates full-stack data science capabilities, from raw data engineering to machine learning, model explainability (SHAP), and a Natural Language GenAI assistant.

## Overview
This repository contains a complete pipeline that:
1. Ingests and cleans actuarial data in ARFF format.
2. Engineers pricing-relevant features (Age bands, simulated premiums, claim frequency/severity).
3. Performs Exploratory Data Analysis (EDA) on key risk factors.
4. Trains baseline Generalized Linear Models (GLMs) and advanced LightGBM models for claim frequency and severity.
5. Explains model predictions using SHAP values.
6. Provides an interactive LLM-powered assistant (via Ollama) to answer business questions over the data.

## Data Source
The dataset is the **freMTPL2** dataset, a standard actuarial benchmark containing risk features and claim records for 678,000 motor insurance policies in France.
- `freMTPL2freq.arff`: Policy characteristics (exposure, vehicle power, driver age, bonus-malus) and claim counts.
- `freMTPL2sev.arff`: Individual claim amounts.

## Methodology
- **Data Engineering**: Data is parsed from WEKA ARFF formats, merged via left-joins, and cleaned by handling nulls, capping extreme outliers, and aggregating severity.
- **Modeling**: 
  - **Frequency**: Modeled using Poisson regression (baseline) and LightGBM (improved), weighted by policy exposure.
  - **Severity**: Modeled using a Gamma GLM (baseline) and LightGBM (improved) on policies with at least one claim.
- **Explainability**: SHAP (SHapley Additive exPlanations) is used to unpack the LightGBM models and extract top risk drivers.

## Key Findings
- **Bonus-Malus** is the strongest predictor of claim frequency, validating its use in the French market.
- **Young Drivers (18-25)** exhibit substantially higher claim frequencies than older cohorts.
- **Urban Concentration**: Regions R82 and R24 have the highest claim rates.
- **Vehicle Power**: Mid-to-high power vehicles (categories 9+) show elevated risk.
- **Severity**: Claim amounts are highly right-skewed, requiring log-transformations and Gamma distributions for accurate modeling.

## Live Dashboard
<!-- TODO: add Power BI publish link and screenshots here -->

## Predictive Model Results
- **Frequency (Poisson Deviance)**: LightGBM (0.345) outperformed baseline Poisson GLM (0.356) by 3%.
- **Severity (Gamma Deviance)**: Baseline Gamma GLM (1.503) slightly outperformed LightGBM (1.566), highlighting that simple GLMs often generalize better on heavy-tailed severity data, though both achieve a Log-MAE of around 0.85-1.02.

## GenAI Assistant
The repository includes a Natural Language query assistant powered by Ollama (or OpenAI) that answers business questions using pre-aggregated KPI tables.

**Example Query:**
> **Q:** Which driver age band has the highest claim frequency, and what is the rate?
> 
> **A:** The 18-25 driver age band has the highest claim frequency, with a rate of 0.170 claims per year.

## Tech Stack
- **Data & ML**: Python (Pandas, NumPy, Scikit-learn, LightGBM, Statsmodels)
- **Explainability**: SHAP
- **GenAI**: Ollama, OpenAI API
- **Testing & Code Quality**: Pytest

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/afrit-med-rayan/insurance-claims-risk-analytics.git
   cd insurance-claims-risk-analytics
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Placement**
   Place `freMTPL2freq.arff` and `freMTPL2sev.arff` in the `data/raw/` folder.

4. **Run the End-to-End Pipeline**
   ```bash
   python src/run_pipeline.py
   ```
   This will execute cleaning, feature engineering, EDA, modeling, and explainability in sequence.

5. **Run the GenAI Assistant**
   ```bash
   python src/genai_assistant.py
   ```

## Project Structure
```text
├── data/
│   ├── raw/                 # Raw ARFF files
│   └── processed/           # Cleaned and engineered features
├── notebooks/               # Jupyter notebooks (EDA, Modeling, Explainability)
├── reports/                 # Output metrics, findings, GenAI examples
│   └── figures/             # EDA and SHAP charts
├── src/                     # Source code modules
│   ├── load_data.py
│   ├── clean_data.py
│   ├── feature_engineering.py
│   ├── eda.py
│   ├── model_frequency.py
│   ├── model_severity.py
│   ├── explainability.py
│   ├── genai_assistant.py
│   └── run_pipeline.py
├── tests/                   # Pytest unit tests
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## License
MIT License
