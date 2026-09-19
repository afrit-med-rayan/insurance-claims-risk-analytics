# Insurance Claims Risk Analytics

End-to-end data analytics pipeline for French motor third-party liability (MTPL) insurance data.
Covers data cleaning, feature engineering, claim frequency and severity modeling, model explainability,
and a natural-language KPI assistant.

## Data

Source: French MTPL2 dataset (OpenML #41214 / #41215)

- `freMTPL2freq` - 677,991 motor insurance policies with claim counts and exposure
- `freMTPL2sev` - 26,639 individual claim records with claim amounts

Raw data files are excluded from version control. Place them in `data/raw/` before running the pipeline.

## Quick Start

```bash
git clone https://github.com/afrit-med-rayan/insurance-claims-risk-analytics.git
cd insurance-claims-risk-analytics
pip install -r requirements.txt
# place freMTPL2freq.arff and freMTPL2sev.arff in data/raw/
python src/run_pipeline.py
```

## Tech Stack

Python 3.10+, pandas, scikit-learn, LightGBM, SHAP, statsmodels, matplotlib, seaborn, Jupyter

## License

MIT
