"""
Claim Severity Modeling.

Target: AvgSeverity
Dataset: Only rows where ClaimNb > 0 (approx 26K rows)
Features: Area, VehPower, VehAge, DrivAge, BonusMalus, VehBrand, VehGas, Density, Region

Trains a baseline Gamma GLM (statsmodels) and an improved LightGBM Gamma model.
Evaluation using Gamma Deviance and Mean Absolute Error on log-transformed target.
"""

import json
import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_gamma_deviance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_and_prep_data(path: str = None) -> tuple:
    """Load features, filter for claims > 0, and split."""
    if path is None:
        path = os.path.join("data", "processed", "features.csv")
        
    print(f"  loading data from {path} ...")
    df = pd.read_csv(path)
    
    # Filter to only policies with claims
    df_claims = df[df["ClaimNb"] > 0].copy()
    
    # Drop rows where AvgSeverity is <= 0 (invalid for Gamma)
    df_claims = df_claims[df_claims["AvgSeverity"] > 0]
    
    features = [
        "Area", "VehPower", "VehAge", "DrivAge", 
        "BonusMalus", "VehBrand", "VehGas", "Density", "Region"
    ]
    target = "AvgSeverity"
    
    X = df_claims[features]
    y = df_claims[target]
    
    # 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test


def build_preprocessor(drop_first: bool = True) -> ColumnTransformer:
    """
    Build scikit-learn preprocessor.
    drop_first=True is needed for statsmodels (avoid perfect collinearity).
    """
    categorical_cols = ["Area", "VehBrand", "VehGas", "Region"]
    numeric_cols = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop="first" if drop_first else None, sparse_output=False), categorical_cols)
        ]
    )
    return preprocessor


def train_baseline(X_train, y_train, preprocessor):
    """Train statsmodels GLM Gamma baseline."""
    print("  training baseline statsmodels Gamma GLM ...")
    X_train_prep = preprocessor.fit_transform(X_train)
    # Add constant for statsmodels
    X_train_prep = sm.add_constant(X_train_prep, has_constant='add')
    
    model = sm.GLM(y_train.values, X_train_prep, family=sm.families.Gamma(link=sm.families.links.log()))
    result = model.fit()
    return result


def predict_baseline(model, X_test, preprocessor):
    X_test_prep = preprocessor.transform(X_test)
    X_test_prep = sm.add_constant(X_test_prep, has_constant='add')
    return model.predict(X_test_prep)


def train_lgbm(X_train, y_train, preprocessor):
    """Train improved LightGBM Gamma model."""
    print("  training improved LightGBM model ...")
    X_train_prep = preprocessor.fit_transform(X_train)
    
    model = lgb.LGBMRegressor(
        objective="gamma",
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42
    )
    
    model.fit(X_train_prep, y_train)
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    return pipeline


def evaluate(name: str, y_true, y_pred) -> dict:
    """Evaluate using Gamma deviance and MAE on log target."""
    # Clip predictions slightly above 0 for stability
    y_pred = np.clip(y_pred, a_min=1e-3, a_max=None)
    
    gamma_dev = mean_gamma_deviance(y_true, y_pred)
    mae_log = mean_absolute_error(np.log1p(y_true), np.log1p(y_pred))
    
    print(f"    {name} Gamma Deviance: {gamma_dev:.4f} | Log-MAE: {mae_log:.4f}")
    return {"gamma_deviance": gamma_dev, "log_mae": mae_log}


def run() -> None:
    """Run severity modeling pipeline."""
    os.makedirs("reports", exist_ok=True)
    
    X_train, X_test, y_train, y_test = load_and_prep_data()
    
    # 1. Baseline (requires drop_first for OneHotEncoder)
    preprocessor_sm = build_preprocessor(drop_first=True)
    sm_model = train_baseline(X_train, y_train, preprocessor_sm)
    sm_preds = predict_baseline(sm_model, X_test, preprocessor_sm)
    base_metrics = evaluate("Baseline", y_test, sm_preds)
    
    # 2. Improved
    preprocessor_lgb = build_preprocessor(drop_first=False)
    lgbm_model = train_lgbm(X_train, y_train, preprocessor_lgb)
    lgbm_preds = lgbm_model.predict(X_test)
    lgbm_metrics = evaluate("LightGBM", y_test, lgbm_preds)
    
    # Save metrics
    metrics = {
        "baseline": base_metrics,
        "lgbm": lgbm_metrics,
        "improvement_pct_deviance": ((base_metrics["gamma_deviance"] - lgbm_metrics["gamma_deviance"]) / base_metrics["gamma_deviance"]) * 100
    }
    
    metrics_path = os.path.join("reports", "model_severity_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"  saved metrics -> {metrics_path}")
    
    # Save the best model
    model_path = os.path.join("reports", "lgbm_severity_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(lgbm_model, f)
    print(f"  saved best model -> {model_path}")


if __name__ == "__main__":
    run()
