"""
Claim Frequency Modeling.

Target: ClaimNb
Features: Area, VehPower, VehAge, DrivAge, BonusMalus, VehBrand, VehGas, Density, Region
Offset/Weight: Exposure

Trains a baseline Poisson GLM (scikit-learn) and an improved LightGBM Poisson model.
Evaluation is done using Poisson Deviance (not RMSE, as variance is proportional to mean).
"""

import json
import os
import pickle

import lightgbm as lgb
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_and_prep_data(path: str = None) -> tuple:
    """Load features and split into train/test sets."""
    if path is None:
        path = os.path.join("data", "processed", "features.csv")
        
    print(f"  loading data from {path} ...")
    df = pd.read_csv(path)
    
    features = [
        "Area", "VehPower", "VehAge", "DrivAge", 
        "BonusMalus", "VehBrand", "VehGas", "Density", "Region"
    ]
    target = "ClaimNb"
    weight = "Exposure"
    
    X = df[features]
    y = df[target]
    w = df[weight]
    
    # 80/20 split
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, w_train, w_test


def build_preprocessor() -> ColumnTransformer:
    """Build a standard scikit-learn preprocessor."""
    categorical_cols = ["Area", "VehBrand", "VehGas", "Region"]
    numeric_cols = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols)
        ]
    )
    return preprocessor


def train_baseline(X_train, y_train, w_train, preprocessor):
    """Train sklearn PoissonRegressor baseline."""
    print("  training baseline PoissonRegressor ...")
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", PoissonRegressor(alpha=1e-4, max_iter=300))
    ])
    pipeline.fit(X_train, y_train, model__sample_weight=w_train)
    return pipeline


def train_lgbm(X_train, y_train, w_train, preprocessor):
    """Train improved LightGBM Poisson model."""
    print("  training improved LightGBM model ...")
    # We preprocess first because LightGBM sklearn API handles it better when explicit
    X_train_prep = preprocessor.fit_transform(X_train)
    
    model = lgb.LGBMRegressor(
        objective="poisson",
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42
    )
    
    # In actuarial modeling, exposure is typically used as a sample weight 
    # or explicitly as an offset. lgb handles sample_weight natively.
    model.fit(X_train_prep, y_train, sample_weight=w_train)
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    return pipeline


def evaluate(model, name: str, X_test, y_test, w_test) -> float:
    """Evaluate using Poisson deviance."""
    preds = model.predict(X_test)
    # Clip predictions slightly above 0 to prevent log(0) in deviance calculation
    preds = pd.Series(preds).clip(lower=1e-6)
    
    deviance = mean_poisson_deviance(y_test, preds, sample_weight=w_test)
    print(f"    {name} Poisson Deviance: {deviance:.4f}")
    return deviance


def run() -> None:
    """Run frequency modeling pipeline."""
    os.makedirs("reports", exist_ok=True)
    
    X_train, X_test, y_train, y_test, w_train, w_test = load_and_prep_data()
    preprocessor = build_preprocessor()
    
    # 1. Baseline
    baseline_model = train_baseline(X_train, y_train, w_train, preprocessor)
    base_dev = evaluate(baseline_model, "Baseline", X_test, y_test, w_test)
    
    # 2. Improved
    lgbm_model = train_lgbm(X_train, y_train, w_train, preprocessor)
    lgbm_dev = evaluate(lgbm_model, "LightGBM", X_test, y_test, w_test)
    
    # Save metrics
    metrics = {
        "baseline_poisson_deviance": base_dev,
        "lgbm_poisson_deviance": lgbm_dev,
        "improvement_pct": ((base_dev - lgbm_dev) / base_dev) * 100
    }
    
    metrics_path = os.path.join("reports", "model_frequency_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"  saved metrics -> {metrics_path}")
    
    # Save the best model
    model_path = os.path.join("reports", "lgbm_frequency_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(lgbm_model, f)
    print(f"  saved best model -> {model_path}")


if __name__ == "__main__":
    run()
