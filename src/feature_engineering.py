"""
Feature engineering for the cleaned MTPL2 dataset.

Creates derived features used in EDA and modeling:
- ClaimFrequency, HasClaim, AvgSeverity
- Age band categoricals for DrivAge and VehAge
- Simulated Premium column (see assumption note below)
- LossRatio

Premium formula assumption
--------------------------
This dataset does not include a real premium column. We simulate one using:

    Premium = BonusMalus * 0.8 + VehPower * 12 + log(1 + Density) * 5

This captures the three main pricing dimensions:
  - BonusMalus: individual claims history (the main tariff lever in France)
  - VehPower: vehicle power as a proxy for accident severity potential
  - Density (log-scaled): urban environment risk

This is a pedagogical approximation intended for portfolio/loss-ratio analysis only,
not a real actuarial tariff.
"""

import os

import numpy as np
import pandas as pd


DRIVAGE_BINS = [17, 25, 40, 60, 100]
DRIVAGE_LABELS = ["18-25", "26-40", "41-60", "61+"]

VEHAGE_BINS = [-1, 2, 7, 15, 200]
VEHAGE_LABELS = ["0-2", "3-7", "8-15", "16+"]


def add_claim_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ClaimFrequency, HasClaim, and AvgSeverity columns.

    ClaimFrequency is claims per year of exposure.
    AvgSeverity is NaN for policies with zero claims.
    """
    df = df.copy()
    df["ClaimFrequency"] = df["ClaimNb"] / df["Exposure"]
    df["HasClaim"] = (df["ClaimNb"] > 0).astype(int)
    df["AvgSeverity"] = np.where(
        df["ClaimNb"] > 0,
        df["ClaimAmount_total"] / df["ClaimNb"],
        np.nan,
    )
    return df


def add_age_bands(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin DrivAge and VehAge into categorical bands.

    DrivAgeBand: 18-25, 26-40, 41-60, 61+
    VehAgeBand:  0-2, 3-7, 8-15, 16+
    """
    df = df.copy()
    df["DrivAgeBand"] = pd.cut(
        df["DrivAge"],
        bins=DRIVAGE_BINS,
        labels=DRIVAGE_LABELS,
        right=True,
    ).astype(str)

    df["VehAgeBand"] = pd.cut(
        df["VehAge"],
        bins=VEHAGE_BINS,
        labels=VEHAGE_LABELS,
        right=True,
    ).astype(str)

    return df


def add_premium_and_loss_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simulated Premium and LossRatio columns.

    See module docstring for the premium formula and its assumptions.
    LossRatio = ClaimAmount_total / Premium (pure loss ratio).
    """
    df = df.copy()
    df["Premium"] = (
        df["BonusMalus"] * 0.8
        + df["VehPower"] * 12
        + np.log1p(df["Density"]) * 5
    )
    df["LossRatio"] = df["ClaimAmount_total"] / df["Premium"]
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps in order.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset from clean_data.clean().

    Returns
    -------
    pd.DataFrame
        Dataset with all new feature columns appended.
    """
    df = add_claim_features(df)
    df = add_age_bands(df)
    df = add_premium_and_loss_ratio(df)
    return df


def run(input_path: str = None, output_path: str = None) -> pd.DataFrame:
    """
    Load cleaned data, apply feature engineering, and save to features.csv.
    """
    if input_path is None:
        input_path = os.path.join("data", "processed", "cleaned.csv")
    if output_path is None:
        output_path = os.path.join("data", "processed", "features.csv")

    print(f"  reading cleaned data from {input_path} ...")
    df = pd.read_csv(input_path)

    print("  building features ...")
    df = build_features(df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  saved features -> {output_path} ({df.shape[0]:,} rows, {df.shape[1]} columns)")

    new_cols = [
        "ClaimFrequency", "HasClaim", "AvgSeverity",
        "DrivAgeBand", "VehAgeBand", "Premium", "LossRatio",
    ]
    print("\n  new feature stats:")
    print(df[new_cols].describe(include="all").to_string())

    return df


if __name__ == "__main__":
    run()
