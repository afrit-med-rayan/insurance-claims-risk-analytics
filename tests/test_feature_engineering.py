"""Tests for the feature engineering module."""

import numpy as np
import pandas as pd
from src.feature_engineering import add_age_bands, add_claim_features


def test_claim_frequency_math():
    """Ensure claim frequency is calculated correctly and severity handles zeroes."""
    df = pd.DataFrame({
        "ClaimNb": [2, 0], 
        "Exposure": [0.5, 1.0], 
        "ClaimAmount_total": [1000, 0]
    })
    
    out = add_claim_features(df)
    
    assert out["ClaimFrequency"].iloc[0] == 4.0
    assert out["ClaimFrequency"].iloc[1] == 0.0
    
    assert out["AvgSeverity"].iloc[0] == 500.0
    assert np.isnan(out["AvgSeverity"].iloc[1]), "Severity must be NaN for 0 claims"
    assert out["HasClaim"].iloc[0] == 1
    assert out["HasClaim"].iloc[1] == 0


def test_age_bands_no_nans():
    """Ensure age bands cover expected actuarial ranges without leaking NaNs."""
    df = pd.DataFrame({
        "DrivAge": [18, 25, 30, 45, 60, 99], 
        "VehAge": [0, 2, 5, 10, 15, 25]
    })
    out = add_age_bands(df)
    
    assert out["DrivAgeBand"].isna().sum() == 0, "NaNs found in DrivAgeBand"
    assert out["VehAgeBand"].isna().sum() == 0, "NaNs found in VehAgeBand"
    
    assert out["DrivAgeBand"].iloc[0] == "18-25"
    assert out["VehAgeBand"].iloc[0] == "0-2"
