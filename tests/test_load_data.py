"""Tests for the data ingestion module."""

import pandas as pd
from src.load_data import merge_datasets


def test_merge_no_duplicate_idpol():
    """Ensure merging freq and sev does not duplicate rows from freq."""
    freq = pd.DataFrame({"IDpol": [1, 2, 3], "Exposure": [1, 1, 1]})
    sev = pd.DataFrame({"IDpol": [1, 1, 3], "ClaimAmount": [100, 50, 200]})
    
    merged = merge_datasets(freq, sev)
    
    assert len(merged) == len(freq), "Merged length must equal freq length"
    assert not merged["IDpol"].duplicated().any(), "Duplicate IDpol generated"


def test_merge_zero_claims_filled():
    """Ensure policies with no claims get ClaimAmount_total = 0."""
    freq = pd.DataFrame({"IDpol": [1, 2], "Exposure": [1, 1]})
    sev = pd.DataFrame({"IDpol": [1], "ClaimAmount": [100]})
    
    merged = merge_datasets(freq, sev)
    
    assert merged.loc[merged["IDpol"] == 2, "ClaimAmount_total"].iloc[0] == 0.0
