"""
Data cleaning for the merged MTPL2 dataset.

Checks for missing values, duplicate policy IDs, and documents outlier
handling decisions. All decisions are logged to reports/data_quality_log.md.
"""

import os
import textwrap
from datetime import datetime

import pandas as pd


LOG_PATH = os.path.join("reports", "data_quality_log.md")


def _append_log(text: str) -> None:
    os.makedirs("reports", exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def _log_decision(title: str, rows_affected: int, action: str, rationale: str) -> None:
    entry = textwrap.dedent(f"""
    ### {title}

    - **Rows affected**: {rows_affected:,}
    - **Action**: {action}
    - **Rationale**: {rationale}
    """)
    _append_log(entry)


def audit_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Report null counts and percentages per column.
    Returns a summary DataFrame.
    """
    null_counts = df.isna().sum()
    null_pct = (null_counts / len(df) * 100).round(2)
    summary = pd.DataFrame({"null_count": null_counts, "null_pct": null_pct})
    summary = summary[summary["null_count"] > 0]

    if summary.empty:
        print("  no missing values found")
    else:
        print("  missing values:")
        print(summary.to_string())

    return summary


def check_duplicates(df: pd.DataFrame) -> int:
    """
    Count duplicate IDpol values in the frequency table.
    Returns the count of duplicated policy IDs.
    """
    n_dupes = df["IDpol"].duplicated().sum()
    if n_dupes > 0:
        print(f"  warning: {n_dupes:,} duplicate IDpol values found")
    else:
        print("  no duplicate IDpol values")
    return n_dupes


def flag_outliers(df: pd.DataFrame) -> tuple:
    """
    Identify rows matching known outlier patterns in the MTPL2 dataset.

    Returns
    -------
    tuple of (cleaned DataFrame, dict with outlier counts)
    """
    report = {}
    original_len = len(df)

    # 1. Driver age: under 18 or over 100
    mask_age = (df["DrivAge"] < 18) | (df["DrivAge"] > 100)
    report["drivage_invalid"] = int(mask_age.sum())
    _log_decision(
        "Driver age outside [18, 100]",
        rows_affected=report["drivage_invalid"],
        action="Rows excluded from cleaned dataset",
        rationale=(
            "Ages below 18 are legally impossible for a licensed driver. "
            "Ages above 100 are almost certainly data entry errors. "
            "This pattern is documented in actuarial tutorials for this dataset."
        ),
    )
    df = df[~mask_age].copy()

    # 2. Exposure: must be in (0, 1]
    mask_exp_zero = df["Exposure"] <= 0
    report["exposure_zero"] = int(mask_exp_zero.sum())
    _log_decision(
        "Exposure <= 0",
        rows_affected=report["exposure_zero"],
        action="Rows excluded (zero or negative exposure has no actuarial meaning)",
        rationale=(
            "Exposure represents the fraction of a year a policy was active. "
            "A value of zero means no time at risk; such rows cannot contribute "
            "meaningful frequency information and are dropped."
        ),
    )
    df = df[~mask_exp_zero].copy()

    mask_exp_high = df["Exposure"] > 1
    report["exposure_gt1"] = int(mask_exp_high.sum())
    _log_decision(
        "Exposure > 1",
        rows_affected=report["exposure_gt1"],
        action="Exposure capped at 1.0",
        rationale=(
            "Exposure cannot exceed 1 year for a single policy period. "
            "Values slightly above 1.0 are rounding artefacts and are capped "
            "rather than dropped to retain the policy record."
        ),
    )
    df.loc[mask_exp_high, "Exposure"] = 1.0

    # 3. Vehicle age: no negative values
    mask_vehage = df["VehAge"] < 0
    report["vehage_negative"] = int(mask_vehage.sum())
    _log_decision(
        "Negative vehicle age",
        rows_affected=report["vehage_negative"],
        action="Rows excluded",
        rationale="Negative vehicle age is a data entry error with no valid interpretation.",
    )
    df = df[~mask_vehage].copy()

    # 4. High claim count relative to short exposure (well-known outliers)
    mask_hf = (df["ClaimNb"] > 4) & (df["Exposure"] < 0.1)
    report["high_freq_outliers"] = int(mask_hf.sum())
    _log_decision(
        "ClaimNb > 4 with Exposure < 0.1",
        rows_affected=report["high_freq_outliers"],
        action="ClaimNb capped at 4 for affected rows",
        rationale=(
            "Several rows with very short exposure (< 5 weeks) carry an implausibly "
            "high claim count. These are documented outlier rows in actuarial "
            "literature for the freMTPL2 dataset. Capping at 4 retains the policies "
            "while limiting the distortion they cause in the Poisson model."
        ),
    )
    df.loc[mask_hf, "ClaimNb"] = 4

    # 5. High BonusMalus (flag only, retain)
    mask_bm = df["BonusMalus"] > 150
    report["bonus_malus_high"] = int(mask_bm.sum())
    _log_decision(
        "BonusMalus > 150",
        rows_affected=report["bonus_malus_high"],
        action="Rows retained, flagged with is_high_risk=True",
        rationale=(
            "BonusMalus above 150 indicates a very high-risk driver. "
            "These are genuine policies and should be included in the model; "
            "they provide important signal for the high end of the risk spectrum. "
            "A flag column is added so the segment can be isolated in EDA."
        ),
    )
    df["is_high_risk"] = (df["BonusMalus"] > 150).astype(int)

    rows_removed = original_len - len(df)
    print(f"  outlier summary: {rows_removed:,} rows removed, {len(df):,} rows retained")
    for k, v in report.items():
        print(f"    {k}: {v:,}")

    return df, report


def clean(df: pd.DataFrame, log_path: str = LOG_PATH) -> pd.DataFrame:
    """
    Full cleaning pipeline. Logs all decisions to log_path and saves
    cleaned data to data/processed/cleaned.csv.

    Parameters
    ----------
    df : pd.DataFrame
        Merged dataset from load_data.merge_datasets().
    log_path : str
        Path to the markdown log file.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """
    global LOG_PATH
    LOG_PATH = log_path

    os.makedirs("reports", exist_ok=True)
    header = textwrap.dedent(f"""
    # Data Quality Log

    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

    Dataset: freMTPL2freq + freMTPL2sev (merged)
    Initial shape: {df.shape[0]:,} rows x {df.shape[1]} columns

    ---
    """)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write(header)

    print("  auditing nulls ...")
    audit_nulls(df)

    print("  checking duplicates ...")
    check_duplicates(df)

    print("  flagging and handling outliers ...")
    df, _ = flag_outliers(df)

    out_path = os.path.join("data", "processed", "cleaned.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  saved cleaned data -> {out_path} ({df.shape[0]:,} rows)")

    _append_log(
        f"\n---\n\n**Final cleaned shape**: {df.shape[0]:,} rows x {df.shape[1]} columns\n"
    )

    return df


if __name__ == "__main__":
    from src.load_data import load_and_merge

    print("loading merged data ...")
    merged = load_and_merge()
    print("cleaning ...")
    cleaned = clean(merged)
    print(f"\ndone. cleaned shape: {cleaned.shape}")
