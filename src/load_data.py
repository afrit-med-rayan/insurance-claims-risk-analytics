"""
Data loading and merging functions for the MTPL2 insurance dataset.

Handles ARFF format parsing (including the 'string' attribute type that
scipy.io.arff does not support), severity aggregation, and the
frequency/severity merge.
"""

import io
import os
import pandas as pd


def _find_data_file(filename: str) -> str:
    """
    Look for a data file in data/raw/ first, then the project root.
    Returns the resolved path or raises FileNotFoundError.
    """
    candidates = [
        os.path.join("data", "raw", filename),
        os.path.join(os.path.dirname(__file__), "..", "data", "raw", filename),
        filename,
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"{filename} not found. Place it in data/raw/ before running."
    )


def _parse_arff(path: str) -> pd.DataFrame:
    """
    Minimal ARFF parser that handles numeric, nominal, and string attributes.
    scipy.io.arff raises NotImplementedError on 'string' type attributes,
    so we parse the header manually and read the @data section as CSV.

    Quoted values (from nominal attributes) have their quotes stripped.
    """
    columns = []
    numeric_cols = []
    data_lines = []
    in_data = False

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if line.lower().startswith("@attribute"):
                parts = line.split(None, 2)
                col_name = parts[1].strip("'\"")
                col_type = parts[2].strip().lower() if len(parts) > 2 else ""
                columns.append(col_name)
                if col_type == "numeric" or col_type == "real" or col_type == "integer":
                    numeric_cols.append(col_name)
            elif line.lower().startswith("@data"):
                in_data = True
            elif in_data and line:
                data_lines.append(line)

    # Strip surrounding quotes from each field (ARFF nominal values are quoted)
    cleaned = []
    for row in data_lines:
        fields = row.split(",")
        fields = [f.strip().strip("'\"") for f in fields]
        cleaned.append(",".join(fields))

    df = pd.read_csv(
        io.StringIO("\n".join(cleaned)),
        header=None,
        names=columns,
    )

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_freq(path: str = None) -> pd.DataFrame:
    """
    Load freMTPL2freq.arff into a DataFrame.

    Parameters
    ----------
    path : str, optional
        Explicit path to the ARFF file. Auto-resolved if None.

    Returns
    -------
    pd.DataFrame
        677,991 rows x 12 columns (IDpol, ClaimNb, Exposure, Area, ...).
    """
    if path is None:
        path = _find_data_file("freMTPL2freq.arff")

    print(f"  loading freq data from {path} ...")
    df = _parse_arff(path)

    df["IDpol"] = df["IDpol"].astype(int)
    df["ClaimNb"] = df["ClaimNb"].astype(int)

    print(f"  freq: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def load_sev(path: str = None) -> pd.DataFrame:
    """
    Load freMTPL2sev.arff into a DataFrame.

    Parameters
    ----------
    path : str, optional
        Explicit path to the ARFF file. Auto-resolved if None.

    Returns
    -------
    pd.DataFrame
        26,639 rows x 2 columns (IDpol, ClaimAmount).
        Multiple rows per policy are possible (one row per claim).
    """
    if path is None:
        path = _find_data_file("freMTPL2sev.arff")

    print(f"  loading severity data from {path} ...")
    df = _parse_arff(path)

    df["IDpol"] = df["IDpol"].astype(int)
    df["ClaimAmount"] = df["ClaimAmount"].astype(float)

    print(f"  sev: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def merge_datasets(freq: pd.DataFrame, sev: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate severity by policy and left-join onto the frequency table.

    Steps:
    - Sum ClaimAmount per IDpol -> ClaimAmount_total
    - Left join so all 677,991 policies are retained
    - Policies with no claims get ClaimAmount_total = 0

    Parameters
    ----------
    freq : pd.DataFrame
        Loaded frequency table.
    sev : pd.DataFrame
        Loaded severity table (one row per individual claim).

    Returns
    -------
    pd.DataFrame
        Merged table with same row count as freq plus ClaimAmount_total column.
    """
    sev_agg = (
        sev.groupby("IDpol", as_index=False)
        .agg(ClaimAmount_total=("ClaimAmount", "sum"))
    )

    merged = freq.merge(sev_agg, on="IDpol", how="left")
    merged["ClaimAmount_total"] = merged["ClaimAmount_total"].fillna(0.0)

    assert merged.shape[0] == freq.shape[0], (
        f"Row count mismatch after merge: {merged.shape[0]} vs {freq.shape[0]}"
    )

    return merged


def load_and_merge(freq_path: str = None, sev_path: str = None) -> pd.DataFrame:
    """
    Full load-and-merge pipeline. Saves result to data/processed/merged.csv.

    Returns
    -------
    pd.DataFrame
        Merged dataset ready for cleaning.
    """
    freq = load_freq(freq_path)
    sev = load_sev(sev_path)
    merged = merge_datasets(freq, sev)

    out_path = os.path.join("data", "processed", "merged.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"  saved merged data -> {out_path} ({merged.shape[0]:,} rows)")

    print("\n  column dtypes:")
    for col, dtype in merged.dtypes.items():
        null_count = merged[col].isna().sum()
        print(f"    {col:<20} {str(dtype):<10} nulls: {null_count}")

    return merged


if __name__ == "__main__":
    df = load_and_merge()
    print(f"\ndone. merged shape: {df.shape}")
