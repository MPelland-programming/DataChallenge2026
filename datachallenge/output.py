"""
FTR Opportunity Output
======================
write_opportunities() validates and writes selected opportunities to CSV.
"""

import pandas as pd


def write_opportunities(df: pd.DataFrame, path: str) -> None:
    """
    Validate and write selected opportunities to a CSV file.

    Converts internal column names (MONTH, PEAKID) to the expected output
    format (TARGET_MONTH, PEAK_TYPE) and runs sanity checks before writing.

    Input df must have columns: EID, MONTH, PEAKID (PEAKID as int: 0=OFF, 1=ON).

    Sanity checks (raises ValueError if any fail):
      - 10 ≤ selections per TARGET_MONTH ≤ 100
      - PEAK_TYPE values are only 'ON' or 'OFF'
      - TARGET_MONTH matches YYYY-MM format
      - No duplicate (TARGET_MONTH, PEAK_TYPE, EID) rows

    Args:
        df:    DataFrame with columns EID, MONTH, PEAKID.
        path:  File path to write the CSV to.
    """
    out = df[['EID', 'MONTH', 'PEAKID']].copy()
    out['PEAK_TYPE'] = out['PEAKID'].map({0: 'OFF', 1: 'ON'})
    out = out.rename(columns={'MONTH': 'TARGET_MONTH'})
    out = out[['TARGET_MONTH', 'PEAK_TYPE', 'EID']]

    # Sanity: PEAK_TYPE must only be 'ON' or 'OFF' (catches unmapped PEAKID values)
    invalid_peak = out[~out['PEAK_TYPE'].isin({'ON', 'OFF'})]
    if not invalid_peak.empty:
        raise ValueError(
            f"Invalid PEAK_TYPE values (expected 'ON' or 'OFF'): "
            f"{invalid_peak['PEAK_TYPE'].unique().tolist()}"
        )

    # Sanity: TARGET_MONTH must match YYYY-MM
    bad_months = out[~out['TARGET_MONTH'].str.match(r'^\d{4}-\d{2}$')]
    if not bad_months.empty:
        raise ValueError(
            f"TARGET_MONTH values not in YYYY-MM format: "
            f"{bad_months['TARGET_MONTH'].unique().tolist()}"
        )

    # Sanity: no duplicates
    dupes = out.duplicated(subset=['TARGET_MONTH', 'PEAK_TYPE', 'EID'])
    if dupes.any():
        raise ValueError(
            f"Duplicate (TARGET_MONTH, PEAK_TYPE, EID) rows found: "
            f"{out[dupes][['TARGET_MONTH', 'PEAK_TYPE', 'EID']].to_dict('records')}"
        )

    # Sanity: 10–100 selections per month
    per_month = out.groupby('TARGET_MONTH').size()
    violations = per_month[(per_month < 10) | (per_month > 100)]
    if not violations.empty:
        raise ValueError(
            f"Months violating the 10–100 selection constraint:\n{violations}"
        )

    out.to_csv(path, index=False)
