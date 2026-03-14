"""
FTR Opportunity Selection Models
=================================
Each function in this module takes a CustomDataLoader, a candidate DataFrame
(columns: EID, MONTH, PEAKID — for the target month M+1), and a cutoff_date,
and returns a scored DataFrame with columns:

    EID, MONTH, PEAKID, PREDICTED_PROFIT

The PREDICTED_PROFIT column is used as a ranking score by
`select_best_predict_and_print()` in loader.py — it does NOT need to be a
true profit estimate; it only needs to rank candidates correctly.

Usage in main.py (inside the per-month loop):
    from datachallenge.selection import score_by_activation_level
    scored = score_by_activation_level(loader, candidates, cutoff_date)
    select_best_predict_and_print(scored)
"""

import pandas as pd
from datachallenge.loader import CustomDataLoader


def score_by_activation_level(
    loader: CustomDataLoader,
    candidates: pd.DataFrame,
    cutoff_date: str,
) -> pd.DataFrame:
    """
    Baseline selector: rank candidates by mean ACTIVATIONLEVEL in monthly sims.

    ACTIVATIONLEVEL represents the intensity of network constraint activation (%)
    for a given element and hour. Higher values indicate stronger expected price
    differentials, which correlates with higher realized prices (PR).
    Confirmed by team analysis to be a strong predictor of profitability.

    The score is computed as the mean of m_ACTIVATIONLEVEL across:
      - all 3 SCENARIOID values
      - all hours of the target month M+1

    Triplets absent from the monthly sims (implicit zero) receive score 0.

    Args:
        loader:       Initialised CustomDataLoader instance.
        candidates:   DataFrame with columns MONTH, PEAKID, EID.
                      All rows should be for the same target month M+1.
        cutoff_date:  'YYYY-MM-DD', the 7th of month M (not used for monthly sims,
                      included for interface consistency).

    Returns:
        DataFrame with columns EID, MONTH, PEAKID, PREDICTED_PROFIT.
        PREDICTED_PROFIT = mean m_ACTIVATIONLEVEL (used as a ranking score).
    """
    monthly_sim = loader.get_monthly_data(candidates)

    if monthly_sim.empty:
        result = candidates[['EID', 'MONTH', 'PEAKID']].copy()
        result['PREDICTED_PROFIT'] = 0.0
        return result

    # Derive MONTH from DATETIME so we can group per triplet
    monthly_sim = monthly_sim.copy()
    monthly_sim['MONTH'] = pd.to_datetime(monthly_sim['DATETIME']).dt.to_period('M').astype(str)

    # Mean ACTIVATIONLEVEL per (EID, MONTH, PEAKID) across all scenarios and hours
    scored = (
        monthly_sim
        .groupby(['EID', 'MONTH', 'PEAKID'], as_index=False)['m_ACTIVATIONLEVEL']
        .mean()
        .rename(columns={'m_ACTIVATIONLEVEL': 'PREDICTED_PROFIT'})
    )

    # Left-join so candidates with no sim data get score 0
    result = candidates[['EID', 'MONTH', 'PEAKID']].merge(
        scored, on=['EID', 'MONTH', 'PEAKID'], how='left'
    )
    result['PREDICTED_PROFIT'] = result['PREDICTED_PROFIT'].fillna(0.0)

    return result[['EID', 'MONTH', 'PEAKID', 'PREDICTED_PROFIT']]
