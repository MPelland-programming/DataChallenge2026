"""
FTR Opportunity Scorers
=======================
Each scorer takes (loader, candidates, cutoff_date) and returns a DataFrame
with columns: EID, MONTH, PEAKID, PREDICTED_PROFIT.

PREDICTED_PROFIT is used only as a ranking score — it does not need to be a
literal profit estimate, just a value that ranks candidates correctly.
"""

from typing import Protocol

import pandas as pd

from datachallenge.loader import CustomDataLoader


class ScorerProtocol(Protocol):
    """Type-checkable interface for all scorer functions."""

    def __call__(
        self,
        loader: CustomDataLoader,
        candidates: pd.DataFrame,
        cutoff_date: str,
    ) -> pd.DataFrame:
        """
        Score candidates for the target month M+1.

        Args:
            loader:       Initialised CustomDataLoader.
            candidates:   DataFrame with columns EID, MONTH, PEAKID.
                          All rows should be for the same target month M+1.
            cutoff_date:  'YYYY-MM-DD', the 7th of month M.

        Returns:
            DataFrame with columns EID, MONTH, PEAKID, PREDICTED_PROFIT.
        """
        ...


def score_by_activation_level(
    loader: CustomDataLoader,
    candidates: pd.DataFrame,
    cutoff_date: str,
) -> pd.DataFrame:
    """
    Reference baseline: rank candidates by mean ACTIVATIONLEVEL in monthly sims.

    !! ANTI-PREDICTIVE — KEPT FOR COMPARISON ONLY !!
    H1 was refuted: higher ACTIVATIONLEVEL is *negatively* associated with
    profitability (AUC = 0.424, Spearman r = −0.120). The threshold found on
    2020–2022 (42.46%) selects zero triplets in 2023. Do not use this scorer
    as a primary signal.

    ACTIVATIONLEVEL is on a **0–100 percent scale** (confirmed by threshold
    analysis: best threshold ≈ 42, not ≈ 0.42).

    Triplets absent from the monthly sims (implicit zero) receive score 0.

    Args:
        loader:       Initialised CustomDataLoader instance.
        candidates:   DataFrame with columns EID, MONTH, PEAKID.
                      All rows should be for the same target month M+1.
        cutoff_date:  'YYYY-MM-DD', the 7th of month M. Not used to filter
                      monthly sims (those are for M+1 and always available);
                      included for interface consistency with ScorerProtocol.

    Returns:
        DataFrame with columns EID, MONTH, PEAKID, PREDICTED_PROFIT.
        PREDICTED_PROFIT = mean m_ACTIVATIONLEVEL (used as a ranking score).
    """
    monthly_sim = loader.get_monthly_data(candidates)

    if monthly_sim.empty:
        result = candidates[['EID', 'MONTH', 'PEAKID']].copy()
        result['PREDICTED_PROFIT'] = 0.0
        return result

    monthly_sim = monthly_sim.copy()
    monthly_sim['MONTH'] = pd.to_datetime(monthly_sim['DATETIME']).dt.to_period('M').astype(str)

    scored = (
        monthly_sim
        .groupby(['EID', 'MONTH', 'PEAKID'], as_index=False)['m_ACTIVATIONLEVEL']
        .mean()
        .rename(columns={'m_ACTIVATIONLEVEL': 'PREDICTED_PROFIT'})
    )

    result = candidates[['EID', 'MONTH', 'PEAKID']].merge(
        scored, on=['EID', 'MONTH', 'PEAKID'], how='left'
    )
    result['PREDICTED_PROFIT'] = result['PREDICTED_PROFIT'].fillna(0.0)

    return result[['EID', 'MONTH', 'PEAKID', 'PREDICTED_PROFIT']]


def score_by_historical_profit_rate(
    loader: CustomDataLoader,
    candidates: pd.DataFrame,
    cutoff_date: str,
) -> pd.DataFrame:
    """
    New baseline: rank candidates by historical profitability rate per (EID, PEAKID).

    Score = fraction of months where PROFIT > 0, computed over all months
    strictly before the cutoff month (M-1 and earlier). This captures H4
    ("chronic winners"): 68.9% of (EID, PEAKID) pairs with ≥6 months of
    history have a win rate above 50%.

    Anti-leak guarantee: only months strictly before M (cutoff_month) are used.
    Month M itself is excluded because its realized prices are not yet complete
    at the 7th — get_price_cost_profit with a filterdf containing month M would
    fetch all of M's realized prices (the full month, past day 7), violating the
    anti-leak rule. Restricting filterdf to MONTH < cutoff_month avoids this.

    Fallback for EIDs with no history (new elements): global average profit
    rate across all known pairs in the same historical window.

    Args:
        loader:       Initialised CustomDataLoader.
        candidates:   DataFrame with columns EID, MONTH, PEAKID.
                      All rows should be for the same target month M+1.
        cutoff_date:  'YYYY-MM-DD', the 7th of month M.

    Returns:
        DataFrame with columns EID, MONTH, PEAKID, PREDICTED_PROFIT.
        PREDICTED_PROFIT = historical win rate in [0.0, 1.0].
    """
    cutoff_month = cutoff_date[:7]  # 'YYYY-MM'

    # Triplets visible at cutoff, restricted to months strictly before M.
    # Excluding M avoids partial-month price data and any downstream leak:
    # get_price_cost_profit joins on MONTH and would pull the full month's
    # realized prices even though we only have 7 days at decision time.
    historical = loader.get_all_triplets(cutoff_date)
    historical = historical[historical['MONTH'] < cutoff_month].copy()

    if historical.empty:
        result = candidates[['EID', 'MONTH', 'PEAKID']].copy()
        result['PREDICTED_PROFIT'] = 0.0
        return result

    # get_price_cost_profit with this filterdf is safe: all months are < M,
    # so no realized prices from M or M+1 are fetched.
    profit_df = loader.get_price_cost_profit(historical)
    profit_df['profitable'] = (profit_df['PROFIT'] > 0).astype(float)

    # Win rate per (EID, PEAKID) — collapse over historical months
    win_rate = (
        profit_df
        .groupby(['EID', 'PEAKID'], as_index=False)['profitable']
        .mean()
        .rename(columns={'profitable': 'PREDICTED_PROFIT'})
    )

    # Global average as fallback for EIDs with no history
    global_avg = float(profit_df['profitable'].mean()) if not profit_df.empty else 0.0

    # Join on (EID, PEAKID) only — MONTH in candidates is M+1, not in history
    result = candidates[['EID', 'MONTH', 'PEAKID']].merge(
        win_rate, on=['EID', 'PEAKID'], how='left'
    )
    result['PREDICTED_PROFIT'] = result['PREDICTED_PROFIT'].fillna(global_avg)

    return result[['EID', 'MONTH', 'PEAKID', 'PREDICTED_PROFIT']]
