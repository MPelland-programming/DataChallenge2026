"""
FTR Opportunity Selection
=========================
select_opportunities() enforces the 10–100 constraint and returns a filtered
DataFrame ready for write_opportunities() in output.py.
"""

from typing import Protocol

import pandas as pd


class SelectorProtocol(Protocol):
    """Type-checkable interface for selection functions."""

    def __call__(
        self,
        scored_df: pd.DataFrame,
        min_opp: int = 10,
        max_opp: int = 100,
    ) -> pd.DataFrame:
        """
        Select opportunities from a scored DataFrame.

        Args:
            scored_df:  DataFrame with columns EID, MONTH, PEAKID, PREDICTED_PROFIT.
            min_opp:    Minimum number of opportunities to select.
            max_opp:    Maximum number of opportunities to select.

        Returns:
            DataFrame with columns EID, MONTH, PEAKID (no PREDICTED_PROFIT).
        """
        ...


def select_opportunities(
    scored_df: pd.DataFrame,
    min_opp: int = 10,
    max_opp: int = 100,
) -> pd.DataFrame:
    """
    Apply the 10–100 selection constraint and return the selected opportunities.

    Deduplicates by averaging PREDICTED_PROFIT per (EID, MONTH, PEAKID) when
    duplicates are present, then sorts descending. Selects the top n_profitable
    rows (those with PREDICTED_PROFIT > 0), clamped to [min_opp, max_opp].

    Does not write to disk — pass the result to write_opportunities() in output.py.

    Args:
        scored_df:  DataFrame with columns EID, MONTH, PEAKID, PREDICTED_PROFIT.
        min_opp:    Minimum number of opportunities (default 10).
        max_opp:    Maximum number of opportunities (default 100).

    Returns:
        DataFrame with columns EID, MONTH, PEAKID.
    """
    deduped = scored_df.groupby(['EID', 'MONTH', 'PEAKID'], as_index=False)['PREDICTED_PROFIT'].mean()
    sorted_df = deduped.sort_values('PREDICTED_PROFIT', ascending=False)

    n_profitable = int((sorted_df['PREDICTED_PROFIT'] > 0).sum())

    if n_profitable < min_opp:
        selected = sorted_df.head(min_opp)
    elif n_profitable <= max_opp:
        selected = sorted_df.head(n_profitable)
    else:
        selected = sorted_df.head(max_opp)

    return selected[['EID', 'MONTH', 'PEAKID']].reset_index(drop=True)
