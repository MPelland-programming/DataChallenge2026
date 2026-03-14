"""
FTR Candidate Pool Construction
================================
build_candidate_pool() builds the set of (EID, MONTH, PEAKID) triplets to
score for a given target month M+1.
"""

import pandas as pd

from datachallenge.loader import CustomDataLoader


def build_candidate_pool(
    loader: CustomDataLoader,
    cutoff_date: str,
    target_month: str,
) -> pd.DataFrame:
    """
    Build the candidate pool for target month M+1.

    Retrieves all EIDs observed at or before cutoff_date and expands them to
    both PEAKID=0 (OFF-Peak) and PEAKID=1 (ON-Peak) for the target month.

    Design choice: we expand every known EID to both PEAKID values regardless
    of which combinations appear in historical data, because both products exist
    in the FTR market even if not previously observed together.

    Do NOT prune zero-sim or absent EIDs: data exploration showed that EIDs
    with zero ACTIVATIONLEVEL in sims are 91% profitable, and EIDs absent from
    sims entirely are 83% profitable — both higher than non-zero-sim EIDs (68%).
    Any scorer that assigns them score 0 actively harms recall, but the candidate
    pool must still include them so higher-quality scorers can rank them correctly.

    Args:
        loader:        Initialised CustomDataLoader.
        cutoff_date:   'YYYY-MM-DD', the 7th of month M.
        target_month:  'YYYY-MM', the month M+1 being predicted.

    Returns:
        DataFrame with columns EID, MONTH, PEAKID.
    """
    historical = loader.get_all_triplets(cutoff_date)
    known_eids = historical['EID'].unique()

    candidates = pd.DataFrame({
        'EID': list(known_eids) * 2,
        'MONTH': [target_month] * len(known_eids) * 2,
        'PEAKID': [0] * len(known_eids) + [1] * len(known_eids),
    })
    return candidates
