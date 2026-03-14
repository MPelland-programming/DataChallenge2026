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
    Build the candidate pool for target month M+1 from the monthly sim universe.

    Retrieves all (EID, PEAKID) pairs present in the monthly simulation files
    for the target month. This covers ~170k combinations per month — ~28× more
    than the prices+costs universe (~6k), which is heavily survivor-biased.

    Design choice: we use the sim universe as candidates because:
    - prices+costs covers only ~3.5% of the FTR market (already-traded EIDs)
    - The other 96.5% are invisible to any scorer using get_all_triplets
    - Data exploration shows zero-sim EIDs (91% profitable) and absent-from-sim
      EIDs (83% profitable) are MORE profitable than non-zero-sim (68%)

    Do NOT prune zero-sim or absent EIDs: any scorer that assigns them score 0
    actively harms recall, but the candidate pool must still include them so
    higher-quality scorers can rank them correctly.

    Args:
        loader:        Initialised CustomDataLoader.
        cutoff_date:   'YYYY-MM-DD', the 7th of month M. (Unused here — the sim
                       universe is queried by target_month directly — but kept
                       for interface consistency with the rest of the pipeline.)
        target_month:  'YYYY-MM', the month M+1 being predicted.

    Returns:
        DataFrame with columns EID, MONTH, PEAKID.
    """
    sim_universe = loader.get_sim_universe(target_month)

    candidates = sim_universe.copy()
    candidates["MONTH"] = target_month

    return candidates[["EID", "MONTH", "PEAKID"]].reset_index(drop=True)
