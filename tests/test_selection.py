import pandas as pd
import pytest

from datachallenge.selection import score_by_activation_level

CUTOFF_DATE = "2020-07-07"
TARGET_MONTH = "2020-08"  # M+1 when cutoff is 2020-07-07


def _make_candidates(eids, target_month=TARGET_MONTH):
    return pd.DataFrame({
        "EID": list(eids) * 2,
        "MONTH": [target_month] * len(eids) * 2,
        "PEAKID": [0] * len(eids) + [1] * len(eids),
    })


def test_score_output_columns(loader):
    """Result must have exactly EID, MONTH, PEAKID, PREDICTED_PROFIT."""
    triplets = loader.get_all_triplets(CUTOFF_DATE)
    candidates = _make_candidates(triplets["EID"].unique()[:10])

    result = score_by_activation_level(loader, candidates, CUTOFF_DATE)

    assert set(result.columns) == {"EID", "MONTH", "PEAKID", "PREDICTED_PROFIT"}


def test_score_all_candidates_returned(loader):
    """Every candidate triplet must appear in the output (no silent drops)."""
    triplets = loader.get_all_triplets(CUTOFF_DATE)
    sample_eids = triplets["EID"].unique()[:20]
    candidates = _make_candidates(sample_eids)

    result = score_by_activation_level(loader, candidates, CUTOFF_DATE)

    # Same number of rows as input
    assert len(result) == len(candidates)

    # All (EID, MONTH, PEAKID) keys present
    input_keys = set(zip(candidates["EID"], candidates["MONTH"], candidates["PEAKID"]))
    output_keys = set(zip(result["EID"], result["MONTH"], result["PEAKID"]))
    assert input_keys == output_keys


def test_score_nonnegative(loader):
    """ACTIVATIONLEVEL is a percentage, so mean must be >= 0.

    Absent triplets default to 0 (never negative).
    """
    triplets = loader.get_all_triplets(CUTOFF_DATE)
    candidates = _make_candidates(triplets["EID"].unique()[:20])

    result = score_by_activation_level(loader, candidates, CUTOFF_DATE)

    assert (result["PREDICTED_PROFIT"] >= 0).all(), (
        f"Negative scores found:\n{result[result['PREDICTED_PROFIT'] < 0]}"
    )


def test_score_absent_triplets_get_zero(loader):
    """EIDs that appear in history but have no monthly sim for M+1 get score 0.

    We fabricate a fake EID (int 0) that is very unlikely to exist in sims.
    It must appear in the output with PREDICTED_PROFIT = 0.
    """
    fake_eid = 0
    candidates = pd.DataFrame({
        "EID": [fake_eid, fake_eid],
        "MONTH": [TARGET_MONTH, TARGET_MONTH],
        "PEAKID": [0, 1],
    })

    result = score_by_activation_level(loader, candidates, CUTOFF_DATE)

    # EID 0 has no sim data → must default to 0
    for peakid in [0, 1]:
        row = result[(result["EID"] == fake_eid) & (result["PEAKID"] == peakid)]
        assert len(row) == 1
        assert row["PREDICTED_PROFIT"].iloc[0] == 0.0
