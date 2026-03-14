"""
Unit tests for datachallenge/candidates.py (mock-based, no real data required).
"""

import pandas as pd
import pytest
from unittest.mock import MagicMock

from datachallenge.candidates import build_candidate_pool

TARGET_MONTH = "2020-08"
CUTOFF_DATE = "2020-07-07"


def _make_sim_universe(n=5):
    """Simulate ~n (EID, PEAKID) pairs from the monthly sim universe."""
    eids = list(range(1, n + 1))
    return pd.DataFrame({
        "EID": eids * 2,
        "PEAKID": [0] * n + [1] * n,
    })


def _make_loader(sim_df=None):
    loader = MagicMock()
    loader.get_sim_universe.return_value = (
        sim_df if sim_df is not None else _make_sim_universe()
    )
    return loader


# ── Test: output columns ──────────────────────────────────────────────────────


def test_output_columns():
    """Output must have exactly EID, MONTH, PEAKID."""
    loader = _make_loader()
    result = build_candidate_pool(loader, CUTOFF_DATE, TARGET_MONTH)
    assert set(result.columns) == {"EID", "MONTH", "PEAKID"}


# ── Test: MONTH is set to target_month ────────────────────────────────────────


def test_month_is_target_month():
    """Every row must have MONTH = target_month."""
    loader = _make_loader(_make_sim_universe(10))
    result = build_candidate_pool(loader, CUTOFF_DATE, TARGET_MONTH)
    assert (result["MONTH"] == TARGET_MONTH).all()


# ── Test: uses get_sim_universe ───────────────────────────────────────────────


def test_calls_get_sim_universe():
    """build_candidate_pool must call get_sim_universe(target_month)."""
    loader = _make_loader()
    build_candidate_pool(loader, CUTOFF_DATE, TARGET_MONTH)
    loader.get_sim_universe.assert_called_once_with(TARGET_MONTH)


def test_does_not_call_get_all_triplets():
    """build_candidate_pool must NOT call get_all_triplets (old universe)."""
    loader = _make_loader()
    build_candidate_pool(loader, CUTOFF_DATE, TARGET_MONTH)
    loader.get_all_triplets.assert_not_called()


# ── Test: empty sim universe ──────────────────────────────────────────────────


def test_empty_sim_universe_returns_empty():
    """When sim universe is empty, output must also be empty (0 rows)."""
    loader = _make_loader(pd.DataFrame(columns=["EID", "PEAKID"]))
    result = build_candidate_pool(loader, CUTOFF_DATE, TARGET_MONTH)
    assert len(result) == 0
    assert set(result.columns) == {"EID", "MONTH", "PEAKID"}


# ── Test: all sim universe rows appear in output ──────────────────────────────


def test_all_sim_rows_returned():
    """Every (EID, PEAKID) from the sim universe must appear in the output."""
    sim_df = _make_sim_universe(10)
    loader = _make_loader(sim_df)
    result = build_candidate_pool(loader, CUTOFF_DATE, TARGET_MONTH)

    assert len(result) == len(sim_df)
    sim_keys = set(zip(sim_df["EID"], sim_df["PEAKID"]))
    out_keys = set(zip(result["EID"], result["PEAKID"]))
    assert sim_keys == out_keys
