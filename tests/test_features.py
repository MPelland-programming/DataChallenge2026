"""
Unit tests for datachallenge/features.py (mock-based, no real data required).
"""

import math
from unittest.mock import MagicMock

import pandas as pd
import pytest

from datachallenge.features import (
    ALL_COLUMNS,
    FEATURE_COLUMNS,
    KEY_COLUMNS,
    build_feature_matrix,
)

# ── Constants ─────────────────────────────────────────────────────────────────

TARGET_MONTH = "2020-08"
CUTOFF_DATE = "2020-07-07"
CUTOFF_MONTH = "2020-07"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_triplets(n=3):
    eids = list(range(1, n + 1))
    return pd.DataFrame(
        {
            "EID": eids * 2,
            "MONTH": [TARGET_MONTH] * n * 2,
            "PEAKID": [0] * n + [1] * n,
        }
    )


def _make_loader_no_data():
    """Loader where all sim/price/cost calls return empty DataFrames."""
    loader = MagicMock()
    loader.get_monthly_data.return_value = pd.DataFrame(
        columns=["SCENARIOID", "EID", "DATETIME", "PEAKID"]
    )
    loader.get_daily_data.return_value = pd.DataFrame(
        columns=["SCENARIOID", "EID", "DATETIME", "PEAKID"]
    )
    loader.get_all_triplets.return_value = pd.DataFrame(
        columns=["EID", "MONTH", "PEAKID"]
    )
    loader.get_price_cost_profit.return_value = pd.DataFrame(
        columns=["EID", "MONTH", "PEAKID", "PRICE", "COST", "PROFIT"]
    )
    return loader


def _make_monthly_sim_rows(eids, month, n_scenarios=3, act=50.0, psm=10.0):
    """Minimal monthly sim DataFrame (one timestamp per scenario)."""
    rows = []
    for eid in eids:
        for peakid in [0, 1]:
            for scenario in range(1, n_scenarios + 1):
                rows.append(
                    {
                        "SCENARIOID": scenario,
                        "EID": eid,
                        "DATETIME": pd.Timestamp(f"{month}-15"),
                        "PEAKID": peakid,
                        "m_ACTIVATIONLEVEL": act,
                        "m_PSM": psm,
                        "m_WINDIMPACT": 1.0,
                        "m_SOLARIMPACT": 0.5,
                        "m_HYDROIMPACT": 2.0,
                        "m_NONRENEWBALIMPACT": 0.3,
                        "m_EXTERNALIMPACT": 0.2,
                        "m_TRANSMISSIONOUTAGEIMPACT": 3.0,
                        "m_LOADIMPACT": 5.0,
                    }
                )
    return pd.DataFrame(rows)


def _make_profit_df(eids, months, profit_val=1.0):
    rows = []
    for eid in eids:
        for month in months:
            for peakid in [0, 1]:
                rows.append(
                    {
                        "EID": eid,
                        "MONTH": month,
                        "PEAKID": peakid,
                        "PRICE": 10.0,
                        "COST": 10.0 - profit_val,
                        "PROFIT": profit_val,
                    }
                )
    return pd.DataFrame(rows)


# ── Test 1: correct output columns ───────────────────────────────────────────


def test_output_has_correct_columns():
    """Output must have exactly KEY_COLUMNS + all FEATURE_COLUMNS."""
    loader = _make_loader_no_data()
    result = build_feature_matrix(loader, _make_triplets(3), CUTOFF_DATE)
    assert set(result.columns) == set(ALL_COLUMNS)


def test_feature_column_count():
    """FEATURE_COLUMNS must contain exactly 33 features."""
    assert len(FEATURE_COLUMNS) == 33


# ── Test 2: one row per triplet, no duplicates ───────────────────────────────


def test_one_row_per_triplet():
    """Output must have exactly one row per unique input triplet."""
    loader = _make_loader_no_data()
    triplets = _make_triplets(4)
    result = build_feature_matrix(loader, triplets, CUTOFF_DATE)
    assert len(result) == len(triplets)
    keys = result[KEY_COLUMNS].apply(tuple, axis=1)
    assert keys.nunique() == len(result)


def test_duplicate_input_deduplicated():
    """Duplicate rows in input produce a single output row."""
    loader = _make_loader_no_data()
    triplets = pd.DataFrame(
        {"EID": [1, 1], "MONTH": [TARGET_MONTH, TARGET_MONTH], "PEAKID": [0, 0]}
    )
    result = build_feature_matrix(loader, triplets, CUTOFF_DATE)
    assert len(result) == 1


# ── Test 3: no data leakage ───────────────────────────────────────────────────


def test_daily_sim_called_with_cutoff_month():
    """
    get_daily_data must be called with filterdf.MONTH == M (cutoff month), not M+1.
    Requesting M+1 daily sims would require future data not yet available.
    """
    loader = _make_loader_no_data()
    build_feature_matrix(loader, _make_triplets(2), CUTOFF_DATE)

    call_args = loader.get_daily_data.call_args
    assert call_args is not None
    passed_filterdf, passed_cutoff = call_args[0]
    assert (passed_filterdf["MONTH"] == CUTOFF_MONTH).all(), (
        "Daily sim filterdf must only contain MONTH = M, not M+1"
    )
    assert passed_cutoff == CUTOFF_DATE


def test_historical_profit_excludes_cutoff_month():
    """
    get_price_cost_profit for historical features must only be called with
    months strictly before M (anti-leak: M's partial prices would leak).
    """
    loader = MagicMock()
    loader.get_monthly_data.return_value = pd.DataFrame(
        columns=["SCENARIOID", "EID", "DATETIME", "PEAKID"]
    )
    loader.get_daily_data.return_value = pd.DataFrame(
        columns=["SCENARIOID", "EID", "DATETIME", "PEAKID"]
    )
    # get_all_triplets returns a mix: M and a prior month
    loader.get_all_triplets.return_value = pd.DataFrame(
        {
            "EID": [1, 1],
            "MONTH": [CUTOFF_MONTH, "2020-06"],
            "PEAKID": [0, 0],
        }
    )
    loader.get_price_cost_profit.return_value = pd.DataFrame(
        columns=["EID", "MONTH", "PEAKID", "PRICE", "COST", "PROFIT"]
    )

    triplets = pd.DataFrame({"EID": [1], "MONTH": [TARGET_MONTH], "PEAKID": [0]})
    build_feature_matrix(loader, triplets, CUTOFF_DATE)

    # The historical call (for hist_profit_rate) must exclude MONTH = M
    # The cost_proxy_m call may use MONTH = M — identify the historical call
    historical_calls = [
        c for c in loader.get_price_cost_profit.call_args_list
        if not (c[0][0]["MONTH"] == CUTOFF_MONTH).all()
    ]
    for call_arg in historical_calls:
        passed_filterdf = call_arg[0][0]
        assert (passed_filterdf["MONTH"] < CUTOFF_MONTH).all(), (
            "Historical get_price_cost_profit call includes month M — anti-leak violation"
        )


def test_cost_proxy_m_uses_cutoff_month():
    """
    cost_proxy must query month M costs (not historical months, not M+1).
    The cost_proxy_m call to get_price_cost_profit must use filterdf.MONTH == M.
    """
    loader = _make_loader_no_data()
    triplets = pd.DataFrame({"EID": [1], "MONTH": [TARGET_MONTH], "PEAKID": [0]})
    build_feature_matrix(loader, triplets, CUTOFF_DATE)

    cost_m_calls = [
        c for c in loader.get_price_cost_profit.call_args_list
        if (c[0][0]["MONTH"] == CUTOFF_MONTH).all()
    ]
    assert len(cost_m_calls) >= 1, (
        "No get_price_cost_profit call found for MONTH = M (cost_proxy)"
    )


# ── Test 4: seasonal encoding ─────────────────────────────────────────────────


def test_month_sin_january():
    """M+1 = January (month 1): sin(2π·1/12) ≈ 0.5."""
    loader = _make_loader_no_data()
    triplets = pd.DataFrame({"EID": [1], "MONTH": ["2020-01"], "PEAKID": [0]})
    result = build_feature_matrix(loader, triplets, "2019-12-07")
    expected = math.sin(2 * math.pi * 1 / 12)
    assert abs(result["month_sin"].iloc[0] - expected) < 1e-9
    assert abs(expected - 0.5) < 1e-9


def test_month_cos_december():
    """M+1 = December (month 12): cos(2π·12/12) = 1.0."""
    loader = _make_loader_no_data()
    triplets = pd.DataFrame({"EID": [1], "MONTH": ["2020-12"], "PEAKID": [0]})
    result = build_feature_matrix(loader, triplets, "2020-11-07")
    expected = math.cos(2 * math.pi * 12 / 12)
    assert abs(result["month_cos"].iloc[0] - expected) < 1e-9
    assert abs(expected - 1.0) < 1e-9


def test_sin2_plus_cos2_equals_one():
    """sin²(θ) + cos²(θ) = 1 for all 12 months."""
    loader = _make_loader_no_data()
    months = [f"2020-{m:02d}" for m in range(1, 13)]
    triplets = pd.DataFrame({"EID": [1] * 12, "MONTH": months, "PEAKID": [0] * 12})
    result = build_feature_matrix(loader, triplets, CUTOFF_DATE)
    s2c2 = result["month_sin"] ** 2 + result["month_cos"] ** 2
    assert (abs(s2c2 - 1.0) < 1e-9).all()


# ── Test 5: all_scenarios_profitable is binary ────────────────────────────────


def test_all_scenarios_profitable_binary():
    """all_scenarios_profitable must be 0.0 or 1.0 for every row."""
    loader = _make_loader_no_data()
    loader.get_monthly_data.return_value = _make_monthly_sim_rows(
        [1, 2, 3], TARGET_MONTH
    )
    result = build_feature_matrix(loader, _make_triplets(3), CUTOFF_DATE)
    assert result["all_scenarios_profitable"].isin([0.0, 1.0]).all()


def test_all_scenarios_profitable_true_when_all_positive():
    """all_scenarios_profitable = 1 when cost_proxy = 0 and abs_psm > 0."""
    loader = _make_loader_no_data()
    loader.get_monthly_data.return_value = _make_monthly_sim_rows(
        [1], TARGET_MONTH, psm=10.0
    )
    triplets = pd.DataFrame({"EID": [1], "MONTH": [TARGET_MONTH], "PEAKID": [0]})
    result = build_feature_matrix(loader, triplets, CUTOFF_DATE)
    assert result["all_scenarios_profitable"].iloc[0] == 1.0


# ── Test 6: absent from sims → 0.0 ───────────────────────────────────────────


def test_absent_from_sims_get_zero():
    """Triplets not in any sim must receive 0.0 for all sim-derived features."""
    loader = _make_loader_no_data()
    # Only EID=1, PEAKID=0 has monthly sim data
    loader.get_monthly_data.return_value = _make_monthly_sim_rows([1], TARGET_MONTH)
    triplets = _make_triplets(3)
    result = build_feature_matrix(loader, triplets, CUTOFF_DATE)

    sim_features = [
        "m_ACT_mean", "m_ACT_max", "m_PSM_mean", "m_PSM_min", "m_PSM_std",
        "n_hours_active", "mean_wind", "mean_sum_abs_psm", "psm_std_scenarios",
    ]
    absent = result[result["EID"] != 1]
    for col in sim_features:
        assert (absent[col] == 0.0).all(), f"Absent triplet has non-zero {col}"


def test_no_nan_in_output():
    """No NaN values may appear in any feature column."""
    loader = _make_loader_no_data()
    result = build_feature_matrix(loader, _make_triplets(5), CUTOFF_DATE)
    assert not result[FEATURE_COLUMNS].isnull().any().any()


# ── Test 7: hist_profit_rate in [0, 1] ───────────────────────────────────────


def test_hist_profit_rate_in_unit_interval():
    """hist_profit_rate must be in [0.0, 1.0] for every row."""
    loader = MagicMock()
    loader.get_monthly_data.return_value = pd.DataFrame(
        columns=["SCENARIOID", "EID", "DATETIME", "PEAKID"]
    )
    loader.get_daily_data.return_value = pd.DataFrame(
        columns=["SCENARIOID", "EID", "DATETIME", "PEAKID"]
    )
    loader.get_all_triplets.return_value = pd.DataFrame(
        {
            "EID": [1, 1, 2, 2],
            "MONTH": ["2020-05", "2020-06"] * 2,
            "PEAKID": [0, 0, 1, 1],
        }
    )
    # get_price_cost_profit is called twice: once for historical, once for cost_proxy_m
    loader.get_price_cost_profit.return_value = pd.DataFrame(
        {
            "EID": [1, 1, 2, 2],
            "MONTH": ["2020-05", "2020-06", "2020-05", "2020-06"],
            "PEAKID": [0, 0, 1, 1],
            "PRICE": [10.0, 5.0, 10.0, 10.0],
            "COST": [5.0, 10.0, 5.0, 5.0],
            "PROFIT": [5.0, -5.0, 5.0, 5.0],
        }
    )

    result = build_feature_matrix(loader, _make_triplets(3), CUTOFF_DATE)

    assert (result["hist_profit_rate"] >= 0.0).all()
    assert (result["hist_profit_rate"] <= 1.0).all()


def test_hist_profit_rate_fallback_for_new_eid():
    """EIDs with no history receive the global average profit rate."""
    loader = MagicMock()
    loader.get_monthly_data.return_value = pd.DataFrame(
        columns=["SCENARIOID", "EID", "DATETIME", "PEAKID"]
    )
    loader.get_daily_data.return_value = pd.DataFrame(
        columns=["SCENARIOID", "EID", "DATETIME", "PEAKID"]
    )
    # Historical data: only EID 1, 100% win rate → global avg = 1.0
    loader.get_all_triplets.return_value = pd.DataFrame(
        {"EID": [1], "MONTH": ["2020-06"], "PEAKID": [0]}
    )
    loader.get_price_cost_profit.return_value = pd.DataFrame(
        {
            "EID": [1],
            "MONTH": ["2020-06"],
            "PEAKID": [0],
            "PRICE": [10.0],
            "COST": [5.0],
            "PROFIT": [5.0],
        }
    )

    triplets = pd.DataFrame({"EID": [99], "MONTH": [TARGET_MONTH], "PEAKID": [0]})
    result = build_feature_matrix(loader, triplets, CUTOFF_DATE)

    assert result["hist_profit_rate"].iloc[0] == pytest.approx(1.0)


# ── Test 8: cost_proxy uses month M ──────────────────────────────────────────


def test_cost_proxy_uses_month_m_cost():
    """cost_proxy value must match abs(C) for month M when C_M is non-zero."""
    loader = MagicMock()
    loader.get_monthly_data.return_value = pd.DataFrame(
        columns=["SCENARIOID", "EID", "DATETIME", "PEAKID"]
    )
    loader.get_daily_data.return_value = pd.DataFrame(
        columns=["SCENARIOID", "EID", "DATETIME", "PEAKID"]
    )
    loader.get_all_triplets.return_value = pd.DataFrame(
        columns=["EID", "MONTH", "PEAKID"]
    )

    def _price_cost_profit_side_effect(filterdf):
        # cost_proxy_m call: filterdf has MONTH = M → return C = 7.5 for EID 1
        if (filterdf["MONTH"] == CUTOFF_MONTH).all():
            return pd.DataFrame(
                {"EID": [1], "MONTH": [CUTOFF_MONTH], "PEAKID": [0],
                 "PRICE": [0.0], "COST": [7.5], "PROFIT": [-7.5]}
            )
        return pd.DataFrame(columns=["EID", "MONTH", "PEAKID", "PRICE", "COST", "PROFIT"])

    loader.get_price_cost_profit.side_effect = _price_cost_profit_side_effect

    triplets = pd.DataFrame({"EID": [1], "MONTH": [TARGET_MONTH], "PEAKID": [0]})
    result = build_feature_matrix(loader, triplets, CUTOFF_DATE)

    assert result["cost_proxy"].iloc[0] == pytest.approx(7.5)


def test_cost_proxy_fallback_to_hist_median_when_m_absent():
    """When C_M = 0 (absent in parquet), cost_proxy falls back to historical median."""
    loader = MagicMock()
    loader.get_monthly_data.return_value = pd.DataFrame(
        columns=["SCENARIOID", "EID", "DATETIME", "PEAKID"]
    )
    loader.get_daily_data.return_value = pd.DataFrame(
        columns=["SCENARIOID", "EID", "DATETIME", "PEAKID"]
    )
    loader.get_all_triplets.return_value = pd.DataFrame(
        {"EID": [1], "MONTH": ["2020-06"], "PEAKID": [0]}
    )

    def _side_effect(filterdf):
        if (filterdf["MONTH"] == CUTOFF_MONTH).all():
            # C_M = 0 → absent from parquet (sparsified)
            return pd.DataFrame(
                {"EID": [1], "MONTH": [CUTOFF_MONTH], "PEAKID": [0],
                 "PRICE": [0.0], "COST": [0.0], "PROFIT": [0.0]}
            )
        # Historical: median COST = 4.0
        return pd.DataFrame(
            {"EID": [1], "MONTH": ["2020-06"], "PEAKID": [0],
             "PRICE": [10.0], "COST": [4.0], "PROFIT": [6.0]}
        )

    loader.get_price_cost_profit.side_effect = _side_effect

    triplets = pd.DataFrame({"EID": [1], "MONTH": [TARGET_MONTH], "PEAKID": [0]})
    result = build_feature_matrix(loader, triplets, CUTOFF_DATE)

    assert result["cost_proxy"].iloc[0] == pytest.approx(4.0)


# ── Test 9: impact_concentration range ───────────────────────────────────────


def test_impact_concentration_in_unit_interval():
    """impact_concentration must be in [0.0, 1.0] for every row."""
    loader = _make_loader_no_data()
    loader.get_monthly_data.return_value = _make_monthly_sim_rows([1, 2], TARGET_MONTH)
    result = build_feature_matrix(loader, _make_triplets(2), CUTOFF_DATE)
    assert (result["impact_concentration"] >= 0.0).all()
    assert (result["impact_concentration"] <= 1.0).all()


def test_impact_concentration_one_source_dominates():
    """impact_concentration ≈ 1.0 when a single impact source accounts for all congestion."""
    loader = _make_loader_no_data()
    rows = []
    for scenario in [1, 2, 3]:
        rows.append({
            "SCENARIOID": scenario, "EID": 1,
            "DATETIME": pd.Timestamp(f"{TARGET_MONTH}-15"),
            "PEAKID": 0,
            "m_ACTIVATIONLEVEL": 50.0, "m_PSM": 5.0,
            "m_WINDIMPACT": 100.0,   # single dominant source
            "m_SOLARIMPACT": 0.0,
            "m_HYDROIMPACT": 0.0,
            "m_NONRENEWBALIMPACT": 0.0,
            "m_EXTERNALIMPACT": 0.0,
            "m_TRANSMISSIONOUTAGEIMPACT": 0.0,
            "m_LOADIMPACT": 0.0,
        })
    loader.get_monthly_data.return_value = pd.DataFrame(rows)

    triplets = pd.DataFrame({"EID": [1], "MONTH": [TARGET_MONTH], "PEAKID": [0]})
    result = build_feature_matrix(loader, triplets, CUTOFF_DATE)

    assert result["impact_concentration"].iloc[0] == pytest.approx(1.0, abs=1e-6)


# ── Test 10: reproducibility ──────────────────────────────────────────────────


def test_feature_matrix_is_reproducible():
    """Same input must produce identical output on repeated calls."""
    loader = _make_loader_no_data()
    loader.get_monthly_data.return_value = _make_monthly_sim_rows([1, 2], TARGET_MONTH)
    triplets = _make_triplets(2)

    result1 = build_feature_matrix(loader, triplets, CUTOFF_DATE)
    result2 = build_feature_matrix(loader, triplets, CUTOFF_DATE)

    pd.testing.assert_frame_equal(
        result1.reset_index(drop=True), result2.reset_index(drop=True)
    )
