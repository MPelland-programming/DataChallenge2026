"""
Unit tests for datachallenge/scoring.py (mock-based, no real data required).
"""

import pandas as pd
import pytest
from unittest.mock import MagicMock

import numpy as np
from datachallenge.scoring import (
    score_by_activation_level,
    score_by_historical_profit_rate,
    score_by_lasso,
    score_by_lightgbm,
)

TARGET_MONTH = "2020-08"
CUTOFF_DATE = "2020-07-07"
CUTOFF_MONTH = "2020-07"


def _make_candidates(n=5):
    eids = list(range(1, n + 1))
    return pd.DataFrame({
        'EID': eids * 2,
        'MONTH': [TARGET_MONTH] * n * 2,
        'PEAKID': [0] * n + [1] * n,
    })


# ---------------------------------------------------------------------------
# score_by_activation_level
# ---------------------------------------------------------------------------

def test_activation_output_columns():
    """Output must have exactly EID, MONTH, PEAKID, PREDICTED_PROFIT."""
    loader = MagicMock()
    candidates = _make_candidates(3)
    loader.get_monthly_data.return_value = pd.DataFrame(columns=[
        'SCENARIOID', 'EID', 'DATETIME', 'PEAKID', 'm_ACTIVATIONLEVEL'
    ])

    result = score_by_activation_level(loader, candidates, CUTOFF_DATE)

    assert set(result.columns) == {'EID', 'MONTH', 'PEAKID', 'PREDICTED_PROFIT'}


def test_activation_all_candidates_returned():
    """Every candidate triplet must appear in the output."""
    loader = MagicMock()
    candidates = _make_candidates(4)
    loader.get_monthly_data.return_value = pd.DataFrame(columns=[
        'SCENARIOID', 'EID', 'DATETIME', 'PEAKID', 'm_ACTIVATIONLEVEL'
    ])

    result = score_by_activation_level(loader, candidates, CUTOFF_DATE)

    assert len(result) == len(candidates)
    input_keys = set(zip(candidates['EID'], candidates['MONTH'], candidates['PEAKID']))
    output_keys = set(zip(result['EID'], result['MONTH'], result['PEAKID']))
    assert input_keys == output_keys


def test_activation_absent_get_zero():
    """Candidates with no monthly sim row must receive PREDICTED_PROFIT = 0."""
    loader = MagicMock()
    candidates = _make_candidates(3)
    # Sim data only has EID=1/PEAKID=0; others must default to 0
    loader.get_monthly_data.return_value = pd.DataFrame({
        'SCENARIOID': [1],
        'EID': [1],
        'DATETIME': [pd.Timestamp('2020-08-15')],
        'PEAKID': [0],
        'm_ACTIVATIONLEVEL': [60.0],
    })

    result = score_by_activation_level(loader, candidates, CUTOFF_DATE)

    # EIDs 2, 3 and all PEAKID=1 rows have no sim data
    missing = result[(result['EID'] != 1) | (result['PEAKID'] != 0)]
    assert (missing['PREDICTED_PROFIT'] == 0.0).all()


def test_activation_nonnegative():
    """ACTIVATIONLEVEL is a percentage; mean is always >= 0."""
    loader = MagicMock()
    candidates = _make_candidates(5)
    loader.get_monthly_data.return_value = pd.DataFrame({
        'SCENARIOID': [1, 2, 3] * 5,
        'EID': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
        'DATETIME': [pd.Timestamp('2020-08-01')] * 15,
        'PEAKID': [0] * 15,
        'm_ACTIVATIONLEVEL': [10.0, 20.0, 30.0] * 5,
    })

    result = score_by_activation_level(loader, candidates, CUTOFF_DATE)

    assert (result['PREDICTED_PROFIT'] >= 0).all()


def test_activation_empty_sim_returns_all_zero():
    """When get_monthly_data returns empty, all candidates get score 0."""
    loader = MagicMock()
    candidates = _make_candidates(4)
    loader.get_monthly_data.return_value = pd.DataFrame()

    result = score_by_activation_level(loader, candidates, CUTOFF_DATE)

    assert len(result) == len(candidates)
    assert (result['PREDICTED_PROFIT'] == 0.0).all()


# ---------------------------------------------------------------------------
# score_by_historical_profit_rate
# ---------------------------------------------------------------------------

def _make_triplets(eids, months):
    """Cartesian product of eids × months × {0,1} for PEAKID."""
    rows = []
    for eid in eids:
        for month in months:
            for peakid in [0, 1]:
                rows.append({'EID': eid, 'MONTH': month, 'PEAKID': peakid})
    return pd.DataFrame(rows)


def _make_profit_df(eids, months, profit_val=1.0):
    """Profit DataFrame with constant PROFIT for each (EID, MONTH, PEAKID)."""
    rows = []
    for eid in eids:
        for month in months:
            for peakid in [0, 1]:
                rows.append({
                    'EID': eid, 'MONTH': month, 'PEAKID': peakid,
                    'PRICE': 10.0, 'COST': 10.0 - profit_val,
                    'PROFIT': profit_val,
                })
    return pd.DataFrame(rows)


def test_hist_output_columns():
    """Output must have exactly EID, MONTH, PEAKID, PREDICTED_PROFIT."""
    loader = MagicMock()
    candidates = _make_candidates(3)
    historical_months = ['2020-01', '2020-02', '2020-03', '2020-04', '2020-05', '2020-06']
    loader.get_all_triplets.return_value = _make_triplets([1, 2, 3], historical_months + [CUTOFF_MONTH])
    loader.get_price_cost_profit.return_value = _make_profit_df([1, 2, 3], historical_months)

    result = score_by_historical_profit_rate(loader, candidates, CUTOFF_DATE)

    assert set(result.columns) == {'EID', 'MONTH', 'PEAKID', 'PREDICTED_PROFIT'}


def test_hist_all_candidates_returned():
    """Every candidate triplet must appear in the output."""
    loader = MagicMock()
    candidates = _make_candidates(4)
    historical_months = ['2020-05', '2020-06']
    loader.get_all_triplets.return_value = _make_triplets([1, 2, 3, 4], historical_months)
    loader.get_price_cost_profit.return_value = _make_profit_df([1, 2, 3, 4], historical_months)

    result = score_by_historical_profit_rate(loader, candidates, CUTOFF_DATE)

    assert len(result) == len(candidates)
    input_keys = set(zip(candidates['EID'], candidates['MONTH'], candidates['PEAKID']))
    output_keys = set(zip(result['EID'], result['MONTH'], result['PEAKID']))
    assert input_keys == output_keys


def test_hist_no_history_returns_zero():
    """When there is no history before cutoff_month, all candidates get score 0."""
    loader = MagicMock()
    candidates = _make_candidates(3)
    # get_all_triplets only returns month M (cutoff_month) — nothing strictly before it
    loader.get_all_triplets.return_value = _make_triplets([1, 2, 3], [CUTOFF_MONTH])

    result = score_by_historical_profit_rate(loader, candidates, CUTOFF_DATE)

    assert (result['PREDICTED_PROFIT'] == 0.0).all()


def test_hist_rate_is_correct():
    """Win rate is correctly computed as fraction of profitable months."""
    loader = MagicMock()
    candidates = pd.DataFrame({
        'EID': [1],
        'MONTH': [TARGET_MONTH],
        'PEAKID': [0],
    })
    # 3 months of history, 2 profitable and 1 not → win rate = 2/3
    loader.get_all_triplets.return_value = pd.DataFrame({
        'EID': [1, 1, 1],
        'MONTH': ['2020-04', '2020-05', '2020-06'],
        'PEAKID': [0, 0, 0],
    })
    loader.get_price_cost_profit.return_value = pd.DataFrame({
        'EID': [1, 1, 1],
        'MONTH': ['2020-04', '2020-05', '2020-06'],
        'PEAKID': [0, 0, 0],
        'PRICE': [10.0, 10.0, 5.0],
        'COST': [5.0, 5.0, 10.0],
        'PROFIT': [5.0, 5.0, -5.0],
    })

    result = score_by_historical_profit_rate(loader, candidates, CUTOFF_DATE)

    assert len(result) == 1
    assert abs(result['PREDICTED_PROFIT'].iloc[0] - 2 / 3) < 1e-9


def test_hist_new_eid_gets_global_avg():
    """EIDs with no historical data receive the global average profit rate."""
    loader = MagicMock()
    # EID 99 is new (not in history); EIDs 1 and 2 have 100% win rate
    candidates = pd.DataFrame({
        'EID': [99],
        'MONTH': [TARGET_MONTH],
        'PEAKID': [0],
    })
    loader.get_all_triplets.return_value = pd.DataFrame({
        'EID': [1, 2],
        'MONTH': ['2020-06', '2020-06'],
        'PEAKID': [0, 0],
    })
    loader.get_price_cost_profit.return_value = pd.DataFrame({
        'EID': [1, 2],
        'MONTH': ['2020-06', '2020-06'],
        'PEAKID': [0, 0],
        'PRICE': [10.0, 10.0],
        'COST': [5.0, 5.0],
        'PROFIT': [5.0, 5.0],  # both profitable → global avg = 1.0
    })

    result = score_by_historical_profit_rate(loader, candidates, CUTOFF_DATE)

    assert len(result) == 1
    assert result['PREDICTED_PROFIT'].iloc[0] == pytest.approx(1.0)


def test_hist_excludes_cutoff_month():
    """get_price_cost_profit must not be called with month M triplets (anti-leak)."""
    loader = MagicMock()
    candidates = _make_candidates(2)
    # Mix: some rows are month M, some are before
    loader.get_all_triplets.return_value = pd.DataFrame({
        'EID': [1, 1],
        'MONTH': [CUTOFF_MONTH, '2020-06'],  # one in M, one before
        'PEAKID': [0, 0],
    })
    loader.get_price_cost_profit.return_value = pd.DataFrame({
        'EID': [1],
        'MONTH': ['2020-06'],
        'PEAKID': [0],
        'PRICE': [10.0],
        'COST': [5.0],
        'PROFIT': [5.0],
    })

    score_by_historical_profit_rate(loader, candidates, CUTOFF_DATE)

    # Extract the filterdf passed to get_price_cost_profit
    call_args = loader.get_price_cost_profit.call_args
    passed_filterdf = call_args[0][0]
    assert (passed_filterdf['MONTH'] < CUTOFF_MONTH).all(), (
        "Month M was included in the filterdf passed to get_price_cost_profit — anti-leak violation"
    )


def test_hist_score_in_unit_interval():
    """All scores must be in [0.0, 1.0] (they are win rates)."""
    loader = MagicMock()
    candidates = _make_candidates(5)
    historical_months = ['2020-05', '2020-06']
    loader.get_all_triplets.return_value = _make_triplets([1, 2, 3, 4, 5], historical_months)
    loader.get_price_cost_profit.return_value = _make_profit_df([1, 2, 3, 4, 5], historical_months, profit_val=3.0)

    result = score_by_historical_profit_rate(loader, candidates, CUTOFF_DATE)

    assert (result['PREDICTED_PROFIT'] >= 0.0).all()
    assert (result['PREDICTED_PROFIT'] <= 1.0).all()


# ---------------------------------------------------------------------------
# score_by_lasso — helpers
# ---------------------------------------------------------------------------

from datachallenge.features import FEATURE_COLUMNS as FEATURE_COLUMNS_FOR_TEST


def _make_feature_df(loader, triplets_df, cutoff_date):
    """Return a fake feature matrix for the given triplets DataFrame."""
    rng = np.random.default_rng(0)
    df = triplets_df[['EID', 'MONTH', 'PEAKID']].copy().reset_index(drop=True)
    for col in FEATURE_COLUMNS_FOR_TEST:
        df[col] = rng.standard_normal(len(df))
    return df


def _setup_lasso_loader(train_months, train_eids, cand_eids, profit_val=5.0):
    """Create a mock loader suitable for lasso tests."""
    loader = MagicMock()
    all_triplets = _make_triplets(train_eids, train_months + [CUTOFF_MONTH])
    loader.get_all_triplets.return_value = all_triplets
    loader.get_price_cost_profit.return_value = _make_profit_df(train_eids, train_months, profit_val=profit_val)
    return loader


# ---------------------------------------------------------------------------
# score_by_lasso — tests
# ---------------------------------------------------------------------------

def test_lasso_output_columns():
    """Output must have exactly EID, MONTH, PEAKID, PREDICTED_PROFIT."""
    loader = _setup_lasso_loader(['2020-04', '2020-05', '2020-06'], [1, 2, 3], [1, 2, 3])
    candidates = _make_candidates(3)

    with __import__('unittest.mock', fromlist=['patch']).patch(
        'datachallenge.scoring.build_feature_matrix', side_effect=_make_feature_df
    ):
        result = score_by_lasso(loader, candidates, CUTOFF_DATE)

    assert set(result.columns) == {'EID', 'MONTH', 'PEAKID', 'PREDICTED_PROFIT'}


def test_lasso_all_candidates_returned():
    """Every candidate triplet must appear in the output (no rows dropped)."""
    loader = _setup_lasso_loader(['2020-04', '2020-05', '2020-06'], [1, 2, 3, 4], [1, 2, 3, 4])
    candidates = _make_candidates(4)

    with __import__('unittest.mock', fromlist=['patch']).patch(
        'datachallenge.scoring.build_feature_matrix', side_effect=_make_feature_df
    ):
        result = score_by_lasso(loader, candidates, CUTOFF_DATE)

    assert len(result) == len(candidates)
    input_keys = set(zip(candidates['EID'], candidates['MONTH'], candidates['PEAKID']))
    output_keys = set(zip(result['EID'], result['MONTH'], result['PEAKID']))
    assert input_keys == output_keys


def test_lasso_no_history_returns_zero():
    """When there is no history before cutoff_month, all candidates get score 0."""
    loader = MagicMock()
    # Only month M in all_triplets — nothing strictly before it
    loader.get_all_triplets.return_value = _make_triplets([1, 2], [CUTOFF_MONTH])
    candidates = _make_candidates(2)

    result = score_by_lasso(loader, candidates, CUTOFF_DATE)

    assert (result['PREDICTED_PROFIT'] == 0.0).all()


def test_lasso_no_profit_data_returns_zero():
    """When get_price_cost_profit returns empty, all candidates get score 0."""
    loader = MagicMock()
    loader.get_all_triplets.return_value = _make_triplets([1, 2], ['2020-05', '2020-06'])
    loader.get_price_cost_profit.return_value = pd.DataFrame()
    candidates = _make_candidates(2)

    result = score_by_lasso(loader, candidates, CUTOFF_DATE)

    assert (result['PREDICTED_PROFIT'] == 0.0).all()


def test_lasso_predicted_profit_is_numeric():
    """PREDICTED_PROFIT must be numeric (float), not NaN."""
    loader = _setup_lasso_loader(
        ['2020-01', '2020-02', '2020-03', '2020-04', '2020-05', '2020-06'],
        [1, 2, 3, 4, 5],
        [1, 2, 3],
    )
    candidates = _make_candidates(3)

    with __import__('unittest.mock', fromlist=['patch']).patch(
        'datachallenge.scoring.build_feature_matrix', side_effect=_make_feature_df
    ):
        result = score_by_lasso(loader, candidates, CUTOFF_DATE)

    assert result['PREDICTED_PROFIT'].notna().all()
    assert result['PREDICTED_PROFIT'].dtype in [np.float32, np.float64, float]


def test_lasso_excludes_cutoff_month_from_training():
    """get_price_cost_profit must not be called with month M triplets (anti-leak)."""
    loader = MagicMock()
    loader.get_all_triplets.return_value = pd.DataFrame({
        'EID': [1, 1],
        'MONTH': [CUTOFF_MONTH, '2020-06'],
        'PEAKID': [0, 0],
    })
    loader.get_price_cost_profit.return_value = pd.DataFrame({
        'EID': [1], 'MONTH': ['2020-06'], 'PEAKID': [0],
        'PRICE': [10.0], 'COST': [5.0], 'PROFIT': [5.0],
    })
    candidates = _make_candidates(1)

    with __import__('unittest.mock', fromlist=['patch']).patch(
        'datachallenge.scoring.build_feature_matrix', side_effect=_make_feature_df
    ):
        score_by_lasso(loader, candidates, CUTOFF_DATE)

    call_args = loader.get_price_cost_profit.call_args
    passed_filterdf = call_args[0][0]
    assert (passed_filterdf['MONTH'] < CUTOFF_MONTH).all(), (
        "Month M was included in the filterdf passed to get_price_cost_profit — anti-leak violation"
    )


# ---------------------------------------------------------------------------
# score_by_lightgbm — tests
# ---------------------------------------------------------------------------


def test_lightgbm_output_columns():
    """Output must have exactly EID, MONTH, PEAKID, PREDICTED_PROFIT."""
    loader = _setup_lasso_loader(
        ['2020-01', '2020-02', '2020-03', '2020-04', '2020-05', '2020-06'],
        list(range(1, 11)), list(range(1, 11))
    )
    candidates = _make_candidates(5)

    with __import__('unittest.mock', fromlist=['patch']).patch(
        'datachallenge.scoring.build_feature_matrix', side_effect=_make_feature_df
    ):
        result = score_by_lightgbm(loader, candidates, CUTOFF_DATE)

    assert set(result.columns) == {'EID', 'MONTH', 'PEAKID', 'PREDICTED_PROFIT'}


def test_lightgbm_all_candidates_returned():
    """Every candidate triplet must appear in the output (no rows dropped)."""
    loader = _setup_lasso_loader(
        ['2020-01', '2020-02', '2020-03', '2020-04', '2020-05', '2020-06'],
        list(range(1, 11)), list(range(1, 6))
    )
    candidates = _make_candidates(5)

    with __import__('unittest.mock', fromlist=['patch']).patch(
        'datachallenge.scoring.build_feature_matrix', side_effect=_make_feature_df
    ):
        result = score_by_lightgbm(loader, candidates, CUTOFF_DATE)

    assert len(result) == len(candidates)
    input_keys = set(zip(candidates['EID'], candidates['MONTH'], candidates['PEAKID']))
    output_keys = set(zip(result['EID'], result['MONTH'], result['PEAKID']))
    assert input_keys == output_keys


def test_lightgbm_no_history_returns_zero():
    """When there is no history before cutoff_month, all candidates get score 0."""
    loader = MagicMock()
    loader.get_all_triplets.return_value = _make_triplets([1, 2], [CUTOFF_MONTH])
    candidates = _make_candidates(2)

    result = score_by_lightgbm(loader, candidates, CUTOFF_DATE)

    assert (result['PREDICTED_PROFIT'] == 0.0).all()


def test_lightgbm_no_profit_data_returns_zero():
    """When get_price_cost_profit returns empty, all candidates get score 0."""
    loader = MagicMock()
    loader.get_all_triplets.return_value = _make_triplets([1, 2], ['2020-05', '2020-06'])
    loader.get_price_cost_profit.return_value = pd.DataFrame()
    candidates = _make_candidates(2)

    result = score_by_lightgbm(loader, candidates, CUTOFF_DATE)

    assert (result['PREDICTED_PROFIT'] == 0.0).all()


def test_lightgbm_few_training_rows_returns_zero():
    """When fewer than min_child_samples training rows, all candidates get score 0."""
    loader = MagicMock()
    # Only 2 triplets in history (< 100 min_child_samples)
    loader.get_all_triplets.return_value = _make_triplets([1], ['2020-06'])
    loader.get_price_cost_profit.return_value = _make_profit_df([1], ['2020-06'])
    candidates = _make_candidates(2)

    result = score_by_lightgbm(loader, candidates, CUTOFF_DATE)

    assert (result['PREDICTED_PROFIT'] == 0.0).all()


def test_lightgbm_score_in_unit_interval():
    """PREDICTED_PROFIT must be in [0.0, 1.0] (it's predict_proba output)."""
    # Need enough rows to exceed min_child_samples=100
    many_eids = list(range(1, 60))  # 60 EIDs × 2 PEAKIDs × 1 month = 120 rows
    loader = _setup_lasso_loader(
        ['2020-06'], many_eids, list(range(1, 6)), profit_val=5.0
    )
    # Mix profitable and non-profitable
    profit_df = loader.get_price_cost_profit.return_value.copy()
    profit_df.loc[profit_df.index[:len(profit_df) // 2], 'PROFIT'] = -1.0
    loader.get_price_cost_profit.return_value = profit_df
    candidates = _make_candidates(5)

    with __import__('unittest.mock', fromlist=['patch']).patch(
        'datachallenge.scoring.build_feature_matrix', side_effect=_make_feature_df
    ):
        result = score_by_lightgbm(loader, candidates, CUTOFF_DATE)

    assert (result['PREDICTED_PROFIT'] >= 0.0).all()
    assert (result['PREDICTED_PROFIT'] <= 1.0).all()


def test_lightgbm_excludes_cutoff_month_from_training():
    """get_price_cost_profit must not be called with month M triplets (anti-leak)."""
    loader = MagicMock()
    loader.get_all_triplets.return_value = pd.DataFrame({
        'EID': [1, 1],
        'MONTH': [CUTOFF_MONTH, '2020-06'],
        'PEAKID': [0, 0],
    })
    loader.get_price_cost_profit.return_value = pd.DataFrame({
        'EID': [1], 'MONTH': ['2020-06'], 'PEAKID': [0],
        'PRICE': [10.0], 'COST': [5.0], 'PROFIT': [5.0],
    })
    candidates = _make_candidates(1)

    with __import__('unittest.mock', fromlist=['patch']).patch(
        'datachallenge.scoring.build_feature_matrix', side_effect=_make_feature_df
    ):
        score_by_lightgbm(loader, candidates, CUTOFF_DATE)

    call_args = loader.get_price_cost_profit.call_args
    passed_filterdf = call_args[0][0]
    assert (passed_filterdf['MONTH'] < CUTOFF_MONTH).all(), (
        "Month M was included in the filterdf passed to get_price_cost_profit — anti-leak violation"
    )


def test_lightgbm_no_feature_name_warning():
    """score_by_lightgbm must not emit sklearn/LightGBM feature-name warnings.

    Regression test: previously, fitting with a DataFrame and predicting with
    a numpy array triggered: 'X does not have valid feature names, but
    LGBMClassifier was fitted with feature names'. Fixed by passing DataFrames
    to both fit and predict_proba.
    """
    import warnings

    many_eids = list(range(1, 60))  # 120 rows > min_child_samples=100
    loader = _setup_lasso_loader(['2020-06'], many_eids, list(range(1, 6)), profit_val=5.0)
    profit_df = loader.get_price_cost_profit.return_value.copy()
    profit_df.loc[profit_df.index[:len(profit_df) // 2], 'PROFIT'] = -1.0
    loader.get_price_cost_profit.return_value = profit_df
    candidates = _make_candidates(5)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with __import__('unittest.mock', fromlist=['patch']).patch(
            'datachallenge.scoring.build_feature_matrix', side_effect=_make_feature_df
        ):
            score_by_lightgbm(loader, candidates, CUTOFF_DATE)

    feature_name_warnings = [
        w for w in caught if "feature names" in str(w.message).lower()
    ]
    assert len(feature_name_warnings) == 0, (
        f"Unexpected feature name warning(s): {[str(w.message) for w in feature_name_warnings]}"
    )
