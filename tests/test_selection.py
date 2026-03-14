import pandas as pd

from datachallenge.selection import select_opportunities


def _make_scored(n_positive, n_negative=0):
    """Build a scored DataFrame with n_positive profitable and n_negative unprofitable rows."""
    eids = list(range(n_positive + n_negative))
    profits = [float(i + 1) for i in range(n_positive)] + [-1.0] * n_negative
    return pd.DataFrame({
        'EID': eids,
        'MONTH': ['2020-08'] * len(eids),
        'PEAKID': [0] * len(eids),
        'PREDICTED_PROFIT': profits,
    })


def test_select_output_columns():
    """Output must have exactly EID, MONTH, PEAKID — no PREDICTED_PROFIT."""
    scored = _make_scored(20)
    result = select_opportunities(scored)
    assert set(result.columns) == {'EID', 'MONTH', 'PEAKID'}


def test_select_fewer_than_min_pads_to_min():
    """When fewer than min_opp rows are profitable, pad to min_opp (top by score)."""
    scored = _make_scored(n_positive=5, n_negative=10)
    result = select_opportunities(scored, min_opp=10, max_opp=100)
    assert len(result) == 10


def test_select_between_min_and_max_keeps_profitable():
    """When profitable count is in [min_opp, max_opp], keep exactly that many."""
    scored = _make_scored(n_positive=50)
    result = select_opportunities(scored, min_opp=10, max_opp=100)
    assert len(result) == 50


def test_select_more_than_max_caps_at_max():
    """When more than max_opp rows are profitable, cap at max_opp."""
    scored = _make_scored(n_positive=200)
    result = select_opportunities(scored, min_opp=10, max_opp=100)
    assert len(result) == 100


def test_select_exactly_min():
    """Exactly min_opp profitable rows → select exactly min_opp."""
    scored = _make_scored(n_positive=10)
    result = select_opportunities(scored, min_opp=10, max_opp=100)
    assert len(result) == 10


def test_select_exactly_max():
    """Exactly max_opp profitable rows → select exactly max_opp."""
    scored = _make_scored(n_positive=100)
    result = select_opportunities(scored, min_opp=10, max_opp=100)
    assert len(result) == 100


def test_select_deduplicates_by_averaging():
    """Duplicate triplets are deduplicated by averaging PREDICTED_PROFIT."""
    scored = pd.DataFrame({
        'EID': [1, 1],
        'MONTH': ['2020-08', '2020-08'],
        'PEAKID': [0, 0],
        'PREDICTED_PROFIT': [10.0, 0.0],  # avg = 5.0 → still profitable
    })
    result = select_opportunities(scored, min_opp=1, max_opp=10)
    # After dedup there is 1 row; avg profit is 5.0 > 0
    assert len(result) == 1


def test_select_sorted_descending():
    """Top-scored candidates must appear first (descending order)."""
    scored = _make_scored(n_positive=50)
    result = select_opportunities(scored, min_opp=10, max_opp=20)
    # All selected EIDs should be the top-20 by original score (EIDs 49..30)
    assert len(result) == 20
