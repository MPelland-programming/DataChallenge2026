"""
Unit tests for datachallenge/output.py (mock-based, no real data required).
"""

import os
import pandas as pd
import pytest

from datachallenge.output import write_opportunities


def _make_selected(n_per_month=15, months=None, eids=None):
    """Build a valid selection DataFrame (EID, MONTH, PEAKID)."""
    if months is None:
        months = ['2020-08']
    if eids is None:
        eids = list(range(n_per_month))
    rows = []
    for month in months:
        for eid in eids:
            rows.append({'EID': eid, 'MONTH': month, 'PEAKID': eid % 2})
    return pd.DataFrame(rows)


def test_write_creates_file(tmp_path):
    """write_opportunities must create the file at the given path."""
    df = _make_selected(15)
    out = str(tmp_path / 'opportunities.csv')
    write_opportunities(df, out)
    assert os.path.exists(out)


def test_write_correct_columns(tmp_path):
    """Output CSV must have exactly TARGET_MONTH, PEAK_TYPE, EID columns."""
    df = _make_selected(15)
    out = str(tmp_path / 'opportunities.csv')
    write_opportunities(df, out)
    written = pd.read_csv(out)
    assert list(written.columns) == ['TARGET_MONTH', 'PEAK_TYPE', 'EID']


def test_write_peak_type_mapping(tmp_path):
    """PEAKID 0→OFF and 1→ON must be mapped correctly."""
    df = pd.DataFrame({
        'EID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        'MONTH': ['2020-08'] * 15,
        'PEAKID': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    })
    out = str(tmp_path / 'opportunities.csv')
    write_opportunities(df, out)
    written = pd.read_csv(out)
    assert set(written['PEAK_TYPE'].unique()).issubset({'ON', 'OFF'})
    # Check specific mappings
    assert written.loc[written['EID'] == 1, 'PEAK_TYPE'].iloc[0] == 'OFF'
    assert written.loc[written['EID'] == 2, 'PEAK_TYPE'].iloc[0] == 'ON'


def test_write_target_month_format(tmp_path):
    """TARGET_MONTH in output must match YYYY-MM format."""
    df = _make_selected(15, months=['2020-08', '2020-09'])
    out = str(tmp_path / 'opportunities.csv')
    write_opportunities(df, out)
    written = pd.read_csv(out)
    assert written['TARGET_MONTH'].str.match(r'^\d{4}-\d{2}$').all()


def test_write_raises_on_too_few(tmp_path):
    """Fewer than 10 selections for a month must raise ValueError."""
    df = _make_selected(5)  # only 5 rows for 2020-08
    out = str(tmp_path / 'opportunities.csv')
    with pytest.raises(ValueError, match='10'):
        write_opportunities(df, out)


def test_write_raises_on_too_many(tmp_path):
    """More than 100 selections for a month must raise ValueError."""
    df = _make_selected(101)
    out = str(tmp_path / 'opportunities.csv')
    with pytest.raises(ValueError, match='100'):
        write_opportunities(df, out)


def test_write_raises_on_duplicates(tmp_path):
    """Duplicate (TARGET_MONTH, PEAK_TYPE, EID) rows must raise ValueError."""
    df = pd.DataFrame({
        'EID': [1] * 2 + list(range(2, 16)),       # EID 1 appears twice (both OFF)
        'MONTH': ['2020-08'] * 16,
        'PEAKID': [0] * 16,
    })
    out = str(tmp_path / 'opportunities.csv')
    with pytest.raises(ValueError, match='[Dd]uplicate'):
        write_opportunities(df, out)


def test_write_raises_on_invalid_peakid(tmp_path):
    """PEAKID values other than 0 or 1 must raise ValueError after failing mapping."""
    df = pd.DataFrame({
        'EID': list(range(15)),
        'MONTH': ['2020-08'] * 15,
        'PEAKID': [2] * 15,  # invalid: maps to NaN via {0:'OFF', 1:'ON'}
    })
    out = str(tmp_path / 'opportunities.csv')
    with pytest.raises(ValueError, match='PEAK_TYPE'):
        write_opportunities(df, out)


def test_write_multi_month(tmp_path):
    """Multiple months in one DataFrame are all written correctly."""
    df = _make_selected(15, months=['2020-08', '2020-09', '2020-10'])
    out = str(tmp_path / 'opportunities.csv')
    write_opportunities(df, out)
    written = pd.read_csv(out)
    assert set(written['TARGET_MONTH'].unique()) == {'2020-08', '2020-09', '2020-10'}
    for month in ['2020-08', '2020-09', '2020-10']:
        count = (written['TARGET_MONTH'] == month).sum()
        assert 10 <= count <= 100
