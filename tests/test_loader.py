import pandas as pd
import pytest

from datachallenge.loader import select_best_predict_and_print

# A mid-range cutoff date expected to be within the dataset
CUTOFF_DATE = "2020-07-07"
CUTOFF_MONTH = "2020-07"


def test_get_all_triplets_columns_and_cutoff(loader):
    result = loader.get_all_triplets(CUTOFF_DATE)

    assert set(result.columns) == {"MONTH", "PEAKID", "EID"}
    assert (result["MONTH"] <= CUTOFF_MONTH).all(), "Some rows have MONTH > cutoff month"


def test_get_price_cost_profit(loader):
    triplets = loader.get_all_triplets(CUTOFF_DATE)
    result = loader.get_price_cost_profit(filterdf=triplets)

    assert (result["COST"] >= 0).all()
    assert (result["PRICE"] >= 0).all()
    pd.testing.assert_series_equal(
        result["PROFIT"],
        result["PRICE"] - result["COST"],
        check_names=False,
    )


def test_get_daily_data_cutoff(loader):
    triplets = loader.get_all_triplets(CUTOFF_DATE)
    result = loader.get_daily_data(triplets, CUTOFF_DATE)

    if result.empty:
        pytest.skip("No daily sim data for the test period")

    max_dt = pd.to_datetime(result["DATETIME"]).max()
    # The loader allows up to HE24 of the 7th (i.e. 8th 00:00)
    cutoff_ts = pd.Timestamp(CUTOFF_DATE) + pd.DateOffset(days=1)
    assert max_dt <= cutoff_ts, f"Daily sim data contains rows past cutoff: {max_dt}"


def test_select_best_predict_and_print(loader, tmp_path, monkeypatch):
    triplets = loader.get_all_triplets(CUTOFF_DATE)

    resultdf = triplets.copy()
    resultdf["PREDICTED_PROFIT"] = range(1, len(resultdf) + 1)

    monkeypatch.chdir(tmp_path)
    select_best_predict_and_print(resultdf, min_opp=10, max_opp=100)

    out = tmp_path / "opportunities.csv"
    assert out.exists(), "opportunities.csv was not written"

    written = pd.read_csv(out)
    assert 10 <= len(written) <= 100, f"Output has {len(written)} rows, expected 10–100"
