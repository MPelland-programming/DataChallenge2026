import pandas as pd
import pytest

from datachallenge.loader import select_best_predict_and_print

# Fixed reference date used across all tests. Mid-2020 was chosen because the git
# history references July 2020 data, so it is safe to assume rows exist there.
# Keeping one shared date makes test results comparable and deterministic.
CUTOFF_DATE = "2020-07-07"
CUTOFF_MONTH = "2020-07"


def test_get_all_triplets_columns_and_cutoff(loader):
    result = loader.get_all_triplets(CUTOFF_DATE)

    # set() comparison so column order does not matter
    assert set(result.columns) == {"MONTH", "PEAKID", "EID"}

    # Anti-leak rule: the loader must never return rows from months after the
    # cutoff. Returning future months would let the scoring logic peek at data
    # that was not available at decision time (the 7th of month M).
    assert (result["MONTH"] <= CUTOFF_MONTH).all(), "Some rows have MONTH > cutoff month"


def test_get_price_cost_profit(loader):
    triplets = loader.get_all_triplets(CUTOFF_DATE)
    result = loader.get_price_cost_profit(filterdf=triplets)

    # The loader applies ABS() to both raw columns. A negative cost or price
    # here means the ABS() was dropped somewhere, which would silently corrupt
    # the profit signal used for ranking.
    assert (result["COST"] >= 0).all()
    assert (result["PRICE"] >= 0).all()

    # PROFIT must equal PRICE - COST exactly, element-wise. assert_series_equal
    # checks every value; check_names=False because the two series carry
    # different names and that is not what we are testing.
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

    # The loader computes cutoff_he = cutoff_date + 1 day (2020-07-08T00:00:00)
    # to include HE24 of the 7th: the last hourly ending of July 7th is
    # conventionally recorded as midnight on the 8th. So the correct bound is
    # <= 2020-07-08T00:00:00, not <= 2020-07-07.
    cutoff_ts = pd.Timestamp(CUTOFF_DATE) + pd.DateOffset(days=1)
    assert max_dt <= cutoff_ts, f"Daily sim data contains rows past cutoff: {max_dt}"


def test_select_best_predict_and_print(loader, tmp_path, monkeypatch):
    triplets = loader.get_all_triplets(CUTOFF_DATE)

    # Use real triplets so the input is realistic, but attach synthetic
    # PREDICTED_PROFIT = 1, 2, 3, ... The function only cares about ranking,
    # not the actual values, so this is sufficient and fully deterministic.
    resultdf = triplets.copy()
    resultdf["PREDICTED_PROFIT"] = range(1, len(resultdf) + 1)

    # The function writes opportunities.csv to the current working directory
    # (hardcoded). monkeypatch.chdir redirects that write into tmp_path, a
    # pytest built-in that provides a fresh temporary directory per test, so
    # the project root is never polluted and the test is self-cleaning.
    monkeypatch.chdir(tmp_path)
    select_best_predict_and_print(resultdf, min_opp=10, max_opp=100)

    out = tmp_path / "opportunities.csv"
    assert out.exists(), "opportunities.csv was not written"

    written = pd.read_csv(out)
    # Row count must respect the [min_opp, max_opp] contract defined by the
    # function signature. Violating either bound means the selection logic is
    # broken regardless of prediction quality.
    assert 10 <= len(written) <= 100, f"Output has {len(written)} rows, expected 10–100"

    # Output columns must be exactly TARGET_MONTH, PEAK_TYPE, EID — in that order.
    # This is the contract expected by downstream consumers of opportunities.csv.
    assert list(written.columns) == ["TARGET_MONTH", "PEAK_TYPE", "EID"]

    # PEAK_TYPE must only contain "ON" or "OFF" — never raw PEAKID integers.
    assert set(written["PEAK_TYPE"].unique()).issubset({"ON", "OFF"})

    # TARGET_MONTH must follow the YYYY-MM format.
    assert written["TARGET_MONTH"].str.match(r"^\d{4}-\d{2}$").all()
