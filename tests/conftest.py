import glob
import os

import pytest

from datachallenge.loader import CustomDataLoader

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


@pytest.fixture(scope="session")
# create one instance that is reused across the entire session
# because loading data from disk is heavy, and otherwise it would recreate
# it for each test, so it's an optimization
def loader():
    pricefile = os.path.join(DATA_ROOT, "prices", "prices.parquet")
    costfile = os.path.join(DATA_ROOT, "costs", "costs.parquet")

    if not os.path.exists(pricefile) or not os.path.exists(costfile):
        pytest.skip("Local data/ directory not found — skipping integration tests")

    dsimfiles = sorted(glob.glob(os.path.join(DATA_ROOT, "sim_daily", "sim_daily_*.parquet")))
    msimfiles = sorted(glob.glob(os.path.join(DATA_ROOT, "sim_monthly", "sim_monthly_*.parquet")))

    return CustomDataLoader(pricefile, costfile, dsimfiles, msimfiles)
