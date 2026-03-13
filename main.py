import argparse
import os
import pandas as pd
from datachallenge.loader import CustomDataLoader


# Get input
parser = argparse.ArgumentParser()
parser.add_argument('--start-month', type=str, required=True, help='Start month in the format YYYY-MM')
parser.add_argument('--end-month', type=str, required=True, help='End month in the format YYYY-MM')
parser.add_argument('--data-root', type=str, default='data', help='Path to data folder')
args = parser.parse_args()


# Paths perso pour tester seulement
DATA_ROOT = r"G:\.shortcut-targets-by-id\1SR7TxhNQjFyS8f2bbuo9pcLvzNMLBMwY\CSD\data"
costfile = os.path.join(DATA_ROOT, 'costs', 'costs.parquet')
pricefile = os.path.join(DATA_ROOT, 'prices', 'prices.parquet')

# --- Paths ---
# DATA_ROOT = args.data_root
# costfile = os.path.join(DATA_ROOT, 'costs', 'costs.parquet')
# pricefile = os.path.join(DATA_ROOT, 'prices', 'prices.parquet')

# derive years dynamically from the date range
start = pd.Timestamp(args.start_month + '-01')
end = pd.Timestamp(args.end_month + '-01')
all_years = [str(y) for y in range(start.year, end.year + 1)]

dsimfile = [os.path.join(DATA_ROOT, 'sim_daily', f'sim_daily_{y}.parquet')
            for y in all_years if os.path.exists(os.path.join(DATA_ROOT, 'sim_daily', f'sim_daily_{y}.parquet'))]
msimfile = [os.path.join(DATA_ROOT, 'sim_monthly', f'sim_monthly_{y}.parquet')
            for y in all_years if os.path.exists(os.path.join(DATA_ROOT, 'sim_monthly', f'sim_monthly_{y}.parquet'))]

loader = CustomDataLoader(pricefile, costfile, dsimfile, msimfile)

# --- Fix #2 & #3: month-by-month loop with cutoff logic ---
# Generate list of target months M+1 between start and end
target_months = pd.date_range(start, end, freq='MS')  # first day of each target month

all_selections = []

for target_date in target_months:
    target_month = target_date.strftime('%Y-%m')  # M+1
    # Decision month M is the month before the target
    decision_date = target_date - pd.DateOffset(months=1)
    cutoff_date = decision_date.replace(day=7).strftime('%Y-%m-%d')  # 7th of M

    print(f"[Target: {target_month}] Cutoff: {cutoff_date}")

    # --- Build candidate filterdf for M+1 simulations ---
    # Monthly sims for M+1 are available at cutoff
    candidate_filter = pd.DataFrame({
        'MONTH': [target_month],
        'PEAKID': [0]
    })
    # We need both ON and OFF candidates — get all EIDs from monthly sims
    # For now, use historical triplets visible at cutoff as candidate pool
    historical = loader.get_all_triplets(cutoff_date)

    # Get unique EIDs seen historically
    known_eids = historical['EID'].unique()

    # Build candidates: all known EIDs × {ON, OFF} for target_month
    candidates = pd.DataFrame({
        'EID': list(known_eids) * 2,
        'MONTH': [target_month] * len(known_eids) * 2,
        'PEAKID': [0] * len(known_eids) + [1] * len(known_eids),
    })

    # ---------------------------------------------------------
    # TODO: Scoring logic here
    # Load monthly sims for target_month, daily sims up to cutoff,
    # historical profits, then score each candidate.
    # Example:
    #   monthly_sim = loader.get_monthly_data(candidates)
    #   daily_sim = loader.get_daily_data(candidates, cutoff_date)
    #   historical_profit = loader.get_price_cost_profit(filterdf=historical)
    # Then build features, rank, and select top 10-100.
    # ---------------------------------------------------------

    # Placeholder: select top 50 random (replace with real scoring)
    import numpy as np
    n_select = min(50, len(candidates))
    selected = candidates.sample(n=n_select, random_state=42)

    all_selections.append(selected)

# --- Build output CSV in the END
# output = pd.concat(all_selections, ignore_index=True)
# output['PEAK_TYPE'] = output['PEAKID'].map({0: 'OFF', 1: 'ON'})
# output = output.rename(columns={'MONTH': 'TARGET_MONTH'})
# output = output[['TARGET_MONTH', 'PEAK_TYPE', 'EID']].drop_duplicates()

# output.to_csv(os.path.join(DATA_ROOT, 'opportunities.csv'), index=False)
# print(f"Output: {len(output)} rows saved to opportunities.csv")