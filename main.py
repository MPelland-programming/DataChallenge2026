import argparse
import os
import pandas as pd
from datachallenge.loader import CustomDataLoader
from datachallenge.config import settings
from datachallenge.logger import logger
from datachallenge.selection import score_by_activation_level


# Get input
parser = argparse.ArgumentParser(
    description=(
        'Identify profitable FTR (Financial Transmission Rights) opportunities '
        'for a range of target months. For each month M+1 in the given range, '
        'the algorithm uses data available up to the 7th of month M (cutoff) '
        'to select between 10 and 100 ON-Peak/OFF-Peak opportunities. '
        'Results are written to opportunities.csv in the data folder.'
    ),
    epilog=(
        'Examples:\n'
        '  # Full provided dataset (default, no args needed):\n'
        '  python main.py\n'
        '\n'
        '  # Training set only (2020-2022):\n'
        '  python main.py --end-month 2022-12\n'
        '\n'
        '  # Validation set only (2023):\n'
        '  python main.py --start-month 2023-01\n'
        '\n'
        '  # Out-of-sample test (2024, when data is available):\n'
        '  python main.py --start-month 2024-01 --end-month 2024-12\n'
        '\n'
        '  # With explicit data path and debug logging:\n'
        '  python main.py --data-root /path/to/data --log-level DEBUG'
    ),
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument(
    '--start-month',
    type=str,
    default='2020-01',
    metavar='YYYY-MM',
    help='First target month to evaluate (e.g. 2024-01). The cutoff will be set to the 7th of the preceding month. Defaults to 2020-01.',
)
parser.add_argument(
    '--end-month',
    type=str,
    default='2023-12',
    metavar='YYYY-MM',
    help='Last target month to evaluate (e.g. 2024-06). Must be >= start-month. Defaults to 2023-12.',
)
parser.add_argument(
    '--data-root',
    type=str,
    default=None,
    metavar='PATH',
    help='Path to the data folder containing costs/, prices/, sim_daily/, sim_monthly/. Overrides DATA_ROOT in .env.',
)
parser.add_argument(
    '--log-level',
    type=str,
    default=None,
    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
    metavar='LEVEL',
    help='Logging verbosity: DEBUG, INFO, WARNING, or ERROR. Overrides LOG_LEVEL in .env.',
)
args = parser.parse_args()

# Apply CLI overrides
if args.log_level:
    logger.setLevel(args.log_level.upper())

DATA_ROOT = args.data_root if args.data_root else settings.data_root
costfile = os.path.join(DATA_ROOT, 'costs', 'costs.parquet')
pricefile = os.path.join(DATA_ROOT, 'prices', 'prices.parquet')

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

    logger.info(f"[Target: {target_month}] Cutoff: {cutoff_date}")

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

    # Score candidates by mean ACTIVATIONLEVEL in monthly sims for target_month
    scored = score_by_activation_level(loader, candidates, cutoff_date)

    # Apply 10–100 selection constraint (mirrors select_best_predict_and_print logic)
    scored_sorted = scored.sort_values('PREDICTED_PROFIT', ascending=False)
    n_profitable = int((scored_sorted['PREDICTED_PROFIT'] > 0).sum())
    if n_profitable < 10:
        selected = scored_sorted.head(10)[['EID', 'MONTH', 'PEAKID']]
    elif n_profitable <= 100:
        selected = scored_sorted.head(n_profitable)[['EID', 'MONTH', 'PEAKID']]
    else:
        selected = scored_sorted.head(100)[['EID', 'MONTH', 'PEAKID']]

    logger.info(f"[Target: {target_month}] Selected {len(selected)} opportunities "
                f"({n_profitable} with positive ACTIVATIONLEVEL score)")
    all_selections.append(selected)

# --- Build output CSV
output = pd.concat(all_selections, ignore_index=True)
output['PEAK_TYPE'] = output['PEAKID'].map({0: 'OFF', 1: 'ON'})
output = output.rename(columns={'MONTH': 'TARGET_MONTH'})
output = output[['TARGET_MONTH', 'PEAK_TYPE', 'EID']].drop_duplicates()

project_root = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(project_root, 'opportunities.csv')
output.to_csv(output_path, index=False)
logger.info(f"Output: {len(output)} rows saved to {output_path}")
