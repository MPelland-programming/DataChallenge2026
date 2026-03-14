import argparse
import os
import pandas as pd
from datachallenge.loader import CustomDataLoader
from datachallenge.config import settings
from datachallenge.logger import logger
from datachallenge.scoring import score_by_activation_level, score_by_historical_profit_rate, score_by_lasso, score_by_maxime_short
from datachallenge.selection import select_opportunities
from datachallenge.output import write_opportunities
from datachallenge.candidates import build_candidate_pool

# Registry: name → function. Add new scorers/selectors here (one line each).
SCORER_REGISTRY = {
    'activation_level': score_by_activation_level,
    'historical_profit_rate': score_by_historical_profit_rate,
    'maxime_short': score_by_maxime_short,
    'lasso': score_by_lasso,
}
SELECTOR_REGISTRY = {
    'default': select_opportunities,
}

# Best known combination — update this when a better one is found.
# Updated 2026-03-14: activation_level beats historical_profit_rate on both
# 2020-2022 train (F1 0.1337 vs 0.0478) and 2023 val (F1 0.0967 vs 0.0354).
DEFAULT_SCORER = 'activation_level'
DEFAULT_SELECTOR = 'default'

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
        '  python main.py --data-root /path/to/data --log-level DEBUG\n'
        '\n'
        '  # With explicit scorer and selector:\n'
        '  python main.py --scorer historical_profit_rate --selector default'
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
parser.add_argument(
    '--scorer',
    type=str,
    default=DEFAULT_SCORER,
    choices=list(SCORER_REGISTRY),
    metavar='NAME',
    help=f'Scorer to use. Choices: {list(SCORER_REGISTRY)}. Default: {DEFAULT_SCORER}.',
)
parser.add_argument(
    '--selector',
    type=str,
    default=DEFAULT_SELECTOR,
    choices=list(SELECTOR_REGISTRY),
    metavar='NAME',
    help=f'Selector to use. Choices: {list(SELECTOR_REGISTRY)}. Default: {DEFAULT_SELECTOR}.',
)
args = parser.parse_args()

# Apply CLI overrides
if args.log_level:
    logger.setLevel(args.log_level.upper())

scorer_fn = SCORER_REGISTRY[args.scorer]
selector_fn = SELECTOR_REGISTRY[args.selector]
logger.info(f"Using scorer={args.scorer}, selector={args.selector}")

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

# Generate list of target months M+1 between start and end
target_months = pd.date_range(start, end, freq='MS')  # first day of each target month

all_selections = []

for target_date in target_months:
    target_month = target_date.strftime('%Y-%m')  # M+1
    # Decision month M is the month before the target
    decision_date = target_date - pd.DateOffset(months=1)
    cutoff_date = decision_date.replace(day=7).strftime('%Y-%m-%d')  # 7th of M

    logger.info(f"[Target: {target_month}] Cutoff: {cutoff_date}")

    candidates = build_candidate_pool(loader, cutoff_date, target_month)
    scored = scorer_fn(loader, candidates, cutoff_date)
    selected = selector_fn(scored)

    logger.info(f"[Target: {target_month}] Selected {len(selected)} opportunities")
    all_selections.append(selected)

# Build and write output CSV
output = pd.concat(all_selections, ignore_index=True)
project_root = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(project_root, 'opportunities.csv')
write_opportunities(output, output_path)
logger.info(f"Output: {len(output)} rows saved to {output_path}")
