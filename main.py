import argparse
import os
# Get input
parser = argparse.ArgumentParser()
parser.add_argument('--start-month', type=str, required=True, help='Start month in the format YYYY-MM')
parser.add_argument('--end-month', type=str, required=True, help='End month in the format YYYY-MM')
args = parser.parse_args()

#obtain path
cwd = os.getcwd()
filename = os.path.join(cwd, 'data/costs/costs.parquet')