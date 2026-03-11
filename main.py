import argparse

# Get input
parser = argparse.ArgumentParser()
parser.add_argument('--start-month', type=str, required=True, help='Start month in the format YYYY-MM')
parser.add_argument('--end-month', type=str, required=True, help='End month in the format YYYY-MM')
args = parser.parse_args()

import os
import utilities as util
import numpy as np

#obtain path
cwd = os.getcwd()
costfile = os.path.join(cwd, 'data/costs/costs.parquet')
pricefile = os.path.join(cwd, 'data/prices/prices.parquet')

#Find the target variables for all EID, MONTH, PEAKID
baseloader = util.customdataloader(pricefile, costfile)
all_target_vars = baseloader.get_price_cost_profit(filterdf = "all")

#Check how many rare events
#sum(all_target_vars["PROFIT"] > 0)