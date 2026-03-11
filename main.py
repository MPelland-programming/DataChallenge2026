import argparse
import os
import utilities as util
import numpy as np
# Get input
#parser = argparse.ArgumentParser()
#parser.add_argument('--start-month', type=str, required=True, help='Start month in the format YYYY-MM')
#parser.add_argument('--end-month', type=str, required=True, help='End month in the format YYYY-MM')
#args = parser.parse_args()

#obtain path
cwd = os.getcwd()
costfile = os.path.join(cwd, 'data/costs/costs.parquet')
pricefile = os.path.join(cwd, 'data/prices/prices.parquet')

#Find the target variables for all EID, MONTH, PEAKID
baseloader = util.customdataloader(pricefile, costfile)
all_targer_vars = baseloader.get_price_cost_profit(filterdf = "all")