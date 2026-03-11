import argparse

# Get input
parser = argparse.ArgumentParser()
parser.add_argument('--start-month', type=str, required=True, help='Start month in the format YYYY-MM')
parser.add_argument('--end-month', type=str, required=True, help='End month in the format YYYY-MM')
args = parser.parse_args()

import os
import utilities as util
import pandas as pd
import numpy as np

#obtain path your base path is the one where the main is saved.
cwd = os.getcwd()
costfile = os.path.join(cwd, 'data/costs/costs.parquet')
pricefile = os.path.join(cwd, 'data/prices/prices.parquet')
#dsimfile =[os.path.join(cwd, 'data', 'sim_daily', f'sim_daily_{ff}.parquet') for ff in ["2020", "2021", "2022","2023"]]
#msimfile =[os.path.join(cwd, 'data', 'sim_monthly', f'sim_monthly_{ff}.parquet') for ff in ["2020", "2021", "2022","2023"]]
dsimfile =[os.path.join(cwd, 'data', 'sim_daily', f'sim_daily_{ff}.parquet') for ff in ["2021"]]
msimfile =[os.path.join(cwd, 'data', 'sim_monthly', f'sim_monthly_{ff}.parquet') for ff in ["2021"]]

#Find the target variables for all EID, MONTH, PEAKID
baseloader = util.customdataloader(pricefile, costfile, dsimfile, msimfile)
all_ys = baseloader.get_price_cost_profit(filterdf = "all") #Gets all our monthly Ys.
filterdf = all_ys.head()
#Check how many rare events
#sum(all_target_vars["PROFIT"] > 0)