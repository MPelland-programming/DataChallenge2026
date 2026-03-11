import pyarrow.parquet as pq
import pandas as pd
import duckdb

class customdataloader:
    #input: path to price and cost files
    def __init__(self
                 , pricefile: str
                 , costfile: str
                 ):
        self.pricefile = pricefile
        self.costfile = costfile
        self.get_all_triplets()

    def get_all_triplets(self):
        #This function looks through the cost and price files to extract all unique triplets of MONTH, PEAKID, and EID.
        #This is necessary to ensure that we only query for the relevant data when calculating price, cost, and profit.

        all_triplets = duckdb.query(f"""
                SELECT DISTINCT MONTH, PEAKID, EID
                FROM (
                    SELECT p1.MONTH, p1.PEAKID, p1.EID
                    FROM read_parquet('{self.costfile}') p1
                    UNION
                    SELECT strftime(p2.DATETIME, '%Y-%m') AS MONTH, p2.PEAKID, p2.EID
                    FROM read_parquet('{self.pricefile}') p2
                )
            """).to_df()

        self.all_triplets = all_triplets

    def get_price_cost_profit(self, filterdf = "all"):
        # This fun ction takes a list of target triplets (MONTH, PEAKID, EID) and queries the price and cost data to calculate
        # profit for each triplet.
        #filter is a df with three columns: MONTH, PEAKID, EID with nrow = batch size
        if filterdf == "all":
            filterdf = self.all_triplets

        cost = duckdb.query(f"""
                    SELECT p.*
                    FROM read_parquet('{self.costfile}') p
                        INNER JOIN filterdf f
                        ON  p.MONTH = f.MONTH
                        AND p.PEAKID  = f.PEAKID
                        AND p.EID     = f.EID
                    """).to_df()

        #price price summed within triplet
        dailyprice = duckdb.query(f"""
            SELECT p.*
            FROM read_parquet('{self.pricefile}') p
                INNER JOIN filterdf f
                ON  strftime(p.DATETIME, '%Y-%m')   = f.MONTH
                AND p.PEAKID  = f.PEAKID
                AND p.EID     = f.EID
            """).to_df()
        dailyprice['MONTH'] = pd.to_datetime(dailyprice['DATETIME']).dt.to_period('M').astype(str)
        price = dailyprice.groupby(['EID', 'MONTH', 'PEAKID'])["PRICEREALIZED"].sum()

        copri = pd.merge(cost, price, on=['EID', 'MONTH','PEAKID'], how='outer').fillna(0)
        copri = copri.rename(columns={'C': 'COST', 'PRICEREALIZED': 'PRICE'})
        copri['PROFIT'] = copri['PRICE'] - copri['COST']

        return copri















