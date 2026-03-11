import pyarrow.parquet as pq
import pandas as pd
import duckdb

class customdataloader:
    #input: path to price and cost files
    #      pricefile: path to price parquet file
    #      costfile: path to cost parquet file
    #      dsimfile: list of paths to daily simulation parquet files (optional)
    #      msimfile: list of paths to monthly simulation parquet files (optional)

    def __init__(self
                 , pricefile: str
                 , costfile: str
                 , dsimfile = list
                 , msimfile = list
                 ):
        self.pricefile = pricefile
        self.costfile = costfile
        self.dsimulation = dsimfile
        self.msimulation = msimfile
        self.get_all_triplets()
        self.day_sim_col = pq.read_schema(self.dsimulation[0]).names
        self.month_sim_col = pq.read_schema(self.msimulation[0]).names
        self.price_col = pq.read_schema(pricefile).names


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
        if isinstance(filterdf, str) and filterdf == "all":
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

    def get_daily_data(self,filterdf):
        #Extract daily simulation information based on filterdf.
        #Input
        #   filterdf: df with three columns: MONTH, PEAKID, EID with nrow = batch size

        #get list of all unique years in the filterdf
        list_years = pd.to_datetime(filterdf['MONTH']).dt.year.unique().astype(str).tolist()

        #go through all daily simulation files and check if the year falls within the range of the filterdf.

        # create empty df to store daily data with same columns as daily simulation files
        dailysim = pd.DataFrame(columns=self.day_sim_col)

        #loop over daily files
        for dd in self.dsimulation:
            #extract year from file name
            fileyear = dd.split("/")[-1].split(".")[0][-4:]

            if (fileyear in list_years):
                temp = duckdb.query(f"""
                    SELECT d.*
                    FROM read_parquet('{dd}') d
                        INNER JOIN filterdf f
                        ON  strftime(d.DATETIME, '%Y-%m')   = f.MONTH
                        AND d.PEAKID  = f.PEAKID
                        AND d.EID     = f.EID
                    """).to_df()

                #append temp to dailysim
                dailysim = pd.concat([dailysim, temp], ignore_index=True)

            # append d to all column names
            newcolnames = ["d"+col for col in self.day_sim_col]
            dailysim.columns = newcolnames

        return dailysim

    def get_monthly_data(self,filterdf):
        #Extract montlhy simulation information based on filterdf.
        #Input
        #   filterdf: df with three columns: MONTH, PEAKID, EID with nrow = batch size

        #get list of all unique years in the filterdf
        list_years = pd.to_datetime(filterdf['MONTH']).dt.year.unique().astype(str).tolist()

        # create empty df to store daily data with same columns as daily simulation files
        monthlysim = pd.DataFrame(columns=self.month_sim_col)

        #loop over daily files
        for mm in self.dsimulation:
            #extract year from file name
            fileyear = mm.split("/")[-1].split(".")[0][-4:]

            if (fileyear in list_years):
                temp = duckdb.query(f"""
                    SELECT d.*
                    FROM read_parquet('{mm}') d
                        INNER JOIN filterdf f
                        ON  strftime(d.DATETIME, '%Y-%m')   = f.MONTH
                        AND d.PEAKID  = f.PEAKID
                        AND d.EID     = f.EID
                    """).to_df()

                #append temp to dailysim
                monthlysim = pd.concat([monthlysim, temp], ignore_index=True)

            newcolnames = ["m" + col for col in self.month_sim_col]
            monthlysim.columns = newcolnames

        return montlhysim


                    #filterdf = pd.merge(filterdf, daily_data, on=['EID', 'MONTH','PEAKID'], how='left')

    def get_daily_price(self, filterdf):
        #get list of all unique years in the filterdf
        list_years = pd.to_datetime(filterdf['MONTH']).dt.year.unique().astype(str).tolist()

        # create empty df to store daily data with same columns as daily simulation files
        dailyprice = pd.DataFrame(columns=self.price_col)

        dailyprice = duckdb.query(f"""
            SELECT p.*
            FROM read_parquet('{self.pricefile}') p
                INNER JOIN filterdf f
                ON  strftime(p.DATETIME, '%Y-%m')   = f.MONTH
                AND p.PEAKID  = f.PEAKID
                AND p.EID     = f.EID
            """).to_df()

        return dailyprice

    def get_sim_price_profit(self, filterdf):
        #This function merges the daily and monthly simulation data with the price, cost, and profit data based on the filterdf.
        #Input
        #   filterdf: df with three columns: MONTH, PEAKID, EID with nrow = batch size

        monthly_info = self.get_price_cost_profit(filterdf=filterdf)
        daily_price = self.get_daily_price(filterdf)

        daily_sim = self.get_daily_data(filterdf)
        monthly_sim = self.get_monthly_data(filterdf)

        #merged = pd.merge(copri, daily_data, on=['EID', 'MONTH','PEAKID'], how='outer').fillna(0)
        #merged = pd.merge(merged, monthly_data, on=['EID', 'MONTH','PEAKID'], how='outer').fillna(0)

        return (monthly_info, daily_price, daily_sim, monthly_sim)













