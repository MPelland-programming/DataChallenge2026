import os
import pyarrow.parquet as pq
import pandas as pd
import duckdb


class CustomDataLoader:
    def __init__(self, pricefile: str, costfile: str, dsimfile=None, msimfile=None):
        self.pricefile = pricefile
        self.costfile = costfile
        self.dsimulation = dsimfile or []
        self.msimulation = msimfile or []
        self.day_sim_col = pq.read_schema(self.dsimulation[0]).names if self.dsimulation else []
        self.month_sim_col = pq.read_schema(self.msimulation[0]).names if self.msimulation else []
        self.price_col = pq.read_schema(pricefile).names

    def get_all_triplets(self, cutoff_date: str) -> pd.DataFrame:  # rows ~ TripletRow
        """
        Return all unique (MONTH, PEAKID, EID) visible at cutoff.
        cutoff_date: 'YYYY-MM-DD' (the 7th of month M).
        Only includes months <= M (the month of the cutoff).
        """
        cutoff_month = cutoff_date[:7]  # 'YYYY-MM'

        all_triplets = duckdb.query(f"""
            SELECT DISTINCT MONTH, PEAKID, EID
            FROM (
                SELECT p1.MONTH, p1.PEAKID, p1.EID
                FROM read_parquet('{self.costfile}') p1
                WHERE p1.MONTH <= '{cutoff_month}'
                UNION
                SELECT strftime(p2.DATETIME, '%Y-%m') AS MONTH, p2.PEAKID, p2.EID
                FROM read_parquet('{self.pricefile}') p2
                WHERE p2.DATETIME <= '{cutoff_date}T23:59:59'
            )
        """).to_df()

        return all_triplets

    def get_price_cost_profit(self, filterdf="all", cutoff_date=None) -> pd.DataFrame:  # rows ~ ProfitRow
        """
        Calculate profit for each triplet. If filterdf='all', uses all triplets
        visible at cutoff_date.
        """
        if isinstance(filterdf, str) and filterdf == "all":
            if cutoff_date is None:
                raise ValueError("cutoff_date required when filterdf='all'")
            filterdf = self.get_all_triplets(cutoff_date)

        cost = duckdb.query(f"""
            SELECT p.*EXCLUDE (C),
            ABS(p.C) AS C
            FROM read_parquet('{self.costfile}') p
                INNER JOIN filterdf f
                ON  p.MONTH   = f.MONTH
                AND p.PEAKID  = f.PEAKID
                AND p.EID     = f.EID
        """).to_df()

        dailyprice = duckdb.query(f"""
            SELECT p.* EXCLUDE (PRICEREALIZED),
            ABS(p.PRICEREALIZED) AS PRICEREALIZED
            FROM read_parquet('{self.pricefile}') p
                INNER JOIN filterdf f
                ON  strftime(p.DATETIME, '%Y-%m') = f.MONTH
                AND p.PEAKID  = f.PEAKID
                AND p.EID     = f.EID
        """).to_df()
        dailyprice['MONTH'] = pd.to_datetime(dailyprice['DATETIME']).dt.to_period('M').astype(str)
        price = dailyprice.groupby(['EID', 'MONTH', 'PEAKID'])["PRICEREALIZED"].sum().reset_index()

        copri = pd.merge(cost, price, on=['EID', 'MONTH', 'PEAKID'], how='outer').fillna(0)
        copri = copri.rename(columns={'C': 'COST', 'PRICEREALIZED': 'PRICE'})
        copri['PROFIT'] = copri['PRICE'] - copri['COST']

        return copri

    def get_daily_data(self, filterdf, cutoff_date: str) -> pd.DataFrame:  # rows ~ daily sim cols prefixed d_
        """
        Extract daily simulation data respecting cutoff (<=7th of M).
        cutoff_date: 'YYYY-MM-DD' format.
        """
        list_years = pd.to_datetime(filterdf['MONTH']).dt.year.unique().astype(str).tolist()
        chunks = []

        for dd in self.dsimulation:
            fileyear = os.path.basename(dd).split(".")[0][-4:]
            if fileyear not in list_years:
                continue

            # Fix #4: filter DATETIME <= cutoff (8th 00:00 = HE24 of the 7th)
            cutoff_he = (pd.Timestamp(cutoff_date) + pd.DateOffset(days=1)).strftime('%Y-%m-%dT%H:%M:%S')
            temp = duckdb.query(f"""
                SELECT d.*
                FROM read_parquet('{dd}') d
                    INNER JOIN filterdf f
                    ON  strftime(d.DATETIME, '%Y-%m') = f.MONTH
                    AND d.PEAKID  = f.PEAKID
                    AND d.EID     = f.EID
                WHERE d.DATETIME <= '{cutoff_he}'
            """).to_df()

            if not temp.empty:
                chunks.append(temp)

        dailysim = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=self.day_sim_col)

        # Fix #5: prefix only non-key columns
        key_cols = {'SCENARIOID', 'EID', 'DATETIME', 'PEAKID'}
        dailysim = dailysim.rename(columns={c: f"d_{c}" for c in dailysim.columns if c not in key_cols})

        return dailysim

    def get_monthly_data(self, filterdf) -> pd.DataFrame:  # rows ~ monthly sim cols prefixed m_
        """
        Extract monthly simulation data for the months in filterdf.
        Monthly sims for M+1 are produced before the 7th of M, so they are always available.
        """
        list_years = pd.to_datetime(filterdf['MONTH']).dt.year.unique().astype(str).tolist()
        chunks = []

        for mm in self.msimulation:
            fileyear = os.path.basename(mm).split(".")[0][-4:]
            if fileyear not in list_years:
                continue

            temp = duckdb.query(f"""
                SELECT d.*
                FROM read_parquet('{mm}') d
                    INNER JOIN filterdf f
                    ON  strftime(d.DATETIME, '%Y-%m') = f.MONTH
                    AND d.PEAKID  = f.PEAKID
                    AND d.EID     = f.EID
            """).to_df()

            if not temp.empty:
                chunks.append(temp)

        monthlysim = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=self.month_sim_col)

        # Fix #5: prefix only non-key columns
        key_cols = {'SCENARIOID', 'EID', 'DATETIME', 'PEAKID'}
        monthlysim = monthlysim.rename(columns={c: f"m_{c}" for c in monthlysim.columns if c not in key_cols})

        return monthlysim

    def get_daily_price(self, filterdf) -> pd.DataFrame:  # rows ~ PriceRow
        dailyprice = duckdb.query(f"""
            SELECT p.*
            FROM read_parquet('{self.pricefile}') p
                INNER JOIN filterdf f
                ON  strftime(p.DATETIME, '%Y-%m') = f.MONTH
                AND p.PEAKID  = f.PEAKID
                AND p.EID     = f.EID
        """).to_df()
        return dailyprice

    def get_sim_price_profit(self, filterdf, cutoff_date: str) -> tuple:
        """
        Merges daily and monthly simulation data with price, cost, and profit data.
        Input:
            filterdf: df with columns MONTH, PEAKID, EID
            cutoff_date: 'YYYY-MM-DD' format
        Returns:
            tuple of (monthly_info, daily_price, daily_sim, monthly_sim)
        """
        monthly_info = self.get_price_cost_profit(filterdf=filterdf)
        daily_price = self.get_daily_price(filterdf)

        daily_sim = self.get_daily_data(filterdf, cutoff_date)
        monthly_sim = self.get_monthly_data(filterdf)

        return (monthly_info, daily_price, daily_sim, monthly_sim)


def select_best_predict_and_print(resultdf, min_opp=10, max_opp=100):
    """
    This function takes a dataframe containing four columns: EID, MONTH, PEAKID and PREDICTED_PROFIT.
    It removes duplicates based on the triplet (EID, MONTH, PEAKID).
    If more than one row has the same triplet, it keeps the averages of the PREDICTED_PROFIT for those rows.
    Finally, it prints the resulting dataframe.
    :param resultdf:
    :      min_opp: minimum number of opportunities to consider for printing
    :      max_opp: maximum number of opportunities to consider for printing
    :return: none, but prints a
    """

    # Remove duplicates based on the triplet (EID, MONTH, PEAKID) and keep the average of PREDICTED_PROFIT
    resultdf = resultdf.groupby(['EID', 'MONTH', 'PEAKID'], as_index=False)['PREDICTED_PROFIT'].mean()

    #sort the resulting dataframe by PREDICTED_PROFIT in descending order
    resultdf = resultdf.sort_values(by='PREDICTED_PROFIT', ascending=False)

    n_profit_opp = sum(resultdf['PREDICTED_PROFIT'] > 0)

    if n_profit_opp < min_opp:
        #create choice df which contains min_opp rows with the highest PREDICTED_PROFIT
        choice = resultdf.head(min_opp)[['EID', 'MONTH', 'PEAKID']]

    elif n_profit_opp < max_opp:
        choice = resultdf.head(n_profit_opp)[['EID', 'MONTH', 'PEAKID']]
    else:
        choice = resultdf.head(max_opp)[['EID', 'MONTH', 'PEAKID']]

    # Rename columns to the expected output format before writing
    choice = choice.copy()
    choice['PEAKID'] = choice['PEAKID'].map({0: 'OFF', 1: 'ON'})
    choice = choice.rename(columns={'MONTH': 'TARGET_MONTH', 'PEAKID': 'PEAK_TYPE'})
    choice = choice[['TARGET_MONTH', 'PEAK_TYPE', 'EID']]

    #write choice to csv file
    choice.to_csv('opportunities.csv', index=False)
