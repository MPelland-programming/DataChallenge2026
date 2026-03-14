"""
Feature Engineering for FTR Opportunity Selection — V3
=======================================================
Produces a flat feature matrix (one row per triplet) used by all scorers.

Anti-leak guarantee
-------------------
All features are derived from data available at the cutoff (7th of month M):
- Monthly sims for M+1 (produced before the 7th of M, always available).
- Daily sims for M up to day 7 (queried with cutoff_date filter in the loader).
- Costs C for M (PDF section 5.3: "C de M disponible au cutoff").
- Prices/costs for months strictly before M for historical features.

cost_proxy definition
---------------------
cost_proxy = abs(C) for MONTH = M per (EID, PEAKID).
- C for M+1 is forbidden (PDF section 5.4).
- C for M is explicitly listed as available at the cutoff.
- Fallback for triplets absent from M's cost file (sparse = 0): historical
  median of abs(C) for that (EID, PEAKID) over months strictly before M.

Formula change vs V1
--------------------
V3 uses |SUM(PSM)| per (EID, PEAKID, SCENARIOID) for sum_abs_psm_s1/s2,
estimated_profit, and estimated_profit_pessimistic.  The old code used
mean(|PSM|), which understates congestion when PSM values partially cancel.

Usage
-----
    from datachallenge.features import build_feature_matrix, FEATURE_COLUMNS

    features_df = build_feature_matrix(loader, triplets_df, cutoff_date)
    # columns: EID, MONTH, PEAKID + FEATURE_COLUMNS (36 features)
"""

import math
from typing import Optional

import numpy as np
import pandas as pd

from datachallenge.loader import CustomDataLoader

KEY_COLUMNS = ["EID", "MONTH", "PEAKID"]

# 36 feature columns produced by build_feature_matrix (V3).
FEATURE_COLUMNS = [
    # ── Estimated profit ────────────────────────────────────────────────────
    "estimated_profit",             # mean_sum_abs_psm − cost_proxy
    "sum_abs_psm_s1",               # |SUM(PSM)| for scenario 1
    "sum_abs_psm_s2",               # |SUM(PSM)| for scenario 2
    "cost_proxy",                   # abs(C_M); see module docstring
    # ── Consensus inter-scénarios ───────────────────────────────────────────
    "psm_cv_scenarios",             # std(|SUM(PSM)|_s) / mean; model uncertainty
    "estimated_profit_pessimistic", # min per-scenario (|SUM(PSM)|_s − cost_proxy)
    # ── Activation + impacts (monthly sim M+1) ──────────────────────────────
    "mean_activation",              # mean ACTIVATIONLEVEL
    "max_activation",               # max ACTIVATIONLEVEL; captures extreme events
    "pct_high_activation",          # proportion of hours with ACTIVATIONLEVEL > 50
    "mean_wind",                    # mean WINDIMPACT
    "mean_solar",                   # mean SOLARIMPACT
    "mean_hydro",                   # mean HYDROIMPACT
    "mean_nonrenew",                # mean NONRENEWBALIMPACT
    "mean_external",                # mean EXTERNALIMPACT
    "mean_transmission_outage",     # mean TRANSMISSIONOUTAGEIMPACT
    "mean_load",                    # mean LOADIMPACT
    "n_hours_active",               # count of hours with PSM ≠ 0 (monthly sim)
    "impact_concentration",         # max(|src_impacts|) / (sum(|src_impacts|) + 1e-8)
    # ── Daily sim signal (cutoff month M, days 1–7) ─────────────────────────
    "daily_sum_abs_psd",            # mean across scenarios of |SUM(PSD)|
    "daily_mean_activation",        # mean ACTIVATIONLEVEL in daily sim
    "daily_max_activation",         # max ACTIVATIONLEVEL in daily sim
    "daily_mean_trans_outage",      # mean TRANSMISSIONOUTAGEIMPACT in daily sim
    "daily_n_hours_active",         # count of hours with PSD ≠ 0 in daily sim
    # ── Historical features ──────────────────────────────────────────────────
    "hist_win_rate",                # fraction of months profitable for (EID, PEAKID)
    "hist_n_months",                # count of months with historical data
    "hist_mean_profit",             # mean historical realised profit
    "hist_std_profit",              # std of historical profit
    "hist_last_6m_win_rate",        # win rate over last 6 months (recency)
    "hist_consecutive_wins",        # current winning streak (most recent months)
    "hist_seasonal_win_rate",       # win rate for same calendar month in prior years
    "hist_mean_cost",               # mean historical cost
    "hist_mean_price",              # mean historical realised price
    # ── Derived ──────────────────────────────────────────────────────────────
    "profit_per_active_hour",       # estimated_profit / max(n_hours_active, 1)
    "monthly_daily_ratio",          # daily_sum_abs_psd / mean_sum_abs_psm
    "month_sin",                    # sin(2π * month_of_M+1 / 12)
    "month_cos",                    # cos(2π * month_of_M+1 / 12)
]

ALL_COLUMNS = KEY_COLUMNS + FEATURE_COLUMNS

# Source impact columns used for impact_concentration
_IMPACT_SRC_COLS = ["mean_wind", "mean_solar", "mean_hydro", "mean_nonrenew", "mean_external"]


def build_feature_matrix(
    loader: CustomDataLoader,
    triplets_df: pd.DataFrame,
    cutoff_date: str,
    H: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build a V3 feature matrix for FTR opportunity scoring.

    Args:
        loader:       Initialised CustomDataLoader.
        triplets_df:  DataFrame with columns EID, MONTH, PEAKID.
                      MONTH is the target month M+1 (month being predicted).
        cutoff_date:  'YYYY-MM-DD', the 7th of month M (decision cutoff).
        H:            History window in months for historical features. None = all history.

    Returns:
        DataFrame with columns EID, MONTH, PEAKID + FEATURE_COLUMNS (36 features).
        One row per unique input triplet, no duplicates.
        Triplets absent from sims receive 0.0 for sim-derived features.
        Missing hist_win_rate falls back to the global average.
        cost_proxy = abs(C_M), fallback to historical median if C_M is absent.
    """
    triplets = triplets_df[KEY_COLUMNS].drop_duplicates().reset_index(drop=True)
    cutoff_month = cutoff_date[:7]  # 'YYYY-MM'
    window_start = _window_start(cutoff_date, H)

    # ── 1. Monthly sim features (M+1 target month) ──────────────────────────
    monthly_sim = loader.get_monthly_data(triplets)
    ms_feats, ms_scenario_data = _agg_monthly_sim(monthly_sim, triplets)

    # ── 2. Daily sim features (M, days 1–7) ─────────────────────────────────
    daily_filter = triplets[["EID", "PEAKID"]].drop_duplicates().copy()
    daily_filter["MONTH"] = cutoff_month
    daily_sim = loader.get_daily_data(daily_filter, cutoff_date)
    ds_feats = _agg_daily_sim(daily_sim)

    # ── 3. Historical features ───────────────────────────────────────────────
    hist_df, global_win_rate, cost_hist_df = _compute_historical_features(
        loader, cutoff_date, cutoff_month, window_start, triplets
    )

    # ── 4. cost_proxy = abs(C_M), fallback to historical median ─────────────
    cost_proxy_df = _compute_cost_proxy_m(loader, triplets, cutoff_month, cost_hist_df)

    # ── 5. Seasonal features ─────────────────────────────────────────────────
    seasonal_df = _compute_seasonal(triplets)

    # ── 6. Assemble result ───────────────────────────────────────────────────
    result = triplets.copy()
    result = result.merge(ms_feats, on=KEY_COLUMNS, how="left")
    result = result.merge(ds_feats, on=["EID", "PEAKID"], how="left")
    result = result.merge(cost_proxy_df, on=["EID", "PEAKID"], how="left")
    result = result.merge(hist_df, on=["EID", "PEAKID"], how="left")
    result = result.merge(seasonal_df, on=KEY_COLUMNS, how="left")

    # ── 7. Composite features ─────────────────────────────────────────────────
    result = _compute_composites(result, ms_scenario_data, cost_proxy_df)

    # ── 8. hist_win_rate fallback for unseen EIDs ─────────────────────────────
    result["hist_win_rate"] = result["hist_win_rate"].fillna(global_win_rate)

    # ── 9. Ensure all feature columns exist, fill NaN → 0, cast to float64 ─────
    # Left-joins with sparse/empty sim data can leave columns with object dtype.
    # LightGBM (and other sklearn estimators) require numeric dtypes; enforce here.
    for col in FEATURE_COLUMNS:
        if col not in result.columns:
            result[col] = 0.0
        else:
            result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0)

    return result[ALL_COLUMNS]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _window_start(cutoff_date: str, H: Optional[int]) -> Optional[str]:
    if H is None:
        return None
    return (pd.Timestamp(cutoff_date) - pd.DateOffset(months=H)).strftime("%Y-%m")


def _agg_monthly_sim(
    monthly_sim: pd.DataFrame,
    triplets: pd.DataFrame,
) -> tuple:
    """
    Aggregate monthly simulation data for V3 features.

    V3 formula: |SUM(PSM)| per (EID, PEAKID, SCENARIOID) for sum_abs_psm_s1/s2.
    n_hours_active = count of hours with PSM ≠ 0 (averaged across scenarios).

    Returns:
        ms_feats:         DataFrame (KEY_COLUMNS + all monthly sim feature columns).
        ms_scenario_data: DataFrame (KEY_COLUMNS + SCENARIOID + sum_abs_psm_s)
                          for use in pessimistic composite computation.
    """
    zero_feats = [
        "mean_activation", "max_activation", "pct_high_activation",
        "mean_wind", "mean_solar", "mean_hydro", "mean_nonrenew",
        "mean_external", "mean_transmission_outage", "mean_load",
        "n_hours_active", "impact_concentration",
        "sum_abs_psm_s1", "sum_abs_psm_s2",
        "psm_cv_scenarios",
        "_mean_sum_abs_psm",  # internal intermediate
    ]
    empty_feats = triplets[KEY_COLUMNS].copy()
    for col in zero_feats:
        empty_feats[col] = 0.0
    empty_scenario = pd.DataFrame(columns=[*KEY_COLUMNS, "SCENARIOID", "sum_abs_psm_s"])

    if monthly_sim.empty:
        return empty_feats, empty_scenario

    ms = monthly_sim.copy()
    ms["MONTH"] = pd.to_datetime(ms["DATETIME"]).dt.to_period("M").astype(str)

    # ── Triplet-level aggregations (all scenarios, all time) ─────────────────
    col_map = {
        "m_ACTIVATIONLEVEL": [
            ("mean_activation", "mean"),
            ("max_activation", "max"),
        ],
        "m_WINDIMPACT": [("mean_wind", "mean")],
        "m_SOLARIMPACT": [("mean_solar", "mean")],
        "m_HYDROIMPACT": [("mean_hydro", "mean")],
        "m_NONRENEWBALIMPACT": [("mean_nonrenew", "mean")],
        "m_EXTERNALIMPACT": [("mean_external", "mean")],
        "m_TRANSMISSIONOUTAGEIMPACT": [("mean_transmission_outage", "mean")],
        "m_LOADIMPACT": [("mean_load", "mean")],
    }

    rows_by_triplet = ms.groupby(KEY_COLUMNS)

    triplet_feats = {}
    for src_col, name_funcs in col_map.items():
        if src_col not in ms.columns:
            continue
        for feat_name, func in name_funcs:
            if func == "mean":
                triplet_feats[feat_name] = rows_by_triplet[src_col].mean()
            elif func == "max":
                triplet_feats[feat_name] = rows_by_triplet[src_col].max()

    # pct_high_activation: proportion of hours with ACTIVATIONLEVEL > 50
    if "m_ACTIVATIONLEVEL" in ms.columns:
        triplet_feats["pct_high_activation"] = rows_by_triplet["m_ACTIVATIONLEVEL"].apply(
            lambda x: float((x > 50).sum()) / max(len(x), 1)
        )

    feats_df = pd.DataFrame(triplet_feats).reset_index().fillna(0.0)
    ms_feats = triplets[KEY_COLUMNS].merge(feats_df, on=KEY_COLUMNS, how="left")

    # impact_concentration from triplet-level impact means
    src_cols_present = [c for c in _IMPACT_SRC_COLS if c in ms_feats.columns]
    if src_cols_present:
        src = ms_feats[src_cols_present].abs()
        ms_feats["impact_concentration"] = src.max(axis=1) / (src.sum(axis=1) + 1e-8)
    else:
        ms_feats["impact_concentration"] = 0.0

    # ── Per-scenario aggregations (V3 formula: |SUM(PSM)|) ──────────────────
    if "m_PSM" not in ms.columns:
        for col in ["sum_abs_psm_s1", "sum_abs_psm_s2", "psm_cv_scenarios",
                    "n_hours_active", "_mean_sum_abs_psm"]:
            ms_feats[col] = 0.0
        ms_feats = ms_feats.fillna(0.0)
        return ms_feats, empty_scenario

    scenario_agg = (
        ms.groupby([*KEY_COLUMNS, "SCENARIOID"])
        .agg(
            sum_abs_psm_s=("m_PSM", lambda x: abs(x.sum())),
            n_hours_active_s=("m_PSM", lambda x: float((x != 0).sum())),
        )
        .reset_index()
    )

    # Per-scenario |SUM(PSM)| pivoted to s1, s2 columns
    for sid in [1, 2]:
        col_name = f"sum_abs_psm_s{sid}"
        s_data = scenario_agg[scenario_agg["SCENARIOID"] == sid][
            [*KEY_COLUMNS, "sum_abs_psm_s"]
        ].rename(columns={"sum_abs_psm_s": col_name})
        ms_feats = ms_feats.merge(s_data, on=KEY_COLUMNS, how="left")

    # Cross-scenario aggregations (using all 3 scenarios for cv and mean)
    cross = (
        scenario_agg.groupby(KEY_COLUMNS)
        .agg(
            _mean_sum_abs_psm=("sum_abs_psm_s", "mean"),
            _std_sum_abs_psm=("sum_abs_psm_s", "std"),
            n_hours_active=("n_hours_active_s", "mean"),
        )
        .reset_index()
        .fillna(0.0)
    )
    cross["psm_cv_scenarios"] = cross["_std_sum_abs_psm"] / (
        cross["_mean_sum_abs_psm"] + 1e-8
    )
    ms_feats = ms_feats.merge(
        cross[[*KEY_COLUMNS, "_mean_sum_abs_psm", "psm_cv_scenarios", "n_hours_active"]],
        on=KEY_COLUMNS, how="left",
    )
    ms_feats = ms_feats.fillna(0.0)

    ms_scenario_data = scenario_agg[[*KEY_COLUMNS, "SCENARIOID", "sum_abs_psm_s"]]
    return ms_feats, ms_scenario_data


def _agg_daily_sim(daily_sim: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily simulation (M, days 1–7) into per-(EID, PEAKID) features.

    V3 formulas:
    - daily_sum_abs_psd = mean across scenarios of |SUM(PSD)|
    - daily_n_hours_active = count of hours with PSD ≠ 0 (averaged across scenarios)
    - daily_max_activation = max ACTIVATIONLEVEL (new in V3)

    Returns:
        DataFrame with columns EID, PEAKID + daily feature columns.
    """
    out_cols = [
        "EID", "PEAKID",
        "daily_sum_abs_psd", "daily_mean_activation", "daily_max_activation",
        "daily_mean_trans_outage", "daily_n_hours_active",
    ]
    empty = pd.DataFrame(columns=out_cols)

    if daily_sim.empty:
        return empty

    ds = daily_sim.copy()

    # Per-(EID, PEAKID, SCENARIOID) aggregations
    group_keys = ["EID", "PEAKID", "SCENARIOID"]
    if "SCENARIOID" not in ds.columns:
        ds["SCENARIOID"] = 1

    agg_dict = {}
    if "d_PSD" in ds.columns:
        agg_dict["sum_abs_psd"] = ("d_PSD", lambda x: abs(x.sum()))
        agg_dict["daily_n_hours_active_s"] = ("d_PSD", lambda x: float((x != 0).sum()))
    if "d_ACTIVATIONLEVEL" in ds.columns:
        agg_dict["daily_mean_activation_s"] = ("d_ACTIVATIONLEVEL", "mean")
        agg_dict["daily_max_activation_s"] = ("d_ACTIVATIONLEVEL", "max")
    if "d_TRANSMISSIONOUTAGEIMPACT" in ds.columns:
        agg_dict["daily_mean_trans_outage_s"] = ("d_TRANSMISSIONOUTAGEIMPACT", "mean")

    if not agg_dict:
        return empty

    per_scenario = (
        ds.groupby(group_keys)
        .agg(**agg_dict)
        .reset_index()
    )

    # Average across scenarios
    per_eid_peakid = per_scenario.groupby(["EID", "PEAKID"])
    result = ds[["EID", "PEAKID"]].drop_duplicates().copy()

    if "sum_abs_psd" in per_scenario.columns:
        result = result.merge(
            per_eid_peakid["sum_abs_psd"].mean().reset_index().rename(
                columns={"sum_abs_psd": "daily_sum_abs_psd"}
            ),
            on=["EID", "PEAKID"], how="left",
        )
    if "daily_n_hours_active_s" in per_scenario.columns:
        result = result.merge(
            per_eid_peakid["daily_n_hours_active_s"].mean().reset_index().rename(
                columns={"daily_n_hours_active_s": "daily_n_hours_active"}
            ),
            on=["EID", "PEAKID"], how="left",
        )
    if "daily_mean_activation_s" in per_scenario.columns:
        result = result.merge(
            per_eid_peakid["daily_mean_activation_s"].mean().reset_index().rename(
                columns={"daily_mean_activation_s": "daily_mean_activation"}
            ),
            on=["EID", "PEAKID"], how="left",
        )
    if "daily_max_activation_s" in per_scenario.columns:
        result = result.merge(
            per_eid_peakid["daily_max_activation_s"].mean().reset_index().rename(
                columns={"daily_max_activation_s": "daily_max_activation"}
            ),
            on=["EID", "PEAKID"], how="left",
        )
    if "daily_mean_trans_outage_s" in per_scenario.columns:
        result = result.merge(
            per_eid_peakid["daily_mean_trans_outage_s"].mean().reset_index().rename(
                columns={"daily_mean_trans_outage_s": "daily_mean_trans_outage"}
            ),
            on=["EID", "PEAKID"], how="left",
        )

    for col in ["daily_sum_abs_psd", "daily_mean_activation", "daily_max_activation",
                "daily_n_hours_active", "daily_mean_trans_outage"]:
        if col not in result.columns:
            result[col] = 0.0

    return result[out_cols].fillna(0.0)


def _compute_historical_features(
    loader: CustomDataLoader,
    cutoff_date: str,
    cutoff_month: str,
    window_start: Optional[str],
    triplets: pd.DataFrame,
) -> tuple:
    """
    Compute all 9 historical features per (EID, PEAKID) and historical median cost.

    Uses months strictly before M to avoid partial-month price data (anti-leak).

    Returns:
        (hist_df, global_win_rate, cost_hist_df) where:
        - hist_df: EID, PEAKID + all 9 hist_ features
        - global_win_rate: fallback win rate for unseen EIDs
        - cost_hist_df: EID, PEAKID, cost_proxy_hist (historical median |C|)
    """
    empty_hist = pd.DataFrame(columns=[
        "EID", "PEAKID",
        "hist_win_rate", "hist_n_months", "hist_mean_profit", "hist_std_profit",
        "hist_last_6m_win_rate", "hist_consecutive_wins",
        "hist_seasonal_win_rate", "hist_mean_cost", "hist_mean_price",
    ])
    empty_cost = pd.DataFrame(columns=["EID", "PEAKID", "cost_proxy_hist"])

    # target_month = M+1 = first month in triplets
    # We need calendar month of M+1 for seasonal win rate
    target_month_num = None
    if not triplets.empty:
        sample_month = triplets["MONTH"].iloc[0]  # M+1
        try:
            target_month_num = pd.Timestamp(sample_month + "-01").month
        except Exception:
            pass

    historical = loader.get_all_triplets(cutoff_date)
    historical = historical[historical["MONTH"] < cutoff_month].copy()
    if window_start is not None:
        historical = historical[historical["MONTH"] >= window_start].copy()

    if historical.empty:
        return empty_hist, 0.0, empty_cost

    profit_df = loader.get_price_cost_profit(historical)
    if profit_df.empty:
        return empty_hist, 0.0, empty_cost

    profit_df = profit_df.copy()
    profit_df["profitable"] = (profit_df["PROFIT"] > 0).astype(int)

    global_win_rate = float(profit_df["profitable"].mean())

    # ── Base aggregates ──────────────────────────────────────────────────────
    agg = (
        profit_df.groupby(["EID", "PEAKID"])
        .agg(
            hist_win_rate=("profitable", "mean"),
            hist_n_months=("profitable", "count"),
            hist_mean_profit=("PROFIT", "mean"),
            hist_std_profit=("PROFIT", "std"),
            hist_mean_cost=("COST", "mean"),
            hist_mean_price=("PRICE", "mean"),
        )
        .reset_index()
    )
    agg["hist_std_profit"] = agg["hist_std_profit"].fillna(0.0)

    # ── Recency: last 6 months ────────────────────────────────────────────────
    all_months_sorted = sorted(profit_df["MONTH"].unique())
    last_6 = all_months_sorted[-6:] if len(all_months_sorted) >= 6 else all_months_sorted

    rec_6 = (
        profit_df[profit_df["MONTH"].isin(last_6)]
        .groupby(["EID", "PEAKID"])["profitable"]
        .mean()
        .reset_index()
        .rename(columns={"profitable": "hist_last_6m_win_rate"})
    )

    # ── Consecutive wins (streak ending at latest month) ──────────────────────
    def _get_streak(group: pd.DataFrame) -> int:
        sorted_g = group.sort_values("MONTH", ascending=False)
        streak = 0
        for _, row in sorted_g.iterrows():
            if row["profitable"] == 1:
                streak += 1
            else:
                break
        return streak

    streaks = (
        profit_df.groupby(["EID", "PEAKID"])
        .apply(_get_streak)
        .reset_index()
    )
    streaks.columns = ["EID", "PEAKID", "hist_consecutive_wins"]

    # ── Seasonal: same calendar month in prior years ──────────────────────────
    if target_month_num is not None:
        seasonal_mask = pd.to_datetime(profit_df["MONTH"] + "-01").dt.month == target_month_num
        seasonal_data = profit_df[seasonal_mask]
    else:
        seasonal_data = pd.DataFrame()

    if not seasonal_data.empty:
        seasonal = (
            seasonal_data.groupby(["EID", "PEAKID"])["profitable"]
            .mean()
            .reset_index()
            .rename(columns={"profitable": "hist_seasonal_win_rate"})
        )
    else:
        seasonal = pd.DataFrame(columns=["EID", "PEAKID", "hist_seasonal_win_rate"])

    # ── Assemble ──────────────────────────────────────────────────────────────
    hist_df = agg.merge(rec_6, on=["EID", "PEAKID"], how="left")
    hist_df = hist_df.merge(streaks, on=["EID", "PEAKID"], how="left")
    hist_df = hist_df.merge(seasonal, on=["EID", "PEAKID"], how="left")

    for col in ["hist_last_6m_win_rate", "hist_consecutive_wins", "hist_seasonal_win_rate"]:
        hist_df[col] = hist_df[col].fillna(0.0)

    # ── Historical median cost (fallback for absent M cost) ────────────────────
    cost_hist = (
        profit_df.groupby(["EID", "PEAKID"])["COST"]
        .median()
        .abs()
        .reset_index()
        .rename(columns={"COST": "cost_proxy_hist"})
    )

    return hist_df, global_win_rate, cost_hist


def _compute_cost_proxy_m(
    loader: CustomDataLoader,
    triplets: pd.DataFrame,
    cutoff_month: str,
    cost_hist_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute cost_proxy = abs(C) for month M per (EID, PEAKID).

    Fallback: for triplets where C_M == 0 (absent from costs.parquet for month M),
    use the historical median cost instead.

    Returns:
        DataFrame with columns EID, PEAKID, cost_proxy.
    """
    cost_filter = triplets[["EID", "PEAKID"]].drop_duplicates().copy()
    cost_filter["MONTH"] = cutoff_month

    profit_m = loader.get_price_cost_profit(cost_filter)
    cost_m = (
        profit_m.groupby(["EID", "PEAKID"])["COST"]
        .first()
        .abs()
        .reset_index()
        .rename(columns={"COST": "cost_proxy"})
    )

    # Merge historical fallback
    result = cost_m.merge(cost_hist_df, on=["EID", "PEAKID"], how="left")

    if "cost_proxy_hist" in result.columns:
        hist_fallback = result["cost_proxy_hist"].fillna(0.0)
        result["cost_proxy"] = result["cost_proxy"].where(
            result["cost_proxy"] != 0.0, hist_fallback
        )

    return result[["EID", "PEAKID", "cost_proxy"]]


def _compute_seasonal(triplets: pd.DataFrame) -> pd.DataFrame:
    """
    Compute sin/cos seasonal encoding for the target month M+1.

    Returns:
        DataFrame with KEY_COLUMNS + month_sin + month_cos.
    """
    result = triplets[KEY_COLUMNS].copy()
    month_num = pd.to_datetime(result["MONTH"] + "-01").dt.month
    result["month_sin"] = (2 * math.pi * month_num / 12).apply(math.sin)
    result["month_cos"] = (2 * math.pi * month_num / 12).apply(math.cos)
    return result


def _compute_composites(
    result: pd.DataFrame,
    ms_scenario_data: pd.DataFrame,
    cost_proxy_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute composite features that depend on multiple base features.

    V3 composites:
    - estimated_profit = _mean_sum_abs_psm - cost_proxy
    - estimated_profit_pessimistic = min(sum_abs_psm_si - cost_proxy) across scenarios
    - profit_per_active_hour = estimated_profit / max(n_hours_active, 1)
    - monthly_daily_ratio = daily_sum_abs_psd / mean_sum_abs_psm (0 if denom = 0)
    """
    def _col(name: str) -> pd.Series:
        return result[name].fillna(0.0) if name in result.columns else pd.Series(
            0.0, index=result.index
        )

    cp = _col("cost_proxy")
    msp = _col("_mean_sum_abs_psm")  # internal intermediate
    n_act = _col("n_hours_active")
    daily_psd = _col("daily_sum_abs_psd")

    result["estimated_profit"] = msp - cp
    result["profit_per_active_hour"] = result["estimated_profit"] / (n_act.clip(lower=1))
    # Avoid division by zero: use safe division (divide only where msp > 0)
    msp_safe = msp.where(msp > 0, other=1.0)  # replace 0 with 1 to avoid ZeroDivisionError
    result["monthly_daily_ratio"] = np.where(msp > 0, daily_psd / msp_safe, 0.0)

    # estimated_profit_pessimistic = min per-scenario (sum_abs_psm_si - cost_proxy)
    if not ms_scenario_data.empty:
        sc = ms_scenario_data.merge(
            cost_proxy_df[["EID", "PEAKID", "cost_proxy"]], on=["EID", "PEAKID"], how="left"
        )
        sc["cost_proxy"] = sc["cost_proxy"].fillna(0.0)
        sc["scenario_profit"] = sc["sum_abs_psm_s"] - sc["cost_proxy"]

        pess = (
            sc.groupby(KEY_COLUMNS)["scenario_profit"]
            .min()
            .reset_index()
            .rename(columns={"scenario_profit": "estimated_profit_pessimistic"})
        )
        result = result.merge(pess, on=KEY_COLUMNS, how="left")

    return result
