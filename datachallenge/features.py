"""
Feature Engineering for FTR Opportunity Selection
==================================================
Produces a flat feature matrix (one row per triplet) from simulation,
historical price/cost data, and seasonal encoding.

Anti-leak guarantee
-------------------
All features are derived from data available at the cutoff (7th of month M):
- Monthly sims for M+1 (produced before the 7th of M, always available).
- Daily sims for M up to day 7 (queried with cutoff_date filter in the loader).
- Costs C for M and all prior months (PDF section 5.3: "C de M disponible au cutoff").
- Prices for months strictly before M (C for M excluded from price_cost_profit calls
  to avoid pulling partial hourly prices for M past the 7th).

cost_proxy definition
---------------------
cost_proxy = abs(C) for MONTH = M per (EID, PEAKID).
- C for M+1 is forbidden (PDF section 5.4).
- C for M is explicitly listed as available at the cutoff.
- Fallback for triplets absent from M's cost file (sparse = 0): historical median
  of C for that (EID, PEAKID) over months strictly before M, which is more robust
  to null/outlier costs than the mean (per team note).

Usage
-----
    from datachallenge.features import build_feature_matrix, FEATURE_COLUMNS

    features_df = build_feature_matrix(loader, triplets_df, cutoff_date)
    # columns: EID, MONTH, PEAKID + FEATURE_COLUMNS (33 features)
"""

import math
from typing import Optional

import pandas as pd

from datachallenge.loader import CustomDataLoader

KEY_COLUMNS = ["EID", "MONTH", "PEAKID"]

# 33 feature columns produced by build_feature_matrix.
FEATURE_COLUMNS = [
    # ── Monthly sim (M+1 target month, all 3 scenarios) ────────────────────
    "m_ACT_mean",               # mean ACTIVATIONLEVEL; congestion intensity
    "m_ACT_max",                # max ACTIVATIONLEVEL; captures extreme events
    "m_PSM_mean",               # mean PSM; simulated price direction
    "m_PSM_min",                # min PSM; negative spread extremes
    "m_PSM_std",                # std PSM; simulated price volatility
    "n_hours_active",           # hours with ACTIVATIONLEVEL > 0
    "mean_wind",                # mean WINDIMPACT; renewable-driven congestion
    "mean_solar",               # mean SOLARIMPACT; used in impact_concentration
    "mean_hydro",               # mean HYDROIMPACT; hydro-driven congestion
    "mean_nonrenew",            # mean NONRENEWBALIMPACT; non-renewable imbalance
    "mean_external",            # mean EXTERNALIMPACT; used in impact_concentration
    "mean_transmission_outage", # mean TRANSMISSIONOUTAGEIMPACT; structural congestion
    "m_LOADIMPACT_mean",        # mean LOADIMPACT; network demand
    # Per-scenario PSM means (individual scenario signals)
    "sum_abs_psm_s1",           # mean |PSM| for scenario 1
    "sum_abs_psm_s2",           # mean |PSM| for scenario 2
    "sum_abs_psm_s3",           # mean |PSM| for scenario 3
    # ── Daily sim (cutoff month M, days 1–7, all 3 scenarios) ──────────────
    "d_ACT_mean",               # mean ACTIVATIONLEVEL; recent activation signal
    "daily_sum_abs_psd",        # mean |PSD|; recent simulated price magnitude
    "daily_n_hours_active",     # hours with ACTIVATIONLEVEL > 0 in daily sim
    "daily_mean_trans_outage",  # mean TRANSMISSIONOUTAGEIMPACT in daily sim
    # ── Sim-derived interactions ────────────────────────────────────────────
    "act_psm_interaction",      # m_ACT_max * |m_PSM_mean|; congestion × spread
    "mean_sum_abs_psm",         # mean across scenarios of mean(|PSM|); congestion magnitude
    "psm_std_scenarios",        # std of per-scenario mean PSM; model uncertainty
    # ── Composite (require cost_proxy) ─────────────────────────────────────
    "estimated_profit",         # mean_sum_abs_psm − cost_proxy
    "estimated_profit_pessimistic",  # min per-scenario estimated profit
    "all_scenarios_profitable", # 1 if all 3 per-scenario estimates > 0, else 0
    "confidence_adjusted_profit",    # estimated_profit / (1 + psm_std_scenarios)
    "profit_per_active_hour",   # estimated_profit / max(n_hours_active, 1)
    "impact_concentration",     # max(|src_impacts|) / (sum(|src_impacts|) + 1e-8)
    # ── Cost proxy (standalone feature, also used in composites) ───────────
    "cost_proxy",               # abs(C_M); see module docstring
    # ── Historical + seasonal ───────────────────────────────────────────────
    "hist_profit_rate",         # fraction of months profitable for (EID, PEAKID)
    "month_sin",                # sin(2π * month_of_M+1 / 12)
    "month_cos",                # cos(2π * month_of_M+1 / 12)
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
    Build a feature matrix for FTR opportunity scoring.

    Args:
        loader:       Initialised CustomDataLoader.
        triplets_df:  DataFrame with columns EID, MONTH, PEAKID.
                      MONTH is the target month M+1 (month being predicted).
        cutoff_date:  'YYYY-MM-DD', the 7th of month M (decision cutoff).
        H:            History window in months for historical features. None = all history.

    Returns:
        DataFrame with columns EID, MONTH, PEAKID + FEATURE_COLUMNS (33 features).
        One row per unique input triplet, no duplicates.
        Triplets absent from sims receive 0.0 for sim-derived features.
        Missing hist_profit_rate falls back to the global average.
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

    # ── 3. Historical features: hist_profit_rate + cost_proxy fallback ───────
    hist_rate_df, global_avg, cost_hist_df = _compute_historical_features(
        loader, cutoff_date, cutoff_month, window_start
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
    result = result.merge(hist_rate_df, on=["EID", "PEAKID"], how="left")
    result = result.merge(seasonal_df, on=KEY_COLUMNS, how="left")

    # ── 7. Composite features (computed after merge) ─────────────────────────
    result = _compute_composites(result, ms_scenario_data, cost_proxy_df)

    # ── 8. hist_profit_rate fallback for unseen EIDs ─────────────────────────
    result["hist_profit_rate"] = result["hist_profit_rate"].fillna(global_avg)

    # ── 9. Ensure all feature columns exist and fill NaN → 0 ─────────────────
    for col in FEATURE_COLUMNS:
        if col not in result.columns:
            result[col] = 0.0
        else:
            result[col] = result[col].fillna(0.0)

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
    Aggregate monthly simulation data.

    Returns:
        ms_feats:         DataFrame (KEY_COLUMNS + all monthly sim feature columns).
        ms_scenario_data: DataFrame (KEY_COLUMNS + SCENARIOID + abs_psm_s)
                          for use in pessimistic composite computation.
    """
    empty_feats = triplets[KEY_COLUMNS].copy()
    for col in [
        "m_ACT_mean", "m_ACT_max", "m_PSM_mean", "m_PSM_min", "m_PSM_std",
        "n_hours_active", "mean_wind", "mean_solar", "mean_hydro", "mean_nonrenew",
        "mean_external", "mean_transmission_outage", "m_LOADIMPACT_mean",
        "sum_abs_psm_s1", "sum_abs_psm_s2", "sum_abs_psm_s3",
        "mean_sum_abs_psm", "psm_std_scenarios",
    ]:
        empty_feats[col] = 0.0
    empty_scenario = pd.DataFrame(
        columns=[*KEY_COLUMNS, "SCENARIOID", "abs_psm_s"]
    )

    if monthly_sim.empty:
        return empty_feats, empty_scenario

    ms = monthly_sim.copy()
    ms["MONTH"] = pd.to_datetime(ms["DATETIME"]).dt.to_period("M").astype(str)

    # ── Triplet-level aggregations (all scenarios, all time) ─────────────────
    col_map = {
        "m_ACTIVATIONLEVEL": [("m_ACT_mean", "mean"), ("m_ACT_max", "max")],
        "m_PSM": [("m_PSM_mean", "mean"), ("m_PSM_min", "min"), ("m_PSM_std", "std")],
        "m_WINDIMPACT": [("mean_wind", "mean")],
        "m_SOLARIMPACT": [("mean_solar", "mean")],
        "m_HYDROIMPACT": [("mean_hydro", "mean")],
        "m_NONRENEWBALIMPACT": [("mean_nonrenew", "mean")],
        "m_EXTERNALIMPACT": [("mean_external", "mean")],
        "m_TRANSMISSIONOUTAGEIMPACT": [("mean_transmission_outage", "mean")],
        "m_LOADIMPACT": [("m_LOADIMPACT_mean", "mean")],
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
            elif func == "min":
                triplet_feats[feat_name] = rows_by_triplet[src_col].min()
            elif func == "std":
                triplet_feats[feat_name] = rows_by_triplet[src_col].std()

    # n_hours_active: count hours where ACTIVATIONLEVEL > 0
    if "m_ACTIVATIONLEVEL" in ms.columns:
        triplet_feats["n_hours_active"] = rows_by_triplet["m_ACTIVATIONLEVEL"].apply(
            lambda x: float((x > 0).sum())
        )

    feats_df = pd.DataFrame(triplet_feats).reset_index().fillna(0.0)
    ms_feats = triplets[KEY_COLUMNS].merge(feats_df, on=KEY_COLUMNS, how="left")

    # ── Per-scenario aggregations ────────────────────────────────────────────
    if "m_PSM" not in ms.columns:
        # Fill per-scenario and cross-scenario features with 0
        for col in ["sum_abs_psm_s1", "sum_abs_psm_s2", "sum_abs_psm_s3",
                    "mean_sum_abs_psm", "psm_std_scenarios"]:
            ms_feats[col] = 0.0
        ms_feats = ms_feats.fillna(0.0)
        return ms_feats, empty_scenario

    scenario_agg = (
        ms.groupby([*KEY_COLUMNS, "SCENARIOID"])
        .agg(
            psm_mean_s=("m_PSM", "mean"),
            abs_psm_s=("m_PSM", lambda x: x.abs().mean()),
        )
        .reset_index()
    )

    # Per-scenario |PSM| means pivoted to s1, s2, s3 columns
    for sid in [1, 2, 3]:
        col_name = f"sum_abs_psm_s{sid}"
        s_data = scenario_agg[scenario_agg["SCENARIOID"] == sid][
            [*KEY_COLUMNS, "abs_psm_s"]
        ].rename(columns={"abs_psm_s": col_name})
        ms_feats = ms_feats.merge(s_data, on=KEY_COLUMNS, how="left")

    # Cross-scenario aggregations
    cross = (
        scenario_agg.groupby(KEY_COLUMNS)
        .agg(
            psm_std_scenarios=("psm_mean_s", "std"),
            mean_sum_abs_psm=("abs_psm_s", "mean"),
        )
        .reset_index()
        .fillna(0.0)
    )
    ms_feats = ms_feats.merge(cross, on=KEY_COLUMNS, how="left")
    ms_feats = ms_feats.fillna(0.0)

    ms_scenario_data = scenario_agg[[*KEY_COLUMNS, "SCENARIOID", "abs_psm_s"]]
    return ms_feats, ms_scenario_data


def _agg_daily_sim(daily_sim: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily simulation (M, days 1–7) into per-(EID, PEAKID) features.

    Joins back to triplets on (EID, PEAKID) only — daily sims have MONTH = M
    while triplets have MONTH = M+1.

    Returns:
        DataFrame with columns EID, PEAKID, d_ACT_mean, daily_sum_abs_psd,
        daily_n_hours_active, daily_mean_trans_outage.
    """
    out_cols = ["EID", "PEAKID", "d_ACT_mean", "daily_sum_abs_psd",
                "daily_n_hours_active", "daily_mean_trans_outage"]
    empty = pd.DataFrame(columns=out_cols)

    if daily_sim.empty:
        return empty

    ds = daily_sim.copy()
    groups = ds.groupby(["EID", "PEAKID"])

    result = ds[["EID", "PEAKID"]].drop_duplicates().copy()

    if "d_ACTIVATIONLEVEL" in ds.columns:
        result = result.merge(
            groups["d_ACTIVATIONLEVEL"]
            .agg(
                d_ACT_mean="mean",
                daily_n_hours_active=lambda x: float((x > 0).sum()),
            )
            .reset_index(),
            on=["EID", "PEAKID"],
            how="left",
        )

    if "d_PSD" in ds.columns:
        result = result.merge(
            groups["d_PSD"]
            .apply(lambda x: x.abs().mean())
            .reset_index(name="daily_sum_abs_psd"),
            on=["EID", "PEAKID"],
            how="left",
        )

    if "d_TRANSMISSIONOUTAGEIMPACT" in ds.columns:
        result = result.merge(
            groups["d_TRANSMISSIONOUTAGEIMPACT"]
            .mean()
            .reset_index(name="daily_mean_trans_outage"),
            on=["EID", "PEAKID"],
            how="left",
        )

    for col in ["d_ACT_mean", "daily_sum_abs_psd", "daily_n_hours_active",
                "daily_mean_trans_outage"]:
        if col not in result.columns:
            result[col] = 0.0

    return result[out_cols].fillna(0.0)


def _compute_historical_features(
    loader: CustomDataLoader,
    cutoff_date: str,
    cutoff_month: str,
    window_start: Optional[str],
) -> tuple:
    """
    Compute hist_profit_rate per (EID, PEAKID) and historical median cost.

    A single get_price_cost_profit call is made for all historical months
    (strictly before M), shared between hist_profit_rate and cost fallback.

    Returns:
        (hist_rate_df, global_avg, cost_hist_df) where:
        - hist_rate_df: EID, PEAKID, hist_profit_rate
        - global_avg: fallback rate for unseen EIDs
        - cost_hist_df: EID, PEAKID, cost_proxy_hist (historical median |C|)
    """
    empty_rate = pd.DataFrame(columns=["EID", "PEAKID", "hist_profit_rate"])
    empty_cost = pd.DataFrame(columns=["EID", "PEAKID", "cost_proxy_hist"])

    historical = loader.get_all_triplets(cutoff_date)
    historical = historical[historical["MONTH"] < cutoff_month].copy()
    if window_start is not None:
        historical = historical[historical["MONTH"] >= window_start].copy()

    if historical.empty:
        return empty_rate, 0.0, empty_cost

    profit_df = loader.get_price_cost_profit(historical)

    # hist_profit_rate
    profit_df["profitable"] = (profit_df["PROFIT"] > 0).astype(float)
    global_avg = float(profit_df["profitable"].mean())
    hist_rate = (
        profit_df.groupby(["EID", "PEAKID"])["profitable"]
        .mean()
        .reset_index()
        .rename(columns={"profitable": "hist_profit_rate"})
    )

    # Historical median cost (fallback for absent M cost)
    cost_hist = (
        profit_df.groupby(["EID", "PEAKID"])["COST"]
        .median()
        .abs()
        .reset_index()
        .rename(columns={"COST": "cost_proxy_hist"})
    )

    return hist_rate, global_avg, cost_hist


def _compute_cost_proxy_m(
    loader: CustomDataLoader,
    triplets: pd.DataFrame,
    cutoff_month: str,
    cost_hist_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute cost_proxy = abs(C) for month M per (EID, PEAKID).

    Fallback: for triplets where C_M == 0 (absent from costs.parquet for month M,
    since the file is sparsified), use the historical median cost instead.

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

    # Replace 0 (absent in parquet for month M) with historical median fallback.
    # Use .where to avoid pandas dtype errors when the mask selects no rows.
    if "cost_proxy_hist" in result.columns:
        hist_fallback = result["cost_proxy_hist"].fillna(0.0)
        result["cost_proxy"] = result["cost_proxy"].where(
            result["cost_proxy"] != 0.0, hist_fallback
        )

    return result[["EID", "PEAKID", "cost_proxy"]]


def _compute_seasonal(triplets: pd.DataFrame) -> pd.DataFrame:
    """
    Compute sin/cos seasonal encoding for the target month M+1.

    month_number is drawn from the MONTH column (which equals M+1 in triplets_df).

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
    Compute all composite features that require multiple base features to be
    present in the merged result DataFrame.
    """
    _z = pd.Series(0.0, index=result.index)

    def _col(name: str) -> pd.Series:
        return result[name].fillna(0.0) if name in result.columns else _z.copy()

    cp = _col("cost_proxy")
    msp = _col("mean_sum_abs_psm")
    pss = _col("psm_std_scenarios")
    n_act = _col("n_hours_active")
    m_act_max = _col("m_ACT_max")
    m_psm_mean = _col("m_PSM_mean")

    result["act_psm_interaction"] = m_act_max * m_psm_mean.abs()
    result["estimated_profit"] = msp - cp
    result["confidence_adjusted_profit"] = result["estimated_profit"] / (1.0 + pss)
    result["profit_per_active_hour"] = result["estimated_profit"] / (n_act + 1.0)

    # Pessimistic and all_scenarios_profitable (per-scenario)
    if not ms_scenario_data.empty:
        sc = ms_scenario_data.merge(
            cost_proxy_df[["EID", "PEAKID", "cost_proxy"]], on=["EID", "PEAKID"], how="left"
        )
        sc["cost_proxy"] = sc["cost_proxy"].fillna(0.0)
        sc["scenario_profit"] = sc["abs_psm_s"] - sc["cost_proxy"]

        pess = (
            sc.groupby(KEY_COLUMNS)
            .agg(
                estimated_profit_pessimistic=("scenario_profit", "min"),
                all_scenarios_profitable=(
                    "scenario_profit",
                    lambda x: float((x > 0).all()),
                ),
            )
            .reset_index()
        )
        result = result.merge(pess, on=KEY_COLUMNS, how="left")

    # impact_concentration: max(|src_impacts|) / (sum(|src_impacts|) + 1e-8)
    src_cols = [c for c in _IMPACT_SRC_COLS if c in result.columns]
    if src_cols:
        src = result[src_cols].abs()
        result["impact_concentration"] = src.max(axis=1) / (src.sum(axis=1) + 1e-8)

    return result
