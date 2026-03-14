"""
FTR Opportunity Scorers
=======================
Each scorer takes (loader, candidates, cutoff_date) and returns a DataFrame
with columns: EID, MONTH, PEAKID, PREDICTED_PROFIT.

PREDICTED_PROFIT is used only as a ranking score — it does not need to be a
literal profit estimate, just a value that ranks candidates correctly.
"""

from typing import Protocol

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LassoCV, LogisticRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from datachallenge.features import FEATURE_COLUMNS, build_feature_matrix
from datachallenge.loader import CustomDataLoader


class ScorerProtocol(Protocol):
    """Type-checkable interface for all scorer functions."""

    def __call__(
        self,
        loader: CustomDataLoader,
        candidates: pd.DataFrame,
        cutoff_date: str,
    ) -> pd.DataFrame:
        """
        Score candidates for the target month M+1.

        Args:
            loader:       Initialised CustomDataLoader.
            candidates:   DataFrame with columns EID, MONTH, PEAKID.
                          All rows should be for the same target month M+1.
            cutoff_date:  'YYYY-MM-DD', the 7th of month M.

        Returns:
            DataFrame with columns EID, MONTH, PEAKID, PREDICTED_PROFIT.
        """
        ...


def score_by_activation_level(
    loader: CustomDataLoader,
    candidates: pd.DataFrame,
    cutoff_date: str,
) -> pd.DataFrame:
    """
    Reference baseline: rank candidates by mean ACTIVATIONLEVEL in monthly sims.

    !! ANTI-PREDICTIVE — KEPT FOR COMPARISON ONLY !!
    H1 was refuted: higher ACTIVATIONLEVEL is *negatively* associated with
    profitability (AUC = 0.424, Spearman r = −0.120). The threshold found on
    2020–2022 (42.46%) selects zero triplets in 2023. Do not use this scorer
    as a primary signal.

    ACTIVATIONLEVEL is on a **0–100 percent scale** (confirmed by threshold
    analysis: best threshold ≈ 42, not ≈ 0.42).

    Triplets absent from the monthly sims (implicit zero) receive score 0.

    Args:
        loader:       Initialised CustomDataLoader instance.
        candidates:   DataFrame with columns EID, MONTH, PEAKID.
                      All rows should be for the same target month M+1.
        cutoff_date:  'YYYY-MM-DD', the 7th of month M. Not used to filter
                      monthly sims (those are for M+1 and always available);
                      included for interface consistency with ScorerProtocol.

    Returns:
        DataFrame with columns EID, MONTH, PEAKID, PREDICTED_PROFIT.
        PREDICTED_PROFIT = mean m_ACTIVATIONLEVEL (used as a ranking score).
    """
    monthly_sim = loader.get_monthly_data(candidates)

    if monthly_sim.empty:
        result = candidates[['EID', 'MONTH', 'PEAKID']].copy()
        result['PREDICTED_PROFIT'] = 0.0
        return result

    monthly_sim = monthly_sim.copy()
    monthly_sim['MONTH'] = pd.to_datetime(monthly_sim['DATETIME']).dt.to_period('M').astype(str)

    scored = (
        monthly_sim
        .groupby(['EID', 'MONTH', 'PEAKID'], as_index=False)['m_ACTIVATIONLEVEL']
        .mean()
        .rename(columns={'m_ACTIVATIONLEVEL': 'PREDICTED_PROFIT'})
    )

    result = candidates[['EID', 'MONTH', 'PEAKID']].merge(
        scored, on=['EID', 'MONTH', 'PEAKID'], how='left'
    )
    result['PREDICTED_PROFIT'] = result['PREDICTED_PROFIT'].fillna(0.0)

    return result[['EID', 'MONTH', 'PEAKID', 'PREDICTED_PROFIT']]


def score_by_maxime_short(
    loader: CustomDataLoader,
    candidates: pd.DataFrame,
    cutoff_date: str,
) -> pd.DataFrame:
    """
    Two-head supervised scorer: LogisticRegression (P = prob. of profit)
    × Ridge regression (V = predicted profit value). Final score = P × V.

    Walk-forward design: all historical triplets with MONTH strictly before
    the cutoff month M are used as training data.  A 1-month embargo is
    applied implicitly: the feature window for training rows ends at M−1,
    so no data from M (or M+1) leaks into the training features.

    Training set: observed (EID, MONTH, PEAKID) triplets with MONTH < M.
    Labels: CY = (PROFIT > 0) for logistic head; Y = PROFIT for ridge head.
    Ridge sample weights: profitable rows are weighted 2× non-profitable.

    Fallback: if fewer than 2 distinct classes exist in the training labels
    (e.g. first month with no history), all candidates receive score 0.

    Args:
        loader:       Initialised CustomDataLoader.
        candidates:   DataFrame with columns EID, MONTH, PEAKID.
        cutoff_date:  'YYYY-MM-DD', the 7th of month M.

    Returns:
        DataFrame with columns EID, MONTH, PEAKID, PREDICTED_PROFIT.
        PREDICTED_PROFIT = P × V (logistic probability × ridge prediction).
    """
    cutoff_month = cutoff_date[:7]  # 'YYYY-MM'

    # ── 1. Gather training triplets (MONTH < M) ───────────────────────────────
    all_triplets = loader.get_all_triplets(cutoff_date)
    train_triplets = all_triplets[all_triplets["MONTH"] < cutoff_month].copy()

    result_empty = candidates[["EID", "MONTH", "PEAKID"]].copy()
    result_empty["PREDICTED_PROFIT"] = 0.0

    if train_triplets.empty:
        return result_empty

    # ── 2. Labels for training triplets ──────────────────────────────────────
    profit_df = loader.get_price_cost_profit(train_triplets)
    if profit_df.empty:
        return result_empty

    profit_df = profit_df[["EID", "MONTH", "PEAKID", "PROFIT"]].drop_duplicates(
        subset=["EID", "MONTH", "PEAKID"]
    )

    # ── 3. Build feature matrix for training triplets ─────────────────────────
    # Each training row has MONTH = M_hist (historical target months).
    # build_feature_matrix treats MONTH as the target M+1, so we pass the
    # historical month as "target month" and supply a matching cutoff that
    # ends the day-7 of the month before — i.e. we use cutoff_date for the
    # current call only; the feature matrix will use the actual cutoff data
    # available at the *current* cutoff (conservative but correct: all sim
    # data used is available at the current cutoff).
    train_features = build_feature_matrix(loader, train_triplets, cutoff_date)

    # Merge labels into features
    train_df = train_features.merge(
        profit_df, on=["EID", "MONTH", "PEAKID"], how="inner"
    )
    if train_df.empty or train_df["PROFIT"].isna().all():
        return result_empty

    X_train_raw = train_df[FEATURE_COLUMNS].values
    y_profit = train_df["PROFIT"].values
    y_class = (y_profit > 0).astype(int)

    # Need at least 2 classes for logistic regression
    if len(np.unique(y_class)) < 2:
        return result_empty

    # ── 4. Scale features (fit on train only) ────────────────────────────────
    # Features span very different scales: e.g. n_hours_active can be in the
    # thousands while month_sin is in [-1, 1]. Without scaling, lbfgs fails
    # to converge and Ridge coefficients are dominated by high-magnitude
    # features. Scaler is fit on training data only — never on candidates —
    # to avoid any look-ahead into the test distribution.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)

    # ── 5. Train heads ────────────────────────────────────────────────────────
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_train, y_class)

    # Ridge: weight profitable rows 2× non-profitable
    sample_weights = np.where(y_class == 1, 2.0, 1.0)
    ridge = Ridge()
    ridge.fit(X_train, y_profit, sample_weight=sample_weights)

    # ── 6. Build feature matrix for candidates and predict ────────────────────
    cand_features = build_feature_matrix(loader, candidates, cutoff_date)
    result = candidates[["EID", "MONTH", "PEAKID"]].merge(
        cand_features[["EID", "MONTH", "PEAKID"] + FEATURE_COLUMNS],
        on=["EID", "MONTH", "PEAKID"],
        how="left",
    )

    X_cand = scaler.transform(result[FEATURE_COLUMNS].fillna(0.0).values)
    P = lr.predict_proba(X_cand)[:, 1]  # probability of profit
    V = ridge.predict(X_cand)           # predicted profit value

    result["PREDICTED_PROFIT"] = P * V
    return result[["EID", "MONTH", "PEAKID", "PREDICTED_PROFIT"]]


def score_by_historical_profit_rate(
    loader: CustomDataLoader,
    candidates: pd.DataFrame,
    cutoff_date: str,
) -> pd.DataFrame:
    """
    New baseline: rank candidates by historical profitability rate per (EID, PEAKID).

    Score = fraction of months where PROFIT > 0, computed over all months
    strictly before the cutoff month (M-1 and earlier). This captures H4
    ("chronic winners"): 68.9% of (EID, PEAKID) pairs with ≥6 months of
    history have a win rate above 50%.

    Anti-leak guarantee: only months strictly before M (cutoff_month) are used.
    Month M itself is excluded because its realized prices are not yet complete
    at the 7th — get_price_cost_profit with a filterdf containing month M would
    fetch all of M's realized prices (the full month, past day 7), violating the
    anti-leak rule. Restricting filterdf to MONTH < cutoff_month avoids this.

    Fallback for EIDs with no history (new elements): global average profit
    rate across all known pairs in the same historical window.

    Args:
        loader:       Initialised CustomDataLoader.
        candidates:   DataFrame with columns EID, MONTH, PEAKID.
                      All rows should be for the same target month M+1.
        cutoff_date:  'YYYY-MM-DD', the 7th of month M.

    Returns:
        DataFrame with columns EID, MONTH, PEAKID, PREDICTED_PROFIT.
        PREDICTED_PROFIT = historical win rate in [0.0, 1.0].
    """
    cutoff_month = cutoff_date[:7]  # 'YYYY-MM'

    # Triplets visible at cutoff, restricted to months strictly before M.
    # Excluding M avoids partial-month price data and any downstream leak:
    # get_price_cost_profit joins on MONTH and would pull the full month's
    # realized prices even though we only have 7 days at decision time.
    historical = loader.get_all_triplets(cutoff_date)
    historical = historical[historical['MONTH'] < cutoff_month].copy()

    if historical.empty:
        result = candidates[['EID', 'MONTH', 'PEAKID']].copy()
        result['PREDICTED_PROFIT'] = 0.0
        return result

    # get_price_cost_profit with this filterdf is safe: all months are < M,
    # so no realized prices from M or M+1 are fetched.
    profit_df = loader.get_price_cost_profit(historical)
    profit_df['profitable'] = (profit_df['PROFIT'] > 0).astype(float)

    # Win rate per (EID, PEAKID) — collapse over historical months
    win_rate = (
        profit_df
        .groupby(['EID', 'PEAKID'], as_index=False)['profitable']
        .mean()
        .rename(columns={'profitable': 'PREDICTED_PROFIT'})
    )

    # Global average as fallback for EIDs with no history
    global_avg = float(profit_df['profitable'].mean()) if not profit_df.empty else 0.0

    # Join on (EID, PEAKID) only — MONTH in candidates is M+1, not in history
    result = candidates[['EID', 'MONTH', 'PEAKID']].merge(
        win_rate, on=['EID', 'PEAKID'], how='left'
    )
    result['PREDICTED_PROFIT'] = result['PREDICTED_PROFIT'].fillna(global_avg)

    return result[['EID', 'MONTH', 'PEAKID', 'PREDICTED_PROFIT']]


def score_by_lasso(
    loader: CustomDataLoader,
    candidates: pd.DataFrame,
    cutoff_date: str,
) -> pd.DataFrame:
    """
    Lasso regression scorer with walk-forward alpha selection via LassoCV.

    Predicts PROFIT directly (regression only). LassoCV uses TimeSeriesSplit
    cross-validation to select the regularisation parameter alpha from the
    training data available at each cutoff; alpha is re-selected at every
    cutoff so the model adapts as more history becomes available.

    Implicit feature selection: Lasso drives irrelevant feature coefficients
    to zero, which is useful for jury presentation and interpretability.

    Walk-forward design: all historical triplets with MONTH strictly before
    the cutoff month M are used as training data.  Same anti-leak / embargo
    rule as score_by_maxime_short.

    Fallback: if there are fewer than 2 training rows, all candidates receive
    score 0.

    Args:
        loader:       Initialised CustomDataLoader.
        candidates:   DataFrame with columns EID, MONTH, PEAKID.
        cutoff_date:  'YYYY-MM-DD', the 7th of month M.

    Returns:
        DataFrame with columns EID, MONTH, PEAKID, PREDICTED_PROFIT.
        PREDICTED_PROFIT = lasso.predict(X_candidates).
    """
    cutoff_month = cutoff_date[:7]  # 'YYYY-MM'

    # ── 1. Gather training triplets (MONTH < M) ───────────────────────────────
    all_triplets = loader.get_all_triplets(cutoff_date)
    train_triplets = all_triplets[all_triplets["MONTH"] < cutoff_month].copy()

    result_empty = candidates[["EID", "MONTH", "PEAKID"]].copy()
    result_empty["PREDICTED_PROFIT"] = 0.0

    if train_triplets.empty:
        return result_empty

    # ── 2. Labels for training triplets ──────────────────────────────────────
    profit_df = loader.get_price_cost_profit(train_triplets)
    if profit_df.empty:
        return result_empty

    profit_df = profit_df[["EID", "MONTH", "PEAKID", "PROFIT"]].drop_duplicates(
        subset=["EID", "MONTH", "PEAKID"]
    )

    # ── 3. Build feature matrix for training triplets ─────────────────────────
    train_features = build_feature_matrix(loader, train_triplets, cutoff_date)

    train_df = train_features.merge(
        profit_df, on=["EID", "MONTH", "PEAKID"], how="inner"
    )
    if train_df.empty or train_df["PROFIT"].isna().all():
        return result_empty

    if len(train_df) < 2:
        return result_empty

    X_train_raw = train_df[FEATURE_COLUMNS].values
    y_profit = train_df["PROFIT"].values

    # ── 4. Scale features (fit on train only) ────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)

    # ── 5. Fit LassoCV — alpha selected via time-series cross-validation ──────
    # n_splits capped at min(5, n_samples-1) to avoid errors on small datasets.
    n_splits = min(5, len(X_train) - 1)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    lasso = LassoCV(cv=tscv, max_iter=50000, random_state=42)
    lasso.fit(X_train, y_profit)

    # ── 6. Build feature matrix for candidates and predict ────────────────────
    cand_features = build_feature_matrix(loader, candidates, cutoff_date)
    result = candidates[["EID", "MONTH", "PEAKID"]].merge(
        cand_features[["EID", "MONTH", "PEAKID"] + FEATURE_COLUMNS],
        on=["EID", "MONTH", "PEAKID"],
        how="left",
    )

    X_cand = scaler.transform(result[FEATURE_COLUMNS].fillna(0.0).values)
    result["PREDICTED_PROFIT"] = lasso.predict(X_cand)

    return result[["EID", "MONTH", "PEAKID", "PREDICTED_PROFIT"]]


def score_by_lightgbm(
    loader: CustomDataLoader,
    candidates: pd.DataFrame,
    cutoff_date: str,
) -> pd.DataFrame:
    """
    LightGBM classifier scorer with walk-forward training.

    Trains a gradient boosting classifier on historical (EID, MONTH, PEAKID)
    triplets with MONTH strictly before the cutoff month M.  The binary label
    is (PROFIT > 0) and the predicted probability of profit is used as the
    ranking score (PREDICTED_PROFIT).

    LightGBM is preferred over Lasso for the 170k-candidate sim universe:
    it trains in seconds on large datasets (histogram-based splits), handles
    non-linear feature interactions automatically, and requires no StandardScaler.

    Walk-forward design and embargo rule are identical to other scorers:
    - Training data: all historical triplets with MONTH < cutoff_month.
    - No data from month M or M+1 is used in training or feature extraction.

    Fallback: if fewer than min_child_samples (100) training rows are available,
    all candidates receive PREDICTED_PROFIT = 0.0 (not enough data to fit).

    Args:
        loader:       Initialised CustomDataLoader.
        candidates:   DataFrame with columns EID, MONTH, PEAKID.
        cutoff_date:  'YYYY-MM-DD', the 7th of month M.

    Returns:
        DataFrame with columns EID, MONTH, PEAKID, PREDICTED_PROFIT.
        PREDICTED_PROFIT = predict_proba[:, 1] (probability of being profitable).
    """
    MIN_CHILD_SAMPLES = 100
    cutoff_month = cutoff_date[:7]  # 'YYYY-MM'

    # ── 1. Gather training triplets (MONTH < M) ───────────────────────────────
    all_triplets = loader.get_all_triplets(cutoff_date)
    train_triplets = all_triplets[all_triplets["MONTH"] < cutoff_month].copy()

    result_empty = candidates[["EID", "MONTH", "PEAKID"]].copy()
    result_empty["PREDICTED_PROFIT"] = 0.0

    if train_triplets.empty:
        return result_empty

    # ── 2. Labels for training triplets ──────────────────────────────────────
    profit_df = loader.get_price_cost_profit(train_triplets)
    if profit_df.empty:
        return result_empty

    profit_df = profit_df[["EID", "MONTH", "PEAKID", "PROFIT"]].drop_duplicates(
        subset=["EID", "MONTH", "PEAKID"]
    )

    # ── 3. Build feature matrix for training triplets ─────────────────────────
    train_features = build_feature_matrix(loader, train_triplets, cutoff_date)

    train_df = train_features.merge(
        profit_df, on=["EID", "MONTH", "PEAKID"], how="inner"
    )
    if train_df.empty or train_df["PROFIT"].isna().all():
        return result_empty

    if len(train_df) < MIN_CHILD_SAMPLES:
        return result_empty

    X_train = train_df[FEATURE_COLUMNS].values
    y_class = (train_df["PROFIT"].values > 0).astype(int)

    # Need at least 2 classes to train a classifier
    if len(np.unique(y_class)) < 2:
        return result_empty

    # ── 4. Train LightGBM classifier ──────────────────────────────────────────
    # Hyperparams from coworkers (optimised for this dataset).
    # No StandardScaler needed — tree splits are rank-based.
    model = LGBMClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_samples=MIN_CHILD_SAMPLES,
        reg_alpha=2.0,
        reg_lambda=2.0,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_class)

    # ── 5. Build feature matrix for candidates and predict ────────────────────
    cand_features = build_feature_matrix(loader, candidates, cutoff_date)
    result = candidates[["EID", "MONTH", "PEAKID"]].merge(
        cand_features[["EID", "MONTH", "PEAKID"] + FEATURE_COLUMNS],
        on=["EID", "MONTH", "PEAKID"],
        how="left",
    )

    X_cand = result[FEATURE_COLUMNS].fillna(0.0).values
    result["PREDICTED_PROFIT"] = model.predict_proba(X_cand)[:, 1]

    return result[["EID", "MONTH", "PEAKID", "PREDICTED_PROFIT"]]
