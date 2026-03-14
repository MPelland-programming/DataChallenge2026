# FTR Opportunity Selection — Hypotheses & Analysis Guide

## Context

This project is for the MAG Energy Solutions data challenge (2026 edition).
The goal is to build an algorithm that selects between 10 and 100 profitable FTR
(Financial Transmission Rights) opportunities per month, for month M+1, using only
data available up to the 7th of month M (the "cutoff").

**Profitability**: an opportunity `o = (EID, MONTH, PEAKID)` is profitable when:
```
|PR_o| − C_o > 0
```
- `PR_o` = sum of realized hourly prices over the month (from `prices.parquet`); abs taken **after** summing
- `C_o`  = exposure cost for the month (from `costs.parquet`), used as-is (no abs)

> This formula matches `evaluate.py` exactly. Note that `abs(sum(prices)) ≠ sum(abs(prices))`
> when hourly prices can be negative (which occurs in electricity markets).

### Official Scoring Script

`evaluate.py` (read-only, provided by organizers) is the ground truth evaluator.

**Usage**: `python evaluate.py opportunities.csv --start-month YYYY-MM --end-month YYYY-MM`

**What it does**:
1. Loads all prices and costs from `./data/`
2. Computes ground truth: profitable = `abs(sum(PRICEREALIZED)) − C > 0` per triplet
3. Loads and validates `opportunities.csv` (deduplicates, enforces 100-selection max per month by keeping the first 100 rows)
4. Computes F1-score (ON and OFF separately, then averaged) and total net profit
5. Prints a month-by-month breakdown

**Important implementation detail**: if your CSV exceeds 100 selections for a month, `evaluate.py`
keeps the **first 100 rows** in file order — not the best-scoring ones. `select_best_predict_and_print()`
already caps at 100 sorted by predicted profit, so this should never trigger in practice.

**Constraint**: 10–100 selections per month, combining ON-Peak (PEAKID=1) and OFF-Peak (PEAKID=0).

**Scoring**:
- 25% F1-score (precision/recall on profitable triplets)
- 25% net profit (sum of PR−C over all selected opportunities)
- 50% jury (methodology, code quality, interpretability, presentation)

**Evaluation period**: 2020–2023 provided; 2024 released later; **2025 is the final test** (unseen).

---

## Data Overview

All data is sparsified: absence of a row means the value is 0 (not that the combination doesn't exist).

| Dataset          | Granularity | Key columns                                               |
|------------------|-------------|-----------------------------------------------------------|
| `costs`          | Monthly     | EID, MONTH, PEAKID, C (used as-is, no abs)               |
| `prices`         | Hourly      | EID, DATETIME, PEAKID, PRICEREALIZED (sum first, abs after) |
| `sim_monthly`    | Hourly      | SCENARIOID (1–3), EID, DATETIME, PEAKID, ACTIVATIONLEVEL, source impacts, PSM |
| `sim_daily`      | Hourly      | SCENARIOID (1–3), EID, DATETIME, PEAKID, ACTIVATIONLEVEL, source impacts, PSD |

Simulation variables:
- `ACTIVATIONLEVEL`: intensity of network constraint activation, expressed as a **percentage (0–100)**, not a fraction. Confirmed from threshold analysis (best threshold ≈ 42, not ≈ 0.42).
- Source impacts: `WINDIMPACT`, `SOLARIMPACT`, `HYDROIMPACT`, `NONRENEWBALIMPACT`, `EXTERNALIMPACT` — sum to partial contribution of ACTIVATIONLEVEL
- Explanatory vars (overlapping, do NOT sum with impacts): `LOADIMPACT`, `TRANSMISSIONOUTAGEIMPACT`
- `PSM`: simulated price (monthly sim), `PSD`: simulated price (daily sim)

---

## Base Rate — Answered ✓

> *Previously an open question. Resolved by `notebooks/01_data_exploration.ipynb`.*

The challenge's "<5% profitable" refers to the **full cartesian product** (Interpretation 1).

| Scope | Profitable / Total | Rate |
|-------|--------------------|------|
| Non-zero triplets (present in costs or prices) | 19 061 / 26 809 | **71.10%** |
| Full EID × MONTH × PEAKID cartesian product | 19 061 / 296 448 | **6.43%** |

The high non-zero rate (71%) is explained by sparsification: only economically active triplets
are stored, and those naturally skew profitable. The 6.43% cartesian rate aligns with the
challenge's claim.

**Implication**: the real challenge is not to find profitable triplets within a rich dataset —
it's to identify the small fraction of the full space that will be active *and* profitable.
Do not restrict the candidate pool to historically observed triplets only.

---

## Hypotheses

### H1 — ACTIVATIONLEVEL is a strong primary signal — ✗ REFUTED

> *Originally "confirmed by colleagues". Refuted by data analysis (2020–2023).*

| Metric | Value |
|--------|-------|
| AUC (mean ACTIVATIONLEVEL vs profitability) | **0.424** (below 0.5 = worse than random) |
| Spearman rank correlation | **−0.120** (p = 4.1e-86) |
| Best F1 threshold on train 2020–2022 | 42.46% activation |
| F1 on 2023 validation at that threshold | **0.000** |

Higher ACTIVATIONLEVEL is *negatively* associated with profitability. The threshold found on
2020–2022 (42.46%) selects zero triplets in 2023 — the distribution shifted, making the
signal completely unstable across years.

**Why**: ACTIVATIONLEVEL predicts *market activity*, but market activity drives up both PR
and C simultaneously. Profitability (PR − C > 0) requires the realized price to *exceed*
what was priced in at auction time. High activation means the market already priced the
constraint in; residual profit comes from the market being wrong, not from activation level.

**Key insight**: the sims tell you what the market already knows.
Profitability comes from what the market gets wrong.

**Consequence for modeling**: ACTIVATIONLEVEL should not be used as a positive signal.
Using it inversely (`−mean(ACTIVATIONLEVEL)` or `1 / (mean + ε)`) may be worth testing.
Do **not** use it as the primary score.

---

### H2 — PSM approximates realized price PR — ✗ REFUTED

| Metric | Value |
|--------|-------|
| Spearman r (sum(PSM) vs realized PRICE) | **−0.228** (p ≈ 0) |

PSM is *negatively* correlated with realized price. The same reasoning as H1 applies:
simulated prices reflect what the market expects, and FTR profits come from deviations
from those expectations.

**Consequence**: Model A (PSM − C_hist) is **not viable** — both sides of the formula
are unreliable (PSM anti-predicts PR; C has no autocorrelation per H3).

---

### H3 — Historical C is a good proxy for unknown C at M+1 — ✗ REFUTED

| Metric | Value |
|--------|-------|
| Lag-1 autocorrelation (month-to-month), median | **−0.006** |
| Lag-12 autocorrelation (same month of year), median | **−0.029** |

C is essentially a random walk — no persistence month-to-month or seasonally.
Historical C cannot be used to estimate C at M+1.

**Consequence**: any model that requires a C estimate (Model A) loses its foundation.
The only viable path is to predict profitability directly (H4-based) rather than
decomposing into price and cost.

---

### H4 — Some EIDs are "chronically profitable" — ✓ CONFIRMED

> *This is the strongest signal found in the data.*

| Metric | Value |
|--------|-------|
| (EID, PEAKID) pairs with ≥6 months history | 1 229 |
| Pairs with >50% historical win rate | **68.9%** |
| Pairs with >80% historical win rate | **44.9%** |
| Seasonal pairs (EID, PEAKID, month-of-year) with ≥2 years, >50% win rate | **56.6%** |

Certain network elements are structurally and repeatedly profitable, likely due to
persistent grid topology constraints (chronic transmission bottlenecks, hydro corridors, etc.).
This signal is backward-looking but stable across years.

**Usage**: `score = historical_profit_rate(EID, PEAKID)` over all months strictly before
the cutoff. This is the recommended **new baseline** (replaces ACTIVATIONLEVEL).

**Limitation**: new EIDs with no history cannot be scored this way. Fall back to a global
average or a sim-based signal (possibly inverted) for those.

---

### H5 — Scenario consensus signals reliability — ✗ MARGINAL / NOT USEFUL

| Metric | Value |
|--------|-------|
| Consensus AUC (`mean / std_across_scenarios`) | **0.451** |
| Plain mean AUC | 0.424 |
| Point-biserial r (−std vs profitability), global | +0.095 (p = 2.8e-55) |
| Point-biserial r (−std vs profitability), high-activation subset | −0.035 (p = 6.4e-05) |

The consensus score marginally improves AUC (0.424 → 0.451) but both remain below 0.5.
Globally, lower scenario std is weakly associated with higher profitability — but among
high-activation triplets (the ones we'd actually consider selecting), the effect reverses.
Not a useful standalone signal.

**Consequence**: H5 can be dropped from further modeling. If a consensus feature is ever
used in a regression model, it should be combined with H4-based features and validated
carefully on a holdout year.

---

### H6 — Source impacts reveal the nature of the opportunity (interpretability)

> *Not yet analyzed quantitatively. Retained for jury presentation value.*

The source impacts (WIND, SOLAR, HYDRO, NONRENEWBAL, EXTERNAL) explain *why* ACTIVATIONLEVEL
is high. Even though ACTIVATIONLEVEL itself is not a useful predictive signal, the source
impacts may still be valuable for:
1. **Interpretability**: explaining to the jury *why* certain EIDs are selected.
2. **Feature engineering**: renewable-driven impacts (weather-dependent) may behave
   differently from load or transmission-driven impacts, even if the overall signal is weak.

**Status**: analysis deferred. Low priority until H4-based model is established.

---

## Zero Pruning — Do Not Prune ✓

> *Resolved by `notebooks/01_data_exploration.ipynb`.*

| Group | Count | Profitable | Rate |
|-------|-------|------------|------|
| Zero ACTIVATIONLEVEL in sims | 373 | 340 | **91.15%** |
| Non-zero ACTIVATIONLEVEL in sims | 22 033 | 15 066 | 68.38% |
| Absent from sims entirely | 4 403 | 3 655 | 83.01% |

Zero-sim and absent EIDs are *more* profitable than non-zero-sim EIDs. Pruning them would
actively harm performance. This reinforces H1 being refuted: the sims anti-predict
profitability. Do not filter the candidate pool based on sim presence or activation level.

---

## Data Exploration Checklist — All Complete ✓

> See `notebooks/01_data_exploration.ipynb` for full results.

- [x] **Base rate decomposition** → 71.10% non-zero, 6.43% cartesian (see above)
- [x] **ACTIVATIONLEVEL vs profitability** → H1 refuted (AUC 0.424, r = −0.12)
- [x] **ACTIVATIONLEVEL threshold** → best threshold 42.46% on train; F1 = 0 on 2023
- [x] **PSM vs realized PR** → H2 refuted (Spearman r = −0.228)
- [x] **C stability** → H3 refuted (lag-1 median = −0.006)
- [x] **Scenario consensus** → H5 marginal (AUC 0.451 vs 0.424, reverses in high-act subset)
- [x] **Chronic winners** → H4 confirmed (68.9% pairs >50% win rate)
- [x] **Zero pruning** → not safe (zero-sim triplets 91% profitable)

---

## Modeling Approaches (updated priority)

### Model 0b — New Baseline: Historical Profitability Rate *(recommended next step)*

Score = `historical_profit_rate(EID, PEAKID)` = fraction of months profitable over all
months strictly before the cutoff, per (EID, PEAKID) pair.

Pros: simple, interpretable, directly validated by H4, stable across years.
Cons: blind to forward-looking information; new EIDs score 0 (need a fallback).

Fallback for new EIDs: global average profit rate, or seasonal average if available.

### Model 0 — Old Baseline: ACTIVATIONLEVEL ranking *(disproven, kept for reference)*

Score = `mean(m_ACTIVATIONLEVEL)` across all 3 scenarios and all hours of M+1.

**This model is anti-predictive.** H1 was refuted. Kept as a reference/comparison point
only. Its inverted form (`−mean(ACTIVATIONLEVEL)`) may be worth testing.

See `datachallenge/scoring.py :: score_by_activation_level`.

### Model A — Simulation price − estimated cost *(not viable)*

Score = `sum(PSM_mean)` − `C_historical_mean`.

**Not viable**: H2 (PSM anti-correlates with PR) and H3 (C has no autocorrelation) both
refuted. Do not pursue unless new evidence changes these findings.

### Model B — Historical profitability + (inverted) sim signal

Score = `alpha × historical_profit_rate + (1−alpha) × (−normalized_activation)`

Pros: combines the confirmed H4 signal with a (possibly inverted) forward-looking component.
Cons: requires tuning alpha; the inverted sim signal needs validation before use.

### Model C — Logistic / linear regression

Features: `historical_profit_rate`, `−ACTIVATIONLEVEL_mean`, `ACTIVATIONLEVEL_std` (across
scenarios), seasonal dummy (month-of-year), PEAKID.
Target: binary profitable (logistic) or expected profit (linear).

Pros: principled, handles feature interactions, interpretable via coefficients.
Cons: more complex; requires careful time-series cross-validation (no future leakage).

---

## Implementation Notes

- Scorer functions live in `datachallenge/scoring.py`. Each returns a DataFrame with
  `EID, MONTH, PEAKID, PREDICTED_PROFIT` for use with the selection pipeline.
- Selection logic lives in `datachallenge/selection.py` (10–100 constraint).
- Output validation and CSV writing live in `datachallenge/output.py`.
- Candidate pool construction is in `datachallenge/candidates.py` — see docstring for
  the rationale behind expanding all known EIDs to both PEAKID values.
- Time-series cross-validation: train on years N–(N+k), validate on N+k+1. Never shuffle.
- **Do not prune** zero-sim or absent EIDs from the candidate pool (see zero pruning above).
- `m_ACTIVATIONLEVEL` is on a **0–100 (percent)** scale, not 0–1.
