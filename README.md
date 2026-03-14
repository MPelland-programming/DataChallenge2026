# MAG Energy Solutions - Data Challenge 2026

## 1. Contexte
MAG Energy Solutions est un chef de file dans le trading d'électricité en Amérique du Nord. Le trading d'électricité se situe à l'intersection de l'ingénierie et de la finance, nécessitant de relier des signaux opérationnels à des opportunités financières concrètes. 

Ce projet se concentre sur les produits des FTR (Financial Transmission Rights). L'issue économique se joue dans l'écart entre ce qui était anticipé et ce qui se matérialise.

## 2. Objectif du Projet
L'objectif est de construire un algorithme qui identifie les situations les plus susceptibles de générer de la valeur selon deux profils horaires (On-Peak et Off-Peak) pour le mois suivant (M+1). 

* **La mission** : Sélectionner entre 10 et 100 opportunités par mois (total ON + OFF combinés).
* **Définition de la profitabilité** : Une opportunité est profitable lorsque PR_o - C_o > 0 (le prix réalisé mensuel est supérieur au coût d'exposition).
* **Livrable final** : Un fichier CSV nommé opportunities.csv listant les sélections.

## 3. La Contrainte Temporelle (Règle Anti-Fuite)
La difficulté principale du défi réside dans le moment de la décision (cutoff).
* L'analyse est supposée être réalisée le 7 du mois M (inclus), avec les données accessibles jusqu'à 23:59:59.
* **Information autorisée** : Historique des marchés jusqu'à M-1, prix réalisés de M jusqu'au 7 inclus, et simulations prospectives mensuelles pour M et M+1.
* **Information strictement interdite** : Les prix réalisés ou les coûts du mois M+1, et les simulations journalières pour les jours après le 7 de M.

## 4. Données Disponibles
Les données marché sont sparsifiées (seules les valeurs non nulles sont fournies, l'absence équivaut à 0) et réparties dans quatre dossiers :
* `/data/costs` : Coûts d'exposition mensuels.
* `/data/prices` : Prix réalisés horaires.
* `/data/sim_monthly` : Simulations mensuelles avec 3 scénarios.
* `/data/sim_daily` : Simulations journalières avec 3 scénarios.

---

## 5. Install & Run

### Prerequisites
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (fast Python package manager)

### Setup

```bash
# Install dependencies
uv sync

# Copy the example environment file and fill in DATA_ROOT
cp .env.example .env
# Edit .env and set DATA_ROOT to the path of your local data folder
```

### Running

```bash
# Full dataset (default scorer, default selector — no args needed):
python main.py

# Explicit date range:
python main.py --start-month 2020-01 --end-month 2022-12

# Choose a specific scorer or selector:
python main.py --scorer historical_profit_rate --selector default

# All overrides:
python main.py --start-month 2020-01 --end-month 2022-12 \
    --scorer activation_level --selector default \
    --data-root /path/to/data \
    --log-level DEBUG
```

The script writes `opportunities.csv` at the **project root** (same directory as `main.py`).

Available scorers: `activation_level` (default), `historical_profit_rate`. Available selectors: `default`. Adding a new one requires only one line in the corresponding registry dict at the top of `main.py`.

---

## 6. Project Structure

```
DataChallenge2026/
├── main.py                  # Entry point — month-by-month loop, scorer/selector registry, CLI
├── eval_wrapper.py          # Evaluation orchestrator — runs main.py + evaluate.py, writes JSON
├── evaluate.py              # Official scoring script (read-only, provided by organizers)
├── datachallenge/           # Core Python package
│   ├── __init__.py
│   ├── config.py            # Settings loaded from .env (DATA_ROOT, LOG_LEVEL, LOG_FILE)
│   ├── logger.py            # Shared logger (stderr + optional file handler)
│   ├── loader.py            # CustomDataLoader — queries parquet files via DuckDB
│   ├── schemas.py           # TypedDict definitions documenting DataFrame column schemas
│   ├── candidates.py        # build_candidate_pool — expands known EIDs to both PEAKID values
│   ├── scoring.py           # ScorerProtocol + score_by_activation_level / score_by_historical_profit_rate
│   ├── selection.py         # SelectorProtocol + select_opportunities (10–100 constraint)
│   └── output.py            # write_opportunities — validates and writes opportunities.csv
├── tests/                   # Unit + integration tests (33 tests, all passing)
│   ├── conftest.py          # Shared pytest fixtures (loader instance)
│   ├── test_loader.py       # Integration tests for loader methods
│   ├── test_scoring.py      # Unit tests for scoring functions (mock-based)
│   ├── test_selection.py    # Unit tests for select_opportunities
│   └── test_output.py       # Unit tests for write_opportunities
├── results/                 # Committed evaluation results — one JSON per (scorer, selector, period)
│   ├── activation_level__default__202001_202212.json
│   ├── activation_level__default__202301_202312.json
│   ├── historical_profit_rate__default__202001_202212.json
│   └── historical_profit_rate__default__202301_202312.json
├── notebooks/               # Data exploration notebooks (committed)
├── HYPOTHESES.md            # Data analysis, hypothesis verdicts, modeling approach
├── .env.example             # Template for environment variables
├── requirements.txt         # Pinned dependencies (use uv sync to install)
└── data/                    # Local data folder (git-ignored)
    ├── costs/costs.parquet
    ├── prices/prices.parquet
    ├── sim_daily/sim_daily_<year>.parquet
    └── sim_monthly/sim_monthly_<year>.parquet
```

---

## 7. Evaluating Your Output

### Official script — `evaluate.py`

Provided by MAG Energy Solutions (read-only). Reads `opportunities.csv` and computes the two quantitative grading axes against realized data.

```bash
python evaluate.py opportunities.csv --start-month 2020-01 --end-month 2023-12
```

Outputs: F1-score (ON/OFF separately + average), net profit, month-by-month breakdown.

> `evaluate.py` reads data from `./data/` relative to its location. Run it from the project root. **Do not modify it.**

### Evaluation wrapper — `eval_wrapper.py`

Automates the full loop: runs `main.py`, then `evaluate.py`, parses the output, and saves a JSON result to `results/`.

```bash
# Standard run — generates opportunities.csv then evaluates:
python eval_wrapper.py --scorer activation_level --selector default \
    --start-month 2020-01 --end-month 2022-12

# Re-evaluate an existing opportunities.csv without re-running main.py:
python eval_wrapper.py --scorer activation_level --selector default \
    --start-month 2020-01 --end-month 2022-12 --dry-run
```

Output is saved to `results/{scorer}__{selector}__{start}_{end}.json` with aggregate metrics and a per-month breakdown. Results in `results/` are committed — they are the record of what was tried and when.

---

## 8. Recommended Train / Validation Split

The provided dataset covers **2020–2023**. A 2024 out-of-sample set will be distributed later for robustness testing. Final scoring uses 2025 (never provided).

| Purpose | Period | Command |
|---------|--------|---------|
| Default evaluation (full provided set) | 2020-01 → 2023-12 | `python main.py` |
| Training only | 2020-01 → 2022-12 | `python main.py --start-month 2020-01 --end-month 2022-12` |
| Validation only | 2023-01 → 2023-12 | `python main.py --start-month 2023-01 --end-month 2023-12` |
| Out-of-sample (2024, when available) | 2024-01 → 2024-12 | `python main.py --start-month 2024-01 --end-month 2024-12` |

The anti-leakage cutoff is enforced automatically for every month — there is no risk of data contamination between periods regardless of the date range chosen.

---

## 9. Methodological Approach

Two scoring strategies are implemented in `datachallenge/scoring.py` and selectable via `--scorer`:

### `activation_level` (current default)
Score = `mean(m_ACTIVATIONLEVEL)` across all 3 monthly simulation scenarios for month M+1. Triplets absent from the sims receive score 0. The top candidates by score are selected (10–100 per month).

Despite the hypothesis analysis showing ACTIVATIONLEVEL is negatively associated with profitability at the population level (AUC=0.424), this scorer outperforms the historical baseline in empirical evaluation — see Section 10 for details.

### `historical_profit_rate`
Score = fraction of months where `|PR| − C > 0`, computed over all months strictly before the cutoff (month M and earlier). Uses only price/cost data available at decision time — no future leakage. EIDs with no history receive the global average win rate as a fallback.

**Selection pipeline** (`datachallenge/selection.py`): deduplicates by averaging scores per triplet, sorts descending, selects the top *n* where *n* = number of candidates with score > 0, clamped to [10, 100]. **Output** (`datachallenge/output.py`): writes `opportunities.csv` with columns `TARGET_MONTH`, `PEAK_TYPE`, `EID`.

**Anti-leak guarantee**: the 7th-of-month cutoff is strictly enforced in the loader and in every scorer. See `HYPOTHESES.md` for full data analysis.

---

## 10. Results and Analysis

Full per-run results (with monthly breakdowns) are in `results/`. Evaluation performed with `eval_wrapper.py` on 2020–2022 (train) and 2023 (validation).

### Model comparison table

| Scorer | Selector | Period | F1 avg | F1 OFF | F1 ON | Precision | Recall | Net Profit |
|--------|----------|--------|--------|--------|-------|-----------|--------|------------|
| `activation_level` | `default` | 2020–2022 (train) | **0.1337** | 0.1362 | 0.1312 | 0.360 | 0.082 | **3,471,724** |
| `historical_profit_rate` | `default` | 2020–2022 (train) | 0.0478 | 0.0493 | 0.0464 | 0.132 | 0.029 | 783,256 |
| `activation_level` | `default` | 2023 (val) | **0.0967** | 0.0977 | 0.0957 | 0.202 | 0.064 | **132,999** |
| `historical_profit_rate` | `default` | 2023 (val) | 0.0354 | 0.0308 | 0.0400 | 0.074 | 0.029 | 17,641 |

### Interpretation

**`activation_level` wins on all metrics** across both periods, despite H1 in `HYPOTHESES.md` concluding it is anti-predictive at the population level (AUC=0.424, Spearman r=−0.12). The likely explanation: the anti-predictive finding applies to threshold-based selection (select all above 42% activation → F1=0 in 2023), whereas **top-k ranking** still concentrates more active, better-documented constraints in the top 100 — and those constraints, even if already priced in on average, involve larger price swings that generate higher absolute profit.

**`historical_profit_rate` underperforms** its theoretical motivation (H4: 69% of pairs have >50% win rate). The most likely cause is that the historical signal is strongly diluted by the candidate pool expansion: the pool includes all known EIDs × 2 PEAKID values, meaning many candidates share similar historical rates and the ranking does not spread as effectively across months.

**Current recommendation**: use `activation_level + default` (the default with no extra arguments). The next improvement to explore is an inverted or hybrid signal (see `HYPOTHESES.md` § Model B) or candidate pool refinement.