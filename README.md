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
python main.py --start-month 2024-01 --end-month 2024-06
```

Optional overrides:

```bash
python main.py --start-month 2024-01 --end-month 2024-06 \
    --data-root /path/to/data \
    --log-level DEBUG
```

The script writes `opportunities.csv` at the **project root** (same directory as `main.py`).

---

## 6. Project Structure

```
DataChallenge2026/
├── main.py                  # Entry point — month-by-month loop, CLI argument parsing
├── datachallenge/           # Core Python package
│   ├── __init__.py
│   ├── config.py            # Settings loaded from .env (DATA_ROOT, LOG_LEVEL, LOG_FILE)
│   ├── logger.py            # Shared logger (stderr + optional file handler)
│   ├── loader.py            # CustomDataLoader — queries parquet files via DuckDB
│   └── schemas.py           # TypedDict definitions documenting DataFrame column schemas
├── tests/                   # Integration tests (requires real data under data/)
│   ├── conftest.py          # Shared pytest fixtures (loader instance)
│   └── test_loader.py       # Tests for loader methods and output generation
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

`evaluate.py` is the official scoring script provided by MAG Energy Solutions. It reads your `opportunities.csv` and computes the two quantitative axes of the grading rubric against the actual realized data.

```bash
python evaluate.py opportunities.csv --start-month 2020-01 --end-month 2023-12
```

It outputs:
- **Axe 1 — F1-score** (precision, recall, F1 for ON-Peak and OFF-Peak separately, then averaged)
- **Axe 2 — Net profit** (sum of `|PR| − C` over all selected opportunities)
- A month-by-month breakdown (selections, TP, FP, profit per month)

> **Note**: `evaluate.py` reads data directly from `./data/` relative to its own location and does not accept a `--data-root` override. Run it from the project root with data in `data/`.

> **Do not modify** `evaluate.py` — it is read-only and provided by the organizers.

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

> **TODO**: Describe the scoring/selection approach here once implemented.
>
> Expected content:
> - Feature engineering from `sim_monthly`, `sim_daily`, historical prices and costs
> - Model or heuristic used to estimate `PREDICTED_PROFIT` per triplet (EID, MONTH, PEAKID)
> - How scenarios (SCENARIOID 1, 2, 3) are combined or used individually
> - How the 10–100 opportunity constraint is applied
> - Key design decisions and their justifications
> - Any domain-driven signals leveraged (ACTIVATIONLEVEL, impact variables, PSM/PSD)

---

## 10. Results and Analysis

> **TODO**: Fill in once the scoring algorithm is implemented and run on the validation set.
>
> Expected content:
> - F1-score (precision/recall) on the 2023 validation set, broken down by ON-Peak and OFF-Peak
> - Total net profit on the validation set
> - Distribution of selected opportunities per month (count, profitable fraction)
> - Comparison with baseline (e.g. random selection)
> - Error analysis: common false-positive and false-negative patterns
> - Potential improvements and open questions