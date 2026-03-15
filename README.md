# MAG Energy Solutions - Data Challenge 2026

## 1. Contexte

MAG Energy Solutions est un chef de file dans le trading d'électricité en Amérique du Nord. Le trading d'électricité se situe à l'intersection de l'ingénierie et de la finance, nécessitant de relier des signaux opérationnels à des opportunités financières concrètes.

Ce projet se concentre sur les produits des FTR (Financial Transmission Rights). L'issue économique se joue dans l'écart entre ce qui était anticipé et ce qui se matérialise : pour chaque élément du réseau électrique, un prix réalisé (PR) horaire se concrétise au fil du mois, tandis qu'un coût d'exposition (C) mensuel est fixé par le marché. L'objectif est de sélectionner en amont les opportunités où PR − C > 0.

## 2. Objectif du Projet

L'objectif est de construire un algorithme qui identifie, à partir de données de simulations disponibles au moment de la décision (7 du mois M), les opportunités les plus susceptibles de générer de la valeur pour le mois M+1, selon deux profils horaires (On-Peak et Off-Peak).

* **La mission** : Sélectionner entre 10 et 100 opportunités par mois (total ON + OFF combinés).
* **Définition de la profitabilité** : Une opportunité `o = (EID, MONTH, PEAKID)` est profitable lorsque `PR_o − C_o > 0` (la somme des prix réalisés horaires du mois dépasse le coût d'exposition).
* **Livrable final** : Un fichier CSV nommé `opportunities.csv` listant les sélections.

## 3. La Contrainte Temporelle (Règle Anti-Fuite)

La difficulté principale du défi réside dans le moment de la décision (cutoff).

* L'analyse est supposée être réalisée le **7 du mois M** (inclus), avec les données accessibles jusqu'à 23:59:59. En convention Hour Ending (HE), la dernière entrée valable est horodatée le 8 du mois M à 00:00:00.
* **Informations autorisées** : Historique complet des prix/coûts jusqu'à M−1, prix réalisés de M jusqu'au 7 inclus, coût C de M, simulations mensuelles historiques + prospectives pour M et M+1, simulations journalières jusqu'au 7 de M.
* **Informations strictement interdites** : Les prix réalisés ou les coûts du mois M+1, les prix réalisés de M après le 7, les simulations journalières après le 7 de M, tout agrégat dérivé de données interdites.

L'anti-fuite est appliquée à **trois couches indépendantes** dans le code :
1. **Couche données** (`loader.py`) : les requêtes DuckDB filtrent par `DATETIME <= cutoff_date`.
2. **Couche features** (`features.py`) : l'historique est restreint aux mois strictement avant M.
3. **Couche scoring** (`scoring.py`) : chaque scorer exclut explicitement le mois M de l'historique d'entraînement.

## 4. Données Disponibles

Les données marché sont **sparsifiées** (seules les valeurs non nulles sont fournies ; l'absence équivaut à 0) et réparties dans quatre dossiers :

| Dossier | Contenu | Granularité | Colonnes clés |
|---------|---------|-------------|---------------|
| `/data/costs` | Coûts d'exposition mensuels | Mensuelle | `EID, MONTH, PEAKID, C` |
| `/data/prices` | Prix réalisés horaires | Horaire | `EID, DATETIME, PEAKID, PRICEREALIZED` |
| `/data/sim_monthly` | Simulations mensuelles (3 scénarios) | Horaire | `SCENARIOID, EID, DATETIME, PEAKID, ACTIVATIONLEVEL, impacts..., PSM` |
| `/data/sim_daily` | Simulations journalières (3 scénarios) | Horaire | `SCENARIOID, EID, DATETIME, PEAKID, ACTIVATIONLEVEL, impacts..., PSD` |

**Note de cohérence** : dans les données, le profil horaire est codé via `PEAKID` (0 = OFF, 1 = ON). Dans la sortie CSV, on utilise `PEAK_TYPE` (ON ou OFF).

---

## 5. Installation et Exécution

### Prérequis

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (gestionnaire de paquets Python rapide)

### Installation

```bash
# Installer les dépendances
uv sync

# Copier le fichier d'environnement et configurer DATA_ROOT
cp .env.example .env
# Éditer .env et définir DATA_ROOT vers le dossier de données local
```

### Exécution

```bash
# Jeu complet fourni (scorer et sélecteur par défaut, aucun argument requis) :
python main.py

# Plage de dates explicite :
python main.py --start-month 2020-01 --end-month 2022-12

# Choisir un scorer spécifique :
python main.py --scorer lightgbm

# Tous les paramètres :
python main.py --start-month 2020-01 --end-month 2022-12 \
    --scorer maxime_short --selector default \
    --data-root /path/to/data \
    --log-level DEBUG
```

Le script écrit `opportunities.csv` à la **racine du projet** (même répertoire que `main.py`).

**Scorers disponibles** : `activation_level` (défaut), `historical_profit_rate`, `maxime_short`, `lasso`, `lightgbm`.
**Sélecteurs disponibles** : `default`.

---

## 6. Structure du Projet

```
DataChallenge2026/
├── main.py                  # Point d'entrée — boucle mois par mois, registre scorer/sélecteur, CLI
├── eval_wrapper.py          # Orchestrateur d'évaluation — exécute main.py + evaluate.py, écrit JSON
├── evaluate.py              # Script de scoring officiel (lecture seule, fourni par les organisateurs)
├── datachallenge/           # Package Python principal
│   ├── __init__.py
│   ├── config.py            # Configuration chargée depuis .env (DATA_ROOT, LOG_LEVEL, LOG_FILE)
│   ├── logger.py            # Logger partagé (stderr + fichier optionnel)
│   ├── loader.py            # CustomDataLoader — requêtes parquet via DuckDB (6 méthodes)
│   ├── schemas.py           # Définitions TypedDict documentant les schémas de colonnes
│   ├── candidates.py        # build_candidate_pool — construit le pool candidat depuis l'univers sim
│   ├── features.py          # build_feature_matrix — ingénierie de 36 features (V3)
│   ├── scoring.py           # 5 scorers : activation_level, historical_profit_rate, maxime_short, lasso, lightgbm
│   ├── selection.py         # select_opportunities — applique la contrainte 10–100
│   └── output.py            # write_opportunities — valide et écrit opportunities.csv
├── tests/                   # Tests unitaires + intégration (33 tests)
│   ├── conftest.py          # Fixtures pytest partagées (instance du loader)
│   ├── test_loader.py       # Tests d'intégration pour les méthodes du loader
│   ├── test_candidates.py   # Tests du pool candidat
│   ├── test_features.py     # Tests de la matrice de features
│   ├── test_scoring.py      # Tests unitaires des scorers (mocks)
│   ├── test_selection.py    # Tests de la sélection
│   └── test_output.py       # Tests de validation de la sortie
├── results/                 # Résultats d'évaluation — un JSON par (scorer, sélecteur, période)
├── notebooks/               # Notebooks d'exploration des données
├── HYPOTHESES.md            # Analyse des données, verdicts d'hypothèses, approche de modélisation
├── .env.example             # Template des variables d'environnement
├── requirements.txt         # Dépendances Python
└── data/                    # Dossier de données local (git-ignored)
    ├── costs/costs.parquet
    ├── prices/prices.parquet
    ├── sim_daily/sim_daily_<year>.parquet
    └── sim_monthly/sim_monthly_<year>.parquet
```

---

## 7. Évaluation

### Script officiel — `evaluate.py`

Fourni par MAG Energy Solutions (lecture seule). Lit `opportunities.csv` et calcule les deux axes quantitatifs de notation (F1-score et profit net) à partir des données réalisées.

```bash
python evaluate.py opportunities.csv --start-month 2020-01 --end-month 2023-12
```

> `evaluate.py` lit les données depuis `./data/` relativement à son emplacement. L'exécuter depuis la racine du projet. **Ne pas le modifier.**

### Wrapper d'évaluation — `eval_wrapper.py`

Automatise la boucle complète : exécute `main.py`, puis `evaluate.py`, parse la sortie et sauvegarde un JSON dans `results/`.

```bash
# Exécution standard — génère opportunities.csv puis évalue :
python eval_wrapper.py --scorer activation_level --selector default \
    --start-month 2020-01 --end-month 2022-12

# Ré-évaluer un opportunities.csv existant sans ré-exécuter main.py :
python eval_wrapper.py --scorer activation_level --selector default \
    --start-month 2020-01 --end-month 2022-12 --dry-run
```

---

## 8. Découpage Entraînement / Validation

Le jeu de données fourni couvre **2020–2023**. Un jeu out-of-sample 2024 sera distribué ultérieurement. Le scoring final utilise 2025 (jamais fourni).

| Période | Usage | Commande |
|---------|-------|----------|
| 2020-01 → 2023-12 | Évaluation complète (jeu fourni) | `python main.py` |
| 2020-01 → 2022-12 | Entraînement | `python main.py --end-month 2022-12` |
| 2023-01 → 2023-12 | Validation | `python main.py --start-month 2023-01` |
| 2024-01 → 2024-12 | Out-of-sample (quand disponible) | `python main.py --start-month 2024-01 --end-month 2024-12` |

L'anti-fuite est appliquée automatiquement pour chaque mois — il n'y a aucun risque de contamination entre périodes, quel que soit l'intervalle de dates choisi.

---

## 9. Approche Méthodologique

### 9.1. Pipeline Général

Pour chaque mois cible M+1, le pipeline s'exécute comme suit :

```
1. build_candidate_pool()   → Charger tous les (EID, PEAKID) de l'univers de simulation mensuelle
                               pour M+1 (~170k paires par mois)
2. scorer_fn()              → Scorer chaque candidat avec le scorer sélectionné
3. select_opportunities()   → Dédupliquer, trier par score décroissant,
                               sélectionner les top-N avec score > 0, borné à [10, 100]
4. write_opportunities()    → Écrire le CSV : TARGET_MONTH, PEAK_TYPE, EID
```

### 9.2. Stratégies de Scoring

Cinq scorers sont implémentés dans `datachallenge/scoring.py`, sélectionnables via `--scorer` :

#### `activation_level` (défaut actuel)

**Type** : Heuristique non-supervisée (sans entraînement).

**Score** : `mean(ACTIVATIONLEVEL)` à travers les 3 scénarios de simulation mensuelle pour le mois M+1. Les triplets absents des simulations reçoivent un score de 0.

**Justification** : Malgré l'analyse d'hypothèses montrant que ACTIVATIONLEVEL est négativement associé à la profitabilité au niveau de la population (AUC = 0.424, Spearman r = −0.12), le scoring par **classement top-k** est robuste au changement de distribution entre périodes. Les seuils absolus (ex. sélectionner tout au-dessus de 42%) collapsent à 0 sélection en 2023, alors que le top-k reste stable. Ce scorer concentre les contraintes les plus actives et documentées dans le top 100 — celles-ci impliquent des swings de prix plus importants en valeur absolue.

#### `historical_profit_rate`

**Type** : Heuristique basée sur l'historique (sans entraînement).

**Score** : Fraction des mois historiques où `PROFIT > 0` pour chaque paire `(EID, PEAKID)`, calculée sur les mois strictement avant M. Les EIDs sans historique reçoivent le taux de profit moyen global comme fallback.

**Justification** : Exploite l'observation que 68.9% des paires avec ≥6 mois d'historique ont un taux de profit supérieur à 50% (« chronic winners »). Cependant, le signal est dilué par l'expansion du pool candidat (~170k paires).

#### `maxime_short` (supervisé, deux têtes)

**Type** : Modèle supervisé hybride LogisticRegression + Ridge.

**Score** : `P × V` où P = probabilité de profit (tête logistique) et V = profit prédit (tête Ridge).

**Entraînement** : Walk-forward — tous les triplets historiques avec MONTH < M. Un embargo de 1 mois est appliqué implicitement. La tête logistique utilise `class_weight="balanced"` pour compenser le déséquilibre des classes (<5% profitables). La tête Ridge pondère les exemples profitables 2× les non-profitables.

**Features** : Matrice complète de 36 features (voir Section 10).

#### `lasso`

**Type** : Régression Lasso avec sélection automatique de features.

**Score** : Prédiction directe du PROFIT par régression.

**Entraînement** : Walk-forward, avec sélection de l'hyperparamètre alpha via `LassoCV` et `TimeSeriesSplit`. Alpha est recalculé à chaque cutoff pour s'adapter à l'historique disponible. La pénalité L1 envoie les coefficients des features non-pertinentes vers zéro, offrant une sélection de features implicite utile pour l'interprétabilité.

**Features** : Matrice complète de 36 features (voir Section 10).

#### `lightgbm`

**Type** : Classificateur par gradient boosting (LightGBM).

**Score** : `predict_proba[:, 1]` = probabilité d'être profitable.

**Entraînement** : Walk-forward, label binaire `(PROFIT > 0)`. Entraîne en quelques secondes sur les ~170k candidats grâce aux splits par histogrammes. Ne nécessite pas de StandardScaler (les splits d'arbres sont basés sur les rangs).

**Hyperparamètres** : `n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, min_child_samples=100, reg_alpha=2.0, reg_lambda=2.0`.

**Features** : Matrice complète de 36 features (voir Section 10).

### 9.3. Logique de Sélection

La fonction `select_opportunities()` applique la contrainte de 10–100 opportunités par mois :

1. **Dédupliquer** : moyenner le score par triplet `(EID, MONTH, PEAKID)`.
2. **Trier** : par score décroissant.
3. **Compter** : n_profitable = nombre de candidats avec score > 0.
4. **Sélectionner** : si n_profitable < 10 → top 10 ; si n_profitable > 100 → top 100 ; sinon → tous les positifs.

---

## 10. Ingénierie des Features (36 features)

Le module `datachallenge/features.py` construit une matrice de features plate (une ligne par triplet) utilisée par les trois scorers supervisés (`maxime_short`, `lasso`, `lightgbm`). Toutes les features respectent la contrainte anti-fuite : elles sont dérivées uniquement des données disponibles au cutoff (7 du mois M).

### 10.1. Features de profit estimé (4 features)

| Feature | Description | Source |
|---------|-------------|--------|
| `estimated_profit` | `mean(|SUM(PSM)|) − cost_proxy` : différence entre le revenu simulé moyen et le coût proxy | Sim mensuelle M+1 + Coûts M |
| `sum_abs_psm_s1` | `|SUM(PSM)|` pour le scénario 1 — revenu simulé total en valeur absolue | Sim mensuelle M+1 |
| `sum_abs_psm_s2` | `|SUM(PSM)|` pour le scénario 2 | Sim mensuelle M+1 |
| `cost_proxy` | `abs(C_M)` : coût d'exposition du mois M en valeur absolue. Fallback : médiane historique des `|C|` pour les triplets absents de M | Coûts M |

### 10.2. Features de consensus inter-scénarios (2 features)

| Feature | Description | Source |
|---------|-------------|--------|
| `psm_cv_scenarios` | Coefficient de variation `std(|SUM(PSM)|) / mean(|SUM(PSM)|)` entre les 3 scénarios — mesure l'incertitude du modèle | Sim mensuelle M+1 |
| `estimated_profit_pessimistic` | `min(|SUM(PSM)|_s − cost_proxy)` sur les 3 scénarios — profit estimé dans le pire cas | Sim mensuelle M+1 + Coûts M |

### 10.3. Features d'activation et d'impacts (12 features)

Issues des simulations mensuelles pour M+1, agrégées sur les 3 scénarios.

| Feature | Description |
|---------|-------------|
| `mean_activation` | Moyenne de `ACTIVATIONLEVEL` — intensité moyenne de l'opportunité (en %) |
| `max_activation` | Maximum de `ACTIVATIONLEVEL` — capture les événements extrêmes |
| `pct_high_activation` | Proportion des heures avec `ACTIVATIONLEVEL > 50%` — fréquence d'activité élevée |
| `mean_wind` | Moyenne de `WINDIMPACT` — contribution de l'éolien à l'intensité |
| `mean_solar` | Moyenne de `SOLARIMPACT` — contribution du solaire |
| `mean_hydro` | Moyenne de `HYDROIMPACT` — contribution de l'hydraulique |
| `mean_nonrenew` | Moyenne de `NONRENEWBALIMPACT` — contribution des non-renouvelables |
| `mean_external` | Moyenne de `EXTERNALIMPACT` — contribution des facteurs externes |
| `mean_transmission_outage` | Moyenne de `TRANSMISSIONOUTAGEIMPACT` — impact des pannes de transmission |
| `mean_load` | Moyenne de `LOADIMPACT` — impact de la charge |
| `n_hours_active` | Nombre d'heures avec `PSM ≠ 0` (moyenné sur les scénarios) — proxy de la durée d'activité |
| `impact_concentration` | `max(|src_impacts|) / sum(|src_impacts|)` — mesure la concentration : un impact domine-t-il ? |

**Note** : Les impacts "par source" (wind, solar, hydro, nonrenew, external) constituent une somme partielle de `ACTIVATIONLEVEL`. Les variables `LOADIMPACT` et `TRANSMISSIONOUTAGEIMPACT` sont des variables explicatives avec chevauchements et ne doivent pas être sommées avec les impacts par source.

### 10.4. Features de simulation journalière (5 features)

Issues des simulations journalières pour le mois M, jours 1 à 7 (avant le cutoff).

| Feature | Description |
|---------|-------------|
| `daily_sum_abs_psd` | Moyenne sur les 3 scénarios de `|SUM(PSD)|` — signal prix court-terme |
| `daily_mean_activation` | Moyenne de `ACTIVATIONLEVEL` dans les sims journalières |
| `daily_max_activation` | Maximum de `ACTIVATIONLEVEL` dans les sims journalières |
| `daily_mean_trans_outage` | Moyenne de `TRANSMISSIONOUTAGEIMPACT` dans les sims journalières |
| `daily_n_hours_active` | Nombre d'heures avec `PSD ≠ 0` dans les sims journalières |

### 10.5. Features historiques (9 features)

Calculées sur les mois strictement avant M pour chaque paire `(EID, PEAKID)`.

| Feature | Description |
|---------|-------------|
| `hist_win_rate` | Fraction des mois historiques profitables. Fallback : taux moyen global pour les EIDs sans historique |
| `hist_n_months` | Nombre de mois historiques avec données — proxy de l'ancienneté de l'élément |
| `hist_mean_profit` | Profit moyen historique |
| `hist_std_profit` | Écart-type du profit historique — proxy de la volatilité |
| `hist_last_6m_win_rate` | Taux de profit sur les 6 derniers mois — signal de récence |
| `hist_consecutive_wins` | Série de victoires consécutives (mois profitables en continu) en partant du plus récent |
| `hist_seasonal_win_rate` | Taux de profit pour le même mois calendaire dans les années précédentes — capture la saisonnalité |
| `hist_mean_cost` | Coût moyen historique |
| `hist_mean_price` | Prix réalisé moyen historique |

### 10.6. Features dérivées (4 features)

| Feature | Description |
|---------|-------------|
| `profit_per_active_hour` | `estimated_profit / max(n_hours_active, 1)` — rentabilité par heure d'activité |
| `monthly_daily_ratio` | `daily_sum_abs_psd / mean_sum_abs_psm` — rapport entre signal court-terme et signal mensuel |
| `month_sin` | `sin(2π × mois_M+1 / 12)` — encodage circulaire saisonnier |
| `month_cos` | `cos(2π × mois_M+1 / 12)` — encodage circulaire saisonnier (composante complémentaire) |

---

## 11. Résultats et Analyse
 
Les résultats détaillés (avec ventilations mensuelles) sont dans `results/`. L'évaluation a été réalisée en walk-forward : pour chaque mois à prédire, le modèle est entraîné uniquement sur les mois précédents.
 
### Tableau comparatif des scorers
 
| Scorer | Période | F1 avg | F1 OFF | F1 ON | Precision | Recall | Profit Net |
|--------|---------|--------|--------|-------|-----------|--------|------------|
| `lightgbm` | 2022–2024 (walk-forward) | **0.187** | 0.190 | 0.184 | **49.4%** | **11.5%** | **5,343,589** |
| `activation_level` | 2020–2022 (train) | 0.134 | 0.136 | 0.131 | 36.0% | 8.2% | 3,471,724 |
| `activation_level` | 2023 (val) | 0.097 | 0.098 | 0.096 | 20.2% | 6.4% | 132,999 |
| `historical_profit_rate` | 2020–2022 (train) | 0.048 | 0.049 | 0.046 | 13.2% | 2.9% | 783,256 |
 
### Pourquoi LightGBM domine
 
Le scorer `lightgbm` surpasse tous les autres sur les trois métriques pour deux raisons fondamentales :
 
1. **Combinaison multivarié de signaux.** Les heuristiques simples (`activation_level`, `historical_profit_rate`) utilisent une seule dimension. Le LightGBM exploite les 36 features simultanément et apprend des interactions non-linéaires — par exemple, un EID avec haute activation ET faible dispersion inter-scénarios ET un historique de profitabilité est beaucoup plus susceptible d'être profitable qu'un EID avec seulement une haute activation.
 
2. **Classifier vs ranking brut.** Le LightGBM Classifier optimise directement la frontière profitable/non-profitable, ce qui est aligné avec le F1-score. Les heuristiques produisent un score continu sans notion de seuil de décision.
 
### Analyse du recall structurellement plafonné
 
Le recall de 11.5% peut sembler faible, mais il est contraint structurellement. Avec ~400 opportunités profitables par mois dans l'univers complet et un maximum de 100 sélections autorisées, le recall théorique maximal est ~25%. Notre recall de 11.5% représente environ la moitié de ce maximum.
 
Le levier principal pour améliorer le F1 est la **précision** : chaque faux positif converti en vrai positif augmente simultanément la précision et le recall.
 
### Analyse SHAP — Interprétabilité du modèle
 
L'analyse SHAP (SHapley Additive exPlanations) révèle que le modèle combine trois familles de signaux complémentaires :
 
**Signaux historiques (dominants)**
- `hist_win_rate` et `hist_n_months` sont les deux features les plus importants. Les EID chroniquement profitables tendent à le rester, reflétant la nature structurelle de la congestion du réseau.
- `hist_consecutive_wins` est très discriminant : une longue série de mois profitables consécutifs est un signal fort de congestion persistante.
 
**Signaux de simulation (complémentaires)**
- `max_activation` est le 3ème feature le plus important, avec une relation inversée : une activation très élevée pousse vers NON profitable. Explication métier : les EID très activés sont déjà bien pricés par le marché — le coût d'exposition absorbe le signal.
- `sum_abs_psm_s2` et `n_hours_active` fournissent le signal prospectif des simulations mensuelles.
- `psm_cv_scenarios` (consensus inter-scénarios) agit comme filtre de qualité : à profit estimé égal, les EID où les 3 scénarios convergent sont 3× plus souvent profitables.
 
**Signal court-terme (affinement)**
- `daily_n_hours_active` confirme si la situation prédite par les simulations mensuelles est déjà observable dans les données récentes des 7 premiers jours du mois M.
 
### Note sur ACTIVATIONLEVEL
 
L'ACTIVATIONLEVEL est anti-prédictif lorsqu'utilisé seul comme heuristique de ranking (AUC = 0.424). Cependant, `max_activation` est le 3ème feature le plus important dans le modèle LightGBM. Cette apparente contradiction s'explique : en isolation, une haute activation ne prédit pas la profitabilité car le marché la price dans le coût. Mais dans un modèle multivarié, l'activation interagit avec d'autres features (coût proxy, historique, consensus) pour distinguer les EID où le marché a sous-estimé le signal de ceux où il l'a correctement anticipé.
 
### Évolution des itérations de modélisation
 
| Version | F1 | Profit | Changement clé |
|---------|-----|--------|----------------|
| Regressor, top-50, params défaut | 0.178* | 3,438K$ | Baseline LightGBM |
| Top-100 au lieu de top-50 | 0.253* | 4,503K$ | Recall doublé |
| Hyperparamètres optimisés | 0.261* | 5,239K$ | Grid search walk-forward |
| Classifier + features historiques | 0.444* | 5,258K$ | Signal historique ajouté |
| Features V3 (prunés) | 0.491* | 5,344K$ | Retrait de 7 features redondants |
| Évaluation univers complet (réel) | **0.187** | **5,344K$** | Recall mesuré sur ~167K triplets |
 
*F1 calculé sur l'univers filtré (surestimé). Le F1 réel sur l'univers complet est 0.187.
 
**Recommandation** : utiliser `--scorer lightgbm` pour les meilleures performances.
