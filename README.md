# Rootcause

[![CI](https://github.com/smkwray/rootcause/actions/workflows/ci.yml/badge.svg)](https://github.com/smkwray/rootcause/actions/workflows/ci.yml)

**Economic Policy, Poverty, and Crime in U.S. Counties, 2000--2024**

A county-level empirical study examining the relationships between economic policy, poverty, and recorded crime. Tests whether changes in minimum wage, state earned income tax credits (EITC), SNAP eligibility rules, and TANF benefit levels affect crime rates -- and whether poverty and crime reinforce each other.

**[View the results site](https://smkwray.github.io/rootcause/)**

---

## What This Project Asks

Does economic policy affect crime? Does poverty cause crime -- or does crime deepen poverty? This project tests both directions using 25 years of county-level U.S. data and multiple causal inference methods. Policy-to-crime lanes test whether minimum wage, EITC, SNAP, and TANF changes affect recorded crime. A separate bidirectional analysis places poverty-to-crime and crime-to-poverty on equal methodological footing. Each lane is evaluated through a systematic credibility framework that includes conventional fixed effects, machine-learning estimators, border-county designs, staggered-adoption checks, and falsification tests.

## Headline Findings

### Policy and Crime

- **No single policy lever shows a clear, consistent causal effect on recorded crime** across all estimation methods.
- **Minimum wage** is the primary research lane but shows mixed signals: TWFE is positive and insignificant for both crime types; DML is marginally significant but switches sign for property crime (negative) vs violent crime (positive). Border-county designs do not confirm either direction.
- **State EITC** results are method-sensitive. Pre-trends pass, but TWFE is weak while DML is marginal for violent crime. Support trimming attenuates the signal.
- **SNAP BBCE** and **TANF** lanes remain exploratory due to failed pre-trends and poor covariate overlap, respectively.
- **Tighter identification designs** (border-county comparisons, staggered adoption, support trimming) tend to weaken rather than strengthen initial broad-sample estimates.

### Poverty and Crime: Both Directions

The project tests whether poverty causes crime *and* whether crime deepens poverty, using the same panel, estimators, and diagnostics for both directions.

- **Poverty to crime:** DML finds significant positive associations for both violent and property crime. FE is mixed (significant for property, not violent). However, the poverty-to-property placebo lead is also significant, raising concern about trending rather than causation. Covariate imbalance is high (max SMD > 2).
- **Crime to poverty:** DML finds small but statistically significant positive associations in both directions. FE is significant only for property-to-poverty. The coefficients are near zero in practical terms (crime rate per 100,000 moving poverty rate in percentage points). Placebo-lead concerns apply to property crime as well.
- **Neither direction passes the full credibility battery.** Both suffer from high covariate imbalance and/or significant placebo leads, preventing causal claims.

## Data Sources

All data comes from publicly available U.S. federal sources:

| Source | Variables | Coverage |
|--------|-----------|----------|
| Census SAIPE | Poverty rate, median household income, population | 100% |
| BLS Local Area Unemployment Statistics | Unemployment rate | 99.7% |
| BEA Regional Economic Accounts | Per-capita personal income | 98.3% |
| Census County Business Patterns | Establishments, employment, payroll | 95.9% |
| HUD Fair Market Rents | Rent levels by bedroom count | 99.5% |
| FHFA House Price Index | County-level house price indices | 86.5% |
| American Community Survey 5-Year | Race/ethnicity, age, education demographics | 64.0% |
| UKCPR Welfare Rules Database | State EITC rates, TANF benefit levels | 96.0% |
| FBI Uniform Crime Reporting | Violent and property crime counts and rates | 96.4% |
| State Policy Variables | Effective minimum wage, SNAP BBCE status | 100% |

The panel contains **78,529 county-year observations** across **3,158 counties** from **2000 to 2024**.

## Methods

The analysis applies multiple estimation strategies to each policy lane:

- **Two-Way Fixed Effects (TWFE):** County and year fixed effects with event-study pre-trend tests.
- **Double/Debiased Machine Learning (DML):** Cross-fitted histogram gradient boosting nuisance models partial out high-dimensional confounders before estimating treatment effects. County-year analyses use county-grouped sample splitting and a two-way within transform over county and year before nuisance fitting, so the same county does not appear in both train and test folds and the ML stage is panel-aware by default.
- **Border-County Design:** Adjacent cross-state county pairs compared within pairs over time.
- **Staggered-Adoption Estimator:** Stacked not-yet-treated approach to avoid heterogeneous timing bias.
- **Robustness:** Population weighting, county detrending, strict coverage filters, placebo leads, support-trimmed samples.
- **Falsification:** Negative-control outcomes (slow-moving demographics) test for residual confounding.

The authoritative lane registry, panel metadata, titles, tiers, treatments, and outcomes live in [`configs/project.yaml`](configs/project.yaml). That now covers both the main policy-to-crime lanes and the exploratory bidirectional poverty/crime lanes. Backend scripts and report builders read those definitions through the typed loader in [`src/povcrime/config.py`](src/povcrime/config.py).

## Credibility Framework

Every policy lane is evaluated against a battery of checks:

| Check | What It Tests |
|-------|---------------|
| Event-study pre-trends | Whether parallel trends hold before treatment |
| Covariate overlap | Balance between treated and control groups |
| Support trimming | Sensitivity to propensity-score overlap |
| Border-county design | Whether local comparisons reproduce national estimates |
| Negative-control outcomes | Whether the policy also "affects" outcomes it shouldn't |
| Staggered-adoption ATT | Robustness to heterogeneous treatment timing |

Currently, **no policy lane passes all checks**. Every lane carries a "caution" verdict.

## Limitations

This study has fundamental constraints that no amount of robustness checking can fully resolve:

- **No external crime benchmark.** The FBI UCR program is the sole crime data source. There is no second county-year crime measure (e.g., victimization surveys, 911 call data) to validate that patterns reflect actual crime rather than reporting or participation changes. Coverage-sensitivity checks are used instead, but they cannot rule out measurement-driven results.
- **Unconfoundedness is assumed, not tested.** DML removes observable confounders via machine learning, but state-level policy choices correlate with unmeasured characteristics (political environment, institutional capacity, cultural factors) that also affect crime. Fixed effects absorb time-invariant county traits and common year shocks, but slow-moving unobserved trends remain a concern — as the negative-control tests confirm.
- **ACS controls are pooled 5-year estimates.** Demographic variables (race, age, education) come from ACS 5-year pooled estimates designed for small-area reliability, not precise annual measurement. Using them as year-by-year DML confounders is methodologically loose, and they are unavailable before 2005.
- **Staggered treatment timing is not fully resolved.** The staggered-adoption estimator returns "not interpretable" for minimum wage, meaning pre-period coefficients are absorbed. Standard TWFE estimates may carry Goodman-Bacon heterogeneous-timing bias that is acknowledged but not decomposed.
- **Crime data reflects recorded crime, not actual crime.** FBI UCR reporting practices vary across agencies and over time. Changes in reporting can mimic changes in crime.
- **Results are sensitive to estimator choice.** TWFE and DML disagree on sign, magnitude, and significance for the same policy lane. This is transparently reported but limits the strength of any conclusion.
- **Border design is underpowered.** Only 798 cross-state adjacent county pairs are available, limiting statistical power for detecting moderate effects.

## Backend Quickstart

Install the project and development tools:

```bash
python -m pip install -e .[dev]
```

Run a fast local verification pass:

```bash
make smoke
```

`make smoke` includes a fixture-backed script smoke test that runs the real `build_panel -> build_qa_report -> build_app_artifacts -> build_final_report -> refresh_public_data` chain against tiny local fixtures with no network access.

Run the standard backend pipeline:

```bash
make full
```

If you want the steps individually:

```bash
make download
make build
make qa
make baseline
make dml
make overlap
make report
make app
make public-data
```

Notes:

- `make download` now includes the normal FBI county-fallback path when county-level crime needs to be constructed from the raw FBI files.
- `make build` requires county-level FBI crime by default for the county-year panel. Use `python scripts/build_panel.py --allow-missing-county-crime` only for debugging or partial builds.
- `make fbi-county-fallback` is available if you want to build the county-level FBI fallback explicitly.

## Viewing the Site

The static results site is published at **[smkwray.github.io/rootcause](https://smkwray.github.io/rootcause/)**.

To view locally:

```bash
cd docs
python3 -m http.server
# Open http://localhost:8000
```

## Refreshing the Public Data

The site reads from a sanitized data snapshot at `docs/assets/data/site_data.json`. To update it after re-running the backend pipeline:

```bash
make public-data
```

This reads `outputs/app/results_summary.json` and `outputs/app/credibility_summary.json`, strips internal file paths, and writes the public snapshot.

---
