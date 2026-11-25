# rootcause

**[View the interactive results site](https://smkwray.github.io/rootcause/)**

`rootcause` is a county-level study of economic policy, poverty, and recorded crime in the United States, covering 3,158 counties from 2000 to 2024 (78,529 county-years). It tests whether changes in the minimum wage, state Earned Income Tax Credits (EITC), broad-based categorical eligibility (BBCE) for the Supplemental Nutrition Assistance Program (SNAP), and Temporary Assistance for Needy Families (TANF) benefit levels affect recorded crime, and whether poverty and crime feed each other.

No policy shows a consistent effect on crime across estimators, and tighter designs weaken the broad-sample estimates. Neither poverty-to-crime nor crime-to-poverty passes the full set of credibility checks. Every analysis currently carries a "caution" verdict, so none of the results supports a causal claim.

## Headline findings

Policy and crime:

- Minimum wage, the primary analysis, gives mixed signals. Two-way fixed effects (TWFE) estimates are positive and not significant for violent and property crime. Double/debiased machine learning (DML) estimates are marginally significant but negative for property crime and positive for violent crime. Border-county comparisons confirm neither direction.
- State EITC results depend on the estimator. Pre-trends pass, TWFE is weak, and DML is marginal for violent crime; trimming to the region of common support shrinks the violent-crime estimate.
- SNAP BBCE fails the pre-trend test and TANF has poor covariate overlap, so both remain exploratory.

Poverty and crime, estimated both ways with the same panel and methods:

- Poverty to crime: DML finds significant positive associations with violent and property crime, and fixed effects find one for property crime only. A significant placebo lead for property crime points to shared trends rather than causation, and covariate balance is poor (maximum standardized mean difference above 2).
- Crime to poverty: DML finds small, significant positive associations for both crime types, and fixed effects find one for property crime only. The effects are near zero in practical terms, and the property-crime placebo lead is also significant.

## Method and credibility checks

Each analysis is estimated with TWFE (county and year fixed effects, with event-study pre-trend tests) and with cross-fitted DML using gradient-boosting nuisance models, county-grouped sample splits, and a county-and-year within transform. Each then faces the same checks:

| Check | What it tests |
|---|---|
| Event-study pre-trends | Whether treated and comparison counties trended alike before treatment |
| Covariate overlap | Balance between treated and comparison counties |
| Support trimming | Sensitivity to dropping counties with poor propensity-score overlap |
| Border-county pairs | Whether adjacent cross-state counties reproduce the national estimate |
| Negative-control outcomes | Whether the policy also "moves" slow demographic outcomes it should not affect |
| Staggered-adoption ATT | Average treatment effect on the treated, using not-yet-treated counties as controls |

Robustness runs add population weighting, county detrending, strict coverage filters, and placebo leads. Analysis definitions live in [`configs/project.yaml`](configs/project.yaml).

## Data sources

All data are public. Coverage is the share of county-year rows with data from that source.

| Source | Variables | Coverage |
|---|---|---|
| Census Small Area Income and Poverty Estimates (SAIPE) | Poverty rate, median household income, population | 100% |
| BLS Local Area Unemployment Statistics (LAUS) | Unemployment rate | 99.7% |
| BEA Regional Economic Accounts | Per-capita personal income | 98.3% |
| Census County Business Patterns | Establishments, employment, payroll | 95.9% |
| HUD Fair Market Rents | Rent by bedroom count | 99.5% |
| FHFA House Price Index | County house price indices | 86.5% |
| American Community Survey (ACS) 5-year | Race and ethnicity, age, education | 64.0% |
| University of Kentucky Center for Poverty Research (UKCPR) Welfare Rules Database | State EITC rates, TANF benefit levels | 96.0% |
| FBI Uniform Crime Reporting (UCR) Program | Violent and property crime counts and rates | 96.4% |
| State policy variables | Effective minimum wage, SNAP BBCE status | 100% |

API keys are optional. To use them, copy `.env.example` to `.env` and fill in the keys you have.

## Reproduce

Requires Python 3.11 or later. The package is named `povcrime`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
make smoke     # fixture-based tests, no network
make full      # download, build the panel, QA, baseline, DML, overlap, report, site data
```

`make full` runs the core chain. The other checks have their own targets: `make robustness`, `make border`, `make support-trim`, `make falsification`, `make min-wage-id`, `make bidirectional`, `make crime-validation`, and `python scripts/run_staggered_att.py`. Run `make report app public-data` afterwards to include their results in the report and site. To preview the site locally, run `python3 -m http.server --directory docs` and open http://localhost:8000.

Outputs are written to `outputs/` (not committed). `make public-data` copies the site summary to `docs/assets/data/site_data.json` with local file paths removed.

## Limitations

- FBI UCR is the only crime source, so changes in reporting or agency participation can look like changes in crime.
- DML assumes no unobserved confounding, but state policy choices track unmeasured political and institutional factors; the negative-control failures suggest slow-moving trends remain.
- ACS controls are pooled 5-year estimates, unavailable before 2005, and are loose year-by-year controls.
- The staggered-adoption estimator is not interpretable for the minimum wage, and TWFE estimates may carry bias from staggered timing that is not decomposed.
- TWFE and DML disagree on sign, size, and significance for the same analysis.
- The border design has only 798 adjacent cross-state county pairs, too few to detect moderate effects.

## License

MIT. See [LICENSE](LICENSE).
