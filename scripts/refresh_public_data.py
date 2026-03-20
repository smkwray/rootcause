#!/usr/bin/env python3
"""Refresh the public data snapshot for the GitHub Pages site.

Reads backend artifacts from outputs/app/ and writes a sanitized
JSON bundle to docs/assets/data/site_data.json.

Usage:
    python scripts/refresh_public_data.py
"""

import json
import math
import sys
from pathlib import Path

from povcrime.analysis import get_bidirectional_lane
from povcrime.config import get_config
from povcrime.reports.contracts import load_credibility_summary, load_results_summary

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs" / "app"
DEST = ROOT / "docs" / "assets" / "data"

SOURCE_LABELS = {
    "saipe": "Census SAIPE",
    "laus": "BLS Local Area Unemployment Statistics",
    "bea": "BEA Regional Economic Accounts",
    "acs": "American Community Survey 5-Year Estimates",
    "census_cbp": "Census County Business Patterns",
    "fhfa_hpi": "FHFA House Price Index",
    "hud_fmr": "HUD Fair Market Rents",
    "ukcpr_welfare": "UKCPR Welfare Rules Database",
    "fbi_crime": "FBI Uniform Crime Reporting Program",
    "policy": "State Policy Variables",
}

SOURCE_DESCRIPTIONS = {
    "saipe": "County poverty rates, median household income, population",
    "laus": "County unemployment rates",
    "bea": "Per-capita personal income",
    "acs": "Race/ethnicity, age, education demographics",
    "census_cbp": "Establishment counts, employment, payroll",
    "fhfa_hpi": "County-level house price indices",
    "hud_fmr": "Fair market rent by bedroom count, rent-to-income ratio",
    "ukcpr_welfare": "State EITC rates, TANF benefit levels",
    "fbi_crime": "Violent and property crime counts and rates",
    "policy": "Effective minimum wage, SNAP broad-based categorical eligibility",
}

TIER_LABELS = {
    "primary": "Primary",
    "secondary": "Secondary",
    "exploratory": "Exploratory",
}

STATUS_DESCRIPTIONS = {
    "primary_mixed_signal": "Mixed signals across estimation methods",
    "secondary_method_sensitive": "Results shift between estimators",
    "exploratory_failed_pretrends": "Event-study pre-trends fail",
    "exploratory_low_signal": "Low statistical signal",
}


def build_site_data():
    config = get_config()
    results_path = OUTPUTS / "results_summary.json"
    credibility_path = OUTPUTS / "credibility_summary.json"

    for p in [results_path, credibility_path]:
        if not p.exists():
            print(f"Missing required file: {p}", file=sys.stderr)
            sys.exit(1)

    results = load_results_summary(results_path)
    credibility = load_credibility_summary(credibility_path)

    # Build credibility lookup by slug
    cred_lookup = {lane["slug"]: lane for lane in credibility.get("lanes", [])}

    # Panel overview
    panel = results["panel"]
    sources = []
    for src in panel.get("available_sources", []):
        name = src["name"]
        sources.append(
            {
                "id": name,
                "label": SOURCE_LABELS.get(name, name),
                "description": SOURCE_DESCRIPTIONS.get(name, ""),
                "columns": src["columns"],
                "coverage_pct": round(src["share"] * 100, 1),
            }
        )

    # Determine tier for each lane
    research_lanes = results.get("research_lanes", {})
    tier_lookup = {}
    for tier, slugs in research_lanes.items():
        for slug in slugs:
            tier_lookup[slug] = tier

    # Build lane results
    lanes = []
    for est in results.get("estimands", []):
        slug = est["slug"]
        lane = {
            "slug": slug,
            "title": est["title"],
            "tier": tier_lookup.get(slug, "exploratory"),
            "tier_label": TIER_LABELS.get(tier_lookup.get(slug, "exploratory"), "Exploratory"),
        }

        # Baseline TWFE
        if "baseline" in est:
            bl = est["baseline"]
            tr = bl.get("treatment_row", {})
            lane["twfe"] = {
                "coefficient": round(tr.get("coefficient", 0), 4),
                "std_error": round(tr.get("std_error", 0), 4),
                "p_value": round(tr.get("p_value", 1), 4),
                "ci_lower": round(tr.get("ci_lower", 0), 2),
                "ci_upper": round(tr.get("ci_upper", 0), 2),
            }
            pt = bl.get("pretrend", {})
            if pt:
                p_val = pt.get("p_value")
                if p_val is not None and isinstance(p_val, float) and math.isnan(p_val):
                    p_val = None
                lane["pretrend"] = {
                    "p_value": p_val,
                    "pass": pt.get("pass"),
                }

        # DML
        if "dml" in est and est["dml"] is not None:
            d = est["dml"]
            lane["dml"] = {
                "theta": round(d.get("theta", 0), 4),
                "se": round(d.get("se", 0), 4),
                "p_value": round(d.get("p_value", 1), 4),
                "ci_lower": round(d.get("ci_lower", 0), 2),
                "ci_upper": round(d.get("ci_upper", 0), 2),
            }

        # Credibility checks
        if slug in cred_lookup:
            cred = cred_lookup[slug]
            lane["credibility"] = {
                "verdict": cred.get("verdict", "unknown"),
                "frontend_status": cred.get("frontend_status", ""),
                "status_description": STATUS_DESCRIPTIONS.get(
                    cred.get("frontend_status", ""), ""
                ),
                "headline_eligible": cred.get("headline_eligible", False),
                "checks": [
                    {
                        "name": c["name"],
                        "status": c["status"],
                        "detail": c["detail"],
                    }
                    for c in cred.get("checks", [])
                ],
            }

        lanes.append(lane)

    # Bidirectional poverty-crime (exploratory)
    exploratory = results.get("exploratory", {}) or {}
    bidir = exploratory.get("bidirectional_poverty_crime") or {}
    bidir_public = []
    for est in bidir.get("estimands", []):
        lane = get_bidirectional_lane(est["label"], config=config)
        fe = est.get("baseline_fe", {})
        dml = est.get("dml", {})
        ovlp = est.get("overlap", {})

        entry = {
            "label": est["label"],
            "title": lane.title if lane is not None else est["title"],
            "treatment": lane.treatment if lane is not None else est.get("treatment", ""),
            "outcome": lane.outcome if lane is not None else est.get("outcome", ""),
            "headline": est.get("headline", ""),
            "fe_coefficient": round(fe.get("coefficient", 0), 4),
            "fe_p_value": round(fe.get("p_value", 1), 4),
            "fe_ci_lower": round(fe.get("ci_lower", 0), 2),
            "fe_ci_upper": round(fe.get("ci_upper", 0), 2),
            "fe_n_obs": fe.get("n_obs_used"),
            "dml_theta": round(dml.get("theta", 0), 4),
            "dml_p_value": round(dml.get("p_value", 1), 4),
            "dml_ci_lower": round(dml.get("ci_lower", 0), 2),
            "dml_ci_upper": round(dml.get("ci_upper", 0), 2),
        }

        if ovlp:
            entry["overlap"] = {
                "max_abs_smd": round(ovlp.get("max_abs_smd", 0), 2),
                "oof_r2": round(ovlp.get("oof_r2", 0), 2),
                "residual_to_treatment_std": round(
                    ovlp.get("residual_to_treatment_std", 0), 2
                ),
            }

        # Extract placebo lead from robustness
        for rob in est.get("robustness", []):
            if rob.get("spec") == "placebo_lead1_high_coverage":
                entry["placebo_lead"] = {
                    "coefficient": round(rob.get("coefficient", 0), 4),
                    "p_value": round(rob.get("p_value", 1), 4),
                }
                break

        bidir_public.append(entry)

    site_data = {
        "generated_date": results.get("generated_date", ""),
        "panel": {
            "rows": panel["rows"],
            "counties": panel["counties"],
            "year_min": panel["year_min"],
            "year_max": panel["year_max"],
            "years": panel["year_max"] - panel["year_min"] + 1,
            "violent_rows": panel["violent_rows"],
            "property_rows": panel["property_rows"],
            "crime_data_level": panel.get("crime_data_level", "missing"),
        },
        "sources": sources,
        "lanes": lanes,
        "bidirectional": bidir_public,
    }

    data_json = json.dumps(site_data, indent=2)

    DEST.mkdir(parents=True, exist_ok=True)
    out_path = DEST / "site_data.json"
    out_path.write_text(data_json + "\n")
    print(f"Wrote {out_path.relative_to(ROOT)}")

    print("Site shell reads docs/assets/data/site_data.json directly.")


if __name__ == "__main__":
    build_site_data()
