"""Exploratory scaffold for reverse-direction analyses."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd


_REVERSE_SPECS = [
    {"treatment": "violent_crime_rate", "outcome": "poverty_rate", "label": "violent_to_poverty"},
    {"treatment": "property_crime_rate", "outcome": "poverty_rate", "label": "property_to_poverty"},
    {"treatment": "violent_crime_rate", "outcome": "unemployment_rate", "label": "violent_to_unemployment"},
    {"treatment": "property_crime_rate", "outcome": "unemployment_rate", "label": "property_to_unemployment"},
]


def build_reverse_direction_scaffold(
    *,
    panel: pd.DataFrame,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write an exploratory reverse-direction scaffold as markdown and JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = []
    for spec in _REVERSE_SPECS:
        cols = [spec["treatment"], spec["outcome"]]
        available = all(col in panel.columns for col in cols)
        usable = int(panel[cols].dropna().shape[0]) if available else 0
        specs.append(
            {
                **spec,
                "available": available,
                "usable_rows": usable,
            }
        )

    json_path = output_dir / "reverse_direction_specs.json"
    md_path = output_dir / "reverse_direction_scaffold.md"

    json_path.write_text(
        json.dumps(
            {
                "generated_date": date.today().isoformat(),
                "status": "exploratory_only",
                "notes": [
                    "These designs are intentionally exploratory and should not be presented as headline causal estimates without a separate identification argument.",
                    "Crime-to-poverty directions are vulnerable to simultaneity, lag structure ambiguity, and post-treatment control contamination.",
                ],
                "specs": specs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Reverse-Direction Exploratory Scaffold",
        "",
        f"Generated: {date.today().isoformat()}",
        "",
        "## Status",
        "",
        "- Scope: exploratory only.",
        "- Recommended use: backend planning / future extension, not frontend headline interpretation.",
        "",
        "## Candidate Specs",
        "",
        "| Label | Treatment | Outcome | Available | Usable Rows |",
        "|-------|-----------|---------|-----------|-------------|",
    ]
    for spec in specs:
        lines.append(
            f"| {spec['label']} | {spec['treatment']} | {spec['outcome']} | "
            f"{spec['available']} | {spec['usable_rows']:,} |"
        )
    lines.extend(
        [
            "",
            "## Design Warnings",
            "",
            "- Crime and poverty likely move jointly with shared shocks, so reverse-direction FE alone is not a strong design.",
            "- Lag structure needs explicit justification before estimation.",
            "- Some standard controls in the forward models may become post-treatment in the reverse direction and should be reconsidered.",
            "",
            "## Suggested Next Build Order",
            "",
            "1. Start with lagged crime predictors rather than contemporaneous crime rates.",
            "2. Use county and year fixed effects with a reduced control set audited for post-treatment risk.",
            "3. Treat any resulting estimates as exploratory until a clearer design is established.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path
