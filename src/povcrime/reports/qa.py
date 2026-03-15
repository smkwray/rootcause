"""Data-quality report generation for the county-year panel."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from povcrime.processing.coverage import coverage_summary


def build_data_quality_report(
    panel: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Write a markdown QA report for the analysis panel."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# Data Quality Report",
        "",
        "## Panel Overview",
        f"- Rows: {len(panel):,}",
        f"- Counties: {panel['county_fips'].nunique():,}",
        f"- Years: {panel['year'].min()}-{panel['year'].max()}",
        f"- Columns: {len(panel.columns)}",
    ]

    if "low_coverage" in panel.columns:
        n_low = int(panel["low_coverage"].sum())
        lines.append(
            f"- Low-coverage rows: {n_low:,} "
            f"({100 * n_low / max(len(panel), 1):.1f}%)"
        )

    lines.extend(
        [
            "",
            "## Missingness Snapshot",
            "| Column | Missing Rows | Missing Share |",
            "|--------|--------------|---------------|",
        ]
    )

    missing = (
        panel.isna()
        .sum()
        .sort_values(ascending=False)
        .rename("missing_rows")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    missing["missing_share"] = missing["missing_rows"] / max(len(panel), 1)

    for _, row in missing.head(12).iterrows():
        lines.append(
            f"| {row['column']} | {int(row['missing_rows']):,} | "
            f"{100 * row['missing_share']:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Coverage Comparison",
            "| Sample | Rows | Counties | Mean Source Share | Mean Control Completeness |",
            "|--------|------|----------|-------------------|---------------------------|",
        ]
    )

    for label, frame in _sample_frames(panel):
        source_share = frame["source_share"].mean() if "source_share" in frame else float("nan")
        control = (
            frame["control_completeness"].mean()
            if "control_completeness" in frame
            else float("nan")
        )
        lines.append(
            f"| {label} | {len(frame):,} | {frame['county_fips'].nunique():,} | "
            f"{source_share:.3f} | {control:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Treatment Availability",
            f"- Minimum wage rows available: {_non_null_count(panel, 'effective_min_wage'):,}",
            f"- SNAP BBCE rows available: {_non_null_count(panel, 'broad_based_cat_elig'):,}",
            f"- Violent crime rows available: {_non_null_count(panel, 'violent_crime_count'):,}",
            f"- Property crime rows available: {_non_null_count(panel, 'property_crime_count'):,}",
        ]
    )

    min_wage_events = _count_min_wage_events(panel)
    if min_wage_events is not None:
        lines.append(f"- State-year minimum-wage changes: {min_wage_events:,}")

    snap_treated = _treated_rows(panel, "broad_based_cat_elig")
    if snap_treated is not None:
        lines.append(f"- SNAP BBCE treated rows: {snap_treated:,}")

    if {"state_fips", "year", "source_share"}.issubset(panel.columns):
        lines.extend(
            [
                "",
                "## Lowest-Coverage State-Years",
                "| State FIPS | Year | Counties | Mean Source Share | Low-Coverage Rows |",
                "|------------|------|----------|-------------------|-------------------|",
            ]
        )
        summary = coverage_summary(panel).sort_values(
            ["mean_source_share", "n_low_coverage", "state_fips", "year"],
            ascending=[True, False, True, True],
        )
        for _, row in summary.head(10).iterrows():
            low_rows = int(row["n_low_coverage"]) if "n_low_coverage" in row else 0
            lines.append(
                f"| {row['state_fips']} | {int(row['year'])} | "
                f"{int(row['n_counties']):,} | {row['mean_source_share']:.3f} | "
                f"{low_rows:,} |"
            )

    if _non_null_count(panel, "violent_crime_count") == 0:
        lines.extend(
            [
                "",
                "## Blocking Issue",
                "- Crime outcomes are currently absent from the panel. "
                "Baseline and DML estimation should remain blocked until a "
                "county-level FBI fallback file or another county-level crime "
                "source is loaded.",
            ]
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _sample_frames(panel: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Return the report's sample comparison frames."""
    frames = [("Full sample", panel)]
    if "low_coverage" in panel.columns:
        frames.append(("High coverage only", panel.loc[~panel["low_coverage"]]))
    return frames


def _non_null_count(panel: pd.DataFrame, column: str) -> int:
    """Return the non-null count for a column or zero when unavailable."""
    if column not in panel.columns:
        return 0
    return int(panel[column].notna().sum())


def _treated_rows(panel: pd.DataFrame, column: str) -> int | None:
    """Count treated rows for a binary treatment indicator."""
    if column not in panel.columns:
        return None
    return int((panel[column] == 1).sum())


def _count_min_wage_events(panel: pd.DataFrame) -> int | None:
    """Count state-year minimum wage changes."""
    required = {"state_fips", "year", "effective_min_wage"}
    if not required.issubset(panel.columns):
        return None

    state_year = (
        panel.groupby(["state_fips", "year"])["effective_min_wage"]
        .first()
        .reset_index()
        .sort_values(["state_fips", "year"])
    )
    changes = state_year.groupby("state_fips")["effective_min_wage"].diff()
    return int((changes.fillna(0).abs() > 0.01).sum())
