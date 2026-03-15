"""Data-source manifest for the povcrime project.

Each data source is represented by a :class:`SourceInfo` dataclass.
Use :func:`get_sources` to retrieve the full list and
:func:`export_manifest` to write it as JSON.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class SourceInfo:
    """Metadata for a single data source."""

    name: str
    url: str
    description: str
    frequency: str
    geography: str
    priority: int  # 1 = highest


_SOURCES: list[SourceInfo] = [
    SourceInfo(
        name="SAIPE",
        url="https://www.census.gov/programs-surveys/saipe.html",
        description=(
            "Small Area Income and Poverty Estimates — "
            "county-level poverty and median household income"
        ),
        frequency="annual",
        geography="county",
        priority=1,
    ),
    SourceInfo(
        name="ACS 5-Year",
        url="https://www.census.gov/programs-surveys/acs",
        description=(
            "American Community Survey 5-Year Estimates — "
            "demographic and socioeconomic characteristics"
        ),
        frequency="annual (5-year rolling)",
        geography="county",
        priority=1,
    ),
    SourceInfo(
        name="LAUS",
        url="https://www.bls.gov/lau/",
        description=(
            "Local Area Unemployment Statistics — "
            "county-level labor force, employment, and unemployment"
        ),
        frequency="monthly / annual",
        geography="county",
        priority=1,
    ),
    SourceInfo(
        name="BEA County Income",
        url="https://www.bea.gov/data/income-saving/personal-income-county-metro-and-other-areas",
        description=(
            "Bureau of Economic Analysis county-level personal income "
            "and per-capita income"
        ),
        frequency="annual",
        geography="county",
        priority=2,
    ),
    SourceInfo(
        name="Census County Business Patterns",
        url="https://www.census.gov/programs-surveys/cbp.html",
        description=(
            "County Business Patterns — county-level establishments, "
            "employment, and annual payroll"
        ),
        frequency="annual",
        geography="county",
        priority=2,
    ),
    SourceInfo(
        name="Census County Adjacency",
        url="https://www2.census.gov/geo/docs/reference/county_adjacency.txt",
        description=(
            "Public Census county adjacency text file for building cross-state "
            "border-county pair designs"
        ),
        frequency="periodic",
        geography="county-pair",
        priority=2,
    ),
    SourceInfo(
        name="HUD Fair Market Rents",
        url="https://www.huduser.gov/portal/datasets/fmr.html",
        description=(
            "HUD User Fair Market Rent history — county-level nominal "
            "rent benchmarks by bedroom size"
        ),
        frequency="annual",
        geography="county",
        priority=2,
    ),
    SourceInfo(
        name="FHFA County HPI",
        url="https://www.fhfa.gov/data/hpi/datasets",
        description=(
            "Federal Housing Finance Agency county-level house price indexes "
            "(all-transactions, annual)"
        ),
        frequency="annual",
        geography="county",
        priority=2,
    ),
    SourceInfo(
        name="USDA SNAP Policy",
        url="https://www.ers.usda.gov/data-products/snap-policy-database/",
        description=(
            "SNAP Policy Database — state-level SNAP eligibility rules, "
            "benefit levels, and waivers"
        ),
        frequency="annual",
        geography="state",
        priority=2,
    ),
    SourceInfo(
        name="UKCPR National Welfare Data",
        url="https://ukcpr.org/resources/national-welfare-data",
        description=(
            "University of Kentucky Center for Poverty Research state panel "
            "covering TANF/AFDC benefit levels, EITC generosity, and related "
            "transfer-program variables"
        ),
        frequency="annual",
        geography="state",
        priority=2,
    ),
    SourceInfo(
        name="DOL Minimum Wage",
        url="https://www.dol.gov/agencies/whd/minimum-wage/state",
        description=(
            "Department of Labor state minimum wage data — "
            "historical state-level minimum wage rates"
        ),
        frequency="annual",
        geography="state",
        priority=2,
    ),
    SourceInfo(
        name="FBI Crime Data Explorer",
        url="https://cde.ucr.cjis.gov/",
        description=(
            "FBI Crime Data Explorer — agency- and county-level "
            "Uniform Crime Reporting (UCR) data"
        ),
        frequency="annual",
        geography="county / agency",
        priority=1,
    ),
]


def get_sources() -> list[SourceInfo]:
    """Return the list of registered data sources."""
    return list(_SOURCES)


def export_manifest(path: str | Path | None = None) -> Path:
    """Write the source manifest to a JSON file.

    Parameters
    ----------
    path : str | Path | None
        Destination file.  Defaults to ``outputs/source_manifest.json``
        relative to the project root.

    Returns
    -------
    Path
        The path the manifest was written to.
    """
    if path is None:
        from povcrime.config import get_config

        cfg = get_config()
        path = cfg.output_dir / "source_manifest.json"

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump([asdict(s) for s in _SOURCES], fh, indent=2)
    return out
