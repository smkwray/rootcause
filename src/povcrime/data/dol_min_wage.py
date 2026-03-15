"""DOL Minimum Wage adapter.

Source
------
U.S. Department of Labor, Wage and Hour Division -- State Minimum Wage Laws.
https://www.dol.gov/agencies/whd/minimum-wage/state

Historical state minimum-wage rates are compiled from DOL publications and
supplementary academic datasets.  Federal rates come from the FLSA history.

Update frequency: Irregular (when states enact changes).
Geographic coverage: All 50 states + DC (state-level).
Time coverage: 2000-2024.

Output Schema
-------------
state_fips         : str   -- 2-digit state FIPS code (zero-padded)
year               : int   -- Calendar year
state_min_wage     : float -- State-enacted minimum wage (USD, nominal)
federal_min_wage   : float -- Federal minimum wage (USD, nominal)
effective_min_wage : float -- max(state, federal) minimum wage (USD)
"""

from __future__ import annotations

import csv
import logging
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

from povcrime.config import ProjectConfig
from povcrime.data.base import BaseAdapter

logger = logging.getLogger(__name__)

# Expected output columns.
EXPECTED_COLUMNS: list[str] = [
    "state_fips",
    "year",
    "state_min_wage",
    "federal_min_wage",
    "effective_min_wage",
]

# ---------------------------------------------------------------------------
# Federal minimum wage history (FLSA).  For years between changes the
# previous value is carried forward.
# ---------------------------------------------------------------------------
_FEDERAL_WAGE_CHANGES: dict[int, float] = {
    1997: 5.15,
    2007: 5.85,
    2008: 6.55,
    2009: 7.25,
}


def _federal_min_wage(year: int) -> float:
    """Return the federal minimum wage in effect for *year*."""
    wage = 5.15  # default for years >= 1997
    for change_year in sorted(_FEDERAL_WAGE_CHANGES):
        if year >= change_year:
            wage = _FEDERAL_WAGE_CHANGES[change_year]
    return wage


# ---------------------------------------------------------------------------
# State name -> 2-digit FIPS mapping (50 states + DC)
# ---------------------------------------------------------------------------
STATE_NAME_TO_FIPS: dict[str, str] = {
    "Alabama": "01",
    "Alaska": "02",
    "Arizona": "04",
    "Arkansas": "05",
    "California": "06",
    "Colorado": "08",
    "Connecticut": "09",
    "Delaware": "10",
    "District of Columbia": "11",
    "Florida": "12",
    "Georgia": "13",
    "Hawaii": "15",
    "Idaho": "16",
    "Illinois": "17",
    "Indiana": "18",
    "Iowa": "19",
    "Kansas": "20",
    "Kentucky": "21",
    "Louisiana": "22",
    "Maine": "23",
    "Maryland": "24",
    "Massachusetts": "25",
    "Michigan": "26",
    "Minnesota": "27",
    "Mississippi": "28",
    "Missouri": "29",
    "Montana": "30",
    "Nebraska": "31",
    "Nevada": "32",
    "New Hampshire": "33",
    "New Jersey": "34",
    "New Mexico": "35",
    "New York": "36",
    "North Carolina": "37",
    "North Dakota": "38",
    "Ohio": "39",
    "Oklahoma": "40",
    "Oregon": "41",
    "Pennsylvania": "42",
    "Rhode Island": "44",
    "South Carolina": "45",
    "South Dakota": "46",
    "Tennessee": "47",
    "Texas": "48",
    "Utah": "49",
    "Vermont": "50",
    "Virginia": "51",
    "Washington": "53",
    "West Virginia": "54",
    "Wisconsin": "55",
    "Wyoming": "56",
}

# Reverse lookup (FIPS -> name) for convenience.
_FIPS_TO_STATE_NAME: dict[str, str] = {v: k for k, v in STATE_NAME_TO_FIPS.items()}

# DOL history table URL (primary scraping target).
_DOL_HISTORY_URL = (
    "https://www.dol.gov/agencies/whd/minimum-wage/state/history"
)

# Alternative URL patterns that DOL has used.
_DOL_ALT_URLS = [
    "https://www.dol.gov/agencies/whd/state/minimum-wage/history",
    "https://www.dol.gov/whd/minwage/america/newMinWageHist.htm",
]


def _parse_wage_cell(text: str) -> float:
    """Parse a wage value from a table cell, returning 0.0 for blanks or
    non-numeric entries (e.g. 'No minimum wage law')."""
    text = text.strip().replace("$", "").replace(",", "")
    # Handle annotations like "8.25(1)" or footnotes
    # Take only the first number-like token
    for token in text.split():
        cleaned = ""
        for ch in token:
            if ch.isdigit() or ch == ".":
                cleaned += ch
            else:
                break
        if cleaned:
            try:
                return float(cleaned)
            except ValueError:
                continue
    return 0.0


class DOLMinWageAdapter(BaseAdapter):
    """Adapter for DOL historical state minimum-wage data.

    The adapter tries to scrape the DOL state minimum wage history page.
    If that fails (the page structure changes often), it falls back to:
    1. A CSV file manually placed at ``data/raw/dol_min_wage/state_min_wage.csv``
    2. A hardcoded federal-only dataset (all states get the federal floor).
    """

    def __init__(self, config: ProjectConfig) -> None:
        self._config = config
        self._raw_dir = config.raw_dir / "dol_min_wage"

    # ------------------------------------------------------------------ #
    # download
    # ------------------------------------------------------------------ #
    def download(self) -> None:
        """Download or compile state minimum-wage data.

        Tries to scrape the DOL WHD state minimum wage history table.
        If the page is unavailable or has no parseable table, saves a
        placeholder CSV with federal-only data and logs a warning.

        Stores results in ``data/raw/dol_min_wage/``.
        """
        self._raw_dir.mkdir(parents=True, exist_ok=True)

        # Try primary and alternative URLs.
        for url in [_DOL_HISTORY_URL, *_DOL_ALT_URLS]:
            try:
                logger.info("Trying DOL URL: %s", url)
                resp = requests.get(url, timeout=30)
                if resp.status_code != 200:
                    logger.warning(
                        "DOL returned status %d for %s", resp.status_code, url
                    )
                    continue

                soup = BeautifulSoup(resp.text, "lxml")
                tables = soup.find_all("table")
                if not tables:
                    logger.warning("No tables found at %s", url)
                    continue

                # Save raw HTML for reproducibility.
                html_path = self._raw_dir / "dol_state_history.html"
                html_path.write_text(resp.text, encoding="utf-8")
                logger.info("Saved raw HTML to %s", html_path)

                # The DOL page splits history across multiple tables
                # (e.g. 1968-1981, 1988-1998, 2000-2006, ...).  Parse
                # all of them and concatenate.
                frames: list[pd.DataFrame] = []
                for tbl in tables:
                    parsed = self._parse_dol_table(tbl)
                    if parsed is not None and len(parsed) > 0:
                        frames.append(parsed)

                if frames:
                    df = pd.concat(frames, ignore_index=True)
                    # Keep only the latest entry per (state, year) in
                    # case tables overlap.
                    df = df.drop_duplicates(
                        subset=["state_fips", "year"], keep="last"
                    )
                    csv_path = self._raw_dir / "state_min_wage.csv"
                    df.to_csv(csv_path, index=False)
                    logger.info(
                        "Parsed %d DOL tables: %d rows -> %s",
                        len(frames),
                        len(df),
                        csv_path,
                    )
                    return

            except requests.RequestException as exc:
                logger.warning("Request to %s failed: %s", url, exc)

        # Fallback: generate federal-only dataset.
        logger.warning(
            "Could not scrape DOL table. Generating federal-only fallback CSV. "
            "For state-level data, manually place a CSV at %s "
            "with columns: state_name,year,state_min_wage",
            self._raw_dir / "state_min_wage.csv",
        )
        self._write_federal_fallback()

    def _parse_dol_table(self, table: Any) -> pd.DataFrame | None:
        """Parse an HTML ``<table>`` element from the DOL history page.

        The expected layout has states as rows and years as columns.
        Returns a long-format DataFrame or *None* on failure.
        """
        rows = table.find_all("tr")
        if len(rows) < 2:
            return None

        # Header: first cell is state name, rest are years.
        # Year headers may have annotations like "1968 (a)" -- extract
        # just the leading 4-digit year.
        header_cells = rows[0].find_all(["th", "td"])
        years: list[int] = []
        for cell in header_cells[1:]:
            txt = cell.get_text(strip=True)
            # Take the first whitespace-delimited token as the year.
            token = txt.split()[0] if txt.split() else txt
            try:
                years.append(int(token))
            except ValueError:
                continue

        if not years:
            return None

        records: list[dict[str, Any]] = []
        for row in rows[1:]:
            cells = row.find_all(["th", "td"])
            if not cells:
                continue
            state_name = cells[0].get_text(strip=True)
            fips = STATE_NAME_TO_FIPS.get(state_name)
            if fips is None:
                continue
            for i, yr in enumerate(years):
                cell_idx = i + 1
                if cell_idx < len(cells):
                    wage = _parse_wage_cell(cells[cell_idx].get_text(strip=True))
                else:
                    wage = 0.0
                records.append(
                    {
                        "state_fips": fips,
                        "year": yr,
                        "state_min_wage": wage,
                    }
                )

        if not records:
            return None
        return pd.DataFrame(records)

    def _write_federal_fallback(self) -> None:
        """Write a CSV with all states getting the federal minimum only."""
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self._raw_dir / "state_min_wage.csv"
        with open(csv_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["state_fips", "year", "state_min_wage"])
            for fips in sorted(STATE_NAME_TO_FIPS.values()):
                for yr in range(self._config.start_year, self._config.end_year + 1):
                    # State wage of 0 means no state law / below federal.
                    writer.writerow([fips, yr, 0.0])
        logger.info("Wrote federal-only fallback CSV to %s", csv_path)

    # ------------------------------------------------------------------ #
    # load
    # ------------------------------------------------------------------ #
    def load(self) -> pd.DataFrame:
        """Load minimum-wage data into a standardised DataFrame.

        Reads the CSV produced by :meth:`download` (or a manually placed
        CSV with columns ``state_fips, year, state_min_wage``), then adds
        federal and effective wage columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns defined in ``EXPECTED_COLUMNS``.
            ``effective_min_wage`` = max(state_min_wage, federal_min_wage).
        """
        csv_path = self._raw_dir / "state_min_wage.csv"
        if not csv_path.exists():
            msg = (
                f"Raw CSV not found at {csv_path}. "
                "Run download() first or place a CSV with columns: "
                "state_fips, year, state_min_wage"
            )
            raise FileNotFoundError(msg)

        df = pd.read_csv(csv_path, dtype={"state_fips": str})

        # Ensure state_fips is zero-padded to 2 digits.
        df["state_fips"] = df["state_fips"].str.zfill(2)

        # Filter to configured year range.
        df = df[
            (df["year"] >= self._config.start_year)
            & (df["year"] <= self._config.end_year)
        ].copy()

        # Add federal and effective wages.
        df["federal_min_wage"] = df["year"].apply(_federal_min_wage)
        df["effective_min_wage"] = df[["state_min_wage", "federal_min_wage"]].max(
            axis=1
        )

        # Select and order columns.
        df = df[EXPECTED_COLUMNS].sort_values(["state_fips", "year"]).reset_index(
            drop=True
        )
        return df

    # ------------------------------------------------------------------ #
    # validate
    # ------------------------------------------------------------------ #
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate minimum-wage DataFrame.

        Checks
        ------
        - All expected columns are present.
        - No duplicate (state_fips, year) rows.
        - effective_min_wage >= federal_min_wage for all rows.
        - All wage values are non-negative.

        Returns
        -------
        pd.DataFrame
            The validated DataFrame.

        Raises
        ------
        ValueError
            If any validation check fails.
        """
        # Column check.
        missing = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing:
            msg = f"Missing columns: {missing}"
            raise ValueError(msg)

        # Duplicate check.
        dupes = df.duplicated(subset=["state_fips", "year"], keep=False)
        if dupes.any():
            n_dupes = dupes.sum()
            msg = f"Found {n_dupes} duplicate (state_fips, year) rows"
            raise ValueError(msg)

        # Effective >= federal.
        bad_eff = df["effective_min_wage"] < df["federal_min_wage"]
        if bad_eff.any():
            n_bad = bad_eff.sum()
            msg = f"{n_bad} rows have effective_min_wage < federal_min_wage"
            raise ValueError(msg)

        # Non-negative wages.
        for col in ("state_min_wage", "federal_min_wage", "effective_min_wage"):
            neg = df[col] < 0
            if neg.any():
                msg = f"{neg.sum()} rows have negative {col}"
                raise ValueError(msg)

        logger.info("Validation passed: %d rows", len(df))
        return df

    # ------------------------------------------------------------------ #
    # get_metadata
    # ------------------------------------------------------------------ #
    def get_metadata(self) -> dict[str, Any]:
        """Return metadata for the DOL minimum-wage data source."""
        return {
            "source": "U.S. Department of Labor, Wage and Hour Division",
            "url": "https://www.dol.gov/agencies/whd/minimum-wage/state",
            "description": (
                "State and federal minimum wage history. State rates are "
                "from DOL publications; federal rates from FLSA history."
            ),
            "geographic_coverage": "All 50 states + DC (state-level)",
            "time_coverage": (
                f"{self._config.start_year}-{self._config.end_year}"
            ),
            "update_frequency": "Irregular",
            "output_columns": EXPECTED_COLUMNS,
        }
