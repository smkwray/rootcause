"""FBI Crime Data adapter.

Source
------
FBI Uniform Crime Reporting (UCR) Program / National Incident-Based
Reporting System (NIBRS), accessed via the Crime Data Explorer.
https://cde.ucr.cjis.gov/

**IMPORTANT:** These are RECORDED crimes -- offenses known to and logged by
participating law-enforcement agencies.  They are NOT a measure of total
victimization.  Under-reporting varies by crime type, jurisdiction, and
agency participation.  The UCR-to-NIBRS transition (completed 2021) creates
additional comparability challenges across years.

Update frequency: Annual (released ~fall for prior calendar year).
Geographic coverage: Participating agencies, aggregated to county.
Time coverage: UCR county-level data available ~1960-present.

Two-tier data strategy
----------------------
**Primary (state-level estimates):** The CDE API ``/estimate/state/``
endpoint provides FBI-adjusted crime estimates that account for
non-reporting agencies.  These are assigned to each state as a whole with
``county_fips`` set to ``{state_fips}000`` (state-total row).

**Fallback (county-level file):** A manually-placed county-level CSV with
the expected schema can be loaded via :meth:`load_county_file` for higher
geographic resolution (e.g., from Jacob Kaplan's concatenated UCR files).

Output Schema
-------------
county_fips          : str   -- 5-digit county FIPS code (zero-padded)
state_fips           : str   -- 2-digit state FIPS code (zero-padded)
year                 : int   -- Calendar year
violent_crime_count  : int   -- Total recorded violent crimes
property_crime_count : int   -- Total recorded property crimes
population_covered   : int   -- Population served by reporting agencies
agencies_reporting   : int   -- Number of agencies reporting in the county
reported_month_share : float -- Fraction of agency-months with data (0-1)
coverage_pass_flag   : bool  -- True if coverage meets quality threshold
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import requests

from povcrime.config import ProjectConfig
from povcrime.data.base import BaseAdapter

logger = logging.getLogger(__name__)

# Expected output columns.
EXPECTED_COLUMNS: list[str] = [
    "county_fips",
    "state_fips",
    "year",
    "violent_crime_count",
    "property_crime_count",
    "population_covered",
    "agencies_reporting",
    "reported_month_share",
    "coverage_pass_flag",
]

# FBI CDE API configuration.
_CDE_API_KEY = os.environ.get("FBI_CDE_API_KEY", "")
_CDE_BASE = "https://api.usa.gov/crime/fbi/cde"
_ESTIMATE_URL = (
    "{base}/estimate/state/{state_abbrev}/{offense}"
    "?from={from_year}&to={to_year}&API_KEY={api_key}"
)
_OFFENSE_TYPES = ("violent-crime", "property-crime")
_REQUEST_HEADERS = {
    "Accept": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
}
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_RETRY_SLEEP_SECONDS = 2.0
_MANUAL_FILE_CANDIDATES = (
    "county_crime.parquet",
    "county_crime.csv",
    "fbi_county_crime.parquet",
    "fbi_county_crime.csv",
)

# State abbreviations mapped to 2-digit FIPS codes (50 states + DC).
_STATE_FIPS: dict[str, str] = {
    "AL": "01",
    "AK": "02",
    "AZ": "04",
    "AR": "05",
    "CA": "06",
    "CO": "08",
    "CT": "09",
    "DE": "10",
    "DC": "11",
    "FL": "12",
    "GA": "13",
    "HI": "15",
    "ID": "16",
    "IL": "17",
    "IN": "18",
    "IA": "19",
    "KS": "20",
    "KY": "21",
    "LA": "22",
    "ME": "23",
    "MD": "24",
    "MA": "25",
    "MI": "26",
    "MN": "27",
    "MS": "28",
    "MO": "29",
    "MT": "30",
    "NE": "31",
    "NV": "32",
    "NH": "33",
    "NJ": "34",
    "NM": "35",
    "NY": "36",
    "NC": "37",
    "ND": "38",
    "OH": "39",
    "OK": "40",
    "OR": "41",
    "PA": "42",
    "RI": "44",
    "SC": "45",
    "SD": "46",
    "TN": "47",
    "TX": "48",
    "UT": "49",
    "VT": "50",
    "VA": "51",
    "WA": "53",
    "WV": "54",
    "WI": "55",
    "WY": "56",
}


class FBICrimeAdapter(BaseAdapter):
    """Adapter for FBI UCR / NIBRS recorded-crime data.

    Uses a **two-tier** approach:

    1. **Primary**: State-level estimated crime counts from the FBI Crime Data
       Explorer (CDE) API.  The CDE ``/estimate/state/`` endpoint provides
       FBI-adjusted estimates that account for non-reporting agencies.
       Because these are state-level, ``county_fips`` is set to
       ``{state_fips}000`` (a state-total pseudo-county code).
    2. **Fallback**: County-level data loaded from a manually-placed CSV via
       :meth:`load_county_file`.

    **Reminder:** All crime counts in this dataset are *recorded* crimes
    (offenses known to law enforcement), not total victimization.
    Under-reporting is a first-order concern for any analysis using these
    data.
    """

    def __init__(self, config: ProjectConfig) -> None:
        self._cfg = config
        self._raw_dir: Path = config.raw_dir / "fbi_crime"

    # ------------------------------------------------------------------ #
    # download
    # ------------------------------------------------------------------ #
    def download(self) -> None:
        """Download FBI crime data from the Crime Data Explorer API.

        For each state abbreviation, fetches the ``/estimate/state/`` endpoint
        for both ``violent-crime`` and ``property-crime`` offense types across
        the configured year range.  Raw JSON responses are saved to
        ``data/raw/fbi_crime/{state_abbrev}_{offense}.json``.

        Files that already exist are skipped (idempotent).
        """
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        session = requests.Session()
        n_downloaded = 0

        from_year = self._cfg.start_year
        to_year = self._cfg.end_year

        for state_abbrev in _STATE_FIPS:
            for offense in _OFFENSE_TYPES:
                out_path = self._raw_dir / f"{state_abbrev}_{offense}.json"
                if out_path.exists():
                    logger.info(
                        "FBI CDE %s %s already downloaded, skipping.",
                        state_abbrev,
                        offense,
                    )
                    continue

                url = _ESTIMATE_URL.format(
                    base=_CDE_BASE,
                    state_abbrev=state_abbrev,
                    offense=offense,
                    from_year=from_year,
                    to_year=to_year,
                    api_key=_CDE_API_KEY,
                )
                logger.info(
                    "Downloading FBI CDE estimate: %s %s (%d-%d) ...",
                    state_abbrev,
                    offense,
                    from_year,
                    to_year,
                )

                try:
                    resp = self._request_with_retries(session, url)
                    resp.raise_for_status()
                except requests.RequestException as exc:
                    logger.warning(
                        "Failed to download FBI CDE %s %s: %s. Skipping.",
                        state_abbrev,
                        offense,
                        exc,
                    )
                    continue

                try:
                    payload = resp.json()
                except ValueError as exc:
                    logger.warning(
                        "Invalid JSON for FBI CDE %s %s: %s. Skipping.",
                        state_abbrev,
                        offense,
                        exc,
                    )
                    continue

                # Persist raw response with state/offense metadata.
                wrapped = {
                    "state_abbrev": state_abbrev,
                    "state_fips": _STATE_FIPS[state_abbrev],
                    "offense_type": offense,
                    "from_year": from_year,
                    "to_year": to_year,
                    "api_response": payload,
                }
                with open(out_path, "w") as fh:
                    json.dump(wrapped, fh, indent=2)

                logger.info(
                    "Saved FBI CDE %s %s -> %s",
                    state_abbrev,
                    offense,
                    out_path,
                )
                n_downloaded += 1

        if n_downloaded == 0:
            logger.warning(
                "FBI CDE download produced no usable files. "
                "Place a county-level fallback file at %s. Expected schema: %s",
                self._raw_dir / _MANUAL_FILE_CANDIDATES[0],
                EXPECTED_COLUMNS,
            )

    def _request_with_retries(
        self,
        session: requests.Session,
        url: str,
    ) -> requests.Response:
        """Issue a GET request with small backoff for transient API failures."""
        last_response: requests.Response | None = None
        last_exc: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = session.get(
                    url,
                    timeout=60,
                    headers=_REQUEST_HEADERS,
                )
                if response.status_code not in _RETRYABLE_STATUS_CODES:
                    return response

                last_response = response
                logger.warning(
                    "FBI CDE request returned %d on attempt %d/%d.",
                    response.status_code,
                    attempt,
                    _MAX_RETRIES,
                )
            except requests.RequestException as exc:
                last_exc = exc
                logger.warning(
                    "FBI CDE request failed on attempt %d/%d: %s",
                    attempt,
                    _MAX_RETRIES,
                    exc,
                )

            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_SLEEP_SECONDS * attempt)

        if last_response is not None:
            return last_response
        if last_exc is not None:
            raise last_exc
        raise requests.RequestException("FBI CDE request failed unexpectedly.")

    # ------------------------------------------------------------------ #
    # load
    # ------------------------------------------------------------------ #
    def load(self) -> pd.DataFrame:
        """Load downloaded FBI CDE state-estimate JSON files.

        Reads all ``{state_abbrev}_{offense}.json`` files from the raw
        directory and builds a state-year DataFrame.

        Since the CDE estimates are state-level, ``county_fips`` is set to
        ``{state_fips}000`` (state-total code).  Coverage metrics
        ``reported_month_share`` and ``agencies_reporting`` are set to
        ``NaN`` because the CDE estimates are already adjusted for
        non-reporting.  ``coverage_pass_flag`` is ``True`` (the FBI
        estimation procedure accounts for missing data).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns defined in ``EXPECTED_COLUMNS``.
        """
        manual_path = self._find_manual_county_file()
        if manual_path is not None:
            logger.info(
                "Loading county-level FBI fallback file from %s.", manual_path
            )
            return self.load_county_file(manual_path)

        json_files = sorted(self._raw_dir.glob("*_violent-crime.json"))
        if not json_files:
            raise FileNotFoundError(
                f"No FBI crime raw files found in {self._raw_dir}. "
                "Run download() first or place a county-level fallback file "
                f"at one of: {', '.join(_MANUAL_FILE_CANDIDATES)}"
            )

        # Parse all violent-crime and property-crime files per state.
        # Build a dict keyed by (state_fips, year) -> row data.
        rows: dict[tuple[str, int], dict[str, Any]] = {}

        for fp in self._raw_dir.glob("*.json"):
            with open(fp) as fh:
                wrapped = json.load(fh)

            state_fips = wrapped["state_fips"]
            offense_type = wrapped["offense_type"]
            api_response = wrapped["api_response"]

            # The CDE API returns different shapes depending on the
            # endpoint version.  Handle the common patterns:
            #   1. dict with "results" key containing a list of year records
            #   2. list of year records directly
            #   3. dict with year keys mapping to values
            records = _extract_year_records(api_response)

            for rec in records:
                year = _extract_year(rec)
                if year is None:
                    continue

                key = (state_fips, year)
                if key not in rows:
                    rows[key] = {
                        "state_fips": state_fips,
                        "year": year,
                        "violent_crime_count": np.nan,
                        "property_crime_count": np.nan,
                        "population_covered": np.nan,
                    }

                count = _extract_crime_count(rec)
                population = _extract_population(rec)

                if offense_type == "violent-crime":
                    rows[key]["violent_crime_count"] = count
                elif offense_type == "property-crime":
                    rows[key]["property_crime_count"] = count

                # Population may appear in either response; take whichever
                # is non-null (they should agree).
                if population is not None and not np.isnan(population):
                    rows[key]["population_covered"] = population

        if not rows:
            raise ValueError(
                "No year-records could be parsed from the downloaded FBI CDE "
                f"files in {self._raw_dir}."
            )

        df = pd.DataFrame(list(rows.values()))

        # Construct county_fips as state-total pseudo-code.
        df["county_fips"] = df["state_fips"] + "000"

        # Coverage columns: NaN for estimated data (already adjusted).
        df["agencies_reporting"] = np.nan
        df["reported_month_share"] = np.nan
        df["coverage_pass_flag"] = True

        # Coerce types.
        for col in ("violent_crime_count", "property_crime_count"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["population_covered"] = pd.to_numeric(
            df["population_covered"], errors="coerce"
        )
        df["year"] = df["year"].astype(int)

        # Filter to the configured year range.
        df = df[
            (df["year"] >= self._cfg.start_year)
            & (df["year"] <= self._cfg.end_year)
        ].copy()

        # Reorder columns to match expected schema.
        df = df[EXPECTED_COLUMNS].reset_index(drop=True)
        return df

    def _find_manual_county_file(self) -> Path | None:
        """Return the first supported county-level fallback file if present."""
        for filename in _MANUAL_FILE_CANDIDATES:
            candidate = self._raw_dir / filename
            if candidate.exists():
                return candidate
        return None

    # ------------------------------------------------------------------ #
    # load_county_file
    # ------------------------------------------------------------------ #
    def load_county_file(self, path: str | Path) -> pd.DataFrame:
        """Load a manually-placed county-level crime CSV.

        This supports the fallback tier: a user-provided CSV (e.g., from
        Jacob Kaplan's concatenated UCR files or ICPSR downloads) that
        already has county-level crime counts.

        Parameters
        ----------
        path : str | Path
            Path to a CSV file that must contain at least the columns in
            ``EXPECTED_COLUMNS``.  Extra columns are ignored.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns defined in ``EXPECTED_COLUMNS``.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"County-level crime file not found: {path}"
            )

        if path.suffix.lower() == ".parquet":
            raw = pd.read_parquet(path)
            raw = raw.astype(
                {col: str for col in ["county_fips", "state_fips"] if col in raw}
            )
        else:
            raw = pd.read_csv(path, dtype=str)

        # Ensure required columns are present.
        missing = set(EXPECTED_COLUMNS) - set(raw.columns)
        if missing:
            raise ValueError(
                f"County-level CSV is missing required columns: {missing}. "
                f"Expected: {EXPECTED_COLUMNS}"
            )

        df = raw[EXPECTED_COLUMNS].copy()

        # Standardise FIPS codes.
        df["county_fips"] = df["county_fips"].str.zfill(5)
        df["state_fips"] = df["state_fips"].str.zfill(2)

        # Coerce numeric columns.
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
        for col in ("violent_crime_count", "property_crime_count"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["population_covered"] = pd.to_numeric(
            df["population_covered"], errors="coerce"
        )
        df["agencies_reporting"] = pd.to_numeric(
            df["agencies_reporting"], errors="coerce"
        )
        df["reported_month_share"] = pd.to_numeric(
            df["reported_month_share"], errors="coerce"
        )
        df["coverage_pass_flag"] = _coerce_bool_flag(df["coverage_pass_flag"])

        # Filter to configured year range.
        df = df[
            (df["year"] >= self._cfg.start_year)
            & (df["year"] <= self._cfg.end_year)
        ].copy()

        df = df.reset_index(drop=True)
        return df

    # ------------------------------------------------------------------ #
    # validate
    # ------------------------------------------------------------------ #
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate FBI crime DataFrame.

        Checks
        ------
        - All expected columns are present.
        - No duplicate (county_fips, year) rows.
        - Crime counts are non-negative.
        - reported_month_share is between 0 and 1 (where non-null).
        - population_covered is coerced to missing when non-positive.

        Returns
        -------
        pd.DataFrame
            The validated (and possibly cleaned) DataFrame.
        """
        # --- column presence ---
        missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"FBI crime DataFrame is missing columns: {missing_cols}"
            )

        # --- duplicates ---
        dup_mask = df.duplicated(subset=["county_fips", "year"], keep="first")
        n_dups = dup_mask.sum()
        if n_dups > 0:
            logger.warning(
                "Dropping %d duplicate (county_fips, year) rows.", n_dups
            )
            df = df[~dup_mask].copy()

        # --- crime counts non-negative ---
        for col in ("violent_crime_count", "property_crime_count"):
            neg = df[col].dropna() < 0
            if neg.any():
                logger.warning(
                    "Dropping %d rows with negative %s.", neg.sum(), col
                )
                df = df[~(df[col] < 0)].copy()

        # --- reported_month_share in [0, 1] (where non-null) ---
        rms = df["reported_month_share"].dropna()
        if len(rms) > 0:
            bad_rms = (rms < 0) | (rms > 1)
            if bad_rms.any():
                logger.warning(
                    "Dropping %d rows with reported_month_share outside "
                    "[0, 1].",
                    bad_rms.sum(),
                )
                bad_idx = bad_rms[bad_rms].index
                df = df.drop(bad_idx).copy()

        # County-level fallback files may not carry a clean covered-population
        # denominator. Preserve those rows and mark the denominator missing.
        bad_pop = df["population_covered"].notna() & (df["population_covered"] <= 0)
        if bad_pop.any():
            logger.warning(
                "Coercing %d rows with population_covered <= 0 to missing.",
                bad_pop.sum(),
            )
            df.loc[bad_pop, "population_covered"] = np.nan

        df = df.reset_index(drop=True)
        return df

    # ------------------------------------------------------------------ #
    # get_metadata
    # ------------------------------------------------------------------ #
    def get_metadata(self) -> dict[str, Any]:
        """Return metadata for the FBI crime data source."""
        return {
            "source_name": "FBI Crime Data Explorer (CDE)",
            "url": "https://cde.ucr.cjis.gov/",
            "api_endpoint": f"{_CDE_BASE}/estimate/state/",
            "data_level": "state-level estimates (county_fips = state + 000)",
            "update_frequency": "Annual",
            "geographic_coverage": "50 states + DC (state-level estimates)",
            "time_coverage": f"{self._cfg.start_year}-{self._cfg.end_year}",
            "expected_columns": EXPECTED_COLUMNS,
            "notes": (
                "Primary tier uses CDE state-level estimated crime counts "
                "which are adjusted by the FBI for non-reporting agencies. "
                "These are NOT raw agency counts. Coverage metrics "
                "(reported_month_share, agencies_reporting) are NaN for "
                "estimated data. Preferred county-year workflow uses a "
                "manual fallback file placed in data/raw/fbi_crime/ "
                "with one of the supported fallback names. Use "
                "load_county_file() for county-level data from external "
                "sources (e.g., Kaplan UCR files). "
                "All counts are RECORDED crimes only, not total "
                "victimization."
            ),
        }


# ====================================================================== #
# Private helpers for parsing CDE API responses
# ====================================================================== #


def _extract_year_records(api_response: Any) -> list[dict[str, Any]]:
    """Extract a list of year-level records from a CDE API response.

    The CDE API has returned different response shapes across versions.
    This function handles the known patterns:

    1. ``{"results": [...]}`` -- list of year-dicts under a "results" key.
    2. ``[{...}, {...}]`` -- bare list of year-dicts.
    3. ``{"2018": {...}, "2019": {...}}`` -- year keys at top level.
    """
    if isinstance(api_response, list):
        return api_response

    if isinstance(api_response, dict):
        # Pattern 1: "results" key.
        if "results" in api_response:
            results = api_response["results"]
            if isinstance(results, list):
                return results

        # Pattern 3: top-level year keys (numeric strings).
        year_records = []
        for key, val in api_response.items():
            if isinstance(key, str) and key.isdigit() and isinstance(val, dict):
                rec = {**val, "year": int(key)}
                year_records.append(rec)
        if year_records:
            return year_records

        # If the dict itself looks like a single record, wrap it.
        if "year" in api_response:
            return [api_response]

    return []


def _extract_year(record: dict[str, Any]) -> int | None:
    """Extract the year integer from a CDE record."""
    for key in ("year", "data_year", "Year", "DATA_YEAR"):
        if key in record:
            try:
                return int(record[key])
            except (ValueError, TypeError):
                continue
    return None


def _extract_crime_count(record: dict[str, Any]) -> float:
    """Extract the crime count from a CDE record.

    Tries several field names that the API has used across versions.
    Returns ``NaN`` if no count field is found.
    """
    for key in (
        "value",
        "count",
        "violent_crime",
        "property_crime",
        "total",
        "offense_count",
    ):
        if key in record:
            try:
                return float(record[key])
            except (ValueError, TypeError):
                continue
    return np.nan


def _extract_population(record: dict[str, Any]) -> float:
    """Extract population from a CDE record.

    Returns ``NaN`` if no population field is found.
    """
    for key in ("population", "state_population", "pop", "Population"):
        if key in record:
            try:
                return float(record[key])
            except (ValueError, TypeError):
                continue
    return np.nan


def _coerce_bool_flag(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y"})
