"""County-level FBI crime fallback from CDE Return A master files.

This module builds a county-year fallback file for the project when the
public FBI API is unavailable. It uses live official CDE downloads:

- annual Return A master files from ``cde.ucr.cjis.gov``
- live agency-by-state county mappings from ``/LATEST/agency/byStateAbbr/{state}``
- a legacy agency reference bundle from the FBI CDE frontend repo
- Census county code reference data

The parser extracts agency-year violent and property crime counts from the
annual Return A fixed-width master files, matches agencies to counties, and
aggregates to county-year. The annual master format does not expose a clean
official month-reporting indicator in this fallback path, so covered
population is approximated from stable agency-population header fields and
reporting-share fields are left missing.
"""

from __future__ import annotations

from collections import defaultdict
import gzip
import json
import logging
from pathlib import Path
import re
from typing import Any
from zipfile import ZipFile

import pandas as pd
import requests

from povcrime.data.fbi_crime import EXPECTED_COLUMNS, _STATE_FIPS

logger = logging.getLogger(__name__)

_SIGNED_URL_ENDPOINT = (
    "https://cde.ucr.cjis.gov/LATEST/s3/signedurl?key=nibrs/master/reta/reta-{year}.zip"
)
_LIVE_AGENCY_ENDPOINT = "https://cde.ucr.cjis.gov/LATEST/agency/byStateAbbr/{state}"
_LEGACY_AGENCY_BUNDLE_URL = (
    "https://raw.githubusercontent.com/fbi-cde/crime-data-frontend/master/"
    "public/data/agencies-by-state.json.gz"
)
_COUNTY_CODES_2020_URL = (
    "https://www2.census.gov/geo/docs/reference/codes2020/national_county2020.txt"
)

_HEADER_SLICE = slice(120, 240)
_CITY_SLICE = slice(120, 140)
_AGENCY_POP_PRIMARY_SLICE = slice(44, 53)
_AGENCY_POP_SECONDARY_SLICE = slice(89, 98)

# Annual Return A master-file block offsets discovered by matching against the
# FBI's own sample `agency_sums_view` dump.
_BLOCK_START = 305
_BLOCK_LEN = 590
_COUNT_POS = {
    "murder": 157,
    "rape_total": 167,
    "robbery_total": 182,
    "assault_gun": 212,
    "assault_cut": 217,
    "assault_other": 222,
    "assault_hff": 227,
    "burglary_total": 237,
    "larceny_total": 257,
    "mvt_total": 262,
}

_COUNTY_SUFFIXES = (
    " COUNTY",
    " PARISH",
    " BOROUGH",
    " CENSUS AREA",
    " MUNICIPALITY",
    " CITY AND BOROUGH",
    " CITY AND COUNTY",
    " CITY",
)
_NORMALIZATION_REPLACEMENTS = {
    "SHERIFF'S OFFICE": "SHERIFFSOFFICE",
    "SHERIFFS OFFICE": "SHERIFFSOFFICE",
    "POLICE DEPARTMENT": "POLICEDEPARTMENT",
    "POLICE DEPT.": "POLICEDEPARTMENT",
    "POLICE DEPT": "POLICEDEPARTMENT",
    "POLICE DPT": "POLICEDEPARTMENT",
    "STATE POLICE: ": "STATEPOLICE",
    "STATE POLICE ": "STATEPOLICE",
    "DEPARTMENT OF PUBLIC SAFETY": "DPS",
    "DEPARTMENT OF ENVIRONMENTAL MANAGEMENT": "ENVIRONMENTALMANAGEMENT",
    "UNIVERSITY OF ": "UNIVOF",
    "AIRPRT": "AIRPORT",
    "INTL": "INTERNATIONAL",
    "&": "AND",
    ":": "",
}
_COUNTY_ALIASES = {
    ("02", "VALDEZCORDOVA"): "02261",
    ("46", "SHANNON"): "46113",
}
_AGENCY_KEY_SLICE = slice(3, 10)


def build_county_fallback(
    *,
    start_year: int,
    end_year: int,
    raw_dir: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Build the county-year fallback file from official Return A downloads."""
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
        }
    )

    cache_dir = raw_dir / "reta_master"
    cache_dir.mkdir(parents=True, exist_ok=True)

    county_lookup = _load_county_lookup(session)
    agency_ref = _load_agency_reference(session, raw_dir)

    rows: list[dict[str, Any]] = []
    unmatched_rows = 0

    for year in range(start_year, end_year + 1):
        zip_path = _download_year_zip(session, year, cache_dir)
        logger.info("Parsing FBI Return A master file for %d from %s", year, zip_path)

        yearly_rows, yearly_unmatched = _parse_year(
            zip_path=zip_path,
            year=year,
            agency_ref=agency_ref,
            county_lookup=county_lookup,
        )
        rows.extend(yearly_rows)
        unmatched_rows += yearly_unmatched

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No county-year FBI rows could be parsed from Return A files.")

    agg = (
        df.groupby(["county_fips", "state_fips", "year"], as_index=False)
        .agg(
            violent_crime_count=("violent_crime_count", "sum"),
            property_crime_count=("property_crime_count", "sum"),
            agencies_reporting=("agency_key", "nunique"),
            population_covered=("population_covered", lambda s: s.sum(min_count=1)),
            reported_month_share=("reported_month_share", "mean"),
        )
    )
    agg["coverage_pass_flag"] = False
    agg = agg[EXPECTED_COLUMNS].sort_values(["county_fips", "year"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".csv":
        agg.to_csv(output_path, index=False)
    else:
        agg.to_parquet(output_path, engine="pyarrow", index=False)

    logger.info(
        "Built FBI county fallback: %s rows across %s counties (%s unmatched agency rows).",
        len(agg),
        agg["county_fips"].nunique(),
        unmatched_rows,
    )
    return agg


def _parse_year(
    *,
    zip_path: Path,
    year: int,
    agency_ref: dict[str, dict[str, Any]],
    county_lookup: dict[tuple[str, str], str],
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    unmatched = 0

    with ZipFile(zip_path) as zf:
        member = zf.namelist()[0]
        with zf.open(member) as fh:
            for raw in fh:
                line = raw.decode("latin1").rstrip("\n\r")
                state_abbr = line[3:5]
                if state_abbr not in _STATE_FIPS:
                    continue

                county_fips, agency_key = _match_county_fips(
                    line=line,
                    state_abbr=state_abbr,
                    agency_ref=agency_ref,
                    county_lookup=county_lookup,
                )
                if county_fips is None:
                    unmatched += 1
                    continue

                violent = (
                    _annual_sum(line, _COUNT_POS["murder"])
                    + _annual_sum(line, _COUNT_POS["rape_total"])
                    + _annual_sum(line, _COUNT_POS["robbery_total"])
                    + _annual_sum(line, _COUNT_POS["assault_gun"])
                    + _annual_sum(line, _COUNT_POS["assault_cut"])
                    + _annual_sum(line, _COUNT_POS["assault_other"])
                    + _annual_sum(line, _COUNT_POS["assault_hff"])
                )
                property_count = (
                    _annual_sum(line, _COUNT_POS["burglary_total"])
                    + _annual_sum(line, _COUNT_POS["larceny_total"])
                    + _annual_sum(line, _COUNT_POS["mvt_total"])
                )
                agency_population = _extract_agency_population(line)

                rows.append(
                    {
                        "county_fips": county_fips,
                        "state_fips": _STATE_FIPS[state_abbr],
                        "year": year,
                        "violent_crime_count": violent,
                        "property_crime_count": property_count,
                        "population_covered": agency_population,
                        "agencies_reporting": 1,
                        "reported_month_share": float("nan"),
                        "coverage_pass_flag": False,
                        "agency_key": agency_key,
                    }
                )

    return rows, unmatched


def _match_county_fips(
    *,
    line: str,
    state_abbr: str,
    agency_ref: dict[str, dict[str, Any]],
    county_lookup: dict[tuple[str, str], str],
) -> tuple[str | None, str]:
    state_ref = agency_ref[state_abbr]
    header_norm = _normalize_text(line[_HEADER_SLICE])
    city_norm = _normalize_text(line[_CITY_SLICE])
    agency_key = line[_AGENCY_KEY_SLICE].strip()

    county_name_norm: str | None = None

    # 1. Direct match on the agency-family key derived from FBI ORIs.
    county_name_norm = state_ref["agency_key_to_county"].get(agency_key)

    # 2. Exact agency-name substring match against official FBI reference names.
    for name_norm, county_norm in state_ref["agency_names"]:
        if county_name_norm is None and name_norm and name_norm in header_norm:
            county_name_norm = county_norm
            break

    # 3. County appears directly in the header for sheriff/county agencies.
    if county_name_norm is None:
        for county_norm in state_ref["county_names"]:
            if county_norm and county_norm in header_norm:
                county_name_norm = county_norm
                break

    # 4. Unique city -> county match from the FBI reference sets.
    if county_name_norm is None and city_norm:
        candidates = state_ref["city_to_counties"].get(city_norm, set())
        if len(candidates) == 1:
            county_name_norm = next(iter(candidates))

    if county_name_norm is None:
        return None, agency_key or header_norm

    key = (_STATE_FIPS[state_abbr], county_name_norm)
    county_fips = county_lookup.get(key)
    if county_fips is None:
        county_fips = _COUNTY_ALIASES.get(key)
    return county_fips, agency_key or header_norm


def _annual_sum(line: str, pos: int) -> int:
    total = 0
    for month in range(12):
        block_start = _BLOCK_START + month * _BLOCK_LEN
        total += _parse_count_field(line[block_start + pos:block_start + pos + 5])
    return total


def _extract_agency_population(line: str) -> int | None:
    """Extract a conservative agency-population proxy from the header.

    The annual Return A master records expose two stable 9-digit population-like
    header fields. Their exact semantics differ across agency types, but both
    track official FBI agency population closely in aggregate. We use the
    smaller non-zero value to avoid systematic overstatement when the fields
    disagree.
    """
    candidates = [
        _parse_header_int(line[_AGENCY_POP_PRIMARY_SLICE]),
        _parse_header_int(line[_AGENCY_POP_SECONDARY_SLICE]),
    ]
    positive = [value for value in candidates if value > 0]
    if not positive:
        return None
    return min(positive)


def _parse_header_int(raw: str) -> int:
    match = re.match(r"\d+", raw)
    return int(match.group(0)) if match else 0


def _parse_count_field(raw: str) -> int:
    match = re.match(r"\d+", raw)
    return int(match.group(0)) if match else 0


def _download_year_zip(session: requests.Session, year: int, cache_dir: Path) -> Path:
    out_path = cache_dir / f"reta-{year}.zip"
    if out_path.exists():
        return out_path

    signed_resp = session.get(_SIGNED_URL_ENDPOINT.format(year=year), timeout=60)
    signed_resp.raise_for_status()
    signed_payload = signed_resp.json()
    signed_url = next(iter(signed_payload.values()))

    logger.info("Downloading FBI Return A zip for %d", year)
    with session.get(signed_url, timeout=120, stream=True) as resp:
        resp.raise_for_status()
        with out_path.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
    return out_path


def _load_agency_reference(
    session: requests.Session,
    raw_dir: Path,
) -> dict[str, dict[str, Any]]:
    cache_dir = raw_dir / "agency_reference"
    cache_dir.mkdir(parents=True, exist_ok=True)
    legacy_path = cache_dir / "agencies-by-state.json.gz"

    if not legacy_path.exists():
        logger.info("Downloading legacy FBI agency reference bundle.")
        with session.get(_LEGACY_AGENCY_BUNDLE_URL, timeout=60, stream=True) as resp:
            resp.raise_for_status()
            with legacy_path.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        fh.write(chunk)

    with gzip.open(legacy_path, "rt") as fh:
        legacy = json.load(fh)

    ref: dict[str, dict[str, Any]] = {}
    for state_abbr in _STATE_FIPS:
        state_cache = cache_dir / f"{state_abbr}.json"
        if state_cache.exists():
            live = json.loads(state_cache.read_text())
        else:
            resp = session.get(_LIVE_AGENCY_ENDPOINT.format(state=state_abbr), timeout=60)
            resp.raise_for_status()
            live = resp.json()
            state_cache.write_text(json.dumps(live))

        agency_names: set[tuple[str, str]] = set()
        county_names: set[str] = set()
        city_to_counties: defaultdict[str, set[str]] = defaultdict(set)
        agency_key_to_counties: defaultdict[str, set[str]] = defaultdict(set)

        for ori, item in legacy.get(state_abbr, {}).items():
            agency_norm = _normalize_text(item.get("agency_name", ""))
            county_norm = _normalize_county(item.get("primary_county", ""))
            city_norm = _city_key(item.get("agency_name", ""))
            agency_key = _agency_key_from_ori(ori)
            if agency_norm and county_norm:
                agency_names.add((agency_norm, county_norm))
            if county_norm:
                county_names.add(county_norm)
            if agency_key and county_norm:
                agency_key_to_counties[agency_key].add(county_norm)
            if city_norm and county_norm:
                city_to_counties[city_norm].add(county_norm)

        for county_name, agencies in live.items():
            county_norm = _normalize_county(county_name)
            if county_norm:
                county_names.add(county_norm)
            for item in agencies:
                agency_norm = _normalize_text(item["agency_name"])
                city_norm = _city_key(item["agency_name"])
                agency_key = _agency_key_from_ori(item.get("ori", ""))
                if agency_norm and county_norm:
                    agency_names.add((agency_norm, county_norm))
                if agency_key and county_norm:
                    agency_key_to_counties[agency_key].add(county_norm)
                if city_norm and county_norm:
                    city_to_counties[city_norm].add(county_norm)

        ref[state_abbr] = {
            "agency_names": sorted(agency_names, key=lambda x: len(x[0]), reverse=True),
            "county_names": sorted(county_names, key=len, reverse=True),
            "city_to_counties": city_to_counties,
            "agency_key_to_county": {
                key: next(iter(counties))
                for key, counties in agency_key_to_counties.items()
                if len(counties) == 1
            },
        }

    return ref


def _load_county_lookup(session: requests.Session) -> dict[tuple[str, str], str]:
    resp = session.get(_COUNTY_CODES_2020_URL, timeout=60)
    resp.raise_for_status()

    rows = [
        line.strip().split("|")
        for line in resp.text.splitlines()
        if line.strip()
    ]
    lookup: dict[tuple[str, str], str] = {}
    for state_abbr, state_fips, county_fips, _, county_name, *_ in rows[1:]:
        if state_abbr not in _STATE_FIPS:
            continue
        name_norm = _normalize_county(county_name)
        lookup[(state_fips, name_norm)] = f"{state_fips}{county_fips}"

    # Historical names that still appear in older FBI agency references.
    lookup[("02", "VALDEZCORDOVA")] = "02261"
    lookup[("46", "SHANNON")] = "46113"
    return lookup


def _normalize_text(value: str) -> str:
    value = (value or "").upper()
    for src, dst in _NORMALIZATION_REPLACEMENTS.items():
        value = value.replace(src, dst)
    value = re.sub(r"[^A-Z0-9]+", "", value)
    return value


def _agency_key_from_ori(ori: str) -> str:
    return str(ori).strip().upper()[:7]


def _normalize_county(value: str) -> str:
    value = (value or "").upper().replace("-", " ")
    for suffix in _COUNTY_SUFFIXES:
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    return _normalize_text(value)


def _city_key(agency_name: str) -> str:
    key = _normalize_text(agency_name)
    for suffix in (
        "POLICEDEPARTMENT",
        "SHERIFFSOFFICE",
        "STATEPOLICEHEADQUARTERS",
        "STATEPOLICE",
        "TRIBALPOLICEDEPARTMENT",
        "INTERNATIONALAIRPORT",
        "AIRPORT",
        "UNIVOF",
        "UNIVERSITY",
    ):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
            break
    return key
