"""Tests for the FBI Return A county fallback builder helpers."""

from __future__ import annotations

from povcrime.data.fbi_reta_master import (
    _agency_key_from_ori,
    _extract_agency_population,
    _match_county_fips,
)


def test_agency_key_from_ori_uses_stable_prefix():
    assert _agency_key_from_ori("RI0040900") == "RI00409"
    assert _agency_key_from_ori("RIDI00300") == "RIDI003"


def test_match_county_fips_prefers_agency_key_lookup():
    line = "138RI004092 11450916".ljust(120) + "PROVIDENCE              R I   CHIEF OF POLICE".ljust(120)
    agency_ref = {
        "RI": {
            "agency_key_to_county": {"RI00409": "PROVIDENCE"},
            "agency_names": [],
            "county_names": [],
            "city_to_counties": {},
        }
    }
    county_lookup = {("44", "PROVIDENCE"): "44007"}

    county_fips, agency_key = _match_county_fips(
        line=line,
        state_abbr="RI",
        agency_ref=agency_ref,
        county_lookup=county_lookup,
    )

    assert county_fips == "44007"
    assert agency_key == "RI00409"


def test_extract_agency_population_uses_smaller_nonzero_header_value():
    line = [" "] * 240
    line[44:53] = list("000178640")
    line[89:98] = list("000178887")

    assert _extract_agency_population("".join(line)) == 178640
