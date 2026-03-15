"""Tests for povcrime.processing.panel."""

from __future__ import annotations

import pandas as pd
import pytest

from povcrime.processing.panel import (
    build_county_year_panel,
    compute_rates,
    validate_panel_keys,
)


def test_compute_rates_basic():
    df = pd.DataFrame({
        "crime_count": [50, 100, 200],
        "pop": [100000, 200000, 50000],
    })
    rates = compute_rates(df, "crime_count", "pop", per=100_000)
    assert rates.iloc[0] == pytest.approx(50.0)
    assert rates.iloc[1] == pytest.approx(50.0)
    assert rates.iloc[2] == pytest.approx(400.0)


def test_compute_rates_zero_population():
    df = pd.DataFrame({
        "crime_count": [10, 20],
        "pop": [0, 100000],
    })
    rates = compute_rates(df, "crime_count", "pop")
    assert pd.isna(rates.iloc[0])
    assert rates.iloc[1] == pytest.approx(20.0)


def test_validate_panel_keys_unique():
    df = pd.DataFrame({
        "county_fips": ["01001", "01001", "01003"],
        "year": [2020, 2021, 2020],
    })
    result = validate_panel_keys(df)
    assert len(result) == 3


def test_validate_panel_keys_duplicates():
    df = pd.DataFrame({
        "county_fips": ["01001", "01001", "01003"],
        "year": [2020, 2020, 2020],
    })
    with pytest.raises(ValueError, match="duplicate"):
        validate_panel_keys(df)


def test_build_county_year_panel_adds_business_and_rent_derivatives(config):
    sources = {
        "saipe": pd.DataFrame(
            {
                "county_fips": ["01001"],
                "state_fips": ["01"],
                "year": [2020],
                "poverty_rate": [15.0],
                "median_hh_income": [50000.0],
                "poverty_count": [8000],
                "population": [100000],
            }
        ),
        "census_cbp": pd.DataFrame(
            {
                "county_fips": ["01001"],
                "state_fips": ["01"],
                "year": [2020],
                "cbp_establishments": [2000],
                "cbp_employment": [25000],
                "cbp_annual_payroll": [1250000],
            }
        ),
        "hud_fmr": pd.DataFrame(
            {
                "county_fips": ["01001"],
                "state_fips": ["01"],
                "year": [2020],
                "fair_market_rent_0br": [700.0],
                "fair_market_rent_1br": [800.0],
                "fair_market_rent_2br": [900.0],
                "fair_market_rent_3br": [1100.0],
                "fair_market_rent_4br": [1300.0],
            }
        ),
    }

    panel = build_county_year_panel(sources, config)

    assert panel.loc[0, "cbp_establishments_per_1k"] == pytest.approx(20.0)
    assert panel.loc[0, "cbp_employment_per_capita"] == pytest.approx(0.25)
    assert panel.loc[0, "cbp_payroll_per_employee"] == pytest.approx(50000.0)
    assert panel.loc[0, "rent_to_income_ratio_2br"] == pytest.approx(0.216)
