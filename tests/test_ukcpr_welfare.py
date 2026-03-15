"""Tests for UKCPR welfare parsing."""

from __future__ import annotations

import pandas as pd

from povcrime.data.ukcpr_welfare import _reshape_ukcpr_welfare


def test_reshape_ukcpr_welfare_builds_expected_schema():
    raw = pd.DataFrame(
        {
            "state_fips": [6, 8],
            "year": [2000, 2000],
            "State EITC Rate": [0.0, 0.1],
            "Refundable State EITC (1=Yes)": [0, 1],
            "AFDC/TANF Benefit for 2-Person family": [505, 281],
            "AFDC/TANF Benefit for 3-person family": [626, 356],
            "AFDC/TANF benefit for 4-person family": [746, 432],
        }
    )

    df = _reshape_ukcpr_welfare(raw)

    assert df.columns.tolist() == [
        "state_fips",
        "year",
        "state_eitc_rate",
        "state_eitc_refundable",
        "tanf_benefit_2_person",
        "tanf_benefit_3_person",
        "tanf_benefit_4_person",
    ]
    assert df.loc[0, "state_fips"] == "06"
    assert df.loc[1, "state_eitc_rate"] == 0.1
    assert df.loc[1, "state_eitc_refundable"] == 1
