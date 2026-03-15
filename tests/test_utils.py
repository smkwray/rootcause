"""Tests for povcrime.utils module."""

from __future__ import annotations

from povcrime.utils import standardize_fips


def test_standardize_fips_pads_correctly():
    """Default width=5 should left-pad with zeros."""
    assert standardize_fips("1") == "00001"
    assert standardize_fips("12345") == "12345"


def test_standardize_fips_width_3():
    """width=3 should produce 3-character FIPS codes."""
    assert standardize_fips("1", width=3) == "001"
