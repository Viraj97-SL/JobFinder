"""
JobForge AI — UK Location Normalisation Tests.

Covers the postcode-area lookup, known-city substring fallback, remote
detection, and the no-space raw-postcode style ("WC2B5LX") seen in real
scraped data.
"""

from __future__ import annotations

from src.jobforge.utils.geography import normalize_location


def test_bare_city_name():
    assert normalize_location("London") == "London"


def test_postcode_with_space():
    assert normalize_location("London, WC2B 5LX") == "London"


def test_postcode_without_space():
    """Real scraped data includes postcodes with no internal space."""
    assert normalize_location("WC2B5LX") == "London"


def test_manchester_postcode():
    assert normalize_location("Manchester, M1 1AA") == "Manchester"


def test_city_name_substring_without_postcode():
    assert normalize_location("Cambridge Science Park") == "Cambridge"


def test_remote_detection():
    assert normalize_location("Remote, UK") == "Remote"
    assert normalize_location("Fully remote (UK based)") == "Remote"


def test_unrecognised_location_falls_back_to_other_uk():
    assert normalize_location("Some Village, Unknownshire") == "Other UK"


def test_empty_or_none_is_unknown():
    assert normalize_location("") == "Unknown"
    assert normalize_location(None) == "Unknown"


def test_postcode_takes_priority_over_remote_keyword():
    """A postcode is a stronger signal than an incidental 'remote' mention."""
    assert normalize_location("London, WC2B 5LX (hybrid, 2 days remote)") == "London"
