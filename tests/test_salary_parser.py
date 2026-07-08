"""
JobForge AI — Salary Period Parser Tests.

Covers period detection (annual/daily/hourly/unknown) and annual-equivalent
normalisation, including the garbage-value filtering the plan called out
("£0k-£0k", "£1k-£1k" style misparsed entries must never enter the median).
"""

from __future__ import annotations

from src.jobforge.utils.salary_parser import detect_salary_period, normalize_to_annual


class TestDetectSalaryPeriod:
    def test_explicit_annual_cue(self):
        assert detect_salary_period("£60,000 - £80,000 per annum", 60000, 80000) == "annual"

    def test_explicit_daily_cue(self):
        assert detect_salary_period("Day rate: £700-£800, outside IR35", 700, 800) == "daily"

    def test_explicit_hourly_cue(self):
        assert detect_salary_period("Rate: £45-£55 per hour", 45, 55) == "hourly"

    def test_pa_abbreviation_is_annual(self):
        assert detect_salary_period("Salary 45k pa", 45000, 45000) == "annual"

    def test_no_cue_falls_back_to_plausible_annual(self):
        assert detect_salary_period("Great team, hybrid working", 55000, 75000) == "annual"

    def test_no_cue_and_implausible_annual_is_unknown(self):
        """Small numbers with no period cue at all are unclassifiable, not guessed as day rate."""
        assert detect_salary_period("Great team, hybrid working", 700, 800) == "unknown"

    def test_no_salary_values_is_unknown(self):
        assert detect_salary_period("No salary disclosed", None, None) == "unknown"

    def test_garbage_zero_values_are_unknown(self):
        assert detect_salary_period("Competitive salary", 0, 0) == "unknown"

    def test_false_positive_word_not_treated_as_period_cue(self):
        """'spa' contains 'pa' as a substring but must not trigger the annual p.a. cue check."""
        assert detect_salary_period("Office has a spa and gym", 700, 800) == "unknown"


class TestNormalizeToAnnual:
    def test_annual_passthrough(self):
        assert normalize_to_annual(60000, 80000, "annual") == (60000, 80000)

    def test_daily_converts_to_annual_equivalent(self):
        annual_min, annual_max = normalize_to_annual(700, 800, "daily")
        assert annual_min == 700 * 220
        assert annual_max == 800 * 220

    def test_hourly_converts_to_annual_equivalent(self):
        annual_min, annual_max = normalize_to_annual(45, 55, "hourly")
        assert annual_min == 45 * 1650
        assert annual_max == 55 * 1650

    def test_unknown_period_yields_no_annual_values(self):
        assert normalize_to_annual(700, 800, "unknown") == (None, None)

    def test_garbage_low_annual_is_dropped(self):
        """'£1k-£1k' style misparsed annual figures must be excluded, not treated as real."""
        assert normalize_to_annual(1000, 1000, "annual") == (None, None)

    def test_implausible_daily_rate_is_dropped_not_projected(self):
        """A 'daily' figure too large to be a real day rate (misclassified annual figure) is dropped."""
        annual_min, annual_max = normalize_to_annual(50000, 60000, "daily")
        assert annual_min is None
        assert annual_max is None

    def test_implausible_hourly_rate_is_dropped_not_projected(self):
        annual_min, annual_max = normalize_to_annual(5000, 6000, "hourly")
        assert annual_min is None
        assert annual_max is None

    def test_none_values_pass_through_as_none(self):
        assert normalize_to_annual(None, None, "annual") == (None, None)
