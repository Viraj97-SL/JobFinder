"""
JobForge AI — Salary Divergence Guard Tests.

The weekly snapshot median and 90-day rolling median have diverged publicly
before (~£46k vs ~£73k). This guard must trip whenever the gap exceeds the
threshold, so it's never silently published as two contradictory figures.
"""

from __future__ import annotations

from src.jobforge.analytics.validation import check_salary_divergence


def test_no_divergence_within_threshold():
    result = check_salary_divergence(weekly_median=72000, rolling_median=70000)
    assert result["diverges"] is False


def test_divergence_trips_above_threshold():
    """~£46k vs ~£73k is the real-world case that motivated this guard."""
    result = check_salary_divergence(weekly_median=46000, rolling_median=73000)
    assert result["diverges"] is True
    assert result["pct_diff"] > 0.15


def test_divergence_exactly_at_threshold_does_not_trip():
    result = check_salary_divergence(weekly_median=115000, rolling_median=100000, threshold=0.15)
    assert result["diverges"] is False


def test_missing_weekly_median_does_not_trip():
    result = check_salary_divergence(weekly_median=None, rolling_median=70000)
    assert result["diverges"] is False
    assert result["pct_diff"] is None


def test_missing_rolling_median_does_not_trip():
    result = check_salary_divergence(weekly_median=70000, rolling_median=None)
    assert result["diverges"] is False


def test_custom_threshold_is_respected():
    result = check_salary_divergence(weekly_median=80000, rolling_median=70000, threshold=0.05)
    assert result["diverges"] is True
