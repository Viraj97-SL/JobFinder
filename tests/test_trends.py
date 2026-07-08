"""
JobForge AI — Trend Classification Tests.

Covers the two acceptance cases called out in the upgrade plan directly:
a skill entering from zero classifies "New"; a skill on a steady climb
classifies "Accelerating". Also covers the statistical Rising/Cooling
noise-suppression behaviour (1.5).
"""

from __future__ import annotations

from datetime import datetime

import pytest

from src.jobforge.analytics.trends import (
    classify_rising_cooling,
    classify_trend,
    linear_trend,
    week_start,
)


class TestClassifyTrend:
    def test_skill_entering_from_zero_is_new(self):
        """CI/CD: absent for most of the window, then appears."""
        result = classify_trend([0, 0, 0, 0, 6])
        assert result["trend"] == "New"

    def test_steady_climb_is_accelerating(self):
        """Python: a consistent, near-linear rise week over week."""
        result = classify_trend([10, 14, 19, 24, 29, 35])
        assert result["trend"] == "Accelerating"
        assert result["slope"] > 0

    def test_steady_decline_is_cooling(self):
        result = classify_trend([40, 34, 29, 23, 18, 12])
        assert result["trend"] == "Cooling"
        assert result["slope"] < 0

    def test_flat_series_is_stable(self):
        result = classify_trend([20, 21, 19, 20, 21, 20])
        assert result["trend"] == "Stable"

    def test_empty_series_is_stable(self):
        result = classify_trend([])
        assert result["trend"] == "Stable"

    def test_never_appearing_skill_is_not_new(self):
        """All-zero history (never posted at all) is Stable, not New."""
        result = classify_trend([0, 0, 0, 0, 0])
        assert result["trend"] == "Stable"


class TestLinearTrend:
    def test_perfect_line_has_r_squared_of_one(self):
        slope, r_squared = linear_trend([1, 2, 3, 4, 5])
        assert slope == pytest.approx(1.0)
        assert r_squared == pytest.approx(1.0)

    def test_constant_series_has_zero_slope(self):
        slope, r_squared = linear_trend([5, 5, 5, 5])
        assert slope == 0.0
        assert r_squared == 0.0

    def test_single_point_is_degenerate(self):
        slope, r_squared = linear_trend([5])
        assert slope == 0.0
        assert r_squared == 0.0


class TestClassifyRisingCooling:
    def test_rising_on_consistent_uptrend(self):
        # 6 weeks, last 3 trending up and above their own mean
        assert classify_rising_cooling([10, 11, 9, 12, 15, 18]) == "Rising"

    def test_cooling_on_consistent_downtrend(self):
        assert classify_rising_cooling([10, 9, 11, 15, 12, 8]) == "Cooling"

    def test_single_week_noise_does_not_flip_to_rising(self):
        """A one-off spike shouldn't read as Rising if the short-term trend is flat/down."""
        # Trailing 3 weeks: 20, 19, 20 — roughly flat despite noisy history before it
        assert classify_rising_cooling([5, 30, 2, 20, 19, 20]) == "Stable"

    def test_insufficient_history_is_stable(self):
        assert classify_rising_cooling([10, 12], window=3) == "Stable"


def test_week_start_returns_monday():
    # 2026-03-04 is a Wednesday
    dt = datetime(2026, 3, 4, 15, 30)
    result = week_start(dt)
    assert result.weekday() == 0
    assert result == datetime(2026, 3, 2)
