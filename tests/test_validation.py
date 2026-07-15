"""
JobForge AI — Salary Divergence Guard Tests.

The weekly snapshot median and 90-day rolling median have diverged publicly
before (~£46k vs ~£73k). This guard must trip whenever the gap exceeds the
threshold, so it's never silently published as two contradictory figures.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure

from src.jobforge.analytics.validation import (
    check_card_not_empty,
    check_no_nan_or_inf,
    check_no_overlapping_text,
    check_salary_divergence,
    enforce_min_sample,
)


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


def test_enforce_min_sample_suppresses_below_threshold():
    """n=2 shouldn't publish a precise-looking median like £92,000."""
    result = enforce_min_sample({"n": 2, "p25": 88000, "p50": 92000, "p75": 96000}, min_n=5)

    assert result["suppressed"] is True
    assert result["n"] == 2
    assert result["p25"] is None
    assert result["p50"] is None
    assert result["p75"] is None


def test_enforce_min_sample_passes_through_at_or_above_threshold():
    percentiles = {"n": 5, "p25": 50000, "p50": 60000, "p75": 70000}

    result = enforce_min_sample(percentiles, min_n=5)

    assert result["suppressed"] is False
    assert result["p50"] == 60000


def test_enforce_min_sample_missing_n_defaults_to_suppressed():
    result = enforce_min_sample({}, min_n=5)

    assert result["suppressed"] is True
    assert result["n"] == 0


# ── Render-Validation Gate (Phase 4.3 — chart output half) ──────────────────


def test_check_no_overlapping_text_trips_on_deliberate_collision():
    """Two labels placed at the identical spot must be flagged as overlapping."""
    fig = Figure(figsize=(4, 4), dpi=100)
    ax = fig.add_subplot()
    ax.text(0.5, 0.5, "Label A", fontsize=20, transform=ax.transAxes)
    ax.text(0.5, 0.5, "Label B", fontsize=20, transform=ax.transAxes)

    result = check_no_overlapping_text(fig)

    assert result["has_overlap"] is True
    assert len(result["overlapping_pairs"]) >= 1


def test_check_no_overlapping_text_passes_on_well_spaced_labels():
    """Labels placed far apart on the figure should never be flagged."""
    fig = Figure(figsize=(8, 8), dpi=100)
    ax = fig.add_subplot()
    ax.text(0.05, 0.95, "Top Left", fontsize=10, transform=ax.transAxes)
    ax.text(0.95, 0.05, "Bottom Right", fontsize=10, transform=ax.transAxes, ha="right")

    result = check_no_overlapping_text(fig)

    assert result["has_overlap"] is False
    assert result["overlapping_pairs"] == []


def test_check_card_not_empty_false_when_axes_has_no_artists():
    """An Axes with no bars/lines/text drawn is the 'empty card' bug class."""
    fig = Figure(figsize=(4, 4), dpi=100)
    ax = fig.add_subplot()

    assert check_card_not_empty(ax) is False


def test_check_card_not_empty_true_when_axes_has_content():
    fig = Figure(figsize=(4, 4), dpi=100)
    ax = fig.add_subplot()
    ax.bar(["a", "b"], [1, 2])

    assert check_card_not_empty(ax) is True


def test_check_no_nan_or_inf_false_when_nan_present():
    assert check_no_nan_or_inf([1.0, 2.0, float("nan")]) is False


def test_check_no_nan_or_inf_false_when_inf_present():
    assert check_no_nan_or_inf([1.0, float("inf"), 3.0]) is False
    assert check_no_nan_or_inf([1.0, float("-inf"), 3.0]) is False


def test_check_no_nan_or_inf_true_for_clean_values():
    assert check_no_nan_or_inf([1.0, 2.5, 100]) is True


def test_check_no_nan_or_inf_ignores_none_as_legitimate_missing_data():
    """A suppressed percentile (None) is missing data, not corrupt input."""
    assert check_no_nan_or_inf([1.0, None, 3.0]) is True


def test_check_no_nan_or_inf_true_for_empty_sequence():
    assert check_no_nan_or_inf([]) is True
