"""
JobForge AI — Time-Series Trend Analysis.

Windowing and trend-classification helpers shared by MarketAnalyzer's
week-over-week deltas, per-skill trend classification, and the statistical
Rising/Cooling skill lists. Kept separate from market_analyzer.py because
the maths here (linear regression, week bucketing) is independent of how
the underlying data is loaded from the DB.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# A skill with fewer than this many total zero-followed-by-nonzero weeks of
# history isn't "New" so much as noise — require it to have been genuinely
# absent before appearing.
NEW_SKILL_ZERO_WEEKS = 2
# Below this |slope|, week-over-week movement is treated as noise, not trend.
STABLE_SLOPE_THRESHOLD = 0.3
# Minimum R^2 for a slope to be called "Accelerating"/"Cooling" rather than
# a weak/noisy fit.
TREND_R_SQUARED_THRESHOLD = 0.8


def week_start(dt: datetime) -> datetime:
    """Return the Monday 00:00 of the ISO week containing dt."""
    date_only = datetime(dt.year, dt.month, dt.day)
    return date_only - timedelta(days=date_only.weekday())


def weekly_series(timestamps: pd.Series, values: pd.Series | None = None) -> pd.Series:
    """
    Bucket a series of timestamps into weekly counts (or summed values).

    Parameters
    ----------
    timestamps : parseable datetime strings/objects
    values     : optional weights to sum per week; defaults to a count of 1 per row

    Returns
    -------
    pd.Series indexed by week-start date (chronological), one entry per week
    that has at least one observation — gaps are NOT filled with zero here;
    callers that need a dense weekly grid should reindex.
    """
    ts = pd.to_datetime(timestamps)
    weeks = ts.apply(week_start)
    if values is None:
        return weeks.value_counts().sort_index()
    return pd.Series(values.to_numpy(), index=weeks).groupby(level=0).sum().sort_index()


def linear_trend(y: list[float] | np.ndarray) -> tuple[float, float]:
    """
    Fit a simple linear trend to y (evenly spaced, one point per period).

    Returns (slope, r_squared). Degenerate inputs (fewer than 2 points, or
    zero variance) return (0.0, 0.0) rather than raising.
    """
    y_arr = np.asarray(y, dtype=float)
    n = len(y_arr)
    if n < 2 or np.all(y_arr == y_arr[0]):
        return 0.0, 0.0

    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, y_arr, 1)
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y_arr - y_pred) ** 2))
    ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), r_squared


def classify_trend(weekly_counts: list[float]) -> dict[str, object]:
    """
    Classify a weekly count series as Accelerating / Cooling / Stable / New.

    Label rules (checked in order):
      1. New       — absent (zero) for all but the most recent weeks, then appears.
      2. Accelerating — R^2 > 0.8 and positive slope.
      3. Cooling      — R^2 > 0.8 and negative slope.
      4. Stable       — |slope| below the noise threshold.
      5. Cooling/Accelerating — weak-fit fallback, sign of slope decides.
    """
    n = len(weekly_counts)
    if n == 0:
        return {"trend": "Stable", "slope": 0.0, "r_squared": 0.0}

    zero_prefix_len = n - 1
    if (
        n > NEW_SKILL_ZERO_WEEKS
        and all(c == 0 for c in weekly_counts[:zero_prefix_len])
        and weekly_counts[-1] > 0
    ):
        return {"trend": "New", "slope": 0.0, "r_squared": 0.0}

    slope, r_squared = linear_trend(weekly_counts)

    if r_squared > TREND_R_SQUARED_THRESHOLD and slope > 0:
        trend = "Accelerating"
    elif r_squared > TREND_R_SQUARED_THRESHOLD and slope < 0:
        trend = "Cooling"
    elif abs(slope) < STABLE_SLOPE_THRESHOLD:
        trend = "Stable"
    else:
        trend = "Cooling" if slope < 0 else "Accelerating"

    return {"trend": trend, "slope": round(slope, 3), "r_squared": round(r_squared, 3)}


def classify_rising_cooling(weekly_counts: list[float], window: int = 3) -> str:
    """
    Statistically robust Rising/Cooling/Stable label over the trailing window.

    Rising  = positive slope over the trailing `window` weeks AND the current
              week's value is at or above the trailing-window mean.
    Cooling = the mirror condition (negative slope, current <= mean).
    Stable  = neither condition holds, or not enough history yet.

    This suppresses single-week noise: a one-off spike doesn't flip the
    label unless it also drags the short-term trend line with it.
    """
    if len(weekly_counts) < window:
        return "Stable"

    trailing = weekly_counts[-window:]
    slope, _ = linear_trend(trailing)
    mean = sum(trailing) / len(trailing)
    current = trailing[-1]

    if slope > 0 and current >= mean:
        return "Rising"
    if slope < 0 and current <= mean:
        return "Cooling"
    return "Stable"
