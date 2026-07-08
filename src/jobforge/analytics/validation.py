"""
JobForge AI — Analytics Validation Gate.

The 90-day rolling salary median and a single week's snapshot median have
previously diverged enough (~£46k vs ~£73k) to publish contradictory figures
across the LinkedIn series and marketforge.digital. This module doesn't
silently pick one — it flags the divergence so it can never ship unnoticed.

The full render/data validation gate (Phase 4.3) builds on this; this module
currently covers the salary divergence check called out in Phase 1.4.
"""

from __future__ import annotations

DEFAULT_DIVERGENCE_THRESHOLD = 0.15


def check_salary_divergence(
    weekly_median: float | None,
    rolling_median: float | None,
    threshold: float = DEFAULT_DIVERGENCE_THRESHOLD,
) -> dict[str, object]:
    """
    Compare a single-week salary snapshot median against the canonical
    90-day rolling median and flag if they diverge beyond `threshold`.

    Returns a dict always safe to serialise into report metadata:
      {"diverges": bool, "weekly_median": ..., "rolling_median": ..., "pct_diff": float | None}

    If either figure is missing (no disclosed-salary data for that window),
    divergence can't be assessed — reported as False rather than raising,
    since "insufficient data" is a different condition from "diverges".
    """
    if weekly_median is None or rolling_median is None or rolling_median == 0:
        return {
            "diverges": False,
            "weekly_median": weekly_median,
            "rolling_median": rolling_median,
            "pct_diff": None,
        }

    pct_diff = abs(weekly_median - rolling_median) / rolling_median

    return {
        "diverges": pct_diff > threshold,
        "weekly_median": weekly_median,
        "rolling_median": rolling_median,
        "pct_diff": round(pct_diff, 4),
    }
