"""
JobForge AI — Analytics Validation Gate.

The 90-day rolling salary median and a single week's snapshot median have
previously diverged enough (~£46k vs ~£73k) to publish contradictory figures
across the LinkedIn series and marketforge.digital. This module doesn't
silently pick one — it flags the divergence so it can never ship unnoticed.

This module covers both halves of the Phase 4.3 validation gate:
  - the *data* half: no percentile-based figure ships below a minimum sample
    size, and the salary divergence guard from Phase 1.4.
  - the *render* half (Phase 4.2/4.3): chart label collisions, empty cards,
    and NaN/inf in chart inputs, wired into scripts/generate_carousel.py so a
    slide that fails one of these checks is never written to disk — the run
    exits non-zero and prints which slide needs manual review instead.
"""

from __future__ import annotations

import math
from typing import Sequence

from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.text import Text

DEFAULT_DIVERGENCE_THRESHOLD = 0.15
DEFAULT_MIN_SAMPLE_N = 5


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


def enforce_min_sample(
    percentiles: dict[str, object],
    min_n: int = DEFAULT_MIN_SAMPLE_N,
) -> dict[str, object]:
    """
    Null out a percentile summary's p25/p50/p75 when its sample size is
    below `min_n`, instead of publishing a precise-looking median computed
    from a handful of data points (e.g. "MLOps median: £92,000" from n=2).

    `n` is always preserved and a `suppressed` flag is added, so callers can
    render "insufficient data (n=2)" rather than the figure just vanishing
    with no explanation.
    """
    n = int(percentiles.get("n", 0) or 0)
    if n >= min_n:
        return {**percentiles, "suppressed": False}
    return {"n": n, "p25": None, "p50": None, "p75": None, "suppressed": True}


def check_no_overlapping_text(fig: Figure) -> dict[str, object]:
    """
    Walk every visible `Text` artist on a rendered Figure (titles, tick
    labels, legends, manually-placed labels — `fig.findobj` catches all of
    them) and flag any pair whose pixel bounding boxes overlap.

    `get_window_extent()` only returns real (not stale, zero-sized) pixel
    coordinates once the figure has an active renderer, so this calls
    `fig.canvas.draw()` first — cheap relative to the risk of shipping a
    slide with two labels stacked on top of each other.

    Returns a dict always safe to serialise / log:
      {"has_overlap": bool, "overlapping_pairs": [(text_a, text_b), ...]}
    """
    if not hasattr(fig.canvas, "get_renderer"):
        # A bare `Figure()` has no renderer-capable canvas until one is
        # attached (matplotlib's default FigureCanvasBase can't draw) —
        # attach a headless Agg canvas so this check works for any Figure,
        # not just ones built through pyplot or already wired for saving.
        FigureCanvasAgg(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    texts = [
        text for text in fig.findobj(match=Text)
        if text.get_visible() and text.get_text().strip()
    ]
    boxes = [(text, text.get_window_extent(renderer=renderer)) for text in texts]

    overlapping_pairs: list[tuple[str, str]] = []
    for i, (text_a, box_a) in enumerate(boxes):
        for text_b, box_b in boxes[i + 1:]:
            if box_a.overlaps(box_b):
                overlapping_pairs.append((text_a.get_text(), text_b.get_text()))

    return {"has_overlap": bool(overlapping_pairs), "overlapping_pairs": overlapping_pairs}


def check_card_not_empty(ax: Axes) -> bool:
    """
    Returns False when an Axes has zero content artists — catches the
    "empty card" bug class where a slide renders a title/frame but the
    actual data (bars, lines, markers, manually-placed text) never got
    drawn, e.g. a silently-skipped `ax.bar(...)` call.

    Deliberately ignores axis ticks/labels and the title (always present on
    a fresh Axes) and only counts artists that represent drawn content:
    lines, patches (bars, wedges, boxes), collections (scatter), images,
    manually-added text, and bar/pie containers.
    """
    content_artist_count = (
        len(ax.lines)
        + len(ax.patches)
        + len(ax.collections)
        + len(ax.images)
        + len(ax.texts)
        + len(ax.containers)
    )
    return content_artist_count > 0


def check_no_nan_or_inf(values: Sequence[float]) -> bool:
    """
    Returns False if any NaN or +/-inf is present in a sequence of chart
    input values, before it's handed to matplotlib. `None` entries are
    treated as legitimate "no data" (e.g. a suppressed percentile) rather
    than corrupt input, so they don't trip this check.
    """
    for value in values:
        if value is None:
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(numeric_value) or math.isinf(numeric_value):
            return False
    return True
