"""
JobForge AI — LinkedIn Carousel Slide Builders (Phase 4.2).

Each `build_slide_*` function takes the single schema-validated MarketReport
(src/jobforge/models/report.py) and returns a matplotlib Figure — nothing
here re-derives figures from raw DB queries, it only reads fields already
assembled by MarketAnalyzer.build_market_report(). Light theme throughout
(white/near-white surface, dark ink) since these slides ship as LinkedIn
carousel images, not a dark-mode dashboard.

Palette values below are the validated light-mode instance from the
project's dataviz skill reference palette (categorical hues in fixed order,
one hue for magnitude, status colors reserved for up/down/neutral deltas).

Every builder validates its own numeric chart inputs with
`check_no_nan_or_inf` *before* handing them to matplotlib, raising
`SlideRenderError` on failure so the caller (scripts/generate_carousel.py)
can skip writing that slide instead of shipping a corrupted chart.
"""

from __future__ import annotations

from typing import Callable

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

from jobforge.analytics.validation import check_no_nan_or_inf
from jobforge.models.report import MarketReport

# ── Slide geometry (portrait, LinkedIn carousel aspect ratio) ───────────────
SLIDE_WIDTH_PX = 1080
SLIDE_HEIGHT_PX = 1350
DPI = 100
FIGSIZE = (SLIDE_WIDTH_PX / DPI, SLIDE_HEIGHT_PX / DPI)

# ── Light-theme palette (see dataviz skill: references/palette.md) ─────────
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"
MAGENTA = "#e87ba4"
ORANGE = "#eb6834"
CATEGORICAL_CYCLE = [BLUE, AQUA, YELLOW, GREEN, VIOLET, RED, MAGENTA, ORANGE]

STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"

FONT_FAMILY = "sans-serif"

DIRECTION_ARROW = {"up": "↑", "down": "↓", "flat": "→", "unknown": "→"}
DIRECTION_COLOR = {"up": STATUS_GOOD, "down": STATUS_CRITICAL, "flat": INK_MUTED, "unknown": INK_MUTED}


class SlideRenderError(Exception):
    """Raised when a slide's numeric chart inputs fail validation before plotting."""


def _require_finite(values: list[float], slide_name: str) -> None:
    if not check_no_nan_or_inf(values):
        raise SlideRenderError(f"{slide_name}: NaN/inf detected in chart input data")


def _new_slide(title: str, subtitle: str | None = None) -> Figure:
    """Blank portrait slide with the standard header — every slide starts here."""
    fig = Figure(figsize=FIGSIZE, dpi=DPI, facecolor=SURFACE)
    FigureCanvasAgg(fig)

    fig.text(0.06, 0.965, title, fontsize=28, fontweight="bold",
              color=INK_PRIMARY, family=FONT_FAMILY, va="top", ha="left")
    if subtitle:
        fig.text(0.06, 0.915, subtitle, fontsize=14,
                  color=INK_SECONDARY, family=FONT_FAMILY, va="top", ha="left")
    return fig


def _footer(fig: Figure, text: str) -> None:
    fig.text(0.06, 0.025, text, fontsize=10, color=INK_MUTED, family=FONT_FAMILY, va="bottom", ha="left")


# ── Slide 1 — Headline stats card ───────────────────────────────────────────

def build_slide_headline(report: MarketReport) -> Figure:
    meta = report.metadata
    top_skill = report.top_skills[0] if report.top_skills else None
    salary_p50 = report.salary_percentiles.p50

    numeric_inputs: list[float] = [meta.total_jobs]
    if top_skill is not None:
        numeric_inputs.append(top_skill[1])
    if salary_p50 is not None:
        numeric_inputs.append(salary_p50)
    _require_finite(numeric_inputs, "slide_1_headline")

    fig = _new_slide(
        "This Week in UK AI/ML Hiring",
        f"{meta.window_days}-day window · generated {meta.generated_at:%d %b %Y}",
    )

    ax = fig.add_axes((0.04, 0.30, 0.92, 0.50))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    salary_text = f"£{salary_p50:,.0f}" if salary_p50 is not None else "Insufficient data"
    salary_color = INK_PRIMARY if salary_p50 is not None else INK_MUTED
    top_skill_value = f"{top_skill[0]}\n({top_skill[1]} mentions)" if top_skill else "No data"

    cards = [
        ("Total Jobs Tracked", f"{meta.total_jobs:,}", INK_PRIMARY),
        ("Top Skill in Demand", top_skill_value, BLUE),
        ("Median Salary", salary_text, salary_color),
    ]

    card_width = 0.28
    gap = 0.04
    x = 0.02
    for label, value, color in cards:
        box = FancyBboxPatch(
            (x, 0.05), card_width, 0.9,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=1.2, edgecolor=BASELINE, facecolor="#ffffff",
            transform=ax.transAxes,
        )
        ax.add_patch(box)
        ax.text(x + card_width / 2, 0.72, label, ha="center", va="center",
                 fontsize=12, color=INK_SECONDARY, transform=ax.transAxes, wrap=True)
        ax.text(x + card_width / 2, 0.40, value, ha="center", va="center",
                 fontsize=18, fontweight="bold", color=color, transform=ax.transAxes)
        x += card_width + gap

    if report.salary_divergence.diverges:
        weekly = report.salary_divergence.weekly_median
        rolling = report.salary_divergence.rolling_median
        pct = report.salary_divergence.pct_diff or 0.0
        banner = (
            f"Salary divergence flagged: weekly £{weekly:,.0f} vs "
            f"90-day rolling £{rolling:,.0f} ({pct:.0%} apart)"
        )
        fig.text(0.5, 0.16, banner, ha="center", va="center", fontsize=12,
                  color="#ffffff", family=FONT_FAMILY,
                  bbox={"boxstyle": "round,pad=0.6", "facecolor": STATUS_CRITICAL, "edgecolor": "none"})

    _footer(fig, "JobForge AI — Market Intelligence")
    return fig


# ── Slide 2 — Top skills bar chart + WoW delta strip ────────────────────────

def build_slide_top_skills(report: MarketReport) -> Figure:
    top_skills = list(reversed(report.top_skills[:10]))
    names = [skill for skill, _ in top_skills]
    counts = [count for _, count in top_skills]
    _require_finite(counts, "slide_2_top_skills")

    fig = _new_slide("Top Skills in Demand", "Mentions across tracked job postings")

    ax = fig.add_axes((0.32, 0.36, 0.60, 0.46))
    if counts:
        bars = ax.barh(names, counts, color=BLUE, height=0.6, zorder=3)
        ax.bar_label(bars, padding=4, color=INK_PRIMARY, fontsize=11)
    else:
        ax.text(0.5, 0.5, "No skill data available", ha="center", va="center",
                fontsize=12, color=INK_MUTED, transform=ax.transAxes)
        # No bars means matplotlib falls back to default 0..1 tick marks on
        # both axes, and their "0.0" labels can land on top of one another
        # at the origin corner — blank the ticks since there's no data axis
        # to label anyway.
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="y", colors=INK_PRIMARY, labelsize=12)
    ax.tick_params(axis="x", colors=INK_MUTED, labelsize=10)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # Market pulse strip — WoW deltas for the aggregate metrics in `deltas`.
    pulse_ax = fig.add_axes((0.04, 0.08, 0.92, 0.16))
    pulse_ax.axis("off")
    metric_order = ["total_volume", "sponsorship_rate", "startup_share", "salary_median"]
    metric_labels = {
        "total_volume": "Postings",
        "sponsorship_rate": "Sponsorship",
        "startup_share": "Startup Share",
        "salary_median": "Salary Median",
    }
    slot = 1.0 / len(metric_order)
    for i, key in enumerate(metric_order):
        delta = report.deltas.get(key)
        cx = slot * (i + 0.5)
        if delta is None or delta.pct_change is None:
            value_text, color = "n/a", INK_MUTED
        else:
            arrow = DIRECTION_ARROW.get(delta.direction, "→")
            color = DIRECTION_COLOR.get(delta.direction, INK_MUTED)
            value_text = f"{arrow} {delta.pct_change:+.0%}"
        pulse_ax.text(cx, 0.70, metric_labels[key], ha="center", va="center",
                       fontsize=11, color=INK_SECONDARY, transform=pulse_ax.transAxes)
        pulse_ax.text(cx, 0.20, value_text, ha="center", va="center",
                       fontsize=15, fontweight="bold", color=color, transform=pulse_ax.transAxes)

    _footer(fig, "Week-over-week change vs the prior 7-day window")
    return fig


# ── Slide 3 — Rising / cooling skills ───────────────────────────────────────

def build_slide_rising_cooling(report: MarketReport) -> Figure:
    rising = report.rising_cooling_skills.get("rising", [])[:10]
    cooling = report.rising_cooling_skills.get("cooling", [])[:10]

    fig = _new_slide("Rising & Cooling Skills", "Trailing 6-week trend classification")

    left_ax = fig.add_axes((0.05, 0.10, 0.42, 0.74))
    right_ax = fig.add_axes((0.53, 0.10, 0.42, 0.74))

    columns = [
        (left_ax, "Rising", rising, STATUS_GOOD, "▲"),
        (right_ax, "Cooling", cooling, STATUS_CRITICAL, "▼"),
    ]
    for ax, title, items, color, arrow in columns:
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.98, title, ha="center", va="top", fontsize=18,
                 fontweight="bold", color=color, transform=ax.transAxes)
        if items:
            step = min(0.82 / len(items), 0.11)
            for i, skill in enumerate(items):
                y = 0.86 - i * step
                ax.text(0.5, y, f"{arrow}  {skill}", ha="center", va="top",
                         fontsize=13, color=INK_PRIMARY, transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "No skills in this category", ha="center", va="center",
                     fontsize=12, color=INK_MUTED, transform=ax.transAxes)

    _footer(fig, "Statistically smoothed classification — single-week noise excluded")
    return fig


# ── Slide 4 — Salary percentiles by role category ───────────────────────────

def build_slide_salary_by_category(report: MarketReport) -> Figure:
    categories = sorted(
        report.salary_by_category.items(),
        key=lambda kv: (kv[1].p50 is None, -(kv[1].p50 or 0)),
    )[:8]

    numeric_inputs: list[float] = []
    for _, percentiles in categories:
        for value in (percentiles.p25, percentiles.p50, percentiles.p75):
            if value is not None:
                numeric_inputs.append(value)
    _require_finite(numeric_inputs, "slide_4_salary_by_category")

    fig = _new_slide("Salary by Role Category", "P25–P75 annual salary range (GBP)")

    ax = fig.add_axes((0.30, 0.12, 0.62, 0.68))
    ax.set_facecolor(SURFACE)

    if not categories:
        ax.text(0.5, 0.5, "No salary data available", ha="center", va="center",
                 fontsize=13, color=INK_MUTED, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    else:
        labels = []
        for y, (category, percentiles) in enumerate(categories):
            labels.append(category)
            # SalaryPercentiles has no "suppressed" field on the schema — the
            # min-sample-size gate (enforce_min_sample) already nulls p25/p50/
            # p75 for below-threshold categories, so p50 is None *is* the
            # suppression signal here; n is still preserved for the label.
            if percentiles.p50 is None:
                ax.text(0.02, y, f"Insufficient data (n={percentiles.n})", va="center",
                        ha="left", fontsize=11, color=INK_MUTED,
                        transform=ax.get_yaxis_transform())
                continue
            ax.plot([percentiles.p25, percentiles.p75], [y, y], color=BLUE,
                     linewidth=6, solid_capstyle="round", alpha=0.35, zorder=2)
            ax.plot(percentiles.p50, y, marker="o", markersize=9, color=BLUE, zorder=3)
            ax.text(percentiles.p50, y + 0.30, f"£{percentiles.p50:,.0f}",
                     ha="center", fontsize=10, color=INK_PRIMARY)

        ax.set_yticks(range(len(categories)))
        ax.set_yticklabels(labels, fontsize=12, color=INK_PRIMARY)
        # Explicit (inverted) ylim with a fixed half-row margin on both ends
        # — autoscale + invert_yaxis() alone clamps the last row's data limit
        # to *exactly* the axis edge (no margin), which lets a suppressed
        # row's "insufficient data" text sit directly on top of the x-axis
        # spine instead of above it.
        ax.set_ylim(len(categories) - 0.5, -0.5)
        ax.set_xlabel("Annual salary (£)", color=INK_SECONDARY, fontsize=11)
        ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="x", colors=INK_MUTED, labelsize=10)

    _footer(fig, "Ranges below the minimum sample size are suppressed, never estimated")
    return fig


# ── Slide 5 — Geographic + company-stage distribution ───────────────────────

def build_slide_geo_company_stage(report: MarketReport) -> Figure:
    geo = report.geographic_distribution[:8]
    stage = sorted(report.company_stage_distribution.items(), key=lambda kv: -kv[1])

    numeric_inputs = [count for _, count in geo] + [count for _, count in stage]
    _require_finite(numeric_inputs, "slide_5_geo_company_stage")

    fig = _new_slide("Where the Roles Are", "Geography & company-stage breakdown")

    geo_ax = fig.add_axes((0.32, 0.56, 0.60, 0.30))
    geo_ax.set_title("Top Locations", loc="left", fontsize=13, color=INK_SECONDARY)
    if geo:
        names = [region for region, _ in reversed(geo)]
        counts = [count for _, count in reversed(geo)]
        bars = geo_ax.barh(names, counts, color=AQUA, height=0.6, zorder=3)
        geo_ax.bar_label(bars, padding=4, fontsize=10, color=INK_PRIMARY)
        geo_ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
    else:
        geo_ax.text(0.5, 0.5, "No geographic data available", ha="center", va="center",
                     fontsize=12, color=INK_MUTED, transform=geo_ax.transAxes)
        geo_ax.set_xticks([])
        geo_ax.set_yticks([])
        geo_ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    geo_ax.set_facecolor(SURFACE)
    geo_ax.spines[["top", "right", "left"]].set_visible(False)
    geo_ax.spines["bottom"].set_color(BASELINE)
    geo_ax.tick_params(axis="y", labelsize=10, colors=INK_PRIMARY)
    geo_ax.tick_params(axis="x", labelsize=9, colors=INK_MUTED)
    geo_ax.set_axisbelow(True)

    stage_ax = fig.add_axes((0.14, 0.10, 0.72, 0.32))
    stage_ax.set_title("Company Stage", loc="left", fontsize=13, color=INK_SECONDARY)
    if stage:
        stage_names = [name for name, _ in stage]
        stage_counts = [count for _, count in stage]
        colors = [CATEGORICAL_CYCLE[i % len(CATEGORICAL_CYCLE)] for i in range(len(stage_names))]
        stage_ax.pie(
            stage_counts, labels=stage_names, colors=colors, autopct="%1.0f%%",
            textprops={"color": INK_PRIMARY, "fontsize": 10},
            wedgeprops={"edgecolor": SURFACE, "linewidth": 1.5},
        )
        stage_ax.axis("equal")
    else:
        stage_ax.text(0.5, 0.5, "No company-stage data available", ha="center", va="center",
                       fontsize=12, color=INK_MUTED, transform=stage_ax.transAxes)
        stage_ax.axis("off")

    _footer(fig, "JobForge AI — Market Intelligence")
    return fig


SLIDE_BUILDERS: list[tuple[str, Callable[[MarketReport], Figure]]] = [
    ("01_headline.png", build_slide_headline),
    ("02_top_skills.png", build_slide_top_skills),
    ("03_rising_cooling.png", build_slide_rising_cooling),
    ("04_salary_by_category.png", build_slide_salary_by_category),
    ("05_geo_company_stage.png", build_slide_geo_company_stage),
]
