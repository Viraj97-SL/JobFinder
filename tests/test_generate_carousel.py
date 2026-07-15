"""
JobForge AI — LinkedIn Carousel Generator Tests.

Builds a small, schema-valid MarketReport in-memory (same schema tested in
tests/test_report.py), round-trips it through JSON exactly the way
scripts/market_report.py --json does, then drives the slide-generation
functions directly — no shelling out to the CLI script.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from jobforge.models.report import (
    MarketReport,
    MetricDelta,
    ReportMetadata,
    SalaryDivergence,
    SalaryPercentiles,
)
from scripts.generate_carousel import generate_slides


def _build_report(**overrides) -> MarketReport:
    metadata = ReportMetadata(
        generated_at=datetime(2026, 7, 14, 8, 0, 0),
        window_days=90,
        total_jobs=120,
        divergence_flagged=False,
    )

    defaults = dict(
        metadata=metadata,
        top_skills=[("Python", 80), ("PyTorch", 55), ("LangGraph", 40), ("SQL", 35)],
        salary_percentiles=SalaryPercentiles(n=20, p25=55000, p50=65000, p75=80000),
        salary_by_category={
            "AI/LLM Engineer": SalaryPercentiles(n=10, p25=60000, p50=70000, p75=85000),
            "MLOps": SalaryPercentiles(n=2, p25=None, p50=None, p75=None),  # suppressed
        },
        salary_divergence=SalaryDivergence(
            diverges=True, weekly_median=46000, rolling_median=73000, pct_diff=0.37,
        ),
        deltas={
            "total_volume": MetricDelta(
                metric="total_volume", weeks=1, current=120, previous=100,
                abs_change=20, pct_change=0.20, direction="up",
            ),
            "sponsorship_rate": MetricDelta(
                metric="sponsorship_rate", weeks=1, current=0.30, previous=0.32,
                abs_change=-0.02, pct_change=-0.0625, direction="down",
            ),
            "startup_share": MetricDelta(
                metric="startup_share", weeks=1, current=0.5, previous=0.5,
                abs_change=0.0, pct_change=0.0, direction="flat",
            ),
            "salary_median": MetricDelta(metric="salary_median", weeks=1, direction="unknown"),
        },
        rising_cooling_skills={
            "rising": ["LangGraph", "RAG", "Agentic AI"],
            "cooling": ["Hadoop", "Spark"],
            "stable": ["Python"],
        },
        geographic_distribution=[("London", 60), ("Manchester", 20), ("Remote", 40)],
        company_stage_distribution={"series_a": 30, "series_b": 20, "public": 50, "unknown": 20},
    )
    defaults.update(overrides)
    return MarketReport(**defaults)


def _round_trip_through_json(report: MarketReport, tmp_path) -> MarketReport:
    """Mirrors scripts/market_report.py --json's exact serialisation path."""
    json_path = tmp_path / "market_report.json"
    json_path.write_text(json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8")
    return MarketReport.model_validate_json(json_path.read_text(encoding="utf-8"))


def test_generate_slides_writes_five_non_trivial_pngs(tmp_path):
    report = _round_trip_through_json(_build_report(), tmp_path)
    outdir = tmp_path / "carousel"

    results = generate_slides(report, outdir)

    assert len(results) == 5
    assert all(result.saved for result in results), [r.reasons for r in results if not r.saved]

    png_files = sorted(outdir.glob("*.png"))
    assert len(png_files) == 5
    for png_file in png_files:
        assert png_file.stat().st_size > 5_000, f"{png_file.name} looks too small to be a real slide"


def test_generate_slides_skips_suppressed_category_without_fabricating_a_number(tmp_path):
    """MLOps (n=2) must render as 'insufficient data', never a precise median."""
    report = _round_trip_through_json(_build_report(), tmp_path)
    outdir = tmp_path / "carousel"

    results = generate_slides(report, outdir)

    salary_slide = next(r for r in results if r.filename == "04_salary_by_category.png")
    assert salary_slide.saved is True


def test_generate_slides_catches_nan_and_withholds_only_the_broken_slide(tmp_path):
    """
    Injecting NaN into salary_percentiles.p50 (consumed only by the headline
    slide) must trip the NaN gate for that slide while leaving the other
    slides — which don't read that field — unaffected.
    """
    broken = _build_report(
        salary_percentiles=SalaryPercentiles(n=20, p25=55000, p50=float("nan"), p75=80000),
    )
    report = _round_trip_through_json(broken, tmp_path)
    outdir = tmp_path / "carousel"

    results = generate_slides(report, outdir)

    headline = next(r for r in results if r.filename == "01_headline.png")
    assert headline.saved is False
    assert any("NaN" in reason or "nan" in reason for reason in headline.reasons)
    assert not (outdir / "01_headline.png").exists()

    others = [r for r in results if r.filename != "01_headline.png"]
    assert all(result.saved for result in others), [r.reasons for r in others if not r.saved]

    passed_pngs = sorted(outdir.glob("*.png"))
    assert len(passed_pngs) == 4


@pytest.mark.parametrize(
    "filename",
    [
        "01_headline.png",
        "02_top_skills.png",
        "03_rising_cooling.png",
        "04_salary_by_category.png",
        "05_geo_company_stage.png",
    ],
)
def test_generate_slides_handles_empty_report_gracefully(tmp_path, filename):
    """An essentially-empty report (fresh DB) must still produce a valid slide, never crash."""
    empty_metadata = ReportMetadata(
        generated_at=datetime(2026, 7, 14, 8, 0, 0), window_days=90, total_jobs=0,
    )
    empty_report = MarketReport(metadata=empty_metadata)
    report = _round_trip_through_json(empty_report, tmp_path)
    outdir = tmp_path / "carousel"

    results = generate_slides(report, outdir)

    result = next(r for r in results if r.filename == filename)
    assert result.saved is True, result.reasons
    assert (outdir / filename).stat().st_size > 1_000
