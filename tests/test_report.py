"""
JobForge AI — MarketReport Schema & Builder Tests.

MarketReport is meant to be the single source of truth the LinkedIn carousel,
email digest, and marketforge.digital all consume, instead of each surface
re-deriving figures from separate MarketAnalyzer calls (the root cause of the
salary-median divergence that shipped inconsistently before).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy import text

import jobforge.memory.dedup_store as dedup_store_module
from jobforge.analytics.market_analyzer import MarketAnalyzer
from jobforge.memory.dedup_store import get_engine, init_database
from jobforge.models.report import MarketReport


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_jobforge.db"
    monkeypatch.setattr(dedup_store_module.settings, "database_url", f"sqlite:///{db_path}")
    dedup_store_module._engine = None
    init_database()
    yield
    dedup_store_module._engine = None


def _insert_row(job_id: str, scraped_at: datetime, **overrides) -> None:
    row = {
        "job_id": job_id,
        "dedup_hash": f"hash_{job_id}",
        "run_id": "run1",
        "title": "AI Engineer",
        "company": "Acme",
        "location": "London",
        "source": "adzuna",
        "salary_min": None,
        "salary_max": None,
        "salary_period": "unknown",
        "salary_annual_min": None,
        "salary_annual_max": None,
        "work_model": "remote",
        "company_stage": "series_a",
        "is_startup": 0,
        "role_category": "AI/LLM Engineer",
        "region": "London",
        "offers_sponsorship": None,
        "employer_is_licensed_sponsor": None,
        "matched_skills_json": "[]",
        "scraped_at": scraped_at.isoformat(),
    }
    row.update(overrides)
    with get_engine().connect() as conn:
        conn.execute(
            text("""
                INSERT INTO job_analytics
                    (job_id, dedup_hash, run_id, title, company, location, source,
                     salary_min, salary_max, salary_period, salary_annual_min, salary_annual_max,
                     work_model, company_stage, is_startup, role_category, region,
                     offers_sponsorship, employer_is_licensed_sponsor, matched_skills_json, scraped_at)
                VALUES
                    (:job_id, :dedup_hash, :run_id, :title, :company, :location, :source,
                     :salary_min, :salary_max, :salary_period, :salary_annual_min, :salary_annual_max,
                     :work_model, :company_stage, :is_startup, :role_category, :region,
                     :offers_sponsorship, :employer_is_licensed_sponsor, :matched_skills_json, :scraped_at)
            """),
            row,
        )
        conn.commit()


def test_build_market_report_returns_valid_schema_on_empty_db(isolated_db):
    """No jobs yet (fresh DB) — should still produce a schema-valid, empty report."""
    report = MarketAnalyzer().build_market_report()

    assert isinstance(report, MarketReport)
    assert report.metadata.total_jobs == 0
    assert report.metadata.divergence_flagged is False
    assert report.top_skills == []
    assert report.salary_percentiles.n == 0


def test_build_market_report_metadata_reflects_window_and_volume(isolated_db):
    now = datetime.utcnow()
    _insert_row("j1", now - timedelta(days=1))
    _insert_row("j2", now - timedelta(days=2))

    report = MarketAnalyzer(lookback_days=30).build_market_report()

    assert report.metadata.window_days == 30
    assert report.metadata.total_jobs == 2


def test_build_market_report_suppresses_low_sample_percentiles(isolated_db):
    """A single disclosed salary (n=1) must not surface a precise-looking median."""
    now = datetime.utcnow()
    _insert_row(
        "j1", now, salary_period="annual", salary_annual_min=90000, salary_annual_max=90000,
        role_category="MLOps",
    )

    report = MarketAnalyzer().build_market_report()

    assert report.salary_percentiles.n == 1
    assert report.salary_percentiles.p50 is None
    assert report.salary_by_category["MLOps"].p50 is None


def test_build_market_report_keeps_percentiles_at_or_above_min_sample(isolated_db):
    now = datetime.utcnow()
    for value in [40000, 50000, 60000, 70000, 80000]:
        _insert_row(
            f"j_{value}", now, salary_period="annual",
            salary_annual_min=value, salary_annual_max=value,
        )

    report = MarketAnalyzer().build_market_report()

    assert report.salary_percentiles.n == 5
    assert report.salary_percentiles.p50 == 60000


def test_build_market_report_flags_salary_divergence(isolated_db):
    now = datetime.utcnow()
    for value in [70000, 72000, 73000, 74000, 76000]:
        _insert_row(f"rolling_{value}", now - timedelta(days=30), salary_period="annual",
                    salary_annual_min=value, salary_annual_max=value)
    for value in [44000, 45000, 46000, 47000, 48000]:
        _insert_row(f"weekly_{value}", now - timedelta(days=1), salary_period="annual",
                    salary_annual_min=value, salary_annual_max=value)

    report = MarketAnalyzer(lookback_days=90).build_market_report()

    assert report.metadata.divergence_flagged is True
    assert report.salary_divergence.diverges is True


def test_build_market_report_includes_valid_funnel_field(isolated_db):
    """funnel (5.3) must always be present and shaped correctly, even with no run_history rows."""
    _insert_row("j1", datetime.utcnow())

    report = MarketAnalyzer().build_market_report()

    assert isinstance(report.funnel, dict)
    assert "runs" in report.funnel
    assert "aggregate" in report.funnel
    assert report.funnel["aggregate"]["n_runs"] == 0


def test_build_market_report_funnel_reflects_run_history(isolated_db):
    with get_engine().connect() as conn:
        conn.execute(
            text("""
                INSERT INTO run_history
                    (run_id, started_at, completed_at, status, total_scraped,
                     total_after_dedup, total_after_prescreen, total_scored, total_qualified)
                VALUES
                    ('run1', :ts, :ts, 'complete', 100, 80, 40, 40, 10)
            """),
            {"ts": datetime.utcnow().isoformat()},
        )
        conn.commit()

    report = MarketAnalyzer().build_market_report()

    assert report.funnel["aggregate"]["n_runs"] == 1
    assert report.funnel["runs"][0]["counts"]["total_scraped"] == 100
    assert report.funnel["runs"][0]["drop_rates"]["total_scraped_to_total_after_dedup"] == 20.0


def test_build_market_report_round_trips_through_json(isolated_db):
    """model_dump(mode="json") is what scripts/market_report.py --json emits."""
    _insert_row("j1", datetime.utcnow())

    report = MarketAnalyzer().build_market_report()
    dumped = report.model_dump(mode="json")

    assert isinstance(dumped["metadata"]["generated_at"], str)
    reparsed = MarketReport.model_validate(dumped)
    assert reparsed.metadata.total_jobs == 1
