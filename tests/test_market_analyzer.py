"""
JobForge AI — Market Analyzer Salary Stats Tests.

Verifies salary_stats() computes the headline figure purely from annual-period
salaries, so a contractor day rate or an unparseable garbage value never
corrupts the published median (the divergence bug the salary parser fixes).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest
from sqlalchemy import text

import jobforge.memory.dedup_store as dedup_store_module
from jobforge.analytics.market_analyzer import MarketAnalyzer
from jobforge.memory.dedup_store import AnalyticsStore, get_engine, init_database
from jobforge.models.job import RawJob


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_jobforge.db"
    monkeypatch.setattr(dedup_store_module.settings, "database_url", f"sqlite:///{db_path}")
    dedup_store_module._engine = None
    init_database()
    yield
    dedup_store_module._engine = None


def _job(job_id: str, title: str, salary_min, salary_max, salary_period) -> RawJob:
    return RawJob(
        job_id=job_id,
        title=title,
        company="Acme",
        location="London",
        salary_min=salary_min,
        salary_max=salary_max,
        salary_period=salary_period,
        description="Great role.",
        url=f"https://example.com/{job_id}",
        source="adzuna",
    )


def test_salary_stats_excludes_day_rate_and_garbage(isolated_db):
    store = AnalyticsStore()
    store.log_job(_job("j1", "AI Engineer", 60000, 80000, "annual"), run_id="run1")
    store.log_job(_job("j2", "Contract ML Engineer", 700, 800, "daily"), run_id="run1")
    store.log_job(_job("j3", "Garbage Listing", 0, 0, "unknown"), run_id="run1")

    stats = MarketAnalyzer().salary_stats()

    assert stats["disclosed_count"] == 1, "only the genuinely annual role should count"
    assert stats["avg_min"] == 60000
    assert stats["avg_max"] == 80000


def test_salary_stats_empty_when_no_disclosed_salary(isolated_db):
    store = AnalyticsStore()
    store.log_job(_job("j1", "AI Engineer", None, None, "unknown"), run_id="run1")

    stats = MarketAnalyzer().salary_stats()

    assert stats == {"disclosed_count": 0}


# ── Raw row insertion helper ────────────────────────────────────────────────
# metric_deltas / skill_trajectories need precise control over scraped_at
# and matched_skills_json that AnalyticsStore.log_job (which always stamps
# "now") can't give us — insert directly into job_analytics instead.

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
                     offers_sponsorship, matched_skills_json, scraped_at)
                VALUES
                    (:job_id, :dedup_hash, :run_id, :title, :company, :location, :source,
                     :salary_min, :salary_max, :salary_period, :salary_annual_min, :salary_annual_max,
                     :work_model, :company_stage, :is_startup, :role_category, :region,
                     :offers_sponsorship, :matched_skills_json, :scraped_at)
            """),
            row,
        )
        conn.commit()


class TestMetricDeltas:
    def test_wow_total_volume_math(self, isolated_db):
        now = datetime.utcnow()
        # Current week: 3 jobs. Previous week: 1 job.
        _insert_row("cur1", now - timedelta(days=1))
        _insert_row("cur2", now - timedelta(days=2))
        _insert_row("cur3", now - timedelta(days=3))
        _insert_row("prev1", now - timedelta(days=9))

        result = MarketAnalyzer().metric_deltas("total_volume", weeks=1)

        assert result["current"] == 3
        assert result["previous"] == 1
        assert result["abs_change"] == 2
        assert result["pct_change"] == 200.0
        assert result["direction"] == "up"

    def test_wow_salary_median(self, isolated_db):
        now = datetime.utcnow()
        _insert_row(
            "cur1", now - timedelta(days=1),
            salary_period="annual", salary_annual_min=60000, salary_annual_max=60000,
        )
        _insert_row(
            "prev1", now - timedelta(days=9),
            salary_period="annual", salary_annual_min=50000, salary_annual_max=50000,
        )

        result = MarketAnalyzer().metric_deltas("salary_median", weeks=1)

        assert result["current"] == 60000
        assert result["previous"] == 50000
        assert result["direction"] == "up"

    def test_missing_previous_window_yields_unknown_direction(self, isolated_db):
        now = datetime.utcnow()
        _insert_row("cur1", now - timedelta(days=1))

        result = MarketAnalyzer().metric_deltas("total_volume", weeks=1)

        assert result["current"] == 1
        assert result["previous"] is None
        assert result["direction"] == "unknown"

    def test_skill_mention_delta(self, isolated_db):
        now = datetime.utcnow()
        _insert_row("cur1", now - timedelta(days=1), matched_skills_json=json.dumps(["Python", "LangGraph"]))
        _insert_row("cur2", now - timedelta(days=2), matched_skills_json=json.dumps(["Python"]))
        _insert_row("prev1", now - timedelta(days=9), matched_skills_json=json.dumps(["Python"]))

        result = MarketAnalyzer().metric_deltas("Python", weeks=1)

        assert result["current"] == 2
        assert result["previous"] == 1
        assert result["direction"] == "up"


class TestSkillTrajectories:
    def test_new_skill_classification(self, isolated_db):
        now = datetime.utcnow()
        # "CI/CD" absent for 4 weeks, then appears in the most recent week.
        for week_offset in [28, 21, 14, 7]:
            _insert_row(
                f"old_{week_offset}", now - timedelta(days=week_offset),
                matched_skills_json=json.dumps(["Python"] * 5),  # padding so Python is the top skill too
            )
        _insert_row("new1", now - timedelta(days=1), matched_skills_json=json.dumps(["CI/CD"]))

        trajectories = MarketAnalyzer(lookback_days=35).skill_trajectories(lookback_weeks=5, top_n=20)

        assert "CI/CD" in trajectories
        assert trajectories["CI/CD"]["trend"] == "New"

    def test_steady_climb_classification(self, isolated_db):
        now = datetime.utcnow()
        # Python mentioned in a steadily increasing number of jobs each week.
        weekly_mention_counts = [2, 4, 6, 8, 10, 12]
        for i, count in enumerate(weekly_mention_counts):
            week_offset = (len(weekly_mention_counts) - i) * 7
            for j in range(count):
                _insert_row(
                    f"py_{i}_{j}", now - timedelta(days=week_offset - 1),
                    matched_skills_json=json.dumps(["Python"]),
                )

        trajectories = MarketAnalyzer(lookback_days=60).skill_trajectories(lookback_weeks=8, top_n=5)

        assert trajectories["Python"]["trend"] == "Accelerating"


class TestRoleAndSalarySegmentation:
    def test_role_category_distribution(self, isolated_db):
        _insert_row("j1", datetime.utcnow(), role_category="AI/LLM Engineer")
        _insert_row("j2", datetime.utcnow(), role_category="AI/LLM Engineer")
        _insert_row("j3", datetime.utcnow(), role_category="Data Scientist")

        dist = MarketAnalyzer().role_category_distribution()

        assert dist["AI/LLM Engineer"] == 2
        assert dist["Data Scientist"] == 1

    def test_salary_percentiles_true_quantiles(self, isolated_db):
        now = datetime.utcnow()
        for value in [40000, 50000, 60000, 70000, 80000]:
            _insert_row(
                f"j_{value}", now, salary_period="annual",
                salary_annual_min=value, salary_annual_max=value,
            )

        percentiles = MarketAnalyzer().salary_percentiles()

        assert percentiles["n"] == 5
        assert percentiles["p50"] == 60000

    def test_salary_by_category_segments_correctly(self, isolated_db):
        now = datetime.utcnow()
        _insert_row("j1", now, role_category="MLOps", salary_period="annual",
                    salary_annual_min=90000, salary_annual_max=90000)
        _insert_row("j2", now, role_category="Data Scientist", salary_period="annual",
                    salary_annual_min=50000, salary_annual_max=50000)

        by_category = MarketAnalyzer().salary_by_category()

        assert by_category["MLOps"]["p50"] == 90000
        assert by_category["Data Scientist"]["p50"] == 50000

    def test_salary_by_seniority_segments_correctly(self, isolated_db):
        now = datetime.utcnow()
        _insert_row("j1", now, title="Senior AI Engineer", salary_period="annual",
                    salary_annual_min=90000, salary_annual_max=90000)
        _insert_row("j2", now, title="Junior AI Engineer", salary_period="annual",
                    salary_annual_min=40000, salary_annual_max=40000)

        by_seniority = MarketAnalyzer().salary_by_seniority()

        assert by_seniority["Senior"]["p50"] == 90000
        assert by_seniority["Junior"]["p50"] == 40000

    def test_salary_divergence_check_trips_on_real_world_gap(self, isolated_db):
        now = datetime.utcnow()
        # 90-day rolling distribution centred around ~73k
        for value in [70000, 72000, 73000, 74000, 76000]:
            _insert_row(f"rolling_{value}", now - timedelta(days=30), salary_period="annual",
                        salary_annual_min=value, salary_annual_max=value)
        # This week's snapshot centred around ~46k
        for value in [44000, 45000, 46000, 47000, 48000]:
            _insert_row(f"weekly_{value}", now - timedelta(days=1), salary_period="annual",
                        salary_annual_min=value, salary_annual_max=value)

        result = MarketAnalyzer(lookback_days=90).salary_divergence_check()

        assert result["diverges"] is True


class TestGeographyAndCompanyStage:
    def test_geographic_distribution_uses_stored_region(self, isolated_db):
        _insert_row("j1", datetime.utcnow(), region="London")
        _insert_row("j2", datetime.utcnow(), region="London")
        _insert_row("j3", datetime.utcnow(), region="Manchester")

        dist = dict(MarketAnalyzer().geographic_distribution())

        assert dist["London"] == 2
        assert dist["Manchester"] == 1

    def test_geographic_distribution_falls_back_for_missing_region(self, isolated_db):
        """Rows scraped before the region column existed have region=NULL."""
        _insert_row("j1", datetime.utcnow(), region=None, location="Cambridge Science Park")

        dist = dict(MarketAnalyzer().geographic_distribution())

        assert dist["Cambridge"] == 1

    def test_company_stage_distribution(self, isolated_db):
        _insert_row("j1", datetime.utcnow(), company_stage="series_a")
        _insert_row("j2", datetime.utcnow(), company_stage="series_a")
        _insert_row("j3", datetime.utcnow(), company_stage="enterprise")

        dist = MarketAnalyzer().company_stage_distribution()

        assert dist["series_a"] == 2
        assert dist["enterprise"] == 1


class TestRisingCoolingSkills:
    def test_rising_and_cooling_lists(self, isolated_db):
        now = datetime.utcnow()
        rising_counts = [5, 10, 15]     # weeks -2, -1, 0 (current)
        cooling_counts = [15, 10, 5]
        for i, count in enumerate(rising_counts):
            week_offset = (len(rising_counts) - 1 - i) * 7
            for j in range(count):
                _insert_row(f"rising_{i}_{j}", now - timedelta(days=week_offset),
                            matched_skills_json=json.dumps(["RisingSkill"]))
        for i, count in enumerate(cooling_counts):
            week_offset = (len(cooling_counts) - 1 - i) * 7
            for j in range(count):
                _insert_row(f"cooling_{i}_{j}", now - timedelta(days=week_offset),
                            matched_skills_json=json.dumps(["CoolingSkill"]))

        result = MarketAnalyzer(lookback_days=30).rising_cooling_skills(lookback_weeks=3, top_n=10)

        assert "RisingSkill" in result["rising"]
        assert "CoolingSkill" in result["cooling"]
