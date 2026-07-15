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


def _insert_seen_job(dedup_hash: str, first_seen_at: datetime, last_seen_at: datetime, times_seen: int = 1) -> None:
    with get_engine().connect() as conn:
        conn.execute(
            text("""
                INSERT INTO seen_jobs
                    (dedup_hash, job_id, title, company, source, first_seen_at, last_seen_at, times_seen)
                VALUES (:h, :h, 'AI Engineer', 'Acme', 'adzuna', :first, :last, :times)
            """),
            {"h": dedup_hash, "first": first_seen_at.isoformat(), "last": last_seen_at.isoformat(), "times": times_seen},
        )
        conn.commit()


class TestSkillCoOccurrence:
    def test_top_pairs_meeting_min_count(self, isolated_db):
        now = datetime.utcnow()
        for i in range(3):
            _insert_row(f"j{i}", now, matched_skills_json=json.dumps(["Python", "LangChain", "PostgreSQL"]))
        _insert_row("solo", now, matched_skills_json=json.dumps(["Rust"]))

        pairs = MarketAnalyzer().skill_co_occurrence(min_count=3)

        pair_sets = [set(p["skills"]) for p in pairs]
        assert {"Python", "LangChain"} in pair_sets
        assert {"Python", "PostgreSQL"} in pair_sets

    def test_pairs_below_min_count_excluded(self, isolated_db):
        now = datetime.utcnow()
        _insert_row("j1", now, matched_skills_json=json.dumps(["Rust", "Kubernetes"]))

        pairs = MarketAnalyzer().skill_co_occurrence(min_count=3)

        assert pairs == []


class TestPostingPersistence:
    def test_median_days_by_category(self, isolated_db):
        now = datetime.utcnow()
        _insert_row("j1", now, role_category="MLOps")
        _insert_seen_job("hash_j1", now - timedelta(days=10), now)

        _insert_row("j2", now, role_category="Data Scientist")
        _insert_seen_job("hash_j2", now - timedelta(days=2), now)

        result = MarketAnalyzer().posting_persistence()

        assert result["MLOps"]["median_days"] == pytest.approx(10.0, abs=0.1)
        assert result["Data Scientist"]["median_days"] == pytest.approx(2.0, abs=0.1)


class TestSalaryBySkill:
    def test_skill_meeting_min_n_shows_premium(self, isolated_db):
        now = datetime.utcnow()
        for i in range(15):
            _insert_row(f"base_{i}", now, salary_period="annual",
                        salary_annual_min=50000, salary_annual_max=50000,
                        matched_skills_json=json.dumps(["Python"]))
        for i in range(15):
            _insert_row(f"k8s_{i}", now, salary_period="annual",
                        salary_annual_min=90000, salary_annual_max=90000,
                        matched_skills_json=json.dumps(["Kubernetes"]))

        result = MarketAnalyzer().salary_by_skill(min_n=15)

        assert result["Kubernetes"]["n"] == 15
        assert result["Kubernetes"]["premium_pct"] > 0

    def test_skill_below_min_n_suppressed(self, isolated_db):
        now = datetime.utcnow()
        for i in range(3):
            _insert_row(f"j_{i}", now, salary_period="annual",
                        salary_annual_min=90000, salary_annual_max=90000,
                        matched_skills_json=json.dumps(["RareSkill"]))

        result = MarketAnalyzer().salary_by_skill(min_n=15)

        assert "RareSkill" not in result


class TestWorkModelTrend:
    def test_weekly_split_by_model(self, isolated_db):
        now = datetime.utcnow()
        _insert_row("r1", now - timedelta(days=1), work_model="remote")
        _insert_row("r2", now - timedelta(days=2), work_model="remote")
        _insert_row("h1", now - timedelta(days=8), work_model="hybrid")

        result = MarketAnalyzer(lookback_days=30).work_model_trend(lookback_weeks=4)

        assert "remote" in result
        assert "hybrid" in result
        assert sum(result["remote"]) == 2
        assert sum(result["hybrid"]) == 1


class TestSponsorRegisterRate:
    def test_reports_licensed_vs_not_vs_unknown_separately(self, isolated_db):
        _insert_row("j1", datetime.utcnow(), employer_is_licensed_sponsor=1)
        _insert_row("j2", datetime.utcnow(), employer_is_licensed_sponsor=1)
        _insert_row("j3", datetime.utcnow(), employer_is_licensed_sponsor=0)
        _insert_row("j4", datetime.utcnow(), employer_is_licensed_sponsor=None)

        result = MarketAnalyzer().sponsor_register_rate()

        assert result["total"] == 4
        assert result["licensed_sponsor"] == 2
        assert result["licensed_sponsor_pct"] == 50.0
        assert result["not_licensed"] == 1
        assert result["unknown"] == 1

    def test_distinct_from_jd_stated_sponsorship(self, isolated_db):
        """An employer can hold a licence without the JD mentioning it, and vice versa."""
        _insert_row(
            "j1", datetime.utcnow(),
            offers_sponsorship=None,  # JD doesn't mention it
            employer_is_licensed_sponsor=1,  # but the register says they hold a licence
        )

        sponsorship = MarketAnalyzer().sponsorship_rate()
        register = MarketAnalyzer().sponsor_register_rate()

        assert sponsorship["sponsoring"] == 0
        assert register["licensed_sponsor"] == 1


def _insert_run(run_id: str, started_at: datetime, **overrides) -> None:
    row = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "completed_at": started_at.isoformat(),
        "status": "complete",
        "total_scraped": None,
        "total_after_dedup": None,
        "total_after_prescreen": None,
        "total_scored": None,
        "total_qualified": None,
    }
    row.update(overrides)
    with get_engine().connect() as conn:
        conn.execute(
            text("""
                INSERT INTO run_history
                    (run_id, started_at, completed_at, status, total_scraped,
                     total_after_dedup, total_after_prescreen, total_scored, total_qualified)
                VALUES
                    (:run_id, :started_at, :completed_at, :status, :total_scraped,
                     :total_after_dedup, :total_after_prescreen, :total_scored, :total_qualified)
            """),
            row,
        )
        conn.commit()


class TestPipelineFunnel:
    """
    Funnel: scraped -> dedup -> prescreen -> scored -> qualified (5.3).
    Drop rate between consecutive stages is 100 * (from - to) / from.
    """

    def test_drop_rate_math_single_run(self, isolated_db):
        _insert_run(
            "run1", datetime.utcnow(),
            total_scraped=100, total_after_dedup=80,
            total_after_prescreen=40, total_scored=40, total_qualified=10,
        )

        funnel = MarketAnalyzer().pipeline_funnel(lookback_runs=10)

        assert len(funnel["runs"]) == 1
        run = funnel["runs"][0]
        assert run["counts"] == {
            "total_scraped": 100,
            "total_after_dedup": 80,
            "total_after_prescreen": 40,
            "total_scored": 40,
            "total_qualified": 10,
        }
        assert run["drop_rates"]["total_scraped_to_total_after_dedup"] == 20.0
        assert run["drop_rates"]["total_after_dedup_to_total_after_prescreen"] == 50.0
        assert run["drop_rates"]["total_after_prescreen_to_total_scored"] == 0.0
        assert run["drop_rates"]["total_scored_to_total_qualified"] == 75.0

    def test_aggregate_averages_counts_and_drop_rates_across_runs(self, isolated_db):
        now = datetime.utcnow()
        _insert_run(
            "run1", now - timedelta(days=1),
            total_scraped=100, total_after_dedup=50,
            total_after_prescreen=50, total_scored=50, total_qualified=50,
        )
        _insert_run(
            "run2", now,
            total_scraped=200, total_after_dedup=100,
            total_after_prescreen=100, total_scored=100, total_qualified=100,
        )

        funnel = MarketAnalyzer().pipeline_funnel(lookback_runs=10)

        assert funnel["aggregate"]["n_runs"] == 2
        assert funnel["aggregate"]["counts"]["total_scraped"] == 150.0
        # Both runs drop 50% scraped -> dedup, so the average must also be 50%.
        assert funnel["aggregate"]["drop_rates"]["total_scraped_to_total_after_dedup"] == 50.0

    def test_missing_stage_counts_yield_none_drop_rate_not_zero(self, isolated_db):
        """A run that predates this instrumentation shouldn't fake a 0%/100% drop."""
        _insert_run(
            "run1", datetime.utcnow(),
            total_scraped=100, total_after_dedup=None,
            total_after_prescreen=None, total_scored=None, total_qualified=None,
        )

        funnel = MarketAnalyzer().pipeline_funnel(lookback_runs=10)

        run = funnel["runs"][0]
        assert run["counts"]["total_after_dedup"] is None
        assert run["drop_rates"]["total_scraped_to_total_after_dedup"] is None
        assert funnel["aggregate"]["counts"]["total_after_dedup"] is None

    def test_empty_run_history_returns_empty_funnel(self, isolated_db):
        funnel = MarketAnalyzer().pipeline_funnel()

        assert funnel == {"runs": [], "aggregate": {"counts": {}, "drop_rates": {}, "n_runs": 0}}

    def test_lookback_runs_limits_window(self, isolated_db):
        now = datetime.utcnow()
        for i in range(15):
            _insert_run(
                f"run{i}", now - timedelta(days=i),
                total_scraped=10, total_after_dedup=10,
                total_after_prescreen=10, total_scored=10, total_qualified=10,
            )

        funnel = MarketAnalyzer().pipeline_funnel(lookback_runs=5)

        assert len(funnel["runs"]) == 5


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
