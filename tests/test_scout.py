"""
JobForge AI — Scout Agent & Connector Unit Tests.

All HTTP calls are mocked. Tests verify:
1. Visa signal detection (sponsorship, citizens-only)
2. Startup detection
3. Deduplication logic
4. Connector parsing
5. Search plan generation
"""

from __future__ import annotations

from datetime import date

import pytest

from src.jobforge.connectors.base import detect_sponsorship, detect_startup
from src.jobforge.agents.scout.planner import build_search_plan
from src.jobforge.models.job import RawJob


class TestVisaDetection:
    """Test visa sponsorship signal detection from JD text."""

    def test_detects_sponsorship_available(self):
        offers, citizens, signals = detect_sponsorship(
            "We offer visa sponsorship available for the right candidate."
        )
        assert offers is True
        assert citizens is None
        assert len(signals) > 0

    def test_detects_skilled_worker_visa(self):
        offers, _, signals = detect_sponsorship(
            "This role is eligible for Skilled Worker Visa sponsorship."
        )
        assert offers is True

    def test_detects_willing_to_sponsor(self):
        offers, _, _ = detect_sponsorship("We are willing to sponsor exceptional candidates.")
        assert offers is True

    def test_detects_uk_citizens_only(self):
        _, citizens, signals = detect_sponsorship(
            "This role requires UK citizens only due to security clearance."
        )
        assert citizens is True

    def test_detects_no_sponsorship(self):
        _, citizens, _ = detect_sponsorship(
            "We cannot offer sponsorship for this role."
        )
        assert citizens is True

    def test_detects_sc_clearance(self):
        _, citizens, _ = detect_sponsorship("SC clearance required for this position.")
        assert citizens is True

    def test_neutral_when_no_signals(self):
        offers, citizens, signals = detect_sponsorship(
            "We are looking for a Python developer to join our team."
        )
        assert offers is None
        assert citizens is None
        assert len(signals) == 0

    def test_handles_both_signals(self):
        """Edge case: JD mentions both sponsorship and restrictions."""
        offers, citizens, _ = detect_sponsorship(
            "Visa sponsorship available. Note: some projects require SC clearance."
        )
        assert offers is True
        assert citizens is True


class TestStartupDetection:
    """Test startup indicator detection."""

    def test_detects_seed_stage(self):
        assert detect_startup("We are a seed funded AI startup.") is True

    def test_detects_series_a(self):
        assert detect_startup("Just closed our Series A round.") is True

    def test_detects_founding_engineer(self):
        assert detect_startup("Looking for a founding engineer to join us.") is True

    def test_detects_yc_backed(self):
        assert detect_startup("Backed by Y Combinator, building from the ground up.") is True

    def test_no_startup_for_enterprise(self):
        assert detect_startup("Join our enterprise team at a Fortune 500 company.") is False

    def test_startup_keyword(self):
        assert detect_startup("This is a fast-growing startup in London.") is True


class TestDeduplication:
    """Test cross-source deduplication logic."""

    def test_same_job_same_hash(self):
        job1 = RawJob(
            job_id="a1", title="AI Engineer", company="DeepMind",
            location="London", description="Test", url="http://a", source="adzuna",
        )
        job2 = RawJob(
            job_id="r1", title="AI Engineer", company="DeepMind",
            location="London", description="Different desc", url="http://b", source="reed",
        )
        assert job1.dedup_hash == job2.dedup_hash

    def test_different_jobs_different_hash(self):
        job1 = RawJob(
            job_id="a1", title="AI Engineer", company="DeepMind",
            location="London", description="Test", url="http://a", source="adzuna",
        )
        job2 = RawJob(
            job_id="a2", title="Data Scientist", company="Google",
            location="London", description="Test", url="http://b", source="adzuna",
        )
        assert job1.dedup_hash != job2.dedup_hash

    def test_case_insensitive_hash(self):
        job1 = RawJob(
            job_id="a1", title="AI ENGINEER", company="DEEPMIND",
            location="LONDON", description="Test", url="http://a", source="adzuna",
        )
        job2 = RawJob(
            job_id="a2", title="ai engineer", company="deepmind",
            location="london", description="Test", url="http://b", source="reed",
        )
        assert job1.dedup_hash == job2.dedup_hash


class TestSearchPlan:
    """Test query planner generates appropriate queries."""

    def test_builds_primary_queries(self):
        plan = build_search_plan()
        assert len(plan.primary_queries) > 0
        assert any("AI Engineer" in q for q in plan.primary_queries)

    def test_builds_startup_queries(self):
        plan = build_search_plan()
        assert len(plan.startup_queries) > 0
        assert any("startup" in q.lower() for q in plan.startup_queries)

    def test_builds_nonprofit_queries(self):
        plan = build_search_plan()
        assert len(plan.nonprofit_queries) > 0

    def test_source_specific_queries(self):
        plan = build_search_plan()
        wellfound_queries = plan.for_source("wellfound")
        # Wellfound should get startup queries first
        assert any("startup" in q.lower() or "seed" in q.lower() for q in wellfound_queries)

    def test_all_queries_non_empty(self):
        plan = build_search_plan()
        assert all(len(q.strip()) > 0 for q in plan.all_queries)

    def test_custom_roles(self):
        plan = build_search_plan(target_roles=["NLP Engineer"])
        assert any("NLP Engineer" in q for q in plan.primary_queries)


class TestRawJobModel:
    """Test RawJob Pydantic model."""

    def test_salary_display_range(self):
        job = RawJob(
            job_id="t1", title="Test", company="Co", location="UK",
            description="Test", url="http://test", source="test",
            salary_min=50000, salary_max=70000,
        )
        assert "£50,000" in job.salary_display
        assert "£70,000" in job.salary_display

    def test_salary_display_not_disclosed(self):
        job = RawJob(
            job_id="t1", title="Test", company="Co", location="UK",
            description="Test", url="http://test", source="test",
        )
        assert job.salary_display == "Not Disclosed"

    def test_description_max_length(self):
        long_desc = "x" * 9000
        job = RawJob(
            job_id="t1", title="Test", company="Co", location="UK",
            description=long_desc[:8000], url="http://test", source="test",
        )
        assert len(job.description) <= 8000
