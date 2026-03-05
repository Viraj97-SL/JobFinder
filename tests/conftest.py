"""
JobForge AI — Test Configuration & Shared Fixtures.

All external API calls are mocked. No real HTTP requests in unit tests.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.jobforge.models.job import RawJob


@pytest.fixture
def sample_raw_jobs() -> list[RawJob]:
    """A batch of realistic test jobs with various visa/startup signals."""
    return [
        RawJob(
            job_id="test_001",
            title="AI Engineer",
            company="DeepMind",
            location="London",
            salary_min=65000,
            salary_max=95000,
            description="We are looking for an AI Engineer with experience in LangGraph, "
                        "multi-agent systems, and production ML. Visa sponsorship available "
                        "for the right candidate. Experience with PyTorch and FastAPI preferred.",
            url="https://deepmind.google/careers/ai-engineer",
            source="adzuna",
            posted_date=date(2026, 3, 1),
            work_model="hybrid",
        ),
        RawJob(
            job_id="test_002",
            title="Data Scientist",
            company="Arclet AI",
            location="London, Remote",
            salary_min=45000,
            salary_max=65000,
            description="Early-stage startup looking for a Data Scientist. Series A funded. "
                        "Python, SQL, machine learning required. We are a seed stage company "
                        "building the future of AI-powered analytics.",
            url="https://arclet.ai/careers/ds",
            source="wellfound",
            posted_date=date(2026, 3, 2),
            work_model="remote",
        ),
        RawJob(
            job_id="test_003",
            title="ML Engineer",
            company="UK Gov Department",
            location="London",
            description="ML Engineer for classified project. Must be a UK citizen. "
                        "SC clearance required. Python, TensorFlow, Kubernetes.",
            url="https://gov.uk/careers/ml",
            source="reed",
            posted_date=date(2026, 3, 3),
            work_model="onsite",
        ),
        RawJob(
            job_id="test_004",
            title="Computer Vision Engineer",
            company="Wayve",
            location="London",
            salary_min=70000,
            salary_max=100000,
            description="Join our autonomous driving team. Computer Vision, PyTorch, "
                        "YOLOv8, 3D perception. Skilled Worker Visa sponsorship offered. "
                        "Experience with medical imaging or autonomous systems a plus.",
            url="https://wayve.ai/careers/cv-engineer",
            source="career_pages",
            posted_date=date(2026, 3, 1),
            work_model="hybrid",
            is_startup=True,
            company_stage="series_b",
        ),
    ]


@pytest.fixture
def duplicate_jobs(sample_raw_jobs: list[RawJob]) -> list[RawJob]:
    """Same jobs with different IDs/sources — tests deduplication."""
    dupes = []
    for job in sample_raw_jobs[:2]:
        dupe = job.model_copy()
        dupe.job_id = f"dupe_{job.job_id}"
        dupe.source = "reed"
        dupes.append(dupe)
    return sample_raw_jobs + dupes
