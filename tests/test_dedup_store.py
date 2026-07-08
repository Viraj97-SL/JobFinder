"""
JobForge AI — Fuzzy Deduplication Tests.

Verifies the MinHash LSH near-duplicate layer added on top of the exact-hash
cross-run dedup in memory/dedup_store.py: reworded reposts of the same role
should collapse, unrelated roles must not.
"""

from __future__ import annotations

import pytest

import src.jobforge.memory.dedup_store as dedup_store_module
from src.jobforge.memory.dedup_store import DedupStore, compute_minhash, init_database
from src.jobforge.models.job import RawJob


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """Point the shared engine at a fresh temp SQLite file for this test only."""
    db_path = tmp_path / "test_jobforge.db"
    monkeypatch.setattr(dedup_store_module.settings, "database_url", f"sqlite:///{db_path}")
    dedup_store_module._engine = None
    init_database()
    yield
    dedup_store_module._engine = None


def _job(job_id: str, title: str, company: str, description: str, location: str = "London") -> RawJob:
    return RawJob(
        job_id=job_id,
        title=title,
        company=company,
        location=location,
        description=description,
        url=f"https://example.com/{job_id}",
        source="adzuna",
    )


def test_exact_duplicate_is_filtered(isolated_db):
    store = DedupStore()
    job = _job("j1", "AI Engineer", "Acme", "Build ML pipelines with PyTorch and FastAPI.")
    dupe = job.model_copy(update={"job_id": "j1_repost", "source": "reed"})

    assert len(store.filter_new([job])) == 1
    assert len(store.filter_new([dupe])) == 0


def test_near_duplicate_jd_is_caught_by_fuzzy_layer(isolated_db):
    store = DedupStore()
    shared_description = (
        "We need a senior machine learning engineer to build production ML "
        "systems using PyTorch, FastAPI, and Kubernetes for our content team."
    )
    original = _job("j1", "Senior ML Engineer", "Acme Corp", shared_description)
    reworded = _job(
        "j2", "Senior Machine Learning Engineer — Content Intelligence", "Acme Corp", shared_description
    )

    assert len(store.filter_new([original])) == 1
    assert len(store.filter_new([reworded])) == 0, (
        "reworded repost of the same JD at the same company should be a fuzzy duplicate"
    )


def test_unrelated_roles_are_not_collapsed(isolated_db):
    store = DedupStore()
    role_a = _job(
        "j1", "AI Engineer", "Acme Corp",
        "Build multi-agent LLM systems with LangGraph and LangChain for our platform team.",
    )
    role_b = _job(
        "j2", "Data Engineer", "Different Co",
        "Design and maintain Spark and Airflow pipelines for a petabyte-scale warehouse.",
    )

    assert len(store.filter_new([role_a])) == 1
    assert len(store.filter_new([role_b])) == 1, "unrelated roles must not be treated as duplicates"


def test_lsh_index_reloads_across_store_instances(isolated_db):
    """A fresh DedupStore() (e.g. the next pipeline run) must reload prior signatures."""
    shared_description = (
        "We need a senior machine learning engineer to build production ML "
        "systems using PyTorch, FastAPI, and Kubernetes for our content team."
    )
    original = _job("j1", "Senior ML Engineer", "Acme Corp", shared_description)
    DedupStore().filter_new([original])

    reworded = _job(
        "j2", "Senior Machine Learning Engineer — Content Intelligence", "Acme Corp", shared_description
    )
    result = DedupStore().filter_new([reworded])

    assert len(result) == 0


def test_compute_minhash_is_stable_for_identical_content():
    mh1 = compute_minhash("Acme", "Build ML systems with PyTorch.")
    mh2 = compute_minhash("Acme", "Build ML systems with PyTorch.")
    assert mh1.jaccard(mh2) == 1.0
