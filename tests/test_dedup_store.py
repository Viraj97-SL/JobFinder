"""
JobForge AI — Fuzzy Deduplication Tests.

Verifies the MinHash LSH near-duplicate layer added on top of the exact-hash
cross-run dedup in memory/dedup_store.py: reworded reposts of the same role
should collapse, unrelated roles must not.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import text

import src.jobforge.memory.dedup_store as dedup_store_module
from src.jobforge.memory.dedup_store import (
    DedupStore,
    ReportArchive,
    RunHistory,
    compute_minhash,
    get_engine,
    init_database,
)
from src.jobforge.models.job import RawJob
from src.jobforge.models.report import MarketReport, ReportMetadata


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


# ── Report Archive (4.4) ─────────────────────────────────────────────────────
# market_report_archive is append-only by design: every archive() call must
# INSERT a new row so a past week's published figures can be reconstructed
# exactly as published, even as the live DB's rolling windows move forward.


def _sample_report(total_jobs: int = 5, window_days: int = 90) -> MarketReport:
    return MarketReport(
        metadata=ReportMetadata(
            generated_at=datetime(2026, 7, 1, 12, 0, 0),
            window_days=window_days,
            total_jobs=total_jobs,
            divergence_flagged=False,
        ),
        top_skills=[("Python", 10), ("LangGraph", 5)],
    )


def test_archive_and_get_latest_round_trip(isolated_db):
    archive = ReportArchive()
    report = _sample_report(total_jobs=7)

    archive.archive(report)
    latest = archive.get_latest()

    assert latest is not None
    assert latest["window_days"] == 90
    assert latest["report"]["metadata"]["total_jobs"] == 7
    assert latest["report"]["top_skills"] == [["Python", 10], ["LangGraph", 5]]

    reparsed = MarketReport.model_validate(latest["report"])
    assert reparsed.metadata.total_jobs == 7


def test_get_latest_returns_none_when_archive_empty(isolated_db):
    assert ReportArchive().get_latest() is None


def test_archive_is_append_only_not_overwritten(isolated_db):
    """Archiving twice must produce two rows, never an UPDATE in place."""
    archive = ReportArchive()
    archive.archive(_sample_report(total_jobs=1))
    archive.archive(_sample_report(total_jobs=2))

    all_rows = archive.list_all()

    assert len(all_rows) == 2
    assert [row["report"]["metadata"]["total_jobs"] for row in all_rows] == [1, 2]
    assert archive.get_latest()["report"]["metadata"]["total_jobs"] == 2


def test_list_all_returns_oldest_first(isolated_db):
    archive = ReportArchive()
    archive.archive(_sample_report(total_jobs=1))
    archive.archive(_sample_report(total_jobs=2))
    archive.archive(_sample_report(total_jobs=3))

    rows = archive.list_all()

    assert [row["report"]["metadata"]["total_jobs"] for row in rows] == [1, 2, 3]


# ── Pipeline Funnel Columns (5.3) ────────────────────────────────────────────
# run_history gained total_after_dedup / total_after_prescreen / total_scored
# columns. complete_run() already builds a dynamic UPDATE from **kwargs, so
# passing these new field names through should just work with no code change.


def test_run_history_columns_exist_after_init(isolated_db):
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("PRAGMA table_info(run_history)")).fetchall()
    column_names = {row[1] for row in rows}

    assert {"total_after_dedup", "total_after_prescreen", "total_scored"}.issubset(column_names)


def test_complete_run_accepts_new_funnel_stage_kwargs(isolated_db):
    run_history = RunHistory()
    run_history.start_run("run_funnel_1")

    run_history.complete_run(
        "run_funnel_1",
        total_scraped=100,
        total_after_dedup=80,
        total_after_prescreen=40,
        total_scored=40,
        total_qualified=10,
    )

    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT total_scraped, total_after_dedup, total_after_prescreen, "
                "total_scored, total_qualified, status FROM run_history WHERE run_id = :rid"
            ),
            {"rid": "run_funnel_1"},
        ).fetchone()

    assert row == (100, 80, 40, 40, 10, "complete")
