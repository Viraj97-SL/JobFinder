"""
JobForge AI — Persistent Memory Layer.

SQLAlchemy-backed stores that work with both SQLite (local dev) and
PostgreSQL (Railway production). The database URL is read from settings,
which picks up DATABASE_URL from the environment on Railway.

Stores:
1. Deduplication: Track seen job hashes across runs
2. Run History: Store pipeline telemetry and score distributions
3. Score Calibration: Historical match score data for self-reflection
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import structlog
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from jobforge.config.settings import settings

logger = structlog.get_logger(__name__)


def _make_sync_url(url: str) -> str:
    """Convert async driver URL to synchronous equivalent for SQLAlchemy."""
    return (
        url
        .replace("postgresql+asyncpg://", "postgresql+psycopg2://")
        .replace("sqlite+aiosqlite://", "sqlite://")
    )


_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        sync_url = _make_sync_url(settings.database_url)
        _engine = create_engine(sync_url, pool_pre_ping=True)
        # Log without credentials
        safe_url = sync_url.split("@")[-1] if "@" in sync_url else sync_url
        logger.info("database.engine.created", url=safe_url)
    return _engine


def init_database() -> None:
    """Create all tables if they don't exist. Safe to call multiple times."""
    engine = get_engine()
    is_pg = engine.dialect.name == "postgresql"
    id_col = "id BIGSERIAL PRIMARY KEY" if is_pg else "id INTEGER PRIMARY KEY AUTOINCREMENT"

    ddl_statements = [
        """
        CREATE TABLE IF NOT EXISTS seen_jobs (
            dedup_hash TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            title TEXT NOT NULL,
            company TEXT NOT NULL,
            source TEXT NOT NULL,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            times_seen INTEGER DEFAULT 1
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS run_history (
            run_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            status TEXT DEFAULT 'running',
            total_scraped INTEGER DEFAULT 0,
            total_qualified INTEGER DEFAULT 0,
            total_cvs_generated INTEGER DEFAULT 0,
            average_score REAL DEFAULT 0.0,
            email_sent INTEGER DEFAULT 0,
            llm_cost_usd REAL DEFAULT 0.0,
            metadata_json TEXT
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS score_history (
            {id_col},
            run_id TEXT NOT NULL,
            job_hash TEXT NOT NULL,
            overall_score REAL NOT NULL,
            cv_variant TEXT,
            offers_sponsorship INTEGER,
            is_startup INTEGER,
            scored_at TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES run_history(run_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS score_cache (
            job_hash TEXT PRIMARY KEY,
            score_json TEXT NOT NULL,
            cached_at TEXT NOT NULL,
            expires_at TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_seen_jobs_hash ON seen_jobs(dedup_hash)",
        "CREATE INDEX IF NOT EXISTS idx_score_history_run ON score_history(run_id)",
        "CREATE INDEX IF NOT EXISTS idx_score_cache_expires ON score_cache(expires_at)",
    ]

    with engine.connect() as conn:
        for stmt in ddl_statements:
            conn.execute(text(stmt))
        conn.commit()

    logger.info("database.init.complete")


class DedupStore:
    """
    Cross-run job deduplication using content hashes.

    A job is considered a duplicate if its (title + company + location) hash
    has been seen in any previous run. Duplicates are logged but not reprocessed.
    """

    def __init__(self) -> None:
        self._engine = get_engine()

    def is_seen(self, dedup_hash: str) -> bool:
        with self._engine.connect() as conn:
            row = conn.execute(
                text("SELECT 1 FROM seen_jobs WHERE dedup_hash = :h"),
                {"h": dedup_hash},
            ).fetchone()
        return row is not None

    def mark_seen(self, dedup_hash: str, job_id: str, title: str, company: str, source: str) -> None:
        now = datetime.utcnow().isoformat()
        with self._engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO seen_jobs
                        (dedup_hash, job_id, title, company, source, first_seen_at, last_seen_at, times_seen)
                    VALUES (:h, :jid, :t, :co, :src, :now, :now, 1)
                    ON CONFLICT(dedup_hash) DO UPDATE SET
                        last_seen_at = :now,
                        times_seen   = seen_jobs.times_seen + 1
                """),
                {"h": dedup_hash, "jid": job_id, "t": title, "co": company, "src": source, "now": now},
            )
            conn.commit()

    def filter_new(self, jobs: list) -> list:
        """Return only jobs not previously seen. Mark all as seen."""
        new_jobs = []
        for job in jobs:
            if not self.is_seen(job.dedup_hash):
                self.mark_seen(job.dedup_hash, job.job_id, job.title, job.company, job.source)
                new_jobs.append(job)
        return new_jobs

    def get_total_seen(self) -> int:
        with self._engine.connect() as conn:
            row = conn.execute(text("SELECT COUNT(*) FROM seen_jobs")).fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        pass  # Engine is shared; lifecycle managed at module level


class RunHistory:
    """Track pipeline run telemetry over time."""

    def __init__(self) -> None:
        self._engine = get_engine()

    def start_run(self, run_id: str) -> None:
        with self._engine.connect() as conn:
            conn.execute(
                text("INSERT INTO run_history (run_id, started_at) VALUES (:rid, :ts)"),
                {"rid": run_id, "ts": datetime.utcnow().isoformat()},
            )
            conn.commit()

    def complete_run(self, run_id: str, **kwargs) -> None:
        params = dict(kwargs)
        params["run_id"] = run_id
        params["_completed_at"] = datetime.utcnow().isoformat()
        set_clauses = ", ".join(f"{k} = :{k}" for k in kwargs)
        with self._engine.connect() as conn:
            conn.execute(
                text(
                    f"UPDATE run_history SET completed_at = :_completed_at, "
                    f"status = 'complete', {set_clauses} WHERE run_id = :run_id"
                ),
                params,
            )
            conn.commit()

    def log_scores(self, run_id: str, scores: list[dict]) -> None:
        """Batch insert score history for calibration analysis."""
        now = datetime.utcnow().isoformat()
        rows = [
            {
                "run_id": run_id,
                "job_hash": s["job_hash"],
                "overall_score": s["overall_score"],
                "cv_variant": s.get("cv_variant"),
                "offers_sponsorship": s.get("offers_sponsorship", 0),
                "is_startup": s.get("is_startup", 0),
                "scored_at": now,
            }
            for s in scores
        ]
        with self._engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO score_history
                        (run_id, job_hash, overall_score, cv_variant,
                         offers_sponsorship, is_startup, scored_at)
                    VALUES
                        (:run_id, :job_hash, :overall_score, :cv_variant,
                         :offers_sponsorship, :is_startup, :scored_at)
                """),
                rows,
            )
            conn.commit()

    def get_recent_score_stats(self, last_n_runs: int = 5) -> dict:
        """Get aggregate score statistics for self-reflection / calibration."""
        with self._engine.connect() as conn:
            row = conn.execute(
                text("""
                    SELECT AVG(overall_score),
                           MIN(overall_score),
                           MAX(overall_score),
                           COUNT(*)
                    FROM score_history
                    WHERE run_id IN (
                        SELECT run_id FROM run_history
                        ORDER BY started_at DESC LIMIT :n
                    )
                """),
                {"n": last_n_runs},
            ).fetchone()

        if row and row[3] > 0:
            return {
                "avg_score": round(row[0], 1),
                "min_score": round(row[1], 1),
                "max_score": round(row[2], 1),
                "total_scored": row[3],
            }
        return {"avg_score": 0, "min_score": 0, "max_score": 0, "total_scored": 0}

    def close(self) -> None:
        pass


class ScoreCache:
    """
    Persistent LLM score cache keyed on job content hash.

    If a job re-appears within TTL_DAYS (e.g. a re-posted listing),
    we return the cached MatchScore fields instead of calling the LLM again.
    Reduces Gemini Flash calls by ~30-50% on repeat scrapes.
    """

    TTL_DAYS = 14  # 2 weeks — covers weekly run cadence + buffer

    def __init__(self) -> None:
        self._engine = get_engine()

    def get(self, job_hash: str) -> dict | None:
        """Return cached score fields if present and not expired, else None."""
        with self._engine.connect() as conn:
            row = conn.execute(
                text("SELECT score_json FROM score_cache WHERE job_hash = :h AND expires_at > :now"),
                {"h": job_hash, "now": datetime.utcnow().isoformat()},
            ).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, job_hash: str, score_data: dict) -> None:
        """Cache score fields for a job with a TTL_DAYS expiry."""
        now = datetime.utcnow()
        expires = (now + timedelta(days=self.TTL_DAYS)).isoformat()
        with self._engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO score_cache (job_hash, score_json, cached_at, expires_at)
                    VALUES (:h, :sj, :now, :exp)
                    ON CONFLICT(job_hash) DO UPDATE SET
                        score_json = excluded.score_json,
                        cached_at  = excluded.cached_at,
                        expires_at = excluded.expires_at
                """),
                {"h": job_hash, "sj": json.dumps(score_data), "now": now.isoformat(), "exp": expires},
            )
            conn.commit()

    def evict_expired(self) -> int:
        """Delete expired cache entries. Returns count removed."""
        with self._engine.connect() as conn:
            result = conn.execute(
                text("DELETE FROM score_cache WHERE expires_at <= :now"),
                {"now": datetime.utcnow().isoformat()},
            )
            conn.commit()
        return result.rowcount

    def close(self) -> None:
        pass
