"""
JobForge AI — Persistent Memory Layer.

SQLite-backed stores for:
1. Deduplication: Track seen job hashes across runs
2. Run History: Store pipeline telemetry and score distributions
3. Score Calibration: Historical match score data for self-reflection
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import structlog

from jobforge.config.settings import DATA_DIR

logger = structlog.get_logger(__name__)

DB_PATH = DATA_DIR / "jobforge.db"


def get_connection() -> sqlite3.Connection:
    """Get a SQLite connection with WAL mode for concurrent reads."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_database() -> None:
    """Create all tables if they don't exist. Safe to call multiple times."""
    conn = get_connection()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS seen_jobs (
                dedup_hash TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                title TEXT NOT NULL,
                company TEXT NOT NULL,
                source TEXT NOT NULL,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                times_seen INTEGER DEFAULT 1
            );

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
            );

            CREATE TABLE IF NOT EXISTS score_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                job_hash TEXT NOT NULL,
                overall_score REAL NOT NULL,
                cv_variant TEXT,
                offers_sponsorship INTEGER,
                is_startup INTEGER,
                scored_at TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES run_history(run_id)
            );

            -- LLM score cache: reuse scores for jobs seen again within TTL window
            CREATE TABLE IF NOT EXISTS score_cache (
                job_hash TEXT PRIMARY KEY,
                score_json TEXT NOT NULL,
                cached_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_seen_jobs_hash ON seen_jobs(dedup_hash);
            CREATE INDEX IF NOT EXISTS idx_score_history_run ON score_history(run_id);
            CREATE INDEX IF NOT EXISTS idx_score_cache_expires ON score_cache(expires_at);
        """)
        conn.commit()
        logger.info("database.init.complete", path=str(DB_PATH))
    finally:
        conn.close()


class DedupStore:
    """
    Cross-run job deduplication using content hashes.

    A job is considered a duplicate if its (title + company + location) hash
    has been seen in any previous run. Duplicates are logged but not reprocessed.
    """

    def __init__(self) -> None:
        self.conn = get_connection()

    def is_seen(self, dedup_hash: str) -> bool:
        """Check if a job hash exists in the store."""
        row = self.conn.execute(
            "SELECT 1 FROM seen_jobs WHERE dedup_hash = ?", (dedup_hash,)
        ).fetchone()
        return row is not None

    def mark_seen(self, dedup_hash: str, job_id: str, title: str, company: str, source: str) -> None:
        """Insert or update a seen job."""
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            """
            INSERT INTO seen_jobs (dedup_hash, job_id, title, company, source, first_seen_at, last_seen_at, times_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT(dedup_hash) DO UPDATE SET
                last_seen_at = ?,
                times_seen = times_seen + 1
            """,
            (dedup_hash, job_id, title, company, source, now, now, now),
        )
        self.conn.commit()

    def filter_new(self, jobs: list) -> list:
        """Return only jobs not previously seen. Mark all as seen."""
        new_jobs = []
        for job in jobs:
            if not self.is_seen(job.dedup_hash):
                self.mark_seen(job.dedup_hash, job.job_id, job.title, job.company, job.source)
                new_jobs.append(job)
        return new_jobs

    def get_total_seen(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM seen_jobs").fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        self.conn.close()


class RunHistory:
    """Track pipeline run telemetry over time."""

    def __init__(self) -> None:
        self.conn = get_connection()

    def start_run(self, run_id: str) -> None:
        self.conn.execute(
            "INSERT INTO run_history (run_id, started_at) VALUES (?, ?)",
            (run_id, datetime.utcnow().isoformat()),
        )
        self.conn.commit()

    def complete_run(self, run_id: str, **kwargs) -> None:
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values()) + [run_id]
        self.conn.execute(
            f"UPDATE run_history SET completed_at = ?, status = 'complete', {sets} WHERE run_id = ?",
            [datetime.utcnow().isoformat()] + values,
        )
        self.conn.commit()

    def log_scores(self, run_id: str, scores: list[dict]) -> None:
        """Batch insert score history for calibration analysis."""
        now = datetime.utcnow().isoformat()
        self.conn.executemany(
            """
            INSERT INTO score_history (run_id, job_hash, overall_score, cv_variant,
                                       offers_sponsorship, is_startup, scored_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (run_id, s["job_hash"], s["overall_score"], s.get("cv_variant"),
                 s.get("offers_sponsorship", 0), s.get("is_startup", 0), now)
                for s in scores
            ],
        )
        self.conn.commit()

    def get_recent_score_stats(self, last_n_runs: int = 5) -> dict:
        """Get aggregate score statistics for self-reflection / calibration."""
        rows = self.conn.execute(
            """
            SELECT AVG(overall_score) as avg_score,
                   MIN(overall_score) as min_score,
                   MAX(overall_score) as max_score,
                   COUNT(*) as total
            FROM score_history
            WHERE run_id IN (
                SELECT run_id FROM run_history
                ORDER BY started_at DESC LIMIT ?
            )
            """,
            (last_n_runs,),
        ).fetchone()

        if rows and rows["total"] > 0:
            return {
                "avg_score": round(rows["avg_score"], 1),
                "min_score": round(rows["min_score"], 1),
                "max_score": round(rows["max_score"], 1),
                "total_scored": rows["total"],
            }
        return {"avg_score": 0, "min_score": 0, "max_score": 0, "total_scored": 0}

    def close(self) -> None:
        self.conn.close()


class ScoreCache:
    """
    Persistent LLM score cache keyed on job content hash.

    If a job re-appears within TTL_DAYS (e.g. a re-posted listing),
    we return the cached MatchScore fields instead of calling the LLM again.
    Reduces Gemini Flash calls by ~30-50% on repeat scrapes.
    """

    TTL_DAYS = 7

    def __init__(self) -> None:
        self.conn = get_connection()

    def get(self, job_hash: str) -> dict | None:
        """Return cached score fields if present and not expired, else None."""
        row = self.conn.execute(
            "SELECT score_json FROM score_cache WHERE job_hash = ? AND expires_at > ?",
            (job_hash, datetime.utcnow().isoformat()),
        ).fetchone()
        return json.loads(row["score_json"]) if row else None

    def set(self, job_hash: str, score_data: dict) -> None:
        """Cache score fields for a job with a TTL_DAYS expiry."""
        now = datetime.utcnow()
        expires = (now + timedelta(days=self.TTL_DAYS)).isoformat()
        self.conn.execute(
            """
            INSERT INTO score_cache (job_hash, score_json, cached_at, expires_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(job_hash) DO UPDATE SET
                score_json = excluded.score_json,
                cached_at  = excluded.cached_at,
                expires_at = excluded.expires_at
            """,
            (job_hash, json.dumps(score_data), now.isoformat(), expires),
        )
        self.conn.commit()

    def evict_expired(self) -> int:
        """Delete expired cache entries. Returns count removed."""
        cur = self.conn.execute(
            "DELETE FROM score_cache WHERE expires_at <= ?",
            (datetime.utcnow().isoformat(),),
        )
        self.conn.commit()
        return cur.rowcount

    def close(self) -> None:
        self.conn.close()
