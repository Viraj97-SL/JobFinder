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
import re
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import structlog
from datasketch import MinHash, MinHashLSH
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from jobforge.analytics.role_classifier import classify_role_category
from jobforge.config.settings import settings
from jobforge.utils.geography import normalize_location
from jobforge.utils.salary_parser import normalize_to_annual

logger = structlog.get_logger(__name__)

# ── Fuzzy Dedup (MinHash LSH) ────────────────────────────────────────────────
# Exact-hash dedup (title+company+location) misses reworded reposts of the
# same role across boards. This layer catches those via near-duplicate JD text.
#
# Title is deliberately excluded from the shingle set. Reworded titles are
# exactly the noise this layer needs to tolerate ("Senior ML Engineer" vs
# "Senior Machine Learning Engineer — Content Intelligence" at the same
# company). Word-shingling title and JD as one continuous token stream makes
# every shingle near that boundary shift with the title, which drags the
# real Jaccard score below threshold for precisely the reposts we want to
# catch. Company + JD-prefix text carries the actual duplicate signal.
_MINHASH_NUM_PERM = 128
_MINHASH_SCHEME = "affine32"
_FUZZY_JACCARD_THRESHOLD = 0.85
_SHINGLE_SIZE = 4  # word-level shingles
_DESCRIPTION_PREFIX_CHARS = 200


def _normalize_for_shingles(company: str, description: str | None) -> str:
    text_blob = f"{company} {(description or '')[:_DESCRIPTION_PREFIX_CHARS]}".lower()
    text_blob = re.sub(r"[^a-z0-9\s]", " ", text_blob)
    return re.sub(r"\s+", " ", text_blob).strip()


def _word_shingles(text_blob: str, k: int = _SHINGLE_SIZE) -> set[str]:
    tokens = text_blob.split()
    if not tokens:
        return set()
    if len(tokens) < k:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def compute_minhash(company: str, description: str | None) -> MinHash:
    """MinHash signature over word shingles of company + first 200 chars of JD."""
    mh = MinHash(num_perm=_MINHASH_NUM_PERM, scheme=_MINHASH_SCHEME)
    for shingle in _word_shingles(_normalize_for_shingles(company, description)):
        mh.update(shingle.encode("utf-8"))
    return mh


def _minhash_to_bytes(mh: MinHash) -> bytes:
    return mh.hashvalues.tobytes()


def _minhash_from_bytes(blob: bytes) -> MinHash:
    hashvalues = np.frombuffer(bytes(blob), dtype=np.uint32)
    return MinHash(num_perm=len(hashvalues), hashvalues=hashvalues, scheme=_MINHASH_SCHEME)


def _load_lsh_index(engine: Engine) -> MinHashLSH:
    """Rebuild the in-memory LSH index from every previously stored signature."""
    lsh = MinHashLSH(threshold=_FUZZY_JACCARD_THRESHOLD, num_perm=_MINHASH_NUM_PERM)
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT dedup_hash, minhash_signature FROM seen_jobs WHERE minhash_signature IS NOT NULL")
        ).fetchall()
    for dedup_hash, blob in rows:
        lsh.insert(dedup_hash, _minhash_from_bytes(blob))
    return lsh


def _ensure_column(engine: Engine, table: str, column: str, sqlite_type: str, pg_type: str) -> None:
    """Add a column if it doesn't already exist yet (SQLite/Postgres-safe migration)."""
    is_pg = engine.dialect.name == "postgresql"
    with engine.connect() as conn:
        if is_pg:
            exists = conn.execute(
                text(
                    "SELECT 1 FROM information_schema.columns "
                    "WHERE table_name = :t AND column_name = :c"
                ),
                {"t": table, "c": column},
            ).fetchone()
        else:
            rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
            exists = any(row[1] == column for row in rows)
        if not exists:
            col_type = pg_type if is_pg else sqlite_type
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"))
            conn.commit()


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
        """
        CREATE TABLE IF NOT EXISTS job_analytics (
            job_id TEXT PRIMARY KEY,
            dedup_hash TEXT NOT NULL,
            run_id TEXT NOT NULL,
            title TEXT NOT NULL,
            company TEXT NOT NULL,
            location TEXT,
            source TEXT NOT NULL,
            salary_min REAL,
            salary_max REAL,
            work_model TEXT,
            company_stage TEXT,
            is_startup INTEGER DEFAULT 0,
            offers_sponsorship INTEGER,
            overall_score REAL,
            cv_variant TEXT,
            matched_skills_json TEXT,
            scraped_at TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_seen_jobs_hash ON seen_jobs(dedup_hash)",
        "CREATE INDEX IF NOT EXISTS idx_score_history_run ON score_history(run_id)",
        "CREATE INDEX IF NOT EXISTS idx_score_cache_expires ON score_cache(expires_at)",
        "CREATE INDEX IF NOT EXISTS idx_job_analytics_run ON job_analytics(run_id)",
        "CREATE INDEX IF NOT EXISTS idx_job_analytics_scraped ON job_analytics(scraped_at)",
    ]

    with engine.connect() as conn:
        for stmt in ddl_statements:
            conn.execute(text(stmt))
        conn.commit()

    _ensure_column(engine, "seen_jobs", "minhash_signature", "BLOB", "BYTEA")
    _ensure_column(engine, "job_analytics", "salary_period", "TEXT", "TEXT")
    _ensure_column(engine, "job_analytics", "salary_annual_min", "REAL", "REAL")
    _ensure_column(engine, "job_analytics", "salary_annual_max", "REAL", "REAL")
    _ensure_column(engine, "job_analytics", "role_category", "TEXT", "TEXT")
    _ensure_column(engine, "job_analytics", "region", "TEXT", "TEXT")

    logger.info("database.init.complete")


class DedupStore:
    """
    Cross-run job deduplication using content hashes plus fuzzy near-duplicate detection.

    Two layers, cheapest first:
    1. Exact hash — (title + company + location) seen before → duplicate.
    2. Fuzzy (MinHash LSH) — JD text is a near-duplicate (Jaccard >= 0.85) of a
       previously seen job, e.g. the same role reposted under a reworded title
       across boards. Catches reposts that the exact hash misses.

    Duplicates from either layer are logged but not reprocessed.
    """

    def __init__(self) -> None:
        self._engine = get_engine()
        self._lsh = _load_lsh_index(self._engine)

    def is_seen(self, dedup_hash: str) -> bool:
        with self._engine.connect() as conn:
            row = conn.execute(
                text("SELECT 1 FROM seen_jobs WHERE dedup_hash = :h"),
                {"h": dedup_hash},
            ).fetchone()
        return row is not None

    def find_fuzzy_duplicate(self, minhash: MinHash) -> str | None:
        """Return the dedup_hash of a near-duplicate already in the LSH index, if any."""
        matches = self._lsh.query(minhash)
        return matches[0] if matches else None

    def mark_seen(
        self,
        dedup_hash: str,
        job_id: str,
        title: str,
        company: str,
        source: str,
        minhash: MinHash | None = None,
    ) -> None:
        now = datetime.utcnow().isoformat()
        signature = _minhash_to_bytes(minhash) if minhash is not None else None
        with self._engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO seen_jobs
                        (dedup_hash, job_id, title, company, source, first_seen_at, last_seen_at,
                         times_seen, minhash_signature)
                    VALUES (:h, :jid, :t, :co, :src, :now, :now, 1, :sig)
                    ON CONFLICT(dedup_hash) DO UPDATE SET
                        last_seen_at = :now,
                        times_seen   = seen_jobs.times_seen + 1
                """),
                {"h": dedup_hash, "jid": job_id, "t": title, "co": company, "src": source,
                 "now": now, "sig": signature},
            )
            conn.commit()

    def filter_new(self, jobs: list) -> list:
        """Return only jobs not previously seen (exact or fuzzy). Mark all as seen."""
        new_jobs = []
        for job in jobs:
            if self.is_seen(job.dedup_hash):
                continue

            minhash = compute_minhash(job.company, getattr(job, "description", None))
            fuzzy_match = self.find_fuzzy_duplicate(minhash)
            if fuzzy_match is not None:
                logger.info(
                    "dedup.fuzzy_duplicate",
                    job_id=job.job_id,
                    title=job.title,
                    company=job.company,
                    matched_hash=fuzzy_match,
                )
                self.mark_seen(job.dedup_hash, job.job_id, job.title, job.company, job.source, minhash)
                self._lsh.insert(job.dedup_hash, minhash)
                continue

            self.mark_seen(job.dedup_hash, job.job_id, job.title, job.company, job.source, minhash)
            self._lsh.insert(job.dedup_hash, minhash)
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


class AnalyticsStore:
    """
    Logs every scraped job to the job_analytics table for market trend analysis.

    Populated by the Scout Agent post-dedup so only fresh jobs are recorded.
    Score and CV variant columns are updated later by Matchmaker/Tailor.
    """

    def __init__(self) -> None:
        self._engine = get_engine()

    def log_job(self, job: Any, run_id: str, skill_inventory: Any = None) -> None:
        """Insert a new job into the analytics table. Skips on conflict (already logged)."""
        matched_skills: list[str] = []
        if skill_inventory is not None:
            try:
                all_skills = skill_inventory.get_all_skills_flat()
                jd_tokens = set(
                    t.lower()
                    for t in (job.title + " " + (job.description or "")).split()
                    if len(t) >= 2
                )
                matched_skills = [s for s in all_skills if s.lower() in jd_tokens]
            except Exception:
                pass

        salary_period = getattr(job, "salary_period", None) or "unknown"
        salary_annual_min, salary_annual_max = normalize_to_annual(
            getattr(job, "salary_min", None), getattr(job, "salary_max", None), salary_period
        )
        role_category = classify_role_category(job.title, getattr(job, "description", None))
        region = normalize_location(getattr(job, "location", None))

        now = datetime.utcnow().isoformat()
        with self._engine.connect() as conn:
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
                    ON CONFLICT(job_id) DO NOTHING
                """),
                {
                    "job_id": job.job_id,
                    "dedup_hash": job.dedup_hash,
                    "run_id": run_id,
                    "title": job.title,
                    "company": job.company,
                    "location": getattr(job, "location", None),
                    "source": job.source,
                    "salary_min": getattr(job, "salary_min", None),
                    "salary_max": getattr(job, "salary_max", None),
                    "salary_period": salary_period,
                    "salary_annual_min": salary_annual_min,
                    "salary_annual_max": salary_annual_max,
                    "work_model": getattr(job, "work_model", None),
                    "company_stage": getattr(job, "company_stage", None),
                    "is_startup": int(getattr(job, "is_startup", False)),
                    "role_category": role_category,
                    "region": region,
                    "offers_sponsorship": (
                        1 if getattr(job, "offers_sponsorship", None) is True
                        else (0 if getattr(job, "offers_sponsorship", None) is False else None)
                    ),
                    "matched_skills_json": json.dumps(matched_skills),
                    "scraped_at": now,
                },
            )
            conn.commit()

    def update_score(self, job_id: str, overall_score: float, cv_variant: str | None = None) -> None:
        """Update analytics record with Matchmaker score and (optionally) CV variant."""
        with self._engine.connect() as conn:
            conn.execute(
                text("""
                    UPDATE job_analytics
                    SET overall_score = :score, cv_variant = COALESCE(:variant, cv_variant)
                    WHERE job_id = :job_id
                """),
                {"score": overall_score, "variant": cv_variant, "job_id": job_id},
            )
            conn.commit()

    def close(self) -> None:
        pass
