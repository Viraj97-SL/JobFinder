"""
JobForge AI — Centralised Configuration via Pydantic Settings.

All settings are loaded from environment variables or a .env file.
Swap any provider by changing a single env var — no code changes required.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent  # jobforge-ai/
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
_ENV_FILE = str(ROOT_DIR / ".env")


class LLMSettings(BaseSettings):
    """LLM provider configuration — swap models by changing env vars."""

    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=_ENV_FILE, extra="ignore")

    gemini_api_key: str = Field(validation_alias="GEMINI_API_KEY")
    fast_model: str = "gemini-3.5-flash"
    deep_model: str = "gemini-3.1-pro-preview"
    cost_cap_usd: float = 2.00
    temperature: float = 0.1


class JobSourceSettings(BaseSettings):
    """API keys and limits for each job source connector."""

    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")

    adzuna_app_id: str = ""
    adzuna_app_key: str = ""
    reed_api_key: str = ""
    tavily_api_key: str = ""
    serpapi_key: str = ""

    # Per-source daily quotas (respect rate limits)
    adzuna_daily_quota: int = 200
    reed_daily_quota: int = 400
    tavily_daily_quota: int = 100
    # DWP "Find a Job" has no public API and its live successor
    # (jobs.service.gov.uk) sits behind Akamai bot-protection — see
    # connectors/uk_gov_find_a_job.py module docstring. Kept conservative
    # since it is best-effort HTML scraping, not a stable partner API.
    uk_gov_find_a_job_daily_quota: int = 60


class VisaSettings(BaseSettings):
    """
    Graduate Route (PSW) visa context.

    Key logic:
    - User has FULL work rights for 2 years (no sponsorship needed to work).
    - Sponsoring roles are strategically prioritised (long-term UK stay).
    - "UK citizens only" roles are flagged but NOT excluded — user can still
      apply during PSW window for experience, or the requirement may be flexible.
    - Roles offering Skilled Worker Visa sponsorship get a scoring BOOST.
    """

    model_config = SettingsConfigDict(env_prefix="VISA_", env_file=_ENV_FILE, extra="ignore")

    status: str = "psw_graduate_route"
    expiry_date: date = Field(default_factory=lambda: date(2027, 9, 1))
    prioritise_sponsoring: bool = True

    # Scoring adjustments (applied in Matchmaker)
    sponsorship_boost: int = 10      # +10 points if role offers sponsorship
    citizens_only_penalty: int = 5   # -5 points (not excluded, just deprioritised)

    @property
    def days_remaining(self) -> int:
        return max(0, (self.expiry_date - date.today()).days)

    @property
    def is_expired(self) -> bool:
        return self.days_remaining == 0


class PipelineSettings(BaseSettings):
    """Pipeline-level thresholds and behaviour."""

    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")

    match_threshold: int = 70
    embedding_prescreen_threshold: float = 0.45
    max_tailor_retries: int = 2
    daily_run_hour: int = 7
    max_jobs_per_source: int = 50
    # Cost controls
    max_cvs_per_run: int = 20          # Only tailor top-N by score; all jobs still in Excel
    matchmaker_concurrency: int = 15   # Parallel Gemini Flash calls
    tailor_concurrency: int = 5        # Parallel Gemini Pro calls
    # ML pre-screen gate (runs before LLM, cuts ~50% of calls)
    ml_prescreen_enabled: bool = True
    ml_prescreen_threshold: float = 0.30  # Weighted ensemble cutoff [0, 1]
    # UK sponsor register cross-check (2.4) — verifies employers actually hold
    # a Home Office sponsor licence, distinct from JD-stated sponsorship.
    sponsor_register_enabled: bool = True
    sponsor_register_cache_days: int = 7  # Register updates frequently; refresh weekly
    # RAG few-shot context for Tailor Agent
    rag_enabled: bool = True
    rag_top_k: int = 3  # Number of similar past tailoring examples to retrieve


class EmailSettings(BaseSettings):
    """Email dispatch configuration — supports SMTP or Resend."""

    model_config = SettingsConfigDict(env_prefix="SMTP_", env_file=_ENV_FILE, extra="ignore")

    host: str = "smtp.gmail.com"
    port: int = 587
    user: str = ""
    password: str = ""
    recipient_email: str = Field(default="", validation_alias="RECIPIENT_EMAIL")

    # Alternative: Resend
    resend_api_key: str = Field(default="", validation_alias="RESEND_API_KEY")
    email_from: str = Field(default="", validation_alias="EMAIL_FROM")

    @property
    def backend(self) -> Literal["smtp", "resend"]:
        return "resend" if self.resend_api_key else "smtp"


class Settings(BaseSettings):
    """Master settings — aggregates all sub-configs."""

    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")

    llm: LLMSettings = Field(default_factory=LLMSettings)
    sources: JobSourceSettings = Field(default_factory=JobSourceSettings)
    visa: VisaSettings = Field(default_factory=VisaSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    email: EmailSettings = Field(default_factory=EmailSettings)

    # ChromaDB persist directory for RAG tailoring store
    #chroma_db_dir: str = str(DATA_DIR / "chroma_db")
    chroma_db_dir: str = Field(
        default=str(DATA_DIR / "chroma_db"), validation_alias="CHROMA_DB_DIR",
    )


    # Railway auto-sets DATABASE_URL as postgresql://... — we normalise it for asyncpg.
    # Locally falls back to SQLite.
    database_url: str = Field(
        default=f"sqlite+aiosqlite:///{DATA_DIR / 'jobforge.db'}",
        validation_alias="DATABASE_URL",
    )
    log_level: str = "INFO"
    log_format: str = "json"

    @field_validator("database_url", mode="before")
    @classmethod
    def normalise_db_url(cls, v: str) -> str:
        """Convert Railway's postgresql:// to postgresql+asyncpg:// for SQLAlchemy async."""
        if v.startswith("postgresql://") or v.startswith("postgres://"):
            return v.replace("postgres://", "postgresql+asyncpg://", 1).replace(
                "postgresql://", "postgresql+asyncpg://", 1
            )
        return v


# Singleton — import this everywhere
settings = Settings()
