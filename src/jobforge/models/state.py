"""
JobForge AI — LangGraph State Schema.

This TypedDict is the CONTRACT between all agent nodes.
Every field is typed and documented. The state flows through:
  Scout → Matchmaker → Tailor → Dispatcher
"""

from __future__ import annotations

from datetime import datetime
from typing import TypedDict

from jobforge.models.cv import SkillInventory, TailoredCV, TailorError
from jobforge.models.job import RawJob, ScoredJob
from jobforge.models.scoring import MatchSummary


class ScoutMetadata(TypedDict):
    """Telemetry from the Scout Agent."""
    sources_queried: list[str]
    source_counts: dict[str, int]        # source_name -> jobs found
    source_errors: dict[str, str]        # source_name -> error message (if failed)
    total_raw: int
    total_after_dedup: int
    queries_used: list[str]
    duration_seconds: float


class RunLog(TypedDict):
    """Full pipeline telemetry for a single run."""
    run_id: str
    started_at: str
    completed_at: str | None
    status: str                          # "success" | "partial" | "failed"
    total_duration_seconds: float
    scout_metadata: ScoutMetadata
    match_summary: MatchSummary
    cvs_generated: int
    cvs_failed: int
    email_sent: bool
    llm_tokens_used: int
    llm_cost_usd: float
    errors: list[str]


class JobForgeState(TypedDict, total=False):
    """
    The master state object that flows through the LangGraph DAG.

    All fields use `total=False` so agents only write the fields they own.
    Each agent reads upstream fields and writes its own output fields.
    """

    # ── Initialisation (set before graph starts) ──
    skill_inventory: SkillInventory
    run_id: str
    run_started_at: str

    # ── Scout Agent outputs ──
    raw_jobs: list[RawJob]
    deduped_jobs: list[RawJob]
    scout_metadata: ScoutMetadata

    # ── Matchmaker Agent outputs ──
    scored_jobs: list[ScoredJob]
    qualified_jobs: list[ScoredJob]          # >= threshold, ranked desc
    match_summary: MatchSummary

    # ── Tailor Agent outputs ──
    tailored_cvs: list[TailoredCV]
    tailor_errors: list[TailorError]

    # ── Dispatcher Agent outputs ──
    excel_path: str
    email_sent: bool
    run_log: RunLog
