"""
JobForge AI — LangGraph Orchestrator.

The main DAG that wires all four agents into a single pipeline.
State flows: Scout → Matchmaker → Tailor → Dispatcher

Conditional edges:
- If Scout finds 0 new jobs → skip to Dispatcher (send "no new jobs" email)
- If Matchmaker qualifies 0 jobs → skip Tailor, go to Dispatcher
- Tailor has an internal retry loop (max 2 retries per CV)
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import structlog
from langgraph.graph import END, StateGraph

from jobforge.agents.dispatcher.agent import DispatcherAgent
from jobforge.agents.matchmaker.agent import MatchmakerAgent
from jobforge.agents.scout.agent import ScoutAgent
from jobforge.agents.tailor.agent import TailorAgent
from jobforge.memory.dedup_store import RunHistory, init_database
from jobforge.models.state import JobForgeState

logger = structlog.get_logger(__name__)


def _should_skip_to_dispatcher(state: JobForgeState) -> str:
    """Conditional edge: skip Matchmaker if no new jobs found."""
    deduped = state.get("deduped_jobs", [])
    if not deduped:
        logger.info("orchestrator.skip.no_new_jobs")
        return "dispatcher"
    return "matchmaker"


def _should_skip_tailor(state: JobForgeState) -> str:
    """Conditional edge: skip Tailor if no jobs qualified."""
    qualified = state.get("qualified_jobs", [])
    if not qualified:
        logger.info("orchestrator.skip.no_qualified_jobs")
        return "dispatcher"
    return "tailor"


def build_pipeline(watchlist: list[dict] | None = None) -> StateGraph:
    """
    Construct the LangGraph StateGraph DAG.

    Returns a compiled graph ready for invocation.
    """

    # Instantiate agents
    scout = ScoutAgent(watchlist=watchlist)
    matchmaker = MatchmakerAgent()
    tailor = TailorAgent()
    dispatcher = DispatcherAgent()

    # Define the graph
    graph = StateGraph(JobForgeState)

    # Add nodes
    graph.add_node("scout", scout.run)
    graph.add_node("matchmaker", matchmaker.run)
    graph.add_node("tailor", tailor.run)
    graph.add_node("dispatcher", dispatcher.run)

    # Define edges
    graph.set_entry_point("scout")

    # Scout → conditional → Matchmaker or Dispatcher
    graph.add_conditional_edges(
        "scout",
        _should_skip_to_dispatcher,
        {"matchmaker": "matchmaker", "dispatcher": "dispatcher"},
    )

    # Matchmaker → conditional → Tailor or Dispatcher
    graph.add_conditional_edges(
        "matchmaker",
        _should_skip_tailor,
        {"tailor": "tailor", "dispatcher": "dispatcher"},
    )

    # Tailor → Dispatcher
    graph.add_edge("tailor", "dispatcher")

    # Dispatcher → END
    graph.add_edge("dispatcher", END)

    return graph.compile()


async def run_pipeline(
    skill_inventory: Any = None,
    watchlist: list[dict] | None = None,
) -> JobForgeState:
    """
    Execute the full JobForge pipeline.

    Args:
        skill_inventory: Pre-extracted SkillInventory (from skill_inventory.json)
        watchlist: UK AI startup career page URLs

    Returns:
        Final pipeline state with all results.
    """

    # Initialise database
    init_database()

    # Generate run ID
    run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    # Log run start
    run_history = RunHistory()
    try:
        run_history.start_run(run_id)
    finally:
        run_history.close()

    # Build initial state
    initial_state: JobForgeState = {
        "run_id": run_id,
        "run_started_at": datetime.utcnow().isoformat(),
        "skill_inventory": skill_inventory,
        "raw_jobs": [],
        "deduped_jobs": [],
        "scored_jobs": [],
        "qualified_jobs": [],
        "tailored_cvs": [],
        "tailor_errors": [],
        "email_sent": False,
    }

    logger.info("pipeline.start", run_id=run_id)

    # Build and run the graph
    graph = build_pipeline(watchlist=watchlist)
    final_state = await graph.ainvoke(initial_state)

    # Log run completion
    run_history = RunHistory()
    try:
        run_history.complete_run(
            run_id,
            total_scraped=len(final_state.get("raw_jobs", [])),
            total_qualified=len(final_state.get("qualified_jobs", [])),
            total_cvs_generated=len(final_state.get("tailored_cvs", [])),
            email_sent=1 if final_state.get("email_sent") else 0,
        )
    finally:
        run_history.close()

    logger.info(
        "pipeline.complete",
        run_id=run_id,
        scraped=len(final_state.get("raw_jobs", [])),
        qualified=len(final_state.get("qualified_jobs", [])),
        cvs=len(final_state.get("tailored_cvs", [])),
        email=final_state.get("email_sent"),
    )

    return final_state
