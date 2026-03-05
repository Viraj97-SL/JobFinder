"""
JobForge AI — Base Agent & Deep Agent Architecture.

BaseAgent: Simple agent interface (used by Dispatcher).
DeepAgent: Extended agent with planning, memory, tool-use, and self-reflection
           (used by Scout, Matchmaker, Tailor).

Deep Agent Lifecycle:
  1. PLAN    → Analyse inputs, create execution plan
  2. EXECUTE → Run tools, call LLMs, process data
  3. REFLECT → Evaluate own output quality, log insights
  4. OUTPUT  → Return typed state updates
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import structlog

from jobforge.models.state import JobForgeState

logger = structlog.get_logger(__name__)


class BaseAgent(ABC):
    """Minimal agent interface. All agents implement this."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique agent identifier (e.g. 'scout', 'matchmaker')."""
        ...

    @abstractmethod
    async def run(self, state: JobForgeState) -> dict[str, Any]:
        """
        Execute the agent's task and return state updates.

        Args:
            state: Current pipeline state (read upstream fields).

        Returns:
            Dict of state field updates to merge into JobForgeState.
        """
        ...


class DeepAgent(BaseAgent):
    """
    Extended agent with Deep Agent capabilities:
    - Planning: Analyses inputs and creates a structured execution plan
    - Memory: Reads/writes to persistent stores across runs
    - Tool Use: Invokes external tools (APIs, file system, compilers)
    - Self-Reflection: Evaluates own output and logs quality signals

    Subclasses implement the four lifecycle hooks.
    """

    async def run(self, state: JobForgeState) -> dict[str, Any]:
        """Orchestrate the Deep Agent lifecycle."""
        logger.info(f"{self.name}.lifecycle.start")

        # 1. PLAN
        plan = await self.plan(state)
        logger.info(f"{self.name}.plan.complete", plan_summary=str(plan)[:200])

        # 2. EXECUTE
        result = await self.execute(state, plan)
        logger.info(f"{self.name}.execute.complete")

        # 3. REFLECT
        reflection = await self.reflect(state, result)
        logger.info(f"{self.name}.reflect.complete", quality=reflection.get("quality", "unknown"))

        # 4. OUTPUT
        output = await self.output(state, result, reflection)
        logger.info(f"{self.name}.lifecycle.complete")

        return output

    @abstractmethod
    async def plan(self, state: JobForgeState) -> dict[str, Any]:
        """
        Analyse inputs and create an execution plan.

        Returns a plan dict consumed by execute().
        Example: Scout returns {"queries": [...], "sources": [...]}
        """
        ...

    @abstractmethod
    async def execute(self, state: JobForgeState, plan: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the plan using tools, APIs, and LLMs.

        Returns raw results to be evaluated by reflect().
        """
        ...

    @abstractmethod
    async def reflect(self, state: JobForgeState, result: dict[str, Any]) -> dict[str, Any]:
        """
        Self-evaluate output quality. Log insights for cross-run improvement.

        Returns a reflection dict with at minimum {"quality": "good"|"warning"|"poor"}.
        """
        ...

    @abstractmethod
    async def output(
        self, state: JobForgeState, result: dict[str, Any], reflection: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Prepare the final state update dict.

        Combines raw results with reflection insights.
        """
        ...
