"""
JobForge AI — Rate Limiter & LLM Cost Tracker.

Rate limiter: Per-source token bucket with configurable quotas.
Cost tracker: Tracks Gemini token usage and USD spend per run.
"""

from __future__ import annotations

import time
from collections import defaultdict

import structlog

logger = structlog.get_logger(__name__)


class RateLimiter:
    """Simple per-source rate limiter with daily quotas."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = defaultdict(int)
        self._quotas: dict[str, int] = {}
        self._reset_time: float = time.time()

    def set_quota(self, source: str, daily_limit: int) -> None:
        self._quotas[source] = daily_limit

    def can_proceed(self, source: str) -> bool:
        """Check if source has remaining quota."""
        self._maybe_reset()
        quota = self._quotas.get(source, 100)
        return self._counts[source] < quota

    def record_call(self, source: str) -> None:
        self._counts[source] += 1

    def remaining(self, source: str) -> int:
        quota = self._quotas.get(source, 100)
        return max(0, quota - self._counts[source])

    def _maybe_reset(self) -> None:
        """Reset counters every 24 hours."""
        if time.time() - self._reset_time > 86400:
            self._counts.clear()
            self._reset_time = time.time()
            logger.info("rate_limiter.reset")


class CostTracker:
    """
    Track LLM token usage and estimated USD cost per run.

    Pricing (Gemini, as of July 2026 — update if pricing changes):
    - 3.5 Flash:        $1.50 / 1M input tokens, $9.00 / 1M output tokens
    - 3.1 Pro (preview): $2.00 / 1M input tokens, $12.00 / 1M output tokens (<=200k prompt)
    - 2.5 Pro (legacy):  $1.25 / 1M input tokens, $5.00 / 1M output tokens
    - 2.0 Flash (shut down by Google, kept only for historical score_cache entries)
    """

    PRICING = {
        "gemini-3.5-flash":        {"input": 1.50 / 1_000_000, "output": 9.00 / 1_000_000},
        "gemini-3.1-pro-preview":  {"input": 2.00 / 1_000_000, "output": 12.00 / 1_000_000},
        "gemini-2.5-pro":          {"input": 1.25 / 1_000_000, "output": 5.00 / 1_000_000},
        "gemini-2.0-flash":        {"input": 0.075 / 1_000_000, "output": 0.30 / 1_000_000},
    }

    def __init__(self, cost_cap_usd: float = 2.0) -> None:
        self.cost_cap = cost_cap_usd
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self._calls: list[dict] = []

    def record(self, model: str, input_tokens: int, output_tokens: int, agent: str = "") -> None:
        pricing = self.PRICING.get(model, self.PRICING["gemini-3.5-flash"])
        cost = input_tokens * pricing["input"] + output_tokens * pricing["output"]

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost

        self._calls.append({
            "model": model, "input": input_tokens, "output": output_tokens,
            "cost": round(cost, 6), "agent": agent,
        })

        if self.total_cost_usd >= self.cost_cap:
            logger.warning(
                "cost_tracker.cap_reached",
                total=round(self.total_cost_usd, 4),
                cap=self.cost_cap,
            )

    @property
    def is_over_budget(self) -> bool:
        return self.total_cost_usd >= self.cost_cap

    @property
    def summary(self) -> dict:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "total_calls": len(self._calls),
            "budget_remaining": round(self.cost_cap - self.total_cost_usd, 4),
        }
