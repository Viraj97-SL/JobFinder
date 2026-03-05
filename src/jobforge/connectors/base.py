"""
JobForge AI — Job Source Connector ABC.

All job sources implement this interface. Swapping a source means
writing a new class that inherits from JobSourceConnector — no changes
to the Scout Agent or pipeline logic.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

import structlog

from jobforge.models.job import RawJob

logger = structlog.get_logger(__name__)


# ── Visa keyword detection patterns ──
SPONSORSHIP_POSITIVE = [
    r"visa\s+sponsorship\s+(available|offered|provided)?",
    r"willing\s+to\s+sponsor",
    r"skilled\s+worker\s+visa",
    r"sponsor(ship)?\s+for\s+the\s+right\s+candidate",
    r"we\s+can\s+sponsor",
    r"tier\s*2\s+sponsor",
    r"sponsorship\s+available"
]

SPONSORSHIP_NEGATIVE = [
    r"uk\s+citizen(s)?(\s+only)?",
    r"must\s+have\s+(the\s+)?right\s+to\s+work.*without\s+sponsorship",
    r"no\s+(visa\s+)?sponsorship",
    r"cannot\s+offer\s+sponsorship",
    r"not\s+able\s+to\s+sponsor",
    r"without\s+sponsorship",
    r"sc\s+clearance",           # Broadened from "sc clearance required"
    r"dv\s+clearance",           # Broadened
    r"security\s+clearance"      # Added for general security clearance
]

STARTUP_INDICATORS = [
    r"seed\s+(stage|funded|round)",
    r"series\s+[ab]\b",
    r"early[\s-]stage",
    r"pre[\s-]series",
    r"backed\s+by\s+(y\s*combinator|yc|a16z|sequoia)",
    r"founding\s+(engineer|team|member)",
    r"\bstartup\b",
    r"we.?re\s+building\s+from\s+(the\s+)?ground",
]


def detect_sponsorship(text: str) -> tuple[bool | None, bool | None, list[str]]:
    """
    Scan job description for visa sponsorship signals.

    Returns:
        (offers_sponsorship, citizens_only, signal_phrases)
    """
    text_lower = text.lower()
    signals: list[str] = []

    offers = None
    citizens_only = None

    # Check for Positive Signals
    for pattern in SPONSORSHIP_POSITIVE:
        match = re.search(pattern, text_lower)
        if match:
            # Crucial: Ensure it's not a negated phrase (e.g., "no visa sponsorship")
            if not re.search(rf"(?:no|cannot|without|not)\s+{pattern}", text_lower):
                offers = True
                signals.append(match.group())
                break  # Stop checking once we confirm they offer it

    # Check for Negative/Restrictive Signals
    for pattern in SPONSORSHIP_NEGATIVE:
        match = re.search(pattern, text_lower)
        if match:
            citizens_only = True
            signals.append(match.group())
            break  # Stop checking once we confirm restrictions apply

    return offers, citizens_only, signals


def detect_startup(text: str, company: str = "") -> bool:
    """Detect if a role is at a startup based on JD text and company context."""
    combined = f"{company} {text}".lower()
    return any(re.search(p, combined) for p in STARTUP_INDICATORS)


class JobSourceConnector(ABC):
    """
    Abstract base class for all job source connectors.

    To add a new source:
    1. Create a new file in connectors/ (e.g. glassdoor.py)
    2. Implement this ABC
    3. Register it in the Scout Agent's connector registry
    """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Unique identifier for this source (e.g. 'adzuna', 'reed')."""
        ...

    @property
    def daily_quota(self) -> int:
        """Max API calls per day. Override per connector."""
        return 100

    @abstractmethod
    async def search(
        self,
        queries: list[str],
        location: str = "UK",
        max_results_per_query: int = 25,
    ) -> list[RawJob]:
        """
        Execute search queries and return normalised RawJob list.

        Args:
            queries: Search terms (e.g. ["AI Engineer London", "ML Engineer UK remote"])
            location: Geographic filter
            max_results_per_query: Max results per individual query

        Returns:
            List of RawJob objects with all fields populated.
            Visa signals and startup detection should be applied.
        """
        ...

    def enrich_visa_signals(self, job: RawJob) -> RawJob:
        """Post-process: detect sponsorship and citizenship signals from JD text."""
        offers, citizens_only, signals = detect_sponsorship(job.description)
        job.offers_sponsorship = offers
        job.citizens_only = citizens_only
        job.sponsorship_signals = signals
        return job

    def enrich_startup_signals(self, job: RawJob) -> RawJob:
        """Post-process: detect if company is a startup."""
        job.is_startup = detect_startup(job.description, job.company)
        return job

    def enrich(self, job: RawJob) -> RawJob:
        """Apply all enrichment steps."""
        job = self.enrich_visa_signals(job)
        job = self.enrich_startup_signals(job)
        return job

    async def safe_search(
        self,
        queries: list[str],
        location: str = "UK",
        max_results_per_query: int = 25,
    ) -> list[RawJob]:
        """Search with error handling — never crashes the pipeline."""
        try:
            jobs = await self.search(queries, location, max_results_per_query)
            enriched = [self.enrich(job) for job in jobs]
            logger.info(
                "connector.search.complete",
                source=self.source_name,
                jobs_found=len(enriched),
                queries=len(queries),
            )
            return enriched
        except Exception as e:
            logger.error(
                "connector.search.failed",
                source=self.source_name,
                error=str(e),
                error_type=type(e).__name__,
            )
            return []