"""
JobForge AI — ATS Direct Connector (Greenhouse · Lever · Ashby).

Most UK AI startups use one of these three ATS platforms. They expose
public JSON endpoints — no auth, no scraping, clean structured data.

  Greenhouse : boards.greenhouse.io/{token}/jobs          → JSON
  Lever      : api.lever.co/v0/postings/{slug}?mode=json  → JSON
  Ashby      : jobs.ashbyhq.com/{slug}                    → HTML (JSON-LD inside)

Discovery strategy: Tavily searches site:boards.greenhouse.io and
site:jobs.lever.co to find which UK AI companies are on each platform,
then hits the JSON APIs directly for clean structured job data.
"""

from __future__ import annotations

import asyncio
import json
import re
from urllib.parse import urljoin

import httpx
import structlog
from bs4 import BeautifulSoup

from jobforge.config.settings import settings
from jobforge.connectors.base import JobSourceConnector
from jobforge.models.job import RawJob

logger = structlog.get_logger(__name__)

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None  # type: ignore

# ── Known UK AI/ML company tokens on each ATS ─────────────────────────────────
# These are manually verified; Tavily discovery supplements this list at runtime.

GREENHOUSE_TOKENS = [
    "deepmind", "wayve", "graphcore", "benevolentai", "polyai",
    "tractable", "speechmatics", "exscientia", "onfido", "darktrace",
    "thoughtmachine", "quantexa", "multiverse", "luminance", "synthesia",
    "oxbotica", "secondmind", "hadean", "kaedim", "brainomix",
    "automatatechnologies", "kheiron", "perspectum", "huma", "ixico",
    "featurespace", "eigen", "facultyai", "elevenlabs", "physicsx",
    "gradient-labs", "metaview", "granola", "oxa", "dexory",
    "peakai", "cytora", "conjecture",
]

LEVER_SLUGS = [
    "cleo", "monzo", "revolut", "faculty-ai", "stability-ai",
    "wayve", "poly-ai", "tractable", "luminance", "elevenlabs",
    "physicsx", "gradient-labs", "oxa", "dexory", "peak-ai",
    "benevolent-ai", "onfido", "quantexa", "speechmatics",
    "multiverse", "patchwork-health", "ieso-digital-health",
]

ASHBY_SLUGS = [
    "wayve", "faculty", "polyai", "tractable", "luminance",
    "elevenlabs", "physicsx", "gradientlabs", "conjecture",
    "apollo-research", "palisade-research", "kaedim", "granola",
    "metaview", "oxa", "dexory", "digitalgenius",
]

# AI/ML relevance keywords for filtering
AI_KEYWORDS = {
    "machine learning", "ml engineer", "ai engineer", "data scientist",
    "llm", "nlp", "computer vision", "deep learning", "pytorch", "tensorflow",
    "langchain", "langgraph", "neural", "reinforcement learning", "mlops",
    "data engineer", "research scientist", "applied scientist",
    "generative ai", "foundation model", "multimodal", "robotics",
}


class GreenhouseLeverConnector(JobSourceConnector):
    """
    Pulls structured job data directly from Greenhouse, Lever, and Ashby ATS.

    Combines hardcoded known tokens with runtime Tavily discovery to maximise
    coverage of UK AI startups. All three platforms return clean JSON or
    JSON-LD — descriptions are rich and complete.
    """

    source_name = "ats_direct"
    daily_quota = 80

    def __init__(self) -> None:
        self.tavily_key = settings.sources.tavily_api_key
        if TavilyClient and self.tavily_key:
            self._tavily = TavilyClient(api_key=self.tavily_key)
        else:
            self._tavily = None

    async def search(
        self,
        queries: list[str],
        location: str = "UK",
        max_results_per_query: int = 25,
    ) -> list[RawJob]:
        """Fan-out across all three ATS platforms in parallel."""
        greenhouse_tokens = list(GREENHOUSE_TOKENS)
        lever_slugs = list(LEVER_SLUGS)
        ashby_slugs = list(ASHBY_SLUGS)

        # Augment with Tavily-discovered tokens
        if self._tavily:
            discovered = await self._discover_ats_tokens()
            greenhouse_tokens = list(set(greenhouse_tokens + discovered.get("greenhouse", [])))
            lever_slugs = list(set(lever_slugs + discovered.get("lever", [])))
            ashby_slugs = list(set(ashby_slugs + discovered.get("ashby", [])))

        async with httpx.AsyncClient(
            timeout=20.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; JobSearchBot/1.0)"},
        ) as client:
            gh_task = self._fetch_all_greenhouse(client, greenhouse_tokens)
            lv_task = self._fetch_all_lever(client, lever_slugs)
            ash_task = self._fetch_all_ashby(client, ashby_slugs)

            gh_jobs, lv_jobs, ash_jobs = await asyncio.gather(
                gh_task, lv_task, ash_task, return_exceptions=True
            )

        all_jobs: list[RawJob] = []
        for result in [gh_jobs, lv_jobs, ash_jobs]:
            if isinstance(result, list):
                all_jobs.extend(result)

        # Filter to AI/ML relevant roles
        filtered = [j for j in all_jobs if self._is_relevant(j)]

        logger.info(
            "ats_direct.complete",
            total=len(all_jobs),
            relevant=len(filtered),
            greenhouse=len(gh_jobs) if isinstance(gh_jobs, list) else 0,
            lever=len(lv_jobs) if isinstance(lv_jobs, list) else 0,
            ashby=len(ash_jobs) if isinstance(ash_jobs, list) else 0,
        )
        return filtered

    # ── Greenhouse ─────────────────────────────────────────────────────────────

    async def _fetch_all_greenhouse(
        self, client: httpx.AsyncClient, tokens: list[str]
    ) -> list[RawJob]:
        tasks = [self._fetch_greenhouse(client, t) for t in tokens]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        jobs: list[RawJob] = []
        for r in results:
            if isinstance(r, list):
                jobs.extend(r)
        return jobs

    async def _fetch_greenhouse(
        self, client: httpx.AsyncClient, token: str
    ) -> list[RawJob]:
        """Fetch all jobs from a Greenhouse board via public JSON API."""
        url = f"https://boards.greenhouse.io/{token}/jobs"
        try:
            resp = await client.get(url, params={"content": "true"})
            if resp.status_code != 200:
                return []
            data = resp.json()
            jobs = data.get("jobs", [])
            return [self._parse_greenhouse_job(j, token) for j in jobs if j]
        except Exception as e:
            logger.debug("greenhouse.fetch.error", token=token, error=str(e))
            return []

    def _parse_greenhouse_job(self, job: dict, token: str) -> RawJob:
        company = token.replace("-", " ").title()
        location_data = job.get("location", {})
        loc = location_data.get("name", "UK") if isinstance(location_data, dict) else "UK"

        return RawJob(
            job_id=f"gh_{job.get('id', hash(job.get('absolute_url', '')) & 0xFFFFFFFF)}",
            title=job.get("title", ""),
            company=company,
            location=loc,
            description=(job.get("content", "") or "")[:8000],
            url=job.get("absolute_url", f"https://boards.greenhouse.io/{token}/jobs"),
            source="ats_direct",
            is_startup=True,
            company_stage="unknown",
        )

    # ── Lever ──────────────────────────────────────────────────────────────────

    async def _fetch_all_lever(
        self, client: httpx.AsyncClient, slugs: list[str]
    ) -> list[RawJob]:
        tasks = [self._fetch_lever(client, s) for s in slugs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        jobs: list[RawJob] = []
        for r in results:
            if isinstance(r, list):
                jobs.extend(r)
        return jobs

    async def _fetch_lever(
        self, client: httpx.AsyncClient, slug: str
    ) -> list[RawJob]:
        """Fetch all jobs via Lever's public API."""
        url = f"https://api.lever.co/v0/postings/{slug}"
        try:
            resp = await client.get(url, params={"mode": "json", "limit": 100})
            if resp.status_code != 200:
                return []
            postings = resp.json()
            if not isinstance(postings, list):
                return []
            return [self._parse_lever_job(p, slug) for p in postings if p]
        except Exception as e:
            logger.debug("lever.fetch.error", slug=slug, error=str(e))
            return []

    def _parse_lever_job(self, posting: dict, slug: str) -> RawJob:
        company = slug.replace("-", " ").title()
        lists = posting.get("lists", [])
        desc_parts = [posting.get("descriptionPlain", "")]
        for lst in lists:
            desc_parts.append(lst.get("content", ""))
        description = "\n".join(filter(None, desc_parts))[:8000]

        location = posting.get("workplaceType", "")
        categories = posting.get("categories", {})
        if isinstance(categories, dict):
            location = categories.get("location", location) or location

        return RawJob(
            job_id=f"lv_{posting.get('id', hash(posting.get('hostedUrl', '')) & 0xFFFFFFFF)}",
            title=posting.get("text", ""),
            company=company,
            location=location or "UK",
            description=description,
            url=posting.get("hostedUrl", f"https://jobs.lever.co/{slug}"),
            source="ats_direct",
            is_startup=True,
            company_stage="unknown",
        )

    # ── Ashby ──────────────────────────────────────────────────────────────────

    async def _fetch_all_ashby(
        self, client: httpx.AsyncClient, slugs: list[str]
    ) -> list[RawJob]:
        tasks = [self._fetch_ashby(client, s) for s in slugs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        jobs: list[RawJob] = []
        for r in results:
            if isinstance(r, list):
                jobs.extend(r)
        return jobs

    async def _fetch_ashby(
        self, client: httpx.AsyncClient, slug: str
    ) -> list[RawJob]:
        """Fetch jobs from Ashby-hosted career page (JSON-LD)."""
        url = f"https://jobs.ashbyhq.com/{slug}"
        try:
            resp = await client.get(url)
            if resp.status_code != 200:
                return []
            soup = BeautifulSoup(resp.text, "html.parser")
            return self._extract_ashby_jobs(soup, url, slug)
        except Exception as e:
            logger.debug("ashby.fetch.error", slug=slug, error=str(e))
            return []

    def _extract_ashby_jobs(
        self, soup: BeautifulSoup, base_url: str, slug: str
    ) -> list[RawJob]:
        company = slug.replace("-", " ").title()
        jobs: list[RawJob] = []

        # Ashby embeds all jobs as JSON in a __NEXT_DATA__ script tag
        next_data = soup.find("script", id="__NEXT_DATA__")
        if next_data and next_data.string:
            try:
                data = json.loads(next_data.string)
                postings = (
                    data.get("props", {})
                    .get("pageProps", {})
                    .get("jobPostings", [])
                ) or (
                    data.get("props", {})
                    .get("pageProps", {})
                    .get("organizationWithJobs", {})
                    .get("jobPostings", [])
                )
                for p in postings:
                    title = p.get("title", "")
                    job_url = urljoin(base_url, p.get("jobUrl", ""))
                    loc = p.get("locationName", "UK")
                    jobs.append(RawJob(
                        job_id=f"ash_{hash(job_url) & 0xFFFFFFFF}",
                        title=title,
                        company=company,
                        location=loc,
                        description=p.get("descriptionPlain", f"{title} at {company}.")[:8000],
                        url=job_url,
                        source="ats_direct",
                        is_startup=True,
                        company_stage="unknown",
                    ))
                return jobs
            except (json.JSONDecodeError, KeyError):
                pass

        # Fallback: link-based extraction
        for a in soup.find_all("a", href=True):
            text = a.get_text(strip=True)
            href = urljoin(base_url, a["href"])
            if text and len(text) > 5 and any(kw in text.lower() for kw in AI_KEYWORDS):
                jobs.append(RawJob(
                    job_id=f"ash_{hash(href) & 0xFFFFFFFF}",
                    title=text,
                    company=company,
                    location="UK",
                    description=f"{text} at {company}.",
                    url=href,
                    source="ats_direct",
                    is_startup=True,
                    company_stage="unknown",
                ))
        return jobs

    # ── Tavily Discovery ────────────────────────────────────────────────────────

    async def _discover_ats_tokens(self) -> dict[str, list[str]]:
        """Use Tavily to find UK AI companies on each ATS platform."""
        discovery_queries = [
            ("greenhouse", "site:boards.greenhouse.io AI machine learning engineer London UK"),
            ("greenhouse", "site:boards.greenhouse.io data scientist ML engineer United Kingdom"),
            ("lever", "site:jobs.lever.co AI engineer machine learning London UK"),
            ("lever", "site:jobs.lever.co data scientist NLP London United Kingdom"),
            ("ashby", "site:jobs.ashbyhq.com AI ML engineer London UK"),
        ]

        discovered: dict[str, list[str]] = {"greenhouse": [], "lever": [], "ashby": []}

        for platform, query in discovery_queries:
            try:
                results = self._tavily.search(query=query, max_results=10)
                for r in results.get("results", []):
                    token = self._extract_token(r.get("url", ""), platform)
                    if token and token not in discovered[platform]:
                        discovered[platform].append(token)
            except Exception as e:
                logger.debug("ats_discovery.error", platform=platform, error=str(e))

        logger.info(
            "ats_discovery.complete",
            greenhouse=len(discovered["greenhouse"]),
            lever=len(discovered["lever"]),
            ashby=len(discovered["ashby"]),
        )
        return discovered

    @staticmethod
    def _extract_token(url: str, platform: str) -> str | None:
        patterns = {
            "greenhouse": r"boards\.greenhouse\.io/([a-zA-Z0-9_-]+)",
            "lever": r"(?:jobs\.lever\.co|api\.lever\.co/v0/postings)/([a-zA-Z0-9_-]+)",
            "ashby": r"jobs\.ashbyhq\.com/([a-zA-Z0-9_-]+)",
        }
        m = re.search(patterns[platform], url)
        return m.group(1).rstrip("/") if m else None

    # ── Relevance Filter ───────────────────────────────────────────────────────

    @staticmethod
    def _is_relevant(job: RawJob) -> bool:
        text = f"{job.title} {job.description}".lower()
        return any(kw in text for kw in AI_KEYWORDS)
