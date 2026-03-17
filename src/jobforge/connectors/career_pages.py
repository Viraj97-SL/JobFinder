"""
JobForge AI — Company Career Pages Connector (Deep Crawler).

Crawls a curated list of UK AI startup career pages. Goes deep:
  1. Fetches the career index page
  2. Detects embedded ATS iframes (Greenhouse, Lever, Ashby, Workable)
  3. Parses JSON-LD JobPosting structured data
  4. Follows individual job links to extract full JDs
  5. Falls back to link-text extraction if no structured data

The watchlist is in data/startup_watchlist.yaml.
FundingNewsConnector can inject additional discovered companies at runtime.
"""

from __future__ import annotations

import asyncio
import json
import re
from urllib.parse import urljoin, urlparse

import httpx
import structlog
from bs4 import BeautifulSoup

from jobforge.connectors.base import JobSourceConnector
from jobforge.models.job import RawJob

logger = structlog.get_logger(__name__)

# Minimum text length for a job description to be useful
MIN_DESCRIPTION_LENGTH = 80

# AI/ML relevance keywords for filtering
AI_KEYWORDS = {
    "machine learning", "ml engineer", "ai engineer", "data scientist",
    "llm", "nlp", "computer vision", "deep learning", "pytorch", "tensorflow",
    "langchain", "langgraph", "neural network", "reinforcement learning",
    "mlops", "data engineer", "research scientist", "applied scientist",
    "generative ai", "foundation model", "multimodal", "robotics",
    "python", "ml", " ai ", "artificial intelligence",
}

# ATS iframe / embed patterns — if detected, defer to GreenhouseLeverConnector
ATS_EMBED_PATTERNS = {
    "greenhouse": re.compile(r"boards\.greenhouse\.io/embed/job_board\?for=([a-zA-Z0-9_-]+)"),
    "lever": re.compile(r"jobs\.lever\.co/([a-zA-Z0-9_-]+)"),
    "ashby": re.compile(r"jobs\.ashbyhq\.com/([a-zA-Z0-9_-]+)"),
    "workable": re.compile(r"([a-zA-Z0-9_-]+)\.workable\.com"),
}


class CareerPagesConnector(JobSourceConnector):
    """
    Deep crawls curated company career pages for job listings.

    Parsing priority:
      1. JSON-LD structured data (most accurate, richest description)
      2. Embedded ATS detected → extract token for GreenhouseLeverConnector
      3. Follow individual job page links → scrape full JD
      4. Keyword-matched link text (fallback)
    """

    source_name = "career_pages"
    daily_quota = 120

    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9",
    }

    def __init__(self, watchlist: list[dict] | None = None) -> None:
        """
        Args:
            watchlist: List of dicts with keys: company, careers_url, stage
                       Loaded from startup_watchlist.yaml by the Scout Agent.
                       FundingNewsConnector can inject additional entries.
        """
        self.watchlist = watchlist or []

        # Side-channel: discovered ATS tokens for GreenhouseLeverConnector
        self.detected_ats: dict[str, list[str]] = {
            "greenhouse": [], "lever": [], "ashby": [], "workable": []
        }

    async def search(
        self,
        queries: list[str],
        location: str = "UK",
        max_results_per_query: int = 25,
    ) -> list[RawJob]:
        if not self.watchlist:
            logger.info("career_pages.empty_watchlist")
            return []

        # Build keyword set from queries
        keywords = self._build_keywords(queries)

        sem = asyncio.Semaphore(10)  # Max concurrent HTTP connections

        async with httpx.AsyncClient(
            timeout=20.0,
            follow_redirects=True,
            headers=self._HEADERS,
        ) as client:

            async def bounded_crawl(entry: dict) -> list[RawJob]:
                async with sem:
                    return await self._crawl_company(client, entry, keywords)

            tasks = [bounded_crawl(entry) for entry in self.watchlist]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        all_jobs: list[RawJob] = []
        for r in results:
            if isinstance(r, list):
                all_jobs.extend(r)

        logger.info(
            "career_pages.complete",
            companies=len(self.watchlist),
            total=len(all_jobs),
        )
        return all_jobs

    # ── Per-Company Crawl ──────────────────────────────────────────────────────

    async def _crawl_company(
        self,
        client: httpx.AsyncClient,
        entry: dict,
        keywords: set[str],
    ) -> list[RawJob]:
        company = entry["company"]
        url = entry["careers_url"]
        stage = entry.get("stage", "unknown")

        try:
            resp = await client.get(url)
            if resp.status_code != 200:
                logger.debug("career_pages.fetch.skip", company=company, status=resp.status_code)
                return []

            html = resp.text
            soup = BeautifulSoup(html, "html.parser")

            # Priority 1: JSON-LD structured data
            jobs = self._extract_jsonld(soup, url, company, stage)
            if jobs:
                logger.debug("career_pages.jsonld", company=company, found=len(jobs))
                return [j for j in jobs if self._is_relevant(j, keywords)]

            # Priority 2: Detect embedded ATS — record token for GreenhouseLeverConnector
            ats_detected = self._detect_ats_embed(html, url)
            if ats_detected:
                for platform, token in ats_detected.items():
                    if token not in self.detected_ats.get(platform, []):
                        self.detected_ats.setdefault(platform, []).append(token)
                logger.debug("career_pages.ats_detected", company=company, ats=ats_detected)
                # Don't return early — some pages have both embed + link listing

            # Priority 3: Deep crawl — find job links, follow them, extract full JDs
            job_links = self._find_job_links(soup, url, keywords)
            if job_links:
                jobs = await self._deep_crawl_jobs(client, job_links, company, stage)
                if jobs:
                    return jobs

            # Priority 4: Fallback — return matched links with minimal description
            return self._fallback_links(soup, url, company, stage, keywords)

        except Exception as e:
            logger.error("career_pages.error", company=company, error=str(e))
            return []

    # ── JSON-LD Extraction ─────────────────────────────────────────────────────

    def _extract_jsonld(
        self, soup: BeautifulSoup, page_url: str, company: str, stage: str
    ) -> list[RawJob]:
        jobs: list[RawJob] = []
        for script in soup.find_all("script", type="application/ld+json"):
            if not script.string:
                continue
            try:
                data = json.loads(script.string)
                items = data if isinstance(data, list) else [data]
                for item in items:
                    if item.get("@type") in ("JobPosting", "jobPosting"):
                        title = item.get("title", "")
                        if not title:
                            continue
                        loc = self._extract_jsonld_location(item)
                        desc = (item.get("description", "") or "")
                        jobs.append(RawJob(
                            job_id=f"cp_{hash(page_url + title) & 0xFFFFFFFF}",
                            title=title,
                            company=company,
                            location=loc,
                            description=self._clean_html(desc)[:8000],
                            url=item.get("url", page_url),
                            source="career_pages",
                            is_startup=True,
                            company_stage=stage,
                        ))
            except (json.JSONDecodeError, TypeError):
                continue
        return jobs

    @staticmethod
    def _extract_jsonld_location(item: dict) -> str:
        loc_data = item.get("jobLocation", {})
        if isinstance(loc_data, dict):
            addr = loc_data.get("address", {})
            if isinstance(addr, dict):
                return (
                    addr.get("addressLocality", "")
                    or addr.get("addressRegion", "")
                    or "UK"
                )
        remote = item.get("jobLocationType", "")
        if remote == "TELECOMMUTE":
            return "Remote (UK)"
        return "UK"

    # ── ATS Embed Detection ────────────────────────────────────────────────────

    @staticmethod
    def _detect_ats_embed(html: str, url: str) -> dict[str, str]:
        detected: dict[str, str] = {}
        for platform, pattern in ATS_EMBED_PATTERNS.items():
            m = pattern.search(html)
            if m:
                detected[platform] = m.group(1)
        return detected

    # ── Deep Job Page Crawl ────────────────────────────────────────────────────

    def _find_job_links(
        self, soup: BeautifulSoup, base_url: str, keywords: set[str]
    ) -> list[tuple[str, str]]:
        """Find links that look like individual job postings."""
        base_domain = urlparse(base_url).netloc
        links: list[tuple[str, str]] = []
        seen: set[str] = set()

        for a in soup.find_all("a", href=True):
            text = a.get_text(strip=True)
            href = a["href"]

            if not text or len(text) < 5 or len(text) > 120:
                continue

            # Resolve URL
            if href.startswith("/"):
                href = urljoin(base_url, href)
            elif href.startswith("http"):
                # Only follow same-domain or known ATS domains
                link_domain = urlparse(href).netloc
                if link_domain != base_domain and not any(
                    ats in link_domain
                    for ats in ["greenhouse.io", "lever.co", "ashbyhq.com", "workable.com"]
                ):
                    continue
            else:
                continue

            if href in seen:
                continue

            if any(kw in text.lower() for kw in keywords):
                seen.add(href)
                links.append((text.strip(), href))

        return links[:30]  # Cap to avoid hitting too many pages

    async def _deep_crawl_jobs(
        self,
        client: httpx.AsyncClient,
        job_links: list[tuple[str, str]],
        company: str,
        stage: str,
    ) -> list[RawJob]:
        """Follow job page links and extract full JDs."""
        jobs: list[RawJob] = []
        sem = asyncio.Semaphore(5)

        async def fetch_one(title: str, url: str) -> RawJob | None:
            async with sem:
                return await self._extract_job_page(client, title, url, company, stage)

        tasks = [fetch_one(t, u) for t, u in job_links]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, RawJob):
                jobs.append(r)

        return jobs

    async def _extract_job_page(
        self,
        client: httpx.AsyncClient,
        title: str,
        url: str,
        company: str,
        stage: str,
    ) -> RawJob | None:
        """Fetch a single job page and extract the full description."""
        try:
            resp = await client.get(url)
            if resp.status_code != 200:
                return None

            soup = BeautifulSoup(resp.text, "html.parser")

            # Try JSON-LD first
            jsonld_jobs = self._extract_jsonld(soup, url, company, stage)
            if jsonld_jobs:
                return jsonld_jobs[0]

            # Find the main job description container
            description = self._extract_main_content(soup, title, company)
            if len(description) < MIN_DESCRIPTION_LENGTH:
                return None

            # Extract location from page
            location = self._extract_location_from_page(soup)

            return RawJob(
                job_id=f"cp_{hash(url) & 0xFFFFFFFF}",
                title=title,
                company=company,
                location=location,
                description=description[:8000],
                url=url,
                source="career_pages",
                is_startup=True,
                company_stage=stage,
            )
        except Exception as e:
            logger.debug("career_pages.job_page.error", url=url[:60], error=str(e))
            return None

    # ── Content Extraction Helpers ─────────────────────────────────────────────

    @staticmethod
    def _extract_main_content(soup: BeautifulSoup, title: str, company: str) -> str:
        """Extract the main job description text from a job page."""
        # Remove navigation, headers, footers, sidebars
        for tag in soup.find_all(["nav", "header", "footer", "aside", "script", "style"]):
            tag.decompose()

        # Look for common job description container patterns
        candidates = [
            soup.find("div", {"class": re.compile(r"job.?desc|jd|content|posting", re.I)}),
            soup.find("section", {"class": re.compile(r"job|description|content", re.I)}),
            soup.find("article"),
            soup.find("main"),
        ]

        for container in candidates:
            if container:
                text = container.get_text(separator="\n", strip=True)
                if len(text) > MIN_DESCRIPTION_LENGTH:
                    return text[:8000]

        # Final fallback: full page text minus clutter
        body = soup.find("body")
        if body:
            return body.get_text(separator="\n", strip=True)[:8000]

        return f"{title} at {company}. Visit the job page for full details."

    @staticmethod
    def _extract_location_from_page(soup: BeautifulSoup) -> str:
        """Try to extract location from a job page."""
        text = soup.get_text(" ", strip=True)
        m = re.search(
            r"\b(London|Manchester|Edinburgh|Cambridge|Oxford|Bristol|"
            r"Remote|Hybrid|UK Remote|United Kingdom)\b",
            text,
            re.IGNORECASE,
        )
        return m.group(1) if m else "UK"

    # ── Fallback: Link-Only ────────────────────────────────────────────────────

    def _fallback_links(
        self,
        soup: BeautifulSoup,
        page_url: str,
        company: str,
        stage: str,
        keywords: set[str],
    ) -> list[RawJob]:
        """Return keyword-matched links with minimal descriptions (last resort)."""
        jobs: list[RawJob] = []
        seen: set[str] = set()

        for a in soup.find_all("a", href=True):
            text = a.get_text(strip=True)
            href = a["href"]

            if not text or len(text) < 5 or len(text) > 120:
                continue
            if href.startswith("/"):
                href = urljoin(page_url, href)
            if href in seen:
                continue
            if any(kw in text.lower() for kw in keywords):
                seen.add(href)
                jobs.append(RawJob(
                    job_id=f"cp_{hash(href) & 0xFFFFFFFF}",
                    title=text.strip(),
                    company=company,
                    location="UK",
                    description=(
                        f"{text} at {company}. "
                        f"Stage: {stage}. See job page for full description."
                    ),
                    url=href,
                    source="career_pages",
                    is_startup=True,
                    company_stage=stage,
                ))

        return jobs[:20]

    # ── Utilities ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_keywords(queries: list[str]) -> set[str]:
        """Extract individual words from queries to use as keyword filters."""
        base = {
            "engineer", "scientist", "analyst", "ml", "ai", "data",
            "machine learning", "python", "deep learning", "llm", "nlp",
            "computer vision", "mlops", "research",
        }
        for q in queries:
            base.update(q.lower().split())
        return base

    @staticmethod
    def _is_relevant(job: RawJob, keywords: set[str]) -> bool:
        text = f"{job.title} {job.description}".lower()
        return any(kw in text for kw in keywords)

    @staticmethod
    def _clean_html(text: str) -> str:
        """Strip HTML tags from a description string."""
        if "<" in text:
            soup = BeautifulSoup(text, "html.parser")
            return soup.get_text(separator="\n", strip=True)
        return text
