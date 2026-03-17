"""
JobForge AI — Funding News & Startup Discovery Connector.

Monitors UK AI/ML startup funding announcements to discover companies
*before* their roles appear on job boards. Newly funded startups almost
always start hiring within weeks of a round — this connector finds them first.

Flow:
  1. Search Tavily news for recent UK AI funding rounds (seed → Series B)
  2. Extract company domains from news articles
  3. Auto-discover career pages by probing common URL patterns
  4. Parse jobs via JSON-LD structured data or link extraction
  5. Expose `discovered_startups` list so ScoutAgent can enrich the watchlist

Sources targeted: TechCrunch, Sifted, City A.M., Tech Nation, Beauhurst,
                  Crunchbase, Business Insider, The Times Tech.
"""

from __future__ import annotations

import asyncio
import json
import re
from urllib.parse import urljoin, urlparse

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


# ── Funding news search queries ────────────────────────────────────────────────
FUNDING_QUERIES = [
    "UK AI startup funding round seed series London 2025 2026",
    "London machine learning startup raised investment hired 2025 2026",
    "UK deep tech AI series A series B funding announced",
    "British AI company seed round artificial intelligence new funding",
    "London AI startup raised million hiring engineers 2026",
    "Sifted UK AI startup funding round 2025 2026",
    "TechCrunch UK AI machine learning startup raises",
]

# Common career page URL paths to probe
CAREER_PATHS = [
    "/careers", "/jobs", "/work-with-us", "/join", "/join-us",
    "/about/careers", "/company/jobs", "/careers/open-roles",
    "/hiring", "/open-positions",
]

# Domains to skip (news sites, social, infra)
SKIP_DOMAINS = {
    "techcrunch.com", "sifted.eu", "twitter.com", "x.com", "linkedin.com",
    "crunchbase.com", "bloomberg.com", "reuters.com", "ft.com",
    "theguardian.com", "bbc.co.uk", "google.com", "youtube.com",
    "facebook.com", "instagram.com", "medium.com", "substack.com",
    "github.com", "notion.so", "docs.google.com", "angel.co",
    "wellfound.com", "glassdoor.com", "indeed.com", "adzuna.co.uk",
    "pitchbook.com", "beauhurst.com", "businessinsider.com",
}

# AI/ML job relevance keywords
AI_JOB_KEYWORDS = {
    "machine learning", "ml engineer", "ai engineer", "data scientist",
    "llm", "nlp", "computer vision", "deep learning", "pytorch", "tensorflow",
    "langchain", "langgraph", "neural network", "reinforcement learning",
    "mlops", "data engineer", "research scientist", "applied scientist",
    "generative ai", "foundation model", "multimodal", "robotics engineer",
}


class FundingNewsConnector(JobSourceConnector):
    """
    Discovers newly funded UK AI startups from news and crawls their career pages.

    Side-channel: after `.search()`, check `self.discovered_startups` for a list
    of newly found companies that can be added to the persistent watchlist.
    """

    source_name = "funding_news"
    daily_quota = 25  # Tavily quota-conscious

    def __init__(self) -> None:
        self.tavily_key = settings.sources.tavily_api_key
        if TavilyClient and self.tavily_key:
            self._tavily = TavilyClient(api_key=self.tavily_key)
        else:
            self._tavily = None

        # Populated after search() — ScoutAgent reads this to enrich watchlist
        self.discovered_startups: list[dict] = []

    async def search(
        self,
        queries: list[str],
        location: str = "UK",
        max_results_per_query: int = 25,
    ) -> list[RawJob]:
        if not self._tavily:
            logger.warning("funding_news.no_tavily", reason="TAVILY_API_KEY not set")
            return []

        # Step 1: Mine funding news for company domains
        candidate_companies = await self._mine_funding_news()
        if not candidate_companies:
            return []

        logger.info("funding_news.candidates_found", count=len(candidate_companies))

        # Step 2: Crawl each company's career page
        all_jobs = await self._crawl_career_pages(candidate_companies)

        logger.info(
            "funding_news.complete",
            companies_probed=len(candidate_companies),
            jobs_found=len(all_jobs),
            startups_discovered=len(self.discovered_startups),
        )
        return all_jobs

    # ── Step 1: Mine Funding News ──────────────────────────────────────────────

    async def _mine_funding_news(self) -> list[dict]:
        """Search Tavily news and extract company domains from articles."""
        seen_domains: dict[str, dict] = {}

        for query in FUNDING_QUERIES[:5]:  # Cap to save quota
            try:
                results = self._tavily.search(
                    query=query,
                    max_results=8,
                    search_depth="advanced",
                    topic="news",
                    include_raw_content=False,
                )
                for result in results.get("results", []):
                    for company in self._extract_companies(result):
                        if company["domain"] not in seen_domains:
                            seen_domains[company["domain"]] = company
            except Exception as e:
                logger.warning("funding_news.query.error", query=query[:40], error=str(e))

        return list(seen_domains.values())

    def _extract_companies(self, result: dict) -> list[dict]:
        """Extract likely company domains from a Tavily news result."""
        companies = []
        text = f"{result.get('title', '')} {result.get('content', '')}"

        # Extract all URLs from content
        for match in re.finditer(r'https?://(?:www\.)?([a-z0-9.\-]+)', text.lower()):
            raw_domain = match.group(1).strip(".")
            # Keep only .ai, .io, .co.uk, .com that look like product companies
            if not any(skip in raw_domain for skip in SKIP_DOMAINS):
                if re.search(r'\.(ai|io|co\.uk|com)$', raw_domain):
                    parts = raw_domain.split(".")
                    if len(parts[0]) > 2:  # Skip very short prefixes
                        name = parts[0].replace("-", " ").title()
                        companies.append({
                            "company": name,
                            "domain": raw_domain,
                            "website": f"https://{raw_domain}",
                            "stage": "recently_funded",
                        })

        return companies[:4]  # Cap per article

    # ── Step 2: Crawl Career Pages ─────────────────────────────────────────────

    async def _crawl_career_pages(self, companies: list[dict]) -> list[RawJob]:
        """Concurrently probe career pages for all candidate companies."""
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-GB,en;q=0.9",
        }

        sem = asyncio.Semaphore(8)  # Max 8 concurrent HTTP connections

        async def bounded_crawl(company: dict) -> list[RawJob]:
            async with sem:
                return await self._find_and_extract_jobs(company, headers)

        async with httpx.AsyncClient(
            timeout=15.0, follow_redirects=True, headers=headers
        ) as client:
            tasks = [self._find_and_extract_jobs_with_client(client, c) for c in companies[:30]]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        jobs: list[RawJob] = []
        for r in results:
            if isinstance(r, list):
                jobs.extend(r)
        return jobs

    async def _find_and_extract_jobs_with_client(
        self, client: httpx.AsyncClient, company: dict
    ) -> list[RawJob]:
        website = company["website"]
        company_name = company["company"]
        stage = company.get("stage", "recently_funded")

        for path in CAREER_PATHS:
            url = f"{website.rstrip('/')}{path}"
            try:
                resp = await client.get(url)
                if resp.status_code == 200 and len(resp.content) > 500:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    jobs = self._extract_jobs(soup, url, company_name, stage)
                    if jobs:
                        self.discovered_startups.append({
                            "company": company_name,
                            "careers_url": url,
                            "stage": stage,
                        })
                        return jobs
            except Exception:
                continue

        return []

    # Needed to satisfy linter — calls above use the _with_client variant
    async def _find_and_extract_jobs(self, company: dict, headers: dict) -> list[RawJob]:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, headers=headers) as c:
            return await self._find_and_extract_jobs_with_client(c, company)

    # ── Job Extraction ─────────────────────────────────────────────────────────

    def _extract_jobs(
        self, soup: BeautifulSoup, page_url: str, company: str, stage: str
    ) -> list[RawJob]:
        """Try JSON-LD structured data first, then fall back to link extraction."""
        jobs = self._extract_jsonld_jobs(soup, page_url, company, stage)
        if jobs:
            return jobs
        return self._extract_linked_jobs(soup, page_url, company, stage)

    def _extract_jsonld_jobs(
        self, soup: BeautifulSoup, page_url: str, company: str, stage: str
    ) -> list[RawJob]:
        """Parse JSON-LD JobPosting structured data."""
        jobs: list[RawJob] = []
        for script in soup.find_all("script", type="application/ld+json"):
            if not script.string:
                continue
            try:
                data = json.loads(script.string)
                # Handle both single object and array
                items = data if isinstance(data, list) else [data]
                for item in items:
                    if item.get("@type") in ("JobPosting", "jobPosting"):
                        title = item.get("title", "")
                        if not self._is_ai_role(title):
                            continue
                        loc_data = item.get("jobLocation", {})
                        if isinstance(loc_data, dict):
                            addr = loc_data.get("address", {})
                            loc = (
                                addr.get("addressLocality", "")
                                or addr.get("addressRegion", "UK")
                            )
                        else:
                            loc = "UK"
                        jobs.append(RawJob(
                            job_id=f"fn_{hash(page_url + title) & 0xFFFFFFFF}",
                            title=title,
                            company=company,
                            location=loc or "UK",
                            description=(item.get("description", "") or "")[:8000],
                            url=item.get("url", page_url),
                            source="funding_news",
                            is_startup=True,
                            company_stage=stage,
                        ))
            except (json.JSONDecodeError, TypeError):
                continue
        return jobs

    def _extract_linked_jobs(
        self, soup: BeautifulSoup, page_url: str, company: str, stage: str
    ) -> list[RawJob]:
        """Extract job links from career page HTML."""
        jobs: list[RawJob] = []
        seen_urls: set[str] = set()

        for a in soup.find_all("a", href=True):
            text = a.get_text(strip=True)
            href = a["href"]

            if not text or len(text) < 5 or len(text) > 120:
                continue

            # Resolve relative URLs
            if href.startswith("/"):
                href = urljoin(page_url, href)
            elif not href.startswith("http"):
                continue

            if href in seen_urls:
                continue

            if self._is_ai_role(text):
                seen_urls.add(href)
                jobs.append(RawJob(
                    job_id=f"fn_{hash(href) & 0xFFFFFFFF}",
                    title=text.strip(),
                    company=company,
                    location="UK",
                    description=(
                        f"{text} at {company} — discovered via funding news. "
                        f"Stage: {stage}. Visit the link for full job description."
                    ),
                    url=href,
                    source="funding_news",
                    is_startup=True,
                    company_stage=stage,
                ))

        return jobs[:15]  # Cap per company

    @staticmethod
    def _is_ai_role(title: str) -> bool:
        t = title.lower()
        return any(kw in t for kw in AI_JOB_KEYWORDS)
