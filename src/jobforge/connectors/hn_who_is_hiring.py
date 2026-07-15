"""
JobForge AI — Hacker News "Who is hiring?" Connector.

Hacker News runs a monthly "Ask HN: Who is hiring?" thread, posted by the
`whoishiring` account on the 1st of each month
(https://news.ycombinator.com/user?id=whoishiring). Each top-level comment is
one job ad, posted directly by the hiring company or a recruiter acting for
them — semi-structured free text, not a queryable API. This connector:

  1. Finds the current thread via the HN Algolia Search API
     (https://hn.algolia.com/api/v1/search_by_date?tags=story&query=Who+is+hiring),
     filtering to stories authored by `whoishiring` whose title matches
     "who is hiring" — sorted by date, so the first match is the latest one.
  2. Fetches the thread item via the official HN Firebase API
     (https://hacker-news.firebaseio.com/v0/item/{id}.json — no key required)
     to get its top-level comment ids (`kids`).
  3. Fetches each top-level comment and parses its free-text HTML body into
     a best-effort RawJob.

Neither the Algolia search endpoint nor the Firebase item API requires an
API key.

Parsing is inherently lossy — postings range from a single pipe-delimited
line ("Company | Role | Location | Remote/Onsite | salary") to multi-
paragraph descriptions with no consistent template. The parser is
deliberately conservative:
  - HTML is stripped from the comment body (HN comments come back as escaped
    HTML with <p> paragraph breaks and <a>/<i> inline formatting).
  - It tries a pipe-delimited first line first ("Company | Role | ...").
  - Falls back to a "Company - Role" / "Company (Role)" pattern.
  - Falls back further to treating the whole first line as both title and
    company if it can't cleanly split — better than dropping a real posting,
    but only when that line is short and doesn't look like a question/reply.
  - Comments that don't yield a plausible title AND company (too short,
    [deleted]/[dead], no discernible line 1) are skipped entirely rather than
    guessed at — this source is noisy enough without inventing structure.
  - Since the thread itself isn't queryable, one thread fetch serves every
    query in `queries` — results are keyword-filtered against a combined set
    built from all queries (same approach as CareerPagesConnector).
"""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, date, datetime

import httpx
import structlog
from bs4 import BeautifulSoup

from jobforge.connectors.base import JobSourceConnector
from jobforge.models.job import RawJob

logger = structlog.get_logger(__name__)

ALGOLIA_SEARCH_URL = "https://hn.algolia.com/api/v1/search_by_date"
FIREBASE_ITEM_URL = "https://hacker-news.firebaseio.com/v0/item/{item_id}.json"

HIRING_THREAD_AUTHOR = "whoishiring"
HIRING_THREAD_TITLE_RE = re.compile(r"who\s+is\s+hiring", re.IGNORECASE)

MIN_COMMENT_LENGTH = 40        # skip near-empty replies/questions
MAX_TOP_LEVEL_COMMENTS = 500   # thread can have 500+ top-level comments; cap per run
CONCURRENT_COMMENT_FETCHES = 15

_PIPE_SPLIT_RE = re.compile(r"\s*\|\s*")
_DASH_SPLIT_RE = re.compile(r"^(?P<company>[^\-(:]{2,80}?)\s*[-(]\s*(?P<title>.{2,150}?)\)?\s*$")


class HNWhoIsHiringConnector(JobSourceConnector):
    """Parses the current month's HN 'Who is hiring?' thread into RawJob postings."""

    source_name = "hn_who_is_hiring"
    # One thread lookup covers the whole run — this isn't a per-query paginated API.
    daily_quota = 5

    def __init__(self) -> None:
        self._request_count = 0

    async def search(
        self,
        queries: list[str],
        location: str = "UK",
        max_results_per_query: int = 25,
    ) -> list[RawJob]:
        if self._request_count >= self.daily_quota:
            logger.warning("hn_who_is_hiring.quota.reached", count=self._request_count)
            return []

        keywords = self._build_keywords(queries)
        max_total = max(max_results_per_query * max(len(queries), 1), max_results_per_query)

        async with httpx.AsyncClient(timeout=20.0) as client:
            try:
                thread_id = await self._find_latest_thread_id(client)
            except (httpx.HTTPError, ValueError) as e:
                logger.error("hn_who_is_hiring.thread_lookup.failed", error=str(e))
                return []

            if thread_id is None:
                logger.warning("hn_who_is_hiring.no_thread_found")
                return []

            try:
                comment_ids = await self._fetch_top_level_comment_ids(client, thread_id)
            except (httpx.HTTPError, ValueError) as e:
                logger.error(
                    "hn_who_is_hiring.thread_fetch.failed", thread_id=thread_id, error=str(e)
                )
                return []

            jobs = await self._fetch_and_parse_comments(client, comment_ids, keywords)

        self._request_count += 1
        logger.info("hn_who_is_hiring.complete", thread_id=thread_id, jobs_found=len(jobs))
        return jobs[:max_total]

    # ── Thread Discovery ───────────────────────────────────────────────────────

    async def _find_latest_thread_id(self, client: httpx.AsyncClient) -> int | None:
        params = {"tags": "story", "query": "Who is hiring"}
        response = await client.get(ALGOLIA_SEARCH_URL, params=params)
        response.raise_for_status()
        data = response.json()

        for hit in data.get("hits", []):
            author = hit.get("author", "")
            title = hit.get("title", "") or ""
            if author == HIRING_THREAD_AUTHOR and HIRING_THREAD_TITLE_RE.search(title):
                object_id = hit.get("objectID")
                if object_id is not None:
                    return int(object_id)
        return None

    async def _fetch_top_level_comment_ids(
        self, client: httpx.AsyncClient, thread_id: int
    ) -> list[int]:
        response = await client.get(FIREBASE_ITEM_URL.format(item_id=thread_id))
        response.raise_for_status()
        item = response.json() or {}
        kids = item.get("kids", [])
        return kids[:MAX_TOP_LEVEL_COMMENTS]

    # ── Comment Fetch & Parse ────────────────────────────────────────────────────

    async def _fetch_and_parse_comments(
        self,
        client: httpx.AsyncClient,
        comment_ids: list[int],
        keywords: set[str],
    ) -> list[RawJob]:
        sem = asyncio.Semaphore(CONCURRENT_COMMENT_FETCHES)

        async def fetch_one(comment_id: int) -> RawJob | None:
            async with sem:
                try:
                    resp = await client.get(FIREBASE_ITEM_URL.format(item_id=comment_id))
                    resp.raise_for_status()
                    item = resp.json()
                except (httpx.HTTPError, ValueError) as e:
                    logger.debug(
                        "hn_who_is_hiring.comment.fetch_failed",
                        comment_id=comment_id,
                        error=str(e),
                    )
                    return None
                return self._parse_comment(item, keywords)

        results = await asyncio.gather(
            *(fetch_one(cid) for cid in comment_ids), return_exceptions=True
        )
        return [r for r in results if isinstance(r, RawJob)]

    def _parse_comment(self, item: dict | None, keywords: set[str]) -> RawJob | None:
        if not item or item.get("deleted") or item.get("dead"):
            return None

        raw_html = item.get("text", "")
        if not raw_html:
            return None

        text = self._clean_comment_html(raw_html)
        if len(text) < MIN_COMMENT_LENGTH:
            return None

        title, company = self._extract_title_and_company(text)
        if not title or not company:
            return None

        if keywords and not any(kw in text.lower() for kw in keywords):
            return None

        comment_id = item.get("id")
        posted_date = self._parse_posted_date(item.get("time"))

        combined = text.lower()
        work_model = "unknown"
        if "remote" in combined:
            work_model = "remote"
        elif "hybrid" in combined:
            work_model = "hybrid"
        elif "on-site" in combined or "onsite" in combined or "in office" in combined:
            work_model = "onsite"

        return RawJob(
            job_id=f"hn_{comment_id}",
            title=title[:200],
            company=company[:200],
            location="Remote/Unspecified — see description (HN thread posting)",
            description=text[:8000],
            url=f"https://news.ycombinator.com/item?id={comment_id}",
            source=self.source_name,
            posted_date=posted_date,
            work_model=work_model,
        )

    # ── Text Parsing Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _clean_comment_html(raw_html: str) -> str:
        """HN comment bodies are escaped HTML with <p> breaks and <a>/<i> tags."""
        soup = BeautifulSoup(raw_html, "html.parser")
        return soup.get_text(separator="\n", strip=True)

    @staticmethod
    def _extract_title_and_company(text: str) -> tuple[str, str]:
        """
        Best-effort split of the comment's first line into (title, company).

        Tries, in order: pipe-delimited ("Company | Role | ..."), then
        "Company - Role" / "Company (Role)", then — only if the line is short
        and not obviously a question/reply — the whole line as both fields.
        """
        first_line = text.split("\n", 1)[0].strip()
        if not first_line:
            return "", ""

        if "|" in first_line:
            parts = [p for p in _PIPE_SPLIT_RE.split(first_line) if p]
            if len(parts) >= 2:
                return parts[1].strip(), parts[0].strip()

        m = _DASH_SPLIT_RE.match(first_line)
        if m:
            return m.group("title").strip(), m.group("company").strip()

        if 5 <= len(first_line) <= 200 and "?" not in first_line:
            return first_line, first_line

        return "", ""

    @staticmethod
    def _parse_posted_date(unix_ts: int | None) -> date | None:
        if not unix_ts:
            return None
        try:
            return datetime.fromtimestamp(unix_ts, tz=UTC).date()
        except (ValueError, OSError, OverflowError):
            return None

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
