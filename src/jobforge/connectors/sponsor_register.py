"""
JobForge AI — UK Sponsor Register Cross-Check (Phase 2.4).

Cross-references employer names against the Home Office's public Register of
Licensed Sponsors (Worker / Temporary Worker routes). This turns "the JD
mentions sponsorship" (an NLP guess against free text — see connectors/base.py
detect_sponsorship) into "this employer legally holds a sponsor licence right
now" (verified against the source-of-truth register). The two are reported as
distinct metrics, never conflated — an employer can hold a licence without
mentioning it in a given JD, and mentioning it isn't proof of a licence.

Source: https://www.gov.uk/government/publications/register-of-licensed-sponsors-workers
The direct CSV asset URL changes every time the Home Office republishes the
register (dated filename, e.g. ..._2026-07-10.csv), so this module resolves
the current link from the publication page rather than hardcoding a URL that
would go stale within days.
"""

from __future__ import annotations

import csv as csv_module
import re
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import structlog
from rapidfuzz import fuzz, process

from jobforge.config.settings import DATA_DIR

logger = structlog.get_logger(__name__)

REGISTER_PUBLICATION_URL = (
    "https://www.gov.uk/government/publications/register-of-licensed-sponsors-workers"
)
CACHE_PATH = DATA_DIR / "cache" / "sponsor_register.csv"
FUZZY_MATCH_THRESHOLD = 92  # rapidfuzz token_sort_ratio, 0-100

_CSV_ASSET_RE = re.compile(r'https://assets\.publishing\.service\.gov\.uk/media/[^"]+\.csv')
_LEGAL_SUFFIX_RE = re.compile(
    r"\b(limited|ltd|llp|plc|inc|incorporated|corp|corporation|"
    r"t\s*/\s*a|trading\s+as)\b\.?",
    re.IGNORECASE,
)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_company_name(name: str) -> str:
    """
    Strip legal suffixes and punctuation so 'Acme Ltd' and 'ACME LIMITED'
    (or 'Acme Ltd T/A Acme Foods') resolve to the same lookup key.
    """
    if not name:
        return ""
    lowered = name.lower()
    lowered = _LEGAL_SUFFIX_RE.sub(" ", lowered)
    lowered = _NON_ALNUM_RE.sub(" ", lowered)
    return _WHITESPACE_RE.sub(" ", lowered).strip()


async def _resolve_current_csv_url(client: httpx.AsyncClient) -> str:
    """
    Scrape the current CSV asset link from the publication page. The direct
    URL isn't stable across Home Office updates, so this can't be hardcoded.
    """
    resp = await client.get(REGISTER_PUBLICATION_URL)
    resp.raise_for_status()
    match = _CSV_ASSET_RE.search(resp.text)
    if not match:
        raise RuntimeError(
            "Could not locate the sponsor register CSV link on the GOV.UK publication page "
            "— the page structure may have changed."
        )
    return match.group(0)


async def download_sponsor_register(
    cache_path: Path = CACHE_PATH,
    max_age_days: int = 7,
    force: bool = False,
) -> Path:
    """
    Download the register CSV to a local cache, refreshing only if stale.
    The register is ~11MB and updates frequently, so a weekly cache keeps
    this cheap without going stale for weeks at a time.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not force and cache_path.exists():
        age = datetime.utcnow() - datetime.utcfromtimestamp(cache_path.stat().st_mtime)
        if age < timedelta(days=max_age_days):
            return cache_path

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        csv_url = await _resolve_current_csv_url(client)
        resp = await client.get(csv_url)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)

    logger.info("sponsor_register.downloaded", path=str(cache_path), bytes=len(resp.content))
    return cache_path


class SponsorRegisterMatcher:
    """
    In-memory lookup answering "does this employer hold a UK sponsor licence?"

    Two-tier matching, cheapest first:
      1. Exact match on the normalised name — covers the vast majority.
      2. Fuzzy match (rapidfuzz token_sort_ratio >= threshold) for near-miss
         spelling/formatting differences the normaliser doesn't catch.
    """

    def __init__(self, csv_path: Path) -> None:
        self._normalized_names: dict[str, str] = {}  # normalized -> original register name
        self._load(csv_path)

    def _load(self, csv_path: Path) -> None:
        with csv_path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv_module.DictReader(f)
            for row in reader:
                name = (row.get("Organisation Name") or "").strip()
                if not name:
                    continue
                normalized = normalize_company_name(name)
                if normalized:
                    self._normalized_names.setdefault(normalized, name)

        logger.info("sponsor_register.loaded", entries=len(self._normalized_names))

    def __len__(self) -> int:
        return len(self._normalized_names)

    def is_licensed_sponsor(self, company: str) -> bool:
        normalized = normalize_company_name(company)
        if not normalized:
            return False
        if normalized in self._normalized_names:
            return True

        match = process.extractOne(
            normalized,
            self._normalized_names.keys(),
            scorer=fuzz.token_sort_ratio,
            score_cutoff=FUZZY_MATCH_THRESHOLD,
        )
        return match is not None
