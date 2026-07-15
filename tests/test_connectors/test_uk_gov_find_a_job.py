"""
JobForge AI — DWP "Find a Job" Connector Tests.

All HTTP calls are mocked via monkeypatching `httpx.AsyncClient.get` — no
network calls are made. The sample HTML mirrors the real `div.search-result`
markup captured from a Wayback Machine snapshot of the (now decommissioned)
findajob.dwp.gov.uk search page — see the connector's module docstring for
the research trail and why this connector can legitimately return an empty
list against the live service (Akamai bot-wall).
"""

from __future__ import annotations

import asyncio
from datetime import date

import httpx

from jobforge.connectors.uk_gov_find_a_job import UkGovFindAJobConnector

SAMPLE_SEARCH_HTML = """
<html><body>
<div class="search-result" data-aid="111">
    <h3 class="govuk-heading-s govuk-!-margin-top-4 govuk-!-margin-bottom-2">
        <a class="govuk-link" href="https://www.jobs.service.gov.uk/details/111">
            Data Scientist
        </a>
    </h3>
    <ul class="govuk-list search-result-details govuk-!-margin-bottom-2">
        <li>07 April 2026</li>
        <li><strong>Acme Analytics Ltd</strong> - <span>Manchester</span></li>
        <li><strong>£45,000 to £60,000 a year</strong></li>
        <li class="govuk-tag govuk-tag--grey">Hybrid</li>
        <li class="govuk-tag govuk-tag--grey">Permanent</li>
    </ul>
    <p class="govuk-body search-result-description">
        We are looking for a data scientist skilled in python and machine
        learning to join our growing analytics team.
    </p>
    <button class="favourite" data-js-favourite="111"></button>
</div>
<div class="search-result" data-aid="222">
    <h3 class="govuk-heading-s govuk-!-margin-top-4 govuk-!-margin-bottom-2">
        <a class="govuk-link" href="https://www.jobs.service.gov.uk/details/222">
            ML Engineer
        </a>
    </h3>
    <ul class="govuk-list search-result-details govuk-!-margin-bottom-2">
        <li>08 April 2026</li>
        <li><strong>Beta AI</strong> - <span>Remote (UK)</span></li>
        <li class="govuk-tag govuk-tag--grey">Fully remote</li>
    </ul>
    <p class="govuk-body search-result-description">
        Remote ML engineer role focused on production PyTorch pipelines.
    </p>
</div>
<div class="search-result" data-aid="333">
    <p class="govuk-body search-result-description">
        Malformed card with no title link — must be skipped, not crash.
    </p>
</div>
</body></html>
"""


def _install_fake_get(monkeypatch, status_code: int = 200, text: str = SAMPLE_SEARCH_HTML) -> None:
    async def fake_get(self, url, params=None, **kwargs):
        return httpx.Response(status_code, text=text, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)


class TestUkGovFindAJobConnector:
    def test_parses_valid_search_results(self, monkeypatch):
        _install_fake_get(monkeypatch)
        connector = UkGovFindAJobConnector()

        jobs = asyncio.run(connector.search(["data scientist"], location="UK"))

        # The third card has no title <a> and must be skipped, not crash the batch.
        assert len(jobs) == 2

        first, second = jobs
        assert first.title == "Data Scientist"
        assert first.company == "Acme Analytics Ltd"
        assert first.location == "Manchester"
        assert first.salary_min == 45000
        assert first.salary_max == 60000
        assert first.posted_date == date(2026, 4, 7)
        assert first.work_model == "hybrid"
        assert first.url == "https://www.jobs.service.gov.uk/details/111"
        assert first.source == "uk_gov_find_a_job"
        assert first.job_id == "ukgov_111"

        assert second.title == "ML Engineer"
        assert second.company == "Beta AI"
        assert second.location == "Remote (UK)"
        assert second.salary_min is None
        assert second.posted_date == date(2026, 4, 8)
        assert second.work_model == "remote"

    def test_non_200_response_returns_empty_list(self, monkeypatch):
        """Simulates the Akamai bot-wall (403) documented in the connector docstring."""
        _install_fake_get(monkeypatch, status_code=403, text="blocked")
        connector = UkGovFindAJobConnector()

        jobs = asyncio.run(connector.search(["data scientist"], location="UK"))

        assert jobs == []

    def test_empty_or_unrecognised_html_returns_empty_list(self, monkeypatch):
        _install_fake_get(monkeypatch, text="<html><body><p>No jobs today</p></body></html>")
        connector = UkGovFindAJobConnector()

        jobs = asyncio.run(connector.search(["data scientist"], location="UK"))

        assert jobs == []

    def test_search_never_raises_on_transport_error(self, monkeypatch):
        async def raising_get(self, url, params=None, **kwargs):
            raise httpx.ConnectError("connection refused", request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx.AsyncClient, "get", raising_get)
        connector = UkGovFindAJobConnector()

        jobs = asyncio.run(connector.search(["data scientist"], location="UK"))

        assert jobs == []

    def test_respects_daily_quota(self, monkeypatch):
        calls = {"count": 0}

        async def fake_get(self, url, params=None, **kwargs):
            calls["count"] += 1
            return httpx.Response(200, text=SAMPLE_SEARCH_HTML, request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
        connector = UkGovFindAJobConnector()
        connector.daily_quota = 2

        asyncio.run(connector.search(["a", "b", "c", "d"], location="UK"))

        assert calls["count"] == 2
