"""
JobForge AI — HN "Who is hiring?" Connector Tests.

All HTTP calls (both the Algolia search endpoint and the Firebase item API)
are mocked via monkeypatching `httpx.AsyncClient.get` — no network calls are
made. Fixtures model the shape those two APIs actually return.
"""

from __future__ import annotations

import asyncio

import httpx

from jobforge.connectors.hn_who_is_hiring import (
    ALGOLIA_SEARCH_URL,
    FIREBASE_ITEM_URL,
    HNWhoIsHiringConnector,
)

THREAD_ID = 44444444

ALGOLIA_RESPONSE = {
    "hits": [
        {"author": "someoneelse", "title": "Ask HN: What are you building?", "objectID": "999"},
        {
            "author": "whoishiring",
            "title": "Ask HN: Who is hiring? (July 2026)",
            "objectID": str(THREAD_ID),
        },
    ]
}

THREAD_ITEM = {"id": THREAD_ID, "kids": [201, 202, 203]}

COMMENT_ITEMS = {
    201: {
        "id": 201,
        "time": 1751328000,
        "text": (
            "Acme Corp | Senior ML Engineer | London, UK (Remote) | Full-time"
            "<p>We are looking for a senior machine learning engineer skilled in "
            "python and pytorch to join our applied research team. Remote friendly, "
            "great benefits, competitive salary."
        ),
    },
    202: {
        "id": 202,
        "deleted": True,
        "text": "",
    },
    203: {
        "id": 203,
        "time": 1751328100,
        "text": (
            "Beta Analytics - Data Scientist"
            "<p>Join our small data team building python and machine learning "
            "pipelines for retail clients across the UK. Great team culture and "
            "remote-friendly setup."
        ),
    },
}


def _install_fake_get(monkeypatch, comment_items: dict | None = None) -> None:
    items = COMMENT_ITEMS if comment_items is None else comment_items

    async def fake_get(self, url, params=None, **kwargs):
        if url == ALGOLIA_SEARCH_URL:
            return httpx.Response(200, json=ALGOLIA_RESPONSE, request=httpx.Request("GET", url))
        if url == FIREBASE_ITEM_URL.format(item_id=THREAD_ID):
            return httpx.Response(200, json=THREAD_ITEM, request=httpx.Request("GET", url))
        for comment_id, item in items.items():
            if url == FIREBASE_ITEM_URL.format(item_id=comment_id):
                return httpx.Response(200, json=item, request=httpx.Request("GET", url))
        return httpx.Response(404, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)


class TestHNWhoIsHiringConnector:
    def test_parses_valid_comments_into_jobs(self, monkeypatch):
        _install_fake_get(monkeypatch)
        connector = HNWhoIsHiringConnector()

        jobs = asyncio.run(
            connector.search(["Machine Learning Engineer", "Data Scientist"], location="UK")
        )

        # Comment 202 is deleted and must be skipped, not crash the batch.
        assert len(jobs) == 2

        by_id = {j.job_id: j for j in jobs}

        pipe_job = by_id["hn_201"]
        assert pipe_job.title == "Senior ML Engineer"
        assert pipe_job.company == "Acme Corp"
        assert pipe_job.url == "https://news.ycombinator.com/item?id=201"
        assert pipe_job.source == "hn_who_is_hiring"
        assert pipe_job.work_model == "remote"

        dash_job = by_id["hn_203"]
        assert dash_job.title == "Data Scientist"
        assert dash_job.company == "Beta Analytics"

    def test_no_matching_thread_returns_empty_list(self, monkeypatch):
        async def fake_get(self, url, params=None, **kwargs):
            if url == ALGOLIA_SEARCH_URL:
                no_match = {
                    "hits": [{"author": "someoneelse", "title": "Unrelated", "objectID": "1"}]
                }
                return httpx.Response(200, json=no_match, request=httpx.Request("GET", url))
            return httpx.Response(404, request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
        connector = HNWhoIsHiringConnector()

        jobs = asyncio.run(connector.search(["Data Scientist"], location="UK"))

        assert jobs == []

    def test_thread_fetch_failure_returns_empty_list(self, monkeypatch):
        async def fake_get(self, url, params=None, **kwargs):
            if url == ALGOLIA_SEARCH_URL:
                return httpx.Response(200, json=ALGOLIA_RESPONSE, request=httpx.Request("GET", url))
            # Thread item fetch (and everything else) errors out.
            return httpx.Response(500, request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
        connector = HNWhoIsHiringConnector()

        jobs = asyncio.run(connector.search(["Data Scientist"], location="UK"))

        assert jobs == []

    def test_malformed_comment_json_is_skipped_not_raised(self, monkeypatch):
        async def fake_get(self, url, params=None, **kwargs):
            if url == ALGOLIA_SEARCH_URL:
                return httpx.Response(200, json=ALGOLIA_RESPONSE, request=httpx.Request("GET", url))
            if url == FIREBASE_ITEM_URL.format(item_id=THREAD_ID):
                return httpx.Response(200, json=THREAD_ITEM, request=httpx.Request("GET", url))
            # Every comment fetch returns invalid JSON — connector must degrade to [].
            return httpx.Response(200, text="not json", request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
        connector = HNWhoIsHiringConnector()

        jobs = asyncio.run(connector.search(["Data Scientist"], location="UK"))

        assert jobs == []

    def test_empty_or_too_short_comment_is_skipped(self, monkeypatch):
        short_items = {
            201: {"id": 201, "time": 1751328000, "text": "too short"},
            202: {"id": 202, "text": ""},
            203: {"id": 203, "time": 1751328100, "text": None},
        }
        _install_fake_get(monkeypatch, comment_items=short_items)
        connector = HNWhoIsHiringConnector()

        jobs = asyncio.run(connector.search(["Data Scientist"], location="UK"))

        assert jobs == []

    def test_respects_daily_quota(self, monkeypatch):
        calls = {"count": 0}

        async def fake_get(self, url, params=None, **kwargs):
            calls["count"] += 1
            if url == ALGOLIA_SEARCH_URL:
                return httpx.Response(200, json=ALGOLIA_RESPONSE, request=httpx.Request("GET", url))
            if url == FIREBASE_ITEM_URL.format(item_id=THREAD_ID):
                return httpx.Response(200, json=THREAD_ITEM, request=httpx.Request("GET", url))
            for comment_id, item in COMMENT_ITEMS.items():
                if url == FIREBASE_ITEM_URL.format(item_id=comment_id):
                    return httpx.Response(200, json=item, request=httpx.Request("GET", url))
            return httpx.Response(404, request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
        connector = HNWhoIsHiringConnector()
        connector.daily_quota = 0

        jobs = asyncio.run(connector.search(["Data Scientist"], location="UK"))

        assert jobs == []
        assert calls["count"] == 0
