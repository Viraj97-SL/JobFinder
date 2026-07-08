"""
JobForge AI — Role Category Classifier.

Rules-based (regex/keyword) classification of a job title into a coarse
role category, so market analytics can segment by "what's happening to AI
Engineer roles vs Data Scientist roles" instead of reporting market-wide
averages only.

Checks run most-specific-first: e.g. "MLOps Engineer" must not fall through
to the generic "ML Engineer" bucket, and "Computer Vision Engineer" must not
fall through to "AI/LLM Engineer" just because it also contains "engineer".
"""

from __future__ import annotations

import re

ROLE_CATEGORIES = (
    "MLOps",
    "Computer Vision",
    "NLP",
    "Research",
    "AI/LLM Engineer",
    "ML Engineer",
    "Data Scientist",
    "Data Engineer",
    "Other",
)

# Ordered (category, pattern) pairs — first match wins.
_RULES: tuple[tuple[str, re.Pattern], ...] = (
    (
        "MLOps",
        re.compile(r"\bmlops\b|\bml\s*ops\b|\bml\s*platform\b|\bmachine\s*learning\s*platform\b|\bml\s*infra", re.IGNORECASE),
    ),
    (
        "Computer Vision",
        re.compile(r"\bcomputer\s*vision\b|\bcv\s*engineer\b|\bvision\s*engineer\b|\bimage\s*recognition\b", re.IGNORECASE),
    ),
    (
        "NLP",
        re.compile(r"\bnlp\b|\bnatural\s*language\s*processing\b", re.IGNORECASE),
    ),
    (
        "Research",
        re.compile(r"\bresearch\s*(scientist|engineer|associate)\b|\bapplied\s*scientist\b", re.IGNORECASE),
    ),
    (
        "AI/LLM Engineer",
        re.compile(
            r"\bai\s*engineer\b|\bllm\s*engineer\b|\bgenerative\s*ai\b|\bgen\s*ai\b|"
            r"\bprompt\s*engineer\b|\bai\s*/\s*ml\s*engineer\b|\bartificial\s*intelligence\s*engineer\b",
            re.IGNORECASE,
        ),
    ),
    (
        "ML Engineer",
        re.compile(r"\bmachine\s*learning\s*engineer\b|\bml\s*engineer\b", re.IGNORECASE),
    ),
    (
        "Data Scientist",
        re.compile(r"\bdata\s*scientist\b", re.IGNORECASE),
    ),
    (
        "Data Engineer",
        re.compile(r"\bdata\s*engineer\b", re.IGNORECASE),
    ),
)


def classify_role_category(title: str, description: str | None = None) -> str:
    """
    Classify a job into one of ROLE_CATEGORIES.

    Title is checked first (the strongest, most deliberate signal). If no
    rule matches the title, the same rules are checked against the
    description as a fallback. Returns "Other" if nothing matches either.
    """
    for category, pattern in _RULES:
        if pattern.search(title):
            return category

    if description:
        for category, pattern in _RULES:
            if pattern.search(description):
                return category

    return "Other"


SENIORITY_LEVELS = ("Principal", "Staff", "Lead", "Senior", "Mid", "Junior", "Unspecified")

# Ordered most-specific-first: "Lead" is checked before "Senior" so
# "Senior Lead Engineer" resolves to the more senior label.
_SENIORITY_RULES: tuple[tuple[str, re.Pattern], ...] = (
    ("Principal", re.compile(r"\bprincipal\b", re.IGNORECASE)),
    ("Staff", re.compile(r"\bstaff\b", re.IGNORECASE)),
    ("Lead", re.compile(r"\blead\b", re.IGNORECASE)),
    ("Senior", re.compile(r"\bsenior\b|\bsr\.?\b", re.IGNORECASE)),
    ("Junior", re.compile(r"\bjunior\b|\bjr\.?\b|\bgraduate\b|\bentry\s*level\b", re.IGNORECASE)),
    ("Mid", re.compile(r"\bmid\s*-?\s*level\b", re.IGNORECASE)),
)


def classify_seniority(title: str) -> str:
    """Classify seniority from a job title. Returns "Unspecified" if no cue matches."""
    for label, pattern in _SENIORITY_RULES:
        if pattern.search(title):
            return label
    return "Unspecified"
