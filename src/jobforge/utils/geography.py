"""
JobForge AI — UK Location Normalisation.

Job board location fields are inconsistent: a bare city name ("London"), a
"City, Remote" combo, or a raw postcode with no space ("WC2B5LX"). This
module normalises all of these to a single city/region label so analytics
can report a real geographic distribution instead of dozens of near-duplicate
location strings.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

_DATA_FILE = Path(__file__).resolve().parents[3] / "data" / "uk_postcode_areas.json"

# Matches a full UK postcode with or without the internal space, e.g.
# "WC2B 5LX" or "WC2B5LX". Captures the leading 1-2 letter postcode area.
_POSTCODE_PATTERN = re.compile(r"\b([A-Z]{1,2})\d[A-Z\d]?\s?\d[A-Z]{2}\b", re.IGNORECASE)

_REMOTE_PATTERN = re.compile(r"\bremote\b|\bwork\s*from\s*home\b|\bwfh\b", re.IGNORECASE)


@lru_cache(maxsize=1)
def _postcode_area_lookup() -> dict[str, str]:
    with open(_DATA_FILE, encoding="utf-8") as f:
        raw: dict[str, str] = json.load(f)
    return {area.upper(): city for area, city in raw.items()}


@lru_cache(maxsize=1)
def _known_cities() -> tuple[str, ...]:
    """Every city/region name in the lookup, longest-first so substring
    matching prefers the more specific name (e.g. "Milton Keynes" before
    a shorter false-positive substring)."""
    return tuple(sorted(set(_postcode_area_lookup().values()), key=len, reverse=True))


def normalize_location(location: str | None) -> str:
    """
    Normalise a raw job location string to a city/region label.

    Priority: explicit postcode > known city name substring > "Remote" > "Other UK".
    """
    if not location or not location.strip():
        return "Unknown"

    postcode_match = _POSTCODE_PATTERN.search(location)
    if postcode_match:
        area = postcode_match.group(1).upper()
        city = _postcode_area_lookup().get(area)
        if city:
            return city

    lowered = location.lower()
    for city in _known_cities():
        if city.lower() in lowered:
            return city

    if _REMOTE_PATTERN.search(location):
        return "Remote"

    return "Other UK"
