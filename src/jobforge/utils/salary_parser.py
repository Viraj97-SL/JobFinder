"""
JobForge AI — Salary Period Parsing & Normalisation.

UK job boards mix pay periods (annual salary, day rate, hourly rate) in the
same salary_min/salary_max fields with no period marker. Contractor day-rate
listings (e.g. £700-£800) averaged straight in with permanent annual salaries
silently corrupt the salary median. This module detects the pay period from
job text and converts day/hour rates to an annual-equivalent figure, kept
separate from the headline (period-pure) annual salary.
"""

from __future__ import annotations

import re
from typing import Literal

SalaryPeriod = Literal["annual", "daily", "hourly", "unknown"]

# Contractor-standard assumptions for day/hour -> annual-equivalent conversion.
WORKING_DAYS_PER_YEAR = 220
HOURS_PER_DAY = 7.5
HOURS_PER_YEAR = WORKING_DAYS_PER_YEAR * HOURS_PER_DAY  # 1650

# Below this, an "annual" figure is almost certainly garbage/misparsed data
# (e.g. the "£0k-£0k", "£1k-£1k" entries seen in the raw feed) rather than a
# real UK salary.
MIN_PLAUSIBLE_ANNUAL = 5_000
# Above these, a "daily"/"hourly" figure is implausible for that period and is
# more likely a misclassified annual figure — don't project it into a
# multi-million "annual-equivalent".
MAX_PLAUSIBLE_DAILY = 3_000
MAX_PLAUSIBLE_HOURLY = 500

_DAILY_PATTERN = re.compile(r"\bper\s*day\b|\bper\s*diem\b|\bday\s*rate\b|\bp\s*/\s*day\b|/\s*day\b", re.IGNORECASE)
_HOURLY_PATTERN = re.compile(r"\bper\s*hour\b|\bhourly\s*rate\b|\bp\s*/\s*h\b|/\s*h(?:ou)?r\b", re.IGNORECASE)
_ANNUAL_PATTERN = re.compile(
    r"\bper\s*annum\b|\bper\s*year\b|\bannual\s*salary\b|\bp\.?a\.?\b|/\s*year\b|/\s*annum\b", re.IGNORECASE
)


def detect_salary_period(text: str, salary_min: float | None, salary_max: float | None) -> SalaryPeriod:
    """
    Detect the pay period for a job's salary figures.

    Priority: explicit textual cue (title/description) > numeric plausibility
    heuristic > "unknown". Explicit cues take priority because they're the
    only reliable signal — the same numeric range can be a plausible annual
    salary for a junior role or a plausible day rate for a contractor.
    """
    if salary_min is None and salary_max is None:
        return "unknown"

    if _DAILY_PATTERN.search(text):
        return "daily"
    if _HOURLY_PATTERN.search(text):
        return "hourly"
    if _ANNUAL_PATTERN.search(text):
        return "annual"

    values = [v for v in (salary_min, salary_max) if v is not None]
    if not values:
        return "unknown"

    # No explicit cue. UK job boards default to annual salary for permanent
    # roles far more often than not, so treat a plausible annual figure as
    # annual; anything too low to be a real annual salary is unclassifiable
    # garbage rather than a guessed day/hour rate.
    if max(values) < MIN_PLAUSIBLE_ANNUAL:
        return "unknown"
    return "annual"


def normalize_to_annual(
    salary_min: float | None, salary_max: float | None, period: SalaryPeriod
) -> tuple[float | None, float | None]:
    """
    Convert salary figures to an annual-equivalent, or None where not computable.

    "unknown" period always yields (None, None) — a value that can't be placed
    on a timescale must never enter the annual median. Values implausible for
    their detected period (e.g. a "daily" rate of 50000) are also dropped
    rather than projected into a nonsensical annual-equivalent.
    """
    if period == "unknown":
        return None, None

    multiplier = {"annual": 1, "daily": WORKING_DAYS_PER_YEAR, "hourly": HOURS_PER_YEAR}[period]
    cap = {"annual": None, "daily": MAX_PLAUSIBLE_DAILY, "hourly": MAX_PLAUSIBLE_HOURLY}[period]

    def convert(value: float | None) -> float | None:
        if value is None:
            return None
        if cap is not None and value > cap:
            return None
        annual = value * multiplier
        return annual if annual >= MIN_PLAUSIBLE_ANNUAL else None

    return convert(salary_min), convert(salary_max)
