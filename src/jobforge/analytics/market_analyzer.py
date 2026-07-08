"""
JobForge AI — Job Market Analyzer.

Reads from the job_analytics and score_history tables to surface
UK AI/ML/DS market trends across pipeline runs.

Usage:
    from jobforge.analytics.market_analyzer import MarketAnalyzer
    report = MarketAnalyzer().generate_text_report()
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta

import pandas as pd
import structlog
from sqlalchemy import text

from jobforge.analytics.role_classifier import classify_seniority
from jobforge.analytics.trends import classify_rising_cooling, classify_trend, week_start
from jobforge.analytics.validation import DEFAULT_DIVERGENCE_THRESHOLD, check_salary_divergence
from jobforge.memory.dedup_store import get_engine
from jobforge.utils.geography import normalize_location

logger = structlog.get_logger(__name__)


class MarketAnalyzer:
    """Aggregate job market intelligence from the analytics DB."""

    def __init__(self, lookback_days: int = 90) -> None:
        self._engine = get_engine()
        self._cutoff = (datetime.utcnow() - timedelta(days=lookback_days)).isoformat()

    # ── Data Loaders ─────────────────────────────────────────────────────────

    def _load_jobs(self) -> pd.DataFrame:
        """Load job_analytics rows within the lookback window."""
        with self._engine.connect() as conn:
            df = pd.read_sql(
                f"SELECT * FROM job_analytics WHERE scraped_at >= '{self._cutoff}'",
                conn,
            )
        return df

    def _load_jobs_between(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Load job_analytics rows scraped in [start, end) — used for windowed deltas."""
        with self._engine.connect() as conn:
            df = pd.read_sql(
                text("SELECT * FROM job_analytics WHERE scraped_at >= :start AND scraped_at < :end"),
                conn,
                params={"start": start.isoformat(), "end": end.isoformat()},
            )
        return df

    def _load_runs(self) -> pd.DataFrame:
        """Load run_history for trend analysis."""
        with self._engine.connect() as conn:
            df = pd.read_sql(
                "SELECT * FROM run_history WHERE status = 'complete' ORDER BY started_at DESC LIMIT 20",
                conn,
            )
        return df

    def _load_score_history(self) -> pd.DataFrame:
        with self._engine.connect() as conn:
            df = pd.read_sql(
                f"""
                SELECT sh.*, rh.started_at AS run_date
                FROM score_history sh
                JOIN run_history rh ON sh.run_id = rh.run_id
                WHERE rh.started_at >= '{self._cutoff}'
                ORDER BY rh.started_at
                """,
                conn,
            )
        return df

    # ── Analysis Methods ─────────────────────────────────────────────────────

    def top_demanded_skills(self, n: int = 10) -> list[tuple[str, int]]:
        """
        Most frequently matched skills across all job postings.
        Based on the matched_skills_json column populated at scrape time.
        """
        df = self._load_jobs()
        if df.empty or "matched_skills_json" not in df.columns:
            return []

        counter: Counter = Counter()
        for raw in df["matched_skills_json"].dropna():
            try:
                skills = json.loads(raw)
                counter.update(skills)
            except (json.JSONDecodeError, TypeError):
                continue

        return counter.most_common(n)

    def work_model_distribution(self) -> dict[str, int]:
        """Count of jobs by work model (remote / hybrid / onsite / unknown)."""
        df = self._load_jobs()
        if df.empty:
            return {}
        counts = df["work_model"].fillna("unknown").value_counts().to_dict()
        return {str(k): int(v) for k, v in counts.items()}

    def salary_stats(self) -> dict[str, float]:
        """
        Headline annual salary stats across jobs with a disclosed *annual* salary.

        Restricted to salary_period == "annual" (detected by utils/salary_parser.py).
        Day-rate/hourly contract roles have an annual-equivalent computed too
        (salary_annual_min/max), but are deliberately excluded from this headline
        figure — mixing contract-equivalent pay into the permanent-role annual
        median is exactly what corrupted this figure before. Unparseable/garbage
        salary values (period == "unknown") are excluded outright.
        """
        df = self._load_jobs()
        if df.empty:
            return {}

        required_cols = {"salary_period", "salary_annual_min", "salary_annual_max"}
        if not required_cols.issubset(df.columns):
            return {"disclosed_count": 0}

        has_salary = df[
            (df["salary_period"] == "annual")
            & (df["salary_annual_min"].notna() | df["salary_annual_max"].notna())
        ]
        if has_salary.empty:
            return {"disclosed_count": 0}

        avg_min = has_salary["salary_annual_min"].dropna().mean()
        avg_max = has_salary["salary_annual_max"].dropna().mean()
        median_mid = (
            ((has_salary["salary_annual_min"].fillna(0) + has_salary["salary_annual_max"].fillna(0)) / 2)
            .replace(0, float("nan"))
            .dropna()
            .median()
        )

        return {
            "disclosed_count": int(len(has_salary)),
            "avg_min": round(avg_min, 0) if pd.notna(avg_min) else None,
            "avg_max": round(avg_max, 0) if pd.notna(avg_max) else None,
            "median_midpoint": round(median_mid, 0) if pd.notna(median_mid) else None,
        }

    def _annual_midpoints(self, df: pd.DataFrame) -> pd.Series:
        """
        Per-job representative annual salary (mean of whichever of
        salary_annual_min/max are disclosed), restricted to period == "annual".
        Used for true percentiles rather than an average-of-averages.
        """
        required_cols = {"salary_period", "salary_annual_min", "salary_annual_max"}
        if df.empty or not required_cols.issubset(df.columns):
            return pd.Series([], dtype=float)
        annual = df[df["salary_period"] == "annual"]
        if annual.empty:
            return pd.Series([], dtype=float)
        midpoints = annual[["salary_annual_min", "salary_annual_max"]].mean(axis=1, skipna=True)
        return midpoints.dropna()

    @staticmethod
    def _percentile_summary(values: pd.Series) -> dict[str, float]:
        return {
            "n": int(len(values)),
            "p25": round(float(values.quantile(0.25)), 0),
            "p50": round(float(values.quantile(0.50)), 0),
            "p75": round(float(values.quantile(0.75)), 0),
        }

    def salary_percentiles(self) -> dict[str, float]:
        """
        True P25/P50/P75 of the annual-salary distribution (not a
        midpoint-of-average like salary_stats()'s legacy fields).
        """
        midpoints = self._annual_midpoints(self._load_jobs())
        if midpoints.empty:
            return {"n": 0}
        return self._percentile_summary(midpoints)

    def salary_by_category(self) -> dict[str, dict[str, float]]:
        """Salary percentiles segmented by role_category (1.3 + 1.4)."""
        df = self._load_jobs()
        if df.empty or "role_category" not in df.columns:
            return {}
        result: dict[str, dict[str, float]] = {}
        for category, group in df.groupby(df["role_category"].fillna("Other")):
            midpoints = self._annual_midpoints(group)
            if not midpoints.empty:
                result[str(category)] = self._percentile_summary(midpoints)
        return result

    def salary_by_seniority(self) -> dict[str, dict[str, float]]:
        """Salary percentiles segmented by seniority parsed from title."""
        df = self._load_jobs()
        if df.empty or "title" not in df.columns:
            return {}
        seniority = df["title"].fillna("").apply(classify_seniority)
        result: dict[str, dict[str, float]] = {}
        for level, group in df.groupby(seniority):
            midpoints = self._annual_midpoints(group)
            if not midpoints.empty:
                result[str(level)] = self._percentile_summary(midpoints)
        return result

    def salary_divergence_check(self, threshold: float = DEFAULT_DIVERGENCE_THRESHOLD) -> dict[str, object]:
        """
        Compare this week's salary snapshot median against the rolling
        (lookback_days) median and flag if they diverge beyond threshold —
        see analytics/validation.py for why this guard exists.
        """
        rolling_median = self.salary_percentiles().get("p50")

        now = datetime.utcnow()
        weekly_df = self._load_jobs_between(now - timedelta(days=7), now)
        weekly_midpoints = self._annual_midpoints(weekly_df)
        weekly_median = float(weekly_midpoints.median()) if not weekly_midpoints.empty else None

        return check_salary_divergence(weekly_median, rolling_median, threshold)

    def role_category_distribution(self) -> dict[str, int]:
        """Count of jobs per role_category (1.3 segmentation)."""
        df = self._load_jobs()
        if df.empty or "role_category" not in df.columns:
            return {}
        counts = df["role_category"].fillna("Other").value_counts().to_dict()
        return {str(k): int(v) for k, v in counts.items()}

    def geographic_distribution(self, n: int = 10) -> list[tuple[str, int]]:
        """
        Top-N UK city/region breakdown (1.6). Uses the region column
        pre-computed at scrape time; falls back to live normalisation for
        older rows scraped before that column existed.
        """
        df = self._load_jobs()
        if df.empty:
            return []
        if "region" in df.columns:
            regions = df["region"]
            missing = regions.isna()
            if missing.any() and "location" in df.columns:
                regions = regions.copy()
                regions.loc[missing] = df.loc[missing, "location"].apply(normalize_location)
        elif "location" in df.columns:
            regions = df["location"].apply(normalize_location)
        else:
            return []
        counts = regions.fillna("Unknown").value_counts()
        return [(str(region), int(cnt)) for region, cnt in counts.head(n).items()]

    def company_stage_distribution(self) -> dict[str, int]:
        """Count of jobs per company_stage (1.6)."""
        df = self._load_jobs()
        if df.empty:
            return {}
        counts = df["company_stage"].fillna("unknown").value_counts().to_dict()
        return {str(k): int(v) for k, v in counts.items()}

    # ── Week-over-Week Deltas & Skill Trends (1.1, 1.2, 1.5) ────────────────

    def _compute_metric(self, df: pd.DataFrame, metric: str) -> float | None:
        """Compute a single scalar metric value over a job_analytics slice."""
        if df.empty:
            return None

        if metric == "total_volume":
            return float(len(df))

        if metric == "sponsorship_rate":
            total = len(df)
            sponsoring = int((df["offers_sponsorship"] == 1).sum())
            return round(100 * sponsoring / total, 1) if total else None

        if metric == "startup_share":
            total = len(df)
            startups = int((df["is_startup"] == 1).sum())
            return round(100 * startups / total, 1) if total else None

        if metric == "salary_median":
            midpoints = self._annual_midpoints(df)
            return float(midpoints.median()) if not midpoints.empty else None

        # Fallback: treat `metric` as a skill name — count of JDs mentioning it.
        if "matched_skills_json" not in df.columns:
            return None
        count = 0
        for raw in df["matched_skills_json"].dropna():
            try:
                if metric in json.loads(raw):
                    count += 1
            except (json.JSONDecodeError, TypeError):
                continue
        return float(count)

    def metric_deltas(self, metric: str, weeks: int = 1) -> dict[str, object]:
        """
        Current-vs-prior-window change for a metric: "total_volume",
        "sponsorship_rate", "startup_share", "salary_median", or any skill
        name (treated as a JD-mention count).

        Returns absolute + relative change and a direction flag, so a report
        can say "Python +12% WoW" instead of restating a flat count.
        """
        now = datetime.utcnow()
        window = timedelta(days=7 * weeks)
        current_start = now - window
        previous_start = current_start - window

        current_value = self._compute_metric(self._load_jobs_between(current_start, now), metric)
        previous_value = self._compute_metric(self._load_jobs_between(previous_start, current_start), metric)

        if current_value is None or previous_value is None:
            return {
                "metric": metric,
                "weeks": weeks,
                "current": current_value,
                "previous": previous_value,
                "abs_change": None,
                "pct_change": None,
                "direction": "unknown",
            }

        abs_change = round(current_value - previous_value, 2)
        pct_change = round(100 * abs_change / previous_value, 1) if previous_value else None
        direction = "up" if abs_change > 0 else ("down" if abs_change < 0 else "flat")

        return {
            "metric": metric,
            "weeks": weeks,
            "current": current_value,
            "previous": previous_value,
            "abs_change": abs_change,
            "pct_change": pct_change,
            "direction": direction,
        }

    def _skill_weekly_counts(self, lookback_weeks: int, top_n: int) -> dict[str, list[int]]:
        """Dense (gap-filled) weekly mention counts for the top_n most-mentioned skills."""
        df = self._load_jobs()
        if df.empty or "matched_skills_json" not in df.columns:
            return {}

        cutoff = datetime.utcnow() - timedelta(weeks=lookback_weeks)
        scraped = pd.to_datetime(df["scraped_at"])
        df = df[scraped >= cutoff]
        if df.empty:
            return {}

        records: list[tuple[datetime, str]] = []
        for scraped_at, raw in zip(pd.to_datetime(df["scraped_at"]), df["matched_skills_json"]):
            try:
                skills = json.loads(raw) if raw else []
            except (json.JSONDecodeError, TypeError):
                continue
            week = week_start(scraped_at.to_pydatetime())
            records.extend((week, skill) for skill in skills)

        if not records:
            return {}

        skill_week_df = pd.DataFrame(records, columns=["week", "skill"])
        top_skills = skill_week_df["skill"].value_counts().head(top_n).index.tolist()
        all_weeks = sorted(skill_week_df["week"].unique())

        result: dict[str, list[int]] = {}
        for skill in top_skills:
            counts_by_week = skill_week_df[skill_week_df["skill"] == skill].groupby("week").size()
            result[skill] = [int(counts_by_week.get(w, 0)) for w in all_weeks]
        return result

    def skill_trajectories(self, lookback_weeks: int = 12, top_n: int = 20) -> dict[str, dict]:
        """
        Per-skill weekly mention history + trend classification
        (Accelerating / Cooling / Stable / New) over the lookback window.
        """
        weekly_counts_by_skill = self._skill_weekly_counts(lookback_weeks, top_n)
        return {
            skill: {"weekly_counts": counts, **classify_trend(counts)}
            for skill, counts in weekly_counts_by_skill.items()
        }

    def rising_cooling_skills(self, lookback_weeks: int = 6, top_n: int = 20) -> dict[str, list[str]]:
        """
        Statistically smoothed Rising/Cooling/Stable skill lists (1.5) —
        trailing-3-week slope + comparison to the trailing mean, so a single
        noisy week doesn't flip the label.
        """
        weekly_counts_by_skill = self._skill_weekly_counts(lookback_weeks, top_n)
        buckets: dict[str, list[str]] = {"rising": [], "cooling": [], "stable": []}
        for skill, counts in weekly_counts_by_skill.items():
            label = classify_rising_cooling(counts)
            buckets[label.lower()].append(skill)
        return buckets

    def sponsorship_rate(self) -> dict[str, object]:
        """Percentage of roles offering visa sponsorship."""
        df = self._load_jobs()
        if df.empty:
            return {}

        total = len(df)
        sponsoring = int((df["offers_sponsorship"] == 1).sum())
        citizens_only = int((df["offers_sponsorship"] == 0).sum())
        unknown = total - sponsoring - citizens_only

        return {
            "total": total,
            "sponsoring": sponsoring,
            "sponsoring_pct": round(100 * sponsoring / total, 1) if total else 0,
            "citizens_only": citizens_only,
            "unknown": unknown,
        }

    def startup_ratio(self) -> dict[str, object]:
        """Ratio of startup vs established company postings."""
        df = self._load_jobs()
        if df.empty:
            return {}
        total = len(df)
        startups = int((df["is_startup"] == 1).sum())
        return {
            "total": total,
            "startups": startups,
            "startup_pct": round(100 * startups / total, 1) if total else 0,
        }

    def top_hiring_companies(self, n: int = 10) -> list[tuple[str, int]]:
        """Companies with the most unique job postings."""
        df = self._load_jobs()
        if df.empty:
            return []
        counts = df["company"].value_counts().head(n)
        return [(str(co), int(cnt)) for co, cnt in counts.items()]

    def source_breakdown(self) -> dict[str, int]:
        """Job count per source connector."""
        df = self._load_jobs()
        if df.empty:
            return {}
        counts = df["source"].value_counts().to_dict()
        return {str(k): int(v) for k, v in counts.items()}

    def score_trend(self) -> list[dict]:
        """Average match score per run (last 10 runs)."""
        df = self._load_score_history()
        if df.empty:
            return []

        trend = (
            df.groupby("run_date")["overall_score"]
            .agg(avg_score="mean", count="count", max_score="max")
            .reset_index()
            .sort_values("run_date", ascending=False)
            .head(10)
        )
        return [
            {
                "run_date": row["run_date"][:10],
                "avg_score": round(row["avg_score"], 1),
                "max_score": round(row["max_score"], 1),
                "jobs_scored": int(row["count"]),
            }
            for _, row in trend.iterrows()
        ]

    def cv_variant_distribution(self) -> dict[str, int]:
        """How often each CV variant was recommended."""
        df = self._load_jobs()
        if df.empty or "cv_variant" not in df.columns:
            return {}
        counts = df["cv_variant"].dropna().value_counts().to_dict()
        return {str(k): int(v) for k, v in counts.items()}

    # ── Report Generator ─────────────────────────────────────────────────────

    def generate_text_report(self) -> str:
        """Generate a plain-text market intelligence summary for email / console."""
        lines: list[str] = []
        sep = "-" * 50

        lines.append("MARKET INTELLIGENCE REPORT")
        lines.append(f"(last 90 days  |  generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC)")
        lines.append(sep)

        # Skills
        skills = self.top_demanded_skills(10)
        if skills:
            lines.append("\nTop Demanded Skills in UK AI/ML/DS Roles:")
            for i, (skill, cnt) in enumerate(skills, 1):
                lines.append(f"  {i:2}. {skill:<30} {cnt} mentions")

        # Work model
        wm = self.work_model_distribution()
        if wm:
            lines.append(f"\nWork Model Distribution:")
            for model, cnt in sorted(wm.items(), key=lambda x: -x[1]):
                lines.append(f"  {model:<12} {cnt}")

        # Salary
        sal = self.salary_stats()
        if sal.get("disclosed_count", 0) > 0:
            lines.append(f"\nSalary (roles with disclosed pay, n={sal['disclosed_count']}):")
            if sal.get("avg_min"):
                lines.append(f"  Avg range: £{sal['avg_min']:,.0f} – £{sal['avg_max']:,.0f}")
            if sal.get("median_midpoint"):
                lines.append(f"  Median midpoint: £{sal['median_midpoint']:,.0f}")

        # Sponsorship
        sp = self.sponsorship_rate()
        if sp.get("total", 0) > 0:
            lines.append(f"\nVisa Sponsorship (n={sp['total']}):")
            lines.append(f"  Offering sponsorship: {sp['sponsoring']} ({sp['sponsoring_pct']}%)")
            lines.append(f"  Citizens only:        {sp['citizens_only']}")
            lines.append(f"  Unknown/not stated:   {sp['unknown']}")

        # Startup ratio
        sr = self.startup_ratio()
        if sr.get("total", 0) > 0:
            lines.append(f"\nStartup vs Established ({sr['total']} total):")
            lines.append(f"  Startups: {sr['startups']} ({sr['startup_pct']}%)")

        # Top companies
        companies = self.top_hiring_companies(8)
        if companies:
            lines.append("\nMost Active Hiring Companies:")
            for co, cnt in companies:
                lines.append(f"  {co:<35} {cnt} roles")

        # Source breakdown
        sources = self.source_breakdown()
        if sources:
            lines.append("\nJobs by Source:")
            for src, cnt in sorted(sources.items(), key=lambda x: -x[1]):
                lines.append(f"  {src:<20} {cnt}")

        # Score trend
        trend = self.score_trend()
        if trend:
            lines.append("\nMatch Score Trend (recent runs):")
            lines.append(f"  {'Date':<12} {'Avg':>6} {'Max':>6} {'Scored':>8}")
            for row in trend[:5]:
                lines.append(
                    f"  {row['run_date']:<12} {row['avg_score']:>6.1f} "
                    f"{row['max_score']:>6.1f} {row['jobs_scored']:>8}"
                )

        # CV variant
        variants = self.cv_variant_distribution()
        if variants:
            lines.append("\nRecommended CV Variant Distribution:")
            for variant, cnt in sorted(variants.items(), key=lambda x: -x[1]):
                lines.append(f"  {variant:<20} {cnt}")

        lines.append(f"\n{sep}")
        return "\n".join(lines)
