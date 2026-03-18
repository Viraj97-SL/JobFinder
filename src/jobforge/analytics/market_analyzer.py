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

from jobforge.memory.dedup_store import get_engine

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
        """Average salary range across all jobs with disclosed salaries."""
        df = self._load_jobs()
        if df.empty:
            return {}

        has_salary = df[(df["salary_min"].notna()) | (df["salary_max"].notna())]
        if has_salary.empty:
            return {"disclosed_count": 0}

        avg_min = has_salary["salary_min"].dropna().mean()
        avg_max = has_salary["salary_max"].dropna().mean()
        median_mid = (
            ((has_salary["salary_min"].fillna(0) + has_salary["salary_max"].fillna(0)) / 2)
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
