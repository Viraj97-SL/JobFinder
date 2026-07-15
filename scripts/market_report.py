"""
JobForge AI — Standalone Market Intelligence Report.

Reads from the local DB and prints a market trend analysis to stdout.
No LLM calls, no API keys required.

Usage:
    python scripts/market_report.py
    python scripts/market_report.py --days 30
    python scripts/market_report.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make sure the src package is importable when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jobforge.memory.dedup_store import ReportArchive, init_database
from jobforge.analytics.market_analyzer import MarketAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser(description="JobForge AI — Market Intelligence Report")
    parser.add_argument("--days", type=int, default=90, help="Lookback window in days (default: 90)")
    parser.add_argument("--json", action="store_true", help="Output structured JSON instead of text")
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Persist this report snapshot to market_report_archive (append-only) before printing",
    )
    args = parser.parse_args()

    init_database()
    analyzer = MarketAnalyzer(lookback_days=args.days)

    if args.json or args.archive:
        report = analyzer.build_market_report()

        if args.archive:
            ReportArchive().archive(report)

        if args.json:
            print(json.dumps(report.model_dump(mode="json"), indent=2))
        else:
            print(analyzer.generate_text_report())
    else:
        print(analyzer.generate_text_report())


if __name__ == "__main__":
    main()
