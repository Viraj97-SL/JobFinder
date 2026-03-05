"""
JobForge AI — CLI Entry Point.

Usage:
    python scripts/run_pipeline.py              # Full pipeline
    python scripts/run_pipeline.py --scout-only # Scout agent only (test API pulls)
    python scripts/run_pipeline.py --dry-run    # No email, just generate outputs
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jobforge.agents.orchestrator import run_pipeline
from jobforge.agents.scout.agent import ScoutAgent
from jobforge.config.settings import DATA_DIR
from jobforge.memory.dedup_store import init_database
from jobforge.models.cv import SkillInventory
from jobforge.utils.logger import setup_logging


def load_skill_inventory() -> SkillInventory | None:
    """Load the pre-extracted skill inventory from JSON."""
    path = DATA_DIR / "skill_inventory.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        return SkillInventory.model_validate(data)
    print(f"[WARN] Skill inventory not found at {path}. Run extract_skill_inventory.py first.")
    return None


def load_watchlist() -> list[dict]:
    """Load the startup watchlist from YAML."""
    path = DATA_DIR / "startup_watchlist.yaml"
    if path.exists():
        try:
            import yaml
            with open(path) as f:
                return yaml.safe_load(f) or []
        except ImportError:
            print("[WARN] PyYAML not installed. Skipping startup watchlist.")
    return []


async def run_scout_only() -> None:
    """Test the Scout Agent in isolation."""
    init_database()
    watchlist = load_watchlist()
    scout = ScoutAgent(watchlist=watchlist)

    # Minimal state for scout-only run
    state = {"skill_inventory": None, "run_id": "test_scout"}
    result = await scout.run(state)

    deduped = result.get("deduped_jobs", [])
    metadata = result.get("scout_metadata", {})

    print(f"\n{'='*60}")
    print(f"SCOUT AGENT TEST RESULTS")
    print(f"{'='*60}")
    print(f"Sources queried:  {metadata.get('sources_queried', [])}")
    print(f"Source counts:    {metadata.get('source_counts', {})}")
    print(f"Source errors:    {metadata.get('source_errors', {})}")
    print(f"Total raw:        {metadata.get('total_raw', 0)}")
    print(f"After dedup:      {metadata.get('total_after_dedup', 0)}")
    print(f"Duration:         {metadata.get('duration_seconds', 0)}s")

    if deduped:
        print(f"\nTop 5 Jobs Found:")
        for i, job in enumerate(deduped[:5], 1):
            sponsor = "✓ Sponsor" if job.offers_sponsorship else ("⚠ UK Only" if job.citizens_only else "—")
            startup = " [STARTUP]" if job.is_startup else ""
            print(f"  {i}. {job.title} @ {job.company}{startup}")
            print(f"     Location: {job.location} | Salary: {job.salary_display} | Visa: {sponsor}")
            print(f"     Source: {job.source} | URL: {job.url[:80]}...")
            print()


async def run_full_pipeline(dry_run: bool = False) -> None:
    """Run the complete 4-agent pipeline."""
    skill_inventory = load_skill_inventory()
    watchlist = load_watchlist()

    final_state = await run_pipeline(
        skill_inventory=skill_inventory,
        watchlist=watchlist,
    )

    print(f"\n{'='*60}")
    print(f"JOBFORGE AI — PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Jobs scraped:     {len(final_state.get('raw_jobs', []))}")
    print(f"Jobs qualified:   {len(final_state.get('qualified_jobs', []))}")
    print(f"CVs generated:    {len(final_state.get('tailored_cvs', []))}")
    print(f"CV errors:        {len(final_state.get('tailor_errors', []))}")
    print(f"Email sent:       {final_state.get('email_sent', False)}")
    print(f"Excel path:       {final_state.get('excel_path', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(description="JobForge AI Pipeline")
    parser.add_argument("--scout-only", action="store_true", help="Run Scout Agent only")
    parser.add_argument("--dry-run", action="store_true", help="No email dispatch")
    args = parser.parse_args()

    setup_logging()

    if args.scout_only:
        asyncio.run(run_scout_only())
    else:
        asyncio.run(run_full_pipeline(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
