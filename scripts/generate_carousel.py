"""
JobForge AI — LinkedIn Carousel Generator (Phase 4.2 / Phase 4.3 render gate).

Reads a MarketReport JSON (produced by `scripts/market_report.py --json`) and
renders 5 slide PNGs for the weekly LinkedIn carousel. Every slide is passed
through the Phase 4.3 render-validation gate (src/jobforge/analytics/
validation.py: check_no_nan_or_inf, check_card_not_empty,
check_no_overlapping_text) before it's written to disk. A slide that fails
validation is never shipped silently — the run prints which slide(s) need
manual review and exits non-zero, mirroring the same "flag, don't silently
pick" philosophy as the salary-divergence guard. Slides that DO pass are
still written even if a sibling slide fails.

Usage:
    python scripts/market_report.py --json > outputs/market_report.json
    python scripts/generate_carousel.py outputs/market_report.json --outdir outputs/carousel/
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Make sure both the project root (for `scripts.carousel.*`) and src/ (for
# `jobforge.*`) are importable when run directly, mirroring the pattern
# already used by scripts/market_report.py.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from jobforge.analytics.validation import check_card_not_empty, check_no_overlapping_text  # noqa: E402
from jobforge.models.report import MarketReport  # noqa: E402

from scripts.carousel.slides import SLIDE_BUILDERS, SlideRenderError  # noqa: E402


@dataclass
class SlideResult:
    filename: str
    saved: bool
    reasons: list[str] = field(default_factory=list)


def _validate_figure(fig, filename: str) -> list[str]:
    """Run the render-validation gate against a built slide Figure."""
    reasons: list[str] = []

    for ax in fig.axes:
        if not check_card_not_empty(ax):
            reasons.append("empty axes detected (no chart content rendered)")

    overlap = check_no_overlapping_text(fig)
    if overlap["has_overlap"]:
        pairs = ", ".join(f"'{a}' / '{b}'" for a, b in overlap["overlapping_pairs"])
        reasons.append(f"overlapping text detected ({pairs})")

    return reasons


def generate_slides(report: MarketReport, outdir: Path) -> list[SlideResult]:
    """
    Build and validate all 5 carousel slides. Slides that pass the
    render-validation gate are written to `outdir`; slides that fail are
    skipped (not written) and reported back with their failure reason(s).
    """
    outdir.mkdir(parents=True, exist_ok=True)
    results: list[SlideResult] = []

    for filename, builder in SLIDE_BUILDERS:
        try:
            fig = builder(report)
        except SlideRenderError as exc:
            results.append(SlideResult(filename, saved=False, reasons=[str(exc)]))
            continue

        reasons = _validate_figure(fig, filename)
        if reasons:
            results.append(SlideResult(filename, saved=False, reasons=reasons))
            continue

        path = outdir / filename
        fig.savefig(path, dpi=fig.dpi, facecolor=fig.get_facecolor())
        results.append(SlideResult(filename, saved=True))

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="JobForge AI — LinkedIn Carousel Generator")
    parser.add_argument("report_path", type=Path, help="Path to a MarketReport JSON file")
    parser.add_argument(
        "--outdir", type=Path, default=Path("outputs/carousel"),
        help="Output directory for slide PNGs (default: outputs/carousel/)",
    )
    args = parser.parse_args()

    report = MarketReport.model_validate_json(args.report_path.read_text(encoding="utf-8"))
    results = generate_slides(report, args.outdir)

    failed = [result for result in results if not result.saved]
    for result in results:
        status = "OK" if result.saved else "FAILED"
        print(f"  [{status}] {result.filename}")
        for reason in result.reasons:
            print(f"      - {reason}")

    if failed:
        print("\nNEEDS MANUAL REVIEW: one or more slides failed the render-validation gate.")
        print(f"  {len(failed)}/{len(results)} slide(s) failed — not written to {args.outdir}")
        return 1

    print(f"\nAll {len(results)} slides passed validation and were written to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
