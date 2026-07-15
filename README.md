# JobForge AI

**Autonomous multi-agent job hunting pipeline** — discovers, evaluates, and prepares personalised applications for Data Science, AI Engineering, and ML Engineering roles across the UK.

Built with **LangGraph** for stateful agent orchestration, **Google Gemini** for LLM-powered reasoning, and a modular connector architecture.

## What It Does

Every morning, JobForge AI:

1. **Scouts** 10+ job sources (Adzuna, Reed, Wellfound, LinkedIn/Indeed proxies, Greenhouse/Lever, recruiter boards, funding news, startup career pages, Hacker News "Who is hiring", UK Gov Find a Job¹) for fresh AI/ML/DS roles
2. **Scores** each job against your CV using a dual-pass engine (embedding pre-screen + 6-dimension LLM evaluation)
3. **Tailors** a job-specific PDF CV for every qualified job (≥70% match), selecting the optimal CV variant
4. **Dispatches** a curated Excel digest + tailored PDFs to your inbox

**It never auto-applies.** The system stops at generating tailored CVs and an email digest.

¹ *DWP's "Find a job" service has no public API and its scrapeable legacy domain has been decommissioned; the current connector is a best-effort fallback against a bot-protected replacement and may return no results until an official API exists — kept in the codebase as a placeholder rather than removed outright.*

The same scraped data also powers a second product: a **weekly UK AI/ML/DS market intelligence report** (skill trends, salary percentiles, sponsor intelligence, geographic/company-stage breakdowns) that feeds a LinkedIn content series and [marketforge.digital](https://marketforge.digital), generated straight from the pipeline's own database — see [Market Intelligence](#market-intelligence) below.

## Key Features

- **Deep Agent Architecture**: Scout, Matchmaker, and Tailor agents use hierarchical planning, persistent memory, and self-reflective evaluation loops
- **Visa Intelligence**: Detects sponsorship signals in job descriptions. PSW-aware scoring — all roles are valid, sponsoring roles get a strategic boost
- **UK Sponsor Licence Cross-Check**: Cross-references every employer against the Home Office's public Register of Licensed Sponsors — turns "the JD mentions sponsorship" (an NLP guess) into "this employer legally holds a sponsor licence right now" (verified against the source of truth), reported as a distinct metric
- **Startup Focus**: Dedicated Wellfound/AngelList connector + curated UK AI startup watchlist
- **Zero Hallucination CV Tailoring**: Skill Inventory ground truth ensures the Tailor Agent never invents skills or metrics
- **Fuzzy Cross-Run Deduplication**: Exact-hash gate plus a MinHash LSH near-duplicate layer catches the same role reposted under a reworded title across boards, not just byte-identical repeats
- **Salary-Aware Parsing**: Detects day-rate vs. annual vs. hourly pay and normalises to an annual-equivalent, so contract day-rates never silently corrupt the salary median
- **Statistically Honest Analytics**: Every public figure is traceable to real scraped data — percentile-based stats are suppressed below a minimum sample size, and a divergence guard flags (rather than silently ships) any inconsistency between a weekly snapshot and the 90-day rolling median

## Quick Start

```bash
# Clone
git clone https://github.com/Viraj97-SL/jobforge-ai.git
cd jobforge-ai

# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # Fill in your API keys

# Extract skill inventory (run once)
python scripts/extract_skill_inventory.py

# Test the Scout Agent
python scripts/run_pipeline.py --scout-only

# Run full pipeline
python scripts/run_pipeline.py
```

## Market Intelligence

The pipeline's own scraped data doubles as a UK AI/ML/DS market intelligence product — no separate scraping, just deeper analysis of what's already in the DB:

```bash
# Text summary to stdout
python scripts/market_report.py

# Full schema-validated JSON (deltas, percentiles, segmentation, divergence flags)
python scripts/market_report.py --json

# Archive this week's report so past figures stay reconstructable
python scripts/market_report.py --archive

# Generate the 5-slide LinkedIn carousel PNGs from a report
python scripts/generate_carousel.py path/to/MarketReport.json --outdir outputs/carousel/
```

`MarketReport` (`src/jobforge/models/report.py`) is the single schema-validated source of truth that the LinkedIn carousel, email digest, and marketforge.digital all consume — no surface re-derives its own figures, which is what eliminates the class of bug where two surfaces publish contradictory numbers. Highlights:

- **Week-over-week deltas & skill trend classification** — Accelerating / Cooling / Stable / New, computed via linear regression over a 12-week window, not hand-labelled
- **Role-category & seniority segmentation** — salary and skill demand broken out by AI/LLM Engineer, ML Engineer, Data Scientist, MLOps, etc. (rules-based classifier, ~93% accuracy on a labelled title set)
- **Geographic & company-stage distribution**, **skill co-occurrence clusters**, **posting-persistence (time-to-fill) signal**, **salary-vs-skill premium** (suppressed below n=15 for statistical honesty)
- **Validation gate** (`src/jobforge/analytics/validation.py`) — a data half (divergence guard, minimum-sample-size suppression) and a render half (no overlapping chart text, no empty cards, no NaN/inf chart inputs); a slide or figure that fails is withheld rather than published, and the run is flagged "needs manual review" instead of silently shipping

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent Orchestration | LangGraph + LangChain |
| LLM | Google Gemini (3.5 Flash + 3.1 Pro) |
| Job Sources | Adzuna, Reed, Wellfound, Greenhouse/Lever, recruiter boards, funding news, Hacker News, UK Gov Find a Job, Tavily, SerpAPI |
| Fuzzy Dedup | MinHash LSH (datasketch) |
| Sponsor Matching | rapidfuzz (UK sponsor licence register cross-check) |
| Analytics | pandas, linear-regression trend classification |
| Charts | Matplotlib (LinkedIn carousel generation) |
| Persistence | SQLite (local) / PostgreSQL (prod) |
| CV Compilation | pdflatex (TeX Live) |
| Excel | Pandas + openpyxl |
| Email | SMTP / Resend |
| Testing | Pytest + pytest-cov (70% coverage gate on analytics/ + memory/) |
| CI/CD | GitHub Actions |

## Architecture

```
[CRON] → Scout Agent → Matchmaker Agent → Tailor Agent → Dispatcher Agent → [EMAIL]
              │                │                │                │
         Fan-out search   Dual-pass score   CV variant select   Excel + PDF
         10+ sources      6 dimensions      LaTeX modify        Email dispatch
         Fuzzy dedup      Visa adjustments  Hallucination check
              │
              ▼
     job_analytics DB ──► MarketAnalyzer ──► MarketReport (schema-validated JSON)
                                                    │
                                    ┌───────────────┼───────────────┐
                                    ▼               ▼               ▼
                            Validation gate   Carousel PNGs   Report archive
                          (data + render)    (LinkedIn)      (append-only)
```

## Visa Context

The system is designed for a **Graduate Route (PSW) visa** holder:
- **Full work rights** for 2 years — can accept any role
- **Sponsoring roles** get a +10 point scoring boost (strategic long-term value)
- **"UK citizens only" roles** get a -5 point penalty but are NOT excluded
- Jobs are tagged with visa status in the Excel digest for informed decision-making

## License

MIT

---

Built by **Viraj Bulugahapitiya** | MSc Data Science | AI Engineer
