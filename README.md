# JobForge AI

**Autonomous multi-agent job hunting pipeline** — discovers, evaluates, and prepares personalised applications for Data Science, AI Engineering, and ML Engineering roles across the UK.

Built with **LangGraph** for stateful agent orchestration, **Google Gemini** for LLM-powered reasoning, and a modular connector architecture.

## What It Does

Every morning, JobForge AI:

1. **Scouts** 6+ job sources (Adzuna, Reed, Wellfound, LinkedIn, Indeed, startup career pages) for fresh AI/ML/DS roles
2. **Scores** each job against your CV using a dual-pass engine (embedding pre-screen + 6-dimension LLM evaluation)
3. **Tailors** a job-specific PDF CV for every qualified job (≥70% match), selecting the optimal CV variant
4. **Dispatches** a curated Excel digest + tailored PDFs to your inbox

**It never auto-applies.** The system stops at generating tailored CVs and an email digest.

## Key Features

- **Deep Agent Architecture**: Scout, Matchmaker, and Tailor agents use hierarchical planning, persistent memory, and self-reflective evaluation loops
- **Visa Intelligence**: Detects sponsorship signals in job descriptions. PSW-aware scoring — all roles are valid, sponsoring roles get a strategic boost
- **Startup Focus**: Dedicated Wellfound/AngelList connector + curated UK AI startup watchlist
- **Zero Hallucination CV Tailoring**: Skill Inventory ground truth ensures the Tailor Agent never invents skills or metrics
- **Cross-Run Deduplication**: SQLite-backed hash store ensures the same job is never re-processed

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

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent Orchestration | LangGraph 0.2.x + LangChain 0.3.x |
| LLM | Google Gemini (Flash + Pro) |
| Job Sources | Adzuna, Reed, Wellfound, Tavily, SerpAPI |
| Persistence | SQLite (local) / PostgreSQL (prod) |
| CV Compilation | pdflatex (TeX Live) |
| Excel | Pandas + openpyxl |
| Email | SMTP / Resend |
| Testing | Pytest + pytest-cov |
| CI/CD | GitHub Actions |

## Architecture

```
[CRON] → Scout Agent → Matchmaker Agent → Tailor Agent → Dispatcher Agent → [EMAIL]
              │                │                │                │
         Fan-out search   Dual-pass score   CV variant select   Excel + PDF
         6+ sources       6 dimensions      LaTeX modify        Email dispatch
         Dedup store      Visa adjustments  Hallucination check
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
