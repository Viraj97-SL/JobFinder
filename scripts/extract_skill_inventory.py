"""
JobForge AI — Skill Inventory Extractor.

Extracts a structured SkillInventory JSON from the master CV PDFs.
This is the Tailor Agent's immutable ground truth.

Usage:
    python scripts/extract_skill_inventory.py

Output: data/skill_inventory.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jobforge.config.settings import DATA_DIR
from jobforge.models.cv import EduEntry, ProjectEntry, SkillInventory, WorkEntry


def build_inventory() -> SkillInventory:
    """
    Build the SkillInventory from known CV data.

    This is manually curated from the three master CVs to ensure
    100% accuracy. The Tailor Agent uses this as its ONLY allowed
    source of truth — no hallucination possible.
    """

    return SkillInventory(
        technical_skills={
            "agent_orchestration": [
                "LangGraph", "LangChain", "Multi-Agent Patterns",
                "Supervisor-Worker", "Router", "Hierarchical Delegation",
                "Fan-Out/Fan-In", "Human-in-the-Loop", "Custom MCP",
                "Prompt Engineering", "Tool Calling", "Command Routing",
                "State Machines", "Checkpointing", "Interrupt",
            ],
            "llm_ecosystem": [
                "Google Gemini", "Gemini Flash", "Gemini Pro",
                "Groq", "Llama-3.3-70B", "Llama-3.1-8B",
                "OpenAI API", "Anthropic Claude", "Hugging Face Transformers",
            ],
            "rag_and_search": [
                "RAG", "ChromaDB", "FAISS", "Vector Databases",
                "Tavily API", "Embedding Search",
            ],
            "observability": [
                "LangSmith Tracing", "structlog", "Weights & Biases",
                "State Inspection", "Retry Policies",
            ],
            "ml_deep_learning": [
                "PyTorch", "TensorFlow", "Keras",
                "3D Swin Transformer", "LSTMs", "XGBoost",
                "Scikit-learn", "YOLOv8", "OpenCV",
                "Computer Vision", "Contrastive Learning",
                "Self-Supervised Learning",
            ],
            "backend_engineering": [
                "Python", "FastAPI", "Pydantic V2",
                "Async/Await", "PostgreSQL", "SQLAlchemy 2.0",
                "SQLite", "REST APIs", "Microservices",
            ],
            "devops_tools": [
                "Docker", "GitHub Actions", "CI/CD",
                "Pytest", "Ruff", "Mypy", "Git",
                "Alembic", "Railway",
            ],
            "data_analysis": [
                "Pandas", "NumPy", "Power BI",
                "Matplotlib", "Seaborn", "SQL",
                "Data Visualization", "Statistics",
            ],
            "cloud": [
                "Azure", "GCP", "Railway", "Vercel",
            ],
            "frontend": [
                "React", "Next.js", "JavaScript",
                "Tailwind CSS", "HTML/CSS",
            ],
        },
        projects=[
            ProjectEntry(
                name="AI News Summarizer",
                short_description="Autonomous multi-agent pipeline with fan-out/fan-in scraping, tiered Gemini routing, and HITL approval",
                technologies=["LangGraph", "Gemini", "FastAPI", "PostgreSQL", "LangSmith", "Docker", "Railway"],
                quantified_metrics=["$5/run cost cap", "4 parallel sources", "Tue/Thu schedule"],
                domain="AI/NLP",
                is_deployed=True,
                github_url="",
            ),
            ProjectEntry(
                name="Pamorya AI Commerce Platform",
                short_description="Multi-agent AI commerce system with Supervisor-Worker architecture, SQL querying, sales flow, and RAG",
                technologies=["LangGraph", "Gemini 2.5", "FastAPI", "Docker", "PostgreSQL", "ChromaDB"],
                quantified_metrics=[],
                domain="Retail/E-commerce",
                is_deployed=True,
            ),
            ProjectEntry(
                name="SATH-CHAKRA AI",
                short_description="Autonomous agentic framework with non-linear state machines, dual-model routing, Playwright rendering",
                technologies=["LangGraph", "Groq Llama-3", "Playwright", "FastAPI", "Docker", "React"],
                quantified_metrics=[],
                domain="Generative AI",
                is_deployed=True,
            ),
            ProjectEntry(
                name="VisionAID",
                short_description="Multi-agent assistive AI for visually impaired with custom MCP coordination",
                technologies=["YOLOv8", "OpenCV", "Whisper", "LangChain", "SQLite", "Tesseract", "Hugging Face"],
                quantified_metrics=["Winner, University of Hertfordshire Agentic AI Competition"],
                domain="Assistive Technology / Computer Vision",
                is_deployed=False,
            ),
            ProjectEntry(
                name="Alzheimer's Detection Tri-Modal Framework",
                short_description="3D Swin Transformer + dual LSTMs with gated fusion for Alzheimer's diagnosis from MRI, clinical, and biomarker data",
                technologies=["PyTorch", "3D Swin Transformer", "LSTMs", "ANTsPy", "NiBabel", "GCP", "W&B"],
                quantified_metrics=["0.8966 Accuracy", "0.9611 AUC-ROC", "0.8337 MCC", "0.9129 G-Mean", "N=187 ADNI"],
                domain="Healthcare / Medical Imaging",
                is_deployed=False,
                github_url="https://github.com/Viraj97-SL/Research-Early-prediction-of-Alzheimer-s",
            ),
            ProjectEntry(
                name="CricOracle 2026",
                short_description="T20 World Cup prediction platform with XGBoost ensemble, LSTM, and genetic algorithm squad optimiser",
                technologies=["XGBoost", "PyTorch", "DEAP", "SHAP", "FastAPI", "Pandas", "scikit-learn"],
                quantified_metrics=["0.887 AUC-ROC", "83.1% Accuracy", "23.5 MAE runs", "2,519 matches", "565K deliveries"],
                domain="Sports Analytics",
                is_deployed=True,
                github_url="https://github.com/Viraj97-SL/CricOracle2026",
            ),
            ProjectEntry(
                name="RepoSentinel",
                short_description="Autonomous multi-agent system for GitHub repo maintenance with gap analysis, content scouting, and auto-PR",
                technologies=["LangGraph", "LangChain", "Gemini", "FastAPI", "PostgreSQL", "GitHub API", "Docker"],
                quantified_metrics=["5-check quality gate", "Score threshold 75/100"],
                domain="Developer Tools / Automation",
                is_deployed=True,
            ),
            ProjectEntry(
                name="GenAI Fashion Stylist",
                short_description="Multi-agent RAG system with virtual try-on, zero hallucination product recommendations",
                technologies=["LangGraph", "LangChain", "ChromaDB", "FastAPI", "Replicate API", "Docker", "Next.js"],
                quantified_metrics=["Zero hallucination product recommendations"],
                domain="Retail / Fashion Tech",
                is_deployed=True,
            ),
            ProjectEntry(
                name="AI/ML/DS Learning Hub",
                short_description="Comprehensive open-source learning resource repository for Data Scientists, ML Engineers, and AI Engineers",
                technologies=["Markdown", "GitHub"],
                quantified_metrics=["3 career tracks", "100+ resources", "50+ projects"],
                domain="Education",
                is_deployed=True,
                github_url="https://github.com/viraj97-sl/ai-ml-ds-learning-hub",
            ),
        ],
        work_experience=[
            WorkEntry(
                role="AI Engineer & Researcher",
                company="Independent",
                location="London, UK",
                date_range="Sep 2025 – Present",
                achievements=[
                    "Designed and deployed five production LLM agent systems end-to-end",
                    "Defined technical standards for code quality (Pytest, Ruff, CI/CD)",
                    "Active contributor to LangGraph and LangChain ecosystem",
                    "Mentoring peers on multi-agent system design patterns",
                ],
                quantified_metrics=["5 production agent systems"],
            ),
            WorkEntry(
                role="Executive – Planning (Operations & Data)",
                company="MAS Holdings",
                location="Sri Lanka",
                date_range="May 2022 – Sep 2024",
                achievements=[
                    "Cross-functional leadership across manufacturing, merchandising, and logistics",
                    "Built Power BI dashboards for real-time KPI tracking",
                    "Translated complex operational data into actionable stakeholder insights",
                    "Managed production planning for Victoria's Secret and Marks & Spencer",
                ],
                quantified_metrics=[
                    "97% capacity utilisation",
                    "95% on-time delivery",
                    "Reduction in WIP inventory",
                ],
            ),
            WorkEntry(
                role="Customer Assistant",
                company="Marks and Spencer",
                location="London, UK",
                date_range="Feb 2025 – Present",
                achievements=[
                    "Operational agility in fast-paced retail",
                    "Inventory management and digital PoS systems",
                ],
                quantified_metrics=[],
            ),
        ],
        education=[
            EduEntry(
                degree="MSc in Data Science",
                institution="University of Hertfordshire, UK",
                date="Sep 2024 – Sep 2025",
                key_modules=["Deep Learning", "Data Mining", "Neural Networks", "Cloud Computing (GCP)", "PySpark"],
            ),
            EduEntry(
                degree="BSc in Logistics Management",
                institution="Dalian Maritime University, China",
                date="Sep 2017 – Oct 2021",
                key_modules=["Supply Chain Optimisation", "Operations Research"],
            ),
        ],
        certifications=[
            "Microsoft Certified: Azure AI Fundamentals (AI-900)",
            "Mathematics for Data Science — 365 Data Science (Nov 2025)",
            "Mentor, Teens in AI (Feb 2026)",
        ],
        quantified_achievements=[
            "97% capacity utilisation",
            "95% on-time delivery",
            "0.8966 Accuracy (Alzheimer's)",
            "0.9611 AUC-ROC (Alzheimer's)",
            "0.8337 MCC (Alzheimer's)",
            "0.9129 G-Mean (Alzheimer's)",
            "N=187 ADNI dataset",
            "0.887 AUC-ROC (CricOracle)",
            "83.1% Accuracy (CricOracle)",
            "23.5 MAE runs (CricOracle)",
            "2,519 matches processed",
            "565,377 ball-by-ball deliveries",
            "$5/run cost cap (AI News)",
            "5 production agent systems",
            "Winner, UH Agentic AI Competition",
            "Zero hallucination product recommendations",
            "Score threshold 75/100 (RepoSentinel)",
        ],
        volunteering=[
            "Mentor, Teens in AI (Feb 2026)",
            "Volunteer, DataKind UK (Aug 2025 – Present)",
            "Volunteer, The Felix Project, UK (2024 – Present)",
            "Director of Finance, Rotaract Club (2022 – 2023)",
        ],
    )


def main():
    inventory = build_inventory()
    output_path = DATA_DIR / "skill_inventory.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(inventory.model_dump(), f, indent=2)

    print(f"Skill Inventory extracted -> {output_path}")
    print(f"  Skills categories:  {len(inventory.technical_skills)}")
    print(f"  Total skills:       {len(inventory.get_all_skills_flat())}")
    print(f"  Projects:           {len(inventory.projects)}")
    print(f"  Work entries:       {len(inventory.work_experience)}")
    print(f"  Certifications:     {len(inventory.certifications)}")
    print(f"  Quantified metrics: {len(inventory.quantified_achievements)}")


if __name__ == "__main__":
    main()
