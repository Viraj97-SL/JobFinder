# ╔══════════════════════════════════════════════════════════════╗
# ║  JobForge AI — Production Dockerfile                        ║
# ║  texlive for pdflatex CV compilation                        ║
# ╚══════════════════════════════════════════════════════════════╝

FROM python:3.11-slim AS base

# System dependencies (texlive for pdflatex + fontawesome5)
RUN apt-get update && apt-get install -y --no-install-recommends \
    texlive-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-xetex \
    latexmk \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (production only)
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

# Copy application code and data
COPY . .

# Create output directories
RUN mkdir -p data outputs/cvs

# Extract skill inventory from master CV templates
RUN python scripts/extract_skill_inventory.py || echo "[WARN] Skill inventory extraction skipped — run manually or ensure .tex files are present"

# Default: run the full pipeline (Railway overrides this with cron)
CMD ["python", "scripts/run_pipeline.py"]
