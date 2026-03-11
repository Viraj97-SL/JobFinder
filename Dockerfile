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

# Copy everything first (editable installs need source present)
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir .

# Create output directories
RUN mkdir -p data outputs/cvs

# Extract skill inventory from master CV templates
RUN python scripts/extract_skill_inventory.py || echo "[WARN] Skill inventory extraction skipped"

# Pre-download SBERT model into image cache (avoids cold-start download on Railway)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Default: run the full pipeline (Railway overrides this with cron)
CMD ["python", "scripts/run_pipeline.py"]