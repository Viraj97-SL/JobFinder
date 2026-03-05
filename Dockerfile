# ╔══════════════════════════════════════════════════════════════╗
# ║  JobForge AI — Production Dockerfile                        ║
# ║  Multi-stage build with TeX Live for CV compilation         ║
# ╚══════════════════════════════════════════════════════════════╝

FROM python:3.11-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    texlive-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    latexmk \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]"

# Copy application code
COPY . .

# Create output directories
RUN mkdir -p data outputs

# Extract skill inventory on build
RUN python scripts/extract_skill_inventory.py

# Default: run the full pipeline
CMD ["python", "scripts/run_pipeline.py"]
