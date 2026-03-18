"""
JobForge AI — RAG Store for Tailor Agent few-shot context.

Persists tailoring decisions to a ChromaDB vector store. Before each
LLM CV-rewrite call, the Tailor Agent retrieves the top-K most similar
past tailoring examples and injects them as few-shot context into the prompt.

Over time this builds a memory of "what worked" for similar roles, making
the Tailor progressively more consistent and role-aware.

Collection schema:
  id:        job_id
  document:  "{title} at {company}: {description_snippet}"  (used for embedding)
  metadata:
    company, role, cv_variant, key_skills (pipe-delimited), scraped_at,
    sections_modified (pipe-delimited), hallucination_passed
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from functools import cached_property
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

COLLECTION_NAME = "tailoring_history"


class RAGStore:
    """
    ChromaDB-backed vector store for Tailor Agent few-shot retrieval.

    Embedding model: all-MiniLM-L6-v2 (same as MLPrescreen — no extra download).
    Persistence: DATA_DIR/chroma_db (local) or a mounted volume (Railway).
    """

    def __init__(self, persist_dir: str | None = None) -> None:
        from jobforge.config.settings import settings
        self._persist_dir = persist_dir or settings.chroma_db_dir

    @cached_property
    def _client(self) -> Any:
        import chromadb
        client = chromadb.PersistentClient(path=self._persist_dir)
        logger.info("rag_store.client.ready", persist_dir=self._persist_dir)
        return client

    @cached_property
    def _ef(self) -> Any:
        """SentenceTransformer embedding function (wraps all-MiniLM-L6-v2)."""
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        return SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    @cached_property
    def _collection(self) -> Any:
        return self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Write ─────────────────────────────────────────────────────────────────

    def store_tailoring(
        self,
        job: Any,                  # ScoredJob
        cv_variant: str,
        sections_modified: list[str],
        hallucination_passed: bool,
    ) -> None:
        """
        Persist a successful tailoring decision to the vector store.
        Called by TailorAgent after each successful _tailor_single().
        """
        try:
            doc_text = self._build_document(job)
            metadata = {
                "company": job.job.company,
                "role": job.job.title,
                "cv_variant": cv_variant,
                "key_skills": "|".join(job.key_matching_skills[:6]),
                "transferable": "|".join(job.transferable_highlights[:4]),
                "sections_modified": "|".join(sections_modified),
                "hallucination_passed": str(hallucination_passed),
                "stored_at": datetime.utcnow().isoformat(),
            }

            self._collection.upsert(
                ids=[job.job.job_id],
                documents=[doc_text],
                metadatas=[metadata],
            )
            logger.debug("rag_store.stored", job_id=job.job.job_id, variant=cv_variant)
        except Exception as e:
            # Never crash the pipeline over RAG failures
            logger.warning("rag_store.store_failed", job_id=job.job.job_id, error=str(e))

    # ── Read ──────────────────────────────────────────────────────────────────

    def find_similar(self, job: Any, top_k: int = 3) -> list[dict]:
        """
        Retrieve top-K most similar past tailoring decisions for a given job.

        Returns a list of dicts with keys: role, company, cv_variant,
        key_skills, sections_modified, transferable.
        Returns [] if fewer than 2 documents are stored (no meaningful context yet).
        """
        try:
            count = self._collection.count()
            if count < 2:
                return []

            query_text = self._build_document(job)
            results = self._collection.query(
                query_texts=[query_text],
                n_results=min(top_k, count),
                include=["metadatas", "distances"],
            )

            examples = []
            for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
                # cosine distance → similarity; skip very dissimilar (>0.8 distance)
                if dist > 0.8:
                    continue
                examples.append({
                    "role": meta.get("role", ""),
                    "company": meta.get("company", ""),
                    "cv_variant": meta.get("cv_variant", ""),
                    "key_skills": meta.get("key_skills", "").split("|"),
                    "transferable": meta.get("transferable", "").split("|"),
                    "sections_modified": meta.get("sections_modified", "").split("|"),
                    "similarity": round(1 - dist, 3),
                })

            return examples
        except Exception as e:
            logger.warning("rag_store.query_failed", error=str(e))
            return []

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_document(job: Any) -> str:
        """Build the text that gets embedded — title + company + skills + description snippet."""
        skills_str = ", ".join(job.key_matching_skills[:6]) if job.key_matching_skills else ""
        desc_snippet = (job.job.description or "")[:800]
        return f"{job.job.title} at {job.job.company}. Skills: {skills_str}. {desc_snippet}"

    @staticmethod
    def format_examples_for_prompt(examples: list[dict]) -> str:
        """
        Format retrieved RAG examples as a few-shot context block for the LLM prompt.
        Returns an empty string if no examples.
        """
        if not examples:
            return ""

        lines = [
            "\nFEW-SHOT CONTEXT — Similar roles you have tailored CVs for previously:",
            "(Use these only as style/emphasis guidance. Do NOT copy their content.)",
        ]
        for i, ex in enumerate(examples, 1):
            skills = ", ".join(ex["key_skills"][:5]) if ex["key_skills"] else "N/A"
            sections = ", ".join(ex["sections_modified"]) if ex["sections_modified"] else "N/A"
            lines.append(
                f"\n  Example {i} (similarity={ex['similarity']}):\n"
                f"    Role:      {ex['role']} at {ex['company']}\n"
                f"    CV Variant: {ex['cv_variant']}\n"
                f"    Key Skills Emphasised: {skills}\n"
                f"    Sections Modified: {sections}"
            )

        return "\n".join(lines)
