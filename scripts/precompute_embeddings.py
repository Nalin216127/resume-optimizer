"""Precompute Gemini embeddings for the built-in knowledge base.

    GEMINI_API_KEY=... python scripts/precompute_embeddings.py

Writes resume_optimizer/data/embeddings-<model>-<dim>.jsonl so deployments embed the knowledge base
without any API calls (the free tier allows ~100 embedded texts per minute). Rerun after editing
data/knowledge.jsonl; it takes about a minute per 90 entries.
"""
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resume_optimizer.config import Settings  # noqa: E402
from resume_optimizer.vectorstore import (  # noqa: E402
    SEED_KNOWLEDGE, GeminiEmbedder, _embed_text, _normalise_doc, cache_key, precomputed_path, save_precomputed,
)

settings = replace(Settings(), embedding_backend="gemini")
if not settings.gemini_api_key:
    sys.exit("Set GEMINI_API_KEY")
embedder = GeminiEmbedder(settings)
embedder._precomputed = {}  # recompute everything
embedder.max_wait = 70      # wait out the free tier's per-minute limit
texts = [_embed_text(_normalise_doc(d)) for d in SEED_KNOWLEDGE]
vectors = embedder.embed_many(texts, "RETRIEVAL_DOCUMENT")
path = precomputed_path(embedder.model, embedder.dim)
save_precomputed(path, {cache_key("RETRIEVAL_DOCUMENT", t): v for t, v in zip(texts, vectors)})
print(f"wrote {len(texts)} vectors to {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")
