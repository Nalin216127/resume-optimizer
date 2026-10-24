"""Knowledge retrieval for the Context agent, backed by Milvus.

The store holds resume-writing guidelines, exemplar bullets and role profiles.
The Context agent retrieves the most relevant entries for each job description
and hands them to the Writer as grounded guidance.
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Dict, List, Optional, Protocol, Union

from .config import Settings

log = logging.getLogger(__name__)


class HashingEmbedder:
    """Dependency-free embedding via the hashing trick over unigrams + bigrams.

    Deterministic and fast, but lexical: it matches shared words, not meaning. Used offline and as
    the fallback when the hosted embedding API is unavailable.
    """

    # Cosine similarity at or above which a draft claim counts as traced to the original resume.
    # Calibrated on rewritten-but-true bullets (lowest 0.55) vs invented ones (highest 0.18).
    support_threshold = 0.35

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.tag = f"hash{dim}"
        self.label = "hashing (lexical)"

    def embed(self, text: str, task: str = "RETRIEVAL_QUERY") -> List[float]:
        tokens = re.findall(r"[a-z0-9+#]+", text.lower())
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        vec = [0.0] * self.dim
        for feat in features:
            h = int(hashlib.blake2b(feat.encode(), digest_size=8).hexdigest(), 16)
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_many(self, texts: List[str], task: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
        return [self.embed(t, task) for t in texts]


class GeminiEmbedder:
    """Semantic embeddings from Gemini's hosted embedding API (free tier; nothing runs locally).

    Batches requests, caches repeated texts (the same resume is embedded on every review) and
    L2-normalises vectors, which Google recommends for reduced output dimensions.
    """

    # Calibrated on gemini-embedding-001 at 768 dims: true claims scored >= 0.94, invented ones <= 0.85.
    support_threshold = 0.90
    BATCH = 100
    CACHE_LIMIT = 4096

    def __init__(self, settings: Settings):
        from google import genai
        from google.genai import types

        self._types = types
        self.client = genai.Client(api_key=settings.gemini_api_key, http_options=types.HttpOptions(timeout=30_000))
        self.model = settings.gemini_embedding_model
        self.dim = settings.gemini_embedding_dim
        self.tag = f"gemini{self.dim}"
        self.label = f"Gemini {self.model}"
        self._cache: Dict[tuple, List[float]] = {}

    def embed(self, text: str, task: str = "RETRIEVAL_QUERY") -> List[float]:
        return self.embed_many([text], task)[0]

    def embed_many(self, texts: List[str], task: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
        missing = list(dict.fromkeys(t for t in texts if (task, t) not in self._cache))
        for i in range(0, len(missing), self.BATCH):
            chunk = missing[i:i + self.BATCH]
            result = self.client.models.embed_content(
                model=self.model, contents=chunk,
                config=self._types.EmbedContentConfig(task_type=task, output_dimensionality=self.dim),
            )
            for text, emb in zip(chunk, result.embeddings):
                values = list(emb.values)
                norm = math.sqrt(sum(v * v for v in values)) or 1.0
                self._cache[(task, text)] = [v / norm for v in values]
        if len(self._cache) > self.CACHE_LIMIT:
            self._cache = dict(list(self._cache.items())[-self.CACHE_LIMIT // 2:])
        return [self._cache[(task, t)] for t in texts]


Embedder = Union[HashingEmbedder, GeminiEmbedder]


def build_embedder(settings: Settings) -> Embedder:
    backend = settings.embedding_backend
    if backend == "auto":
        backend = "gemini" if settings.gemini_api_key else "hash"
    if backend == "gemini":
        try:
            embedder = GeminiEmbedder(settings)
            embedder.embed("startup check")  # fail fast here rather than mid-request
            log.info("Using %s embeddings (dim=%d)", embedder.model, embedder.dim)
            return embedder
        except Exception as exc:  # noqa: BLE001 - any failure means fall back
            log.warning("Gemini embeddings unavailable (%s); falling back to the hashing embedder", exc)
    elif backend != "hash":
        raise ValueError(f"Unknown EMBEDDING_BACKEND: {settings.embedding_backend}")
    return HashingEmbedder(settings.embedding_dim)


class KnowledgeStore(Protocol):
    embedder: Embedder

    def add(self, docs: List[Dict[str, str]]) -> int: ...
    def search(self, query: str, k: int, kind: Optional[str] = None) -> List[Dict[str, object]]: ...
    def count(self) -> int: ...
    def healthy(self) -> bool: ...


class InMemoryStore:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.rows: List[Dict[str, object]] = []

    def add(self, docs):
        vectors = self.embedder.embed_many([f"{d['title']} {d['text']}" for d in docs], "RETRIEVAL_DOCUMENT")
        self.rows.extend({**d, "vector": v} for d, v in zip(docs, vectors))
        return len(docs)

    def search(self, query, k, kind=None):
        q = self.embedder.embed(query, "RETRIEVAL_QUERY")
        scored = [
            ({key: r[key] for key in ("title", "text", "kind")}, sum(a * b for a, b in zip(q, r["vector"])))
            for r in self.rows
            if kind is None or r["kind"] == kind
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [{**doc, "score": round(score, 4)} for doc, score in scored[:k]]

    def count(self):
        return len(self.rows)

    def healthy(self):
        return True


class MilvusStore:
    def __init__(self, settings: Settings, embedder: Embedder):
        from pymilvus import MilvusClient

        self.embedder = embedder
        # One collection per embedder (e.g. resume_knowledge_gemini768): vectors from different
        # embedders or dimensions must never share a collection.
        self.collection = f"{settings.milvus_collection}_{embedder.tag}"
        self.client = MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token or "")
        if not self.client.has_collection(self.collection):
            # Quick-setup schema: auto int64 id + float vector; title/text/kind live in dynamic fields.
            self.client.create_collection(
                collection_name=self.collection,
                dimension=embedder.dim,
                metric_type="COSINE",
                auto_id=True,
                enable_dynamic_field=True,
            )
            log.info("Created Milvus collection %s (dim=%d)", self.collection, embedder.dim)

    def add(self, docs):
        vectors = self.embedder.embed_many([f"{d['title']} {d['text']}" for d in docs], "RETRIEVAL_DOCUMENT")
        rows = [
            {"vector": v, "title": d["title"], "text": d["text"], "kind": d.get("kind", "guideline")}
            for d, v in zip(docs, vectors)
        ]
        result = self.client.insert(collection_name=self.collection, data=rows)
        return int(result.get("insert_count", len(rows)))

    def search(self, query, k, kind=None):
        hits = self.client.search(
            collection_name=self.collection,
            data=[self.embedder.embed(query, "RETRIEVAL_QUERY")],
            limit=k,
            filter=f'kind == "{kind}"' if kind else "",
            output_fields=["title", "text", "kind"],
        )
        return [{**hit["entity"], "score": round(float(hit["distance"]), 4)} for hit in hits[0]]

    def count(self):
        # count(*) with Strong consistency sees unflushed inserts; collection stats can lag behind.
        rows = self.client.query(self.collection, filter="", output_fields=["count(*)"], consistency_level="Strong")
        return int(rows[0]["count(*)"]) if rows else 0

    def healthy(self):
        try:
            self.client.list_collections()
            return True
        except Exception:  # noqa: BLE001 - any failure means not ready
            return False


SEED_KNOWLEDGE: List[Dict[str, str]] = [
    {"kind": "guideline", "title": "XYZ bullet formula",
     "text": "Write bullets as: accomplished X, measured by Y, by doing Z. Lead with the outcome, then the method."},
    {"kind": "guideline", "title": "Quantify impact",
     "text": "Quantify impact with percentages, revenue, latency, cost savings, users served or time saved. Never invent numbers; use scale words if no metric exists."},
    {"kind": "guideline", "title": "Strong action verbs",
     "text": "Open each bullet with a strong past-tense action verb such as Architected, Led, Reduced, Shipped, Automated. Avoid 'responsible for' and 'worked on'."},
    {"kind": "guideline", "title": "ATS keyword alignment",
     "text": "Mirror exact keywords and tool names from the job description where the candidate genuinely has the experience. ATS parsers match literal terms."},
    {"kind": "guideline", "title": "ATS-safe formatting",
     "text": "Use standard headings (Summary, Experience, Projects, Education, Skills), single column, no tables or images, consistent date formats."},
    {"kind": "guideline", "title": "Concise bullets",
     "text": "Keep bullets to one or two lines, roughly 12 to 28 words. Three to six bullets per role, most recent role gets the most."},
    {"kind": "guideline", "title": "Tailored summary",
     "text": "The summary should name the target role, years of experience, two or three signature strengths and one headline achievement."},
    {"kind": "guideline", "title": "Truthfulness",
     "text": "Never fabricate employers, titles, dates, metrics or skills. Rephrase and reorder real experience; flag gaps instead of hiding them."},
    {"kind": "example_bullet", "title": "Backend engineering",
     "text": "Reduced p99 API latency 45% by redesigning the caching layer with Redis and async I/O in FastAPI, serving 2M daily requests."},
    {"kind": "example_bullet", "title": "ML / LLM engineering",
     "text": "Architected a retrieval-augmented generation pipeline over Milvus, improving answer accuracy 22% on internal evals while cutting token cost 30%."},
    {"kind": "example_bullet", "title": "Platform / DevOps",
     "text": "Migrated 40 services to Kubernetes with Helm and GitHub Actions CI/CD, cutting deploy time from 45 to 8 minutes."},
    {"kind": "example_bullet", "title": "Data engineering",
     "text": "Built Airflow and Spark data pipelines processing 3TB/day, raising data freshness SLA from 24h to 1h."},
    {"kind": "example_bullet", "title": "Leadership",
     "text": "Led a cross-functional team of 6 engineers to ship a self-serve onboarding flow, lifting activation 18% in one quarter."},
    {"kind": "role_profile", "title": "AI / ML Engineer",
     "text": "Hiring managers look for LLM application development, LangChain or LangGraph agents, vector databases, evaluation, Python, model deployment and MLOps."},
    {"kind": "role_profile", "title": "Backend Engineer",
     "text": "Hiring managers look for API design, distributed systems, SQL and NoSQL databases, caching, observability, testing and cloud deployment."},
    {"kind": "role_profile", "title": "Data Scientist",
     "text": "Hiring managers look for statistics, experimentation and A/B testing, SQL, Python, machine learning, and communicating insights to stakeholders."},
]


def build_store(settings: Settings) -> KnowledgeStore:
    embedder = build_embedder(settings)
    store: KnowledgeStore
    if settings.vector_backend == "milvus":
        store = MilvusStore(settings, embedder)
    elif settings.vector_backend == "memory":
        store = InMemoryStore(embedder)
    else:
        raise ValueError(f"Unknown VECTOR_BACKEND: {settings.vector_backend}")
    if settings.seed_on_startup:
        seed(store)
    return store


def seed(store: KnowledgeStore) -> int:
    if store.count() > 0:
        return 0
    inserted = store.add(SEED_KNOWLEDGE)
    log.info("Seeded knowledge store with %d documents", inserted)
    return inserted
