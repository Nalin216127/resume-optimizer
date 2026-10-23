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
from typing import Dict, List, Optional, Protocol

from .config import Settings

log = logging.getLogger(__name__)


class HashingEmbedder:
    """Dependency-free embedding via the hashing trick over unigrams + bigrams.

    Deterministic and fast; swap in a neural embedder by implementing ``embed``.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        tokens = re.findall(r"[a-z0-9+#]+", text.lower())
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        vec = [0.0] * self.dim
        for feat in features:
            h = int(hashlib.blake2b(feat.encode(), digest_size=8).hexdigest(), 16)
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class KnowledgeStore(Protocol):
    def add(self, docs: List[Dict[str, str]]) -> int: ...
    def search(self, query: str, k: int, kind: Optional[str] = None) -> List[Dict[str, object]]: ...
    def count(self) -> int: ...
    def healthy(self) -> bool: ...


class InMemoryStore:
    def __init__(self, embedder: HashingEmbedder):
        self.embedder = embedder
        self.rows: List[Dict[str, object]] = []

    def add(self, docs):
        for d in docs:
            self.rows.append({**d, "vector": self.embedder.embed(f"{d['title']} {d['text']}")})
        return len(docs)

    def search(self, query, k, kind=None):
        q = self.embedder.embed(query)
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
    def __init__(self, settings: Settings, embedder: HashingEmbedder):
        from pymilvus import MilvusClient

        self.embedder = embedder
        self.collection = settings.milvus_collection
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
        rows = [
            {"vector": self.embedder.embed(f"{d['title']} {d['text']}"), "title": d["title"], "text": d["text"],
             "kind": d.get("kind", "guideline")}
            for d in docs
        ]
        result = self.client.insert(collection_name=self.collection, data=rows)
        return int(result.get("insert_count", len(rows)))

    def search(self, query, k, kind=None):
        hits = self.client.search(
            collection_name=self.collection,
            data=[self.embedder.embed(query)],
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
    embedder = HashingEmbedder(settings.embedding_dim)
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
