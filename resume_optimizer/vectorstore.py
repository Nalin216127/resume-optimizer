"""Knowledge retrieval for the Context agent, backed by Milvus.

The store holds resume-writing guidelines, exemplar bullets and role profiles.
The Context agent retrieves the most relevant entries for each job description
and hands them to the Writer as grounded guidance.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import re
import time
import uuid
from array import array
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

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
    # Lexical vectors rank worse than BM25 (eval hit@1 0.57 vs 0.82), so BM25 leads hybrid search.
    preferred_method = "keyword"

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

    The free tier allows about 100 embedded texts per minute, and every text in a batch counts.
    So the knowledge base ships with precomputed vectors (data/embeddings-<model>-<dim>.jsonl) and
    start-up makes no embedding calls; anything not precomputed is embedded on demand in small
    batches, waiting out per-minute limits. Repeated texts are cached (the same resume is embedded
    on every review) and vectors are L2-normalised, as Google recommends for reduced dimensions.
    """

    # Calibrated on gemini-embedding-001 at 768 dims: true claims scored >= 0.94, invented ones <= 0.85.
    support_threshold = 0.90
    preferred_method = "semantic"   # eval hit@1 1.00 vs 0.82 for BM25
    BATCH = 90
    CACHE_LIMIT = 4096

    def __init__(self, settings: Settings):
        from google import genai
        from google.genai import errors, types

        self._types, self._errors = types, errors
        self.client = genai.Client(api_key=settings.gemini_api_key, http_options=types.HttpOptions(timeout=30_000))
        self.model = settings.gemini_embedding_model
        self.dim = settings.gemini_embedding_dim
        self.tag = f"gemini{self.dim}"
        self.label = f"Gemini {self.model}"
        self._cache: Dict[tuple, List[float]] = {}
        self._precomputed = load_precomputed(precomputed_path(self.model, self.dim))
        # Seconds to wait out a per-minute limit. 0 during user requests, so callers fall back at once
        # (retrieval to keyword search, the evidence check to lexical vectors); offline scripts raise it.
        self.max_wait = 0

    def embed(self, text: str, task: str = "RETRIEVAL_QUERY") -> List[float]:
        return self.embed_many([text], task)[0]

    def embed_many(self, texts: List[str], task: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
        for t in texts:
            if (task, t) not in self._cache:
                vec = self._precomputed.get(cache_key(task, t))
                if vec is not None:
                    self._cache[(task, t)] = vec
        missing = list(dict.fromkeys(t for t in texts if (task, t) not in self._cache))
        for i in range(0, len(missing), self.BATCH):
            chunk = missing[i:i + self.BATCH]
            for text, values in zip(chunk, self._embed_api(chunk, task)):
                norm = math.sqrt(sum(v * v for v in values)) or 1.0
                self._cache[(task, text)] = [v / norm for v in values]
        result = [self._cache[(task, t)] for t in texts]
        if len(self._cache) > self.CACHE_LIMIT:
            self._cache = dict(list(self._cache.items())[-self.CACHE_LIMIT // 2:])
        return result

    def _embed_api(self, chunk: List[str], task: str) -> List[List[float]]:
        from .llm import _retry_delay

        for attempt in range(3):
            try:
                response = self.client.models.embed_content(
                    model=self.model, contents=chunk,
                    config=self._types.EmbedContentConfig(task_type=task, output_dimensionality=self.dim),
                )
                return [list(e.values) for e in response.embeddings]
            except self._errors.APIError as exc:
                delay = _retry_delay(exc)
                if exc.code == 429 and delay is not None and delay <= self.max_wait and attempt < 2:
                    log.warning("Embedding rate limit; waiting %.0fs", delay)
                    time.sleep(delay + 1)
                    continue
                raise
        raise RuntimeError("unreachable")


def cache_key(task: str, text: str) -> str:
    return hashlib.sha256(f"{task}\n{text}".encode()).hexdigest()[:32]


def precomputed_path(model: str, dim: int) -> Path:
    return Path(__file__).parent / "data" / f"embeddings-{model}-{dim}.jsonl"


def load_precomputed(path: Path) -> Dict[str, List[float]]:
    """Vectors stored as base64 float32, keyed by cache_key(task, text)."""
    if not path.exists():
        return {}
    out: Dict[str, List[float]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[row["key"]] = list(array("f", base64.b64decode(row["v"])))
    return out


def save_precomputed(path: Path, items: Dict[str, List[float]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for key, vec in items.items():
            f.write(json.dumps({"key": key, "v": base64.b64encode(array("f", vec).tobytes()).decode()}) + "\n")


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

    def add(self, docs: List[Dict[str, Any]]) -> int: ...
    def search(self, query: str, k: int, kind: Optional[str] = None) -> List[Dict[str, Any]]: ...
    def all_docs(self) -> List[Dict[str, Any]]: ...
    def count(self) -> int: ...
    def healthy(self) -> bool: ...


# Metadata kept with every knowledge entry (besides the vector).
DOC_FIELDS = ("id", "kind", "title", "text", "roles", "seniority", "skills")


def _normalise_doc(d: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": d.get("id") or f"doc-{uuid.uuid4().hex[:12]}",
        "kind": d.get("kind", "guideline"),
        "title": d["title"],
        "text": d["text"],
        "roles": list(d.get("roles") or ["any"]),
        "seniority": d.get("seniority", "any"),
        "skills": list(d.get("skills") or []),
    }


def _embed_text(d: Dict[str, Any]) -> str:
    return f"{d['title']}. {d['text']} Skills: {', '.join(d.get('skills') or [])}"


class InMemoryStore:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.rows: List[Dict[str, Any]] = []

    def add(self, docs):
        docs = [_normalise_doc(d) for d in docs]
        vectors = self.embedder.embed_many([_embed_text(d) for d in docs], "RETRIEVAL_DOCUMENT")
        self.rows.extend({**d, "vector": v} for d, v in zip(docs, vectors))
        return len(docs)

    def search(self, query, k, kind=None):
        q = self.embedder.embed(query, "RETRIEVAL_QUERY")
        scored = [
            ({f: r[f] for f in DOC_FIELDS}, sum(a * b for a, b in zip(q, r["vector"])))
            for r in self.rows
            if kind is None or r["kind"] == kind
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [{**doc, "score": round(score, 4)} for doc, score in scored[:k]]

    def all_docs(self):
        return [{f: r[f] for f in DOC_FIELDS} for r in self.rows]

    def count(self):
        return len(self.rows)

    def healthy(self):
        return True


class MilvusStore:
    QUERY_LIMIT = 16384  # Milvus caps unfiltered query results

    def __init__(self, settings: Settings, embedder: Embedder):
        from pymilvus import MilvusClient

        self.embedder = embedder
        # One collection per embedder and knowledge-base version (e.g. resume_knowledge_gemini768_kb2):
        # vectors from different embedders, dimensions or corpora must never share a collection.
        self.collection = f"{settings.milvus_collection}_{embedder.tag}_{KNOWLEDGE_VERSION}"
        self.client = MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token or "")
        if not self.client.has_collection(self.collection):
            # Quick-setup schema: auto int64 pk + float vector; metadata lives in dynamic fields.
            self.client.create_collection(
                collection_name=self.collection,
                dimension=embedder.dim,
                metric_type="COSINE",
                auto_id=True,
                enable_dynamic_field=True,
            )
            log.info("Created Milvus collection %s (dim=%d)", self.collection, embedder.dim)

    def add(self, docs):
        docs = [_normalise_doc(d) for d in docs]
        vectors = self.embedder.embed_many([_embed_text(d) for d in docs], "RETRIEVAL_DOCUMENT")
        # The Milvus primary key is auto-generated, so the knowledge id is stored as doc_id.
        rows = [{"vector": v, **{("doc_id" if f == "id" else f): d[f] for f in DOC_FIELDS}} for d, v in zip(docs, vectors)]
        result = self.client.insert(collection_name=self.collection, data=rows)
        return int(result.get("insert_count", len(rows)))

    def _out_fields(self) -> List[str]:
        return ["doc_id"] + [f for f in DOC_FIELDS if f != "id"]

    @staticmethod
    def _from_row(row: Dict[str, Any]) -> Dict[str, Any]:
        return {f: (row.get("doc_id") if f == "id" else row.get(f)) for f in DOC_FIELDS}

    def search(self, query, k, kind=None):
        hits = self.client.search(
            collection_name=self.collection,
            data=[self.embedder.embed(query, "RETRIEVAL_QUERY")],
            limit=k,
            filter=f'kind == "{kind}"' if kind else "",
            output_fields=self._out_fields(),
        )
        return [{**self._from_row(hit["entity"]), "score": round(float(hit["distance"]), 4)} for hit in hits[0]]

    def all_docs(self):
        rows = self.client.query(self.collection, filter="", output_fields=self._out_fields(),
                                 limit=self.QUERY_LIMIT, consistency_level="Strong")
        return [self._from_row(r) for r in rows]

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


# Bump when data/knowledge.jsonl changes in a way existing Milvus collections should not keep.
KNOWLEDGE_VERSION = "kb2"
KNOWLEDGE_FILE = Path(__file__).parent / "data" / "knowledge.jsonl"


def load_knowledge(path: Path = KNOWLEDGE_FILE) -> List[Dict[str, Any]]:
    """The built-in knowledge base: guidelines, role profiles and example bullets tagged by role."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


SEED_KNOWLEDGE: List[Dict[str, Any]] = load_knowledge()


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
