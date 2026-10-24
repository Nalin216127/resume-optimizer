"""Hybrid retrieval over the knowledge base: semantic (vector store) + keyword (BM25).

Semantic search catches paraphrases ("make the site load faster" finds a Core Web Vitals bullet);
BM25 catches exact terms and keeps retrieval working when the embedding API is unavailable.
Results are filtered by kind, role and seniority using each entry's metadata, and every result
records which method(s) found it.

How the two are combined was decided by measurement (eval/eval_retrieval.py): with Gemini
embeddings, semantic ranking alone beat both BM25 and Reciprocal Rank Fusion, so the default
"cascade" lets the stronger method rank and the other only add what it missed. Weighted RRF is
kept as an option for comparison.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .textutils import _STOPWORDS
from .vectorstore import KnowledgeStore

log = logging.getLogger(__name__)

RRF_K = 60          # standard Reciprocal Rank Fusion constant
CANDIDATES = 40     # per-method candidates fetched before filtering and fusion


def tokenize(text: str) -> List[str]:
    """Lowercase terms that keep tech punctuation (c++, ci/cd, node.js) and drop stopwords."""
    tokens = re.findall(r"[a-z0-9][a-z0-9+#./-]*[a-z0-9+#]|[a-z0-9]", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def doc_text(doc: Dict[str, Any]) -> str:
    return " ".join([doc.get("title", ""), doc.get("text", ""), " ".join(doc.get("skills") or [])])


class BM25Index:
    """Okapi BM25 over title + text + skill tags (k1=1.5, b=0.75)."""

    def __init__(self, docs: Sequence[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = list(docs)
        self.tfs = [Counter(tokenize(doc_text(d))) for d in self.docs]
        self.lengths = [sum(tf.values()) for tf in self.tfs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df: Counter = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, n: int) -> List[tuple]:
        terms = [t for t in tokenize(query) if t in self.idf]
        if not terms:
            return []
        scores = []
        for i, tf in enumerate(self.tfs):
            s = 0.0
            for t in terms:
                f = tf.get(t)
                if f:
                    norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avg_len or 1))
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            if s > 0:
                scores.append((i, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]


def _matches(doc: Dict[str, Any], kind: Optional[str], roles: Optional[Iterable[str]],
             seniority: Optional[str]) -> bool:
    if kind and doc.get("kind") != kind:
        return False
    if roles:
        doc_roles = set(doc.get("roles") or ["any"])
        if "any" not in doc_roles and not doc_roles & set(roles):
            return False
    if seniority and doc.get("seniority", "any") not in ("any", seniority):
        return False
    return True


class HybridRetriever:
    def __init__(self, store: KnowledgeStore):
        self.store = store
        self.refresh()

    @property
    def primary_method(self) -> str:
        """Which method ranks results. Measured on eval/retrieval_queries.jsonl: with Gemini embeddings
        semantic search alone is best (hit@1 1.00 vs 0.82 for BM25, and RRF fusion lowered it to
        0.90-0.93); with the lexical hashing fallback, BM25 is best."""
        return getattr(self.store.embedder, "preferred_method", "semantic")

    def refresh(self) -> None:
        self.docs = self.store.all_docs()
        self.by_id = {d["id"]: d for d in self.docs}
        self.bm25 = BM25Index(self.docs)

    def add(self, docs: List[Dict[str, Any]]) -> int:
        inserted = self.store.add(docs)
        self.refresh()
        return inserted

    def search(self, query: str, k: int = 5, kind: Optional[str] = None, roles: Optional[Iterable[str]] = None,
               seniority: Optional[str] = None, mode: str = "hybrid", fusion: str = "cascade",
               keyword_weight: float = 1.0) -> List[Dict[str, Any]]:
        """mode: "hybrid" (default), "semantic" or "keyword".

        Hybrid fusion:
        - "cascade" (default): the primary method (see primary_method) ranks the results; the other
          method only adds entries the primary missed and marks entries both found. If semantic
          search fails (embedding API down or rate limited), keyword search answers alone.
        - "rrf": weighted Reciprocal Rank Fusion, kept for comparison in the eval.
        """
        lists: Dict[str, List[str]] = {}
        if mode in ("hybrid", "semantic"):
            try:
                lists["semantic"] = [h["id"] for h in self.store.search(query, k=CANDIDATES)]
            except Exception as exc:  # noqa: BLE001 - embedding API down or rate limited: keyword-only
                log.warning("Semantic search unavailable (%s); using keyword search only", exc)
                lists["semantic"] = []
        if mode in ("hybrid", "keyword"):
            lists["keyword"] = [self.docs[i]["id"] for i, _ in self.bm25.search(query, CANDIDATES)]
        for method in lists:  # apply metadata filters before ranking
            lists[method] = [i for i in lists[method] if i in self.by_id and _matches(self.by_id[i], kind, roles, seniority)]

        found_by: Dict[str, List[str]] = {}
        for method, ids in lists.items():
            for doc_id in ids:
                found_by.setdefault(doc_id, []).append(method)

        if mode != "hybrid":
            order = lists[mode]
            scores = {d: 1.0 / (RRF_K + r) for r, d in enumerate(order, 1)}
        elif fusion == "rrf":
            weights = {"semantic": 1.0, "keyword": keyword_weight}
            scores = {}
            for method, ids in lists.items():
                for r, d in enumerate(ids, 1):
                    scores[d] = scores.get(d, 0.0) + weights[method] / (RRF_K + r)
            order = sorted(scores, key=scores.get, reverse=True)
        else:
            primary = self.primary_method if lists.get(self.primary_method) else (
                "keyword" if lists.get("keyword") else "semantic")
            secondary = "keyword" if primary == "semantic" else "semantic"
            order = list(lists.get(primary, [])) + [d for d in lists.get(secondary, []) if d not in set(lists.get(primary, []))]
            scores = {d: 1.0 / (RRF_K + r) for r, d in enumerate(order, 1)}

        return [{**{f: self.by_id[d].get(f) for f in ("id", "kind", "title", "text", "roles", "seniority", "skills")},
                 "score": round(scores[d], 4), "matched_by": found_by.get(d, [])} for d in order[:k]]

    def multi_search(self, queries: Sequence[str], k: int = 5, **filters: Any) -> List[Dict[str, Any]]:
        """Run several focused queries (e.g. one per required skill) and fuse their rankings with RRF."""
        queries = [q for q in queries if q.strip()]
        embed_many = getattr(self.store.embedder, "embed_many", None)
        if embed_many and queries:
            try:
                embed_many(list(queries), "RETRIEVAL_QUERY")  # warms the embedder cache in one batched call
            except Exception as exc:  # noqa: BLE001 - search() falls back to keyword-only per query
                log.warning("Could not pre-embed queries (%s)", exc)
        fused: Dict[str, Dict[str, Any]] = {}
        for q in queries:
            for rank, hit in enumerate(self.search(q, k=CANDIDATES // 2, **filters), 1):
                entry = fused.setdefault(hit["id"], {**hit, "score": 0.0, "matched_by": set(), "queries": []})
                entry["score"] += 1.0 / (RRF_K + rank)
                entry["matched_by"].update(hit["matched_by"])
                entry["queries"].append(q)
        top = sorted(fused.values(), key=lambda e: e["score"], reverse=True)[:k]
        return [{**e, "score": round(e["score"], 4), "matched_by": sorted(e["matched_by"]),
                 "queries": e["queries"][:3]} for e in top]

    def detect_roles(self, text: str) -> List[str]:
        """Role families whose profiles best match a job description, best first.

        The hybrid winner is the primary role. A second role is added only when semantic and keyword
        search both rank it in their top two (e.g. an "ML platform" job is ml_ai + devops); RRF scores
        alone sit too close together to threshold on.
        """
        hybrid = self.search(text, k=1, kind="role_profile")
        if not hybrid:
            return []
        primary = [r for r in hybrid[0].get("roles") or [] if r != "any"]
        sem = {r for h in self.search(text, k=2, kind="role_profile", mode="semantic") for r in h.get("roles") or []}
        kw = {r for h in self.search(text, k=2, kind="role_profile", mode="keyword") for r in h.get("roles") or []}
        secondary = sorted((sem & kw) - set(primary) - {"any"})
        return primary + secondary[:1]
