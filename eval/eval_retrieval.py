"""Retrieval evaluation: semantic vs keyword (BM25) vs hybrid (RRF) on labelled queries.

    python eval/eval_retrieval.py                 # hashing embedder (offline)
    GEMINI_API_KEY=... python eval/eval_retrieval.py --embedder gemini

Each query in eval/retrieval_queries.jsonl lists the example bullets that answer it and whether
it uses the corpus's exact terms ("exact") or different words ("paraphrase"). Reports hit@1/3/5
and MRR@10 per retrieval mode, overall and per query style.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resume_optimizer.config import Settings  # noqa: E402
from resume_optimizer.retrieval import HybridRetriever  # noqa: E402
from resume_optimizer.vectorstore import SEED_KNOWLEDGE, InMemoryStore, build_embedder  # noqa: E402

MODES = ("semantic", "keyword", "hybrid")
WEIGHTS = (1.0, 0.5, 0.3)  # keyword weights tried for hybrid
STYLES = ("exact", "paraphrase", "rare_term")


def evaluate(retriever: HybridRetriever, queries: list) -> dict:
    retriever.store.embedder.embed_many([q["query"] for q in queries], "RETRIEVAL_QUERY")  # one batched call
    runs = [("semantic", None), ("keyword", None), ("cascade", None)] + [("hybrid", w) for w in WEIGHTS]
    results = {}
    for mode, weight in runs:
        rows = []
        for q in queries:
            if mode == "cascade":
                hits = retriever.search(q["query"], k=10, kind="example_bullet")  # the default
            else:
                hits = retriever.search(q["query"], k=10, kind="example_bullet", mode=mode, fusion="rrf",
                                        keyword_weight=weight or 1.0)
            ids = [h["id"] for h in hits]
            rank = next((i + 1 for i, d in enumerate(ids) if d in set(q["relevant"])), None)
            rows.append({"style": q["style"], "rank": rank})
        name = {"cascade": "hybrid cascade*"}.get(mode, mode) if weight is None else f"hybrid rrf kw={weight}"
        results[name] = rows
    return results


def summarise(rows: list) -> dict:
    n = len(rows) or 1
    hit = lambda k: sum(1 for r in rows if r["rank"] and r["rank"] <= k) / n  # noqa: E731
    mrr = sum(1 / r["rank"] for r in rows if r["rank"]) / n
    return {"n": len(rows), "hit@1": hit(1), "hit@3": hit(3), "hit@5": hit(5), "mrr@10": mrr}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedder", choices=["hash", "gemini"], default="hash")
    parser.add_argument("--json", action="store_true", help="print machine-readable results")
    args = parser.parse_args()

    settings = replace(Settings(), embedding_backend=args.embedder)
    embedder = build_embedder(settings)
    if args.embedder == "gemini" and type(embedder).__name__ != "GeminiEmbedder":
        sys.exit("Gemini embeddings unavailable: set GEMINI_API_KEY")
    if hasattr(embedder, "max_wait"):
        embedder.max_wait = 70  # offline script: wait out per-minute limits instead of failing
    store = InMemoryStore(embedder)
    store.add(SEED_KNOWLEDGE)
    queries = [json.loads(line) for line in open(Path(__file__).with_name("retrieval_queries.jsonl"))]
    results = evaluate(HybridRetriever(store), queries)

    table = {}
    for mode, rows in results.items():
        table[mode] = {"all": summarise(rows)}
        for style in STYLES:
            table[mode][style] = summarise([r for r in rows if r["style"] == style])
    if args.json:
        print(json.dumps({"embedder": embedder.label, "results": table}, indent=2))
        return
    print(f"Embedder: {embedder.label} | corpus: {len(SEED_KNOWLEDGE)} entries | queries: {len(queries)} "
          "| * = default\n")
    print(f"{'mode':20s} {'subset':11s} {'n':>3s} {'hit@1':>6s} {'hit@3':>6s} {'hit@5':>6s} {'MRR@10':>7s}")
    for mode in table:
        for subset in ("all",) + STYLES:
            m = table[mode][subset]
            print(f"{mode:20s} {subset:11s} {m['n']:3d} {m['hit@1']:6.2f} {m['hit@3']:6.2f} {m['hit@5']:6.2f} {m['mrr@10']:7.3f}")


if __name__ == "__main__":
    main()
