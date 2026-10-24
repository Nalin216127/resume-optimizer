# Retrieval evaluation

`eval/eval_retrieval.py` scores retrieval over the 151-entry knowledge base with 40 hand-labelled queries (`eval/retrieval_queries.jsonl`). Each query lists the example bullets that answer it. There are three query styles:

- **exact (12):** uses the corpus's own terms, e.g. "Terraform infrastructure as code".
- **paraphrase (20):** different words, same meaning, e.g. "lower our cloud bill" or "tests that randomly fail".
- **rare_term (8):** a single uncommon tool name, e.g. "Debezium" or "k6".

Metrics are hit@k (a relevant entry appears in the top k) and MRR@10, the mean reciprocal rank of the first relevant entry.

## Gemini embeddings (`gemini-embedding-001`, 768 dims)

| Method | hit@1 | hit@3 | hit@5 | MRR@10 | paraphrase hit@1 |
|---|---|---|---|---|---|
| Semantic (vectors) | **1.00** | 1.00 | 1.00 | **1.000** | **1.00** |
| Keyword (BM25) | 0.82 | 0.85 | 0.85 | 0.842 | 0.65 |
| **Hybrid cascade (default)** | **1.00** | 1.00 | 1.00 | **1.000** | **1.00** |
| Hybrid RRF, keyword weight 1.0 | 0.90 | 1.00 | 1.00 | 0.946 | 0.80 |
| Hybrid RRF, keyword weight 0.5 | 0.93 | 1.00 | 1.00 | 0.958 | 0.85 |
| Hybrid RRF, keyword weight 0.3 | 0.93 | 1.00 | 1.00 | 0.958 | 0.85 |

## Offline fallback (hashing embedder, lexical)

| Method | hit@1 | hit@3 | hit@5 | MRR@10 |
|---|---|---|---|---|
| Semantic (lexical vectors) | 0.57 | 0.65 | 0.68 | 0.615 |
| Keyword (BM25) | **0.82** | 0.85 | 0.85 | **0.842** |
| **Hybrid cascade (default)** | **0.82** | 0.85 | 0.85 | **0.842** |
| Hybrid RRF, keyword weight 1.0 | 0.65 | 0.78 | 0.80 | 0.722 |

## What this decided

- **Semantic embeddings matter most for paraphrases.** BM25 found only 65% of paraphrased queries at rank 1; Gemini embeddings found all of them.
- **Reciprocal Rank Fusion hurt top-1 accuracy here**, at every keyword weight tried. RRF scores for neighbouring ranks are nearly equal, so even a small keyword contribution can push the right answer from #1 to #2.
- **The best single method depends on the embedder:** semantic with Gemini, BM25 with the lexical fallback. The default **cascade** therefore lets the embedder's preferred method rank the results. The other method only adds entries the first one missed and marks entries both found, which shows up as badges in the UI.
- **BM25 is the safety net.** If the embedding API is rate limited mid-request, retrieval answers with keyword search instead of failing.

Caveats: the queries were written for this corpus by the same author, so the absolute numbers flatter every method. The comparison between methods is what matters. Rerun after changing the knowledge base:

```bash
python eval/eval_retrieval.py                              # offline
GEMINI_API_KEY=... python eval/eval_retrieval.py --embedder gemini
```
