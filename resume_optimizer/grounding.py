"""Evidence check: is each claim in the draft traceable to the original resume?

Stage one of a two-stage grounding check (the pattern MedRAG's safety agent uses): every draft
bullet and summary sentence is embedded and compared with every line of the original resume.
Claims whose best match is weak are flagged and handed to the Reviewer LLM, which makes the final
call in the request it already sends, so the check costs no extra LLM calls.

Fails closed: if the embedding API errors, the check reruns with the local hashing embedder
instead of being skipped.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from .schemas import EvidenceItem
from .vectorstore import Embedder, HashingEmbedder

log = logging.getLogger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_BULLET = re.compile(r"^\s*(?:[-*•▪◦·]|\d+[.)])\s+")


def resume_evidence(resume_text: str) -> List[str]:
    """Lines and sentences of the original resume that a claim could be traced to."""
    pieces: List[str] = []
    for line in resume_text.splitlines():
        line = _BULLET.sub("", line).strip()
        if len(line.split()) < 2:
            continue
        pieces.extend(p.strip() for p in _SENTENCE_SPLIT.split(line) if len(p.split()) >= 2)
    pieces.append(" ".join(resume_text.split()))  # whole resume, for claims that synthesise several lines
    return list(dict.fromkeys(pieces))


def draft_claims(draft: Dict[str, Any]) -> List[str]:
    claims = [s.strip() for s in _SENTENCE_SPLIT.split(draft.get("summary", "")) if len(s.split()) >= 4]
    for section in draft.get("sections", []):
        claims.extend(b.strip() for b in section.get("bullets", []) if len(b.split()) >= 3)
    return list(dict.fromkeys(claims))


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def evidence_check(resume_text: str, draft: Dict[str, Any], embedder: Embedder) -> List[EvidenceItem]:
    sources, claims = resume_evidence(resume_text), draft_claims(draft)
    if not sources or not claims:
        return []
    try:
        src_vecs = embedder.embed_many(sources, "SEMANTIC_SIMILARITY")
        claim_vecs = embedder.embed_many(claims, "SEMANTIC_SIMILARITY")
        threshold = embedder.support_threshold
    except Exception as exc:  # noqa: BLE001 - never skip the check; degrade to lexical matching
        log.warning("Evidence check: embedding API failed (%s); using the hashing embedder", exc)
        fallback = HashingEmbedder()
        src_vecs = fallback.embed_many(sources)
        claim_vecs = fallback.embed_many(claims)
        threshold = fallback.support_threshold

    items = []
    for claim, vec in zip(claims, claim_vecs):
        scores = [_dot(vec, s) for s in src_vecs[:-1]] or [0.0]  # best single line, not the whole-resume blob
        best = max(range(len(scores)), key=scores.__getitem__)
        whole = _dot(vec, src_vecs[-1])
        score = max(scores[best], whole)
        items.append(EvidenceItem(
            claim=claim, best_match=sources[best], score=round(score, 3), supported=score >= threshold,
        ))
    return items
