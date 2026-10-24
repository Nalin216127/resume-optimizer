"""Deterministic text helpers: keyword extraction, resume parsing, markdown rendering.

These ground the LLM agents (keyword coverage is computed, not guessed) and
power the offline mock LLM.
"""
from __future__ import annotations

import io
import re
from collections import Counter
from typing import Dict, Iterable, List, Tuple

from .schemas import ResumeDraft

_STOPWORDS = set(
    """a about above after again all also an and any are as at be because been being below between both but by can
    could did do does doing down during each etc few for from further had has have having he her here hers him his how
    i if in into is it its itself just like may me more most must my no nor not of off on once only or other our ours
    out over own per plus same she should so some such than that the their them then there these they this those
    through to too under until up us very via was we were what when where which while who whom why will with within
    would you your yours role team work working years year experience strong ability skills including across using
    based new well looking join ideal candidate responsibilities requirements qualifications preferred required
    bonus nice must-have degree equivalent knowledge understanding excellent good great help build building make
    company products product customers customer world best day every high level engineer engineers engineering
    senior junior lead staff principal applications application systems reliable agentic need software
    professional hands-on proven solid deep""".split()
)

# Multi-word / punctuated technical terms that naive tokenisation would split.
_PHRASES = [
    "machine learning", "deep learning", "natural language processing", "large language models", "computer vision",
    "data pipelines", "distributed systems", "system design", "ci/cd", "rest api", "rest apis", "event-driven",
    "a/b testing", "infrastructure as code", "vector database", "vector databases", "retrieval-augmented generation",
    "multi-agent", "human-in-the-loop", "prompt engineering", "cross-functional", "stakeholder management",
    "google cloud", "node.js", "react native", "spring boot", "scikit-learn", "power bi", "github actions",
]

STRONG_VERBS = {
    "helped build": "Built", "helped develop": "Developed", "worked on": "Engineered", "responsible for": "Owned", "helped": "Drove", "helped with": "Contributed to",
    "was part of": "Partnered on", "did": "Delivered", "made": "Built", "handled": "Managed", "used": "Leveraged",
    "involved in": "Led", "participated in": "Collaborated on", "assisted": "Supported", "tasked with": "Spearheaded",
}

_BULLET_RE = re.compile(r"^\s*(?:[-*•▪◦·]|\d+[.)])\s+")
# A number that is not a bare year (1990-2039) counts as a quantified outcome.
_METRIC_RE = re.compile(r"\$\s?\d|\b(?!(?:199\d|20[0-3]\d)\b)\d[\d,.]*\b")


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_keywords(text: str, limit: int = 25) -> List[str]:
    """Rank salient terms: known phrases first, then capitalised/technical tokens, then frequent words."""
    low = normalise(text)
    found: List[str] = [p for p in _PHRASES if p in low]

    tokens = re.findall(r"[A-Za-z][A-Za-z0-9+#./-]*[A-Za-z0-9+#]|[A-Za-z]", text)
    counts: Counter = Counter()
    for tok in tokens:
        key = tok.lower().strip(".")
        if len(key) < 2 or key in _STOPWORDS or any(key in p.split() for p in found):
            continue
        weight = 1.0
        if tok[0].isupper() and not tok.isupper():
            weight += 0.5  # Proper nouns: frameworks, tools, vendors
        if tok.isupper() and len(tok) <= 6:
            weight += 1.0  # Acronyms: AWS, SQL, LLM
        if re.search(r"[0-9+#./]", tok):
            weight += 1.0  # C++, Node.js, K8s
        counts[key] += weight
    found.extend(k for k, _ in counts.most_common(limit * 2) if k not in found)
    return found[:limit]


def split_requirements(job_description: str) -> Tuple[List[str], List[str]]:
    """Split JD keywords into must-have vs nice-to-have using the section they appear under."""
    must_text: List[str] = []
    nice_text: List[str] = []
    bucket = must_text
    for line in job_description.splitlines():
        low = line.lower()
        if any(k in low for k in ("nice to have", "preferred", "bonus", "plus if")):
            bucket = nice_text
        elif any(k in low for k in ("requirement", "must have", "qualifications", "you have", "what you bring")):
            bucket = must_text
        bucket.append(line)
    must = extract_keywords("\n".join(must_text), 15)
    nice = [k for k in extract_keywords("\n".join(nice_text), 10) if k not in must]
    return must, nice


def _tokens(text: str) -> set:
    """Meaningful lowercase terms: drops stopwords and bare numbers like '4+' or '2021'."""
    return {
        t for t in re.findall(r"[a-z0-9][a-z0-9+#./-]*[a-z0-9+#]|[a-z0-9]", text.lower())
        if t not in _STOPWORDS and not re.fullmatch(r"\d+\+?", t)
    }


def keyword_present(keyword: str, text: str) -> bool:
    """Exact phrase match, or (for longer phrases) at least two thirds of its meaningful words present.

    LLMs sometimes return whole requirement sentences ("4+ years of Python experience") as keywords;
    a literal substring test would score those as always missing.
    """
    phrase = re.escape(normalise(keyword))
    if re.search(rf"(?<![a-z0-9]){phrase}(?![a-z0-9])", normalise(text)):
        return True
    words = _tokens(keyword)
    if not words:
        return False
    return len(words & _tokens(text)) / len(words) >= 2 / 3


def keyword_coverage(keywords: Iterable[str], text: str) -> Tuple[float, List[str]]:
    keywords = [k for k in keywords if k]
    if not keywords:
        return 1.0, []
    missing = [k for k in keywords if not keyword_present(k, text)]
    return 1 - len(missing) / len(keywords), missing


def has_metric(text: str) -> bool:
    return bool(_METRIC_RE.search(text))


def parse_resume(resume_text: str) -> Dict[str, List[str]]:
    """Group bullet lines under the nearest preceding heading. Non-bullet prose goes to 'Summary'."""
    sections: Dict[str, List[str]] = {}
    current = "Summary"
    for raw in resume_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _BULLET_RE.match(line):
            sections.setdefault(current, []).append(_BULLET_RE.sub("", line).strip())
        elif len(line) < 90 and (line.isupper() or line.endswith(":") or line.startswith("#") or "|" in line or "—" in line):
            current = line.strip("#: ").strip()
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(line)
    return {k: v for k, v in sections.items() if v}


def strengthen_bullet(bullet: str) -> str:
    text = bullet.strip().rstrip(".")
    low = text.lower()
    for weak, strong in sorted(STRONG_VERBS.items(), key=lambda kv: -len(kv[0])):
        if low.startswith(weak):
            text = strong + text[len(weak):]
            break
    return text[0].upper() + text[1:] if text else text


def render_markdown(draft: ResumeDraft) -> str:
    out = [f"# {draft.headline}", "", draft.summary, ""]
    for section in draft.sections:
        out.append(f"## {section.title}")
        out.extend(f"- {b}" for b in section.bullets)
        out.append("")
    if draft.skills:
        out += ["## Skills", ", ".join(draft.skills), ""]
    return "\n".join(out).strip() + "\n"


def file_to_text(filename: str, data: bytes) -> str:
    """Extract plain text from an uploaded PDF / TXT / MD resume. Raises ValueError on unreadable input."""
    if filename.lower().endswith(".pdf"):
        from pypdf import PdfReader

        try:
            reader = PdfReader(io.BytesIO(data))
            return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        except Exception as exc:  # noqa: BLE001 - pypdf raises many types on bad input
            raise ValueError(f"Could not read PDF: {exc}") from exc
    try:
        return data.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise ValueError("Upload a PDF, .txt or .md file") from exc
