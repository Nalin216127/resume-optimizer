"""LLM backends.

Agents call ``llm.structured(...)`` and always get back a validated Pydantic
model. Two implementations:

* ``AnthropicLLM`` — Claude via the official Anthropic SDK structured outputs.
* ``MockLLM`` — deterministic heuristics so the whole system (graph, API, UI,
  tests) runs offline without an API key.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Protocol, Type, TypeVar

from pydantic import BaseModel

from . import textutils as tu
from .config import Settings
from .schemas import (
    ContextBrief,
    ResumeDraft,
    ResumeSection,
    ReviewIssue,
    ReviewReport,
    ReviewScores,
)

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when the model cannot produce a usable structured response."""


class LLM(Protocol):
    def structured(
        self, *, task: str, system: str, prompt: str, schema: Type[T], payload: Dict[str, Any], effort: str
    ) -> T: ...


# --------------------------------------------------------------------- Claude
class AnthropicLLM:
    def __init__(self, settings: Settings):
        import anthropic  # imported lazily so the mock path has no hard dependency

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=4, timeout=300)
        self.model = settings.anthropic_model
        self.fallback = settings.anthropic_refusal_fallback

    def structured(self, *, task, system, prompt, schema, payload, effort):
        kwargs: Dict[str, Any] = dict(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,
            output_config={"effort": effort},
        )
        try:
            if self.fallback:
                # Server-side refusal fallback: on a policy decline the API re-runs the
                # request on a fallback model inside the same call.
                response = self.client.beta.messages.parse(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
                )
            else:
                response = self.client.messages.parse(**kwargs)
        except self._anthropic.RateLimitError as exc:
            raise LLMError(f"{task}: rate limited by the Claude API, retry later") from exc
        except self._anthropic.BadRequestError as exc:
            raise LLMError(f"{task}: request rejected by the Claude API: {exc.message}") from exc
        except self._anthropic.APIStatusError as exc:
            raise LLMError(f"{task}: Claude API error {exc.status_code}") from exc
        except self._anthropic.APIConnectionError as exc:
            raise LLMError(f"{task}: could not reach the Claude API") from exc

        if response.stop_reason == "refusal":
            raise LLMError(f"{task}: the model declined this request")
        if response.stop_reason == "max_tokens":
            raise LLMError(f"{task}: response hit max_tokens before completing")
        parsed = response.parsed_output
        if parsed is None:
            raise LLMError(f"{task}: model returned no parseable output")
        return parsed


# ----------------------------------------------------------------------- Mock
class MockLLM:
    """Heuristic stand-in for Claude. Good enough to exercise every workflow path."""

    def structured(self, *, task, system, prompt, schema, payload, effort):
        handler = {"context": self._context, "writer": self._writer, "reviewer": self._reviewer}[task]
        return schema.model_validate(handler(payload).model_dump())

    # Context agent ---------------------------------------------------------
    def _context(self, p: Dict[str, Any]) -> ContextBrief:
        jd, resume = p["job_description"], p["resume_text"]
        must, nice = tu.split_requirements(jd)
        low_resume = tu.normalise(resume)
        strengths = [k for k in must + nice if k.lower() in low_resume]
        gaps = [k for k in must if k.lower() not in low_resume]
        title = p.get("target_role") or next((l.strip() for l in jd.splitlines() if l.strip()), "Target role")
        seniority = next((s for s in ("principal", "staff", "lead", "senior", "junior") if s in jd.lower()), "mid")
        guidance = [d["text"] for d in p.get("retrieved", [])][:4]
        return ContextBrief(
            target_role=title[:80], seniority=seniority, must_have_keywords=must, nice_to_have_keywords=nice,
            candidate_strengths=strengths, gaps=gaps, writing_guidance=guidance,
        )

    # Writer agent ----------------------------------------------------------
    def _writer(self, p: Dict[str, Any]) -> ResumeDraft:
        ctx = p["context"]
        parsed = tu.parse_resume(p["resume_text"])
        summary_lines = parsed.pop("Summary", [])
        skills_src = [k for k in list(parsed) if "skill" in k.lower()]
        resume_skills: List[str] = []
        for key in skills_src:
            for line in parsed.pop(key):
                resume_skills += [s.strip() for s in re.split(r"[,;|]", line) if s.strip()]

        low_resume = tu.normalise(p["resume_text"])
        matched = [k for k in ctx["must_have_keywords"] + ctx["nice_to_have_keywords"] if k.lower() in low_resume]
        skills = list({s.lower(): s for s in reversed(matched + resume_skills)}.values())[::-1][:20]

        sections = []
        for title, bullets in parsed.items():
            improved = [tu.strengthen_bullet(b) for b in bullets]
            # Lead each section with the bullets that carry metrics or target keywords.
            improved.sort(key=lambda b: (not tu.has_metric(b), not any(k.lower() in b.lower() for k in matched)))
            sections.append(ResumeSection(title=title, bullets=improved))

        role = ctx["target_role"]
        strengths = ", ".join(ctx["candidate_strengths"][:4]) or "delivering results"
        base = " ".join(summary_lines[1:3]) if len(summary_lines) > 1 else ""
        summary = f"{ctx['seniority'].title()}-level professional targeting {role}, with proven strength in {strengths}."
        if base:
            summary += " " + base

        changes = ["Rewrote bullets to open with strong action verbs", "Reordered bullets to lead with measurable impact"]
        if p.get("review"):
            changes.append("Addressed reviewer issues: " + "; ".join(i["problem"] for i in p["review"]["report"]["issues"][:3]))
        if p.get("human_feedback"):
            changes.append(f"Applied human feedback: {p['human_feedback']}")
            summary += " " + p["human_feedback"].strip().rstrip(".") + "."
        return ResumeDraft(
            headline=f"{summary_lines[0] if summary_lines else 'Candidate'} — {role}"[:120],
            summary=summary, sections=sections, skills=skills, change_log=changes,
        )

    # Reviewer agent --------------------------------------------------------
    def _reviewer(self, p: Dict[str, Any]) -> ReviewReport:
        ctx, draft_md, original = p["context"], p["draft_markdown"], p["resume_text"]
        coverage, missing = tu.keyword_coverage(ctx["must_have_keywords"], draft_md)
        bullets = [l[2:] for l in draft_md.splitlines() if l.startswith("- ")]
        with_metrics = [b for b in bullets if tu.has_metric(b)]
        impact = 10 * len(with_metrics) / max(len(bullets), 1)
        avg_words = sum(len(b.split()) for b in bullets) / max(len(bullets), 1)
        clarity = 9.0 if 10 <= avg_words <= 28 else 6.5
        original_numbers = set(re.findall(r"\d[\d,.]*", original))
        fabricated = [b for b in bullets if any(n not in original_numbers for n in re.findall(r"\d[\d,.]*", b))]

        issues = [
            ReviewIssue(severity="medium", section="Experience", problem=f"No quantified outcome: '{b[:60]}'",
                        suggestion="Add a metric (%, $, time saved, scale) if the original resume supports one")
            for b in bullets if not tu.has_metric(b)
        ][:4]
        if missing:
            issues.insert(0, ReviewIssue(severity="high", section="Skills/Experience",
                                         problem="Missing must-have keywords: " + ", ".join(missing[:6]),
                                         suggestion="Surface these only where the candidate has real experience"))
        return ReviewReport(
            scores=ReviewScores(
                keyword_alignment=round(10 * coverage, 1), impact_quantification=round(max(impact, 3.0), 1),
                clarity_concision=clarity, ats_formatting=9.0, truthfulness=10.0 - min(len(fabricated) * 3, 10),
            ),
            strengths=[f"{len(with_metrics)} of {len(bullets)} bullets show measurable impact"],
            issues=issues, missing_keywords=missing, possible_fabrications=fabricated,
        )


def build_llm(settings: Settings) -> LLM:
    if settings.llm_provider == "mock":
        log.warning("LLM_PROVIDER=mock: using offline heuristic agents, not Claude")
        return MockLLM()
    if settings.llm_provider == "anthropic":
        return AnthropicLLM(settings)
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")
