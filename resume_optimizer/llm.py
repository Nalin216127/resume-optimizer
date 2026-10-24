"""LLM backends.

Agents call ``llm.structured(...)`` and always get back a validated Pydantic
model. Two implementations:

* ``AnthropicLLM`` — Claude via the official Anthropic SDK structured outputs.
* ``GeminiLLM`` — Google Gemini via the google-genai SDK with JSON-schema output.
* ``MockLLM`` — deterministic heuristics so the whole system (graph, API, UI,
  tests) runs offline without an API key.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Protocol, Type, TypeVar

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


# --------------------------------------------------------------------- Gemini
class _QuotaExhausted(Exception):
    """This model can't serve now: a long 429 (e.g. a daily free-tier cap) or repeated 5xx overload.

    The caller moves on to the next model and skips this one for ``retry_after`` seconds.
    """

    def __init__(self, model: str, retry_after: float, reason: str = "quota exhausted"):
        super().__init__(model)
        self.model, self.retry_after, self.reason = model, retry_after, reason


def _retry_delay(exc: Any) -> Optional[float]:
    """Seconds from a google.rpc.RetryInfo detail (e.g. "37s"), if the error carries one."""
    details = (getattr(exc, "details", None) or {}).get("error", {}).get("details", [])
    for d in details:
        if d.get("@type", "").endswith("RetryInfo") and str(d.get("retryDelay", "")).endswith("s"):
            try:
                return float(d["retryDelay"][:-1])
            except ValueError:
                return None
    return None


def _human_delay(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes // 60}h {minutes % 60}m" if minutes >= 60 else f"{max(minutes, 1)}m"


class GeminiLLM:
    RETRYABLE = {429, 500, 502, 503, 504}
    MAX_INLINE_WAIT = 60  # wait out per-minute limits; anything longer moves to the next model
    OVERLOAD_COOLDOWN = 120  # after repeated 5xx, skip that model for this long
    # Agent effort → Gemini thinking budget (tokens). Capped even at "high": unbounded thinking made
    # single Flash calls take over a minute.
    THINKING_BUDGET = {"low": 512, "medium": 2048, "high": 6144}

    def __init__(self, settings: Settings):
        import httpx
        from google import genai  # imported lazily: only needed when LLM_PROVIDER=gemini
        from google.genai import errors, types

        if not settings.gemini_api_key:
            raise ValueError("LLM_PROVIDER=gemini requires GEMINI_API_KEY")
        self._errors, self._types, self._transport_error = errors, types, httpx.TransportError
        # Without a timeout, a dropped connection (e.g. the host sleeping) hangs the request forever.
        self.client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(timeout=settings.llm_timeout_seconds * 1000),
        )
        self.model = settings.gemini_model
        # Quotas are per model, so when one model's quota runs out the next one usually still works.
        self.models = [self.model] + [m for m in settings.gemini_fallback_models if m != self.model]
        self._exhausted_until: Dict[str, float] = {}

    def structured(self, *, task, system, prompt, schema, payload, effort):
        budget = self.THINKING_BUDGET.get(effort)
        config = self._types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=schema.model_json_schema(),
            max_output_tokens=16000,
            thinking_config=self._types.ThinkingConfig(thinking_budget=budget) if budget else None,
        )
        response = self._generate_with_fallback(task, prompt, config)

        feedback = getattr(response, "prompt_feedback", None)
        if feedback is not None and getattr(feedback, "block_reason", None):
            raise LLMError(f"{task}: Gemini blocked the prompt ({feedback.block_reason})")
        finish = response.candidates[0].finish_reason if response.candidates else None
        finish_name = getattr(finish, "name", str(finish))
        if finish_name == "MAX_TOKENS":
            raise LLMError(f"{task}: response hit max_output_tokens before completing")
        if finish_name not in (None, "STOP", "FINISH_REASON_UNSPECIFIED"):
            raise LLMError(f"{task}: Gemini stopped early ({finish_name})")
        try:
            return schema.model_validate_json(response.text or "")
        except ValueError as exc:
            raise LLMError(f"{task}: Gemini returned output that does not match the schema") from exc

    def _generate_with_fallback(self, task, prompt, config):
        now = time.time()
        available = [m for m in self.models if self._exhausted_until.get(m, 0) <= now]
        reasons: Dict[str, str] = {}
        for model in available:
            try:
                return self._generate(task, model, prompt, config)
            except _QuotaExhausted as exc:
                self._exhausted_until[model] = time.time() + exc.retry_after
                reasons[model] = exc.reason
                log.warning("gemini %s: %s on %s, skipping it for %s; trying next model",
                            task, exc.reason, model, _human_delay(exc.retry_after))
        soonest = min(self._exhausted_until.get(m, now) for m in self.models) - time.time()
        if reasons and all(r == "overloaded" for r in reasons.values()):
            raise LLMError(f"{task}: Gemini is overloaded right now (HTTP 503 on {', '.join(reasons)}). "
                           "Press Retry in a minute; finished steps are kept.")
        raise LLMError(
            f"{task}: Gemini quota exhausted for {', '.join(self.models)} (free tier allows few requests per day). "
            f"Resets in about {_human_delay(max(soonest, 60))}; use a paid key or retry later."
        )

    def _generate(self, task, model, prompt, config, attempts: int = 4):
        for attempt in range(attempts):
            started = time.monotonic()
            try:
                response = self.client.models.generate_content(model=model, contents=prompt, config=config)
                log.info("gemini %s: %s %.1fs (attempt %d)", task, model, time.monotonic() - started, attempt + 1)
                return response
            except self._errors.APIError as exc:
                delay = _retry_delay(exc)
                if exc.code == 429 and delay is not None and delay > self.MAX_INLINE_WAIT:
                    raise _QuotaExhausted(model, delay) from exc
                if exc.code in self.RETRYABLE and attempt < attempts - 1:
                    time.sleep(delay if delay is not None else 2 ** (attempt + 1))  # 2s, 4s, 8s
                    continue
                if exc.code == 429:
                    raise _QuotaExhausted(model, delay or self.MAX_INLINE_WAIT) from exc
                if exc.code >= 500:
                    # Still failing after retries: the model is overloaded, so try the next one.
                    raise _QuotaExhausted(model, self.OVERLOAD_COOLDOWN, "overloaded") from exc
                if exc.code == 404:
                    raise LLMError(f"{task}: Gemini model {model} is unavailable to this key: set GEMINI_MODEL") from exc
                if exc.code in (400, 401, 403):
                    raise LLMError(f"{task}: Gemini rejected the request ({exc.code}): check GEMINI_API_KEY and GEMINI_MODEL") from exc
                raise LLMError(f"{task}: Gemini API error {exc.code}") from exc
            except (self._transport_error, OSError) as exc:  # timeouts, dropped connections, DNS failures
                log.warning("gemini %s: %s after %.1fs (attempt %d)", task, type(exc).__name__,
                            time.monotonic() - started, attempt + 1)
                if attempt < attempts - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise LLMError(f"{task}: could not reach the Gemini API (timed out or connection lost)") from exc


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
    if settings.llm_provider == "gemini":
        return GeminiLLM(settings)
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")
