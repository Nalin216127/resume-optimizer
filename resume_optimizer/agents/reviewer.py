"""Reviewer agent: scores the draft against a rubric and decides pass / revise.

The LLM's rubric scores are blended with a deterministic keyword-coverage
measurement, so the pass/fail gate can't be talked into passing a draft that
is missing the job's must-have terms.
"""
from __future__ import annotations

import json

from ..config import Settings
from ..grounding import evidence_check
from ..llm import LLM
from ..schemas import ReviewReport, ReviewResult
from ..state import ResumeState
from ..textutils import keyword_coverage
from ..vectorstore import Embedder

SYSTEM = """You are the Reviewer agent in a resume optimization team: a demanding recruiter and ATS \
expert. Score the draft resume from 0 to 10 on each rubric dimension:

- keyword_alignment: uses the job's must-have terms where the candidate has the experience
- impact_quantification: bullets show measurable outcomes
- clarity_concision: bullets are crisp, one to two lines, active voice
- ats_formatting: standard headings, plain structure, parseable
- truthfulness: every claim is supported by the original resume (10 = fully supported)

Compare the draft with the original resume and list any claim, number or skill that the original does \
not support under possible_fabrications. Give specific, actionable issues the Writer can fix. Score \
honestly: a 9 or 10 means a recruiter would not change anything."""

WEIGHTS = {
    "keyword_alignment": 0.25,
    "impact_quantification": 0.2,
    "clarity_concision": 0.15,
    "ats_formatting": 0.1,
    "truthfulness": 0.3,
}


def make_reviewer_agent(llm: LLM, settings: Settings, embedder: Embedder):
    def reviewer_agent(state: ResumeState) -> dict:
        ctx = state["context"]
        evidence = evidence_check(state["resume_text"], state["draft"], embedder)
        weak = [e for e in evidence if not e.supported]
        payload = {"resume_text": state["resume_text"], "draft_markdown": state["draft_markdown"], "context": ctx,
                   "weak_evidence": [e.model_dump() for e in weak]}
        prompt = (
            f"<original_resume>\n{state['resume_text']}\n</original_resume>\n\n"
            f"<job_requirements>\n{json.dumps({k: ctx[k] for k in ('target_role', 'must_have_keywords', 'nice_to_have_keywords')}, indent=2)}\n</job_requirements>\n\n"
            f"<draft_resume>\n{state['draft_markdown']}\n</draft_resume>"
        )
        if weak:
            listed = "\n".join(f"- {e.claim}  (closest original line: \"{e.best_match}\")" for e in weak)
            prompt += (
                "\n\n<weak_evidence>\nAn automatic similarity check could not clearly trace these draft claims to "
                "the original resume. Check each one: if the original does not support it, list it under "
                f"possible_fabrications.\n{listed}\n</weak_evidence>"
            )
        report = llm.structured(task="reviewer", system=SYSTEM, prompt=prompt, schema=ReviewReport,
                                payload=payload, effort=settings.reviewer_effort)

        coverage, missing = keyword_coverage(ctx["must_have_keywords"], state["draft_markdown"])
        scores = report.scores.model_dump()
        rubric = sum(scores[k] * w for k, w in WEIGHTS.items())
        overall = round(0.8 * rubric + 0.2 * (10 * coverage), 2)
        # A suspected fabrication always sends the draft back, whatever the score.
        passed = overall >= state["quality_threshold"] and not report.possible_fabrications
        if not report.missing_keywords:
            report.missing_keywords = missing

        result = ReviewResult(report=report, keyword_coverage=round(coverage, 3), overall=overall,
                              verdict="pass" if passed else "revise", evidence=evidence)
        return {
            "review": result.model_dump(),
            "reviewed_version": state["draft_version"],
            "history": [{"agent": "reviewer", "message": (
                f"Scored draft v{state['draft_version']}: {overall:.1f}/10, keyword coverage {coverage:.0%}, "
                f"{len(evidence) - len(weak)}/{len(evidence)} claims traced to your resume, "
                f"{len(report.issues)} issues → {result.verdict}")}],
        }

    return reviewer_agent
