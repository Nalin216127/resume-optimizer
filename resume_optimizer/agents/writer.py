"""Writer agent: rewrites the resume against the context brief and any feedback."""
from __future__ import annotations

import json

from ..config import Settings
from ..llm import LLM
from ..schemas import ResumeDraft
from ..state import ResumeState
from ..textutils import render_markdown

SYSTEM = """You are the Writer agent in a resume optimization team. You rewrite a candidate's resume \
so it is tailored to a specific job, following the Context agent's brief.

Keep every fact true to the original resume: employers, titles, dates, metrics and skills must come \
from it. You may rephrase, reorder, merge, cut and sharpen, but never invent numbers or experience; a \
recruiter who checks a fabricated claim discards the whole application. Where the job wants something \
the resume lacks, leave it out rather than imply it.

Open bullets with strong action verbs, lead with outcomes, keep each bullet to one or two lines, and \
use the job description's exact terms where the candidate has the matching experience. When reviewer \
or human feedback is provided, address every point and say how in the change log. Human feedback takes \
priority over reviewer feedback."""


def make_writer_agent(llm: LLM, settings: Settings):
    def writer_agent(state: ResumeState) -> dict:
        # The latest human feedback stays in force for every pass of the round, including reviewer-driven ones.
        human_feedback = state.get("human_feedback")
        review = state.get("review") if state.get("draft") else None
        payload = {
            "resume_text": state["resume_text"],
            "context": state["context"],
            "previous_draft": state.get("draft"),
            "review": review,
            "human_feedback": human_feedback,
        }
        parts = [
            f"<original_resume>\n{state['resume_text']}\n</original_resume>",
            f"<context_brief>\n{json.dumps(state['context'], indent=2)}\n</context_brief>",
        ]
        if state.get("draft"):
            parts.append(f"<previous_draft>\n{state['draft_markdown']}\n</previous_draft>")
        if review:
            parts.append(f"<reviewer_feedback>\n{json.dumps(review['report'], indent=2)}\n</reviewer_feedback>")
        if human_feedback:
            parts.append(f"<human_feedback>\n{human_feedback}\n</human_feedback>")
        parts.append("Produce the next version of the tailored resume.")

        draft = llm.structured(task="writer", system=SYSTEM, prompt="\n\n".join(parts), schema=ResumeDraft,
                               payload=payload, effort=settings.writer_effort)
        version = state.get("draft_version", 0) + 1
        triggered_by_reviewer = bool(state.get("draft")) and not state.get("revision_requested")
        return {
            "draft": draft.model_dump(),
            "draft_markdown": render_markdown(draft),
            "draft_version": version,
            "revision_count": state.get("revision_count", 0) + 1,
            "auto_revisions": state.get("auto_revisions", 0) + (1 if triggered_by_reviewer else 0),
            "revision_requested": False,
            "history": [{"agent": "writer", "message": f"Wrote draft v{version}: " + "; ".join(draft.change_log[:3])}],
        }

    return writer_agent
