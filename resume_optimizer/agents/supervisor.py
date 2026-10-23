"""Supervisor: decides which agent acts next.

Routing is a deterministic policy over state rather than an LLM call. The
workflow's shape is fixed, so a rule-based supervisor is cheaper, auditable and
can't loop forever; the LLM effort goes into the agents that do the work.
"""
from __future__ import annotations

from typing import Tuple

from ..state import ResumeState


def _route(state: ResumeState) -> Tuple[str, str]:
    if state.get("approved"):
        return "end", "Resume approved by the human reviewer"
    if not state.get("context"):
        return "context", "No job context yet: analysing the job description and retrieving guidance"
    if not state.get("draft"):
        return "writer", "Context ready: producing the first draft"
    if state.get("revision_requested"):
        return "writer", "Human requested changes: revising the draft"
    if state.get("reviewed_version", 0) < state.get("draft_version", 0):
        return "reviewer", f"Draft v{state['draft_version']} has not been reviewed"

    review = state["review"]
    budget_left = state.get("auto_revisions", 0) < state.get("max_auto_revisions", 2)
    if review["verdict"] == "revise" and budget_left:
        return "writer", (
            f"Score {review['overall']:.1f} is below {state['quality_threshold']:.1f}: automatic revision "
            f"{state.get('auto_revisions', 0) + 1}/{state['max_auto_revisions']}"
        )
    why = "passed review" if review["verdict"] == "pass" else "auto-revision budget exhausted"
    return "human", f"Draft {why} (score {review['overall']:.1f}): waiting for human approval"


def supervisor(state: ResumeState) -> dict:
    nxt, reason = _route(state)
    return {"next_agent": nxt, "history": [{"agent": "supervisor", "message": f"→ {nxt}: {reason}"}]}
