"""Human-in-the-loop gate.

Execution pauses here via LangGraph's ``interrupt``; the checkpointer persists
the full state, and the API resumes the thread with the human's decision.
"""
from __future__ import annotations

from langgraph.types import interrupt

from ..state import ResumeState


def human_review(state: ResumeState) -> dict:
    decision = interrupt({
        "type": "approval",
        "draft_version": state["draft_version"],
        "overall": state["review"]["overall"],
        "prompt": "Approve this draft, or request a revision with feedback.",
    })
    action = (decision or {}).get("action")
    feedback = ((decision or {}).get("feedback") or "").strip()

    if action == "approve":
        return {"approved": True, "history": [{"agent": "human", "message": "Approved the draft"}]}
    if action == "revise" and feedback:
        return {
            "revision_requested": True,
            "human_feedback": feedback,
            "auto_revisions": 0,  # fresh auto-revision budget for the new round
            "history": [{"agent": "human", "message": f"Requested revision: {feedback}"}],
        }
    # Malformed or empty decision: stay at the gate rather than guessing.
    return {"history": [{"agent": "human", "message": "No valid decision received; awaiting input again"}]}
