"""Shared LangGraph state for the resume optimization workflow."""
from __future__ import annotations

import operator
from typing import Any, Dict, List, Optional

from typing_extensions import Annotated, TypedDict


class ResumeState(TypedDict, total=False):
    # Inputs
    resume_text: str
    job_description: str
    target_role: Optional[str]
    quality_threshold: float
    max_auto_revisions: int

    # Agent outputs (stored as plain dicts so checkpoints serialise cleanly)
    retrieved: List[Dict[str, Any]]       # knowledge used by the Context agent (role profile + guidelines)
    exemplars: List[Dict[str, Any]]       # example bullets retrieved for the Writer
    target_roles: List[str]               # detected role families, e.g. ["ml_ai"]
    seniority: str
    context: Dict[str, Any]
    draft: Dict[str, Any]
    draft_markdown: str
    review: Dict[str, Any]

    # Control flow
    draft_version: int        # incremented by the Writer
    reviewed_version: int     # draft_version the Reviewer last scored
    auto_revisions: int       # Writer passes triggered by the Reviewer since the last human turn
    revision_count: int       # total Writer passes
    human_feedback: Optional[str]
    revision_requested: bool  # human asked for changes, Writer has not yet run
    approved: bool
    next_agent: str

    # Append-only trace shown in the UI
    history: Annotated[List[Dict[str, str]], operator.add]
