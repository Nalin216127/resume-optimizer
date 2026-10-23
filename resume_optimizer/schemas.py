"""Structured contracts exchanged between agents and exposed by the API.

Every LLM call returns one of these Pydantic models, so agent outputs are
validated before they ever reach graph state.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------- agent outputs
class ContextBrief(BaseModel):
    """Produced by the Context agent: what the target job wants and how the candidate maps to it."""

    target_role: str = Field(description="Job title being targeted")
    seniority: str = Field(description="e.g. junior, mid, senior, staff, lead")
    must_have_keywords: List[str] = Field(description="Hard requirements / skills from the job description")
    nice_to_have_keywords: List[str] = Field(description="Preferred skills from the job description")
    candidate_strengths: List[str] = Field(description="Evidence in the resume that matches the job")
    gaps: List[str] = Field(description="Requirements with no supporting evidence in the resume")
    writing_guidance: List[str] = Field(description="Concrete rewrite guidance drawn from retrieved best practices")


class ResumeSection(BaseModel):
    title: str = Field(description="Section heading, e.g. 'Experience — Acme Corp, Senior Engineer (2021–2024)'")
    bullets: List[str]


class ResumeDraft(BaseModel):
    """Produced by the Writer agent."""

    headline: str = Field(description="One-line professional headline")
    summary: str = Field(description="2-3 sentence professional summary tailored to the role")
    sections: List[ResumeSection]
    skills: List[str]
    change_log: List[str] = Field(description="What changed versus the previous version and why")


class ReviewIssue(BaseModel):
    severity: Literal["high", "medium", "low"]
    section: str
    problem: str
    suggestion: str


class ReviewScores(BaseModel):
    keyword_alignment: float = Field(ge=0, le=10)
    impact_quantification: float = Field(ge=0, le=10)
    clarity_concision: float = Field(ge=0, le=10)
    ats_formatting: float = Field(ge=0, le=10)
    truthfulness: float = Field(ge=0, le=10, description="10 = every claim is supported by the original resume")


class ReviewReport(BaseModel):
    """Produced by the Reviewer agent."""

    scores: ReviewScores
    strengths: List[str]
    issues: List[ReviewIssue]
    missing_keywords: List[str]
    possible_fabrications: List[str] = Field(description="Claims in the draft not supported by the original resume")


class ReviewResult(BaseModel):
    """Reviewer output after deterministic grounding is blended in (stored in state)."""

    report: ReviewReport
    keyword_coverage: float
    overall: float
    verdict: Literal["pass", "revise"]


# ---------------------------------------------------------------- API contracts
class StartSessionRequest(BaseModel):
    resume_text: str = Field(min_length=50)
    job_description: str = Field(min_length=50)
    target_role: Optional[str] = None
    quality_threshold: Optional[float] = Field(default=None, ge=0, le=10)
    max_auto_revisions: Optional[int] = Field(default=None, ge=0, le=5)


class FeedbackRequest(BaseModel):
    action: Literal["approve", "revise"]
    feedback: str = ""


class KnowledgeDoc(BaseModel):
    title: str
    text: str = Field(min_length=10)
    kind: Literal["guideline", "example_bullet", "role_profile"] = "guideline"


class KnowledgeIngestRequest(BaseModel):
    documents: List[KnowledgeDoc] = Field(min_length=1)


class AgentEvent(BaseModel):
    agent: str
    message: str


class SessionView(BaseModel):
    session_id: str
    status: Literal["running", "awaiting_human", "approved", "failed"]
    revision_count: int = 0
    context: Optional[ContextBrief] = None
    draft: Optional[ResumeDraft] = None
    draft_markdown: Optional[str] = None
    review: Optional[ReviewResult] = None
    history: List[AgentEvent] = []
    error: Optional[str] = None
