"""Structured contracts exchanged between agents and exposed by the API.

Every LLM call returns one of these Pydantic models, so agent outputs are
validated before they ever reach graph state.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------- agent outputs
class ContextBrief(BaseModel):
    """Produced by the Context agent: what the target job wants and how the candidate maps to it."""

    target_role: str = Field(description="Job title being targeted")
    seniority: str = Field(description="e.g. junior, mid, senior, staff, lead")
    must_have_keywords: List[str] = Field(
        description="Hard requirements as short ATS keywords of 1-4 words in the job's own wording, e.g. 'LangGraph', 'Kubernetes', 'CI/CD'")
    nice_to_have_keywords: List[str] = Field(description="Preferred skills as short keywords of 1-4 words")
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


class HealthCheck(BaseModel):
    ok: bool


class ProviderStatus(BaseModel):
    provider: str
    ok: bool
    seconds: float
    detail: str = ""


class EvidenceItem(BaseModel):
    """One draft claim and the closest line of the original resume (embedding similarity)."""

    claim: str
    best_match: str
    score: float
    supported: bool


class ReviewResult(BaseModel):
    """Reviewer output after deterministic grounding is blended in (stored in state)."""

    report: ReviewReport
    keyword_coverage: float
    overall: float
    verdict: Literal["pass", "revise"]
    evidence: List[EvidenceItem] = []


# ---------------------------------------------------------------- API contracts
class StartSessionRequest(BaseModel):
    resume_text: str = Field(min_length=50)
    job_description: str = Field(min_length=50)
    target_role: Optional[str] = None
    # Optional client-chosen id, so a client that loses its connection mid-run can still find the session.
    session_id: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    quality_threshold: Optional[float] = Field(default=None, ge=0, le=10)
    max_auto_revisions: Optional[int] = Field(default=None, ge=0, le=5)


class FeedbackRequest(BaseModel):
    action: Literal["approve", "revise"]
    feedback: str = ""


class KnowledgeDoc(BaseModel):
    title: str
    text: str = Field(min_length=10)
    kind: Literal["guideline", "example_bullet", "role_profile"] = "guideline"
    id: Optional[str] = None
    roles: List[str] = Field(default_factory=lambda: ["any"], description="Role families, e.g. backend, ml_ai; 'any' for all")
    seniority: Literal["any", "junior", "mid", "senior"] = "any"
    skills: List[str] = Field(default_factory=list)


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
    # RAG sources: knowledge used by the Context agent and example bullets given to the Writer.
    retrieved: List[Dict[str, Any]] = []
    exemplars: List[Dict[str, Any]] = []
    target_roles: List[str] = []
    error: Optional[str] = None
