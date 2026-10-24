"""Context agent: understands the job and retrieves grounded writing guidance from Milvus."""
from __future__ import annotations

import json

from ..config import Settings
from ..llm import LLM
from ..schemas import ContextBrief
from ..state import ResumeState
from ..retrieval import HybridRetriever

SYSTEM = """You are the Context agent in a resume optimization team. You analyse a job description \
against a candidate's resume and produce a brief that the Writer agent will follow.

Extract the hard requirements and preferred skills using the job description's own wording, since \
applicant tracking systems match literal terms. Identify which requirements the resume already \
evidences and which it does not. Turn the retrieved best-practice notes into short, concrete guidance \
for this specific candidate. Base strengths only on what the resume actually says."""


def detect_seniority(text: str) -> str:
    low = text.lower()
    if any(w in low for w in ("principal", "staff ", "lead ", "senior", "sr.", "head of")):
        return "senior"
    if any(w in low for w in ("junior", "graduate", "new grad", "entry level", "entry-level", "intern")):
        return "junior"
    return "mid"


def make_context_agent(llm: LLM, retriever: HybridRetriever, settings: Settings):
    def context_agent(state: ResumeState) -> dict:
        query = f"{state.get('target_role') or ''}\n{state['job_description']}"
        roles = retriever.detect_roles(query)
        seniority = detect_seniority(f"{state.get('target_role') or ''} {state['job_description'][:600]}")
        # The role's hiring profile, plus guidelines that fit this job and seniority (hybrid search).
        profiles = retriever.search(query, k=1, kind="role_profile", roles=roles or None)
        guidelines = retriever.search(query, k=max(settings.retrieval_top_k - 1, 3), kind="guideline",
                                      seniority=seniority)
        retrieved = profiles + guidelines

        payload = {
            "resume_text": state["resume_text"],
            "job_description": state["job_description"],
            "target_role": state.get("target_role"),
            "retrieved": retrieved,
        }
        prompt = (
            f"<job_description>\n{state['job_description']}\n</job_description>\n\n"
            f"<resume>\n{state['resume_text']}\n</resume>\n\n"
            f"<target_role>{state.get('target_role') or 'infer from the job description'}</target_role>\n\n"
            f"<retrieved_guidance>\n{json.dumps(retrieved, indent=2)}\n</retrieved_guidance>"
        )
        brief = llm.structured(task="context", system=SYSTEM, prompt=prompt, schema=ContextBrief,
                               payload=payload, effort=settings.context_effort)
        return {
            "retrieved": retrieved,
            "target_roles": roles,
            "seniority": seniority,
            "context": brief.model_dump(),
            "history": [{"agent": "context", "message": (
                f"Detected role {', '.join(roles) or 'unknown'} ({seniority}); retrieved {len(retrieved)} knowledge entries "
                f"(hybrid search); {len(brief.must_have_keywords)} must-have keywords, "
                f"{len(brief.gaps)} gaps")}],
        }

    return context_agent
