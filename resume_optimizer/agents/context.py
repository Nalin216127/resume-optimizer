"""Context agent: understands the job and retrieves grounded writing guidance from Milvus."""
from __future__ import annotations

import json

from ..config import Settings
from ..llm import LLM
from ..schemas import ContextBrief
from ..state import ResumeState
from ..vectorstore import KnowledgeStore

SYSTEM = """You are the Context agent in a resume optimization team. You analyse a job description \
against a candidate's resume and produce a brief that the Writer agent will follow.

Extract the hard requirements and preferred skills using the job description's own wording, since \
applicant tracking systems match literal terms. Identify which requirements the resume already \
evidences and which it does not. Turn the retrieved best-practice notes into short, concrete guidance \
for this specific candidate. Base strengths only on what the resume actually says."""


def make_context_agent(llm: LLM, store: KnowledgeStore, settings: Settings):
    def context_agent(state: ResumeState) -> dict:
        query = f"{state.get('target_role') or ''}\n{state['job_description']}"
        k = settings.retrieval_top_k
        retrieved = store.search(query, k=k // 2, kind="guideline") + store.search(query, k=k - k // 2)
        retrieved = list({d["text"]: d for d in retrieved}.values())  # dedupe across the two queries

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
            "context": brief.model_dump(),
            "history": [{"agent": "context", "message": (
                f"Retrieved {len(retrieved)} knowledge entries; {len(brief.must_have_keywords)} must-have keywords, "
                f"{len(brief.gaps)} gaps")}],
        }

    return context_agent
