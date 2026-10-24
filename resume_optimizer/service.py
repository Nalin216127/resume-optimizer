"""Session service: the one place that drives the graph. Used by the API and by the UI's embedded mode."""
from __future__ import annotations

import logging
import uuid
from typing import Any, Callable, Dict, List, Optional

from langgraph.types import Command

from .config import Settings, get_settings
from .graph import build_checkpointer, build_graph
from .llm import LLM, LLMError, build_llm, check_providers
from .schemas import FeedbackRequest, KnowledgeDoc, SessionView, StartSessionRequest
from .vectorstore import KnowledgeStore, build_store

log = logging.getLogger(__name__)

# Upper bound on node executions per invoke; the supervisor's own budget normally ends far sooner.
RECURSION_LIMIT = 50


class SessionNotFound(KeyError):
    pass


# Called with (agent, message) as each agent finishes, so a UI can show live progress.
ProgressCallback = Callable[[str, str], None]


class ResumeService:
    def __init__(self, settings: Optional[Settings] = None, llm: Optional[LLM] = None,
                 store: Optional[KnowledgeStore] = None):
        self.settings = settings or get_settings()
        self.llm = llm or build_llm(self.settings)
        self.store = store or build_store(self.settings)
        self.graph = build_graph(self.llm, self.store, self.settings, build_checkpointer(self.settings))

    def _config(self, session_id: str) -> Dict[str, Any]:
        return {"configurable": {"thread_id": session_id}, "recursion_limit": RECURSION_LIMIT}

    def _run(self, session_id: str, graph_input: Any, on_event: Optional[ProgressCallback] = None) -> SessionView:
        try:
            for chunk in self.graph.stream(graph_input, self._config(session_id), stream_mode="updates"):
                if on_event is None:
                    continue
                for update in chunk.values():
                    for event in (update or {}).get("history", []) if isinstance(update, dict) else []:
                        on_event(event["agent"], event["message"])
        except LLMError as exc:
            log.warning("session %s failed: %s", session_id, exc)
            return self.get(session_id, error=str(exc))
        return self.get(session_id)

    def start(self, req: StartSessionRequest, on_event: Optional[ProgressCallback] = None) -> SessionView:
        session_id = req.session_id or uuid.uuid4().hex
        if req.session_id and self.graph.get_state(self._config(session_id)).values:
            raise ValueError("A session with this id already exists")
        initial = {
            "resume_text": req.resume_text,
            "job_description": req.job_description,
            "target_role": req.target_role,
            "quality_threshold": req.quality_threshold if req.quality_threshold is not None else self.settings.quality_threshold,
            "max_auto_revisions": req.max_auto_revisions if req.max_auto_revisions is not None else self.settings.max_auto_revisions,
            "draft_version": 0,
            "reviewed_version": 0,
            "auto_revisions": 0,
            "revision_count": 0,
            "approved": False,
            "revision_requested": False,
            "history": [],
        }
        return self._run(session_id, initial, on_event)

    def feedback(self, session_id: str, req: FeedbackRequest,
                 on_event: Optional[ProgressCallback] = None) -> SessionView:
        view = self.get(session_id)
        if view.status != "awaiting_human":
            raise ValueError(f"Session is {view.status}, not awaiting human feedback")
        if req.action == "revise" and not req.feedback.strip():
            raise ValueError("Feedback text is required to request a revision")
        return self._run(session_id, Command(resume=req.model_dump()), on_event)

    def retry(self, session_id: str, on_event: Optional[ProgressCallback] = None) -> SessionView:
        """Resume a failed run from the last completed agent; finished agents are not re-run."""
        view = self.get(session_id)
        if view.status != "failed":
            raise ValueError(f"Session is {view.status}; only failed sessions can be retried")
        return self._run(session_id, None, on_event)

    def get(self, session_id: str, error: Optional[str] = None) -> SessionView:
        snapshot = self.graph.get_state(self._config(session_id))
        values = snapshot.values
        if not values:
            raise SessionNotFound(session_id)
        waiting = any(task.interrupts for task in snapshot.tasks)
        # Runs are synchronous, so pending nodes with no interrupt mean the last run stopped on an error.
        stalled = bool(snapshot.next) and not waiting
        if error or stalled:
            status = "failed"
            error = error or "The last run stopped before finishing. Retry to continue from the last completed step."
        elif values.get("approved"):
            status = "approved"
        elif waiting:
            status = "awaiting_human"
        else:
            status = "running"
        return SessionView(
            session_id=session_id,
            status=status,
            revision_count=values.get("revision_count", 0),
            context=values.get("context"),
            draft=values.get("draft"),
            draft_markdown=values.get("draft_markdown"),
            review=values.get("review"),
            history=values.get("history", []),
            error=error,
        )

    def provider_status(self, every_model: bool = False) -> List[Dict[str, Any]]:
        return check_providers(self.llm, every_model)

    def add_knowledge(self, docs: List[KnowledgeDoc]) -> int:
        return self.store.add([d.model_dump() for d in docs])
