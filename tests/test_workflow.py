from dataclasses import replace

import pytest

from resume_optimizer.agents.supervisor import _route
from resume_optimizer.llm import LLMError, MockLLM
from resume_optimizer.samples import SAMPLE_JOB, SAMPLE_RESUME
from resume_optimizer.schemas import FeedbackRequest, StartSessionRequest
from resume_optimizer.service import ResumeService, SessionNotFound
from resume_optimizer.vectorstore import HashingEmbedder, InMemoryStore, SEED_KNOWLEDGE


def start(service, **kw):
    return service.start(StartSessionRequest(resume_text=SAMPLE_RESUME, job_description=SAMPLE_JOB, **kw))


def agents_in(view):
    return [e.agent for e in view.history]


def test_runs_until_human_gate(service):
    view = start(service)
    assert view.status == "awaiting_human"
    assert view.context and view.context.must_have_keywords
    assert view.draft and view.draft_markdown.startswith("# ")
    assert view.review and 0 <= view.review.overall <= 10
    order = [a for a in agents_in(view) if a != "supervisor"]
    assert order[:3] == ["context", "writer", "reviewer"]


def test_reviewer_loop_respects_auto_revision_budget(service):
    # An unreachable threshold forces every review to fail.
    view = start(service, quality_threshold=10.0, max_auto_revisions=2)
    assert view.status == "awaiting_human"
    assert view.review.verdict == "revise"
    assert view.revision_count == 3  # first draft + 2 automatic revisions
    assert agents_in(view).count("reviewer") == 3


def test_zero_budget_goes_straight_to_human(service):
    view = start(service, quality_threshold=10.0, max_auto_revisions=0)
    assert view.revision_count == 1
    assert view.status == "awaiting_human"


def test_human_revise_then_approve(service):
    view = start(service, max_auto_revisions=0)
    passes = view.revision_count

    view = service.feedback(view.session_id, FeedbackRequest(action="revise", feedback="Highlight the RAG project"))
    assert view.status == "awaiting_human"
    assert view.revision_count > passes
    assert any("Highlight the RAG project" in c for c in view.draft.change_log)

    view = service.feedback(view.session_id, FeedbackRequest(action="approve"))
    assert view.status == "approved"
    assert agents_in(view)[-2:] == ["human", "supervisor"]


def test_feedback_validation(service):
    view = start(service)
    with pytest.raises(ValueError):
        service.feedback(view.session_id, FeedbackRequest(action="revise", feedback="  "))
    service.feedback(view.session_id, FeedbackRequest(action="approve"))
    with pytest.raises(ValueError):  # already approved
        service.feedback(view.session_id, FeedbackRequest(action="approve"))


def test_unknown_session(service):
    with pytest.raises(SessionNotFound):
        service.get("nope")


def test_llm_failure_marks_session_failed(settings):
    class Failing(MockLLM):
        def structured(self, *, task, **kw):
            if task == "reviewer":
                raise LLMError("reviewer: the model declined this request")
            return super().structured(task=task, **kw)

    view = start(ResumeService(settings, llm=Failing()))
    assert view.status == "failed"
    assert "declined" in view.error


def test_fabricated_metric_blocks_pass(settings):
    class Fabricating(MockLLM):
        def _writer(self, p):
            draft = super()._writer(p)
            draft.sections[0].bullets.append("Grew revenue 900% for 5000000 users")
            return draft

    view = start(ResumeService(replace(settings, quality_threshold=0.0), llm=Fabricating()), max_auto_revisions=0)
    assert view.review.report.possible_fabrications
    assert view.review.verdict == "revise"


def test_supervisor_routing_table():
    base = {"quality_threshold": 8.0, "max_auto_revisions": 1, "draft_version": 1, "reviewed_version": 1}
    assert _route({})[0] == "context"
    assert _route({"context": {"x": 1}})[0] == "writer"
    assert _route({**base, "context": {1: 1}, "draft": {1: 1}, "reviewed_version": 0})[0] == "reviewer"
    reviewed = {**base, "context": {1: 1}, "draft": {1: 1}, "review": {"verdict": "revise", "overall": 5.0}}
    assert _route({**reviewed, "auto_revisions": 0})[0] == "writer"
    assert _route({**reviewed, "auto_revisions": 1})[0] == "human"
    assert _route({**reviewed, "revision_requested": True, "auto_revisions": 1})[0] == "writer"
    assert _route({**reviewed, "approved": True})[0] == "end"


def test_retrieval_is_relevant():
    store = InMemoryStore(HashingEmbedder(384))
    store.add(SEED_KNOWLEDGE)
    hits = store.search("Kubernetes Helm CI/CD deploy services", k=3)
    assert any("Kubernetes" in h["text"] for h in hits)
    assert all(h["kind"] == "guideline" for h in store.search("anything", k=3, kind="guideline"))


def test_failed_session_retries_from_last_completed_step(settings):
    class FlakyReviewer(MockLLM):
        def __init__(self):
            self.calls, self.fail_next_review = [], True

        def structured(self, *, task, **kw):
            self.calls.append(task)
            if task == "reviewer" and self.fail_next_review:
                self.fail_next_review = False
                raise LLMError("reviewer: Gemini quota exhausted")
            return super().structured(task=task, **kw)

    llm = FlakyReviewer()
    svc = ResumeService(settings, llm=llm)
    view = start(svc, max_auto_revisions=0)
    assert view.status == "failed"
    assert svc.get(view.session_id).status == "failed"  # still failed when re-read later

    view = svc.retry(view.session_id)
    assert view.status == "awaiting_human"
    assert llm.calls == ["context", "writer", "reviewer", "reviewer"]  # context/writer not re-run
    with pytest.raises(ValueError):
        svc.retry(view.session_id)


def test_keyword_coverage_tolerates_sentence_keywords():
    from resume_optimizer.textutils import keyword_coverage

    text = "Built production APIs with FastAPI in Python for 5 years; Docker and CI/CD with GitHub Actions."
    cov, missing = keyword_coverage(
        ["FastAPI", "Production APIs with FastAPI", "4+ years of software engineering experience in Python",
         "Docker and Kubernetes deployment experience", "LangGraph"], text)
    assert missing == ["Docker and Kubernetes deployment experience", "LangGraph"]
    assert cov == 0.6


def test_keyword_match_is_whole_word():
    from resume_optimizer.textutils import keyword_present

    assert not keyword_present("AI", "Maintained services")
    assert keyword_present("AI", "Built AI tooling")
    assert keyword_present("CI/CD", "Set up CI/CD with GitHub Actions")
    assert keyword_present("C++", "Wrote C++ services")
