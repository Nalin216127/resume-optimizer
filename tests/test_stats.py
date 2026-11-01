import re

import pytest
from fastapi.testclient import TestClient

from resume_optimizer import api, stats
from resume_optimizer.llm import LLMError, MockLLM
from resume_optimizer.samples import SAMPLE_JOB, SAMPLE_RESUME
from resume_optimizer.schemas import FeedbackRequest, StartSessionRequest
from resume_optimizer.service import ResumeService
from resume_optimizer.stats import OPTIMIZATION, VISIT, MemoryCounters, MilvusCounters, SiteCounters


def start(service, **kw):
    return service.start(StartSessionRequest(resume_text=SAMPLE_RESUME, job_description=SAMPLE_JOB, **kw))


def test_memory_counters_split_by_event_and_day(monkeypatch):
    counters = MemoryCounters()
    monkeypatch.setattr(stats, "_today", lambda: "2026-10-31")
    counters.record(VISIT)
    monkeypatch.setattr(stats, "_today", lambda: "2026-11-01")
    counters.record(VISIT)
    counters.record(VISIT)
    counters.record(OPTIMIZATION)
    s = counters.stats()
    assert (s.visits, s.visits_today, s.optimizations) == (3, 2, 1)
    assert "resets on restart" in s.storage


def test_counter_failures_never_raise():
    class Broken(SiteCounters):
        def _insert(self, event, day):
            raise ConnectionError("cluster asleep")

        def _count(self, event, day=None):
            raise ConnectionError("cluster asleep")

    Broken().record(VISIT)
    assert Broken().stats() is None


class FakeMilvus:
    """Just enough of MilvusClient: rows in a list, filters like event == "x" and day == "y"."""

    def __init__(self, count_star=True):
        self.collections, self.count_star = {}, count_star

    def has_collection(self, name):
        return name in self.collections

    def create_collection(self, collection_name, **kw):
        self.collections[collection_name] = []

    def insert(self, collection_name, data):
        self.collections[collection_name].extend(data)

    def query(self, collection_name, filter, output_fields, **kw):
        conds = dict(re.findall(r'(\w+) == "([^"]*)"', filter))
        rows = [r for r in self.collections[collection_name] if all(r.get(k) == v for k, v in conds.items())]
        if output_fields == ["count(*)"]:
            if not self.count_star:
                raise RuntimeError("count(*) not supported")
            return [{"count(*)": len(rows)}]
        return rows


@pytest.mark.parametrize("count_star", [True, False])
def test_milvus_counters(count_star):
    client = FakeMilvus(count_star)
    counters = MilvusCounters(client, "resume_knowledge_events", "Zilliz Cloud (Milvus)")
    assert client.has_collection("resume_knowledge_events")
    counters.record(VISIT)
    counters.record(OPTIMIZATION)
    counters.record(VISIT)
    s = counters.stats()
    assert (s.visits, s.visits_today, s.optimizations, s.storage) == (2, 2, 1, "Zilliz Cloud (Milvus)")
    # A second instance (an app restart) reuses the collection and keeps the counts.
    assert MilvusCounters(client, "resume_knowledge_events", "Zilliz Cloud (Milvus)").stats().visits == 2


def test_service_counts_each_optimized_session_once(service):
    view = start(service)
    service.feedback(view.session_id, FeedbackRequest(action="revise", feedback="Shorter summary"))
    service.feedback(view.session_id, FeedbackRequest(action="approve"))
    start(service)
    service.record_visit()
    s = service.site_stats()
    assert (s.visits, s.optimizations) == (1, 2)


def test_failed_start_counts_once_after_retry(settings):
    class FlakyReviewer(MockLLM):
        failed = False

        def structured(self, *, task, **kw):
            if task == "reviewer" and not self.failed:
                self.failed = True
                raise LLMError("reviewer: quota exhausted")
            return super().structured(task=task, **kw)

    svc = ResumeService(settings, llm=FlakyReviewer())
    view = start(svc, max_auto_revisions=0)
    assert view.status == "failed" and svc.site_stats().optimizations == 0
    svc.retry(view.session_id)
    assert svc.site_stats().optimizations == 1


def test_api_visits_and_stats():
    with TestClient(api.app) as client:
        assert client.get("/api/v1/stats").json()["visits"] == 0
        assert client.post("/api/v1/visits").json() == {"status": "recorded"}
        client.post("/api/v1/sessions", json={"resume_text": SAMPLE_RESUME, "job_description": SAMPLE_JOB})
        body = client.get("/api/v1/stats").json()
        assert (body["visits"], body["visits_today"], body["optimizations"]) == (1, 1, 1)
