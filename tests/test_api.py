import pytest
from fastapi.testclient import TestClient

from resume_optimizer import api
from resume_optimizer.samples import SAMPLE_JOB, SAMPLE_RESUME


@pytest.fixture
def client():
    with TestClient(api.app) as c:
        yield c


def test_health_and_ready(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    ready = client.get("/readyz").json()
    assert ready["status"] == "ready" and ready["knowledge_docs"] > 0


def test_session_lifecycle(client):
    r = client.post("/api/v1/sessions", json={"resume_text": SAMPLE_RESUME, "job_description": SAMPLE_JOB})
    assert r.status_code == 200, r.text
    sid = r.json()["session_id"]
    assert r.json()["status"] == "awaiting_human"

    assert client.get(f"/api/v1/sessions/{sid}").json()["session_id"] == sid

    r = client.post(f"/api/v1/sessions/{sid}/feedback", json={"action": "revise", "feedback": "Shorter summary"})
    assert r.json()["status"] == "awaiting_human"

    r = client.post(f"/api/v1/sessions/{sid}/feedback", json={"action": "approve"})
    assert r.json()["status"] == "approved"

    r = client.post(f"/api/v1/sessions/{sid}/feedback", json={"action": "approve"})
    assert r.status_code == 409
    assert client.post(f"/api/v1/sessions/{sid}/retry").status_code == 409


def test_provider_status(client):
    rows = client.get("/api/v1/providers").json()
    assert rows == [{"provider": "offline demo agents", "ok": True, "seconds": rows[0]["seconds"], "detail": ""}]


def test_start_with_session_id(client):
    body = {"resume_text": SAMPLE_RESUME, "job_description": SAMPLE_JOB, "session_id": "b" * 32}
    assert client.post("/api/v1/sessions", json=body).json()["session_id"] == "b" * 32
    assert client.post("/api/v1/sessions", json=body).status_code == 409
    assert client.post("/api/v1/sessions", json={**body, "session_id": "not-hex"}).status_code == 422


def test_errors(client):
    assert client.get("/api/v1/sessions/missing").status_code == 404
    assert client.post("/api/v1/sessions", json={"resume_text": "short", "job_description": "short"}).status_code == 422


def test_knowledge_and_extract(client):
    r = client.post("/api/v1/knowledge", json={"documents": [
        {"title": "Security", "text": "Mention SOC 2 and threat modelling for security roles", "kind": "role_profile"}]})
    assert r.json() == {"inserted": 1}
    r = client.post("/api/v1/extract-text", files={"file": ("cv.txt", SAMPLE_RESUME.encode())})
    assert r.json()["text"].startswith("Alex Rivera")


def test_auth_enforced(client, monkeypatch):
    monkeypatch.setattr(api, "settings", api.settings.__class__(api_auth_token="secret"))
    assert client.get("/api/v1/sessions/x").status_code == 401
    assert client.get("/api/v1/sessions/x", headers={"X-API-Key": "secret"}).status_code == 404
