"""Streamlit front end.

Two backends:
* API mode (``API_URL`` set) — talks to the FastAPI service, as in Docker/Kubernetes.
* Embedded mode (no ``API_URL``) — runs the LangGraph service in-process, for
  single-container hosting such as Streamlit Community Cloud.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resume_optimizer.samples import SAMPLE_JOB, SAMPLE_RESUME  # noqa: E402


def _secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, os.getenv(name, default)))
    except Exception:  # noqa: BLE001 - no secrets.toml present
        return os.getenv(name, default)


API_URL = _secret("API_URL").rstrip("/")
API_KEY = _secret("API_AUTH_TOKEN")


class ApiBackend:
    label = "FastAPI service"

    def __init__(self, url: str, key: str):
        import requests

        self.http = requests.Session()
        self.url = url
        if key:
            self.http.headers["X-API-Key"] = key

    def _call(self, method: str, path: str, **kw) -> Dict[str, Any]:
        r = self.http.request(method, f"{self.url}{path}", timeout=600, **kw)
        if r.status_code >= 400:
            raise RuntimeError(r.json().get("detail", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text)
        return r.json()

    def start(self, payload):
        return self._call("POST", "/api/v1/sessions", json=payload)

    def feedback(self, sid, action, text):
        return self._call("POST", f"/api/v1/sessions/{sid}/feedback", json={"action": action, "feedback": text})

    def retry(self, sid):
        return self._call("POST", f"/api/v1/sessions/{sid}/retry")

    def extract(self, name, data):
        return self._call("POST", "/api/v1/extract-text", files={"file": (name, data)})["text"]


class EmbeddedBackend:
    LABELS = {"anthropic": "Claude", "gemini": "Gemini", "mock": "offline demo agents"}

    def __init__(self, provider: str):
        from dataclasses import replace

        from resume_optimizer.config import Settings
        from resume_optimizer.service import ResumeService

        if provider == "anthropic":
            os.environ["ANTHROPIC_API_KEY"] = _secret("ANTHROPIC_API_KEY")  # the Anthropic SDK reads it from the env
        settings = replace(
            Settings(),
            llm_provider=provider,
            gemini_api_key=_secret("GEMINI_API_KEY"),
            gemini_model=_secret("GEMINI_MODEL", Settings().gemini_model),
            anthropic_model=_secret("ANTHROPIC_MODEL", Settings().anthropic_model),
            vector_backend="memory",
            checkpoint_backend="memory",
        )
        self.svc = ResumeService(settings)
        self.label = f"in-process ({self.LABELS.get(provider, provider)})"

    def start(self, payload):
        from resume_optimizer.schemas import StartSessionRequest

        return self.svc.start(StartSessionRequest(**payload)).model_dump()

    def feedback(self, sid, action, text):
        from resume_optimizer.schemas import FeedbackRequest

        try:
            return self.svc.feedback(sid, FeedbackRequest(action=action, feedback=text)).model_dump()
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

    def retry(self, sid):
        try:
            return self.svc.retry(sid).model_dump()
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

    def extract(self, name, data):
        from resume_optimizer.textutils import file_to_text

        try:
            return file_to_text(name, data)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc


def embedded_provider() -> str:
    """Decided from the current secrets on every run, so adding or changing a key takes effect without a redeploy."""
    explicit = _secret("LLM_PROVIDER")
    if explicit:
        return explicit
    return "gemini" if _secret("GEMINI_API_KEY") else "anthropic" if _secret("ANTHROPIC_API_KEY") else "mock"


@st.cache_resource
def get_backend(provider: str, key_fingerprint: str):
    # Arguments are the cache key: a new provider or a rotated key builds a fresh backend.
    return ApiBackend(API_URL, API_KEY) if API_URL else EmbeddedBackend(provider)


def key_fingerprint() -> str:
    import hashlib

    material = "|".join(_secret(n) for n in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_MODEL", "ANTHROPIC_MODEL"))
    return hashlib.sha256(material.encode()).hexdigest()[:16]


# ------------------------------------------------------------------------ UI
st.set_page_config(page_title="Resume Optimizer", page_icon="📄", layout="wide")
backend = get_backend(embedded_provider(), key_fingerprint())
ss = st.session_state
ss.setdefault("session", None)
ss.setdefault("resume_text", SAMPLE_RESUME)
ss.setdefault("job_text", SAMPLE_JOB)

with st.sidebar:
    st.header("Settings")
    st.caption(f"Backend: {backend.label}")
    threshold = st.slider("Reviewer pass threshold", 5.0, 10.0, 8.0, 0.5)
    max_auto = st.slider("Max automatic revisions", 0, 5, 2)
    target_role = st.text_input("Target role (optional)")
    if st.button("Start over", use_container_width=True):
        ss.session = None
        st.rerun()
    st.divider()
    st.markdown(
        "**Agents**\n\n"
        "1. **Supervisor** routes work\n2. **Context** reads the job + retrieves guidance from Milvus\n"
        "3. **Writer** tailors the resume\n4. **Reviewer** scores it and sends it back if needed\n"
        "5. **You** approve or request changes"
    )

st.title("📄 Multi-Agent Resume Optimizer")


def run(fn, *args):
    ss.last_error = None
    try:
        with st.spinner("Agents are working…"):
            ss.session = fn(*args)
    except Exception as exc:  # noqa: BLE001 - surface any backend failure in the UI
        ss.last_error = f"Request failed: {exc}"


if ss.get("last_error"):
    st.error(ss.last_error)


if ss.session is None:
    left, right = st.columns(2)
    with left:
        st.subheader("Your resume")
        upload = st.file_uploader("Upload PDF / TXT / MD", type=["pdf", "txt", "md"])
        if upload is not None and ss.get("_uploaded") != upload.name:
            try:
                ss.resume_text = backend.extract(upload.name, upload.getvalue())
                ss._uploaded = upload.name
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not read file: {exc}")
        ss.resume_text = st.text_area("Resume text", ss.resume_text, height=420)
    with right:
        st.subheader("Job description")
        ss.job_text = st.text_area("Job description text", ss.job_text, height=500)
    if st.button("Optimize resume", type="primary", use_container_width=True):
        run(backend.start, {
            "resume_text": ss.resume_text, "job_description": ss.job_text, "target_role": target_role or None,
            "quality_threshold": threshold, "max_auto_revisions": max_auto,
        })
        st.rerun()
    st.stop()

s = ss.session
status_badge = {"awaiting_human": "🟡 Awaiting your review", "approved": "🟢 Approved",
                "failed": "🔴 Failed", "running": "🔵 Running"}[s["status"]]
review = s.get("review") or {}
c1, c2, c3, c4 = st.columns(4)
c1.metric("Status", status_badge)
c2.metric("Overall score", f"{review.get('overall', 0):.1f} / 10")
c3.metric("Keyword coverage", f"{review.get('keyword_coverage', 0):.0%}")
c4.metric("Writer passes", s.get("revision_count", 0))
if s.get("error"):
    st.error(s["error"])
if s["status"] == "failed" and st.button("🔁 Retry from the last completed step", type="primary"):
    run(backend.retry, s["session_id"])
    st.rerun()

tab_draft, tab_review, tab_context, tab_trace, tab_original = st.tabs(
    ["Draft", "Review", "Context", "Agent trace", "Original"])

with tab_draft:
    if s.get("draft_markdown"):
        st.markdown(s["draft_markdown"])
        with st.expander("What changed"):
            for change in s["draft"]["change_log"]:
                st.markdown(f"- {change}")
        st.download_button("Download markdown", s["draft_markdown"], file_name="optimized_resume.md")

with tab_review:
    if review:
        report = review["report"]
        for name, score in report["scores"].items():
            st.progress(min(score / 10, 1.0), text=f"{name.replace('_', ' ').title()}: {score:.1f}")
        if report["possible_fabrications"]:
            st.warning("Possible unsupported claims:\n\n" + "\n".join(f"- {f}" for f in report["possible_fabrications"]))
        if report["missing_keywords"]:
            st.info("Missing keywords: " + ", ".join(report["missing_keywords"]))
        for issue in report["issues"]:
            icon = {"high": "🔴", "medium": "🟠", "low": "🟢"}[issue["severity"]]
            st.markdown(f"{icon} **{issue['section']}** — {issue['problem']}  \n↳ _{issue['suggestion']}_")

with tab_context:
    if s.get("context"):
        ctx = s["context"]
        a, b = st.columns(2)
        a.markdown("**Must-have keywords**\n\n" + ", ".join(ctx["must_have_keywords"]))
        a.markdown("**Nice to have**\n\n" + ", ".join(ctx["nice_to_have_keywords"]))
        b.markdown("**Strengths found**\n\n" + "\n".join(f"- {x}" for x in ctx["candidate_strengths"]))
        b.markdown("**Gaps**\n\n" + "\n".join(f"- {x}" for x in ctx["gaps"]))
        st.markdown("**Writing guidance**\n\n" + "\n".join(f"- {x}" for x in ctx["writing_guidance"]))

with tab_trace:
    for event in s.get("history", []):
        st.markdown(f"`{event['agent']:>10}` {event['message']}")

with tab_original:
    st.text(ss.resume_text)

if s["status"] == "awaiting_human":
    st.divider()
    st.subheader("Your decision")
    feedback = st.text_area("Feedback for the Writer (required to request a revision)",
                            placeholder="e.g. Emphasise the RAG search project and shorten the summary")
    a, b = st.columns(2)
    if a.button("✅ Approve", type="primary", use_container_width=True):
        run(backend.feedback, s["session_id"], "approve", "")
        st.rerun()
    if b.button("✏️ Request revision", use_container_width=True, disabled=not feedback.strip()):
        run(backend.feedback, s["session_id"], "revise", feedback)
        st.rerun()
