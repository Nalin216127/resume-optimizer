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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import theme  # noqa: E402


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

    def start(self, payload, on_event=None):
        return self._call("POST", "/api/v1/sessions", json=payload)

    def feedback(self, sid, action, text, on_event=None):
        return self._call("POST", f"/api/v1/sessions/{sid}/feedback", json={"action": action, "feedback": text})

    def retry(self, sid, on_event=None):
        return self._call("POST", f"/api/v1/sessions/{sid}/retry")

    def provider_status(self, every_model=False):
        return self._call("GET", "/api/v1/providers", params={"every_model": str(every_model).lower()})

    def extract(self, name, data):
        return self._call("POST", "/api/v1/extract-text", files={"file": (name, data)})["text"]


class EmbeddedBackend:
    def __init__(self, provider: str):
        from dataclasses import replace

        from resume_optimizer.config import Settings
        from resume_optimizer.service import ResumeService

        base = Settings()

        def models(name: str, default):
            raw = _secret(name)
            return [m.strip() for m in raw.split(",") if m.strip()] if raw else default

        settings = replace(
            base,
            llm_provider=provider,
            gemini_api_key=_secret("GEMINI_API_KEY"),
            gemini_model=_secret("GEMINI_MODEL", base.gemini_model),
            gemini_fallback_models=models("GEMINI_FALLBACK_MODELS", base.gemini_fallback_models),
            openrouter_api_key=_secret("OPENROUTER_API_KEY"),
            openrouter_models=models("OPENROUTER_MODELS", base.openrouter_models),
            openai_compat_base_url=_secret("OPENAI_COMPAT_BASE_URL"),
            openai_compat_api_key=_secret("OPENAI_COMPAT_API_KEY"),
            openai_compat_models=models("OPENAI_COMPAT_MODELS", base.openai_compat_models),
            anthropic_api_key=_secret("ANTHROPIC_API_KEY"),
            anthropic_model=_secret("ANTHROPIC_MODEL", base.anthropic_model),
            vector_backend="memory",
            checkpoint_backend="memory",
        )
        self.svc = ResumeService(settings)
        self.label = f"{self.svc.llm.label}"
        self.embeddings = self.svc.store.embedder.label

    def start(self, payload, on_event=None):
        from resume_optimizer.schemas import StartSessionRequest

        return self.svc.start(StartSessionRequest(**payload), on_event).model_dump()

    def feedback(self, sid, action, text, on_event=None):
        from resume_optimizer.schemas import FeedbackRequest

        try:
            return self.svc.feedback(sid, FeedbackRequest(action=action, feedback=text), on_event).model_dump()
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

    def provider_status(self, every_model=False):
        return self.svc.provider_status(every_model)

    def retry(self, sid, on_event=None):
        try:
            return self.svc.retry(sid, on_event).model_dump()
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

    def extract(self, name, data):
        from resume_optimizer.textutils import file_to_text

        try:
            return file_to_text(name, data)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc


def embedded_provider() -> str:
    """LLM_PROVIDER from secrets, else "auto": every provider with a key, in failover order."""
    return _secret("LLM_PROVIDER") or "auto"


@st.cache_resource(show_spinner="Starting the agents…")
def get_backend(provider: str, key_fingerprint: str):
    # Arguments are the cache key: a new provider or a rotated key builds a fresh backend.
    return ApiBackend(API_URL, API_KEY) if API_URL else EmbeddedBackend(provider)


@st.cache_data(ttl=600, show_spinner=False)
def cached_provider_status(provider: str, key_fingerprint: str):
    # Cached for 10 minutes so repeated clicks by visitors can't burn the free quotas.
    return get_backend(provider, key_fingerprint).provider_status()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_model_status(provider: str, key_fingerprint: str):
    # One request per model, so cached for an hour.
    return get_backend(provider, key_fingerprint).provider_status(every_model=True)


def key_fingerprint() -> str:
    import hashlib

    names = ("GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_FALLBACK_MODELS", "OPENROUTER_API_KEY", "OPENROUTER_MODELS",
             "OPENAI_COMPAT_BASE_URL", "OPENAI_COMPAT_API_KEY", "OPENAI_COMPAT_MODELS", "ANTHROPIC_API_KEY",
             "ANTHROPIC_MODEL")
    material = "|".join(_secret(n) for n in names)
    return hashlib.sha256(material.encode()).hexdigest()[:16]


# ------------------------------------------------------------------------ UI
st.set_page_config(page_title="Resume Optimizer", page_icon="📄", layout="wide")
backend = get_backend(embedded_provider(), key_fingerprint())
ss = st.session_state
ss.setdefault("session", None)
ss.setdefault("resume_text", SAMPLE_RESUME)
ss.setdefault("job_text", SAMPLE_JOB)
if "theme_mode" not in ss:
    # ?theme=light in the URL survives reloads and shared links.
    ss.theme_mode = "light" if st.query_params.get("theme") == "light" else "dark"
THEME_OPTIONS = {"🌙 Dark": "dark", "☀️ Light": "light"}
if "theme_choice" in ss and ss.theme_choice is None:
    # The active option was clicked again and deselected: re-select the current mode.
    ss.theme_choice = next(k for k, v in THEME_OPTIONS.items() if v == ss.theme_mode)

s = ss.session
status = s["status"] if s else None

hero_col, toggle_col = st.columns([3, 1.4], vertical_alignment="top")
with toggle_col:
    # Two fixed options keep the widget identity stable (a relabelled widget is reset by newer
    # Streamlit) and always show which mode is active.
    choice = st.segmented_control(
        "Theme", list(THEME_OPTIONS), key="theme_choice", label_visibility="collapsed",
        default=next(k for k, v in THEME_OPTIONS.items() if v == ss.theme_mode),
    )
if choice is not None:
    ss.theme_mode = THEME_OPTIONS[choice]
else:
    st.rerun()  # re-highlight the current mode (see the deselect guard above)
st.query_params["theme"] = ss.theme_mode
st.markdown(theme.page_css(ss.theme_mode), unsafe_allow_html=True)
with hero_col:
    st.markdown(theme.hero(status), unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown(f'<div class="ro-backend">Models: <b>{theme.esc(backend.label)}</b></div>', unsafe_allow_html=True)
    if getattr(backend, "embeddings", None):
        st.markdown(f'<div class="ro-backend">Embeddings: <b>{theme.esc(backend.embeddings)}</b></div>',
                    unsafe_allow_html=True)
    st.divider()
    threshold = st.slider("Reviewer pass threshold", 5.0, 10.0, 8.0, 0.5,
                          help="Drafts scoring below this go back to the Writer")
    max_auto = st.slider("Max automatic revisions", 0, 5, 2,
                         help="Writer ⇄ Reviewer rounds before the draft comes to you")
    target_role = st.text_input("Target role (optional)", placeholder="e.g. Senior AI Engineer")
    with st.expander("Provider status"):
        st.caption("Check providers sends one tiny request per provider (cached 10 min). "
                   "Test every model sends one per model (cached 1 hour).")
        a, b = st.columns(2)
        if a.button("Check providers", use_container_width=True):
            with st.spinner("Checking…"):
                ss.provider_status = cached_provider_status(embedded_provider(), key_fingerprint())
        if b.button("Test every model", use_container_width=True):
            with st.spinner("Testing each model…"):
                ss.provider_status = cached_model_status(embedded_provider(), key_fingerprint())
        for row in ss.get("provider_status", []):
            mark = "✅" if row["ok"] else "❌"
            st.markdown(f"{mark} **{row['provider']}** · {row['seconds']}s")
            if row["detail"]:
                # One line per model; escape ':' so ":free" doesn't render as an emoji shortcode.
                for part in row["detail"].split("; "):
                    st.caption(part.replace(":", "\\:"))
    st.divider()
    if st.button("↺ Start over", use_container_width=True):
        ss.session = None
        st.rerun()


AGENT_LABELS = {"supervisor": "🧭 Supervisor", "context": "🎯 Context", "writer": "✍️ Writer",
                "reviewer": "🔍 Reviewer", "human": "🧑 You"}


def run(fn, *args):
    """Run a backend call, streaming each agent step into a status panel as it finishes."""
    ss.last_error = None
    status_box = st.status("Agents are working…", expanded=True)

    def on_event(agent: str, message: str) -> None:
        label = AGENT_LABELS.get(agent, agent)
        if agent != "supervisor":
            status_box.update(label=f"{label} finished. Working on the next step…")
        status_box.markdown(f"**{label}** · {message}")

    try:
        ss.session = fn(*args, on_event=on_event)
        failed = ss.session.get("status") == "failed"
        status_box.update(label="Stopped with an error" if failed else "Done", state="error" if failed else "complete",
                          expanded=False)
    except Exception as exc:  # noqa: BLE001 - surface any backend failure in the UI
        ss.last_error = f"Request failed: {exc}"
        status_box.update(label="Request failed", state="error")


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


if ss.get("last_error"):
    st.error(ss.last_error)

# ----------------------------------------------------------------- input page
if s is None:
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True, key="card_resume"):
            html(theme.section("📄 Your resume"))
            upload = st.file_uploader("Upload PDF / TXT / MD", type=["pdf", "txt", "md"])
            if upload is not None and ss.get("_uploaded") != upload.name:
                try:
                    ss.resume_text = backend.extract(upload.name, upload.getvalue())
                    ss._uploaded = upload.name
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not read file: {exc}")
            ss.resume_text = st.text_area("Or paste it here", ss.resume_text, height=400)
    with right:
        with st.container(border=True, key="card_job"):
            html(theme.section("🎯 Target job"))
            ss.job_text = st.text_area("Job description", ss.job_text, height=505)
    st.write("")
    if st.button("✨ Optimize my resume", type="primary", use_container_width=True):
        run(backend.start, {
            "resume_text": ss.resume_text, "job_description": ss.job_text, "target_role": target_role or None,
            "quality_threshold": threshold, "max_auto_revisions": max_auto,
        })
        st.rerun()
    st.stop()

# --------------------------------------------------------------- results page
review = s.get("review") or {}
status_pill = {
    "awaiting_human": theme.pill("● Your turn", "info"),
    "approved": theme.pill("✓ Approved", "success"),
    "failed": theme.pill("✕ Failed", "danger"),
    "running": theme.pill("● Running", "info"),
}[status]
overall = review.get("overall", 0.0)
html(theme.stats([
    {"label": "Status", "html": status_pill},
    {"label": "Overall score", "html": theme.stat_value(f"{overall:.1f}", "/ 10")},
    {"label": "Keyword coverage", "html": theme.stat_value(f"{review.get('keyword_coverage', 0):.0%}")},
    {"label": "Writer passes", "html": theme.stat_value(str(s.get("revision_count", 0)))},
]))
if s.get("error"):
    st.error(s["error"])
if status == "failed" and st.button("🔁 Retry from the last completed step", type="primary"):
    run(backend.retry, s["session_id"])
    st.rerun()

tab_draft, tab_review, tab_context, tab_trace, tab_original = st.tabs(
    ["📝 Draft", "📊 Review", "🎯 Context", "🧭 Agent trace", "📄 Original"])

with tab_draft:
    if s.get("draft_markdown"):
        with st.container(border=True, key="paper"):
            st.markdown(s["draft_markdown"])
        a, b = st.columns([3, 1], vertical_alignment="center")
        with a.expander("What changed in this version"):
            for change in s["draft"]["change_log"]:
                st.markdown(f"- {change}")
        b.download_button("⬇ Download .md", s["draft_markdown"], file_name="optimized_resume.md",
                          use_container_width=True)

with tab_review:
    if review:
        report = review["report"]
        left, right = st.columns([1, 1], gap="large")
        with left:
            html(theme.section("Rubric scores"))
            html("".join(theme.bar(name.replace("_", " ").title(), score) for name, score in report["scores"].items()))
        with right:
            if report["possible_fabrications"]:
                html(theme.warning_card("Possibly unsupported claims", report["possible_fabrications"]))
            html(theme.section("Missing keywords"))
            html(theme.chips(report["missing_keywords"], "danger"))
            if report["strengths"]:
                html(theme.section("Strengths"))
                html(theme.chips(report["strengths"], "success"))
        evidence = review.get("evidence") or []
        if evidence:
            weak = [e for e in evidence if not e["supported"]]
            html(theme.section(f"Evidence check · {len(evidence) - len(weak)} of {len(evidence)} claims "
                               "traced to your original resume"))
            html("".join(theme.evidence_card(e["claim"], e["best_match"], e["score"]) for e in weak)
                 or theme.chips(["Every claim matches something in your original resume"], "success"))
            with st.expander("All claims and their closest original line"):
                for e in evidence:
                    mark = "✅" if e["supported"] else "⚠️"
                    st.markdown(f"{mark} **{e['score']:.2f}** · {e['claim']}  \n↳ _{e['best_match']}_")
        html(theme.section(f"Issues ({len(report['issues'])})"))
        html("".join(theme.issue_card(i["severity"], i["section"], i["problem"], i["suggestion"])
                     for i in report["issues"]) or theme.chips(["No issues raised"], "success"))

with tab_context:
    if s.get("context"):
        ctx = s["context"]
        html(f'<p class="ro-sub">Targeting <b>{theme.esc(ctx["target_role"])}</b> · seniority '
             f'<b>{theme.esc(ctx["seniority"])}</b></p>')
        a, b = st.columns(2, gap="large")
        with a:
            html(theme.section("Must-have keywords") + theme.chips(ctx["must_have_keywords"]))
            html(theme.section("Nice to have") + theme.chips(ctx["nice_to_have_keywords"]))
        with b:
            html(theme.section("Evidence found in your resume") + theme.chips(ctx["candidate_strengths"], "success"))
            html(theme.section("Gaps") + theme.chips(ctx["gaps"], "danger"))
        html(theme.section("Writing guidance (retrieved from the knowledge base)"))
        for tip in ctx["writing_guidance"]:
            st.markdown(f"- {tip}")

with tab_trace:
    html(theme.timeline(s.get("history", [])))

with tab_original:
    with st.container(border=True, key="original"):
        st.text(ss.resume_text)

if status == "awaiting_human":
    st.write("")
    with st.container(border=True, key="decision"):
        html(theme.section("🧑‍⚖️ Your decision"))
        st.caption("Approve the draft as it is, or tell the Writer what to change. "
                   "A revision runs another Writer → Reviewer round.")
        feedback = st.text_area("Feedback for the Writer", label_visibility="collapsed",
                                placeholder="e.g. Lead with the RAG search project and keep the summary to two sentences")
        a, b = st.columns(2)
        if a.button("✅ Approve draft", type="primary", use_container_width=True):
            run(backend.feedback, s["session_id"], "approve", "")
            st.rerun()
        if b.button("✏️ Request revision", use_container_width=True, disabled=not feedback.strip()):
            run(backend.feedback, s["session_id"], "revise", feedback)
            st.rerun()
