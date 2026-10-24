"""Design tokens, page CSS and small HTML components for the Streamlit UI.

Streamlit's own theme option is server-wide, so switching it would change the
theme for every visitor. Instead the base theme stays dark (see
.streamlit/config.toml) and each session injects CSS built from one of two
token sets. The toggle only swaps the token set, per visitor.

Every value that comes from a user or an LLM goes through ``esc`` before it is
placed in HTML.
"""
from __future__ import annotations

from html import escape
from typing import Dict, Iterable, List, Optional

TOKENS: Dict[str, Dict[str, str]] = {
    "dark": {
        "bg": "#0B0F19",
        "bg-glow": "rgba(99, 102, 241, 0.16)",
        "surface": "#121826",
        "surface-2": "#1A2233",
        "border": "#273049",
        "text": "#E6E9F2",
        "muted": "#9AA3B8",
        "accent": "#A5B4FC",          # text-on-dark accent (links, active tab)
        "button": "#4F46E5",          # filled buttons: white text stays ~6:1
        "button-hover": "#4338CA",
        "success": "#34D399",
        "warning": "#FBBF24",
        "danger": "#F87171",
        "chip": "rgba(165, 180, 252, 0.12)",
        "success-soft": "rgba(52, 211, 153, 0.12)",
        "warning-soft": "rgba(251, 191, 36, 0.12)",
        "danger-soft": "rgba(248, 113, 113, 0.12)",
        "shadow": "0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.25)",
        "toolbar-icon-filter": "none",
    },
    "light": {
        "bg": "#F5F6FA",
        "bg-glow": "rgba(79, 70, 229, 0.10)",
        "surface": "#FFFFFF",
        "surface-2": "#F0F2F8",
        "border": "#E2E6EF",
        "text": "#141A2A",
        "muted": "#5B6478",
        "accent": "#4F46E5",
        "button": "#4F46E5",
        "button-hover": "#4338CA",
        "success": "#047857",
        "warning": "#B45309",
        "danger": "#B91C1C",
        "chip": "rgba(79, 70, 229, 0.08)",
        "success-soft": "rgba(4, 120, 87, 0.08)",
        "warning-soft": "rgba(180, 83, 9, 0.08)",
        "danger-soft": "rgba(185, 28, 28, 0.07)",
        "shadow": "0 1px 2px rgba(20,26,42,.06), 0 8px 24px rgba(20,26,42,.06)",
        # Toolbar icons (Share/star/edit/GitHub) are images coloured for the dark base theme: darken them.
        "toolbar-icon-filter": "brightness(0) opacity(0.62)",
    },
}


def esc(value: object) -> str:
    return escape(str(value), quote=True)


def page_css(mode: str) -> str:
    t = TOKENS[mode]
    variables = "\n".join(f"  --ro-{k}: {v};" for k, v in t.items())
    return f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root {{
{variables}
}}

/* ---------- canvas ---------- */
[data-testid="stApp"] {{
  background: radial-gradient(1200px 500px at 15% -10%, var(--ro-bg-glow), transparent 60%), var(--ro-bg);
  color: var(--ro-text);
}}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stToolbarActionButtonIcon"] {{ filter: var(--ro-toolbar-icon-filter); }}
[data-testid="stToolbar"] [data-testid="stIconMaterial"], [data-testid="stToolbar"] button,
[data-testid="stToolbarActionButtonLabel"], [data-testid="stMainMenu"] svg {{ color: var(--ro-text) !important; }}
[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"] {{ color: var(--ro-muted) !important; }}
/* Keep content (and the theme toggle) below Streamlit's fixed header, which captures clicks. */
[data-testid="stMainBlockContainer"] {{ padding-top: 4.5rem; max-width: 1200px; }}
[data-testid="stApp"] p, [data-testid="stApp"] li, [data-testid="stApp"] label,
[data-testid="stApp"] h1, [data-testid="stApp"] h2, [data-testid="stApp"] h3, [data-testid="stApp"] h4,
[data-testid="stApp"] textarea, [data-testid="stApp"] input, [data-testid="stApp"] button {{
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}}
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li, [data-testid="stApp"] h1, [data-testid="stApp"] h2,
[data-testid="stApp"] h3, [data-testid="stWidgetLabel"] p {{ color: var(--ro-text); }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ color: var(--ro-muted); }}
[data-testid="stMarkdownContainer"] a {{ color: var(--ro-accent); }}
hr {{ border-color: var(--ro-border) !important; }}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] > div:first-child {{ background: var(--ro-surface); border-right: 1px solid var(--ro-border); }}
[data-testid="stSidebar"] * {{ color: var(--ro-text); }}

/* ---------- inputs ---------- */
[data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input {{
  background: var(--ro-surface-2) !important; color: var(--ro-text) !important;
  -webkit-text-fill-color: var(--ro-text); caret-color: var(--ro-accent);
  border-radius: 10px; font-size: 0.92rem; line-height: 1.5;
}}
/* Older Streamlit wraps inputs in data-baseweb elements, newer ones in *RootElement test ids: style both. */
[data-testid="stTextArea"] [data-baseweb="textarea"], [data-testid="stTextInput"] [data-baseweb="input"],
[data-testid="stTextArea"] [data-baseweb="textarea"] > div, [data-testid="stTextInput"] [data-baseweb="base-input"],
[data-testid="stTextAreaRootElement"], [data-testid="stTextInputRootElement"] {{
  border-color: var(--ro-border) !important; border-radius: 10px; background: var(--ro-surface-2) !important;
}}
[data-testid="stTextArea"] [data-baseweb="textarea"]:focus-within, [data-testid="stTextInput"] [data-baseweb="input"]:focus-within,
[data-testid="stTextAreaRootElement"]:focus-within, [data-testid="stTextInputRootElement"]:focus-within {{
  border-color: var(--ro-button) !important; box-shadow: 0 0 0 3px var(--ro-chip);
}}

/* ---------- theme selector (segmented control) ---------- */
[data-testid="stButtonGroup"], [data-testid="stButtonGroup"] > div {{
  display: flex; flex-wrap: nowrap !important; justify-content: flex-end; gap: 0;
}}
[data-testid="stButtonGroup"] button {{ padding: .3rem .7rem; min-height: 2.1rem; font-size: .85rem; }}
[data-testid="stButtonGroup"] button {{
  background: var(--ro-surface); border: 1px solid var(--ro-border); color: var(--ro-muted); font-weight: 600;
  white-space: nowrap;
}}
[data-testid="stButtonGroup"] button p {{ color: inherit !important; }}
[data-testid="stBaseButton-segmented_controlActive"] {{
  background: var(--ro-button) !important; border-color: var(--ro-button) !important; color: #fff !important;
}}

/* ---------- toggle ---------- */
[data-testid="stCheckbox"] label > div:first-of-type {{ background: var(--ro-border) !important; }}
[data-testid="stCheckbox"] label > div:first-of-type > div {{ background: #FFFFFF !important; box-shadow: 0 1px 3px rgba(0,0,0,.35); }}
[data-testid="stCheckbox"] label:has(input:checked) > div:first-of-type {{ background: var(--ro-button) !important; }}
[data-testid="stCheckbox"] label p {{ font-weight: 600; white-space: nowrap; }}
[data-testid="stTextArea"] textarea::placeholder {{ color: var(--ro-muted); -webkit-text-fill-color: var(--ro-muted); }}
[data-testid="stFileUploaderDropzone"] {{
  background: var(--ro-surface-2); border: 1.5px dashed var(--ro-border); border-radius: 12px;
}}
[data-testid="stFileUploaderDropzone"] * {{ color: var(--ro-muted); }}
/* Uploaded-file chip: stFileChip on newer Streamlit, stFileUploaderFile on older. */
[data-testid="stFileChip"], [data-testid="stFileUploaderFile"] {{
  background: var(--ro-surface-2) !important; border: 1px solid var(--ro-border); border-radius: 10px;
}}
[data-testid="stFileChip"] *, [data-testid="stFileUploaderFile"] * {{ color: var(--ro-text) !important; }}
[data-testid="stFileUploader"] [data-testid="stBaseButton-minimal"] {{ color: var(--ro-muted) !important; }}
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] * {{ color: var(--ro-text) !important; }}

/* ---------- buttons ---------- */
[data-testid="stBaseButton-primary"] {{
  background: var(--ro-button); border: 1px solid var(--ro-button); color: #fff; border-radius: 10px;
  font-weight: 600; box-shadow: 0 6px 18px rgba(79,70,229,.25); transition: transform .08s ease, background .15s ease;
}}
[data-testid="stBaseButton-primary"]:hover {{ background: var(--ro-button-hover); border-color: var(--ro-button-hover); color: #fff; }}
[data-testid="stBaseButton-primary"]:active {{ transform: translateY(1px); }}
[data-testid="stBaseButton-primary"] p {{ color: #fff !important; }}
[data-testid="stBaseButton-secondary"] {{
  background: var(--ro-surface); border: 1px solid var(--ro-border); color: var(--ro-text); border-radius: 10px; font-weight: 500;
}}
[data-testid="stBaseButton-secondary"]:hover {{ border-color: var(--ro-accent); color: var(--ro-accent); }}
[data-testid="stBaseButton-secondary"]:disabled {{ opacity: .45; }}
[data-testid="stDownloadButton"] button {{ border-radius: 10px; }}

/* ---------- containers ---------- */
.st-key-card_resume, .st-key-card_job, .st-key-paper, .st-key-original, .st-key-decision {{
  background: var(--ro-surface); border: 1px solid var(--ro-border) !important; border-radius: 16px;
  box-shadow: var(--ro-shadow); padding: 1.1rem 1.2rem 1.2rem;
}}
.st-key-decision {{ border-left: 4px solid var(--ro-button) !important; }}
.st-key-paper h1 {{ font-size: 1.6rem; font-weight: 800; letter-spacing: -.02em; margin-bottom: .2rem; }}
.st-key-paper h2 {{
  font-size: .78rem; text-transform: uppercase; letter-spacing: .09em; color: var(--ro-accent) !important;
  border-bottom: 1px solid var(--ro-border); padding-bottom: .35rem; margin-top: 1.2rem;
}}
.st-key-paper li {{ margin-bottom: .25rem; line-height: 1.55; }}
[data-testid="stExpander"] details {{ background: var(--ro-surface-2); border: 1px solid var(--ro-border); border-radius: 12px; }}
[data-testid="stExpander"] summary * {{ color: var(--ro-text); }}

/* ---------- tabs ---------- */
[data-baseweb="tab-list"] {{ gap: .25rem; border-bottom: 1px solid var(--ro-border); }}
button[role="tab"] {{ background: transparent; border-radius: 8px 8px 0 0; padding: .45rem .9rem; }}
button[role="tab"] p {{ color: var(--ro-muted); font-weight: 500; }}
button[role="tab"][aria-selected="true"] p {{ color: var(--ro-accent); font-weight: 600; }}
[data-baseweb="tab-highlight"] {{ background-color: var(--ro-accent); }}
[data-baseweb="tab-border"] {{ display: none; }}

/* ---------- live progress panel ---------- */
[data-testid="stExpander"] details[open] summary, [data-testid="stStatusWidget"] * {{ color: var(--ro-text); }}

/* ---------- alerts ---------- */
[data-testid="stAlert"] {{ border-radius: 12px; }}

/* ---------- custom components ---------- */
.ro-hero {{ display: flex; flex-direction: column; gap: .55rem; margin-bottom: 1.2rem; }}
.ro-eyebrow {{
  align-self: flex-start; font-size: .72rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
  color: var(--ro-accent); background: var(--ro-chip); border: 1px solid var(--ro-border);
  padding: .25rem .6rem; border-radius: 999px;
}}
.ro-title {{
  font-size: clamp(1.9rem, 3.6vw, 2.7rem); font-weight: 800; letter-spacing: -.03em; line-height: 1.1; margin: 0;
  color: var(--ro-text);
}}
.ro-title span {{
  background: linear-gradient(90deg, #6366F1, #22D3EE); -webkit-background-clip: text; background-clip: text; color: transparent;
}}
.ro-sub {{ color: var(--ro-muted); font-size: 1rem; max-width: 720px; margin: 0; }}
.ro-pipeline {{ display: flex; flex-wrap: wrap; align-items: center; gap: .4rem; margin-top: .35rem; }}
.ro-step {{
  display: inline-flex; align-items: center; gap: .4rem; font-size: .82rem; font-weight: 500;
  padding: .32rem .7rem; border-radius: 999px; border: 1px solid var(--ro-border);
  background: var(--ro-surface); color: var(--ro-muted);
}}
.ro-step.done {{ color: var(--ro-success); border-color: var(--ro-success); background: var(--ro-success-soft); }}
.ro-step.active {{ color: #fff; background: var(--ro-button); border-color: var(--ro-button); box-shadow: 0 0 0 4px var(--ro-chip); }}
.ro-arrow {{ color: var(--ro-muted); font-size: .8rem; }}

.ro-section {{ font-size: .78rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; color: var(--ro-muted); margin: .2rem 0 .5rem; }}
.ro-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: .8rem; margin: .4rem 0 1.1rem; }}
.ro-stat {{ background: var(--ro-surface); border: 1px solid var(--ro-border); border-radius: 14px; padding: .85rem 1rem; box-shadow: var(--ro-shadow); }}
.ro-stat .label {{ font-size: .75rem; color: var(--ro-muted); text-transform: uppercase; letter-spacing: .07em; font-weight: 600; }}
.ro-stat .value {{ font-size: 1.55rem; font-weight: 750; color: var(--ro-text); margin-top: .15rem; letter-spacing: -.02em; }}
.ro-stat .value small {{ font-size: .9rem; color: var(--ro-muted); font-weight: 500; }}
.ro-pill {{ display: inline-flex; align-items: center; gap: .35rem; font-size: .9rem; font-weight: 600; padding: .2rem .65rem; border-radius: 999px; margin-top: .3rem; }}
.ro-pill.success {{ color: var(--ro-success); background: var(--ro-success-soft); }}
.ro-pill.warning {{ color: var(--ro-warning); background: var(--ro-warning-soft); }}
.ro-pill.danger {{ color: var(--ro-danger); background: var(--ro-danger-soft); }}
.ro-pill.info {{ color: var(--ro-accent); background: var(--ro-chip); }}

.ro-bar {{ margin: .55rem 0 .85rem; }}
.ro-bar .row {{ display: flex; justify-content: space-between; font-size: .88rem; color: var(--ro-text); margin-bottom: .3rem; }}
.ro-bar .row b {{ font-weight: 650; }}
.ro-bar .track {{ height: 8px; border-radius: 999px; background: var(--ro-surface-2); overflow: hidden; border: 1px solid var(--ro-border); }}
.ro-bar .fill {{ height: 100%; border-radius: 999px; }}

.ro-chips {{ display: flex; flex-wrap: wrap; gap: .35rem; margin: .2rem 0 .9rem; }}
.ro-chip {{ font-size: .8rem; padding: .22rem .6rem; border-radius: 999px; background: var(--ro-chip); color: var(--ro-text); border: 1px solid var(--ro-border); }}
.ro-chip.success {{ background: var(--ro-success-soft); color: var(--ro-success); border-color: transparent; }}
.ro-chip.danger {{ background: var(--ro-danger-soft); color: var(--ro-danger); border-color: transparent; }}

.ro-card {{ background: var(--ro-surface); border: 1px solid var(--ro-border); border-radius: 12px; padding: .75rem .9rem; margin-bottom: .6rem; }}
.ro-card .head {{ display: flex; align-items: center; gap: .5rem; font-weight: 600; color: var(--ro-text); font-size: .92rem; }}
.ro-card .body {{ color: var(--ro-text); font-size: .9rem; margin-top: .3rem; }}
.ro-card .hint {{ color: var(--ro-muted); font-size: .86rem; margin-top: .25rem; }}
.ro-card.warning {{ border-color: var(--ro-warning); background: var(--ro-warning-soft); }}
.ro-sev {{ font-size: .68rem; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; padding: .12rem .45rem; border-radius: 6px; }}
.ro-sev.high {{ color: var(--ro-danger); background: var(--ro-danger-soft); }}
.ro-sev.medium {{ color: var(--ro-warning); background: var(--ro-warning-soft); }}
.ro-sev.low {{ color: var(--ro-success); background: var(--ro-success-soft); }}

.ro-timeline {{ position: relative; margin: .4rem 0 0 .4rem; padding-left: 1.2rem; border-left: 2px solid var(--ro-border); }}
.ro-event {{ position: relative; margin-bottom: .75rem; }}
.ro-event::before {{
  content: ""; position: absolute; left: -1.62rem; top: .3rem; width: .7rem; height: .7rem; border-radius: 50%;
  background: var(--dot); box-shadow: 0 0 0 3px var(--ro-bg);
}}
.ro-event .who {{ font-size: .72rem; font-weight: 700; text-transform: uppercase; letter-spacing: .07em; color: var(--dot); }}
.ro-event .what {{ font-size: .9rem; color: var(--ro-text); }}
.ro-backend {{ font-size: .8rem; color: var(--ro-muted); }}
.ro-backend b {{ color: var(--ro-accent); }}
</style>"""


# ---------------------------------------------------------------- components
STAGES = ["Supervisor", "Context", "Writer", "Reviewer", "You"]


def hero(status: Optional[str]) -> str:
    done: Iterable[str] = []
    active: Optional[str] = None
    if status == "awaiting_human":
        done, active = STAGES[:4], "You"
    elif status == "approved":
        done = STAGES
    elif status == "failed":
        done = STAGES[:1]
    steps = []
    for i, name in enumerate(STAGES):
        cls = "active" if name == active else "done" if name in done else ""
        icon = "✓ " if cls == "done" else ""
        steps.append(f'<span class="ro-step {cls}">{icon}{name}</span>')
        if i < len(STAGES) - 1:
            steps.append('<span class="ro-arrow">→</span>')
    return (
        '<div class="ro-hero">'
        '<span class="ro-eyebrow">LangGraph · Multi-agent · Human-in-the-loop</span>'
        '<h1 class="ro-title">Resume <span>Optimizer</span></h1>'
        '<p class="ro-sub">Five agents tailor your resume to a specific job. They extract what the role needs, '
        'rewrite without inventing facts, score the result, and wait for your approval.</p>'
        f'<div class="ro-pipeline">{"".join(steps)}</div>'
        "</div>"
    )


def score_tone(score: float) -> str:
    return "success" if score >= 8 else "warning" if score >= 6 else "danger"


def stats(items: List[Dict[str, str]]) -> str:
    cards = "".join(
        f'<div class="ro-stat"><div class="label">{esc(i["label"])}</div>{i["html"]}</div>' for i in items
    )
    return f'<div class="ro-stats">{cards}</div>'


def stat_value(value: str, suffix: str = "") -> str:
    tail = f" <small>{esc(suffix)}</small>" if suffix else ""
    return f'<div class="value">{esc(value)}{tail}</div>'


def pill(text: str, tone: str) -> str:
    return f'<span class="ro-pill {tone}">{esc(text)}</span>'


def bar(label: str, score: float) -> str:
    tone = score_tone(score)
    pct = max(0.0, min(score / 10, 1.0)) * 100
    return (
        f'<div class="ro-bar"><div class="row"><span>{esc(label)}</span><b>{score:.1f}</b></div>'
        f'<div class="track"><div class="fill" style="width:{pct:.0f}%;background:var(--ro-{tone})"></div></div></div>'
    )


def chips(values: Iterable[str], tone: str = "") -> str:
    items = "".join(f'<span class="ro-chip {tone}">{esc(v)}</span>' for v in values)
    return f'<div class="ro-chips">{items or "<span class=ro-chip>None</span>"}</div>'


def section(title: str) -> str:
    return f'<div class="ro-section">{esc(title)}</div>'


def issue_card(severity: str, area: str, problem: str, suggestion: str) -> str:
    return (
        f'<div class="ro-card"><div class="head"><span class="ro-sev {esc(severity)}">{esc(severity)}</span>{esc(area)}</div>'
        f'<div class="body">{esc(problem)}</div><div class="hint">↳ {esc(suggestion)}</div></div>'
    )


def warning_card(title: str, lines: Iterable[str]) -> str:
    body = "".join(f"<li>{esc(line)}</li>" for line in lines)
    return f'<div class="ro-card warning"><div class="head">⚠️ {esc(title)}</div><ul class="body">{body}</ul></div>'


AGENT_COLORS = {
    "supervisor": "#94A3B8", "context": "#22D3EE", "writer": "#818CF8", "reviewer": "#F59E0B", "human": "#34D399",
}


def timeline(events: Iterable[Dict[str, str]]) -> str:
    rows = "".join(
        f'<div class="ro-event" style="--dot:{AGENT_COLORS.get(e["agent"], "#94A3B8")}">'
        f'<div class="who">{esc(e["agent"])}</div><div class="what">{esc(e["message"])}</div></div>'
        for e in events
    )
    return f'<div class="ro-timeline">{rows}</div>'
