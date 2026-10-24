# Multi-Agent Resume Optimizer

[![Live demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nalin-resume-optimizer.streamlit.app)
[![CI](https://github.com/Nalin216127/resume-optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/Nalin216127/resume-optimizer/actions/workflows/ci.yml)

**Live demo:** https://nalin-resume-optimizer.streamlit.app. The demo runs in embedded mode, with an in-memory store instead of Milvus. It uses Gemini when `GEMINI_API_KEY` is set in its secrets, Claude when `ANTHROPIC_API_KEY` is set, and the offline demo agents otherwise.

A resume tailoring system built on **LangGraph**. A Supervisor coordinates Context, Writer and Reviewer agents, then pauses for a **human-in-the-loop** approval before anything is final. A **FastAPI** backend and a **Streamlit** UI sit on top. **Milvus** holds the retrieval knowledge base. The stack ships with **Docker Compose** and **Kubernetes** manifests.

```
                       ┌──────────────────────── Supervisor ────────────────────────┐
                       │  deterministic routing policy over graph state             │
                       └───┬───────────────┬───────────────┬───────────────┬────────┘
                           ▼               ▼               ▼               ▼
                       Context          Writer   ⇄     Reviewer          Human
                  JD analysis +     tailored draft   rubric scores +   interrupt():
                  Milvus retrieval  (no invented     keyword grounding approve / revise
                                     facts)          + fabrication gate
```

## How the workflow runs

1. **Supervisor** checks the state and picks the next agent. Routing is rule-based on purpose: the workflow has a fixed shape, so rules are auditable, cost nothing and can't loop forever.
2. **Context agent** pulls guidelines, example bullets and role profiles from Milvus. It then writes a `ContextBrief` with must-have and nice-to-have keywords, the candidate's strengths and gaps, and writing guidance.
3. **Writer agent** produces a structured `ResumeDraft` from the original resume, the brief, and any reviewer or human feedback. It isn't allowed to invent employers, dates, metrics or skills.
4. **Reviewer agent** scores the draft on five dimensions: keyword alignment, impact, clarity, ATS formatting and truthfulness. It also lists issues and possible fabrications. Its score is blended with a deterministic keyword-coverage measure. **Any suspected fabrication fails the draft, whatever the score.**
5. If the draft fails, the Supervisor sends it back to the Writer, up to `max_auto_revisions` times.
6. **Human gate**: LangGraph `interrupt()` pauses the thread and the checkpointer saves it. The user then either approves, or asks for a revision with feedback. A revision starts a new Writer → Reviewer round with a fresh auto-revision budget.

All agent outputs are Pydantic models produced through structured outputs, so they are validated before they reach graph state.

## LLM providers

Set `LLM_PROVIDER` to choose a provider:

| `LLM_PROVIDER` | Model | Notes |
|---|---|---|
| `anthropic` | Claude Opus 5.5 (`ANTHROPIC_MODEL`) | Uses `messages.parse` structured outputs, with a server-side refusal fallback |
| `gemini` | `gemini-flash-latest` (`GEMINI_MODEL`) | JSON-schema output; each agent's effort sets Gemini's thinking budget |
| `mock` | none | Deterministic offline heuristics, used by the tests and offline demos |

Gemini specifics:

- **Quota fallback.** Free-tier Gemini keys get about 20 requests per day per model, and one optimization uses 4–8 requests. When the main model's quota runs out, the backend switches to the models in `GEMINI_FALLBACK_MODELS` (default `gemini-flash-lite-latest`). It stops using the exhausted model until Google's reset time.
- **Rate limits.** If Google asks for a wait of 60 s or less, the backend waits and retries.
- **Timeouts.** Every request times out after `LLM_TIMEOUT_SECONDS` (default 180), so a dropped connection can't hang a session.

If an LLM call still fails, the session is marked `failed`. `POST /api/v1/sessions/{id}/retry` (or **Retry** in the UI) continues from the last completed agent, without re-running finished steps.

## Project layout

```
resume_optimizer/
  agents/        supervisor.py, context.py, writer.py, reviewer.py, human.py
  graph.py       StateGraph wiring + checkpointer factory (memory / sqlite / postgres)
  service.py     session lifecycle used by both the API and the UI
  api.py         FastAPI app
  llm.py         Claude backend + offline MockLLM
  vectorstore.py Milvus store, in-memory store, hashing embedder, seed knowledge
  seed.py        one-off seeding entrypoint (Kubernetes Job)
ui/streamlit_app.py
docker/          api.Dockerfile, ui.Dockerfile
docker-compose.yml
k8s/base, k8s/overlays/production
tests/
```

## Quick start (local, no Docker)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q
```

To run with a real model (save a key to `.env` without echoing it: `sh scripts/set_key.sh GEMINI_API_KEY`):

```bash
export ANTHROPIC_API_KEY=sk-ant-...      # or: export LLM_PROVIDER=gemini GEMINI_API_KEY=...
VECTOR_BACKEND=memory uvicorn resume_optimizer.api:app --port 8000
API_URL=http://localhost:8000 streamlit run ui/streamlit_app.py
```

Set `LLM_PROVIDER=mock` to run everything offline. The mock uses deterministic heuristic agents and needs no API key.

**Embedded mode:** start Streamlit without `API_URL` and the UI runs the agents in-process, with an in-memory vector store and checkpointer. This is the mode to use for single-container hosting.

## Docker Compose (Milvus + API + UI)

```bash
cp .env.example .env        # add ANTHROPIC_API_KEY, or set LLM_PROVIDER=mock
docker compose up --build
```

- UI: http://localhost:8501
- API docs: http://localhost:8000/docs
- Milvus: localhost:19530. It runs standalone with embedded etcd and local storage, so there are no separate etcd or MinIO containers.

Sessions are saved to SQLite on a volume, so paused sessions survive an API restart.

No Docker Desktop? On macOS, [Colima](https://github.com/abiosoft/colima) works:

```bash
brew install colima docker docker-compose kubectl
colima start --cpu 6 --memory 10 --kubernetes
```

## Kubernetes

```bash
kubectl apply -f k8s/base/namespace.yaml
kubectl -n resume-optimizer create secret generic resume-optimizer-secrets \
  --from-literal=ANTHROPIC_API_KEY=sk-ant-... \
  --from-literal=API_AUTH_TOKEN=$(openssl rand -hex 24) \
  --from-literal=POSTGRES_PASSWORD=<pw> \
  --from-literal=DATABASE_URL=postgresql://resume:<pw>@postgres:5432/resume
kubectl apply -k k8s/overlays/production
```

To try it on a local cluster (Colima/k3s, kind, minikube), use the `local` overlay. It runs the offline agents with one replica each and no ingress:

```bash
docker compose build && docker tag resume-optimizer-api ghcr.io/nalin216127/resume-optimizer-api:main \
  && docker tag resume-optimizer-ui ghcr.io/nalin216127/resume-optimizer-ui:main
kubectl apply -k k8s/overlays/local     # after creating the secret as above
kubectl -n resume-optimizer port-forward svc/ui 8501:8501
```

What the manifests include:

- **API**: Deployment with 2–10 replicas behind an HPA, plus a PodDisruptionBudget and topology spread. Startup, readiness and liveness probes are set. Pods run as non-root with a read-only root filesystem and all capabilities dropped.
- **Postgres** StatefulSet for the LangGraph checkpointer. Any replica can resume a session that is waiting on a human.
- **Milvus** standalone StatefulSet (embedded etcd, local storage, PVC). For a highly available setup, switch to the Milvus Helm chart or Operator in cluster mode.
- A **seed Job** loads the knowledge base once, so replicas don't race to seed it.
- **Ingress** with TLS (cert-manager) and a 600 s read timeout, because agent runs can take minutes.
- **NetworkPolicies**: only the API (and the seed Job) can reach the data stores.

Images live at `ghcr.io/nalin216127/resume-optimizer-{api,ui}`. CI builds and pushes them on every push to `main` (tag `main`), and on `v*` tags (semver tags).

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/sessions` | Start a run; returns once the draft is waiting for human review |
| GET | `/api/v1/sessions/{id}` | Current state: draft, review, context, agent trace |
| POST | `/api/v1/sessions/{id}/feedback` | `{"action": "approve"}` or `{"action": "revise", "feedback": "..."}` |
| POST | `/api/v1/sessions/{id}/retry` | Continue a failed run from the last completed agent |
| POST | `/api/v1/knowledge` | Add guidelines, example bullets or role profiles to Milvus |
| POST | `/api/v1/extract-text` | PDF / TXT / MD upload → plain text |
| GET | `/healthz`, `/readyz` | Probes |

If `API_AUTH_TOKEN` is set, every `/api` call needs an `X-API-Key` header.

## Configuration

See [`.env.example`](.env.example). The main settings:

- `LLM_PROVIDER`: `anthropic`, `gemini` or `mock`.
- `ANTHROPIC_MODEL`: defaults to `claude-opus-5-5`.
- `GEMINI_MODEL`, `GEMINI_FALLBACK_MODELS`: default to `gemini-flash-latest` and `gemini-flash-lite-latest`.
- `CONTEXT_EFFORT`, `WRITER_EFFORT`, `REVIEWER_EFFORT`: effort level per agent. Defaults are `low`, `high` and `medium`.
- `ANTHROPIC_REFUSAL_FALLBACK`: turns on server-side refusal fallbacks (Claude API only).
- `QUALITY_THRESHOLD`, `MAX_AUTO_REVISIONS`: the Reviewer's pass bar and the automatic revision budget.
- `CHECKPOINT_BACKEND`: `memory`, `sqlite` or `postgres`.
