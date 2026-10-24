"""Failover chain, OpenAI-compatible provider, provider selection, embeddings and the evidence check."""
from dataclasses import replace
from types import SimpleNamespace

import httpx
import openai
import pytest

from resume_optimizer import llm as llm_mod
from resume_optimizer.grounding import evidence_check
from resume_optimizer.llm import ChainLLM, GeminiLLM, LLMError, MockLLM, OpenAICompatLLM, build_llm, provider_names
from resume_optimizer.samples import SAMPLE_RESUME
from resume_optimizer.schemas import ReviewScores
from resume_optimizer.vectorstore import HashingEmbedder, build_embedder

GOOD = '{"keyword_alignment": 8, "impact_quantification": 7, "clarity_concision": 9, "ats_formatting": 9, "truthfulness": 10}'
_REQ = httpx.Request("POST", "https://example.test/v1/chat/completions")


def status_error(cls, code, headers=None):
    return cls("boom", response=httpx.Response(code, request=_REQ, headers=headers or {}), body=None)


def chat_reply(text=GOOD, finish="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish, message=SimpleNamespace(content=text))])


class FakeCompletions:
    def __init__(self, *outcomes):
        self.outcomes, self.calls = list(outcomes), []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def compat(*outcomes, models=("m1", "m2")):
    g = OpenAICompatLLM("OpenRouter", "https://example.test/v1", "key", list(models), 30)
    g.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(*outcomes)))
    return g


def call(g):
    return g.structured(task="reviewer", system="sys", prompt="p", schema=ReviewScores, payload={}, effort="low")


# ------------------------------------------------------------ OpenAI-compatible
def test_compat_parses_fenced_json_and_sends_schema():
    g = compat(chat_reply("```json\n" + GOOD + "\n```"))
    assert call(g).truthfulness == 10
    sent = g.client.chat.completions.calls[0]
    assert sent["response_format"] == {"type": "json_object"}
    assert '"truthfulness"' in sent["messages"][0]["content"]  # schema is in the system prompt


def test_compat_repairs_invalid_json_once():
    g = compat(chat_reply('{"keyword_alignment": 99}'), chat_reply(GOOD))
    assert call(g).clarity_concision == 9
    repair = g.client.chat.completions.calls[1]["messages"][-1]["content"]
    assert "does not match the schema" in repair


def test_compat_rate_limit_moves_to_next_model_and_cools_down():
    g = compat(status_error(openai.RateLimitError, 429, {"retry-after": "3600"}), chat_reply(), chat_reply())
    call(g)
    call(g)
    assert [c["model"] for c in g.client.chat.completions.calls] == ["m1", "m2", "m2"]


def test_compat_retries_without_json_mode_when_unsupported():
    g = compat(openai.BadRequestError("response_format not supported", response=httpx.Response(400, request=_REQ),
                                      body=None), chat_reply())
    call(g)
    assert "response_format" not in g.client.chat.completions.calls[1]


def test_compat_all_models_failing_raises():
    g = compat(status_error(openai.InternalServerError, 503), status_error(openai.NotFoundError, 404))
    with pytest.raises(LLMError, match="OpenRouter unavailable"):
        call(g)


# ------------------------------------------------------------------ the chain
class Failing:
    label = "Broken"

    def structured(self, *, task, **kw):
        raise LLMError(f"{task}: quota exhausted")


def test_chain_falls_through_to_next_provider():
    chain = ChainLLM([Failing(), MockLLM()])
    assert chain.label == "Broken → offline demo agents"
    out = chain.structured(task="context", system="", prompt="", schema=llm_mod.ContextBrief,
                           payload={"job_description": "Requirements\n- Python", "resume_text": "Python dev",
                                    "target_role": "Dev", "retrieved": []}, effort="low")
    assert out.target_role == "Dev"


def test_chain_reports_every_failure():
    with pytest.raises(LLMError, match="every model provider failed"):
        ChainLLM([Failing(), Failing()]).structured(task="writer", system="", prompt="", schema=ReviewScores,
                                                     payload={}, effort="low")


# ---------------------------------------------------------- provider selection
def test_auto_picks_providers_with_keys(settings):
    s = replace(settings, llm_provider="auto", gemini_api_key="g", openrouter_api_key="o", anthropic_api_key="")
    assert provider_names(s) == ["gemini", "openrouter"]
    assert provider_names(replace(s, gemini_api_key="", openrouter_api_key="")) == ["mock"]
    assert provider_names(replace(s, llm_provider="openrouter, gemini")) == ["openrouter", "gemini"]


def test_build_llm_makes_a_chain(settings):
    s = replace(settings, llm_provider="auto", gemini_api_key="g", openrouter_api_key="o")
    built = build_llm(s)
    assert isinstance(built, ChainLLM) and built.label == "Gemini → OpenRouter"
    assert isinstance(built.providers[0], GeminiLLM)


# ------------------------------------------------------------------ embeddings
def test_embedder_falls_back_when_gemini_fails(settings, monkeypatch):
    class Boom:
        def __init__(self, *_):
            raise RuntimeError("no network")

    monkeypatch.setattr("resume_optimizer.vectorstore.GeminiEmbedder", Boom)
    emb = build_embedder(replace(settings, embedding_backend="auto", gemini_api_key="g"))
    assert isinstance(emb, HashingEmbedder)


def test_evidence_check_flags_invented_claims():
    draft = {"summary": "", "sections": [{"title": "x", "bullets": [
        "Reduced nightly runtime from 6 hours to 2 hours by migrating batch jobs to Airflow",
        "Led a team of 12 engineers deploying Kubernetes clusters on GCP for 10 million users",
    ]}]}
    items = evidence_check(SAMPLE_RESUME, draft, HashingEmbedder())
    assert [i.supported for i in items] == [True, False]
    assert "Airflow" in items[0].best_match


def test_evidence_check_fails_closed_to_lexical():
    class Broken(HashingEmbedder):
        support_threshold = 0.99

        def embed_many(self, texts, task="x"):
            raise RuntimeError("API down")

    draft = {"summary": "", "sections": [{"title": "x", "bullets": ["Wrote unit tests and set up GitHub Actions CI for 12 services"]}]}
    items = evidence_check(SAMPLE_RESUME, draft, Broken())
    assert items and items[0].supported  # checked with the hashing fallback, not skipped



def test_check_providers_reports_each_link():
    from resume_optimizer.llm import check_providers

    rows = check_providers(ChainLLM([Failing(), MockLLM()]))
    assert [(r["provider"], r["ok"]) for r in rows] == [("Broken", False), ("offline demo agents", True)]
    assert "quota exhausted" in rows[0]["detail"]



def test_compat_error_includes_provider_reason():
    body = {"error": {"message": "No endpoints found matching your data policy", "code": 404}}
    err = openai.NotFoundError("404", response=httpx.Response(404, request=_REQ), body=body)
    g = compat(err, models=("m1",))
    with pytest.raises(LLMError, match="data policy"):
        call(g)



def test_every_model_check_reports_each_model():
    from resume_optimizer.llm import check_providers

    g = compat(status_error(openai.RateLimitError, 429), chat_reply('{"ok": true}'), models=("busy", "good"))
    rows = check_providers(g, every_model=True)
    assert [(r["provider"], r["ok"]) for r in rows] == [("OpenRouter / busy", False), ("OpenRouter / good", True)]
    assert "rate limited" in rows[0]["detail"]


def test_quick_check_names_the_model_that_answered():
    from resume_optimizer.llm import check_providers

    g = compat(status_error(openai.RateLimitError, 429), chat_reply('{"ok": true}'), models=("busy", "good"))
    rows = check_providers(g)
    assert rows == [{"provider": "OpenRouter", "ok": True, "seconds": rows[0]["seconds"], "detail": "answered by good"}]
