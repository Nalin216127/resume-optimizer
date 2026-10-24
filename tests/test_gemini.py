"""GeminiLLM against a fake client: no network, no key needed."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
from google.genai import errors

from resume_optimizer import llm as llm_mod
from resume_optimizer.llm import GeminiLLM, LLMError, build_llm
from resume_optimizer.schemas import ReviewScores

GOOD = '{"keyword_alignment": 8, "impact_quantification": 7, "clarity_concision": 9, "ats_formatting": 9, "truthfulness": 10}'


def reply(text=GOOD, finish="STOP", block=None):
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name=finish))],
        prompt_feedback=SimpleNamespace(block_reason=block) if block else None,
    )


class FakeModels:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.models = []

    def generate_content(self, *, model, contents, config):
        self.calls.append(config)
        self.models.append(model)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def api_error(code, retry_delay=None):
    details = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay}] if retry_delay else []
    return errors.APIError(code, {"error": {"code": code, "message": "boom", "status": "X", "details": details}})


@pytest.fixture
def gemini(settings, monkeypatch):
    monkeypatch.setattr(llm_mod.time, "sleep", lambda _: None)

    def make(*outcomes, fallbacks=()):
        g = GeminiLLM(replace(settings, gemini_api_key="test-key", gemini_model="main",
                              gemini_fallback_models=list(fallbacks)))
        g.client = SimpleNamespace(models=FakeModels(*outcomes))
        return g

    return make


def call(g):
    return g.structured(task="reviewer", system="sys", prompt="p", schema=ReviewScores, payload={}, effort="low")


def test_parses_structured_output_and_sends_schema(gemini):
    g = gemini(reply())
    assert call(g).truthfulness == 10
    config = g.client.models.calls[0]
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema["properties"]["truthfulness"]["maximum"] == 10
    assert config.system_instruction == "sys"


def test_retries_rate_limits_then_succeeds(gemini):
    g = gemini(api_error(429), api_error(503), reply())
    assert call(g).clarity_concision == 9
    assert len(g.client.models.calls) == 3


@pytest.mark.parametrize("outcome,message", [
    (api_error(403), "check GEMINI_API_KEY"),
    (reply(finish="MAX_TOKENS"), "max_output_tokens"),
    (reply(finish="SAFETY"), r"stopped early \(SAFETY\)"),
    (reply(block="PROHIBITED_CONTENT"), "blocked the prompt"),
    (reply(text='{"keyword_alignment": 99}'), "does not match the schema"),
])
def test_failures_become_llm_errors(gemini, outcome, message):
    with pytest.raises(LLMError, match=message):
        call(gemini(outcome))


def test_retries_timeouts(gemini):
    import httpx

    g = gemini(httpx.ReadTimeout("slow"), reply())
    assert call(g).ats_formatting == 9
    with pytest.raises(LLMError, match="timed out or connection lost"):
        call(gemini(*[httpx.ConnectTimeout("down")] * 4))


def test_client_has_timeout(settings):
    g = GeminiLLM(replace(settings, gemini_api_key="test-key", llm_timeout_seconds=42))
    assert g.client._api_client._http_options.timeout == 42_000


def test_gives_up_after_repeated_rate_limits(gemini):
    with pytest.raises(LLMError, match="quota exhausted"):
        call(gemini(*[api_error(429)] * 4))


def test_daily_quota_falls_back_to_next_model_and_remembers(gemini):
    g = gemini(api_error(429, "6982s"), reply(), reply(), fallbacks=["lite"])
    assert call(g).truthfulness == 10
    assert call(g).truthfulness == 10
    # Primary hit once, then skipped while its quota is exhausted.
    assert g.client.models.models == ["main", "lite", "lite"]


def test_short_retry_delay_is_waited_out_on_same_model(gemini, monkeypatch):
    waits = []
    monkeypatch.setattr(llm_mod.time, "sleep", waits.append)
    g = gemini(api_error(429, "20s"), reply(), fallbacks=["lite"])
    call(g)
    assert waits == [20.0] and g.client.models.models == ["main", "main"]


def test_all_models_exhausted_reports_reset_time(gemini):
    g = gemini(api_error(429, "6982s"), api_error(429, "7000s"), fallbacks=["lite"])
    with pytest.raises(LLMError, match="Resets in about 1h 56m"):
        call(g)


def test_unavailable_model(gemini):
    with pytest.raises(LLMError, match="unavailable to this key"):
        call(gemini(api_error(404)))


@pytest.mark.parametrize("effort,budget", [("low", 512), ("medium", 2048), ("high", 6144)])
def test_effort_maps_to_thinking_budget(gemini, effort, budget):
    g = gemini(reply())
    g.structured(task="t", system="s", prompt="p", schema=ReviewScores, payload={}, effort=effort)
    config = g.client.models.calls[0]
    assert (config.thinking_config.thinking_budget if config.thinking_config else None) == budget


def test_requires_key(settings):
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        build_llm(replace(settings, llm_provider="gemini", gemini_api_key=""))


def test_persistent_overload_falls_back_to_next_model(gemini):
    g = gemini(*[api_error(503)] * 4, reply(), fallbacks=["lite"])
    assert call(g).truthfulness == 10
    assert g.client.models.models == ["main"] * 4 + ["lite"]


def test_overload_everywhere_says_retry(gemini):
    g = gemini(*[api_error(503)] * 8, fallbacks=["lite"])
    with pytest.raises(LLMError, match="overloaded right now"):
        call(g)
