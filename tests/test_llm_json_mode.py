"""The provider-side half of the strict JSON control contract.

The model is asked for JSON-only output so a control payload cannot drift into
prose, and the answer is parsed strictly either way. These tests pin both: the
request carries the JSON hint, and an endpoint that does not implement the hint
is retried *without* it rather than failing the step.
"""

import sys
import types as pytypes
from types import SimpleNamespace

import pytest

from furti_ai.brain import DeepSeekClient
from furti_ai.config import Settings


def make_settings(**overrides) -> Settings:
    defaults = {"deepseek_api_key": "test-key", "deepseek_model": "deepseek-test"}
    defaults.update(overrides)
    return Settings(**defaults)


class FakeCompletions:
    def __init__(self, responder):
        self.responder = responder

    def create(self, **kwargs):
        return self.responder(kwargs)


class FakeOpenAI:
    """Minimal stand-in for the OpenAI SDK surface the client uses."""

    def __init__(self, responder):
        self.chat = SimpleNamespace(completions=FakeCompletions(responder))


def install_fake_openai(monkeypatch, responder):
    calls: list[dict] = []

    def recording(kwargs):
        calls.append(kwargs)
        return responder(kwargs)

    module = pytypes.ModuleType("openai")
    module.OpenAI = lambda **_: FakeOpenAI(recording)
    monkeypatch.setitem(sys.modules, "openai", module)
    return calls


def completion(text: str) -> SimpleNamespace:
    message = SimpleNamespace(content=text, tool_calls=None)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def test_chat_text_asks_for_a_json_object(monkeypatch):
    calls = install_fake_openai(monkeypatch, lambda _kwargs: completion('{"ok": true}'))

    raw = DeepSeekClient(make_settings()).chat_text("system", "user", purpose="plan")

    assert raw == '{"ok": true}'
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert calls[0]["temperature"] == 0.0


def test_chat_vision_asks_for_a_json_object(monkeypatch):
    calls = install_fake_openai(monkeypatch, lambda _kwargs: completion('{"ok": true}'))

    DeepSeekClient(make_settings()).chat_vision("system", "user", "AAAA")

    assert calls[0]["response_format"] == {"type": "json_object"}


def test_a_rejected_json_mode_is_retried_without_it(monkeypatch):
    def responder(kwargs):
        if "response_format" in kwargs:
            raise RuntimeError("response_format is not supported by this model")
        return completion('{"ok": true}')

    calls = install_fake_openai(monkeypatch, responder)

    raw = DeepSeekClient(make_settings()).chat_text("system", "user", purpose="plan")

    assert raw == '{"ok": true}'
    assert len(calls) == 2
    assert "response_format" in calls[0]
    assert "response_format" not in calls[1]


def test_an_unrelated_failure_is_not_retried_for_nothing(monkeypatch):
    def responder(_kwargs):
        raise RuntimeError("connection reset by peer")

    calls = install_fake_openai(monkeypatch, responder)

    with pytest.raises(RuntimeError, match="connection reset"):
        DeepSeekClient(make_settings()).chat_text("system", "user")

    assert len(calls) == 1


def test_json_mode_can_be_switched_off(monkeypatch):
    calls = install_fake_openai(monkeypatch, lambda _kwargs: completion('{"ok": true}'))

    DeepSeekClient(make_settings(llm_json_mode=False)).chat_text("system", "user")

    assert "response_format" not in calls[0]
