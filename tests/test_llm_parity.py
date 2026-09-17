"""Provider parity: the Gemini client must match the DeepSeek client's surface.

Gemini was the less-used path and drifted: no function calling, no JSON response
mode, no legacy-SDK support, and a different failure behaviour. These tests pin
the parity down with a fake ``google.genai`` SDK, so no network or API key is
involved.
"""

import sys
import types as pytypes
from types import SimpleNamespace

import pytest

from furti_ai.brain import (
    DeepSeekClient,
    GeminiClient,
    _plan_tool_schema,
    _strip_openapi_extras,
)
from furti_ai.config import Settings


# --------------------------------------------------------------- fake genai
class FakePart:
    @staticmethod
    def from_text(text):
        return {"kind": "text", "text": text}

    @staticmethod
    def from_bytes(data, mime_type):
        return {"kind": "bytes", "data": data, "mime_type": mime_type}


def make_types(schema_cls=None):
    namespace = SimpleNamespace()
    namespace.Part = FakePart
    namespace.FunctionDeclaration = lambda **kwargs: {"function_declaration": kwargs}
    namespace.Tool = lambda **kwargs: {"tool": kwargs}
    namespace.ToolConfig = lambda **kwargs: {"tool_config": kwargs}
    namespace.FunctionCallingConfig = lambda **kwargs: {"function_calling_config": kwargs}
    namespace.GenerateContentConfig = lambda **kwargs: {"config": kwargs}
    namespace.Schema = schema_cls
    return namespace


def install_modern_genai(monkeypatch, responder, schema_cls=None):
    """Register a fake `google.genai` (the modern SDK layout)."""
    calls: list[dict] = []

    class FakeModels:
        def generate_content(self, model, contents, config=None):
            calls.append({"model": model, "contents": contents, "config": config})
            return responder(model, contents, config)

    class FakeClient:
        def __init__(self, api_key=None, **kwargs):
            self.api_key = api_key
            self.models = FakeModels()

    google = pytypes.ModuleType("google")
    genai = pytypes.ModuleType("google.genai")
    genai.types = make_types(schema_cls)
    genai.Client = FakeClient
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    return calls


def install_legacy_genai(monkeypatch, responder, schema_cls=None):
    """Register a fake `google.generativeai` (the legacy SDK layout)."""
    calls: list[dict] = []

    class FakeGenerativeModel:
        def __init__(self, model):
            self.model = model

        def generate_content(self, contents, generation_config=None):
            calls.append(
                {"model": self.model, "contents": contents, "config": generation_config}
            )
            return responder(self.model, contents, generation_config)

    google = pytypes.ModuleType("google")  # deliberately has no `genai` attribute
    legacy = pytypes.ModuleType("google.generativeai")
    legacy.types = make_types(schema_cls)
    legacy.configure = lambda api_key=None: calls.append({"api_key": api_key})
    legacy.GenerativeModel = FakeGenerativeModel
    google.generativeai = legacy
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.generativeai", legacy)
    monkeypatch.delitem(sys.modules, "google.genai", raising=False)
    return calls


def response(
    text=None,
    function_calls=None,
    parts=None,
    prompt_tokens=120,
    candidate_tokens=30,
    thoughts=7,
):
    usage = SimpleNamespace(
        prompt_token_count=prompt_tokens,
        candidates_token_count=candidate_tokens,
        thoughts_token_count=thoughts,
    )
    candidates = (
        [SimpleNamespace(content=SimpleNamespace(parts=parts))] if parts else []
    )
    return SimpleNamespace(
        text=text, candidates=candidates, usage_metadata=usage, function_calls=function_calls
    )


def make_settings(**overrides) -> Settings:
    defaults = {"google_api_key": "test-key", "gemini_model": "gemini-test"}
    defaults.update(overrides)
    return Settings(**defaults)


PLAN = {
    "action": "click",
    "bbox": {"x": 1, "y": 2, "width": 30, "height": 40},
    "description": "click the button",
}


# ------------------------------------------------------------ function call
def test_gemini_requests_the_plan_tool_with_forced_mode(monkeypatch):
    calls = install_modern_genai(monkeypatch, lambda *_: response(function_calls=[SimpleNamespace(args=PLAN)]))
    client = GeminiClient(make_settings())

    plan = client.chat_with_vision("find the button", "AAAA")

    assert plan == PLAN
    config = calls[0]["config"]["config"]
    declaration = config["tools"][0]["tool"]["function_declarations"][0]
    assert declaration["function_declaration"]["name"] == "plan_action"
    calling = config["tool_config"]["tool_config"]["function_calling_config"]
    assert calling["function_calling_config"]["mode"] == "ANY"
    assert calling["function_calling_config"]["allowed_function_names"] == ["plan_action"]
    assert config["temperature"] == 0.0


def test_gemini_sends_the_screenshot_as_a_png_part(monkeypatch):
    calls = install_modern_genai(monkeypatch, lambda *_: response(function_calls=[SimpleNamespace(args=PLAN)]))
    client = GeminiClient(make_settings())

    client.chat_with_vision("prompt", "AAAA")

    kinds = [part["kind"] for part in calls[0]["contents"]]
    assert kinds == ["text", "text", "bytes"]
    assert calls[0]["contents"][2]["mime_type"] == "image/png"


def test_gemini_reads_a_function_call_from_candidate_parts(monkeypatch):
    parts = [SimpleNamespace(function_call=SimpleNamespace(args=PLAN))]
    install_modern_genai(monkeypatch, lambda *_: response(text="ignored", parts=parts))
    client = GeminiClient(make_settings())

    assert client.chat_with_vision("prompt", "AAAA") == PLAN


def test_function_calling_can_be_disabled(monkeypatch):
    calls = install_modern_genai(monkeypatch, lambda *_: response(text='{"a": 1}'))
    client = GeminiClient(make_settings(gemini_use_function_calling=False))

    assert client.chat_with_vision("prompt", "AAAA") == {"a": 1}
    # No tools requested, but JSON mode still asks for a parseable answer.
    assert "tools" not in calls[0]["config"]["config"]
    assert calls[0]["config"]["config"]["response_mime_type"] == "application/json"


def test_tool_rejection_falls_back_to_json_mode(monkeypatch):
    def responder(_model, _contents, config):
        if "tools" in config["config"]:
            raise RuntimeError("Tool calling is not supported for this model")
        return response(text='{"action": "click", "description": "fallback"}')

    calls = install_modern_genai(monkeypatch, responder)
    client = GeminiClient(make_settings())

    plan = client.chat_with_vision("prompt", "AAAA")

    assert plan["description"] == "fallback"
    assert len(calls) == 2


def test_unrelated_error_is_not_swallowed(monkeypatch):
    def responder(_model, _contents, config):
        if "tools" in config["config"]:
            raise RuntimeError("quota exceeded")
        return response(text='{"ok": true}')

    install_modern_genai(monkeypatch, responder)

    with pytest.raises(RuntimeError, match="quota exceeded"):
        GeminiClient(make_settings()).chat_with_vision("prompt", "AAAA")


def test_fenced_json_answer_is_parsed(monkeypatch):
    install_modern_genai(
        monkeypatch,
        lambda *_: response(text='```json\n{"action": "type", "text": "hi"}\n```'),
    )
    plan = GeminiClient(make_settings()).chat_with_vision("prompt", "AAAA")
    assert plan == {"action": "type", "text": "hi"}


def test_unparsable_answer_raises_a_clear_error(monkeypatch):
    install_modern_genai(monkeypatch, lambda *_: response(text="I cannot do that."))
    with pytest.raises(ValueError, match="no usable plan JSON"):
        GeminiClient(make_settings()).chat_with_vision("prompt", "AAAA")


def test_non_object_json_is_rejected(monkeypatch):
    install_modern_genai(monkeypatch, lambda *_: response(text="[1, 2, 3]"))
    with pytest.raises(ValueError):
        GeminiClient(make_settings()).chat_with_vision("prompt", "AAAA")


# ------------------------------------------------------------- text / vision
def test_gemini_chat_text_and_vision_match_the_deepseek_signatures(monkeypatch):
    install_modern_genai(monkeypatch, lambda *_: response(text="an answer"))

    client = GeminiClient(make_settings())

    assert client.chat_text("system", "user", purpose="plan") == "an answer"
    assert client.chat_vision("system", "user", "AAAA", purpose="review") == "an answer"
    # Same public surface as the DeepSeek client, which the planner relies on.
    for name in ("chat_with_vision", "chat_text", "chat_vision"):
        assert callable(getattr(client, name))
    assert client.call_count == 2


def test_gemini_vision_falls_back_to_text_only(monkeypatch):
    def responder(_model, contents, _config):
        if any(isinstance(part, dict) and part.get("kind") == "bytes" for part in contents):
            raise RuntimeError("image payload rejected")
        return response(text="text-only answer")

    calls = install_modern_genai(monkeypatch, responder)

    result = GeminiClient(make_settings()).chat_vision("system", "user", "AAAA")

    assert result == "text-only answer"
    assert len(calls) == 2
    assert all(
        part.get("kind") != "bytes" for part in calls[1]["contents"] if isinstance(part, dict)
    )


def test_gemini_usage_counts_thinking_tokens(monkeypatch):
    install_modern_genai(monkeypatch, lambda *_: response(text="x"))
    recorded = []
    client = GeminiClient(
        make_settings(),
        usage_callback=lambda model, prompt, completion, purpose: recorded.append(
            (model, prompt, completion, purpose)
        ),
    )

    client.chat_text("s", "u", purpose="plan")

    assert recorded == [("gemini-test", 120, 37, "plan")]


# --------------------------------------------------------------- legacy SDK
def test_legacy_sdk_is_supported(monkeypatch):
    calls = install_legacy_genai(monkeypatch, lambda *_: response(function_calls=[SimpleNamespace(args=PLAN)]))

    client = GeminiClient(make_settings())

    assert client._modern is False
    assert client.chat_with_vision("prompt", "AAAA") == PLAN
    # The legacy path configures the SDK with the key and builds a model.
    assert calls[0]["api_key"] == "test-key"
    assert calls[1]["model"] == "gemini-test"


def test_legacy_sdk_reads_function_calls_from_candidate_parts(monkeypatch):
    parts = [SimpleNamespace(function_call=SimpleNamespace(args={"action": "scroll"}))]
    install_legacy_genai(monkeypatch, lambda *_: response(parts=parts))

    plan = GeminiClient(make_settings()).chat_with_vision("prompt", "AAAA")

    assert plan == {"action": "scroll"}


def test_legacy_sdk_chat_text(monkeypatch):
    install_legacy_genai(monkeypatch, lambda *_: response(text="legacy answer"))
    client = GeminiClient(make_settings())

    assert client.chat_text("s", "u") == "legacy answer"
    assert client.call_count == 1


def test_missing_key_is_reported(monkeypatch):
    install_modern_genai(monkeypatch, lambda *_: response(text="x"))
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        GeminiClient(Settings(google_api_key=""))


# ------------------------------------------------------------------- schema
def test_gemini_schema_is_derived_from_the_openai_one(monkeypatch):
    seen = {}

    class FakeSchema:
        @classmethod
        def model_validate(cls, payload):
            seen.update(payload)
            return payload

    install_modern_genai(
        monkeypatch,
        lambda *_: response(function_calls=[SimpleNamespace(args=PLAN)]),
        schema_cls=FakeSchema,
    )

    GeminiClient(make_settings()).chat_with_vision("prompt", "AAAA")

    openai_schema = _plan_tool_schema()["function"]["parameters"]
    assert set(seen["properties"]) == set(openai_schema["properties"])
    assert seen["required"] == openai_schema["required"]
    # Gemini's Schema type rejects these, so they must be stripped.
    assert "additionalProperties" not in seen["properties"]["params"]


def test_openapi_extras_are_stripped_recursively():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"a": {"type": "string", "default": "x"}},
    }
    cleaned = _strip_openapi_extras(schema)
    assert "additionalProperties" not in cleaned
    assert "default" not in cleaned["properties"]["a"]


def test_schema_falls_back_to_a_plain_dict_without_a_schema_type(monkeypatch):
    calls = install_modern_genai(
        monkeypatch,
        lambda *_: response(function_calls=[SimpleNamespace(args=PLAN)]),
        schema_cls=None,
    )
    GeminiClient(make_settings()).chat_with_vision("prompt", "AAAA")
    declaration = calls[0]["config"]["config"]["tools"][0]["tool"]["function_declarations"][0]
    assert isinstance(declaration["function_declaration"]["parameters"], dict)


def test_deepseek_and_gemini_agree_on_the_tool_name():
    schema = _plan_tool_schema()
    assert schema["function"]["name"] == "plan_action"
    assert "additionalProperties" in schema["function"]["parameters"]["properties"]["params"]


def test_deepseek_client_still_exposes_the_same_surface():
    """Parity is symmetric: DeepSeek must not have lost anything either."""
    for name in ("chat_with_vision", "chat_text", "chat_vision"):
        assert callable(getattr(DeepSeekClient, name))
    for name in ("chat_with_vision", "chat_text", "chat_vision"):
        assert callable(getattr(GeminiClient, name))


def test_mock_client_matches_the_protocol():
    from furti_ai.brain import MockLLMClient

    mock = MockLLMClient(plan=PLAN)
    assert mock.chat_with_vision("p", "img")["action"] == "click"
    assert isinstance(mock.chat_text("s", "u"), str)
    assert isinstance(mock.chat_vision("s", "u", "img"), str)
    assert mock.call_count == 3
