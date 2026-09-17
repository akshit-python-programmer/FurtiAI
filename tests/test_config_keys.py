"""Local API keys: the ignored ``keys.json`` file and its two DeepSeek slots.

Keys must never be committed, so these tests use temporary files and fabricated
values only. The second DeepSeek key exists so the reviewer can run on its own
client, its own rate limit and its own model while the primary key plans.
"""

import json
from pathlib import Path

import furti_ai.orchestrator as orchestrator
from furti_ai.config import Settings, _local_key


def write_keys(path: Path, **values) -> Path:
    path.write_text(json.dumps(values), encoding="utf-8")
    return path


def clear_key_env(monkeypatch) -> None:
    for name in ("DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY_2", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)


# ------------------------------------------------------------- file loading
def test_both_deepseek_keys_load_from_the_local_file(tmp_path, monkeypatch):
    clear_key_env(monkeypatch)
    monkeypatch.setenv(
        "FURTI_KEYS_FILE",
        str(
            write_keys(
                tmp_path / "keys.json",
                deepseek_api_key="primary-key",
                deepseek_api_key_2="secondary-key",
                google_api_key="google-key",
            )
        ),
    )

    settings = Settings()

    assert settings.deepseek_api_key == "primary-key"
    assert settings.deepseek_api_key_2 == "secondary-key"
    assert settings.google_api_key == "google-key"


def test_environment_keys_override_the_file(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "FURTI_KEYS_FILE",
        str(
            write_keys(
                tmp_path / "keys.json",
                deepseek_api_key="from-file",
                deepseek_api_key_2="secondary-from-file",
            )
        ),
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "from-environment")

    settings = Settings()

    assert settings.deepseek_api_key == "from-environment"
    # The second slot is independent, so it still comes from the file.
    assert settings.deepseek_api_key_2 == "secondary-from-file"


def test_a_missing_or_broken_key_file_yields_empty_keys(tmp_path, monkeypatch):
    clear_key_env(monkeypatch)
    monkeypatch.setenv("FURTI_KEYS_FILE", str(tmp_path / "absent.json"))
    settings = Settings()
    assert settings.deepseek_api_key == ""
    assert settings.deepseek_api_key_2 == ""

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("FURTI_KEYS_FILE", str(broken))
    assert Settings().deepseek_api_key_2 == ""


def test_an_explicit_key_file_is_authoritative(tmp_path, monkeypatch):
    """A bad FURTI_KEYS_FILE must not silently read some other key file."""
    clear_key_env(monkeypatch)
    decoy = write_keys(tmp_path / "decoy.json", deepseek_api_key_2="decoy-key")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "keys.json").write_text(decoy.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv("FURTI_KEYS_FILE", str(tmp_path / "missing.json"))

    assert Settings().deepseek_api_key_2 == ""


def test_local_key_ignores_blank_values(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "FURTI_KEYS_FILE",
        str(write_keys(tmp_path / "keys.json", deepseek_api_key_2="   ")),
    )

    assert _local_key("deepseek_api_key_2") == ""


# --------------------------------------------------- secondary client wiring
class RecordingClient:
    """Records how a DeepSeek client was constructed."""

    def __init__(
        self,
        settings=None,
        model=None,
        usage_callback=None,
        api_key=None,
        thinking=None,
    ):
        self.settings = settings
        self.api_key = api_key
        self.thinking = thinking
        self._model = model


def make_secondary(monkeypatch, settings):
    monkeypatch.setattr(orchestrator, "DeepSeekClient", RecordingClient)
    return orchestrator.build_secondary_llm(settings, primary_model="deepseek-flash")


def test_secondary_client_uses_the_second_key_on_the_fast_model(monkeypatch):
    """The reviewer runs after every step, so it must stay on the fast model."""
    clear_key_env(monkeypatch)
    settings = Settings(
        deepseek_api_key="primary-key",
        deepseek_api_key_2="secondary-key",
        deepseek_model="deepseek-flash",
    )

    llm, provider = make_secondary(monkeypatch, settings)

    assert provider == orchestrator.SECONDARY_DEEPSEEK_PROVIDER
    assert llm.api_key == "secondary-key"
    assert llm._model == "deepseek-flash"
    assert llm.thinking is False


def test_secondary_client_falls_back_to_the_primary_key(monkeypatch):
    """One DeepSeek key must keep working, just without the second rate limit."""
    clear_key_env(monkeypatch)
    settings = Settings(
        deepseek_api_key="primary-key",
        deepseek_api_key_2="",
        cross_verify_provider="deepseek",
    )

    llm, provider = make_secondary(monkeypatch, settings)

    assert provider == "deepseek"
    assert llm.api_key == "primary-key"


def test_secondary_client_is_absent_without_any_deepseek_key(monkeypatch):
    clear_key_env(monkeypatch)
    settings = Settings(
        deepseek_api_key="",
        deepseek_api_key_2="",
        cross_verify_provider="deepseek",
    )

    llm, provider = make_secondary(monkeypatch, settings)

    assert llm is None
    assert provider == ""


# ----------------------------------------------- escalation = flash + thinking
def test_escalation_defaults_to_flash_with_thinking(monkeypatch):
    """Deep thinking is a mode of the fast model, not a slower model."""
    monkeypatch.delenv("DEEPSEEK_REASONING_MODEL", raising=False)
    monkeypatch.delenv("FURTI_DEEPSEEK_ESCALATION_THINKING", raising=False)

    settings = Settings(deepseek_api_key="primary-key")

    assert settings.deepseek_reasoning_model == "deepseek-flash"
    assert settings.deepseek_escalation_thinking is True


def test_escalation_client_is_built_with_thinking_enabled(monkeypatch):
    monkeypatch.setattr(orchestrator, "DeepSeekClient", RecordingClient)
    settings = Settings(
        deepseek_api_key="primary-key",
        deepseek_model="deepseek-flash",
        deepseek_reasoning_model="deepseek-flash",
    )

    client = orchestrator._make_llm(
        settings, settings.deepseek_reasoning_model, None, thinking=True
    )

    assert client._model == "deepseek-flash"
    assert client.thinking is True
