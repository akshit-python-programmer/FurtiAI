from dataclasses import fields
from pathlib import Path

from app import build_settings, create_parser, settings_from_args
from furti_ai.config import Settings
from furti_ai.gui import FurtiApp


def test_app_parser_without_task_resolves_gui_settings() -> None:
    args = create_parser().parse_args([])
    settings = settings_from_args(args)

    assert settings.llm_provider in {"auto", "deepseek", "gemini"}
    assert settings.cursor_teleport is False


def test_build_settings_rejects_unknown_option() -> None:
    try:
        build_settings(not_a_setting=True)
    except TypeError as exc:
        assert "not_a_setting" in str(exc)
    else:
        raise AssertionError("Unknown settings should raise TypeError")


def test_build_settings_accepts_gui_friendly_overrides(tmp_path: Path) -> None:
    settings = build_settings(
        workspace=tmp_path,
        llm_provider="deepseek",
        deepseek_model="deepseek-flash",
        enable_status_window=False,
    )

    assert settings.workspace == tmp_path
    assert settings.llm_provider == "deepseek"
    assert settings.enable_status_window is False


def test_gui_lists_every_settings_field() -> None:
    all_fields = {field.name for field in fields(Settings)}
    listed = set(FurtiApp._BASIC_FIELDS) | set(FurtiApp._ADVANCED_FIELDS)

    assert all_fields == listed
