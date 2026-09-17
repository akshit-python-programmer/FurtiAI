"""The user/system context file: detection, persistence and prompt rendering.

The point of this profile is that the planner stops guessing: the exact Desktop
path, the installed applications and the user's own notes must reach the prompt,
and a fact the user typed into ``profile.md`` must survive a reload.
"""

import json
from pathlib import Path
from types import SimpleNamespace

from furti_ai.config import Settings
from furti_ai.profile import PROFILE_VERSION, UserContext, build_user_context


def make_settings(tmp_path: Path, **overrides) -> Settings:
    return Settings(workspace=tmp_path, **overrides)


def stub_detection(monkeypatch, context: UserContext, **facts) -> None:
    """Replace the machine probes with fixed values for deterministic tests."""
    defaults = {
        "_identity": lambda: {
            "username": "tester",
            "home": "C:\\Users\\tester",
            "hostname": "TESTBOX",
            "os": "Windows 11",
            "os_version": "10.0.1",
            "python": "3.13",
            "shell": "cmd.exe",
            "workspace": "C:\\ws",
        },
        "_display": lambda: {"width": 2560, "height": 1440, "scaling_percent": 150},
        "_folders": lambda: {
            "desktop": "C:\\Users\\tester\\OneDrive\\Desktop",
            "documents": "C:\\Users\\tester\\Documents",
        },
        "_drives": lambda: [{"path": "C:\\", "total_gb": 500.0, "free_gb": 123.4}],
        "_apps": lambda: ["Notepad", "Chrome", "Visual Studio Code"],
        "_cli_tools": lambda: ["git", "python"],
    }
    defaults.update(facts)
    for name, value in defaults.items():
        monkeypatch.setattr(context, name, value)


# ------------------------------------------------------------------ lifecycle
def test_prepare_writes_both_context_files(tmp_path):
    settings = make_settings(tmp_path)
    settings.ensure_dirs()
    build_user_context(settings)

    assert settings.profile_file.is_file()
    assert settings.profile_report.is_file()
    payload = json.loads(settings.profile_file.read_text(encoding="utf-8"))
    assert payload["version"] == PROFILE_VERSION
    assert payload["identity"]["username"]
    assert "## Notes" in settings.profile_report.read_text(encoding="utf-8")


def test_stale_profile_is_refreshed_on_load(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    settings.ensure_dirs()
    settings.profile_file.write_text(
        json.dumps({"version": 0, "collected_at": "", "notes": ["keep me"]}),
        encoding="utf-8",
    )
    context = UserContext(settings)
    stub_detection(monkeypatch, context)

    data = context.prepare()

    assert data["version"] == PROFILE_VERSION
    assert data["identity"]["hostname"] == "TESTBOX"
    # Refreshing must not lose what the user already taught it.
    assert "keep me" in data["notes"]


def test_fresh_profile_is_not_re_detected(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    settings.ensure_dirs()
    context = build_user_context(settings)
    calls = {"count": 0}

    def counting_identity():
        calls["count"] += 1
        return {}

    monkeypatch.setattr(context, "_identity", counting_identity)
    context.prepare()

    assert calls["count"] == 0  # still fresh, so nothing was re-detected


def test_disabled_profile_renders_nothing(tmp_path):
    settings = make_settings(tmp_path, profile_enabled=False)
    settings.ensure_dirs()
    context = UserContext(settings)

    assert context.prompt_block() == ""
    assert context.prepare() == {}


# ---------------------------------------------------------------------- notes
def test_notes_merge_settings_and_remembered_facts(tmp_path):
    settings = make_settings(
        tmp_path, user_notes="invoices live in D:\\invoices; prefer Chrome"
    )
    settings.ensure_dirs()
    context = build_user_context(settings)

    context.remember("Screenshots go to D:\\pics")
    notes = context.notes()

    assert "invoices live in D:\\invoices" in notes
    assert "prefer Chrome" in notes
    assert "Screenshots go to D:\\pics" in notes


def test_remember_is_idempotent(tmp_path):
    settings = make_settings(tmp_path)
    settings.ensure_dirs()
    context = build_user_context(settings)

    context.remember("I use Chrome")
    context.remember("i use chrome")

    assert sum(1 for note in context.notes() if note.lower() == "i use chrome") == 1


def test_notes_edited_in_the_markdown_mirror_are_loaded(tmp_path):
    settings = make_settings(tmp_path)
    settings.ensure_dirs()
    build_user_context(settings)

    markdown = settings.profile_report.read_text(encoding="utf-8")
    settings.profile_report.write_text(
        markdown.replace(
            "<!-- One fact per line, e.g. 'I keep invoices in D:\\invoices'. -->",
            "<!-- One fact per line -->\n- my team shares templates in D:\\team",
        ),
        encoding="utf-8",
    )

    reloaded = UserContext(settings)

    assert "my team shares templates in D:\\team" in reloaded.notes()


def test_record_task_keeps_a_bounded_history(tmp_path):
    settings = make_settings(tmp_path)
    settings.ensure_dirs()
    context = build_user_context(settings)

    for index in range(30):
        context.record_task(f"task {index}", index % 2 == 0)

    history = context.data()["recent_tasks"]
    assert len(history) == 20
    assert history[-1]["instruction"] == "task 29"
    assert history[-1]["success"] is False


# -------------------------------------------------------------------- prompts
def test_summary_reports_the_detected_facts(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    settings.ensure_dirs()
    context = context_with_facts(tmp_path, monkeypatch, settings)

    summary = context.summary()

    assert "tester on TESTBOX" in summary
    assert "2560x1440 at 150% scaling" in summary
    assert "OneDrive\\Desktop" in summary
    assert "Notepad, Chrome" in summary
    assert "git, python" in summary
    assert "123GB free" in summary


def test_summary_caps_the_application_list(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    settings.ensure_dirs()
    context = build_user_context(settings)
    stub_detection(
        monkeypatch,
        context,
        _apps=lambda: [f"App{index:02d}" for index in range(50)],
    )
    context.refresh(force=True)

    summary = context.summary(max_apps=5)

    assert "App00" in summary
    assert "App04" in summary
    assert "App05" not in summary
    assert "+45 more" in summary


def test_prompt_block_frames_the_context(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, user_notes="never delete D:\\archive")
    settings.ensure_dirs()
    context = context_with_facts(tmp_path, monkeypatch, settings)

    block = context.prompt_block()

    assert block.startswith("Known user and system context")
    assert "never delete D:\\archive" in block
    assert str(settings.profile_file) in block


def test_recent_tasks_appear_in_the_prompt_block(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    settings.ensure_dirs()
    context = context_with_facts(tmp_path, monkeypatch, settings)
    context.record_task("open notepad and save todo.txt", True)

    assert "open notepad and save todo.txt" in context.summary()


def context_with_facts(tmp_path, monkeypatch, settings) -> UserContext:
    context = build_user_context(settings)
    stub_detection(monkeypatch, context)
    context.refresh(force=True)
    return context


def test_missing_profile_file_is_tolerated(tmp_path):
    """A corrupt or unreadable profile must never break a task."""
    settings = make_settings(tmp_path)
    settings.ensure_dirs()
    settings.profile_file.write_text("{not json", encoding="utf-8")

    context = UserContext(settings)

    assert context.data()["notes"] == []
    assert isinstance(context.summary(), str)


def test_setup_is_fail_open_when_detection_explodes(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    settings.ensure_dirs()
    context = UserContext(settings)
    def explode(self):
        raise RuntimeError("probe unavailable")

    monkeypatch.setattr(UserContext, "_drives", explode)
    monkeypatch.setattr(UserContext, "_apps", explode)

    # One broken probe must cost one fact, not the whole profile.
    data = context.refresh(force=True)

    assert data["drives"] == []
    assert data["apps"] == []
    assert data["identity"]["username"]  # the other sections still landed
    assert isinstance(context.summary(), str)


def test_settings_like_object_without_workspace_still_works(monkeypatch):
    settings = SimpleNamespace(
        profile_enabled=True,
        profile_file=Path("profile.json"),
        profile_report=Path("profile.md"),
        user_notes="",
        profile_max_apps=10,
        profile_max_age_days=7,
    )
    context = UserContext(settings)
    assert context.notes() == []
