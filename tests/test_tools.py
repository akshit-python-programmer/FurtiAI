"""Direct tools: the OS-level actions that replace mouse/keyboard chains.

The point of these tests is that a tool step must be fast and honest -- no
screen capture, no OCR, no model review -- and that the destructive-command
guard refuses to erase anything without an explicit opt-in.
"""

import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from furti_ai import windows as windows_module
from furti_ai.brain import _plan_tool_schema
from furti_ai.executor import PlanExecutor
from furti_ai.models import ActionType, coerce_action
from furti_ai.planner import PlanStep, TaskPlan
from furti_ai.tools import (
    DirectToolRunner,
    describe_tool_step,
    is_direct_tool,
)


# ------------------------------------------------------------- vocabulary
@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("launch_app", ActionType.LAUNCH_APP),
        ("open_app", ActionType.LAUNCH_APP),
        ("start_program", ActionType.LAUNCH_APP),
        ("open_url", ActionType.OPEN_PATH),
        ("open_folder", ActionType.OPEN_PATH),
        ("run_command", ActionType.RUN_COMMAND),
        ("execute_command", ActionType.RUN_COMMAND),
        ("save_file", ActionType.WRITE_FILE),
        ("create_file", ActionType.WRITE_FILE),
        ("read_file", ActionType.READ_FILE),
        ("copy_to_clipboard", ActionType.SET_CLIPBOARD),
        ("paste_from_clipboard", ActionType.GET_CLIPBOARD),
        ("bring_to_front", ActionType.FOCUS_WINDOW),
        ("list_windows", ActionType.LIST_WINDOWS),
        ("close_window", ActionType.CLOSE_WINDOW),
        ("minimize", ActionType.MINIMIZE_WINDOW),
        ("maximize_window", ActionType.MAXIMIZE_WINDOW),
        ("sleep", ActionType.WAIT),
        ("delay", ActionType.WAIT),
    ],
)
def test_tool_spellings_resolve(spelling, expected):
    assert coerce_action(spelling) is expected


def test_classic_aliases_still_resolve():
    """The new vocabulary must not shadow the existing mouse/keyboard names."""
    assert coerce_action("left_click") is ActionType.CLICK
    assert coerce_action("hover") is ActionType.MOVE
    assert coerce_action("hotkey") is ActionType.KEY_PRESS
    assert coerce_action("type_text") is ActionType.TYPE
    assert coerce_action("wheel") is ActionType.SCROLL


@pytest.mark.parametrize("spelling", ["close", "focus", "start", "activate"])
def test_ambiguous_bare_verbs_still_fall_back_to_a_click(spelling):
    """``close the tab`` must honour its bbox, not close the whole window.

    These words are deliberately absent from the alias map so an ambiguous
    action name keeps the caller's default (a click on the planned target)
    instead of silently escalating to a window-level tool.
    """
    assert coerce_action(spelling, ActionType.CLICK) is ActionType.CLICK
    assert not is_direct_tool(coerce_action(spelling, ActionType.CLICK))


def test_is_direct_tool_classification():
    assert is_direct_tool(ActionType.LAUNCH_APP)
    assert is_direct_tool(ActionType.GET_CLIPBOARD)
    assert not is_direct_tool(ActionType.CLICK)
    assert not is_direct_tool("click")


def test_reflex_schema_excludes_direct_tools():
    """The template-compiling planner must never be offered a tool action."""
    schema = _plan_tool_schema()
    enum = schema["function"]["parameters"]["properties"]["action"]["enum"]
    assert "click" in enum
    assert "launch_app" not in enum
    assert not any(value in enum for value in ("run_command", "write_file"))


# ----------------------------------------------------------------- fixtures
def make_runner(tmp_path, *, stop: threading.Event | None = None, **settings):
    defaults = {
        "tool_timeout": 10.0,
        "tool_max_output_chars": 4000,
        "allow_shell_commands": True,
        "allow_destructive_commands": False,
        "workspace": tmp_path,
    }
    defaults.update(settings)
    return DirectToolRunner(SimpleNamespace(**defaults), None, stop)


def make_step(action, **kwargs):
    defaults = {
        "index": 1,
        "description": "",
        "action": action,
        "target": None,
        "params": {},
        "window": None,
    }
    defaults.update(kwargs)
    return PlanStep(**defaults)


class ExplodingContext:
    """Fails loudly if a tool step ever asks for screen grounding."""

    def observe(self, instruction, force_fresh=False):
        raise AssertionError("a direct tool step must not capture the screen")


class FakeJournal:
    def __init__(self):
        self.events = []

    def action(self, message, signal="trying"):
        self.events.append(("action", message))

    def confirm(self, message):
        self.events.append(("confirm", message))

    def error(self, message):
        self.events.append(("error", message))


# --------------------------------------------------------------- file tools
def test_write_file_then_read_file_round_trip(tmp_path):
    runner = make_runner(tmp_path)
    target = tmp_path / "notes" / "hello.txt"

    write = runner.run(
        make_step(
            ActionType.WRITE_FILE,
            params={"path": str(target), "content": "hello furti"},
        )
    )
    assert write.ok, write.detail
    assert target.read_text(encoding="utf-8") == "hello furti"

    read = runner.run(make_step(ActionType.READ_FILE, params={"path": str(target)}))
    assert read.ok
    assert "hello furti" in read.output


def test_write_file_append_mode_keeps_existing_text(tmp_path):
    runner = make_runner(tmp_path)
    target = tmp_path / "log.txt"
    runner.run(
        make_step(ActionType.WRITE_FILE, params={"path": str(target), "content": "a"})
    )
    runner.run(
        make_step(
            ActionType.WRITE_FILE,
            params={"path": str(target), "content": "b", "append": True},
        )
    )
    assert target.read_text(encoding="utf-8") == "ab"


def test_write_file_uses_step_text_as_content(tmp_path):
    runner = make_runner(tmp_path)
    target = tmp_path / "typed.txt"
    result = runner.run(
        make_step(
            ActionType.WRITE_FILE,
            params={"path": str(target)},
            text="written straight to disk",
        )
    )
    assert result.ok
    assert target.read_text(encoding="utf-8") == "written straight to disk"


def test_write_file_without_path_fails_cleanly(tmp_path):
    result = make_runner(tmp_path).run(make_step(ActionType.WRITE_FILE))
    assert result.ok is False
    assert "path" in result.detail


def test_read_file_missing_path_reports_failure(tmp_path):
    result = make_runner(tmp_path).run(
        make_step(ActionType.READ_FILE, params={"path": str(tmp_path / "nope.txt")})
    )
    assert result.ok is False
    assert "does not exist" in result.detail


def test_read_file_of_directory_lists_entries(tmp_path):
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    result = make_runner(tmp_path).run(
        make_step(ActionType.READ_FILE, params={"path": str(tmp_path)})
    )
    assert result.ok
    assert result.output.splitlines() == ["a.txt", "b.txt"]


def test_scrape_url_extracts_readable_html_without_browser(tmp_path, monkeypatch):
    from furti_ai import tools

    class FakeResponse:
        headers = {"Content-Type": "text/html; charset=utf-8"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return (
                b"<html><title>Example</title><script>ignore()</script>"
                b"<body><h1>Visible heading</h1><p>Readable content.</p></body></html>"
            )

    monkeypatch.setattr(tools, "urlopen", lambda request, timeout: FakeResponse())
    result = make_runner(tmp_path).run(
        make_step(ActionType.SCRAPE_URL, params={"url": "https://example.com"})
    )

    assert result.ok
    assert result.data["title"] == "Example"
    assert "Visible heading" in result.output
    assert "ignore()" not in result.output


def test_scrape_url_rejects_non_http_urls(tmp_path):
    result = make_runner(tmp_path).run(
        make_step(ActionType.SCRAPE_URL, params={"url": "file:///secret.txt"})
    )

    assert result.ok is False
    assert "http" in result.detail


def test_tool_output_is_truncated(tmp_path):
    runner = make_runner(tmp_path, tool_max_output_chars=10)
    target = tmp_path / "long.txt"
    target.write_text("x" * 500, encoding="utf-8")
    result = runner.run(make_step(ActionType.READ_FILE, params={"path": str(target)}))
    assert "more characters" in result.output


# --------------------------------------------------------------- shell tool
def test_run_command_captures_output_and_exit_code(tmp_path):
    result = make_runner(tmp_path).run(
        make_step(ActionType.RUN_COMMAND, params={"command": "echo furti-tool-test"})
    )
    assert result.ok
    assert "furti-tool-test" in result.output


def test_run_command_reports_nonzero_exit(tmp_path):
    result = make_runner(tmp_path).run(
        make_step(ActionType.RUN_COMMAND, params={"command": "exit 3"})
    )
    assert result.ok is False
    assert "exit code 3" in result.detail


def test_run_command_can_be_disabled(tmp_path):
    runner = make_runner(tmp_path, allow_shell_commands=False)
    result = runner.run(
        make_step(ActionType.RUN_COMMAND, params={"command": "echo nope"})
    )
    assert result.ok is False
    assert "disabled" in result.detail


@pytest.mark.parametrize(
    "command",
    [
        "format C:",
        "rm -rf /",
        "Remove-Item C:\\data -Recurse -Force",
        "DROP TABLE customers",
        "diskpart",
    ],
)
def test_destructive_commands_are_refused(tmp_path, command):
    result = make_runner(tmp_path).run(
        make_step(ActionType.RUN_COMMAND, params={"command": command})
    )
    assert result.ok is False
    assert "destructive" in result.detail


def test_destructive_command_runs_only_with_explicit_confirmation(tmp_path, monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = make_runner(tmp_path).run(
        make_step(
            ActionType.RUN_COMMAND,
            params={"command": "rm -rf /", "confirm": True},
        )
    )
    assert result.ok
    assert calls == ["rm -rf /"]


def test_harmless_command_is_not_blocked(tmp_path, monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    result = make_runner(tmp_path).run(
        make_step(ActionType.RUN_COMMAND, params={"command": "git status"})
    )
    assert result.ok


# ---------------------------------------------------------------- app tools
def test_launch_app_maps_friendly_name_to_executable(tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(
        subprocess, "Popen", lambda argv, **kwargs: started.append(argv) or SimpleNamespace(pid=42)
    )
    monkeypatch.setattr(
        "furti_ai.tools.shutil.which",
        lambda name: "C:\\Windows\\System32\\" + name if name == "notepad.exe" else None,
    )

    result = make_runner(tmp_path).run(
        make_step(ActionType.LAUNCH_APP, params={"app": "notepad", "settle": 0})
    )

    assert result.ok
    assert started and started[0][0].lower().endswith("notepad.exe")
    assert result.data["pid"] == 42


def test_launch_app_passes_arguments(tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(
        subprocess, "Popen", lambda argv, **kwargs: started.append(argv) or SimpleNamespace(pid=7)
    )
    monkeypatch.setattr("furti_ai.tools.shutil.which", lambda name: name)

    runner = make_runner(tmp_path)
    result = runner.run(
        make_step(
            ActionType.LAUNCH_APP,
            params={"app": "notepad", "args": ["C:\\tmp\\a.txt"], "settle": 0},
        )
    )
    assert result.ok
    assert started[0][1:] == ["C:\\tmp\\a.txt"]


def test_launch_app_failure_is_reported_not_raised(tmp_path, monkeypatch):
    monkeypatch.setattr("furti_ai.tools.shutil.which", lambda name: None)
    monkeypatch.setattr("furti_ai.tools._open_externally", lambda target: False)
    result = make_runner(tmp_path).run(
        make_step(ActionType.LAUNCH_APP, params={"app": "no-such-app-xyz"})
    )
    assert result.ok is False
    assert "could not find an application" in result.detail


def test_launch_app_without_name_fails_cleanly(tmp_path):
    result = make_runner(tmp_path).run(make_step(ActionType.LAUNCH_APP))
    assert result.ok is False
    assert "app" in result.detail


def test_open_path_rejects_missing_file(tmp_path):
    result = make_runner(tmp_path).run(
        make_step(ActionType.OPEN_PATH, params={"path": str(tmp_path / "gone.txt")})
    )
    assert result.ok is False
    assert "does not exist" in result.detail


def test_open_path_treats_a_drive_letter_as_a_path_not_a_uri(tmp_path):
    """``C:\\...`` must not be parsed as a URI scheme called ``c``."""
    from furti_ai.tools import _looks_like_uri

    assert _looks_like_uri("https://example.com") is True
    assert _looks_like_uri("ms-settings:") is True
    assert _looks_like_uri("mailto:someone@example.com") is True
    assert _looks_like_uri(r"C:\Users\me\notes.txt") is False
    assert _looks_like_uri(str(tmp_path)) is False


def test_open_path_accepts_urls_without_touching_the_filesystem(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(
        "furti_ai.tools._open_externally", lambda target: opened.append(target) or True
    )
    result = make_runner(tmp_path).run(
        make_step(ActionType.OPEN_PATH, params={"path": "https://example.com"})
    )
    assert result.ok
    assert opened == ["https://example.com"]


# ------------------------------------------------------------- window tools
def test_focus_window_reports_missing_window(tmp_path, monkeypatch):
    monkeypatch.setattr(windows_module, "focus_window", lambda title: False)
    result = make_runner(tmp_path).run(
        make_step(ActionType.FOCUS_WINDOW, params={"window": "Ghost"})
    )
    assert result.ok is False
    assert "no window matched" in result.detail


def test_window_state_tools_use_the_active_window_when_untitled(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(windows_module, "close_window", lambda target: calls.append(("close", target)) or True)
    monkeypatch.setattr(windows_module, "minimize_window", lambda target: calls.append(("min", target)) or True)
    monkeypatch.setattr(windows_module, "maximize_window", lambda target: calls.append(("max", target)) or True)

    runner = make_runner(tmp_path)
    assert runner.run(make_step(ActionType.CLOSE_WINDOW)).ok
    assert runner.run(make_step(ActionType.MINIMIZE_WINDOW)).ok
    assert runner.run(
        make_step(ActionType.MAXIMIZE_WINDOW, params={"window": "Notepad", "settle": 0})
    ).ok

    assert calls == [("close", None), ("min", None), ("max", "Notepad")]


def test_list_windows_returns_titles(tmp_path, monkeypatch):
    monkeypatch.setattr(
        windows_module, "list_windows", lambda: [(1, "Notepad"), (2, "Chrome")]
    )
    result = make_runner(tmp_path).run(make_step(ActionType.LIST_WINDOWS))
    assert result.ok
    assert result.data["windows"] == ["Notepad", "Chrome"]
    assert "Chrome" in result.output


# ------------------------------------------------------------------ waiting
def test_wait_reports_the_pause(tmp_path):
    result = make_runner(tmp_path).run(
        make_step(ActionType.WAIT, params={"seconds": 0})
    )
    assert result.ok
    assert "waited 0.00s" in result.detail


def test_wait_without_seconds_fails_cleanly(tmp_path):
    result = make_runner(tmp_path).run(make_step(ActionType.WAIT))
    assert result.ok is False
    assert "seconds" in result.detail


def test_wait_aborts_immediately_when_stop_is_set(tmp_path):
    stop = threading.Event()
    stop.set()
    runner = make_runner(tmp_path, stop=stop)
    from furti_ai.context import TaskAborted

    with pytest.raises(TaskAborted):
        runner.run(make_step(ActionType.WAIT, params={"seconds": 5}))


def test_unknown_action_is_rejected_by_the_runner(tmp_path):
    result = make_runner(tmp_path).run(make_step(ActionType.CLICK))
    assert result.ok is False
    assert "not a direct tool" in result.detail


# ------------------------------------------------------------- executor glue
def make_tool_executor(tmp_path, **overrides):
    executor = PlanExecutor.__new__(PlanExecutor)
    settings = SimpleNamespace(
        direct_tools=True,
        allow_shell_commands=True,
        allow_destructive_commands=False,
        tool_timeout=10.0,
        tool_max_output_chars=4000,
        templates_dir=tmp_path,
        confidence_threshold=0.8,
        focus_intended_window=False,
    )
    for key, value in overrides.items():
        setattr(settings, key, value)
    executor._settings = settings
    executor._stop = threading.Event()
    executor._journal = FakeJournal()
    executor._context = ExplodingContext()
    executor._cursor_position = lambda: (5, 6)
    return executor


def test_executor_runs_tool_step_without_any_screen_work(tmp_path):
    executor = make_tool_executor(tmp_path)
    target = tmp_path / "fast.txt"
    step = make_step(
        ActionType.WRITE_FILE, params={"path": str(target), "content": "no screenshots"}
    )

    result = executor._execute_tool_step(step)

    assert result.success is True
    assert target.read_text(encoding="utf-8") == "no screenshots"
    # Nothing was compiled into a reflex: a tool call is already the fast path.
    assert result.reflex is None
    assert any("tool latency" in note for note in result.notes)


def test_executor_reports_tool_failure_as_step_failure(tmp_path):
    executor = make_tool_executor(tmp_path)
    step = make_step(ActionType.READ_FILE, params={"path": str(tmp_path / "missing.txt")})

    result = executor._execute_tool_step(step)

    assert result.success is False
    assert result.action_dispatched is False
    assert any("failed" in note for note in result.notes)


def test_executor_honours_the_direct_tools_switch(tmp_path):
    executor = make_tool_executor(tmp_path, direct_tools=False)
    target = tmp_path / "blocked.txt"

    result = executor._execute_tool_step(
        make_step(ActionType.WRITE_FILE, params={"path": str(target), "content": "x"})
    )

    assert result.success is False
    assert not target.exists()
    assert any("disabled" in note for note in result.notes)


def test_executor_resolve_target_skips_the_screen_for_tools(tmp_path):
    executor = make_tool_executor(tmp_path)
    resolution = executor._resolve_target(
        make_step(ActionType.LIST_WINDOWS), None, None
    )
    assert resolution is not None
    assert resolution.capture_coordinates is False
    assert "direct tool" in resolution.anchor_note


def test_executor_perform_action_raises_on_tool_failure(tmp_path):
    executor = make_tool_executor(tmp_path)
    with pytest.raises(RuntimeError):
        executor._perform_action(make_step(ActionType.READ_FILE, params={"path": "nope"}), (0, 0))


# ------------------------------------------------------- end-to-end pipeline
class RecordingContext:
    """Counts screen observations, which tool steps must never trigger."""

    def __init__(self, scene):
        self.scene = scene
        self.observe_calls = 0

    def observe(self, instruction, force_fresh=False):
        self.observe_calls += 1
        return self.scene


class ExplodingLLM:
    """Any chat call is a bug: tool steps must not need the model."""

    def chat_text(self, system, user, purpose=""):
        raise AssertionError(f"unexpected LLM call ({purpose})")

    def chat_vision(self, system, user, image_b64, purpose=""):
        raise AssertionError(f"unexpected LLM call ({purpose})")


class RecordingController:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args))

        return record


def make_pipeline_executor(tmp_path, verify_steps=True):
    import numpy as np

    from furti_ai.config import Settings
    from furti_ai.context import SceneObservation
    from furti_ai.memory import MemoryManager
    from furti_ai.tasklog import TaskJournal

    from furti_ai.planner import TaskPlanner

    settings = Settings(workspace=tmp_path, verify_steps=verify_steps)
    settings.ensure_dirs()
    journal = TaskJournal("tools_pipeline", tmp_path / "reports")
    scene = SceneObservation(
        frame=np.full((240, 320, 3), 180, dtype="uint8"),
        text_lines=[],
        scene_text="",
    )
    context = RecordingContext(scene)
    controller = RecordingController()
    planner = TaskPlanner(ExplodingLLM(), settings, journal, context, threading.Event())
    executor = PlanExecutor(
        settings,
        journal,
        None,
        controller,
        MemoryManager(settings.memory_file),
        planner,
        context,
        threading.Event(),
        "tools_pipeline",
    )
    return executor, context, controller


def test_pipeline_runs_tool_steps_without_screen_or_model_work(tmp_path):
    executor, context, controller = make_pipeline_executor(tmp_path)
    # A regression that routes tools through the generic step loop would call
    # this; the direct-tool path must bypass the review entirely.
    executor._review_step_and_next = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("tool steps must not be reviewed by the model")
    )

    target = tmp_path / "pipeline.txt"
    plan = TaskPlan(
        task_name="t",
        goal="write and read a file",
        steps=[
            make_step(
                ActionType.WRITE_FILE,
                params={"path": str(target), "content": "done without a mouse"},
                description="Write the file",
            ),
            make_step(
                ActionType.READ_FILE,
                params={"path": str(target)},
                description="Confirm the contents",
                index=2,
            ),
            make_step(ActionType.WAIT, params={"seconds": 0}, description="Pause", index=3),
        ],
    )

    report = executor.execute("write and read a file", plan)

    assert report.success is True
    assert [result.success for result in report.results] == [True, True, True]
    assert target.read_text(encoding="utf-8") == "done without a mouse"
    assert context.observe_calls == 0, "tool steps must not capture the screen"
    assert controller.calls == [], "tool steps must not touch the input device"
    # No reflex is compiled for a tool step: the tool call is already instant.
    assert all(result.reflex is None for result in report.results)


def test_pipeline_reports_a_failing_tool_step_without_dispatching_input(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(windows_module, "focus_window", lambda title: False)
    executor, _context, controller = make_pipeline_executor(tmp_path, verify_steps=False)

    plan = TaskPlan(
        task_name="t",
        goal="focus a window",
        steps=[
            make_step(
                ActionType.FOCUS_WINDOW,
                params={"window": "Missing Window"},
                description="Focus it",
            )
        ],
    )

    report = executor.execute("focus a window", plan)

    assert report.success is False
    assert report.results[0].action_dispatched is False
    assert controller.calls == []
    assert any(
        "no window matched" in note for note in report.results[0].notes
    )



# --------------------------------------------------------------- plan shape
def test_plan_step_parses_a_tool_step():
    step = PlanStep.from_dict(
        {"description": "Start Notepad", "action": "launch_app", "params": {"app": "notepad"}},
        1,
    )
    assert step.action is ActionType.LAUNCH_APP
    assert step.params["app"] == "notepad"


def test_plan_describe_shows_tool_arguments():
    plan = TaskPlan(
        task_name="t",
        goal="open notepad",
        steps=[make_step(ActionType.LAUNCH_APP, params={"app": "notepad"}, description="Start it")],
    )
    text = plan.describe()
    assert "launch_app" in text
    assert "notepad" in text


def test_describe_tool_step_summarises_write_payload():
    step = make_step(
        ActionType.WRITE_FILE, params={"path": "out.txt", "content": "abcd"}
    )
    described = describe_tool_step(step)
    assert "out.txt" in described
    assert "4 chars" in described


# ---------------------------------------------------------------- screenshots
class FakeScreen:
    """ScreenCapture stub: a deterministic gradient image."""

    def __init__(self, shape=(120, 200)):
        self.shape = shape
        self.captures = 0

    def capture(self):
        self.captures += 1
        import numpy as np

        frame = np.zeros((self.shape[0], self.shape[1], 3), dtype="uint8")
        frame[:, : self.shape[1] // 2] = 40
        frame[:, self.shape[1] // 2 :] = 220
        return frame


def make_screenshot_runner(tmp_path, screen=None):
    runner = make_runner(tmp_path)
    runner._screen = screen or FakeScreen()
    return runner


def test_screenshot_saves_a_region_and_reports_the_path(tmp_path):
    runner = make_screenshot_runner(tmp_path)
    result = runner.run(
        make_step(
            ActionType.SCREENSHOT,
            params={"region": {"x": 10, "y": 20, "width": 50, "height": 30}},
        )
    )

    assert result.ok, result.detail
    saved = Path(result.data["path"])
    assert saved.is_file()
    assert (result.data["width"], result.data["height"]) == (50, 30)
    assert result.data["region"] == {"x": 10, "y": 20, "width": 50, "height": 30}
    assert str(saved) in result.output
    # The default location is the workspace's screenshots folder.
    assert saved.parent == tmp_path / "screenshots"


def test_screenshot_without_a_region_captures_the_whole_screen(tmp_path):
    runner = make_screenshot_runner(tmp_path, FakeScreen(shape=(60, 80)))
    result = runner.run(make_step(ActionType.SCREENSHOT, params={"label": "whole"}))

    assert result.ok
    assert (result.data["width"], result.data["height"]) == (80, 60)
    assert result.data["region"] is None
    assert "whole" in Path(result.data["path"]).name


def test_screenshot_clamps_a_region_to_the_frame(tmp_path):
    runner = make_screenshot_runner(tmp_path, FakeScreen(shape=(60, 80)))
    result = runner.run(
        make_step(
            ActionType.SCREENSHOT,
            params={"x": 70, "y": 50, "width": 500, "height": 500},
        )
    )

    assert result.ok
    assert result.data["region"] == {"x": 70, "y": 50, "width": 10, "height": 10}


def test_screenshot_honours_an_explicit_path_and_format(tmp_path):
    runner = make_screenshot_runner(tmp_path)
    target = tmp_path / "shots" / "detail.jpg"
    result = runner.run(
        make_step(
            ActionType.SCREENSHOT,
            params={"region": {"x": 0, "y": 0, "width": 20, "height": 20},
                    "path": str(target)},
        )
    )

    assert result.ok
    assert target.is_file()
    assert "JPEG" in result.output


def test_screenshot_scales_down_when_asked(tmp_path):
    runner = make_screenshot_runner(tmp_path, FakeScreen(shape=(100, 100)))
    result = runner.run(make_step(ActionType.SCREENSHOT, params={"scale": 0.5}))
    assert result.ok
    assert (result.data["width"], result.data["height"]) == (50, 50)


# -------------------------------------------------------------- file system
def test_create_folder_makes_parents_and_is_idempotent(tmp_path):
    runner = make_runner(tmp_path)
    nested = tmp_path / "a" / "b" / "c"

    first = runner.run(
        make_step(ActionType.CREATE_FOLDER, params={"path": str(nested)})
    )
    assert first.ok
    assert nested.is_dir()
    assert first.data["created"] is True

    second = runner.run(
        make_step(ActionType.CREATE_FOLDER, params={"path": str(nested)})
    )
    assert second.ok
    assert second.data["created"] is False


def test_create_folder_refuses_to_shadow_a_file(tmp_path):
    target = tmp_path / "already-a-file.txt"
    target.write_text("x", encoding="utf-8")
    result = make_runner(tmp_path).run(
        make_step(ActionType.CREATE_FOLDER, params={"path": str(target)})
    )
    assert result.ok is False
    assert "not a folder" in result.detail


def test_list_dir_separates_folders_from_files(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "note.txt").write_text("hello", encoding="utf-8")
    result = make_runner(tmp_path).run(
        make_step(ActionType.LIST_DIR, params={"path": str(tmp_path)})
    )

    assert result.ok
    assert "[dir]  sub" in result.output
    assert "[file] note.txt (5 bytes)" in result.output
    assert result.data["count"] == 2


def test_list_dir_supports_a_pattern(tmp_path):
    (tmp_path / "a.log").write_text("", encoding="utf-8")
    (tmp_path / "b.txt").write_text("", encoding="utf-8")
    result = make_runner(tmp_path).run(
        make_step(
            ActionType.LIST_DIR, params={"path": str(tmp_path), "pattern": "*.log"}
        )
    )
    assert result.ok
    assert result.data["count"] == 1
    assert "a.log" in result.output


def test_list_dir_reports_a_missing_folder(tmp_path):
    result = make_runner(tmp_path).run(
        make_step(ActionType.LIST_DIR, params={"path": str(tmp_path / "ghost")})
    )
    assert result.ok is False
    assert "does not exist" in result.detail


def test_copy_path_duplicates_a_file(tmp_path):
    source = tmp_path / "src.txt"
    source.write_text("content", encoding="utf-8")
    destination = tmp_path / "nested" / "copy.txt"

    result = make_runner(tmp_path).run(
        make_step(
            ActionType.COPY_PATH,
            params={"source": str(source), "destination": str(destination)},
        )
    )

    assert result.ok, result.detail
    assert destination.read_text(encoding="utf-8") == "content"
    assert source.exists()  # a copy leaves the original alone


def test_copy_path_duplicates_a_folder(tmp_path):
    source = tmp_path / "folder"
    source.mkdir()
    (source / "inner.txt").write_text("x", encoding="utf-8")
    destination = tmp_path / "folder-copy"

    result = make_runner(tmp_path).run(
        make_step(
            ActionType.COPY_PATH,
            params={"source": str(source), "destination": str(destination)},
        )
    )

    assert result.ok
    assert (destination / "inner.txt").is_file()
    assert (source / "inner.txt").is_file()


def test_move_path_renames(tmp_path):
    source = tmp_path / "old.txt"
    source.write_text("x", encoding="utf-8")
    destination = tmp_path / "new.txt"

    result = make_runner(tmp_path).run(
        make_step(
            ActionType.MOVE_PATH,
            params={"source": str(source), "destination": str(destination)},
        )
    )

    assert result.ok
    assert destination.is_file()
    assert not source.exists()


def test_move_path_needs_both_ends(tmp_path):
    source = tmp_path / "old.txt"
    source.write_text("x", encoding="utf-8")
    result = make_runner(tmp_path).run(
        make_step(ActionType.MOVE_PATH, params={"source": str(source)})
    )
    assert result.ok is False
    assert "destination" in result.detail


def test_transfer_rejects_a_missing_source(tmp_path):
    result = make_runner(tmp_path).run(
        make_step(
            ActionType.COPY_PATH,
            params={"source": str(tmp_path / "nope"), "destination": str(tmp_path / "x")},
        )
    )
    assert result.ok is False
    assert "source does not exist" in result.detail


def test_delete_path_requires_confirmation(tmp_path):
    victim = tmp_path / "keep.txt"
    victim.write_text("important", encoding="utf-8")

    result = make_runner(tmp_path).run(
        make_step(ActionType.DELETE_PATH, params={"path": str(victim)})
    )

    assert result.ok is False
    assert "without confirmation" in result.detail
    assert victim.exists()


def test_delete_path_with_confirmation_removes_the_file(tmp_path):
    victim = tmp_path / "gone.txt"
    victim.write_text("bye", encoding="utf-8")

    result = make_runner(tmp_path).run(
        make_step(
            ActionType.DELETE_PATH, params={"path": str(victim), "confirm": True}
        )
    )

    assert result.ok, result.detail
    assert not victim.exists()


def test_delete_path_removes_a_folder_tree(tmp_path):
    folder = tmp_path / "tree" / "inner"
    folder.mkdir(parents=True)
    (folder / "x.txt").write_text("x", encoding="utf-8")

    result = make_runner(tmp_path).run(
        make_step(
            ActionType.DELETE_PATH,
            params={"path": str(tmp_path / "tree"), "confirm": True},
        )
    )

    assert result.ok
    assert not (tmp_path / "tree").exists()


def test_delete_path_protects_the_workspace_and_home(tmp_path):
    runner = make_runner(tmp_path, workspace=tmp_path)
    workspace_result = runner.run(
        make_step(
            ActionType.DELETE_PATH, params={"path": str(tmp_path), "confirm": True}
        )
    )
    assert workspace_result.ok is False
    assert "protected" in workspace_result.detail


def test_find_files_matches_by_pattern_within_depth(tmp_path):
    (tmp_path / "deep" / "deeper").mkdir(parents=True)
    (tmp_path / "top.txt").write_text("", encoding="utf-8")
    (tmp_path / "deep" / "mid.txt").write_text("", encoding="utf-8")
    (tmp_path / "deep" / "deeper" / "bottom.txt").write_text("", encoding="utf-8")

    result = make_runner(tmp_path).run(
        make_step(
            ActionType.FIND_FILES,
            params={"root": str(tmp_path), "pattern": "*.txt", "max_depth": 1},
        )
    )

    assert result.ok
    assert "top.txt" in result.output
    assert "bottom.txt" not in result.output


def test_find_files_honours_the_limit(tmp_path):
    for index in range(5):
        (tmp_path / f"f{index}.txt").write_text("", encoding="utf-8")
    result = make_runner(tmp_path).run(
        make_step(
            ActionType.FIND_FILES,
            params={"root": str(tmp_path), "pattern": "*.txt", "limit": 2},
        )
    )
    assert result.ok
    assert len(result.data["matches"]) == 2


def test_path_info_describes_files_and_missing_paths(tmp_path):
    target = tmp_path / "report.txt"
    target.write_text("hello", encoding="utf-8")
    runner = make_runner(tmp_path)

    present = runner.run(
        make_step(ActionType.PATH_INFO, params={"path": str(target)})
    )
    assert present.ok
    assert present.data["exists"] is True
    assert present.data["kind"] == "file"
    assert present.data["size_bytes"] == 5

    missing = runner.run(
        make_step(ActionType.PATH_INFO, params={"path": str(tmp_path / "nope.txt")})
    )
    assert missing.ok is True
    assert missing.data["exists"] is False
    assert "does not exist" in missing.detail


def test_paths_resolve_tilde_and_environment_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("FURTI_TEST_ROOT", str(tmp_path))
    result = make_runner(tmp_path).run(
        make_step(ActionType.CREATE_FOLDER, params={"path": "%FURTI_TEST_ROOT%/made"})
    )
    assert result.ok, result.detail
    assert (tmp_path / "made").is_dir()


def test_relative_paths_resolve_against_the_workspace(tmp_path):
    runner = make_runner(tmp_path, workspace=tmp_path)
    result = runner.run(
        make_step(ActionType.CREATE_FOLDER, params={"path": "relative/folder"})
    )
    assert result.ok
    assert (tmp_path / "relative" / "folder").is_dir()
