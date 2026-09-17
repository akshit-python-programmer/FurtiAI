"""Window-focus wiring for keyboard/type/scroll dispatch.

The executor must bring the intended application into the foreground *before*
it sends a keystroke or scroll, otherwise the input lands in Furti's own
status window (or whatever else happens to have focus). These tests drive
``PlanExecutor._perform_action`` / ``_resolve_window_title`` with a fake
controller and a monkeypatched Windows layer, so nothing touches the real OS.
"""

import threading
from types import SimpleNamespace

import furti_ai.executor as executor_module
from furti_ai.executor import PlanExecutor
from furti_ai.models import ActionType
from furti_ai.planner import PlanStep


class FakeController:
    def __init__(self):
        self.calls = []

    def press_key(self, key, presses=1):
        self.calls.append(("press_key", key, presses))

    def type_text(self, text):
        self.calls.append(("type_text", text))

    def scroll(self, clicks):
        self.calls.append(("scroll", clicks))

    def click(self, x, y, button="left", clicks=1):
        self.calls.append(("click", x, y, button, clicks))


def make_executor(monkeypatch, *, focus_enabled=True, focus_result=True):
    executor = PlanExecutor.__new__(PlanExecutor)
    executor._settings = SimpleNamespace(
        focus_intended_window=focus_enabled, input_pause=0.0
    )
    executor._stop = threading.Event()
    executor._controller = FakeController()

    journal = SimpleNamespace(thoughts=[], warns=[])
    journal.thought = lambda message: journal.thoughts.append(message)
    journal.warn = lambda message: journal.warns.append(message)
    executor._journal = journal

    focused = []
    monkeypatch.setattr(
        executor_module, "focus_window", lambda title: focused.append(title) or focus_result
    )
    monkeypatch.setattr(
        executor_module,
        "find_window",
        lambda title: 123 if str(title).lower() == "notepad" else None,
    )
    return executor, focused, journal


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


def test_key_press_focuses_explicit_window_before_dispatch(monkeypatch):
    executor, focused, journal = make_executor(monkeypatch)
    step = make_step(
        ActionType.KEY_PRESS, params={"key": "win", "window": "Notepad"}
    )

    executor._perform_action(step, (0, 0), None)

    assert focused == ["Notepad"]
    assert executor._controller.calls == [("press_key", "win", 1)]
    assert any("Notepad" in thought for thought in journal.thoughts)


def test_type_focuses_window_resolved_from_description(monkeypatch):
    executor, focused, journal = make_executor(monkeypatch)
    step = make_step(
        ActionType.TYPE,
        description='Type "hello" into the Notepad window',
        text="hello",
    )

    executor._perform_action(step, (0, 0), None)

    assert focused == ["Notepad"]
    assert executor._controller.calls == [("type_text", "hello")]


def test_scroll_focuses_window(monkeypatch):
    executor, focused, _journal = make_executor(monkeypatch)
    step = make_step(
        ActionType.SCROLL,
        params={"window": "Notepad", "scroll_clicks": -4},
    )

    executor._perform_action(step, (0, 0), None)

    assert focused == ["Notepad"]
    assert executor._controller.calls == [("scroll", -4)]


def test_focus_disabled_skips_window_handling(monkeypatch):
    executor, focused, _journal = make_executor(
        monkeypatch, focus_enabled=False
    )
    step = make_step(
        ActionType.KEY_PRESS, params={"key": "enter", "window": "Notepad"}
    )

    executor._perform_action(step, (0, 0), None)

    assert focused == []
    assert executor._controller.calls == [("press_key", "enter", 1)]


def test_no_window_hint_does_not_focus(monkeypatch):
    executor, focused, _journal = make_executor(monkeypatch)
    step = make_step(
        ActionType.KEY_PRESS,
        params={"key": "enter"},
        description="press enter to confirm",
    )

    executor._perform_action(step, (0, 0), None)

    # find_window only matches "notepad" in these tests, and the description
    # carries no window phrase, so nothing should be focused.
    assert focused == []


def test_click_does_not_trigger_focus(monkeypatch):
    executor, focused, _journal = make_executor(monkeypatch)
    step = make_step(
        ActionType.CLICK,
        target="Notepad",
        params={"window": "Notepad"},
    )

    executor._perform_action(step, (10, 20), None)

    assert focused == []
    assert executor._controller.calls == [("click", 10, 20, "left", 1)]


def test_resolve_window_title_prefers_params_over_description(monkeypatch):
    executor, _focused, _journal = make_executor(monkeypatch)
    step = make_step(
        ActionType.TYPE,
        params={"window": "Notepad"},
        description='Type into the Chrome window',
        text="x",
    )

    # The explicit params.window wins without consulting find_window.
    assert executor._resolve_window_title(step) == "Notepad"


def test_windows_module_helpers_are_importable():
    from furti_ai.windows import (
        find_window,
        list_windows,
        set_process_dpi_aware,
    )

    # Smoke test only: on Windows these touch the real OS, on other platforms
    # they degrade to no-ops. Nothing here may raise.
    assert isinstance(set_process_dpi_aware(), bool)
    assert isinstance(list_windows(), list)
    for hwnd, title in list_windows():
        assert isinstance(hwnd, int)
        assert isinstance(title, str)
    # find_window must never return Furti's own windows.
    assert find_window("Furti AI - Desktop Automation") is None
