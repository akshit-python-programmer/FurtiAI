"""Explicit screen-coordinate actions: move, click-at and drag-at.

The model frequently knows the pixel it wants to act on (a canvas, a slider
handle, a drag that starts where the cursor already is). Before these tests
existed such a step degraded into "no target anchor found" and the action was
never performed, which is why the agent appeared to plan and then do nothing.
"""

import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import furti_ai.executor as executor_module
from furti_ai.config import Settings
from furti_ai.context import SceneObservation
from furti_ai.executor import PlanExecutor, TargetResolution
from furti_ai.memory import MemoryManager
from furti_ai.models import ActionType
from furti_ai.planner import PlanStep, TaskPlan, TaskPlanner
from furti_ai.tasklog import TaskJournal
from furti_ai.verifier import CrossReview


class FakeController:
    def __init__(self):
        self.calls = []

    def move_to(self, x, y):
        self.calls.append(("move_to", x, y))

    def click(self, x, y, button="left", clicks=1):
        self.calls.append(("click", x, y, button, clicks))

    def double_click(self, x, y):
        self.calls.append(("double_click", x, y))

    def right_click(self, x, y):
        self.calls.append(("right_click", x, y))

    def drag(
        self,
        start_x,
        start_y,
        end_x,
        end_y,
        duration=None,
        button="left",
        hold_keys=None,
    ):
        self.calls.append(("drag", start_x, start_y, end_x, end_y, button))

    def press_key(self, key, presses=1):
        self.calls.append(("press_key", key, presses))

    def type_text(self, text):
        self.calls.append(("type_text", text))


class FakeLine:
    def __init__(self, text, center, confidence=0.95):
        self.text = text
        self.center = center
        self.confidence = confidence


def make_scene(text_lines=()):
    """Minimal scene stub for the resolution unit tests."""
    return SimpleNamespace(
        text_lines=list(text_lines),
        icons=[],
        frame=None,
        image_b64=None,
    )


def make_plan():
    return SimpleNamespace(frame=None, steps=[])


def make_executor(monkeypatch, cursor=(500, 400)):
    executor = PlanExecutor.__new__(PlanExecutor)
    executor._settings = SimpleNamespace(
        focus_intended_window=False,
        confidence_threshold=0.8,
        templates_dir=Path("templates"),
    )
    executor._stop = threading.Event()
    executor._controller = FakeController()
    executor._vision = None

    journal = SimpleNamespace(thoughts=[], warns=[], actions=[], errors=[])
    journal.thought = lambda message: journal.thoughts.append(message)
    journal.warn = lambda message: journal.warns.append(message)
    journal.action = lambda message, signal=None: journal.actions.append(
        (message, signal)
    )
    journal.error = lambda message: journal.errors.append(message)
    executor._journal = journal

    # pyautogui.position() is already in mouse space; stub it out so tests
    # never touch the real cursor.
    executor._cursor_position = lambda: cursor
    return executor


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


# ------------------------------------------------------------- explicit pixel
def test_click_step_uses_explicit_pixel_when_no_anchor_matches(monkeypatch):
    executor = make_executor(monkeypatch)
    step = make_step(
        ActionType.CLICK, target="Nowhere", params={"x": "300", "y": "450.6"}
    )

    resolution = executor._resolve_target(step, make_scene(), make_plan())

    assert resolution is not None
    assert resolution.center == (300, 451)
    assert resolution.capture_coordinates is True
    assert "explicit pixel" in resolution.anchor_note


def test_explicit_pixel_still_loses_to_a_live_anchor(monkeypatch):
    executor = make_executor(monkeypatch)
    step = make_step(
        ActionType.CLICK,
        target="Save",
        params={"x": 10, "y": 10},
    )
    scene = make_scene([FakeLine("Save", (700, 800))])

    resolution = executor._resolve_target(step, scene, make_plan())

    assert resolution.center == (700, 800)


def test_type_without_anchor_still_uses_the_focused_window(monkeypatch):
    executor = make_executor(monkeypatch)
    step = make_step(ActionType.TYPE, text="hello")

    resolution = executor._resolve_target(step, make_scene(), make_plan())

    assert resolution.center == (500, 400)
    assert resolution.capture_coordinates is False


# ---------------------------------------------------- type: focus before typing
def test_type_clicks_the_field_before_typing(monkeypatch):
    executor = make_executor(monkeypatch)
    step = make_step(ActionType.TYPE, text="hello", target="Search")

    executor._perform_action(step, (240, 115), None, focus_click=True)

    assert executor._controller.calls == [
        ("click", 240, 115, "left", 1),
        ("type_text", "hello"),
    ]
    assert any("focus the field" in thought for thought in executor._journal.thoughts)


def test_type_without_a_field_does_not_invent_a_click(monkeypatch):
    executor = make_executor(monkeypatch)
    step = make_step(ActionType.TYPE, text="hello")

    executor._perform_action(step, (240, 115), None)

    assert executor._controller.calls == [("type_text", "hello")]


def test_type_focus_click_follows_the_resolved_anchor(monkeypatch):
    executor = make_executor(monkeypatch)
    anchored = TargetResolution((10, 20), None, "OCR text 'Search'")
    fallback = TargetResolution(
        (500, 400), None, "focused window (no anchor)", capture_coordinates=False
    )
    typing = make_step(ActionType.TYPE, text="hello")

    # A real screen anchor means the model named the field: click it first.
    assert executor._type_focus_click(typing, anchored) is True
    # "The focused window" has no field to click.
    assert executor._type_focus_click(typing, fallback) is False
    # The model can opt out (type into whatever already has focus)...
    opting_out = make_step(ActionType.TYPE, text="hello", params={"focus_click": "false"})
    assert executor._type_focus_click(opting_out, anchored) is False
    # ...or in, even when no anchor was found.
    opting_in = make_step(ActionType.TYPE, text="hello", params={"focus_click": True})
    assert executor._type_focus_click(opting_in, fallback) is True
    # Only typing focuses a target this way.
    assert executor._type_focus_click(make_step(ActionType.CLICK), anchored) is False


def test_a_preceding_click_on_the_field_replaces_the_focus_click(monkeypatch):
    executor = make_executor(monkeypatch)
    anchored = TargetResolution((240, 115), None, "OCR text 'Search'")
    executor._last_click_point = (241, 114)  # template match jitter
    executor._last_click_index = 1

    typing = make_step(ActionType.TYPE, text="hello", index=2)

    assert executor._type_focus_click(typing, anchored) is False
    assert any(
        "already clicked" in thought for thought in executor._journal.thoughts
    )


def test_a_click_from_another_step_does_not_suppress_the_focus_click(monkeypatch):
    executor = make_executor(monkeypatch)
    anchored = TargetResolution((240, 115), None, "OCR text 'Search'")
    executor._last_click_point = (240, 115)
    executor._last_click_index = 1

    # Step 3, not step 2: the click was not for this field.
    typing = make_step(ActionType.TYPE, text="hello", index=3)

    assert executor._type_focus_click(typing, anchored) is True


def test_a_dispatched_click_records_its_point_for_the_next_step(monkeypatch):
    executor = make_executor(monkeypatch)

    executor._perform_action(make_step(ActionType.CLICK, index=4), (10, 20), None)

    assert executor._last_click_point == (10, 20)
    assert executor._last_click_index == 4
    assert executor._last_dispatch_at is not None


# ------------------------------------------------------ render delay (timing)
def test_the_next_action_waits_out_the_render_delay(monkeypatch):
    executor = make_executor(monkeypatch)
    executor._settings.render_delay = 0.05
    executor._last_dispatch_at = time.monotonic()

    waited = executor._await_render_delay(make_step(ActionType.CLICK))

    assert 0 < waited <= 0.05
    assert any("finish rendering" in t for t in executor._journal.thoughts)


def test_move_never_waits_because_it_changes_nothing_on_screen(monkeypatch):
    executor = make_executor(monkeypatch)
    executor._settings.render_delay = 5.0
    executor._last_dispatch_at = time.monotonic()

    assert executor._await_render_delay(make_step(ActionType.MOVE)) == 0.0


def test_the_first_action_of_a_run_does_not_wait(monkeypatch):
    executor = make_executor(monkeypatch)
    executor._settings.render_delay = 5.0

    assert executor._await_render_delay(make_step(ActionType.CLICK)) == 0.0


# ------------------------------------------- obstruction protocol (modals)
def test_a_dialog_owning_the_pixel_is_reported_as_an_obstruction(monkeypatch):
    executor = make_executor(monkeypatch)
    monkeypatch.setattr(
        executor_module, "window_at", lambda x, y: (999, "Cookie consent")
    )
    monkeypatch.setattr(
        executor_module, "find_window", lambda title, substring=True: 111
    )
    step = make_step(ActionType.CLICK, target="Export", params={"window": "Chrome"})

    reason = executor._obstruction_reason(step, (350, 220), make_scene())

    assert "Cookie consent" in reason
    assert "Chrome" in reason


def test_the_intended_window_never_obstructs_its_own_target(monkeypatch):
    executor = make_executor(monkeypatch)
    monkeypatch.setattr(
        executor_module, "window_at", lambda x, y: (111, "Google Chrome")
    )
    monkeypatch.setattr(
        executor_module, "find_window", lambda title, substring=True: 111
    )
    step = make_step(ActionType.CLICK, target="Export", params={"window": "Chrome"})

    assert executor._obstruction_reason(step, (350, 220), make_scene()) == ""


def test_nothing_is_reported_when_no_window_is_named(monkeypatch):
    """Without an intended window the only provable overlay is Furti's own."""
    executor = make_executor(monkeypatch)
    monkeypatch.setattr(
        executor_module, "window_at", lambda x, y: (999, "Some other app")
    )
    monkeypatch.setattr(
        executor_module, "find_window", lambda title, substring=True: None
    )
    step = make_step(ActionType.CLICK, target="Export")

    assert executor._obstruction_reason(step, (350, 220), make_scene()) == ""


def test_furtis_own_overlay_is_cleared_by_refocusing_not_by_escape(monkeypatch):
    executor = make_executor(monkeypatch)
    monkeypatch.setattr(
        executor_module,
        "window_at",
        lambda x, y: (999, "Furti AI - live status"),
    )
    monkeypatch.setattr(
        executor_module, "find_window", lambda title, substring=True: 111
    )
    focused: list[str] = []
    monkeypatch.setattr(
        executor_module,
        "focus_window",
        lambda title: focused.append(title) or True,
    )
    step = make_step(ActionType.CLICK, target="Export", params={"window": "Chrome"})

    spent = executor._clear_obstruction(step, (350, 220), make_scene(), 2)

    assert spent == 1
    assert focused == ["Chrome"]
    assert executor._controller.calls == []  # no ESC, no stray click


def test_a_small_dialog_is_dismissed_with_its_close_affordance(monkeypatch):
    executor = make_executor(monkeypatch)
    monkeypatch.setattr(
        executor_module, "window_at", lambda x, y: (999, "Cookie consent")
    )
    monkeypatch.setattr(
        executor_module, "find_window", lambda title, substring=True: 111
    )
    monkeypatch.setattr(executor_module, "window_rect", lambda hwnd: (0, 0, 300, 200))
    scene = make_scene([FakeLine("Close", (500, 60))])
    step = make_step(ActionType.CLICK, target="Export", params={"window": "Chrome"})

    spent = executor._clear_obstruction(step, (350, 220), scene, 2)

    assert spent == 1
    assert executor._controller.calls == [("click", 500, 60, "left", 1)]


def test_a_small_dialog_without_a_label_is_dismissed_with_escape(monkeypatch):
    executor = make_executor(monkeypatch)
    monkeypatch.setattr(
        executor_module, "window_at", lambda x, y: (999, "Cookie consent")
    )
    monkeypatch.setattr(
        executor_module, "find_window", lambda title, substring=True: 111
    )
    monkeypatch.setattr(executor_module, "window_rect", lambda hwnd: (0, 0, 300, 200))
    step = make_step(ActionType.CLICK, target="Export", params={"window": "Chrome"})

    assert executor._clear_obstruction(step, (350, 220), make_scene(), 2) == 1
    assert executor._controller.calls == [("press_key", "esc", 1)]


def test_a_full_size_other_window_is_refocused_not_escaped(monkeypatch):
    """ESC in an unrelated full-screen app would do something else entirely."""
    executor = make_executor(monkeypatch)
    monkeypatch.setattr(
        executor_module, "window_at", lambda x, y: (999, "Another app")
    )
    monkeypatch.setattr(
        executor_module, "find_window", lambda title, substring=True: 111
    )
    monkeypatch.setattr(executor_module, "window_rect", lambda hwnd: (0, 0, 640, 480))
    focused: list[str] = []
    monkeypatch.setattr(
        executor_module, "focus_window", lambda title: focused.append(title) or True
    )
    scene = SimpleNamespace(
        text_lines=[FakeLine("Close", (500, 60))],
        icons=[],
        frame=np.zeros((480, 640, 3), dtype=np.uint8),
        image_b64=None,
    )
    step = make_step(ActionType.CLICK, target="Export", params={"window": "Chrome"})

    assert executor._clear_obstruction(step, (350, 220), scene, 2) == 1
    assert focused == ["Chrome"]
    assert executor._controller.calls == []


def test_a_missing_target_with_a_close_affordance_dismisses_the_overlay(monkeypatch):
    """A covered target reads as "not found"; the Close button is the evidence."""
    executor = make_executor(monkeypatch)
    scene = make_scene([FakeLine("Close", (500, 60))])
    step = make_step(ActionType.CLICK, target="Export")

    assert executor._clear_unresolved_obstruction(step, scene, 2) == 1
    assert executor._controller.calls == [("click", 500, 60, "left", 1)]


def test_a_missing_target_without_evidence_presses_nothing(monkeypatch):
    executor = make_executor(monkeypatch)
    scene = make_scene([FakeLine("Invoice total", (500, 60))])
    step = make_step(ActionType.CLICK, target="Export")

    assert executor._clear_unresolved_obstruction(step, scene, 2) == 0
    assert executor._controller.calls == []


def test_a_step_wanting_to_click_close_is_not_dismissed_by_its_own_label(monkeypatch):
    executor = make_executor(monkeypatch)
    scene = make_scene([FakeLine("Close", (500, 60))])
    step = make_step(ActionType.CLICK, target="Close")

    assert executor._clear_unresolved_obstruction(step, scene, 2) == 0
    assert executor._controller.calls == []


def test_a_furti_overlay_holding_the_foreground_is_released(monkeypatch):
    executor = make_executor(monkeypatch)
    monkeypatch.setattr(
        executor_module,
        "get_foreground_window_title",
        lambda: "Furti AI - live status",
    )
    focused: list[str] = []
    monkeypatch.setattr(
        executor_module,
        "focus_window",
        lambda title: focused.append(title) or True,
    )
    step = make_step(ActionType.CLICK, target="Export", params={"window": "Notepad"})

    assert executor._clear_unresolved_obstruction(step, make_scene(), 2) == 1
    assert focused == ["Notepad"]


# ------------------------------------------------- off-screen point refusal
def make_screen(monkeypatch, rect):
    """Pin the virtual desktop rectangle for the bounds tests."""
    executor = make_executor(monkeypatch)
    monkeypatch.setattr(executor_module, "virtual_screen_rect", lambda: rect)
    return executor


def test_a_point_on_the_virtual_desktop_is_allowed(monkeypatch):
    executor = make_screen(monkeypatch, (0, 0, 1920, 1080))

    assert executor._is_point_on_screen((0, 0)) is True
    assert executor._is_point_on_screen((1919, 1079)) is True


def test_a_point_on_a_second_monitor_is_allowed(monkeypatch):
    """A display left of the primary one legitimately has negative pixels."""
    executor = make_screen(monkeypatch, (-1920, 0, 3840, 1080))

    assert executor._is_point_on_screen((-500, 400)) is True
    assert executor._is_point_on_screen((-2500, 400)) is False


def test_an_off_screen_point_is_refused(monkeypatch):
    executor = make_screen(monkeypatch, (0, 0, 1920, 1080))

    assert executor._is_point_on_screen((2400, 300)) is False
    assert executor._is_point_on_screen((300, -40)) is False


def test_an_unknown_screen_rectangle_does_not_block(monkeypatch):
    """No API answer (non-Windows, or the call failed) must fail open."""
    executor = make_screen(monkeypatch, None)

    assert executor._is_point_on_screen((99999, 99999)) is True


# ------------------------------------------------------------------- move
def test_move_action_repositions_the_cursor_without_clicking(monkeypatch):
    executor = make_executor(monkeypatch)
    step = make_step(ActionType.MOVE, params={"x": 123, "y": 456})

    executor._perform_action(step, (123, 456), None)

    assert executor._controller.calls == [("move_to", 123, 456)]


def test_click_passes_button_and_repeat_count(monkeypatch):
    executor = make_executor(monkeypatch)
    step = make_step(
        ActionType.CLICK, params={"button": "right", "clicks": "2"}
    )

    executor._perform_action(step, (10, 20), None)

    assert executor._controller.calls == [("click", 10, 20, "right", 2)]


def test_click_defaults_to_a_single_left_click(monkeypatch):
    executor = make_executor(monkeypatch)
    step = make_step(ActionType.CLICK)

    executor._perform_action(step, (10, 20), None)

    assert executor._controller.calls == [("click", 10, 20, "left", 1)]


# ------------------------------------------------------------------- drag
def test_drag_starts_at_the_current_cursor_and_drops_at_explicit_pixel(
    monkeypatch,
):
    executor = make_executor(monkeypatch, cursor=(640, 480))
    step = make_step(
        ActionType.DRAG,
        description="drag the slider to the right",
        params={"to_x": 900, "to_y": 480},
    )

    resolution = executor._resolve_target(step, make_scene(), make_plan())

    assert resolution.center == (640, 480)
    # Cursor coordinates are already mouse-space, so they must not be rescaled.
    assert resolution.capture_coordinates is False
    assert resolution.drop_point == (900, 480)
    assert resolution.drop_uses_capture_space is True


def test_drag_can_grab_at_an_explicit_origin_and_drop_by_offset(monkeypatch):
    executor = make_executor(monkeypatch)
    step = make_step(
        ActionType.DRAG,
        params={"from_x": 100, "from_y": 200, "dx": 40, "dy": -10},
    )

    resolution = executor._resolve_target(step, make_scene(), make_plan())

    assert resolution.center == (100, 200)
    assert resolution.capture_coordinates is True
    assert resolution.drop_point == (140, 190)
    # An offset is in the same space as its grab point.
    assert resolution.drop_uses_capture_space is True


def test_drag_top_level_point_is_the_grab_point(monkeypatch):
    executor = make_executor(monkeypatch)
    step = make_step(
        ActionType.DRAG, params={"x": 5, "y": 6, "to_x": 7, "to_y": 8}
    )

    resolution = executor._resolve_target(step, make_scene(), make_plan())

    assert resolution.center == (5, 6)
    assert resolution.drop_point == (7, 8)


def test_drag_with_neither_anchor_nor_drop_point_is_unresolved(monkeypatch):
    executor = make_executor(monkeypatch)
    step = make_step(ActionType.DRAG, description="drag things around")

    assert executor._resolve_target(step, make_scene(), make_plan()) is None


def test_drag_from_cursor_without_a_drop_point_is_unresolved(monkeypatch):
    executor = make_executor(monkeypatch)
    step = make_step(ActionType.DRAG, params={"x": 30, "y": 40})

    # Grabbing is the easy half; releasing at an arbitrary point is not.
    assert executor._resolve_target(step, make_scene(), make_plan()) is None


def test_drag_prefers_a_live_anchor_for_the_grab_point(monkeypatch):
    executor = make_executor(monkeypatch, cursor=(0, 0))
    step = make_step(
        ActionType.DRAG,
        target="Invoice.pdf",
        params={"to_x": 300, "to_y": 300},
    )
    scene = make_scene([FakeLine("Invoice.pdf", (150, 250))])

    resolution = executor._resolve_target(step, scene, make_plan())

    assert resolution.center == (150, 250)
    assert resolution.capture_coordinates is True
    assert resolution.drop_point == (300, 300)


# ------------------------------------------------------- coordinate spaces
def test_drop_point_inherits_the_grab_points_coordinate_space():
    from_cursor = TargetResolution((1, 1), None, "", capture_coordinates=False)
    from_capture = TargetResolution((1, 1), None, "", capture_coordinates=True)
    explicit = TargetResolution(
        (1, 1),
        None,
        "",
        capture_coordinates=False,
        drop_capture_coordinates=True,
    )

    assert from_cursor.drop_uses_capture_space is False
    assert from_capture.drop_uses_capture_space is True
    assert explicit.drop_uses_capture_space is True


def test_step_point_ignores_garbage_and_missing_axes():
    assert PlanExecutor._step_point(make_step(ActionType.CLICK, params={})) is None
    assert (
        PlanExecutor._step_point(
            make_step(ActionType.CLICK, params={"x": 5, "y": None})
        )
        is None
    )
    assert (
        PlanExecutor._step_point(
            make_step(ActionType.CLICK, params={"x": "abc", "y": 5})
        )
        is None
    )


def test_executor_module_still_exposes_the_drag_helpers():
    # Guard against the drag refactor dropping the public helpers the review
    # path relies on.
    assert hasattr(executor_module.PlanExecutor, "_resolve_drag_target")
    assert hasattr(executor_module.PlanExecutor, "_resolve_drag_endpoint")


# --------------------------------------------------- end-to-end dispatch
class StubContext:
    """Hands the executor one fixed observation, like the real context does."""

    def __init__(self, scene):
        self.scene = scene

    def observe(self, instruction, force_fresh=False):
        return self.scene


class StubLLM:
    def __init__(self, response=""):
        self.response = response

    def chat_text(self, system, user, purpose=""):
        return self.response

    def chat_vision(self, system, user, image_b64, purpose=""):
        return self.response


def make_live_executor(tmp_path, input_ctl, *, verify_steps=False):
    settings = Settings(workspace=Path(tmp_path), verify_steps=verify_steps)
    settings.ensure_dirs()
    journal = TaskJournal("points_task", Path(tmp_path) / "reports")
    scene = SceneObservation(
        frame=np.full((480, 640, 3), 200, dtype=np.uint8),
        text_lines=[],
        scene_text="",
    )
    context = StubContext(scene)
    planner = TaskPlanner(StubLLM(), settings, journal, context, threading.Event())
    executor = PlanExecutor(
        settings,
        journal,
        None,
        input_ctl,
        MemoryManager(settings.memory_file),
        planner,
        context,
        threading.Event(),
        "t",
    )
    return executor


def test_move_then_click_at_a_pixel_reaches_the_controller(tmp_path):
    input_ctl = FakeController()
    executor = make_live_executor(tmp_path, input_ctl)
    plan = TaskPlan(
        task_name="t",
        goal="",
        steps=[
            PlanStep(
                index=1,
                description="Hover the canvas",
                action=ActionType.MOVE,
                params={"x": 120, "y": 240},
            ),
            PlanStep(
                index=2,
                description="Click the canvas",
                action=ActionType.CLICK,
                params={"x": 120, "y": 240, "clicks": 1},
            ),
        ],
    )

    report = executor.execute("hover and click the canvas", plan)

    assert report.success
    assert input_ctl.calls == [
        ("move_to", 120, 240),
        ("click", 120, 240, "left", 1),
    ]


def test_drag_to_a_pixel_then_click_reaches_the_controller(tmp_path):
    input_ctl = FakeController()
    executor = make_live_executor(tmp_path, input_ctl)
    plan = TaskPlan(
        task_name="t",
        goal="",
        steps=[
            PlanStep(
                index=1,
                description="Drag the slide to the right",
                action=ActionType.DRAG,
                params={"x": 60, "y": 300, "to_x": 500, "to_y": 300},
            ),
            PlanStep(
                index=2,
                description="Click where it landed",
                action=ActionType.CLICK,
                params={"x": 500, "y": 300},
            ),
        ],
    )

    report = executor.execute("drag the slide then click it", plan)

    assert report.success
    assert input_ctl.calls == [
        ("drag", 60, 300, 500, 300, "left"),
        ("click", 500, 300, "left", 1),
    ]


def test_aliased_action_names_from_model_json_reach_the_controller(tmp_path):
    input_ctl = FakeController()
    executor = make_live_executor(tmp_path, input_ctl)
    steps = [
        PlanStep.from_dict(
            {"description": "Move there", "action": "move_to", "x": 11, "y": 22}, 0
        ),
        PlanStep.from_dict(
            {"description": "Click there", "action": "click_at", "x": 11, "y": 22}, 1
        ),
    ]

    report = executor.execute(
        "move and click", TaskPlan(task_name="t", goal="", steps=steps)
    )

    assert report.success
    assert input_ctl.calls == [
        ("move_to", 11, 22),
        ("click", 11, 22, "left", 1),
    ]


# ------------------------------------------------- post-step review guard
def make_review_executor(tmp_path, response):
    executor = make_live_executor(tmp_path, FakeController(), verify_steps=True)
    executor._context.scene.fresh = True
    executor._planner._fast = StubLLM(response)
    return executor


def test_move_step_is_never_judged_a_visual_failure(tmp_path):
    executor = make_review_executor(
        tmp_path,
        '{"step_ok": false, "step_reason": "cursor did not reach the target", '
        '"evidence_mode": "visual"}',
    )
    step = PlanStep(index=1, description="Hover", action=ActionType.MOVE)

    review = executor._review_step_and_next("hover", step, None)

    assert review.step_ok is True
    assert "no screen change expected" in review.step_reason
    assert review.attempted is True


def test_click_step_still_fails_when_the_review_says_so(tmp_path):
    executor = make_review_executor(
        tmp_path,
        '{"step_ok": false, "step_reason": "the menu did not open", '
        '"evidence_mode": "visual"}',
    )
    step = PlanStep(index=1, description="Open menu", action=ActionType.CLICK)

    review = executor._review_step_and_next("open", step, None)

    assert review.step_ok is False
    assert review.step_reason == "the menu did not open"


# ------------------------------------------------------- parallel review
class ConcurrencyProbe:
    """Records how many review agents were ever in flight at the same time."""

    def __init__(self):
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    @contextmanager
    def track(self):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            yield
        finally:
            with self._lock:
                self.active -= 1


class ProbingLLM(StubLLM):
    """Primary reviewer that reports its own overlap window."""

    def __init__(self, response, probe):
        super().__init__(response)
        self.probe = probe

    def chat_text(self, system, user, purpose=""):
        with self.probe.track():
            time.sleep(0.05)
            return self.response


class FakeMonitor:
    """Stand-in for the secondary route monitor."""

    enabled = True

    def __init__(self, probe, *, approve=True, attempted=True, reason="", error=None):
        self.probe = probe
        self.approve = approve
        self.attempted = attempted
        self.reason = reason
        self.error = error
        self.calls = []

    def review_progress(
        self, instruction, step, *, next_step=None, execution_context=""
    ):
        self.calls.append(
            {
                "instruction": instruction,
                "step": step,
                "next_step": next_step,
                "execution_context": execution_context,
            }
        )
        if self.error is not None:
            raise self.error
        with self.probe.track():
            time.sleep(0.05)
        return CrossReview(
            attempted=self.attempted,
            approved=self.approve,
            reason=self.reason,
            provider="deepseek-secondary",
        )


def make_parallel_executor(tmp_path, monitor, primary_response='{"step_ok": true}'):
    probe = ConcurrencyProbe()
    executor = make_live_executor(tmp_path, FakeController(), verify_steps=True)
    executor._context.scene.fresh = True
    executor._planner._fast = ProbingLLM(primary_response, probe)
    executor._verifier = monitor
    monitor.probe = probe
    return executor, probe


def test_step_review_and_route_monitor_run_in_parallel(tmp_path):
    """Verification must overlap, otherwise it doubles the per-step latency."""
    probe = ConcurrencyProbe()
    monitor = FakeMonitor(probe, reason="on track")
    executor, probe = make_parallel_executor(
        tmp_path, monitor, '{"step_ok": true, "step_reason": "clicked"}'
    )
    monitor.probe = probe
    step = PlanStep(index=1, description="Click Save", action=ActionType.CLICK)

    review = executor._parallel_review("save the file", step, None)

    assert probe.max_active == 2, "the two review agents never overlapped"
    assert review.step_ok is True
    assert monitor.calls and monitor.calls[0]["instruction"] == "save the file"


def test_route_monitor_rejection_fails_the_step(tmp_path):
    probe = ConcurrencyProbe()
    monitor = FakeMonitor(
        probe, approve=False, reason="a popup is covering the page"
    )
    executor, probe = make_parallel_executor(
        tmp_path, monitor, '{"step_ok": true, "step_reason": "clicked"}'
    )
    monitor.probe = probe
    step = PlanStep(index=1, description="Click Read more", action=ActionType.CLICK)

    review = executor._parallel_review("read the article", step, None)

    assert review.step_ok is False
    assert "popup" in review.step_reason
    assert review.next_step_ready is False


def test_route_monitor_approval_keeps_the_primary_verdict(tmp_path):
    probe = ConcurrencyProbe()
    monitor = FakeMonitor(probe, approve=True, reason="still on track")
    executor, probe = make_parallel_executor(
        tmp_path, monitor, '{"step_ok": false, "step_reason": "nothing changed"}'
    )
    monitor.probe = probe
    step = PlanStep(index=1, description="Click Save", action=ActionType.CLICK)

    review = executor._parallel_review("save the file", step, None)

    assert review.step_ok is False
    assert review.step_reason == "nothing changed"


def test_route_monitor_error_never_blocks_the_step(tmp_path):
    probe = ConcurrencyProbe()
    monitor = FakeMonitor(probe, error=RuntimeError("rate limited"))
    executor, probe = make_parallel_executor(
        tmp_path, monitor, '{"step_ok": true, "step_reason": "clicked"}'
    )
    monitor.probe = probe
    step = PlanStep(index=1, description="Click Save", action=ActionType.CLICK)

    review = executor._parallel_review("save the file", step, None)

    assert review.step_ok is True
    assert review.step_reason == "clicked"


def test_parallel_verify_can_be_turned_off(tmp_path):
    probe = ConcurrencyProbe()
    monitor = FakeMonitor(probe, reason="on track")
    executor, probe = make_parallel_executor(tmp_path, monitor)
    monitor.probe = probe
    executor._settings.parallel_verify = False
    step = PlanStep(index=1, description="Click Save", action=ActionType.CLICK)

    executor._parallel_review("save the file", step, None)

    assert monitor.calls == []


# ----------------------------------------------------- placeholder origin
def test_placeholder_origin_point_is_not_clicked(monkeypatch):
    """The schema's (0, 0) placeholder must never become a real click: it is
    pyautogui's abort corner and freezes every later input call."""
    executor = make_executor(monkeypatch)
    step = make_step(ActionType.CLICK, target="Nowhere", params={"x": 0, "y": 0})

    resolution = executor._resolve_target(step, make_scene(), make_plan())

    assert resolution is None


def test_explicit_origin_point_is_honoured_when_requested(monkeypatch):
    executor = make_executor(monkeypatch)
    step = make_step(
        ActionType.CLICK,
        target="Top-left canvas handle",
        params={"x": 0, "y": 0, "explicit_coordinate": True},
    )

    resolution = executor._resolve_target(step, make_scene(), make_plan())

    assert resolution is not None
    assert resolution.center == (0, 0)

