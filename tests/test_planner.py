"""Tests for the multi-step planning/execution pipeline.

Covers plan JSON parsing (defensive cases), model escalation, cost math,
loop signatures and the executor's anchor-resolution/retry/loop-detection
guardrails -- all against deterministic mocks (no network, no screen).
"""

from pathlib import Path
import threading

import numpy as np
import pytest

from furti_ai.config import Settings
from furti_ai.context import SceneObservation
from furti_ai.cost import UsageTracker
from furti_ai.executor import PlanExecutor
from furti_ai.memory import MemoryManager
from furti_ai.models import BoundingBox, ActionType, coerce_action
from furti_ai.jsoncontract import LLMJsonError
from furti_ai.ocr import TextLine
from furti_ai.planner import BudgetExceeded, PlanStep, TaskPlan, TaskPlanner
from furti_ai.tasklog import TaskJournal


class MockLLM:
    """Minimal chat_text/chat_vision client for the planner."""

    def __init__(self, response: str = "", name: str = "fast-mock"):
        self.response = response
        self._model = name
        self.call_count = 0
        self.prompts: list[str] = []

    def chat_text(self, system: str, user: str, purpose: str = "") -> str:
        self.call_count += 1
        self.prompts.append(user)
        return self.response

    def chat_vision(self, system: str, user: str, image_b64: str, purpose: str = "") -> str:
        return self.chat_text(system, user, purpose)


PLAN_JSON = """```json
{
  "goal": "Open the notes app and write a line",
  "reasoning": "Two UI actions in sequence.",
  "requires_smart_model": false,
  "steps": [
    {"description": "Click the Export button", "action": "click",
     "target": "Export", "thought": "It is top-left"},
    {"description": "Type a greeting", "action": "type",
     "target": null, "text": "hello"}
  ]
}
```"""


class FakeContext:
    """Deterministic VisualContextManager stand-in."""

    def __init__(self, observations: list[SceneObservation] | None = None):
        self.observations = observations or []
        self.calls = 0
        self.last_capture_ts = 0.0

    def observe(self, instruction: str, force_fresh: bool = False) -> SceneObservation:
        if not self.observations:
            return SceneObservation(frame=None)
        obs = self.observations[min(self.calls, len(self.observations) - 1)]
        self.calls += 1
        return obs


def make_scene(frame: np.ndarray, lines: list[TextLine]) -> SceneObservation:
    return SceneObservation(
        frame=frame,
        text_lines=lines,
        scene_text="\n".join(f"text:{l.text}" for l in lines),
    )


def make_settings(tmp_path: Path, **overrides) -> Settings:
    return Settings(workspace=Path(tmp_path), **overrides)


def make_journal(tmp_path: Path) -> TaskJournal:
    journal = TaskJournal("test_task", Path(tmp_path) / "reports")
    return journal


class FakePlanner:
    """Planner stand-in that returns the same step forever (loop test)."""

    def __init__(self, replacement: PlanStep):
        self.replacement = replacement
        self.replan_calls = 0
        self._fast = MockLLM()

    def replan_step(self, instruction, failed, reason, scene, attempt) -> PlanStep:
        self.replan_calls += 1
        return self.replacement


class AdaptivePlanner(FakePlanner):
    """Returns a different remaining route after local retries are exhausted."""

    def __init__(self, replacement: PlanStep):
        super().__init__(replacement)
        self.route_replan_calls = 0

    def replan_remaining(
        self,
        instruction,
        current_plan,
        completed_steps,
        failed_step,
        failure_reason,
        scene,
        attempt,
    ):
        self.route_replan_calls += 1
        return TaskPlan(
            task_name=current_plan.task_name,
            goal=current_plan.goal,
            steps=[self.replacement],
        )


class RecordingInput:
    def __init__(self):
        self.clicks: list[tuple[int, int]] = []
        self.moves: list[tuple[int, int]] = []
        self.typed: list[str] = []
        self.pressed: list[tuple[str, int]] = []
        self.scrolls: list[int] = []
        self.drags: list[tuple[tuple[int, int], tuple[int, int], str, object, object]] = []

    def move_to(self, x, y):
        self.moves.append((x, y))

    def click(self, x, y, button="left", clicks=1):
        self.clicks.extend((x, y) for _ in range(max(1, int(clicks))))

    def double_click(self, x, y):
        self.clicks.append((x, y))

    def right_click(self, x, y):
        self.clicks.append((x, y))

    def type_text(self, text):
        self.typed.append(text)

    def press_key(self, key, presses=1):
        self.pressed.append((key, presses))

    def scroll(self, clicks):
        self.scrolls.append(clicks)

    def drag(self, x, y, end_x, end_y, button="left", duration=None, hold_keys=None):
        self.drags.append(
            ((x, y), (end_x, end_y), button, duration, tuple(hold_keys or ()))
        )


# ------------------------------------------------------------------ planner
def test_planner_builds_plan_from_json(tmp_path):
    settings = make_settings(tmp_path)
    llm = MockLLM(response=PLAN_JSON)
    context = FakeContext(
        [SceneObservation(frame=np.zeros((10, 10, 3), dtype=np.uint8))]
    )
    planner = TaskPlanner(llm, settings, make_journal(tmp_path), context, threading.Event())

    plan = planner.plan("click export then type hello")

    assert isinstance(plan, TaskPlan)
    assert len(plan.steps) == 2
    assert plan.model_used == "fast-mock"
    assert plan.steps[0].action == ActionType.CLICK
    assert plan.steps[0].target == "Export"
    assert plan.steps[1].text == "hello"
    assert plan.frame is not None  # planning-time screenshot retained


def test_planner_restores_bbox_from_downscaled_attached_image(tmp_path):
    settings = make_settings(tmp_path)
    llm = MockLLM(
        response=(
            '{"steps": [{"description": "Click Export", "action": "click", '
            '"bbox": {"x": 10, "y": 5, "width": 20, "height": 10}, '
            '"bbox_coordinate_space": "attached_image"}]}'
        )
    )
    scene = SceneObservation(
        frame=np.zeros((100, 200, 3), dtype=np.uint8),
        image_b64="encoded",
        vision_used=True,
        frame_size=(200, 100),
        vision_size=(100, 50),
    )
    planner = TaskPlanner(
        llm,
        settings,
        make_journal(tmp_path),
        FakeContext([scene]),
        threading.Event(),
    )

    plan = planner.plan("click export")

    assert plan.steps[0].bbox == BoundingBox(20, 10, 40, 20)


def test_planner_restores_explicit_point_from_downscaled_attached_image(tmp_path):
    settings = make_settings(tmp_path)
    llm = MockLLM(
        response=(
            '{"steps": [{"description": "Click the canvas", "action": "click_at", '
            '"x": 10, "y": 5, "bbox_coordinate_space": "attached_image"}]}'
        )
    )
    scene = SceneObservation(
        frame=np.zeros((100, 200, 3), dtype=np.uint8),
        image_b64="encoded",
        vision_used=True,
        frame_size=(200, 100),
        vision_size=(100, 50),
    )
    planner = TaskPlanner(
        llm,
        settings,
        make_journal(tmp_path),
        FakeContext([scene]),
        threading.Event(),
    )

    plan = planner.plan("click the canvas")

    step = plan.steps[0]
    # The model answered in attached-image pixels; acting on them unscaled
    # would click half the intended distance.
    assert (step.params["x"], step.params["y"]) == (20, 10)
    assert step.params["point_coordinate_space"] == "full_capture"
    assert step.action is ActionType.CLICK


def test_planner_refuses_an_unknown_action_instead_of_guessing(tmp_path):
    """An unmapped action name must not silently become a click.

    The old behaviour fell back to ``ActionType.CLICK``, which turned any
    hallucinated action ("teleport", "hover_menu", ...) into a real button press
    at whatever anchor the step carried. The planner now rejects the payload,
    escalates to the smarter model, and only gives up if that fails too.
    """
    settings = make_settings(tmp_path)
    llm = MockLLM(
        response='{"steps": [{"description": "x", "action": "teleport", "target": null}]}'
    )
    planner = TaskPlanner(
        llm, settings, make_journal(tmp_path),
        FakeContext([SceneObservation(frame=np.zeros((8, 8, 3), dtype=np.uint8))]),
        threading.Event(),
    )

    with pytest.raises(RuntimeError, match="unknown action 'teleport'"):
        planner.plan("do the thing")


def test_planner_escalates_an_unknown_action_to_the_smart_model(tmp_path):
    settings = make_settings(tmp_path)
    fast = MockLLM(
        response='{"steps": [{"description": "x", "action": "teleport"}]}',
        name="fast-mock",
    )
    smart = MockLLM(response=PLAN_JSON, name="smart-mock")
    planner = TaskPlanner(
        fast, settings, make_journal(tmp_path), FakeContext([]),
        threading.Event(), smart_llm=smart,
    )

    plan = planner.plan("click export then type hello")

    assert plan.steps  # recovered with a valid plan from the smarter model
    assert smart.call_count == 1


def test_planner_escalates_to_smart_model(tmp_path):
    settings = make_settings(tmp_path)
    fast = MockLLM(response="not json at all", name="gemini-flash")
    smart = MockLLM(response=PLAN_JSON, name="gemini-pro")
    planner = TaskPlanner(
        fast, settings, make_journal(tmp_path), FakeContext([]),
        threading.Event(), smart_llm=smart,
    )
    plan = planner.plan("click export then type hello")
    # attempt 0+1 fast fail, attempt 2 escalates to smart and succeeds
    assert fast.call_count == 2
    assert smart.call_count == 1
    assert plan.model_used == "gemini-pro"


def test_planner_respects_step_cap(tmp_path):
    settings = make_settings(tmp_path, max_plan_steps=2)
    steps = [{"description": f"step {i}", "action": "click"} for i in range(5)]
    llm = MockLLM(response='{"steps": ' + str(steps).replace("'", '"') + "}")
    planner = TaskPlanner(
        llm, settings, make_journal(tmp_path), FakeContext([]), threading.Event()
    )
    plan = planner.plan("do many things")
    assert len(plan.steps) == 2


def test_planner_budget_exceeded_stops_planning(tmp_path):
    settings = make_settings(tmp_path, max_llm_calls_per_task=3)
    llm = MockLLM(response=PLAN_JSON)
    llm.call_count = 3  # simulate earlier calls having consumed the budget
    planner = TaskPlanner(
        llm, settings, make_journal(tmp_path), FakeContext([]), threading.Event()
    )
    with pytest.raises(BudgetExceeded):
        planner.plan("anything")


# --------------------------------------------------------------- signatures
def test_plan_step_signature_stable_and_sensitive():
    a = PlanStep(1, "Click the button", ActionType.CLICK, target="btn")
    b = PlanStep(1, "Click the button", ActionType.CLICK, target="btn")
    c = PlanStep(1, "Click the button", ActionType.DOUBLE_CLICK, target="btn")
    assert a.signature() == b.signature()
    assert a.signature() != c.signature()


def test_plan_step_parses_corner_list_bbox():
    # Vision models frequently ignore the dict schema and return
    # [left, top, right, bottom] corner pixels instead of x/y/width/height.
    step = PlanStep.from_dict(
        {"description": "click chrome", "action": "click",
         "target": "Chrome", "bbox": [100, 50, 160, 90]},
        0,
    )
    assert step.bbox == BoundingBox(100, 50, 60, 40)


def test_plan_step_ignores_malformed_corner_list():
    step = PlanStep.from_dict(
        {"description": "click chrome", "action": "click",
         "target": "Chrome", "bbox": [100, 50, 40, 30]},
        0,
    )
    # right <= left means the "corners" reading is nonsense; drop the box.
    assert step.bbox is None


def test_plan_step_parses_window_title():
    step = PlanStep.from_dict(
        {"description": "type the note", "action": "type",
         "text": "hello", "params": {"window": "Notepad"}},
        0,
    )
    assert step.window == "Notepad"
    assert step.params["window"] == "Notepad"

    # A top-level "window" field is also honoured.
    step = PlanStep.from_dict(
        {"description": "press enter", "action": "key_press",
         "params": {"key": "enter"}, "window": "Notepad"},
        0,
    )
    assert step.window == "Notepad"


# ------------------------------------------------------- action vocabulary
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("click", ActionType.CLICK),
        ("click_at", ActionType.CLICK),
        ("left_click", ActionType.CLICK),
        ("move_to", ActionType.MOVE),
        ("Move", ActionType.MOVE),
        ("hover", ActionType.MOVE),
        ("drag_to", ActionType.DRAG),
        ("doubleclick", ActionType.DOUBLE_CLICK),
        ("right_click", ActionType.RIGHT_CLICK),
        ("type_text", ActionType.TYPE),
        ("hotkey", ActionType.KEY_PRESS),
        ("press_key", ActionType.KEY_PRESS),
        ("ctrl+shift+t", ActionType.KEY_PRESS),
        ("mouse_wheel", ActionType.SCROLL),
        # Decorative prefixes/suffixes must not turn a hover into a click.
        ("mouse_move", ActionType.MOVE),
        ("move_mouse", ActionType.MOVE),
        ("MOVE_MOUSE", ActionType.MOVE),
        ("cursor_move", ActionType.MOVE),
        ("mouse-move", ActionType.MOVE),
        ("move the mouse", ActionType.MOVE),
        ("double_click_at", ActionType.DOUBLE_CLICK),
        ("mouse_right_click", ActionType.RIGHT_CLICK),
        ("move_pointer_to", ActionType.MOVE),
        ("hover_over", ActionType.MOVE),
        ("mouse_click", ActionType.CLICK),
    ],
)
def test_model_action_names_coerce_to_canonical_actions(raw, expected):
    assert coerce_action(raw) is expected


def test_decorated_spellings_never_silently_become_a_click():
    # The old parser mapped every unknown name onto the caller's default, so a
    # model saying "mouse_move" produced a real click instead of a hover.
    for raw in ("mouse_move", "move_mouse", "cursor_move", "move_pointer_to"):
        assert coerce_action(raw, ActionType.CLICK) is not ActionType.CLICK


def test_unknown_actions_fall_back_to_the_supplied_default():
    assert coerce_action("teleport", ActionType.CLICK) is ActionType.CLICK
    assert coerce_action(None) is None
    assert coerce_action("") is None


def test_plan_step_accepts_an_aliased_action_name():
    step = PlanStep.from_dict(
        {"description": "hover the menu", "action": "move_to", "x": 10, "y": 20},
        0,
    )

    assert step.action is ActionType.MOVE
    assert (step.params["x"], step.params["y"]) == (10, 20)


def test_plan_step_reads_points_from_every_supported_shape():
    top_level = PlanStep.from_dict(
        {"description": "click", "action": "click", "x": 12, "y": 34}, 0
    )
    point_object = PlanStep.from_dict(
        {"description": "click", "action": "click", "point": {"x": 56, "y": 78}}, 0
    )
    pair = PlanStep.from_dict(
        {"description": "click", "action": "click", "point": [90, 100]}, 0
    )

    assert (top_level.params["x"], top_level.params["y"]) == (12, 34)
    assert (point_object.params["x"], point_object.params["y"]) == (56, 78)
    assert (pair.params["x"], pair.params["y"]) == (90, 100)


def test_plan_step_reads_a_corner_box_given_as_a_point_as_its_centre():
    step = PlanStep.from_dict(
        {"description": "click", "action": "click", "point": [10, 20, 30, 40]}, 0
    )

    assert (step.params["x"], step.params["y"]) == (20, 30)


def test_params_point_wins_only_when_no_top_level_pixel_exists():
    step = PlanStep.from_dict(
        {
            "description": "click",
            "action": "click",
            "params": {"x": 1, "y": 2, "point": {"x": 3, "y": 4}},
        },
        0,
    )

    assert (step.params["x"], step.params["y"]) == (1, 2)


def test_describe_renders_move_and_coordinate_steps():
    plan = TaskPlan(
        task_name="t",
        goal="g",
        steps=[
            PlanStep(
                index=1,
                description="Hover the toolbar",
                action=ActionType.MOVE,
                params={"x": 40, "y": 60},
            ),
            PlanStep(
                index=2,
                description="Park the cursor",
                action=ActionType.MOVE,
            ),
        ],
    )

    rendered = plan.describe()

    assert "to (40, 60)" in rendered
    assert "to current cursor" in rendered


# --------------------------------------------------------------------- cost
def test_usage_tracker_math():
    tracker = UsageTracker()
    tracker.record("gemini-flash", 1000, 500, "plan")
    tracker.record("gemini-flash", 0, 250, "verify")
    summary = tracker.summary()
    assert summary.total_tokens == 1750
    from furti_ai.cost import price_for_model

    prompt_price, completion_price = price_for_model("gemini-flash")
    expected = 1000 * prompt_price / 1_000_000 + 750 * completion_price / 1_000_000
    assert summary.total_cost_usd == pytest.approx(expected, rel=1e-9)
    assert summary.calls == 2


def test_usage_tracker_unknown_model_uses_default_price():
    tracker = UsageTracker()
    tracker.record("mystery-model", 2000, 0, "plan")
    summary = tracker.summary()
    assert summary.total_cost_usd >= 0.0


# ----------------------------------------------------------------- executor
def _step(**kw) -> PlanStep:
    defaults = dict(index=1, description="click", action=ActionType.CLICK)
    defaults.update(kw)
    return PlanStep(**defaults)


def test_executor_resolves_ocr_anchor_and_click(tmp_path):
    settings = make_settings(tmp_path, verify_steps=False, max_step_retries=2)
    settings.ensure_dirs()
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    line = TextLine("Export", BoundingBox(300, 200, 100, 40), 0.99)
    context = FakeContext([make_scene(frame, [line])])
    input_ctl = RecordingInput()
    memory = MemoryManager(settings.memory_file)
    journal = make_journal(tmp_path)
    planner = TaskPlanner(MockLLM(), settings, journal, context, threading.Event())
    executor = PlanExecutor(
        settings, journal, None, input_ctl, memory, planner, context,
        threading.Event(), "t",
    )
    plan = TaskPlan(
        task_name="t", goal="", steps=[_step(target="Export", description="Click Export")]
    )
    report = executor.execute("click export", plan)

    assert report.success
    assert input_ctl.clicks == [(350, 220)]
    assert report.results[0].action_dispatched is True
    assert report.results[0].visually_verified is None
    assert any(event.kind == "CONFIRM" for event in journal._events)
    # a template was saved and a reflex compiled into memory
    assert memory.has_skill("click_export")
    assert list(settings.templates_dir.glob("*.png"))


# ------------------------------------------- rule 1: focus before typing
TYPED_FIELD_PLAN_JSON = """```json
{
  "goal": "Search for invoices",
  "reasoning": "Type the query into the search box.",
  "requires_smart_model": false,
  "steps": [
    {"description": "Type the query into the search box", "action": "type",
     "target": "Search", "text": "invoices", "params": {"window": "Chrome"}}
  ]
}
```"""

PRECLICKED_FIELD_PLAN_JSON = """```json
{
  "goal": "Search for invoices",
  "reasoning": "Click the box, then type.",
  "requires_smart_model": false,
  "steps": [
    {"description": "Click the search box", "action": "click",
     "target": "Search"},
    {"description": "Type the query", "action": "type",
     "target": "Search", "text": "invoices"}
  ]
}
```"""


def make_plan_from(tmp_path, plan_json: str):
    """Run one planning pass over a canned model response."""
    settings = make_settings(tmp_path)
    journal = make_journal(tmp_path)
    planner = TaskPlanner(
        MockLLM(response=plan_json),
        settings,
        journal,
        FakeContext(
            [SceneObservation(frame=np.zeros((10, 10, 3), dtype=np.uint8))]
        ),
        threading.Event(),
    )
    return planner.plan("search for invoices"), journal


def test_plan_split_the_focus_click_out_of_a_typed_field(tmp_path):
    plan, journal = make_plan_from(tmp_path, TYPED_FIELD_PLAN_JSON)

    assert [step.action for step in plan.steps] == [ActionType.CLICK, ActionType.TYPE]
    assert [step.index for step in plan.steps] == [1, 2]
    # The click keeps the field anchor (and its window) so it is still resolved
    # on the live screen, exactly like the type step it protects.
    assert plan.steps[0].target == "Search"
    assert plan.steps[0].window == "Chrome"
    assert plan.steps[1].text == "invoices"
    assert "focus" in plan.steps[0].description.lower()
    thoughts = [
        event.message for event in journal._events if event.kind == "THOUGHT"
    ]
    assert any("Focus-before-typing rule" in message for message in thoughts)


def test_plan_does_not_add_a_second_click_when_one_is_already_planned(tmp_path):
    plan, _journal = make_plan_from(tmp_path, PRECLICKED_FIELD_PLAN_JSON)

    assert [step.action for step in plan.steps] == [ActionType.CLICK, ActionType.TYPE]
    assert len(plan.steps) == 2


def test_plan_leaves_a_targetless_type_step_alone(tmp_path):
    # PLAN_JSON's type step has target=None: it types into the focused control
    # on purpose, so nothing is inserted for it.
    plan, _journal = make_plan_from(tmp_path, PLAN_JSON)

    assert [step.action for step in plan.steps] == [ActionType.CLICK, ActionType.TYPE]
    assert len(plan.steps) == 2


# ------------------------------------- rule: strict control JSON contract
def test_plan_step_refuses_an_unknown_action():
    with pytest.raises(LLMJsonError, match="unknown action 'teleport'"):
        PlanStep.from_dict({"description": "x", "action": "teleport"}, 1)


def test_plan_step_refuses_a_missing_action():
    with pytest.raises(LLMJsonError, match="unknown action"):
        PlanStep.from_dict({"description": "x"}, 1)


def test_plan_step_still_accepts_documented_alias_spellings():
    step = PlanStep.from_dict(
        {"description": "move there", "action": "mouse_move", "x": 10, "y": 20}, 1
    )

    assert step.action is ActionType.MOVE
    assert step.params["x"] == 10


def test_plan_step_refuses_negative_pixels():
    with pytest.raises(LLMJsonError, match="below the allowed minimum"):
        PlanStep.from_dict(
            {"description": "click", "action": "click", "x": -40, "y": 10}, 1
        )


def test_plan_step_refuses_a_non_numeric_pixel():
    with pytest.raises(LLMJsonError, match="must be a number"):
        PlanStep.from_dict(
            {"description": "click", "action": "click", "x": "left", "y": 10}, 1
        )


def test_plan_step_refuses_an_absurd_pixel():
    with pytest.raises(LLMJsonError, match="above the allowed maximum"):
        PlanStep.from_dict(
            {"description": "click", "action": "click", "x": 500000, "y": 10}, 1
        )


def test_executor_refuses_a_target_that_is_off_screen(tmp_path, monkeypatch):
    """A resolved point outside the virtual desktop is never dispatched.

    This is the payload-level version of the same worry: if a mis-scaled or
    hallucinated pixel reaches dispatch, the cursor is thrown off-screen and
    every later click lands somewhere unintended.
    """
    import furti_ai.executor as executor_module

    settings = make_settings(tmp_path, verify_steps=False, max_step_retries=0)
    settings.ensure_dirs()
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    context = FakeContext([make_scene(frame, [])])
    input_ctl = RecordingInput()
    memory = MemoryManager(settings.memory_file)
    journal = make_journal(tmp_path)
    planner = TaskPlanner(MockLLM(), settings, journal, context, threading.Event())
    executor = PlanExecutor(
        settings, journal, None, input_ctl, memory, planner, context,
        threading.Event(), "t",
    )
    monkeypatch.setattr(
        executor_module, "virtual_screen_rect", lambda: (0, 0, 300, 200)
    )
    plan = TaskPlan(
        task_name="t",
        goal="",
        steps=[
            _step(
                description="Click at the far corner",
                params={"x": 4000, "y": 3000},
            )
        ],
    )

    report = executor.execute("click far away", plan)

    assert not report.success
    assert input_ctl.clicks == []
    assert input_ctl.moves == []
    assert any(
        "outside the screen" in note for note in report.results[0].notes
    )


def test_plan_step_refuses_a_negative_bbox_size():
    with pytest.raises(LLMJsonError, match="is negative"):
        PlanStep.from_dict(
            {
                "description": "click",
                "action": "click",
                "bbox": {"x": 10, "y": 10, "width": -5, "height": 5},
            },
            1,
        )


def test_plan_step_refuses_a_bbox_with_junk_numbers():
    with pytest.raises(LLMJsonError, match="bbox must contain integer"):
        PlanStep.from_dict(
            {
                "description": "click",
                "action": "click",
                "bbox": {"x": "a", "y": 1, "width": 5, "height": 5},
            },
            1,
        )


def test_plan_step_treats_a_zero_bbox_as_no_bbox():
    """The documented placeholder must stay a placeholder, not an error."""
    step = PlanStep.from_dict(
        {
            "description": "click save",
            "action": "click",
            "target": "Save",
            "bbox": {"x": 0, "y": 0, "width": 0, "height": 0},
        },
        1,
    )

    assert step.bbox is None


def test_plan_step_refuses_a_type_without_text():
    with pytest.raises(LLMJsonError, match="must carry the text"):
        PlanStep.from_dict(
            {
                "description": "type the query",
                "action": "type",
                "target": "Search",
                "text": "   ",
            },
            1,
        )


def test_plan_step_reads_text_from_params_too():
    step = PlanStep.from_dict(
        {"description": "type", "action": "type", "params": {"text": "hello"}}, 1
    )

    assert step.text == "hello"


def test_plan_step_refuses_a_params_that_is_not_an_object():
    with pytest.raises(LLMJsonError, match="params must be a JSON object"):
        PlanStep.from_dict(
            {"description": "click", "action": "click", "params": [1, 2]}, 1
        )


def test_planner_escalates_a_contract_violation_to_the_smart_model(tmp_path):
    settings = make_settings(tmp_path)
    fast = MockLLM(
        response='{"steps": [{"description": "x", "action": "click", "x": -5}]}',
        name="fast-mock",
    )
    smart = MockLLM(response=PLAN_JSON, name="smart-mock")
    planner = TaskPlanner(
        fast, settings, make_journal(tmp_path), FakeContext([]),
        threading.Event(), smart_llm=smart,
    )

    plan = planner.plan("click export then type hello")

    assert plan.steps
    assert smart.call_count == 1


def test_executor_clicks_a_typed_field_before_typing(tmp_path):
    """A "type at (x, y)" step must focus the field, not the last window."""
    settings = make_settings(tmp_path, verify_steps=False, max_step_retries=0)
    settings.ensure_dirs()
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    line = TextLine("Search", BoundingBox(200, 100, 80, 30), 0.99)
    context = FakeContext([make_scene(frame, [line])])
    input_ctl = RecordingInput()
    memory = MemoryManager(settings.memory_file)
    journal = make_journal(tmp_path)
    planner = TaskPlanner(MockLLM(), settings, journal, context, threading.Event())
    executor = PlanExecutor(
        settings, journal, None, input_ctl, memory, planner, context,
        threading.Event(), "t",
    )
    plan = TaskPlan(
        task_name="t",
        goal="",
        steps=[
            _step(
                action=ActionType.TYPE,
                text="hello",
                target="Search",
                description="Type hello into the search box",
            )
        ],
    )

    report = executor.execute("type hello", plan)

    assert report.success
    # The click lands on the resolved anchor centre (240, 115) first, so the
    # keystrokes go to the field rather than whatever had focus.
    assert input_ctl.clicks == [(240, 115)]
    assert input_ctl.typed == ["hello"]


def test_executor_types_without_clicking_when_no_field_is_named(tmp_path):
    settings = make_settings(tmp_path, verify_steps=False, max_step_retries=0)
    settings.ensure_dirs()
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    context = FakeContext([make_scene(frame, [])])
    input_ctl = RecordingInput()
    memory = MemoryManager(settings.memory_file)
    journal = make_journal(tmp_path)
    planner = TaskPlanner(MockLLM(), settings, journal, context, threading.Event())
    executor = PlanExecutor(
        settings, journal, None, input_ctl, memory, planner, context,
        threading.Event(), "t",
    )
    plan = TaskPlan(
        task_name="t",
        goal="",
        steps=[_step(action=ActionType.TYPE, text="hello")],
    )

    report = executor.execute("type hello", plan)

    assert report.success
    assert input_ctl.clicks == []  # nothing on screen was named
    assert input_ctl.typed == ["hello"]


def test_executor_dismisses_a_popup_covering_the_target(tmp_path, monkeypatch):
    """A modal over the target is dismissed before the target is touched."""
    import furti_ai.executor as executor_module

    settings = make_settings(tmp_path, verify_steps=False, max_step_retries=0)
    settings.ensure_dirs()
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    export = TextLine("Export", BoundingBox(300, 200, 100, 40), 0.99)
    close = TextLine("Close", BoundingBox(500, 60, 60, 24), 0.99)
    context = FakeContext([make_scene(frame, [export, close])])
    state = {"covered": True}

    class PopupInput(RecordingInput):
        """Clicking Close really does remove the popup."""

        def click(self, x, y, button="left", clicks=1):
            super().click(x, y, button, clicks)
            state["covered"] = False

    input_ctl = PopupInput()
    memory = MemoryManager(settings.memory_file)
    journal = make_journal(tmp_path)
    planner = TaskPlanner(MockLLM(), settings, journal, context, threading.Event())
    executor = PlanExecutor(
        settings, journal, None, input_ctl, memory, planner, context,
        threading.Event(), "t",
    )
    monkeypatch.setattr(
        executor_module, "find_window", lambda title, substring=True: 111
    )
    monkeypatch.setattr(
        executor_module,
        "window_at",
        lambda x, y: (999, "Cookie consent")
        if state["covered"]
        else (111, "Google Chrome"),
    )
    monkeypatch.setattr(executor_module, "window_rect", lambda hwnd: (0, 0, 300, 200))
    plan = TaskPlan(
        task_name="t",
        goal="",
        steps=[
            _step(
                target="Export",
                description="Click Export",
                params={"window": "Chrome"},
            )
        ],
    )

    report = executor.execute("click export", plan)

    assert report.success
    # Close first (530, 72), then the real target once the overlay is gone.
    assert input_ctl.clicks == [(530, 72), (350, 220)]


def test_executor_closes_an_overlay_that_hid_the_target(tmp_path):
    """A covered target reads as "not found": dismiss, then retry."""
    settings = make_settings(tmp_path, verify_steps=False, max_step_retries=2)
    settings.ensure_dirs()
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    close = TextLine("Close", BoundingBox(500, 60, 60, 24), 0.99)
    export = TextLine("Export", BoundingBox(300, 200, 100, 40), 0.99)
    context = FakeContext(
        [
            make_scene(frame, [close]),  # attempt 1: the overlay hides Export
            make_scene(frame, [export]),  # after dismissal: Export is reachable
        ]
    )
    input_ctl = RecordingInput()
    memory = MemoryManager(settings.memory_file)
    journal = make_journal(tmp_path)
    planner = TaskPlanner(MockLLM(), settings, journal, context, threading.Event())
    executor = PlanExecutor(
        settings, journal, None, input_ctl, memory, planner, context,
        threading.Event(), "t",
    )
    plan = TaskPlan(
        task_name="t",
        goal="",
        steps=[_step(target="Export", description="Click Export")],
    )

    report = executor.execute("click export", plan)

    assert report.success
    assert input_ctl.clicks == [(530, 72), (350, 220)]


def test_a_planned_focus_click_is_not_repeated_by_the_type_step(tmp_path):
    """click(field) -> type(field) presses the field once, not twice."""
    settings = make_settings(tmp_path, verify_steps=False, max_step_retries=0)
    settings.ensure_dirs()
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    search = TextLine("Search", BoundingBox(200, 100, 80, 30), 0.99)
    context = FakeContext([make_scene(frame, [search])])
    input_ctl = RecordingInput()
    memory = MemoryManager(settings.memory_file)
    journal = make_journal(tmp_path)
    planner = TaskPlanner(MockLLM(), settings, journal, context, threading.Event())
    executor = PlanExecutor(
        settings, journal, None, input_ctl, memory, planner, context,
        threading.Event(), "t",
    )
    plan = TaskPlan(
        task_name="t",
        goal="",
        steps=[
            _step(target="Search", description="Click Search to focus it"),
            _step(
                index=2,
                action=ActionType.TYPE,
                target="Search",
                text="invoices",
                description="Type the query",
            ),
        ],
    )

    report = executor.execute("type the query", plan)

    assert report.success
    assert input_ctl.clicks == [(240, 115)]  # the focus click, exactly once
    assert input_ctl.typed == ["invoices"]


def test_executor_maps_capture_anchor_to_input_coordinates(tmp_path):
    class ScaledVision:
        def to_input_point(self, point, _frame_shape):
            return point[0] // 2, point[1] // 2

    settings = make_settings(tmp_path, verify_steps=False)
    settings.ensure_dirs()
    frame = np.full((100, 200, 3), 200, dtype=np.uint8)
    line = TextLine("Export", BoundingBox(100, 40, 40, 20), 0.99)
    context = FakeContext([make_scene(frame, [line])])
    input_ctl = RecordingInput()
    journal = make_journal(tmp_path)
    planner = TaskPlanner(
        MockLLM(),
        settings,
        journal,
        context,
        threading.Event(),
    )
    executor = PlanExecutor(
        settings,
        journal,
        ScaledVision(),
        input_ctl,
        MemoryManager(settings.memory_file),
        planner,
        context,
        threading.Event(),
        "t",
    )

    report = executor.execute(
        "click export",
        TaskPlan(
            task_name="t",
            goal="",
            steps=[_step(target="Export", description="Click Export")],
        ),
    )

    assert report.success
    assert input_ctl.clicks == [(60, 25)]


def test_executor_does_not_match_one_letter_ocr_noise_as_descriptive_target(
    tmp_path,
):
    settings = make_settings(tmp_path, verify_steps=False)
    settings.ensure_dirs()
    frame = np.full((100, 200, 3), 200, dtype=np.uint8)
    noise = TextLine("A", BoundingBox(10, 10, 8, 12), 0.99)
    chrome = TextLine("Chrome", BoundingBox(120, 40, 50, 20), 0.90)
    context = FakeContext([make_scene(frame, [noise, chrome])])
    input_ctl = RecordingInput()
    journal = make_journal(tmp_path)
    planner = TaskPlanner(
        MockLLM(),
        settings,
        journal,
        context,
        threading.Event(),
    )
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

    report = executor.execute(
        "open browser",
        TaskPlan(
            task_name="t",
            goal="",
            steps=[
                _step(
                    target="Chrome icon on taskbar",
                    description="Click Chrome",
                )
            ],
        ),
    )

    assert report.success
    assert input_ctl.clicks == [(145, 50)]


def _ocr_scene(*texts: str) -> SceneObservation:
    lines = [
        TextLine(text, BoundingBox(10 + i * 60, 20, 50, 20), 0.9)
        for i, text in enumerate(texts)
    ]
    return make_scene(np.zeros((80, 400, 3), dtype=np.uint8), lines)


# ------------------------------------------------------------------- drags
def _drag_executor(settings, context, input_ctl, journal, memory=None, planner=None):
    if planner is None:
        planner = TaskPlanner(
            MockLLM(), settings, journal, context, threading.Event()
        )
    return PlanExecutor(
        settings,
        journal,
        None,
        input_ctl,
        memory or MemoryManager(settings.memory_file),
        planner,
        context,
        threading.Event(),
        "t",
    )


def test_executor_drags_anchor_to_a_named_drop_target(tmp_path):
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    grab = TextLine("Report.txt", BoundingBox(300, 200, 100, 40), 0.99)
    drop = TextLine("Trash", BoundingBox(60, 40, 80, 30), 0.99)
    context = FakeContext([make_scene(frame, [grab, drop])])
    input_ctl = RecordingInput()
    settings = make_settings(tmp_path, verify_steps=False)
    settings.ensure_dirs()
    journal = make_journal(tmp_path)
    memory = MemoryManager(settings.memory_file)
    executor = _drag_executor(settings, context, input_ctl, journal, memory)
    step = _step(
        action=ActionType.DRAG,
        target="Report.txt",
        description="Drag Report.txt to Trash",
        params={"to_target": "Trash"},
    )

    report = executor.execute(
        "drag the report to the trash", TaskPlan(task_name="t", goal="", steps=[step])
    )

    assert report.success
    assert input_ctl.drags == [((350, 220), (100, 55), "left", None, ())]
    # The reflex replays in input space, so the delta must be stored there.
    skill = memory.get_skill("drag_report_txt_to_trash")
    assert skill is not None
    assert skill.metadata["drag_delta"] == [-250, -165]
    assert skill.metadata["drag_button"] == "left"


def test_executor_drag_offset_drops_relative_to_the_grab_point(tmp_path):
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    grab = TextLine("Slider", BoundingBox(100, 100, 60, 20), 0.99)
    context = FakeContext([make_scene(frame, [grab])])
    input_ctl = RecordingInput()
    journal = make_journal(tmp_path)
    settings = make_settings(tmp_path, verify_steps=False)
    settings.ensure_dirs()
    executor = _drag_executor(settings, context, input_ctl, journal)
    step = _step(
        action=ActionType.DRAG,
        target="Slider",
        description="Nudge the slider",
        params={"dx": 40, "dy": -25, "button": "right", "hold_keys": "shift"},
    )

    report = executor.execute(
        "nudge the slider", TaskPlan(task_name="t", goal="", steps=[step])
    )

    assert report.success
    assert input_ctl.drags == [((130, 110), (170, 85), "right", None, ("shift",))]


def test_executor_fails_a_drag_whose_drop_target_is_not_on_screen(tmp_path):
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    grab = TextLine("Report.txt", BoundingBox(300, 200, 100, 40), 0.99)
    context = FakeContext([make_scene(frame, [grab])])
    input_ctl = RecordingInput()
    journal = make_journal(tmp_path)
    settings = make_settings(tmp_path, verify_steps=False)
    settings.ensure_dirs()
    step = _step(
        action=ActionType.DRAG,
        target="Report.txt",
        description="Drag Report.txt to Archive",
        params={"to_target": "Archive"},
    )
    executor = _drag_executor(
        settings, context, input_ctl, journal, planner=FakePlanner(step)
    )

    report = executor.execute(
        "file the report", TaskPlan(task_name="t", goal="", steps=[step])
    )

    # Dropping a payload at a guessed coordinate is worse than not dragging.
    assert not report.success
    assert input_ctl.drags == []


def test_executor_drag_without_a_drop_hint_is_reported_as_a_failure(tmp_path):
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    grab = TextLine("Report.txt", BoundingBox(300, 200, 100, 40), 0.99)
    context = FakeContext([make_scene(frame, [grab])])
    input_ctl = RecordingInput()
    journal = make_journal(tmp_path)
    settings = make_settings(tmp_path, verify_steps=False)
    settings.ensure_dirs()
    step = _step(
        action=ActionType.DRAG,
        target="Report.txt",
        description="Drag Report.txt somewhere",
        params={},
    )
    executor = _drag_executor(
        settings, context, input_ctl, journal, planner=FakePlanner(step)
    )

    report = executor.execute(
        "drag the report", TaskPlan(task_name="t", goal="", steps=[step])
    )

    assert not report.success
    assert input_ctl.drags == []


def test_key_press_step_repeats_the_chord(tmp_path):
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    context = FakeContext([make_scene(frame, [])])
    input_ctl = RecordingInput()
    journal = make_journal(tmp_path)
    settings = make_settings(tmp_path, verify_steps=False)
    settings.ensure_dirs()
    executor = _drag_executor(settings, context, input_ctl, journal)
    step = _step(
        action=ActionType.KEY_PRESS,
        description="Cycle tabs",
        params={"key": "ctrl+tab", "presses": 3},
    )

    report = executor.execute(
        "cycle tabs", TaskPlan(task_name="t", goal="", steps=[step])
    )

    assert report.success
    assert input_ctl.pressed == [("ctrl+tab", 3)]


def test_key_press_step_extracts_key_from_target(tmp_path):
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    context = FakeContext([make_scene(frame, [])])
    input_ctl = RecordingInput()
    journal = make_journal(tmp_path)
    settings = make_settings(tmp_path, verify_steps=False)
    settings.ensure_dirs()
    executor = _drag_executor(settings, context, input_ctl, journal)
    # The route-replan prompt historically dropped params.key and put the
    # keystroke in the target prose.
    step = _step(
        action=ActionType.KEY_PRESS,
        description="Open the Start menu",
        target="Windows key (Start menu)",
    )

    report = executor.execute(
        "open start menu", TaskPlan(task_name="t", goal="", steps=[step])
    )

    assert report.success
    assert input_ctl.pressed == [("win", 1)]


def test_executor_clicks_explicit_bbox_center_without_any_anchor(tmp_path):
    settings = make_settings(tmp_path, verify_steps=False)
    settings.ensure_dirs()
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    context = FakeContext([make_scene(frame, [])])
    input_ctl = RecordingInput()
    journal = make_journal(tmp_path)
    executor = _drag_executor(settings, context, input_ctl, journal)
    # No OCR line, no icon, no plan frame -> the explicit pixel box is the
    # only anchor left. The click must land on its centre.
    step = _step(
        action=ActionType.CLICK,
        target="Chrome taskbar icon",
        bbox=BoundingBox(100, 40, 60, 20),
    )

    report = executor.execute(
        "click chrome", TaskPlan(task_name="t", goal="", steps=[step])
    )

    assert report.success
    assert input_ctl.clicks == [(130, 50)]


def test_best_ocr_target_ignores_word_fragments_inside_a_longer_word():
    # "none" must not match inside "nonexistent": that false anchor used to
    # click an unrelated label on the desktop.
    scene = _ocr_scene("None)", "Untitled")

    assert PlanExecutor._best_ocr_target("zzzzz nonexistent widget", scene) is None


def test_best_ocr_target_still_matches_whole_word_containment():
    scene = _ocr_scene("Save as", "savegame")

    match = PlanExecutor._best_ocr_target("save", scene)

    assert match is not None
    assert match[0].text == "Save as"


def test_best_ocr_target_needs_majority_word_overlap():
    weak = _ocr_scene("report view")

    assert PlanExecutor._best_ocr_target("export report dialog", weak) is None

    strong = _ocr_scene("export report view")
    match = PlanExecutor._best_ocr_target("export report dialog", strong)

    assert match is not None
    assert match[0].text == "export report view"


def test_executor_confirms_dispatch_without_waiting_for_a_review_frame(tmp_path):
    settings = make_settings(tmp_path, verify_steps=True, max_step_retries=1)
    settings.ensure_dirs()
    frame = np.full((120, 160, 3), 200, dtype=np.uint8)
    line = TextLine("Save", BoundingBox(40, 40, 40, 20), 0.99)
    context = FakeContext([make_scene(frame, [line])])
    input_ctl = RecordingInput()
    journal = make_journal(tmp_path)
    planner = TaskPlanner(
        MockLLM(response='{"ok": true}'),
        settings,
        journal,
        context,
        threading.Event(),
    )
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

    report = executor.execute(
        "click save",
        TaskPlan(
            task_name="t",
            goal="",
            steps=[_step(target="Save", description="Click Save")],
        ),
    )

    assert report.success
    assert report.results[0].action_dispatched is True
    assert report.results[0].visually_verified is None
    assert any(
        "post-step review could not capture a fresh frame" in note
        for note in report.results[0].notes
    )
    assert planner._fast.call_count == 0


def test_executor_reviews_current_and_next_step_in_one_api_call(tmp_path):
    settings = make_settings(tmp_path, verify_steps=True, max_step_retries=1)
    settings.ensure_dirs()
    frame = np.full((120, 160, 3), 200, dtype=np.uint8)
    line = TextLine("Save", BoundingBox(40, 40, 40, 20), 0.99)
    review_json = (
        '{"step_ok": true, "step_reason": "button is visible", '
        '"evidence_mode": "both", "visual_required": true, '
        '"next_step": {"ready": true, "reason": "field is ready", '
        '"guidance": "continue"}}'
    )
    context = FakeContext(
        [
            SceneObservation(
                frame=frame,
                text_lines=[line],
                fresh=True,
                image_b64="current-frame",
            ),
            SceneObservation(
                frame=frame,
                text_lines=[line],
                fresh=True,
                image_b64="after-save",
            ),
            SceneObservation(
                frame=frame,
                text_lines=[line],
                fresh=True,
                image_b64="next-frame",
            ),
            SceneObservation(
                frame=frame,
                text_lines=[line],
                fresh=True,
                image_b64="after-type",
            ),
        ]
    )
    input_ctl = RecordingInput()
    journal = make_journal(tmp_path)
    llm = MockLLM(response=review_json)
    planner = TaskPlanner(llm, settings, journal, context, threading.Event())
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

    report = executor.execute(
        "save and type",
        TaskPlan(
            task_name="t",
            goal="",
            steps=[
                _step(index=1, target="Save", description="Click Save"),
                _step(
                    index=2,
                    action=ActionType.TYPE,
                    text="hello",
                    description="Type greeting",
                ),
            ],
        ),
    )

    assert report.success
    assert llm.call_count == 2
    assert all("step_review" in event.message for event in journal._events if event.kind == "AI_OUTPUT")
    assert report.results[0].next_step_ready is True
    assert report.results[0].visually_verified is True
    assert "Next planned step 2" in llm.prompts[0]


def test_executor_replaces_failed_route_without_replaying_completed_steps(tmp_path):
    settings = make_settings(
        tmp_path,
        verify_steps=False,
        max_step_retries=0,
        max_plan_replans=1,
    )
    settings.ensure_dirs()
    frame = np.full((120, 200, 3), 200, dtype=np.uint8)
    missing = TextLine("Old target", BoundingBox(10, 10, 40, 20), 0.99)
    alternate = TextLine("Alternate", BoundingBox(100, 50, 60, 20), 0.99)
    context = FakeContext(
        [
            make_scene(frame, []),
            make_scene(frame, [alternate]),
            make_scene(frame, [alternate]),
        ]
    )
    input_ctl = RecordingInput()
    journal = make_journal(tmp_path)
    replacement = _step(
        target="Alternate",
        description="Use alternate route",
    )
    planner = AdaptivePlanner(replacement)
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
    initial = TaskPlan(
        task_name="t",
        goal="",
        steps=[
            _step(index=1, target="Old target", description="Use old route"),
            _step(index=2, target="Never reached", description="Old next step"),
        ],
    )

    report = executor.execute("complete task", initial)

    assert report.success
    assert planner.route_replan_calls == 1
    assert input_ctl.clicks == [(130, 60)]
    assert report.results[0].superseded is True
    assert report.results[-1].success is True
    assert len(journal.plan_revisions) == 1
    assert journal.plan_revisions[0]["steps"][0]["description"] == (
        "Use alternate route"
    )


def test_planner_logs_thinking_waiting_and_response(tmp_path):
    settings = make_settings(tmp_path)
    journal = make_journal(tmp_path)
    planner = TaskPlanner(
        MockLLM(response=PLAN_JSON),
        settings,
        journal,
        FakeContext([SceneObservation(frame=np.zeros((10, 10, 3), dtype=np.uint8))]),
        threading.Event(),
    )

    planner.plan("click export")

    kinds = [event.kind for event in journal._events]
    assert "THOUGHT" in kinds
    assert "WAIT" in kinds
    assert kinds.index("WAIT") < max(
        index for index, kind in enumerate(kinds) if kind == "THOUGHT"
    )


def test_executor_loop_detection_aborts_step(tmp_path):
    settings = make_settings(tmp_path, verify_steps=False, max_step_retries=2)
    settings.ensure_dirs()
    # The screen never contains the anchor.
    context = FakeContext([make_scene(np.full((100, 100, 3), 10, dtype=np.uint8), [])])
    input_ctl = RecordingInput()
    memory = MemoryManager(settings.memory_file)
    journal = make_journal(tmp_path)
    step = _step(target="Missing Button", description="Click Missing")
    fake_planner = FakePlanner(step)  # replan returns the identical step
    executor = PlanExecutor(
        settings, journal, None, input_ctl, memory, fake_planner, context,
        threading.Event(), "t",
    )
    plan = TaskPlan(task_name="t", goal="", steps=[step])
    report = executor.execute("click missing", plan)

    assert not report.success
    result = report.results[0]
    assert not result.success
    assert "loop detected" in result.notes[-1]
    assert input_ctl.clicks == []
    assert fake_planner.replan_calls == 1  # detected before looping forever


def test_executor_stop_event_raises_task_aborted(tmp_path):
    settings = make_settings(tmp_path, verify_steps=False)
    settings.ensure_dirs()
    stop = threading.Event()
    stop.set()  # already stopped before execution begins
    context = FakeContext([])
    input_ctl = RecordingInput()
    journal = make_journal(tmp_path)
    planner = TaskPlanner(MockLLM(), settings, journal, context, stop)
    executor = PlanExecutor(
        settings, journal, None, input_ctl, MemoryManager(settings.memory_file),
        planner, context, stop, "t",
    )
    from furti_ai.context import TaskAborted

    with pytest.raises(TaskAborted):
        executor.execute("anything", TaskPlan("t", "", [_step()]))


def test_executor_max_consecutive_failures_aborts(tmp_path):
    settings = make_settings(
        tmp_path, verify_steps=False, max_step_retries=1, max_consecutive_failures=1
    )
    settings.ensure_dirs()
    context = FakeContext([make_scene(np.full((50, 50, 3), 5, dtype=np.uint8), [])])
    input_ctl = RecordingInput()
    memory = MemoryManager(settings.memory_file)
    journal = make_journal(tmp_path)
    missing = _step(target="Nope", description="Click Nope")
    executor = PlanExecutor(
        settings, journal, None, input_ctl, memory, FakePlanner(missing), context,
        threading.Event(), "t",
    )
    plan = TaskPlan(
        task_name="t", goal="", steps=[missing, _step(index=2, target="Also Nope")]
    )
    report = executor.execute("do things", plan)

    assert report.aborted
    assert len(report.results) == 1  # second step never ran


# ----------------------------------------------------------------- task agent
def _build_test_agent(tmp_path, monkeypatch, answer: str):
    from furti_ai.agent import TaskAgent
    from furti_ai.cost import UsageTracker

    settings = make_settings(tmp_path, verify_steps=False, enable_status_window=False)
    settings.ensure_dirs()
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    line = TextLine("Export", BoundingBox(300, 200, 100, 40), 0.99)
    context = FakeContext([make_scene(frame, [line])])
    input_ctl = RecordingInput()
    memory = MemoryManager(settings.memory_file)
    journal = make_journal(tmp_path)
    llm = MockLLM(response=PLAN_JSON)
    stop = threading.Event()
    planner = TaskPlanner(llm, settings, journal, context, stop)
    executor = PlanExecutor(
        settings, journal, None, input_ctl, memory, planner, context, stop, "t"
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": answer)
    agent = TaskAgent(
        settings, journal, planner, executor, UsageTracker(), None, None, stop
    )
    return agent, settings, input_ctl


def test_task_agent_requires_confirmation_and_declines(tmp_path, monkeypatch):
    agent, _, input_ctl = _build_test_agent(tmp_path, monkeypatch, "n")
    assert agent.run_task("click export then type hello") is False
    assert input_ctl.clicks == []  # plan shown but nothing executed


def test_task_report_path_is_safe_for_windows_task_names(tmp_path):
    journal = TaskJournal('save: "draft"?', Path(tmp_path) / "reports")

    assert journal.report_path().name == "save___draft.md"


def test_task_agent_confirms_runs_and_reports(tmp_path, monkeypatch):
    agent, settings, input_ctl = _build_test_agent(tmp_path, monkeypatch, "y")
    assert agent.run_task("click export then type hello") is True
    assert input_ctl.clicks == [(350, 220)]
    assert input_ctl.typed == ["hello"]
    # the <task_name>.md report and a compiled reflex template exist
    reports = list(settings.reports_dir.glob("*.md"))
    assert reports
    assert list(settings.templates_dir.glob("*.png"))


def test_task_agent_can_use_non_console_confirmation_callback(tmp_path, monkeypatch):
    agent, settings, input_ctl = _build_test_agent(tmp_path, monkeypatch, "n")
    decisions: list[TaskPlan] = []

    def approve(plan: TaskPlan) -> bool:
        decisions.append(plan)
        return True

    agent._confirmation_callback = approve

    assert agent.run_task("click export then type hello") is True
    assert len(decisions) == 1
    assert input_ctl.clicks == [(350, 220)]
    assert input_ctl.typed == ["hello"]
    assert list(settings.reports_dir.glob("*.md"))


def test_task_agent_extracts_model_user_choice():
    from furti_ai.agent import TaskAgent, UserChoiceRequest

    plan = TaskPlan(
        "task",
        "",
        [
            PlanStep(
                1,
                "Which browser should I use?",
                ActionType.ASK_USER,
                params={"question": "Choose a browser", "options": ["Edge", "Chrome"]},
            )
        ],
    )

    request = TaskAgent._first_user_choice(plan)

    assert request == UserChoiceRequest("Choose a browser", ("Edge", "Chrome"))
