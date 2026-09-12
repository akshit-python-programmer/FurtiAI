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
from furti_ai.models import BoundingBox, ActionType
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
        self.typed: list[str] = []
        self.pressed: list[str] = []
        self.scrolls: list[int] = []

    def click(self, x, y, button="left"):
        self.clicks.append((x, y))

    def double_click(self, x, y):
        self.clicks.append((x, y))

    def right_click(self, x, y):
        self.clicks.append((x, y))

    def type_text(self, text):
        self.typed.append(text)

    def press_key(self, key):
        self.pressed.append(key)

    def scroll(self, clicks):
        self.scrolls.append(clicks)


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


def test_planner_defensive_action_fallback(tmp_path):
    settings = make_settings(tmp_path)
    llm = MockLLM(
        response='{"steps": [{"description": "x", "action": "teleport", "target": null}]}'
    )
    planner = TaskPlanner(
        llm, settings, make_journal(tmp_path),
        FakeContext([SceneObservation(frame=np.zeros((8, 8, 3), dtype=np.uint8))]),
        threading.Event(),
    )
    plan = planner.plan("do the thing")
    assert plan.steps[0].action == ActionType.CLICK  # safe default


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


def test_executor_confirms_dispatch_without_forced_verification_delay(tmp_path):
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
    assert any("deferred by screenshot throttle" in note for note in report.results[0].notes)
    assert planner._fast.call_count == 0


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


def test_task_agent_confirms_runs_and_reports(tmp_path, monkeypatch):
    agent, settings, input_ctl = _build_test_agent(tmp_path, monkeypatch, "y")
    assert agent.run_task("click export then type hello") is True
    assert input_ctl.clicks == [(350, 220)]
    assert input_ctl.typed == ["hello"]
    # the <task_name>.md report and a compiled reflex template exist
    reports = list(settings.reports_dir.glob("*.md"))
    assert reports
    assert list(settings.templates_dir.glob("*.png"))
