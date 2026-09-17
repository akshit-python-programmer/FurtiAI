"""Reflex quality gate, re-alignment, and retirement.

Two behaviours matter here: a reflex is only *created* when it is both
successful and plausibly reusable, and a reflex that stops matching is handed
back to the LLM to be re-aligned (and retired if it has become useless) instead
of being blindly replayed or silently discarded.
"""

import threading
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from furti_ai.config import Settings
from furti_ai.context import SceneObservation
from furti_ai.executor import PlanExecutor
from furti_ai.memory import MemoryManager
from furti_ai.models import ActionType, BoundingBox, Skill
from furti_ai.ocr import TextLine
from furti_ai.planner import PlanStep, TaskPlan, TaskPlanner
from furti_ai.reflex import RealignResult, ReflexRealigner
from furti_ai.tasklog import TaskJournal


# ------------------------------------------------------------------ fixtures
def make_settings(tmp_path: Path, **overrides) -> Settings:
    settings = Settings(workspace=tmp_path, verify_steps=False, **overrides)
    settings.ensure_dirs()
    return settings


def make_journal(tmp_path: Path) -> TaskJournal:
    return TaskJournal("reflex_task", tmp_path / "reports")


class FakeJournal:
    def __init__(self):
        self.events = []

    def warn(self, message):
        self.events.append(("warn", message))

    def thought(self, message):
        self.events.append(("thought", message))

    def action(self, message, signal="trying"):
        self.events.append(("action", message))

    def reflex(self, message):
        self.events.append(("reflex", message))

    def system(self, message):
        self.events.append(("system", message))

    def confirm(self, message):
        self.events.append(("confirm", message))

    def error(self, message):
        self.events.append(("error", message))

    def ai_output(self, output, model, purpose, max_chars=4000):
        self.events.append(("ai_output", purpose))

    def messages(self) -> str:
        return "\n".join(str(payload) for _kind, payload in self.events)


class FakeScreen:
    """A frame with one bright square the realigner is asked to find."""

    def __init__(self, shape=(100, 140), box=(40, 30, 20, 16)):
        self.shape = shape
        self.box = box
        self.captures = 0

    def capture(self):
        self.captures += 1
        frame = np.zeros((self.shape[0], self.shape[1], 3), dtype="uint8")
        x, y, w, h = self.box
        frame[y : y + h, x : x + w] = 255
        return frame


class ScriptedLLM:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []
        self._model = "relocator"

    def chat_vision(self, system, user, image_b64, purpose=""):
        self.calls.append({"system": system, "user": user, "purpose": purpose})
        if self.error is not None:
            raise self.error
        return self.response if isinstance(self.response, str) else str(self.response)

    def chat_text(self, system, user, purpose=""):
        return self.chat_vision(system, user, "", purpose=purpose)


def make_skill(tmp_path: Path, name: str = "click_export", metadata=None) -> Skill:
    template = tmp_path / "templates" / f"{name}.png"
    template.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(template), np.zeros((12, 12, 3), dtype="uint8"))
    skill = Skill(
        name=name,
        template_path=str(template),
        action=ActionType.CLICK,
        # Reflexes are compiled disabled; helpers opt in to stand for a reflex
        # the user has already enabled in the Reflexes tab.
        enabled=True,
        metadata=metadata
        if metadata is not None
        else {
            "description": "Click the Export button",
            "target": "Export",
            "expected_bbox": {"x": 300, "y": 200, "width": 100, "height": 40},
            "anchor": "OCR text 'Export'",
        },
    )
    return skill


# ------------------------------------------------------------ re-alignment
def test_realign_rewrites_the_template_and_metadata(tmp_path):
    settings = make_settings(tmp_path)
    memory = MemoryManager(settings.memory_file)
    skill = make_skill(tmp_path)
    memory.save_skill(skill)
    journal = FakeJournal()
    screen = FakeScreen(box=(50, 40, 20, 16))
    llm = ScriptedLLM(
        '{"found": true, "bbox": {"x": 50, "y": 40, "width": 20, "height": 16}, '
        '"confidence": 0.93, "description": "Export button moved", '
        '"note": "the toolbar moved to the left"}'
    )
    realigner = ReflexRealigner(settings, llm, memory, screen, journal)

    result = realigner.realign(skill, "the stored template did not match")

    assert result.ok is True
    assert result.skill is skill
    assert result.confidence == 0.93
    assert skill.metadata["expected_bbox"] == {"x": 50, "y": 40, "width": 20, "height": 16}
    assert skill.metadata["screen_size"] == [140, 100]
    assert skill.metadata["realign_note"] == "the toolbar moved to the left"
    assert skill.realign_count == 1
    assert skill.failure_count == 0
    # The overwritten template is now the re-located crop, not the stale one.
    stored = cv2.imread(skill.template_path, cv2.IMREAD_COLOR)
    assert stored.shape[:2] == (16, 20)
    assert int(stored.max()) == 255
    # And it was persisted for the next run.
    assert MemoryManager(settings.memory_file).get_skill("click_export") is not None
    assert "re-aligned" in journal.messages().lower()


def test_realign_needs_an_explicit_found_flag(tmp_path):
    settings = make_settings(tmp_path)
    memory = MemoryManager(settings.memory_file)
    skill = make_skill(tmp_path)
    llm = ScriptedLLM('{"bbox": {"x": 1, "y": 1, "width": 10, "height": 10}, "confidence": 0.99}')
    realigner = ReflexRealigner(settings, llm, memory, FakeScreen(), FakeJournal())

    result = realigner.realign(skill)

    assert result.ok is False
    assert "could not find" in result.reason or "no usable bbox" in result.reason


def test_realign_refuses_an_unquantified_guess(tmp_path):
    settings = make_settings(tmp_path)
    skill = make_skill(tmp_path)
    llm = ScriptedLLM('{"found": true, "bbox": {"x": 5, "y": 5, "width": 10, "height": 10}}')
    realigner = ReflexRealigner(settings, llm, MemoryManager(settings.memory_file), FakeScreen(), FakeJournal())

    result = realigner.realign(skill)

    assert result.ok is False
    assert "confidence" in result.reason
    # A rejected guess must not have touched the stored template.
    assert skill.realign_count == 0


def test_realign_refuses_low_confidence(tmp_path):
    settings = make_settings(tmp_path, reflex_min_anchor_confidence=0.9)
    skill = make_skill(tmp_path)
    llm = ScriptedLLM(
        '{"found": true, "bbox": {"x": 5, "y": 5, "width": 10, "height": 10}, "confidence": 0.4}'
    )
    realigner = ReflexRealigner(settings, llm, MemoryManager(settings.memory_file), FakeScreen(), FakeJournal())

    result = realigner.realign(skill)

    assert result.ok is False
    assert "below" in result.reason


@pytest.mark.parametrize(
    "bbox",
    [
        {"x": 0, "y": 0, "width": 2, "height": 2},
        {"x": 0, "y": 0, "width": 4000, "height": 4000},
    ],
)
def test_realign_rejects_unusable_geometry(tmp_path, bbox):
    settings = make_settings(tmp_path)
    skill = make_skill(tmp_path)
    llm = ScriptedLLM(
        '{"found": true, "bbox": '
        + str(bbox).replace("'", '"')
        + ', "confidence": 0.99}'
    )
    realigner = ReflexRealigner(settings, llm, MemoryManager(settings.memory_file), FakeScreen(), FakeJournal())

    result = realigner.realign(skill)

    assert result.ok is False
    assert skill.realign_count == 0


def test_realign_is_disabled_by_setting(tmp_path):
    settings = make_settings(tmp_path, reflex_realign=False)
    realigner = ReflexRealigner(settings, ScriptedLLM("{}"), MemoryManager(settings.memory_file), FakeScreen())

    assert realigner.enabled is False
    assert realigner.realign(make_skill(tmp_path)).ok is False


def test_realign_survives_a_model_error(tmp_path):
    settings = make_settings(tmp_path)
    realigner = ReflexRealigner(
        settings,
        ScriptedLLM(error=RuntimeError("api down")),
        MemoryManager(settings.memory_file),
        FakeScreen(),
        FakeJournal(),
    )

    result = realigner.realign(make_skill(tmp_path))

    assert result.ok is False
    assert "api down" in result.reason


def test_realign_prompt_carries_the_reflex_identity(tmp_path):
    settings = make_settings(tmp_path)
    llm = ScriptedLLM('{"found": false, "note": "gone"}')
    realigner = ReflexRealigner(settings, llm, MemoryManager(settings.memory_file), FakeScreen(), FakeJournal())

    realigner.realign(make_skill(tmp_path), "template match failed")

    prompt = llm.calls[0]["user"]
    assert "Click the Export button" in prompt
    assert "Export" in prompt
    assert "template match failed" in prompt
    assert "140x100" in prompt


# --------------------------------------------------------------- retirement
def test_is_useless_after_repeated_failures(tmp_path):
    settings = make_settings(tmp_path, reflex_retire_failures=3)
    realigner = ReflexRealigner(settings, ScriptedLLM("{}"), MemoryManager(settings.memory_file))
    skill = make_skill(tmp_path)

    assert realigner.is_useless(skill) is False
    skill.failure_count = 3
    assert realigner.is_useless(skill) is True


def test_retire_forgets_the_skill(tmp_path):
    settings = make_settings(tmp_path, reflex_retire_failures=2)
    memory = MemoryManager(settings.memory_file)
    skill = make_skill(tmp_path)
    skill.failure_count = 2
    memory.save_skill(skill)
    journal = FakeJournal()
    realigner = ReflexRealigner(settings, ScriptedLLM("{}"), memory, journal=journal)

    result = realigner.retire(skill)

    assert result.retired is True
    assert memory.get_skill("click_export") is None
    assert "Retired reflex" in journal.messages()


# ------------------------------------------------- orchestrator integration
class StubVision:
    """VisionReflex stub: fails (and optionally succeeds) on demand."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.executed = []

    def execute(self, skill):
        self.executed.append(skill.name)
        return self.outcomes.pop(0) if self.outcomes else False


class StubBrain:
    def __init__(self):
        self.planned = []

    def plan(self, command, name):
        self.planned.append(command)
        return Skill(name=name, template_path="x.png")


def test_orchestrator_realigns_then_replays(tmp_path):
    from furti_ai.orchestrator import AgentOrchestrator

    settings = make_settings(tmp_path)
    memory = MemoryManager(settings.memory_file)
    skill = make_skill(tmp_path, name="click_the_export_button")
    memory.save_skill(skill)
    vision = StubVision([False, True])  # miss, then hit after re-alignment
    brain = StubBrain()
    llm = ScriptedLLM(
        '{"found": true, "bbox": {"x": 50, "y": 40, "width": 20, "height": 16}, '
        '"confidence": 0.95}'
    )
    realigner = ReflexRealigner(settings, llm, memory, FakeScreen(box=(50, 40, 20, 16)), FakeJournal())
    orchestrator = AgentOrchestrator(memory, vision, brain, realigner)

    assert orchestrator.run("click the export button") is True
    assert brain.planned == []  # the reflex was repaired, not re-planned
    assert skill.realign_count == 1
    assert skill.success_count == 1


def test_orchestrator_falls_back_to_the_planner_when_realign_fails(tmp_path):
    from furti_ai.orchestrator import AgentOrchestrator

    settings = make_settings(tmp_path)
    memory = MemoryManager(settings.memory_file)
    memory.save_skill(make_skill(tmp_path, name="click_the_export_button"))
    vision = StubVision([False, True])  # miss, then the newly compiled reflex hits
    brain = StubBrain()
    realigner = ReflexRealigner(
        settings, ScriptedLLM('{"found": false, "note": "not on screen"}'),
        memory, FakeScreen(), FakeJournal(),
    )
    orchestrator = AgentOrchestrator(memory, vision, brain, realigner)

    assert orchestrator.run("click the export button") is True
    assert brain.planned == ["click the export button"]


def test_orchestrator_retires_a_useless_reflex_instead_of_realigning(tmp_path):
    from furti_ai.orchestrator import AgentOrchestrator

    settings = make_settings(tmp_path, reflex_retire_failures=2)
    memory = MemoryManager(settings.memory_file)
    skill = make_skill(tmp_path, name="click_the_export_button")
    skill.failure_count = 1  # one more miss retires it
    memory.save_skill(skill)
    vision = StubVision([False, True])
    brain = StubBrain()
    llm = ScriptedLLM('{"found": true, "bbox": {"x": 1, "y": 1, "width": 9, "height": 9}, "confidence": 0.99}')
    realigner = ReflexRealigner(settings, llm, memory, FakeScreen(), FakeJournal())
    orchestrator = AgentOrchestrator(memory, vision, brain, realigner)

    assert orchestrator.run("click the export button") is True
    # Retired: no re-alignment call, the planner compiled a fresh reflex.
    assert llm.calls == []
    assert brain.planned == ["click the export button"]


def test_orchestrator_skips_a_disabled_reflex(tmp_path):
    from furti_ai.orchestrator import AgentOrchestrator

    settings = make_settings(tmp_path)
    memory = MemoryManager(settings.memory_file)
    skill = make_skill(tmp_path, name="click_the_export_button")
    skill.enabled = False  # compiled but never opted in
    memory.save_skill(skill)
    vision = StubVision([True])  # must not be reached
    brain = StubBrain()
    realigner = ReflexRealigner(
        settings, ScriptedLLM('{"found": false}'), memory, FakeScreen(), FakeJournal()
    )
    orchestrator = AgentOrchestrator(memory, vision, brain, realigner)

    assert orchestrator.run("click the export button") is True
    assert brain.planned == ["click the export button"]
    # The stored reflex was bypassed entirely: no replay, no re-alignment, no
    # counters touched, so a disabled reflex is never penalised.
    assert skill.realign_count == 0
    assert skill.success_count == 0
    assert skill.failure_count == 0


# ------------------------------------------------------------- compile gate
class StubContext:
    def __init__(self, scene):
        self.scene = scene

    def observe(self, instruction, force_fresh=False):
        return self.scene


class RecordingInput:
    def __init__(self):
        self.calls = []

    def click(self, x, y, button="left", clicks=1):
        self.calls.append(("click", x, y))


class StubVisionMapper:
    def to_input_point(self, point, _frame_shape):
        return point


class RecordingPlanner:
    """Planner stub that records how it was asked to recover a failed step."""

    def __init__(self, realign_result=None, replan_result=None, raise_on_realign=False):
        self.realign_result = realign_result
        self.replan_result = replan_result
        self.raise_on_realign = raise_on_realign
        self.realign_calls = 0
        self.replan_calls = 0

    def realign_step(self, instruction, failed, failure_reason, scene, attempt):
        self.realign_calls += 1
        if self.raise_on_realign:
            raise RuntimeError("realign unavailable")
        return self.realign_result or failed

    def replan_step(self, instruction, failed, failure_reason, scene, attempt):
        self.replan_calls += 1
        return self.replan_result or failed


def make_gate_executor(tmp_path, *, scene_lines=(), settings=None, memory=None):
    settings = settings or make_settings(tmp_path)
    frame = np.full((480, 640, 3), 200, dtype="uint8")
    scene = SceneObservation(frame=frame, text_lines=list(scene_lines), scene_text="")
    context = StubContext(scene)
    journal = make_journal(tmp_path)
    planner = TaskPlanner(SimpleNamespace(), settings, journal, context, threading.Event())
    executor = PlanExecutor.__new__(PlanExecutor)
    executor._settings = settings
    executor._journal = journal
    executor._vision = StubVisionMapper()
    executor._controller = RecordingInput()
    executor._memory = memory or MemoryManager(settings.memory_file)
    executor._planner = planner
    executor._context = context
    executor._stop = threading.Event()
    executor._task_slug = "t"
    executor._instruction = ""
    return executor, scene, frame


def gate(executor, step, frame, note, template_name="candidate.png"):
    template = executor._settings.templates_dir / template_name
    cv2.imwrite(str(template), frame[40:60, 40:80])
    return executor._reflex_skip_reason(step, template, SceneObservation(frame=frame), note)


def test_gate_allows_a_verified_ocr_anchor(tmp_path):
    executor, _scene, frame = make_gate_executor(tmp_path)
    step = PlanStep(1, "Click the Export button", ActionType.CLICK, target="Export")

    assert gate(executor, step, frame, "OCR text 'Export'") is None


def test_gate_allows_a_high_confidence_icon_anchor(tmp_path):
    executor, _scene, frame = make_gate_executor(tmp_path)
    step = PlanStep(1, "Click the Export button", ActionType.CLICK, target="icon:export")

    assert gate(executor, step, frame, "icon:export conf=0.94") is None


def test_gate_rejects_a_guessed_anchor(tmp_path):
    executor, _scene, frame = make_gate_executor(tmp_path)
    step = PlanStep(1, "Click the Export button", ActionType.CLICK)

    reason = gate(executor, step, frame, "explicit bbox centre")
    assert reason is not None
    assert "verified visual anchor" in reason


def test_gate_rejects_a_low_confidence_anchor(tmp_path):
    executor, _scene, frame = make_gate_executor(tmp_path)
    step = PlanStep(1, "Click the Export button", ActionType.CLICK)

    reason = gate(executor, step, frame, "planned bbox re-anchored conf=0.61")
    assert reason is not None
    assert "below" in reason


def test_gate_rejects_a_missing_template(tmp_path):
    executor, _scene, _frame = make_gate_executor(tmp_path)
    step = PlanStep(1, "Click the Export button", ActionType.CLICK)

    reason = executor._reflex_skip_reason(step, None, None, "OCR text 'Export'")
    assert reason is not None
    assert "no visual template" in reason


def test_gate_rejects_a_tool_action(tmp_path):
    executor, _scene, frame = make_gate_executor(tmp_path)
    step = PlanStep(1, "Write the note file", ActionType.WRITE_FILE)

    reason = gate(executor, step, frame, "OCR text 'Export'")
    assert reason is not None
    assert "cannot be replayed" in reason


def test_gate_rejects_a_generic_description(tmp_path):
    executor, _scene, frame = make_gate_executor(tmp_path)
    step = PlanStep(1, "click", ActionType.CLICK, target="Export")

    reason = gate(executor, step, frame, "OCR text 'Export'")
    assert reason is not None
    assert "too generic" in reason


def test_gate_rejects_a_one_off_typed_payload(tmp_path):
    executor, _scene, frame = make_gate_executor(tmp_path)
    step = PlanStep(
        1, "Type the email body", ActionType.TYPE, text="x" * 400, target="Body"
    )

    reason = gate(executor, step, frame, "OCR text 'Body'")
    assert reason is not None
    assert "one-off" in reason


def test_gate_rejects_a_tiny_template(tmp_path):
    executor, _scene, frame = make_gate_executor(tmp_path)
    step = PlanStep(1, "Click the Export button", ActionType.CLICK, target="Export")
    tiny = executor._settings.templates_dir / "tiny.png"
    cv2.imwrite(str(tiny), np.zeros((3, 3, 3), dtype="uint8"))

    reason = executor._reflex_skip_reason(
        step, tiny, SceneObservation(frame=frame), "OCR text 'Export'"
    )
    assert reason is not None
    assert "too small" in reason


def test_gate_rejects_a_whole_screen_template(tmp_path):
    executor, _scene, frame = make_gate_executor(tmp_path)
    step = PlanStep(1, "Click the Export button", ActionType.CLICK, target="Export")
    huge = executor._settings.templates_dir / "huge.png"
    cv2.imwrite(str(huge), np.zeros((470, 630, 3), dtype="uint8"))

    reason = executor._reflex_skip_reason(
        step, huge, SceneObservation(frame=frame), "OCR text 'Export'"
    )
    assert reason is not None
    assert "covers" in reason


def test_gate_can_be_disabled(tmp_path):
    settings = make_settings(tmp_path, reflex_enabled=False)
    executor, _scene, frame = make_gate_executor(tmp_path, settings=settings)
    step = PlanStep(1, "Click the Export button", ActionType.CLICK, target="Export")

    reason = gate(executor, step, frame, "OCR text 'Export'")
    assert reason is not None
    assert "disabled" in reason


def test_successful_step_with_a_good_anchor_is_still_compiled(tmp_path):
    """The gate must not become a blanket refusal."""
    line = TextLine("Export", BoundingBox(300, 200, 100, 40), 0.99)
    executor, _scene, _frame = make_gate_executor(tmp_path, scene_lines=[line])
    input_ctl = executor._controller
    plan = TaskPlan(
        task_name="t",
        goal="",
        steps=[PlanStep(1, "Click the Export button", ActionType.CLICK, target="Export")],
    )

    report = executor.execute("click export", plan)

    assert report.success is True
    assert input_ctl.calls == [("click", 350, 220)]
    assert report.results[0].reflex == "click_the_export_button"
    assert executor._memory.has_skill("click_the_export_button")


# ---------------------------------------------------- realign on step failure
def test_failed_step_is_realigned_before_the_route_changes(tmp_path):
    line = TextLine("Export", BoundingBox(300, 200, 100, 40), 0.99)
    executor, _scene, _frame = make_gate_executor(tmp_path, scene_lines=[line])
    realigned = PlanStep(2, "Click the Export button", ActionType.CLICK, target="Export")
    planner = RecordingPlanner(realign_result=realigned)
    executor._planner = planner

    # First attempt cannot anchor (empty scene), then the realigned step can.
    empty = SceneObservation(frame=_frame, text_lines=[], scene_text="")
    hits = [empty, executor._context.scene]
    executor._context = SimpleNamespace(observe=lambda *a, **k: hits.pop(0) if hits else executor._context.scene)

    plan = TaskPlan(
        task_name="t",
        goal="",
        steps=[PlanStep(1, "Click the Export button", ActionType.CLICK, target="Missing")],
    )
    report = executor.execute("click export", plan)

    assert planner.realign_calls == 1
    assert planner.replan_calls == 0
    assert report.success is True


def test_second_failure_re_plans_instead_of_realigning_again(tmp_path):
    executor, _scene, frame = make_gate_executor(tmp_path)
    planner = RecordingPlanner()
    executor._planner = planner
    empty = SceneObservation(frame=frame, text_lines=[], scene_text="")
    executor._context = SimpleNamespace(observe=lambda *a, **k: empty)

    plan = TaskPlan(
        task_name="t",
        goal="",
        steps=[PlanStep(1, "Click the Export button", ActionType.CLICK, target="Missing")],
    )
    executor.execute("click export", plan)

    # Attempt 2 onwards is a route change, not another re-alignment.
    assert planner.realign_calls == 1
    assert planner.replan_calls >= 1


def test_realign_exception_falls_back_to_replanning(tmp_path):
    executor, _scene, frame = make_gate_executor(tmp_path)
    planner = RecordingPlanner(raise_on_realign=True)
    executor._planner = planner
    empty = SceneObservation(frame=frame, text_lines=[], scene_text="")
    executor._context = SimpleNamespace(observe=lambda *a, **k: empty)

    plan = TaskPlan(
        task_name="t",
        goal="",
        steps=[PlanStep(1, "Click the Export button", ActionType.CLICK, target="Missing")],
    )
    executor.execute("click export", plan)

    assert planner.replan_calls >= 1


def test_stale_reflex_is_counted_and_then_retired(tmp_path):
    settings = make_settings(tmp_path, reflex_retire_failures=2)
    memory = MemoryManager(settings.memory_file)
    stale = make_skill(tmp_path, name="click_the_export_button")
    stale.failure_count = 1
    memory.save_skill(stale)
    executor, _scene, _frame = make_gate_executor(
        tmp_path, settings=settings, memory=memory
    )
    step = PlanStep(1, "Click the Export button", ActionType.CLICK)

    executor._retire_stale_reflex(step)

    assert memory.get_skill("click_the_export_button") is None
    assert any(event.kind == "WARN" for event in executor._journal._events)


def test_realign_result_note_is_readable():
    ok = RealignResult(ok=True, skill=Skill(name="click_export", template_path="x"), confidence=0.9)
    assert "re-aligned" in ok.note()
    assert "failed" in RealignResult(ok=False, reason="nope").note()
    assert "retired" in RealignResult(retired=True, reason="kept missing").note()
