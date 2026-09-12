"""Multi-step task planning with tiered models.

Unlike the legacy single-action :class:`BrainPlanner`, :class:`TaskPlanner`
asks the model to decompose a user instruction into an ordered sequence of
steps. Planning consumes the cheap ``fast`` model by default and escalates to
the ``smart`` model when:

* the fast model produces unparsable output twice, or
* the plan itself flags ``requires_smart_model: true`` (complex reasoning), or
* a step keeps failing during execution (handled by the executor).

Every thought the model emits (goal, per-step ``thought`` fields, strategy
notes) is printed to the console through the journal, keeping the reasoning
fully transparent.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from .config import Settings
from .context import SceneObservation, TaskAborted
from .models import ActionPlan, ActionType, BoundingBox
from .tasklog import TaskJournal

logger = logging.getLogger(__name__)

PLAN_SYSTEM_PROMPT = (
    "You are the task planner of Furti AI, a desktop automation agent. "
    "Decompose the user instruction into the smallest ordered list of concrete "
    "UI actions. Use the screen grounding (OCR text with coordinates and "
    "recognised icon templates) to anchor each step. You may also use the "
    "screenshot if it was attached. "
    "Return JSON only (no markdown, no commentary) in exactly this shape:\n"
    '{"goal": "<one-line goal>", '
    '"reasoning": "<short strategy note>", '
    '"requires_smart_model": false, '
    '"steps": [{"step": 1, "description": "<what this does>", '
    '"action": "click|double_click|right_click|type|scroll|key_press", '
    '"target": "<icon:name | exact OCR text | element description | null>", '
    '"bbox": {"x":0,"y":0,"width":0,"height":0} | null, '
    '"bbox_coordinate_space": "full_capture" | "attached_image", '
    '"text": "<text to type when action is type>", '
    '"params": {"key": "enter", "scroll_clicks": 3}, '
    '"thought": "<why this step, one line>"}]}\n'
    "Rules: prefer targets anchored to OCR text or icon templates and give "
    "exact pixel coordinates only when the screenshot clearly shows them; "
    "use full_capture coordinates by default and set "
    "bbox_coordinate_space=attached_image only when the bbox is measured "
    "directly from the attached image dimensions; "
    "never invent coordinates you cannot see; keep the step count minimal; "
    "set requires_smart_model=true only when the task genuinely needs complex "
    "multi-condition reasoning."
)

REPLAN_SYSTEM_PROMPT = (
    "You are the re-planning component of Furti AI. A step of a larger task "
    "failed. Given the current screen grounding, the original step, and the "
    "failure reason, produce exactly ONE replacement action. "
    "Return JSON only (no markdown) in this shape:\n"
    '{"description": "...", "action": "click|double_click|right_click|type|scroll|key_press", '
    '"target": "<icon:name | exact OCR text | element description | null>", '
    '"bbox": null, "bbox_coordinate_space": "full_capture", '
    '"text": null, "params": {}, "thought": "..."}'
)

REPLAN_TASK_SYSTEM_PROMPT = (
    "You are the adaptive route-planner of Furti AI, a desktop automation "
    "agent. The current UI no longer matches the original plan. Re-plan only "
    "the remaining work from the current screen grounding, keeping completed "
    "steps out of the result. Choose a genuinely different route, target "
    "anchor, or navigation action from the failed step; do not repeat the "
    "same description/action/target combination. "
    "Return JSON only in the same plan shape as the main planner, with a "
    "minimal ordered steps list. If no safe route exists, return an empty "
    "steps list."
)


class BudgetExceeded(Exception):
    """Raised when the per-task LLM call budget is exhausted."""


@dataclass
class PlanStep:
    """One planned action of a multi-step task."""

    index: int
    description: str
    action: ActionType
    bbox: Optional[BoundingBox] = None
    text: Optional[str] = None
    target: Optional[str] = None
    params: dict[str, Any] = field(default_factory=dict)
    thought: str = ""
    bbox_space: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any], index: int) -> "PlanStep":
        action = ActionType.CLICK
        try:
            action = ActionType(str(data.get("action", "click")).lower())
        except ValueError:
            pass

        bbox: Optional[BoundingBox] = None
        raw_bbox = data.get("bbox")
        if isinstance(raw_bbox, dict):
            try:
                bbox = BoundingBox.from_dict(raw_bbox)
                if bbox.area <= 0:
                    bbox = None
            except (TypeError, ValueError):
                bbox = None

        params = dict(data.get("params") or {})
        bbox_space = str(
            data.get("bbox_coordinate_space")
            or data.get("coordinate_space")
            or params.get("bbox_coordinate_space")
            or ""
        ).strip().lower()
        return cls(
            index=index,
            description=str(data.get("description", "")).strip(),
            action=action,
            bbox=bbox,
            text=data.get("text") or None,
            target=data.get("target") or None,
            params=params,
            thought=str(data.get("thought", "")).strip(),
            bbox_space=bbox_space,
        )

    def signature(self) -> str:
        """Stable fingerprint used for loop detection."""
        digest = hashlib.sha1(
            f"{self.description}|{self.action.value}|{self.target or ''}".encode(
                "utf-8"
            )
        ).hexdigest()[:12]
        return digest


@dataclass
class TaskPlan:
    """A complete ordered plan produced by the reasoning layer."""

    task_name: str
    goal: str
    steps: list[PlanStep]
    reasoning: str = ""
    model_used: str = ""
    requires_smart: bool = False
    # The planning-time screenshot; crops taken from it act as visual
    # templates when the executor re-anchors steps on the live screen.
    frame: Any = field(default=None, repr=False)

    def describe(self) -> str:
        lines = [f"goal: {self.goal or '(none)'}"]
        for step in self.steps:
            anchor = step.target or (
                f"bbox {step.bbox.x},{step.bbox.y} {step.bbox.width}x{step.bbox.height}"
                if step.bbox
                else "current cursor"
            )
            lines.append(
                f"  {step.index}. {step.description} "
                f"[{step.action.value}] -> {anchor}"
            )
        return "\n".join(lines)


class TaskPlanner:
    """Plans whole tasks into step sequences using tiered LLM models."""

    def __init__(
        self,
        fast_llm: Any,
        settings: Settings,
        journal: TaskJournal,
        visual_context: Any,
        stop_event: Any,
        smart_llm: Any = None,
    ) -> None:
        self._fast = fast_llm
        self._smart = smart_llm
        self._settings = settings
        self._journal = journal
        self._context = visual_context
        self._stop = stop_event

    # ------------------------------------------------------------ public API
    def plan(self, instruction: str) -> TaskPlan:
        """Produce the multi-step plan, escalating models as needed."""
        self._journal.thought(f"Decomposing instruction into steps: {instruction!r}")
        scene = self._context.observe(instruction)

        system = PLAN_SYSTEM_PROMPT
        user = (
            f"User instruction: {instruction}\n\n"
            f"{scene.prompt_block()}\n\n"
            "Produce the multi-step JSON plan now."
        )

        attempts = 0
        last_error: Optional[Exception] = None
        while attempts < 4:
            self._check_stop()
            self._assert_budget("plan")
            model = self._pick_model(attempts)
            try:
                self._journal.thought(
                    f"Asking {self._model_label(model)} to plan "
                    f"(screenshot attached: {scene.vision_used})"
                )
                self._journal.waiting(
                    f"Waiting for {self._model_label(model)} response "
                    "(initial plan)..."
                )
                raw = (
                    model.chat_vision(system, user, scene.image_b64, purpose="plan")
                    if scene.vision_used and scene.image_b64
                    else model.chat_text(system, user, purpose="plan")
                )
                self._journal.thought(
                    f"AI response received from {self._model_label(model)} "
                    "(initial plan)."
                )
                self._journal.ai_output(
                    raw,
                    self._model_name(model),
                    "plan",
                )
                payload = self._parse_json(raw)
                plan = self._build_plan(instruction, payload, self._model_name(model))
                self._map_plan_bboxes_to_frame(plan, scene)
                plan.frame = scene.frame
                return self._finalize_plan(plan)
            except BudgetExceeded:
                raise  # do not retry into a budget we already refuse to spend
            except Exception as exc:  # parse errors, network hiccups
                last_error = exc
                self._journal.warn(f"Planning attempt {attempts + 1} failed: {exc}")
                attempts += 1

        raise RuntimeError(f"Planner gave up after {attempts} attempts: {last_error}")

    def replan_step(
        self,
        instruction: str,
        failed: PlanStep,
        failure_reason: str,
        scene: SceneObservation,
        attempt: int,
    ) -> PlanStep:
        """Ask for a single replacement step for one that failed.

        Escalates to the smart model after the fast model has already failed
        ``max_step_retries`` times on this step.
        """
        self._check_stop()
        self._assert_budget("replan")
        escalate = attempt >= self._settings.max_step_retries
        model = self._smart if escalate and self._smart is not None else self._fast
        if escalate:
            self._journal.model(
                f"Escalating step {failed.index} re-planning to the smarter "
                f"model ({self._model_name(model)})."
            )
        self._journal.thought(
            f"AI is thinking about a replacement for step {failed.index}."
        )
        self._journal.waiting(
            f"Waiting for {self._model_name(model)} response "
            f"(step {failed.index} re-plan)..."
        )

        user = (
            f"Whole task: {instruction}\n\n"
            f"Failed step: {failed.description} [{failed.action.value}]\n"
            f"Failure reason: {failure_reason}\n\n"
            f"{scene.prompt_block()}\n\n"
            "Produce the single replacement step JSON now."
        )
        raw = (
            model.chat_vision(REPLAN_SYSTEM_PROMPT, user, scene.image_b64, purpose="replan")
            if scene.vision_used and scene.image_b64
            else model.chat_text(REPLAN_SYSTEM_PROMPT, user, purpose="replan")
        )
        self._journal.thought(
            f"AI response received from {self._model_name(model)} "
            f"(step {failed.index} re-plan)."
        )
        self._journal.ai_output(
            raw,
            self._model_name(model),
            f"replan_step_{failed.index}",
        )
        payload = self._parse_json(raw)
        step = PlanStep.from_dict(payload, failed.index)
        self._map_step_bbox_to_frame(step, scene)
        if not step.description:
            step.description = failed.description
        self._journal.thought(
            f"Re-plan for step {failed.index}: {step.description} [{step.action.value}]"
        )
        return step

    def replan_remaining(
        self,
        instruction: str,
        current_plan: TaskPlan,
        completed_steps: list[PlanStep],
        failed_step: PlanStep,
        failure_reason: str,
        scene: SceneObservation,
        attempt: int,
    ) -> TaskPlan:
        """Build a replacement route for the unfinished part of a task.

        This is deliberately separate from :meth:`replan_step`: a changed
        modal, navigation state, or dialog can invalidate several future
        actions, not just the anchor for the current action.
        """
        self._check_stop()
        self._assert_budget("replan_task")
        model = (
            self._smart
            if self._smart is not None
            and (attempt >= self._settings.max_step_retries or current_plan.requires_smart)
            else self._fast
        )
        self._journal.thought(
            f"AI is reconsidering the remaining route after step "
            f"{failed_step.index} failed."
        )
        self._journal.waiting(
            f"Waiting for {self._model_name(model)} response "
            "(adaptive route re-plan)..."
        )
        completed = "\n".join(
            f"- {step.index}. {step.description} [{step.action.value}]"
            for step in completed_steps
        ) or "(none)"
        remaining = "\n".join(
            f"- {step.index}. {step.description} [{step.action.value}]"
            for step in current_plan.steps[len(completed_steps):]
        ) or "(none)"
        user = (
            f"Whole task: {instruction}\n\n"
            f"Completed steps:\n{completed}\n\n"
            f"Failed step: {failed_step.description} [{failed_step.action.value}]\n"
            f"Failure reason: {failure_reason}\n\n"
            f"Original remaining plan:\n{remaining}\n\n"
            f"{scene.prompt_block()}\n\n"
            "Return only the replacement steps still needed."
        )
        raw = (
            model.chat_vision(
                REPLAN_TASK_SYSTEM_PROMPT,
                user,
                scene.image_b64,
                purpose="replan_task",
            )
            if scene.vision_used and scene.image_b64
            else model.chat_text(
                REPLAN_TASK_SYSTEM_PROMPT,
                user,
                purpose="replan_task",
            )
        )
        self._journal.thought(
            f"AI response received from {self._model_name(model)} "
            "(adaptive route re-plan)."
        )
        self._journal.ai_output(
            raw,
            self._model_name(model),
            "replan_task",
        )
        payload = self._parse_json(raw)
        replacement = self._build_plan(
            instruction, payload, self._model_name(model)
        )
        self._map_plan_bboxes_to_frame(replacement, scene)
        replacement.frame = scene.frame
        start_index = len(completed_steps) + 1
        for offset, step in enumerate(replacement.steps):
            step.index = start_index + offset
        if not replacement.steps:
            raise ValueError("adaptive re-plan returned no remaining steps")
        self._journal.plan(
            f"Adaptive route ready: {len(replacement.steps)} replacement "
            "step(s).\n"
            + replacement.describe()
        )
        for step in replacement.steps:
            if step.thought:
                self._journal.thought(
                    f"Replacement step {step.index} reasoning: {step.thought}"
                )
        return replacement

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _map_plan_bboxes_to_frame(
        plan: TaskPlan, scene: SceneObservation
    ) -> None:
        """Convert model-image bboxes to full captured-frame pixels."""
        if (
            not scene.vision_used
            or not scene.image_b64
            or scene.frame is None
            or not scene.vision_size
        ):
            return
        frame_width, frame_height = scene.frame.shape[1], scene.frame.shape[0]
        image_width, image_height = scene.vision_size
        if image_width <= 0 or image_height <= 0:
            return
        if (image_width, image_height) == (frame_width, frame_height):
            return

        scale_x = frame_width / image_width
        scale_y = frame_height / image_height
        for step in plan.steps:
            if step.bbox is None:
                continue
            bbox = step.bbox
            if step.bbox_space in {
                "full",
                "full_capture",
                "screen",
                "frame",
            }:
                step.bbox_space = "full_capture"
                continue
            if not (
                0 <= bbox.x
                and 0 <= bbox.y
                and bbox.x + bbox.width <= image_width
                and bbox.y + bbox.height <= image_height
            ):
                # Coordinates outside the attached image are already in the
                # full-capture space (the model had OCR dimensions in view).
                step.bbox_space = "full_capture"
                continue
            step.bbox = BoundingBox(
                x=round(bbox.x * scale_x),
                y=round(bbox.y * scale_y),
                width=max(1, round(bbox.width * scale_x)),
                height=max(1, round(bbox.height * scale_y)),
            ).clamp(frame_width, frame_height)
            step.bbox_space = "full_capture"

    @classmethod
    def _map_step_bbox_to_frame(
        cls, step: PlanStep, scene: SceneObservation
    ) -> None:
        """Apply the image-to-frame conversion to one replacement step."""
        cls._map_plan_bboxes_to_frame(
            TaskPlan(task_name="", goal="", steps=[step]),
            scene,
        )

    def _finalize_plan(self, plan: TaskPlan) -> TaskPlan:
        cap = self._settings.max_plan_steps
        if len(plan.steps) > cap:
            self._journal.warn(
                f"Plan had {len(plan.steps)} steps; truncating to {cap} to "
                f"respect max_plan_steps."
            )
            plan.steps = plan.steps[:cap]
        self._journal.plan(f"Plan ready: {len(plan.steps)} step(s).\n{plan.describe()}")
        for step in plan.steps:
            if step.thought:
                self._journal.thought(f"Step {step.index} reasoning: {step.thought}")
        return plan

    def _build_plan(
        self, instruction: str, payload: dict[str, Any], model_name: str
    ) -> TaskPlan:
        raw_steps = payload.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError("plan JSON contains no steps list")
        steps = [
            PlanStep.from_dict(entry, index)
            for index, entry in enumerate(raw_steps, start=1)
            if isinstance(entry, dict)
        ]
        if not steps:
            raise ValueError("plan JSON contains no valid steps")
        return TaskPlan(
            task_name=instruction,
            goal=str(payload.get("goal", "")),
            steps=steps,
            reasoning=str(payload.get("reasoning", "")),
            model_used=model_name,
            requires_smart=bool(payload.get("requires_smart_model", False)),
        )

    def _pick_model(self, attempt: int) -> Any:
        if self._smart is not None and attempt >= 2:
            return self._smart
        return self._fast

    def _model_label(self, model: Any) -> str:
        return self._model_name(model)

    @staticmethod
    def _model_name(model: Any) -> str:
        return getattr(model, "_model", getattr(model, "model", type(model).__name__))

    def _assert_budget(self, purpose: str) -> None:
        used = self._calls_used()
        limit = self._settings.max_llm_calls_per_task
        if used >= limit:
            self._journal.error(
                f"LLM call budget exhausted ({used}/{limit}); refusing further "
                f"calls to avoid an endless loop."
            )
            raise BudgetExceeded(f"{used}/{limit} LLM calls used")

    def _calls_used(self) -> int:
        total = int(getattr(self._fast, "call_count", 0))
        if self._smart is not None:
            total += int(getattr(self._smart, "call_count", 0))
        return total

    def _check_stop(self) -> None:
        if self._stop is not None and self._stop.is_set():
            raise TaskAborted("stop hotkey pressed during planning")

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        payload = json.loads(cleaned)
        if not isinstance(payload, dict):
            raise ValueError("plan JSON is not an object")
        return payload
