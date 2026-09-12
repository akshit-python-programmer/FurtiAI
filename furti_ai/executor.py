"""Execution layer for multi-step plans, with dynamic re-anchoring.

Each planned step is executed like this:

1. A fresh (1 s-floored) screen observation grounds the current UI.
2. The step's target is resolved from the *live* screen, in order:
   ``icon:name`` template match -> OCR text substring -> planned bbox cropped
   from the planning-time screenshot and template-matched on the live frame.
3. The action is performed (cursor visibly moves unless teleporting).
4. Optionally the model verifies the step's effect on a new screenshot.
5. Failed steps are re-anchored and retried, then re-planned by the model
   (escalating to the smart tier), bounded by retry/budget guardrails and a
   signature-based loop detector.
6. Every successfully anchored step is compiled into a reusable reflex
   (template + metadata) stored in the memory cache.

The executor checks the stop event between every capture, LLM call and
action, so the kill hotkey aborts safely at the next step boundary.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .config import Settings
from .context import SceneObservation, TaskAborted, VisualContextManager
from .controller import InputController
from .memory import MemoryManager
from .models import ActionType, Skill
from .planner import PlanStep, TaskPlan, TaskPlanner
from .tasklog import TaskJournal
from .vision import VisionReflex

logger = logging.getLogger(__name__)

VERIFY_SYSTEM_PROMPT = (
    "You are the verification component of Furti AI, a desktop automation "
    "agent. A step was just executed. Using the current screen grounding "
    "(OCR text, recognised icons and, if attached, the screenshot), decide "
    "whether the step visibly succeeded. "
    'Answer with JSON only: {"ok": true|false, "reason": "..."}'
)

_OCR_DESCRIPTOR_WORDS = {
    "app",
    "application",
    "bar",
    "button",
    "click",
    "field",
    "icon",
    "link",
    "menu",
    "on",
    "open",
    "tab",
    "taskbar",
    "the",
    "window",
}


def _normalise_ocr_text(value: str) -> str:
    """Normalise OCR and model text before comparing their labels."""
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))


@dataclass
class StepResult:
    """Outcome of one executed plan step."""

    step: int
    description: str
    success: bool
    attempts: int = 1
    notes: list[str] = field(default_factory=list)
    reflex: Optional[str] = None
    action_dispatched: bool = False
    visually_verified: Optional[bool] = None
    superseded: bool = False


@dataclass(frozen=True)
class TargetResolution:
    """Resolved target plus the coordinate space of its center."""

    center: tuple[int, int]
    template_path: Optional[Path]
    anchor_note: str
    capture_coordinates: bool = True


@dataclass
class ExecutionReport:
    """Aggregate outcome of running a whole plan."""

    results: list[StepResult] = field(default_factory=list)
    aborted: bool = False
    reason: str = ""

    @property
    def success(self) -> bool:
        return not self.aborted and bool(self.results) and all(
            r.success or r.superseded for r in self.results
        )


class PlanExecutor:
    """Executes :class:`TaskPlan` step by step with guardrails."""

    def __init__(
        self,
        settings: Settings,
        journal: TaskJournal,
        vision: VisionReflex,
        controller: InputController,
        memory: MemoryManager,
        planner: TaskPlanner,
        context: VisualContextManager,
        stop_event: Any,
        task_slug: str,
    ) -> None:
        self._settings = settings
        self._journal = journal
        self._vision = vision
        self._controller = controller
        self._memory = memory
        self._planner = planner
        self._context = context
        self._stop = stop_event
        self._task_slug = task_slug or "task"

    # ------------------------------------------------------------- public
    def execute(self, instruction: str, plan: TaskPlan) -> ExecutionReport:
        report = ExecutionReport()
        failures_in_a_row = 0
        plan_replans = 0
        position = 0

        # An index-based loop lets an adaptive re-plan replace the unfinished
        # route without replaying steps that already completed.
        while position < len(plan.steps):
            self._check_stop()
            step = plan.steps[position]
            total = len(plan.steps)
            self._journal.step(
                f"[{position + 1}/{total}] {step.description} "
                f"[action={step.action.value}]"
            )
            self._journal.thought(
                "Continuing immediately; no fixed inter-step delay is applied."
            )
            result = self._execute_step(instruction, step, plan)
            if result.success:
                failures_in_a_row = 0
                if result.reflex:
                    self._journal.reflexes_compiled.append(result.reflex)
                self._record_result(report, result)
                position += 1
            else:
                replacement = self._adaptive_replan(
                    instruction,
                    plan,
                    position,
                    step,
                    result,
                    plan_replans,
                )
                if replacement is not None:
                    failure_reason = (
                        result.notes[-1] if result.notes else "step failed"
                    )
                    result.superseded = True
                    result.notes.append(
                        "superseded by an adaptive re-plan of the remaining route"
                    )
                    self._record_result(report, result)
                    plan_replans += 1
                    self._journal.plan_revisions.append(
                        {
                            "after_step": step.index,
                            "reason": failure_reason,
                            "steps": [
                                {
                                    "step": replacement_step.index,
                                    "description": replacement_step.description,
                                    "action": replacement_step.action.value,
                                }
                                for replacement_step in replacement
                            ],
                        }
                    )
                    self._journal.plan(
                        f"Adaptive route revision {plan_replans} applied after "
                        f"step {step.index}; continuing with a new path."
                    )
                    plan.steps = plan.steps[:position] + replacement
                    failures_in_a_row = 0
                    continue

                self._record_result(report, result)
                failures_in_a_row += 1
                position += 1
                if failures_in_a_row >= self._settings.max_consecutive_failures:
                    report.aborted = True
                    report.reason = (
                        f"{failures_in_a_row} consecutive step failures reached "
                        f"max_consecutive_failures={self._settings.max_consecutive_failures}; "
                        "aborting to avoid an endless loop."
                    )
                    self._journal.error(report.reason)
                    break
        return report

    def _record_result(
        self, report: ExecutionReport, result: StepResult
    ) -> None:
        """Keep the console report and markdown report in sync."""
        report.results.append(result)
        self._journal.step_results.append(
            {
                "step": result.step,
                "description": result.description,
                "success": result.success,
                "superseded": result.superseded,
                "action_dispatched": result.action_dispatched,
                "visually_verified": result.visually_verified,
                "notes": result.notes,
            }
        )

    def _adaptive_replan(
        self,
        instruction: str,
        plan: TaskPlan,
        position: int,
        failed_step: PlanStep,
        result: StepResult,
        plan_replans: int,
    ) -> Optional[list[PlanStep]]:
        """Try a bounded alternate route after a step exhausts local retries."""
        if plan_replans >= self._settings.max_plan_replans:
            self._journal.warn(
                f"Adaptive re-plan budget exhausted "
                f"({plan_replans}/{self._settings.max_plan_replans}); "
                "keeping the original route."
            )
            return None
        replanner = getattr(self._planner, "replan_remaining", None)
        if not callable(replanner):
            return None

        scene = self._context.observe(instruction, force_fresh=True)
        self._journal.warn(
            f"Step {failed_step.index} could not complete. "
            "Re-planning the remaining route from the current screen."
        )
        try:
            replacement_plan = replanner(
                instruction,
                plan,
                plan.steps[:position],
                failed_step,
                result.notes[-1] if result.notes else "step failed",
                scene,
                result.attempts,
            )
        except TaskAborted:
            raise
        except Exception as exc:
            self._journal.warn(f"Adaptive route re-plan failed: {exc}")
            return None

        replacement = list(replacement_plan.steps)
        old_remaining = plan.steps[position:]
        if not replacement or not self._route_changed(old_remaining, replacement):
            self._journal.warn(
                "Adaptive re-plan did not produce a different first route "
                "step; refusing to repeat it."
            )
            return None
        for offset, step in enumerate(replacement):
            step.index = position + offset + 1
        return replacement

    @staticmethod
    def _route_changed(
        old_remaining: list[PlanStep], replacement: list[PlanStep]
    ) -> bool:
        """Require the replacement to avoid repeating the failed first step."""
        if not old_remaining or not replacement:
            return bool(replacement)
        return replacement[0].signature() != old_remaining[0].signature()

    # ------------------------------------------------------- single step
    def _execute_step(
        self, instruction: str, original: PlanStep, plan: TaskPlan
    ) -> StepResult:
        step = original
        seen_signatures: set[str] = set()
        attempts = 0
        action_dispatched = False
        visually_verified: Optional[bool] = None

        while True:
            self._check_stop()
            attempts += 1
            self._journal.action(
                f"Step {step.index}: locating a live target for "
                f"{step.action.value} (attempt {attempts})",
                signal="searching",
            )
            observe_started = time.perf_counter()
            scene = self._context.observe(instruction, force_fresh=True)
            self._journal.thought(
                f"Step {step.index}: screen grounding ready in "
                f"{time.perf_counter() - observe_started:.2f}s "
                f"(fresh={scene.fresh}, OCR={len(scene.text_lines)}, "
                f"icons={len(scene.icons)})."
            )

            resolve_started = time.perf_counter()
            target = self._resolve_target(step, scene, plan)
            resolve_duration = time.perf_counter() - resolve_started
            if target is not None:
                frame_center = target.center
                input_center = (
                    self._to_input_point(frame_center, scene)
                    if target.capture_coordinates
                    else frame_center
                )
                self._journal.action(
                    f"Step {step.index}: {step.description} -> "
                    f"{step.action.value} at input={input_center} "
                    f"(frame={frame_center}, {target.anchor_note}; "
                    f"target search {resolve_duration:.2f}s)"
                )
                try:
                    self._perform_action(step, input_center)
                except Exception as exc:
                    # Input backends expose different exception types. Keep
                    # the failure attached to this step so it can be retried
                    # and reported instead of killing the worker silently.
                    note = f"action dispatch failed: {exc}"
                    self._journal.error(f"Step {step.index}: {note}")
                else:
                    action_dispatched = True
                    self._journal.confirm(
                        f"Step {step.index}: {step.action.value} input "
                        "dispatched successfully."
                    )
                    ok, reason = self._verify_step(instruction, step)
                    if self._settings.verify_steps:
                        visually_verified = (
                            None
                            if reason.startswith("visual verification")
                            else ok
                        )
                    if ok:
                        reflex = self._compile_reflex(
                            step, target.template_path, scene, target.anchor_note
                        )
                        notes = [
                            f"anchor: {target.anchor_note}",
                            "action dispatch confirmed",
                        ]
                        if reason:
                            notes.append(reason)
                        return StepResult(
                            step=step.index,
                            description=step.description,
                            success=True,
                            attempts=attempts,
                            notes=notes,
                            reflex=reflex,
                            action_dispatched=True,
                            visually_verified=visually_verified,
                        )
                    note = f"verification failed: {reason}"
            else:
                note = "no target anchor found on the live screen"
                self._journal.action(
                    f"Step {step.index}: target not found after "
                    f"{resolve_duration:.2f}s; preparing recovery.",
                    signal="searching",
                )

            signature = step.signature()
            if signature in seen_signatures:
                note = (
                    f"loop detected: identical step {step.description!r} was "
                    f"already re-planned; aborting this step"
                )
                self._journal.error(note)
                return StepResult(
                    step=step.index,
                    description=step.description,
                    success=False,
                    attempts=attempts,
                    notes=(
                        ["action dispatch confirmed on an earlier attempt"]
                        if action_dispatched
                        else []
                    )
                    + [note],
                    action_dispatched=action_dispatched,
                    visually_verified=visually_verified,
                )

            if attempts > self._settings.max_step_retries:
                note = (
                    f"giving up after {attempts} attempts "
                    f"(max_step_retries={self._settings.max_step_retries})"
                )
                self._journal.error(
                    f"Step {step.index} {note}: {step.description}"
                )
                return StepResult(
                    step=step.index,
                    description=step.description,
                    success=False,
                    attempts=attempts,
                    notes=(
                        ["action dispatch confirmed on an earlier attempt"]
                        if action_dispatched
                        else []
                    )
                    + [note],
                    action_dispatched=action_dispatched,
                    visually_verified=visually_verified,
                )

            seen_signatures.add(signature)
            self._journal.warn(
                f"Step {step.index} failed (attempt {attempts}): {note}. "
                "Re-planning this step."
            )
            step = self._planner.replan_step(
                instruction, step, note, scene, attempts
            )

    # ---------------------------------------------------- target resolution
    def _resolve_target(
        self,
        step: PlanStep,
        scene: SceneObservation,
        plan: TaskPlan,
    ) -> Optional[TargetResolution]:
        """Resolve a live target and preserve its coordinate-space metadata."""
        action = step.action
        if action in (ActionType.KEY_PRESS,):
            # No visual anchor needed: key chords work on the focused window.
            return TargetResolution(
                self._cursor_position(),
                None,
                "focused window (no anchor)",
                capture_coordinates=False,
            )

        target = (step.target or "").strip()

        if target.lower().startswith("icon:"):
            icon_name = target.split(":", 1)[1].strip()
            for icon in scene.icons:
                if icon.name == icon_name or icon_name in icon.name or icon.name in icon_name:
                    template_path = self._settings.templates_dir / f"{icon.name}.png"
                    return TargetResolution(
                        icon.center,
                        template_path,
                        f"icon:{icon.name} conf={icon.confidence:.2f}",
                    )

        if target:
            candidate = self._best_ocr_target(target, scene)
            if candidate is not None:
                line, _score = candidate
                note = f"OCR text {line.text!r}"
                return TargetResolution(
                    line.center,
                    self._crop_ocr_anchor(scene, line),
                    note,
                )

        if step.bbox is not None and plan.frame is not None:
            crop = self._crop_from_plan_frame(plan.frame, step.bbox)
            if crop is not None and scene.frame is not None and self._vision is not None:
                match = self._vision.locate_on(scene.frame, crop)
                if match is not None:
                    (x, y), confidence, (tw, th) = match
                    if confidence >= self._settings.confidence_threshold:
                        center = (x + tw // 2, y + th // 2)
                        template_path = self._save_crop(crop, step.index)
                        note = f"planned bbox re-anchored conf={confidence:.3f}"
                        return TargetResolution(center, template_path, note)

        if action in (ActionType.TYPE, ActionType.SCROLL):
            # Typing/scroll target the focused window: proceed without an anchor.
            return TargetResolution(
                self._cursor_position(),
                None,
                "focused window (no anchor)",
                capture_coordinates=False,
            )

        return None

    @staticmethod
    def _best_ocr_target(
        target: str,
        scene: SceneObservation,
    ) -> Optional[tuple[Any, int]]:
        """Choose a meaningful OCR match without accepting one-letter noise."""
        needle = _normalise_ocr_text(target)
        if not needle:
            return None
        target_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", needle)
            if len(token) >= 3 and token not in _OCR_DESCRIPTOR_WORDS
        }
        candidates: list[tuple[int, float, Any]] = []
        for line in scene.text_lines:
            hay = _normalise_ocr_text(line.text)
            if not hay:
                continue
            score = 0
            if hay == needle:
                score = 100
            elif needle in hay:
                score = 80
            elif len(hay) >= 3 and hay in needle:
                score = 60
            else:
                line_tokens = set(re.findall(r"[a-z0-9]+", hay))
                overlap = target_tokens.intersection(line_tokens)
                if overlap:
                    score = 40 + min(15, 5 * len(overlap))
            if score:
                candidates.append((score, float(line.confidence), line))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return candidates[0][2], candidates[0][0]

    def _to_input_point(
        self,
        point: tuple[int, int],
        scene: SceneObservation,
    ) -> tuple[int, int]:
        """Convert a capture-space target to the mouse coordinate space."""
        if scene.frame is None or self._vision is None:
            return int(point[0]), int(point[1])
        mapper = getattr(self._vision, "to_input_point", None)
        if not callable(mapper):
            return int(point[0]), int(point[1])
        return mapper(point, scene.frame.shape)

    def _cursor_position(self) -> tuple[int, int]:
        import pyautogui

        return tuple(int(v) for v in pyautogui.position())

    def _crop_from_plan_frame(
        self, frame: np.ndarray, bbox: Any
    ) -> Optional[np.ndarray]:
        try:
            x = max(0, int(bbox.x))
            y = max(0, int(bbox.y))
            w = min(int(bbox.width), frame.shape[1] - x)
            h = min(int(bbox.height), frame.shape[0] - y)
            if w <= 0 or h <= 0:
                return None
            return frame[y : y + h, x : x + w]
        except (AttributeError, TypeError, ValueError):
            return None

    def _crop_ocr_anchor(self, scene: SceneObservation, line: Any) -> Optional[Path]:
        if scene.frame is None:
            return None
        bbox = line.bbox
        pad = 6
        x = max(0, bbox.x - pad)
        y = max(0, bbox.y - pad)
        w = min(bbox.width + 2 * pad, scene.frame.shape[1] - x)
        h = min(bbox.height + 2 * pad, scene.frame.shape[0] - y)
        crop = scene.frame[y : y + h, x : x + w]
        return self._save_crop(crop, f"ocr_{abs(hash(line.text)) % 100000}")

    def _save_crop(self, crop: np.ndarray, label: Any) -> Path:
        self._settings.ensure_dirs()
        safe = self._task_slug.replace("/", "_").replace("\\", "_")
        path = self._settings.templates_dir / f"{safe}_{label}.png"
        cv2.imwrite(str(path), crop)
        return path

    # ------------------------------------------------------------ actuation
    def _perform_action(self, step: PlanStep, center: tuple[int, int]) -> None:
        self._check_stop()
        cx, cy = int(center[0]), int(center[1])
        action = step.action

        if action == ActionType.CLICK:
            self._controller.click(cx, cy)
        elif action == ActionType.DOUBLE_CLICK:
            self._controller.double_click(cx, cy)
        elif action == ActionType.RIGHT_CLICK:
            self._controller.right_click(cx, cy)
        elif action == ActionType.TYPE:
            if not step.text:
                raise ValueError("type action has no text payload")
            self._controller.type_text(str(step.text))
        elif action == ActionType.SCROLL:
            self._controller.scroll(int(step.params.get("scroll_clicks", 3)))
        elif action == ActionType.KEY_PRESS:
            self._controller.press_key(
                str(step.params.get("key") or step.text or "enter")
            )
        else:
            raise ValueError(f"Unsupported action type: {action}")

    # ----------------------------------------------------------- verification
    def _verify_step(self, instruction: str, step: PlanStep) -> tuple[bool, str]:
        """Optionally ask the model whether the step visibly succeeded."""
        if not self._settings.verify_steps:
            return True, "visual effect verification disabled"
        self._check_stop()
        # Do not sleep for a fresh frame here: that made every step pay an
        # avoidable one-second delay. If the capture throttle has not opened,
        # report dispatch success and defer visual checking to a later frame.
        scene = self._context.observe(instruction, force_fresh=True)
        self._journal.thought(
            f"Verifying step {step.index} result "
            f"(fresh screen={scene.fresh})..."
        )
        if not scene.fresh:
            reason = "visual verification deferred by screenshot throttle"
            self._journal.confirm(f"Step {step.index}: {reason}.")
            return True, reason
        try:
            self._journal.thought(
                f"AI is thinking about the visible result of step {step.index}."
            )
            self._journal.waiting(
                f"Waiting for AI response (verify step {step.index})..."
            )
            raw = self._planner._fast.chat_text(
                VERIFY_SYSTEM_PROMPT,
                f"Task: {instruction}\nExecuted step: {step.description}\n\n"
                f"{scene.prompt_block()}",
                purpose="verify",
            )
            self._journal.thought(
                f"AI response received (verify step {step.index})."
            )
            self._journal.ai_output(
                raw,
                getattr(
                    self._planner._fast,
                    "_model",
                    type(self._planner._fast).__name__,
                ),
                f"verify_step_{step.index}",
            )
            import json

            payload = json.loads(_strip_fences(raw))
            ok = bool(payload.get("ok", True))
            reason = str(payload.get("reason", ""))
            self._journal.thought(
                f"Verification for step {step.index}: {'OK' if ok else 'FAIL'} "
                f"{'- ' + reason if reason else ''}"
            )
            return ok, reason
        except Exception as exc:
            reason = f"visual verification unavailable: {exc}"
            self._journal.warn(f"Step {step.index}: {reason}")
            return True, reason

    # ------------------------------------------------------ reflex compiling
    def _compile_reflex(
        self,
        step: PlanStep,
        template_path: Optional[Path],
        scene: SceneObservation,
        anchor_note: str,
    ) -> Optional[str]:
        """Turn a successfully executed step into a reusable cached reflex."""
        if template_path is None or not Path(template_path).exists():
            return None
        name = self._memory.normalize_name(step.description)
        if not name:
            return None
        skill = Skill(
            name=name,
            template_path=str(template_path),
            action=step.action,
            metadata={
                "description": step.description,
                "action": step.action.value,
                "text": step.text,
                "target": step.target,
                "anchor": anchor_note,
                "screen_size": (
                    [scene.frame.shape[1], scene.frame.shape[0]]
                    if scene.frame is not None
                    else None
                ),
                "compiled_by": "multi_step_executor",
            },
        )
        self._memory.save_skill(skill)
        self._journal.reflex(
            f"Compiled reusable reflex {name!r} from step {step.index} "
            f"(template: {template_path.name})"
        )
        return name

    # ---------------------------------------------------------------- utils
    def _check_stop(self) -> None:
        if self._stop is not None and self._stop.is_set():
            raise TaskAborted("stop hotkey pressed during execution")


def _strip_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    return cleaned
