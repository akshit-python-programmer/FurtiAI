"""Execution layer for multi-step plans, with dynamic re-anchoring.

Each planned step is executed like this:

1. A fresh (1 s-floored) screen observation grounds the current UI.
2. The step's target is resolved from the *live* screen, in order:
   ``icon:name`` template match -> OCR text match (whole-word label or
   majority word overlap, so fragments such as ``none`` never match
   ``nonexistent``) -> planned bbox cropped
   from the planning-time screenshot and template-matched on the live frame.
3. The action is performed (cursor visibly moves unless teleporting).
4. The model can review the completed step and readiness of the next step in
   one request, choosing text or visual evidence as appropriate.
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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .config import Settings
from .context import SceneObservation, TaskAborted, VisualContextManager
from .controller import InputController
from .jsoncontract import extract_json_object
from .keyboard import ChordError, parse_chord
from .memory import MemoryManager
from .models import REFLEX_ACTIONS, ActionType, BoundingBox, Skill
from .ocr import AUTO_CROP_SUFFIX
from .planner import PlanStep, TaskPlan, TaskPlanner
from .tasklog import TaskJournal
from .tools import DirectToolRunner, ToolResult, is_direct_tool
from .vision import VisionReflex
from .windows import (
    find_window,
    focus_window,
    get_foreground_window_title,
    is_furti_window,
    virtual_screen_rect,
    window_at,
    window_rect,
)

logger = logging.getLogger(__name__)

#: Two focus clicks closer than this are treated as the same field, so the
#: explicit "click the field" step and the type step's self-healing click do
#: not press the same spot twice.
FOCUS_CLICK_TOLERANCE_PX = 12

#: OCR labels that dismiss a blocking dialog, in preference order: a close
#: affordance is always safer than an accept button, because accepting cookies
#: or a license is not what the task asked for. Within a group the shortest
#: label wins ("Close" is a button; a sentence merely containing the word is
#: not), then the topmost one.
DISMISS_LABEL_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"close", "x", "✕", "×", "dismiss", "no thanks", "not now", "later", "skip"}),
    frozenset({"ok", "okay", "got it", "accept", "allow", "agree", "i agree", "continue", "yes"}),
)


class StepBlockedByVerifier(RuntimeError):
    """Raised when the independent verifier refuses to let a step run.

    The step is *not* dispatched; the executor's normal failure path then
    re-plans with the verifier's reason (and any safer alternative it proposed)
    as the failure note.
    """

STEP_REVIEW_SYSTEM_PROMPT = (
    "You are the post-action review component of Furti AI, a desktop "
    "automation agent. One UI step was just dispatched. Review the current "
    "step and the readiness of the next planned step in one response. Use "
    "text grounding when OCR/icon coordinates are sufficient, use the "
    "screenshot when layout or visual state matters, and report which "
    "evidence you relied on. Return JSON only in this shape (one bare object: "
    "no prose, no markdown fences, no trailing commas):\n"
    '{"step_ok": true|false, "step_reason": "...", '
    '"evidence_mode": "text|visual|both", "visual_required": true|false, '
    '"next_step": {"ready": true|false, "reason": "...", '
    '"guidance": "..."} | null}\n'
    "Focus verification: when the next step types text, presses keyboard keys "
    "or scrolls, check that the control that must receive that input is focused "
    "(a visible caret, a highlighted field border, or the dialog that owns the "
    "input). Report next_step.ready=false with the reason when the field is not "
    "focused yet, when a popup or dialog is covering it, or when the screen did "
    "not change at all -- then say which of those it is in the reason.\n"
    "Do not invent a replacement action. If visual evidence is unavailable, "
    "say so in the reason and avoid failing a step solely because a screenshot "
    "was not attached."
)

# Kept as an alias for integrations that imported the old prompt constant.
VERIFY_SYSTEM_PROMPT = STEP_REVIEW_SYSTEM_PROMPT

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


def _as_float(value: Any) -> Optional[float]:
    """Best-effort numeric read for model-authored action parameters."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_flag(value: Any) -> Optional[bool]:
    """Read a tri-state boolean parameter; ``None`` means "not stated".

    Models emit booleans as ``true``/``"true"``/``"yes"``/``1`` and just as
    often as ``"false"``, which is truthy in Python. Anything unreadable is
    reported as unstated so the caller's default decides.
    """
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    return None


def _is_jsonable(value: Any) -> bool:
    """True when a planner parameter survives a JSON round trip."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_jsonable(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_jsonable(v) for k, v in value.items())
    return False


def _bbox_from_params(value: Any) -> Optional[BoundingBox]:
    """Rebuild a bounding box from a planner payload, ignoring junk."""
    if isinstance(value, BoundingBox):
        return value
    if not isinstance(value, dict):
        return None
    try:
        return BoundingBox.from_dict(value)
    except (KeyError, TypeError, ValueError):
        return None


def _contains_whole_words(text: str, fragment: str) -> bool:
    """True when ``fragment`` occurs in ``text`` as complete words only.

    Word boundaries keep short OCR lines from matching inside longer words:
    ``"none"`` must not match the description ``"nonexistent widget"``.
    """
    if not fragment:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(fragment)}(?![a-z0-9])"
    return re.search(pattern, text) is not None


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
    next_step_ready: Optional[bool] = None
    next_step_note: str = ""
    superseded: bool = False


@dataclass(frozen=True)
class StepReview:
    """Combined post-action assessment for the current and next step."""

    step_ok: bool
    step_reason: str = ""
    evidence_mode: str = "text"
    visual_required: bool = False
    next_step_ready: Optional[bool] = None
    next_step_reason: str = ""
    next_step_guidance: str = ""
    attempted: bool = False


@dataclass(frozen=True)
class TargetResolution:
    """Resolved target plus the coordinate space of its center."""

    center: tuple[int, int]
    template_path: Optional[Path]
    anchor_note: str
    capture_coordinates: bool = True
    #: Drag steps also carry the point the payload is dropped at.
    drop_point: Optional[tuple[int, int]] = None
    drop_note: str = ""
    #: Coordinate space of ``drop_point``. ``None`` means "same as
    #: ``center``" -- a grab point taken from the live cursor is already in
    #: mouse space while its drop offset is too, so the two can differ.
    drop_capture_coordinates: Optional[bool] = None

    @property
    def drop_uses_capture_space(self) -> bool:
        """Whether ``drop_point`` needs the capture-to-mouse conversion."""
        if self.drop_capture_coordinates is None:
            return self.capture_coordinates
        return self.drop_capture_coordinates


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
        verifier: Any = None,
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
        # High-level tools (launch_app, run_command, clipboard, window state,
        # ...) run through this runner and never touch the screen.
        self._tools = DirectToolRunner(settings, journal, stop_event)
        #: Independent second-opinion reviewer (see ``verifier.py``). ``None``
        #: or disabled means the step runs unverified, as before.
        self._verifier = verifier
        #: Set by :meth:`execute` so the verifier can be given the task text.
        self._instruction = ""
        #: Monotonic timestamp of the last dispatched UI action, used to pace
        #: the next interaction so it cannot race the application's render.
        self._last_dispatch_at: Optional[float] = None
        #: The point and step index of the last dispatched click, so a type
        #: step can tell whether the step before it already focused the field.
        self._last_click_point: Optional[tuple[int, int]] = None
        self._last_click_index: Optional[int] = None

    # ------------------------------------------------------------- public
    def execute(self, instruction: str, plan: TaskPlan) -> ExecutionReport:
        report = ExecutionReport()
        failures_in_a_row = 0
        plan_replans = 0
        position = 0
        self._instruction = instruction

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
                "No fixed inter-step sleep is applied; actions are paced by the "
                "render delay (FURTI_RENDER_DELAY) and the post-step review."
            )
            next_step = (
                plan.steps[position + 1]
                if position + 1 < len(plan.steps)
                else None
            )
            result = self._execute_step(
                instruction,
                step,
                plan,
                next_step=next_step,
            )
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

    def close(self) -> None:
        """Release resources owned by the visual grounding context."""
        close_context = getattr(self._context, "close", None)
        if callable(close_context):
            close_context()

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
                "next_step_ready": result.next_step_ready,
                "next_step_note": result.next_step_note,
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
        self,
        instruction: str,
        original: PlanStep,
        plan: TaskPlan,
        next_step: Optional[PlanStep] = None,
    ) -> StepResult:
        step = original
        seen_signatures: set[str] = set()
        attempts = 0
        action_dispatched = False
        visually_verified: Optional[bool] = None
        next_step_ready: Optional[bool] = None
        next_step_note = ""
        #: Obstructions cleared for this step ("clearing is not a retry").
        dismissals = 0
        dismissal_budget = max(
            0, int(getattr(self._settings, "max_obstruction_dismissals", 0) or 0)
        )
        #: Failure reason of the current attempt, set by whichever branch ran.
        note = ""

        while True:
            self._check_stop()
            if is_direct_tool(step.action):
                # No screen observation, no OCR pass, no verification round
                # trip: the tool already knows whether it worked.
                return self._execute_tool_step(step)
            attempts += 1
            #: Set below when this attempt resolved a live target.
            input_center: Optional[tuple[int, int]] = None
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
            if target is not None and target.capture_coordinates:
                # Contract guard: a resolved point that is not on any monitor is
                # a broken control payload (a mis-scaled or hallucinated pixel),
                # and dispatching it would throw the cursor off-screen so every
                # later click lands somewhere unintended. Refuse it and let the
                # re-align/re-plan path look again.
                mapped = self._to_input_point(target.center, scene)
                offscreen = not self._is_point_on_screen(mapped)
                if target.drop_point is not None:
                    # The drop is dispatched in the same coordinate space rule as
                    # the grab point, so it is validated with the same mapping.
                    drop = (
                        self._to_input_point(target.drop_point, scene)
                        if target.drop_uses_capture_space
                        else target.drop_point
                    )
                    offscreen = offscreen or not self._is_point_on_screen(drop)
                if offscreen:
                    note = (
                        f"resolved point {mapped} is outside the screen area; "
                        "refusing to move the cursor there"
                    )
                    self._journal.error(f"Step {step.index}: {note}")
                    target = None
            if target is not None:
                frame_center = target.center
                input_center = (
                    self._to_input_point(frame_center, scene)
                    if target.capture_coordinates
                    else frame_center
                )
                input_drop: Optional[tuple[int, int]] = None
                if target.drop_point is not None:
                    input_drop = (
                        self._to_input_point(target.drop_point, scene)
                        if target.drop_uses_capture_space
                        else target.drop_point
                    )
                drop_detail = (
                    f", drop input={input_drop} ({target.drop_note})"
                    if input_drop is not None
                    else ""
                )
                self._journal.action(
                    f"Step {step.index}: {step.description} -> "
                    f"{step.action.value} at input={input_center} "
                    f"(frame={frame_center}, {target.anchor_note}; "
                    f"target search {resolve_duration:.2f}s){drop_detail}"
                )
                if target.capture_coordinates:
                    # Rule: never act on a point underneath a popup/overlay.
                    # Ask which window owns this pixel first; if something else
                    # does, dismiss it and re-resolve instead of clicking
                    # through it (which is how a modal gets mistaken for the
                    # page it is covering).
                    spent = self._clear_obstruction(
                        step,
                        input_center,
                        scene,
                        dismissal_budget - dismissals,
                    )
                    if spent:
                        dismissals += spent
                        # Clearing an obstruction is not a failed attempt: the
                        # step never ran, so it does not consume the retry
                        # budget, only the dismissal budget.
                        attempts = max(0, attempts - 1)
                        continue
                self._await_render_delay(step)
                try:
                    self._perform_action(
                        step,
                        input_center,
                        input_drop,
                        # Typing needs an insertion point, not just a window:
                        # a step with a real screen anchor means the model
                        # named the field to type into, so click it first to
                        # move the caret there. Without that the text went to
                        # whatever happened to be focused, which is how
                        # "type at the search box" typed into the previous
                        # window instead.
                        focus_click=self._type_focus_click(step, target),
                    )
                except StepBlockedByVerifier as exc:
                    # The independent verifier refused the step. Nothing was
                    # dispatched, and the reason (plus any safer alternative)
                    # becomes the failure note the planner works from.
                    note = str(exc)
                    self._journal.warn(f"Step {step.index}: {note}")
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
                    review = self._parallel_review(
                        instruction,
                        step,
                        next_step,
                    )
                    next_step_ready = review.next_step_ready
                    next_step_note = (
                        review.next_step_reason or review.next_step_guidance
                    )
                    if review.attempted:
                        visually_verified = review.step_ok
                    if review.step_ok:
                        drag_delta = (
                            None
                            if input_drop is None
                            else (
                                input_drop[0] - input_center[0],
                                input_drop[1] - input_center[1],
                            )
                        )
                        reflex = self._compile_reflex(
                            step,
                            target.template_path,
                            scene,
                            target.anchor_note,
                            drag_delta=drag_delta,
                        )
                        notes = [
                            f"anchor: {target.anchor_note}",
                            "action dispatch confirmed",
                        ]
                        if review.step_reason:
                            notes.append(review.step_reason)
                        if review.evidence_mode:
                            notes.append(
                                f"post-step evidence: {review.evidence_mode}"
                            )
                        if review.visual_required and not review.attempted:
                            notes.append(
                                "visual evidence was requested but no fresh "
                                "post-action frame was available"
                            )
                        if next_step is not None and review.next_step_ready is not None:
                            notes.append(
                                "next step "
                                f"{'ready' if review.next_step_ready else 'not ready'}"
                                + (
                                    f": {review.next_step_reason}"
                                    if review.next_step_reason
                                    else ""
                                )
                            )
                        return StepResult(
                            step=step.index,
                            description=step.description,
                            success=True,
                            attempts=attempts,
                            notes=notes,
                            reflex=reflex,
                            action_dispatched=True,
                            visually_verified=visually_verified,
                            next_step_ready=next_step_ready,
                            next_step_note=next_step_note,
                        )
                    note = f"verification failed: {review.step_reason}"
            else:
                note = note or "no target anchor found on the live screen"
                self._journal.action(
                    f"Step {step.index}: target not found after "
                    f"{resolve_duration:.2f}s; preparing recovery.",
                    signal="searching",
                )

            signature = step.signature()
            if signature in seen_signatures:
                note = (
                    f"{note}; loop detected: identical step "
                    f"{step.description!r} was already re-planned; aborting this step"
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
                    next_step_ready=next_step_ready,
                    next_step_note=next_step_note,
                )

            if attempts > self._settings.max_step_retries:
                # Keep the reason the attempt actually failed: "giving up" alone
                # hides whether the anchor was missing, off-screen or refused.
                note = (
                    f"{note}; giving up after {attempts} attempts "
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
                    next_step_ready=next_step_ready,
                    next_step_note=next_step_note,
                )

            self._journal.warn(
                f"Step {step.index} failed (attempt {attempts}): {note}. "
                "Re-aligning this step first."
            )
            # Rule: a step that produced no screen/state change must not simply
            # be repeated. Before re-aligning (which repeats the same action on
            # a new anchor), check whether a modal or an unfocused window is why
            # nothing happened; clearing that is cheaper and more likely to
            # help than re-planning. The check needs a resolved point, so it is
            # skipped when this attempt found no target at all -- "not there" is
            # evidence about the anchor, not about an overlay.
            if input_center is not None and dismissals < dismissal_budget:
                spend = self._clear_obstruction(
                    step,
                    input_center,
                    scene,
                    dismissal_budget - dismissals,
                )
                if spend:
                    dismissals += spend
                    attempts = max(0, attempts - 1)
                    continue
            elif input_center is None and dismissals < dismissal_budget:
                # Nothing could be anchored at all: before re-aligning the same
                # action, rule out an overlay swallowing the target.
                spend = self._clear_unresolved_obstruction(
                    step,
                    scene,
                    dismissal_budget - dismissals,
                )
                if spend:
                    dismissals += spend
                    attempts = max(0, attempts - 1)
                    continue
            realigned = self._try_realign(
                instruction, step, note, scene, attempts
            )
            if realigned is not None:
                # The route is unchanged and the stale reflex for the old
                # anchor was retired; retry against the corrected anchor. The
                # attempt budget still bounds this loop, which is why the
                # failed signature is deliberately not registered as "seen".
                step = realigned
                continue
            seen_signatures.add(signature)
            step = self._planner.replan_step(
                instruction, step, note, scene, attempts
            )

    # --------------------------------------------------------- re-alignment
    def _try_realign(
        self,
        instruction: str,
        step: PlanStep,
        failure_reason: str,
        scene: SceneObservation,
        attempts: int,
    ) -> Optional[PlanStep]:
        """Ask the LLM to re-anchor the *same* step before changing the route.

        A step usually fails because its anchor went stale (scrolled list, moved
        dialog, changed UI), not because the intent was wrong. Re-aligning keeps
        the route, and a subsequent success overwrites the outdated reflex with
        a fresh template. Re-planning the whole action is the fallback, not the
        first response.
        """
        if attempts != 1 or not getattr(self._settings, "reflex_realign", True):
            return None
        if is_direct_tool(step.action):
            return None  # tools never needed a screen anchor to begin with
        realigner = getattr(self._planner, "realign_step", None)
        if not callable(realigner):
            return None
        self._journal.action(
            f"Step {step.index}: redirecting to the LLM to re-align this step "
            "on the live screen.",
            signal="searching",
        )
        try:
            realigned = realigner(instruction, step, failure_reason, scene, attempts)
        except TaskAborted:
            raise
        except Exception as exc:  # noqa: BLE001 - re-planning is the fallback
            self._journal.warn(
                f"Step {step.index}: re-alignment failed ({exc}); "
                "falling back to a route re-plan."
            )
            return None
        if realigned is None:
            return None
        self._retire_stale_reflex(step)
        return realigned

    def _retire_stale_reflex(self, step: PlanStep) -> None:
        """Count a reflex miss and drop the reflex once it is clearly useless.

        Keeping a reflex that never matches costs a screen capture and a
        matching pass on every run, so a stale entry is retired rather than
        retried forever.
        """
        memory = self._memory
        get_skill = getattr(memory, "get_skill", None)
        normalize = getattr(memory, "normalize_name", None)
        if not callable(get_skill) or not callable(normalize):
            return
        name = normalize(step.description)
        skill = get_skill(name)
        if skill is None:
            return
        record_failure = getattr(skill, "record_failure", None)
        if callable(record_failure):
            record_failure()
        limit = int(getattr(self._settings, "reflex_retire_failures", 3) or 0)
        if limit and int(getattr(skill, "failure_count", 0)) >= limit:
            forget = getattr(memory, "forget", None)
            if callable(forget):
                forget(name)
            self._journal.warn(
                f"Retired reflex {name!r}: it failed "
                f"{skill.failure_count} time(s), so it is no longer worth "
                "replaying."
            )
            return
        save = getattr(memory, "save_skill", None)
        if callable(save):
            save(skill)
        self._journal.thought(
            f"Reflex {name!r} missed ({skill.failure_count}/{limit}); "
            "recording the failure and re-aligning it."
        )

    # ---------------------------------------------------- cross-verification
    def _verify_or_raise(self, step: PlanStep) -> None:
        """Block a critical step unless the independent verifier approves it.

        Runs on the *other* provider with its own API key, so a single model
        cannot dispatch an irreversible action on its own. When no second
        provider is configured the step proceeds unverified (fail-open), and
        the journal says so.
        """
        verifier = getattr(self, "_verifier", None)
        if verifier is None:
            return
        enabled = getattr(verifier, "enabled", False)
        if not enabled:
            return
        review_step = getattr(verifier, "review_step", None)
        if not callable(review_step):
            return
        try:
            review = review_step(
                getattr(self, "_instruction", ""),
                step,
                execution_context=self._recent_context(),
            )
        except TaskAborted:
            raise
        except Exception as exc:  # noqa: BLE001 - verification never blocks
            self._journal.warn(
                f"Step {step.index}: cross-verification errored ({exc}); "
                "continuing."
            )
            return
        note = getattr(review, "note", lambda: "")()
        if getattr(review, "attempted", False):
            self._journal.thought(f"Step {step.index}: {note}")
        if not getattr(review, "approved", True) and getattr(review, "attempted", False):
            note_with_alternative = getattr(
                self._settings, "cross_verify_apply_alternative", True
            )
            failure_note = getattr(review, "failure_note", None)
            if callable(failure_note):
                failure = failure_note(include_alternative=bool(note_with_alternative))
            else:
                failure = note
            raise StepBlockedByVerifier(failure)

    def _recent_context(self) -> str:
        """Short summary of what already happened, for the verifier's prompt."""
        results = getattr(self._journal, "step_results", None) or []
        recent = results[-3:]
        if not recent:
            return ""
        return "; ".join(
            f"step {entry.get('step')} {entry.get('description', '')} "
            f"({'ok' if entry.get('success') else 'failed'})"
            for entry in recent
        )

    # ------------------------------------------------------------ direct tools
    @property
    def _tool_runner(self) -> DirectToolRunner:
        """Lazily build the tool runner for executors created via ``__new__``."""
        runner = getattr(self, "_tools", None)
        if runner is None:
            runner = DirectToolRunner(
                self._settings,
                getattr(self, "_journal", None),
                getattr(self, "_stop", None),
            )
            self._tools = runner
        return runner

    def _execute_tool_step(self, step: PlanStep) -> StepResult:
        """Run a direct tool step and report its own outcome as verification.

        These steps ask the OS to do the work (start an app, run a command,
        write a file), so capturing the screen, grounding it with OCR and
        asking the model to review the result would add seconds and tokens to
        an action that already returns a definitive success/failure.
        """
        self._check_stop()
        self._journal.action(
            f"Step {step.index}: running direct tool "
            f"[tool={step.action.value}]",
            signal="trying",
        )
        if not getattr(self._settings, "direct_tools", True):
            note = (
                f"direct tool {step.action.value!r} is disabled "
                "(FURTI_DIRECT_TOOLS=false)"
            )
            self._journal.error(f"Step {step.index}: {note}")
            return StepResult(
                step=step.index,
                description=step.description,
                success=False,
                attempts=1,
                notes=[note],
            )

        started = time.perf_counter()
        try:
            self._verify_or_raise(step)
        except StepBlockedByVerifier as exc:
            note = str(exc)
            self._journal.warn(f"Step {step.index}: {note}")
            return StepResult(
                step=step.index,
                description=step.description,
                success=False,
                attempts=1,
                notes=[f"tool: {step.action.value}", note],
                action_dispatched=False,
            )
        result: ToolResult = self._tool_runner.run(step)
        elapsed = time.perf_counter() - started
        if result.ok:
            # A tool (launching an app, opening a URL) can change the screen too,
            # so the next interaction waits out the render delay as well.
            self._last_dispatch_at = time.monotonic()
        notes = [
            f"tool: {result.tool}",
            f"tool result: {'ok' if result.ok else 'failed'}"
            + (f" ({result.detail})" if result.detail else ""),
            f"tool latency: {elapsed:.2f}s",
        ]
        if result.output:
            notes.append("tool output:")
            notes.extend(result.output.splitlines())
        message = f"Step {step.index} [{result.tool}] {result.detail}".strip()
        if result.output:
            message += f"\n{result.output}"
        if result.ok:
            self._journal.confirm(message)
        else:
            self._journal.error(message)

        return StepResult(
            step=step.index,
            description=step.description,
            success=result.ok,
            attempts=1,
            notes=notes,
            # Nothing to cache: a tool call is already the fast path.
            reflex=None,
            action_dispatched=result.ok,
            visually_verified=None,
            next_step_ready=None,
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
        if is_direct_tool(action):
            # Tools act on the OS, not on a pixel: there is nothing to find on
            # the screen and nothing to convert between coordinate spaces.
            return TargetResolution(
                self._cursor_position(),
                None,
                f"direct tool {action.value} (no screen target)",
                capture_coordinates=False,
            )
        if action in (ActionType.KEY_PRESS,):
            # No visual anchor needed: key chords work on the focused window.
            return TargetResolution(
                self._cursor_position(),
                None,
                "focused window (no anchor)",
                capture_coordinates=False,
            )

        point = self._step_point(step)
        # Planner schemas advertise ``"x": 0, "y": 0`` as the "no explicit
        # pixel" placeholder and models copy it verbatim. Acting on it parks
        # the cursor in a pyautogui abort corner, which freezes every later
        # input call, so a (0, 0) point is only honoured when the step asks
        # for it explicitly.
        explicit_corner = step.params.get("explicit_coordinate")
        wants_corner = explicit_corner is True or str(
            explicit_corner
        ).strip().lower() in {"1", "true", "yes", "on"}
        if point == (0, 0) and not wants_corner:
            point = None

        if action is ActionType.DRAG:
            return self._resolve_drag_target(step, scene, plan, point)

        anchor = self._anchor_for_target(
            step.target or "", step.bbox, scene, plan, step.index
        )
        if anchor is not None:
            center, template_path, note = anchor
            return TargetResolution(center, template_path, note)

        if point is not None:
            # The model named the pixel directly. Reporting "no target anchor
            # found" here (and re-planning) is what made explicit
            # "click at x,y" steps silently never fire.
            return TargetResolution(
                point, None, f"explicit pixel {point[0]},{point[1]}"
            )

        if action in (ActionType.TYPE, ActionType.SCROLL):
            # Typing/scroll target the focused window: proceed without an anchor.
            return TargetResolution(
                self._cursor_position(),
                None,
                "focused window (no anchor)",
                capture_coordinates=False,
            )

        return None

    def _anchor_for_target(
        self,
        target: str,
        bbox: Optional[Any],
        scene: SceneObservation,
        plan: TaskPlan,
        label: Any,
    ) -> Optional[tuple[tuple[int, int], Optional[Path], str]]:
        """Find a live anchor as ``(center, template_path, note)``.

        Prefers a named icon the model was shown, then the step text found on
        the live frame, then the crop the planner drew on the plan frame
        re-anchored on the live frame.
        """
        target = str(target or "").strip()

        if target.lower().startswith("icon:"):
            icon_name = target.split(":", 1)[1].strip()
            for icon in scene.icons:
                if (
                    icon.name == icon_name
                    or icon_name in icon.name
                    or icon.name in icon_name
                ):
                    template_path = self._settings.templates_dir / f"{icon.name}.png"
                    return (
                        icon.center,
                        template_path,
                        f"icon:{icon.name} conf={icon.confidence:.2f}",
                    )

        if target:
            candidate = self._best_ocr_target(target, scene)
            if candidate is not None:
                line, _score = candidate
                return (
                    line.center,
                    self._crop_ocr_anchor(scene, line),
                    f"OCR text {line.text!r}",
                )

        if bbox is not None and plan.frame is not None:
            crop = self._crop_from_plan_frame(plan.frame, bbox)
            if crop is not None and scene.frame is not None and self._vision is not None:
                match = self._vision.locate_on(scene.frame, crop)
                if match is not None:
                    (x, y), confidence, (tw, th) = match
                    if confidence >= self._settings.confidence_threshold:
                        return (
                            (x + tw // 2, y + th // 2),
                            self._save_crop(crop, label),
                            f"planned bbox re-anchored conf={confidence:.3f}",
                        )

        # Last resort: the planner gave us an explicit pixel region. Icons with
        # no OCR text and no matching template (e.g. a taskbar icon the model
        # has never seen cropped) would otherwise be unclickable. Clicking the
        # centre of the requested box keeps the run moving; the post-step
        # review re-plans when the click lands on the wrong thing.
        if bbox is not None:
            try:
                center = bbox.center
            except (AttributeError, TypeError):
                # Some call sites may still pass a raw dict/list.
                try:
                    center = BoundingBox.from_dict(bbox).center
                except (TypeError, ValueError):
                    center = None
            if center is not None and tuple(center) != (0, 0):
                return center, None, "explicit bbox centre"

        return None

    @staticmethod
    def _step_point(step: PlanStep) -> Optional[tuple[int, int]]:
        """Explicit pixel a step acts on, in full-capture coordinates.

        ``PlanStep.from_dict`` normalises every spelling the model uses
        (top-level ``x``/``y``, a ``point`` object, ``params.x``/``params.y``)
        onto ``params``, so one lookup covers them all.
        """
        params = step.params or {}
        x, y = params.get("x"), params.get("y")
        if x is None or y is None:
            return None
        try:
            return int(round(float(x))), int(round(float(y)))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _drag_origin(params: dict[str, Any]) -> Optional[tuple[int, int]]:
        """Explicit ``from_x``/``from_y`` grab point, in full-capture pixels."""
        from_x = _as_float(params.get("from_x"))
        from_y = _as_float(params.get("from_y"))
        if from_x is None or from_y is None:
            return None
        return int(round(from_x)), int(round(from_y))

    def _resolve_drag_target(
        self,
        step: PlanStep,
        scene: SceneObservation,
        plan: TaskPlan,
        point: Optional[tuple[int, int]],
    ) -> Optional[TargetResolution]:
        """Resolve a drag's grab point, tolerating a missing visual anchor.

        Dragging something the cursor already sits on (a slider handle, an
        existing selection, an item picked with an earlier click) has no element
        to anchor on. An explicit pixel or the live cursor position is used
        instead of failing the step with "no target anchor found".
        """
        params = step.params or {}
        anchor = self._anchor_for_target(
            step.target or "", step.bbox, scene, plan, step.index
        )
        if anchor is not None:
            center, template_path, note = anchor
            resolved = self._resolve_drag_endpoint(step, center, scene, plan)
            if resolved is None:
                # A drag needs a real drop point: releasing at an arbitrary
                # coordinate would drop the payload somewhere unintended.
                return None
            drop_point, drop_note, drop_capture = resolved
            return TargetResolution(
                center,
                template_path,
                note,
                drop_point=drop_point,
                drop_note=drop_note,
                drop_capture_coordinates=drop_capture,
            )

        origin = point if point is not None else self._drag_origin(params)
        capture_space = True
        if origin is None:
            # Live cursor coordinates are already in mouse space, so they must
            # not be rescaled like capture-space pixels.
            origin = self._cursor_position()
            capture_space = False
        resolved = self._resolve_drag_endpoint(step, origin, scene, plan)
        if resolved is None:
            return None
        drop_point, drop_note, drop_capture = resolved
        return TargetResolution(
            origin,
            None,
            f"explicit pixel {origin[0]},{origin[1]}"
            if capture_space
            else "current cursor position",
            capture_coordinates=capture_space,
            drop_point=drop_point,
            drop_note=drop_note,
            drop_capture_coordinates=drop_capture,
        )

    def _resolve_drag_endpoint(
        self,
        step: PlanStep,
        start: tuple[int, int],
        scene: SceneObservation,
        plan: TaskPlan,
    ) -> Optional[tuple[tuple[int, int], str, Optional[bool]]]:
        """Resolve where a drag should be released, or ``None`` when unsure.

        A named drop target is re-resolved on the live frame (so the payload is
        never released at a stale coordinate); explicit ``to_x``/``to_y`` and
        ``dx``/``dy`` offsets are escape hatches for targets with no text or
        icon to anchor on. Returns the drop point, a note for the journal, and
        whether it is in capture space (``None`` = same space as the grab
        point).
        """
        params = step.params or {}

        to_target = str(params.get("to_target") or "").strip()
        to_bbox = _bbox_from_params(params.get("to_bbox"))
        if to_target or to_bbox is not None:
            anchor = self._anchor_for_target(
                to_target, to_bbox, scene, plan, f"drop_{step.index}"
            )
            if anchor is None:
                return None
            (center, _template_path, note) = anchor
            return (
                center,
                f"drop on {to_target or 'planned bbox'} ({note})",
                True,
            )

        to_x = _as_float(params.get("to_x"))
        to_y = _as_float(params.get("to_y"))
        if to_x is not None and to_y is not None:
            point = (int(round(to_x)), int(round(to_y)))
            return (point, f"drop at ({point[0]}, {point[1]})", True)

        dx = _as_float(params.get("dx"))
        dy = _as_float(params.get("dy"))
        if dx is not None and dy is not None:
            point = (int(round(start[0] + dx)), int(round(start[1] + dy)))
            # An offset inherits the grab point's coordinate space.
            return (point, f"drop {dx:+g},{dy:+g} from grab point", None)

        return None

    @staticmethod
    def _best_ocr_target(
        target: str,
        scene: SceneObservation,
    ) -> Optional[tuple[Any, int]]:
        """Choose a meaningful OCR match without accepting one-letter noise.

        Substring hits must fall on whole words and a token-overlap hit must
        cover most of the description, so a weak partial match returns ``None``
        (the caller then re-anchors on the plan frame) instead of clicking an
        unrelated label.
        """
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
            elif _contains_whole_words(hay, needle):
                score = 80
            elif _contains_whole_words(needle, hay):
                score = 60
            else:
                line_tokens = set(re.findall(r"[a-z0-9]+", hay))
                overlap = target_tokens.intersection(line_tokens)
                if overlap and 2 * len(overlap) >= len(target_tokens):
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
        # The suffix keeps this one-off anchor crop out of the icon library
        # that IconMatcher advertises to the model as clickable named icons.
        path = self._settings.templates_dir / f"{safe}_{label}{AUTO_CROP_SUFFIX}.png"
        cv2.imwrite(str(path), crop)
        return path

    # ------------------------------------------------------------ actuation
    def _key_from_target(self, target: Optional[str]) -> Optional[str]:
        """Recover a keystroke/chord from a prose target string.

        Some replan paths put the key into ``target`` (e.g. ``"Windows key
        (Start menu)"``) instead of ``params.key``. Extract the chord when it
        is clearly specified so a key_press step does not silently fall back
        to Enter.
        """
        if not target:
            return None
        text = str(target).strip()
        if not text:
            return None

        # A bare chord is the easy case: "ctrl+l", "enter", "page down"...
        try:
            parse_chord(text)
            return text
        except ChordError:
            pass

        lowered = text.lower()

        # Modifier chords embedded in prose: "press Ctrl + L to focus ..."
        chord_match = re.search(
            r"\b((?:ctrl|control|shift|alt|win|windows|super|cmd|command)"
            r"\s*\+\s*){1,3}[a-z0-9]\w*\b",
            lowered,
        )
        if chord_match:
            return (
                chord_match.group(0)
                .replace(" ", "")
                .replace("control", "ctrl")
                .replace("command", "cmd")
                .replace("windows", "win")
                .replace("super", "win")
            )

        phrase_to_key = {
            "windows key": "win",
            "win key": "win",
            "start menu": "win",
            "start key": "win",
            "escape": "esc",
            "esc key": "esc",
            "return": "enter",
            "spacebar": "space",
            "space bar": "space",
            "tab key": "tab",
            "backspace": "backspace",
        }
        for phrase, key in phrase_to_key.items():
            if phrase in lowered:
                return key

        # Function keys: "press f5 to refresh"
        fn_match = re.search(r"\bf([1-9]|1[0-9]|2[0-4])\b", lowered)
        if fn_match:
            return fn_match.group(0)

        # A trailing named key in a sentence: "press Enter", "hit Page Down".
        single = re.search(
            r"\b(enter|esc|escape|tab|backspace|delete|home|end|insert|space"
            r"|pageup|pagedown|page down|page up|up|down|left|right"
            r"|capslock|printscreen|prtsc|alt|ctrl|shift|win)\b",
            lowered,
        )
        if single:
            token = single.group(0)
            aliases = {
                "escape": "esc",
                "page down": "pagedown",
                "page up": "pageup",
                "prtsc": "printscreen",
            }
            return aliases.get(token, token)

        return None

    # -------------------------------------------------------- window focus
    def _resolve_window_title(self, step: PlanStep) -> Optional[str]:
        """Best-effort title of the window that should receive this input."""
        params = step.params or {}
        explicit = (
            params.get("window")
            or params.get("focus_window")
            or params.get("app")
            or step.window
        )
        if explicit:
            title = str(explicit).strip()
            return title or None

        # No explicit window was supplied. Fall back to the step target or a
        # quoted / "<name> window" phrase in the description, but only accept
        # candidates that match a real open window so we never focus a
        # non-existent application.
        candidates: list[str] = []
        if step.target:
            candidates.append(str(step.target).strip())
        desc = str(step.description or "")
        quoted = re.findall(r'["\u201c\u201d]([^"\u201c\u201d]{1,60})["\u201c\u201d]', desc)
        candidates.extend(part.strip() for part in quoted if part.strip())
        windowed = re.findall(
            r"\b([A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*)*)"
            r"\s+(?:window|app|application)\b",
            desc,
        )
        candidates.extend(part.strip() for part in windowed if part.strip())

        for candidate in candidates:
            if candidate and find_window(candidate) is not None:
                return candidate
        return None

    def _type_focus_click(self, step: PlanStep, target: TargetResolution) -> bool:
        """Whether a ``type`` step must click its target before typing.

        A resolved on-screen anchor (``capture_coordinates``) means the model
        pointed at a real field, so the caret has to be put there first.
        A step that fell back to "the focused window" has no field to click and
        is left to the window-focus path. The model can force either behaviour
        with ``params.focus_click``.
        """
        if step.action is not ActionType.TYPE:
            return False
        explicit = _as_flag((step.params or {}).get("focus_click"))
        if explicit is not None:
            return explicit
        if not target.capture_coordinates:
            return False
        # A plan may carry the focus click as its own step (the focus-before-
        # typing rule). Pressing the same field twice would select text or
        # re-trigger a control, so an immediately preceding click on this very
        # point counts as "already focused".
        if self._focus_click_already_happened(step, target.center):
            self._journal.thought(
                f"Step {step.index}: the previous step already clicked "
                f"({target.center[0]}, {target.center[1]}); typing without a "
                "second click."
            )
            return False
        return True

    def _focus_click_already_happened(
        self, step: PlanStep, point: tuple[int, int]
    ) -> bool:
        """True when the step right before this one already clicked ``point``.

        Index adjacency is the guard: the click must come from the immediately
        preceding plan step, so a click from an earlier, unrelated step (or one
        the user performed) cannot suppress the focus click.
        """
        last_point = getattr(self, "_last_click_point", None)
        last_index = getattr(self, "_last_click_index", None)
        if last_point is None or last_index is None:
            return False
        if last_index != int(step.index) - 1:
            return False
        return (
            abs(int(last_point[0]) - int(point[0])) <= FOCUS_CLICK_TOLERANCE_PX
            and abs(int(last_point[1]) - int(point[1])) <= FOCUS_CLICK_TOLERANCE_PX
        )

    # ------------------------------------------------- action timing guard
    def _is_point_on_screen(self, point: tuple[int, int]) -> bool:
        """Whether a resolved point is inside the virtual desktop.

        A model-supplied pixel that is off-screen (a mis-scaled coordinate, a
        hallucinated value, a fallback default) must never be dispatched: on
        Windows the cursor would be clamped or thrown to a corner, and every
        later click would land somewhere unintended. The virtual rectangle is
        used instead of the primary monitor size so a second display at
        negative coordinates still works.
        """
        rect = virtual_screen_rect()
        if rect is None:
            return True  # unknown (non-Windows / API failure): do not block
        left, top, width, height = rect
        x, y = int(point[0]), int(point[1])
        return left <= x < left + width and top <= y < top + height

    def _await_render_delay(self, step: PlanStep) -> float:
        """Pace the next interaction so it cannot race the previous render.

        High-level actions are not chained blindly: a click that navigates,
        opens a popup or re-renders a list needs a moment before the next
        target exists, and a click delivered mid-render is silently dropped.
        A ``move`` only repositions the cursor and changes nothing on screen,
        so it never waits.
        """
        delay = float(getattr(self._settings, "render_delay", 0.0) or 0.0)
        if delay <= 0 or step.action is ActionType.MOVE:
            return 0.0
        last = getattr(self, "_last_dispatch_at", None)
        if last is None:
            return 0.0
        remaining = delay - (time.monotonic() - last)
        if remaining <= 0:
            return 0.0
        self._journal.thought(
            f"Step {step.index}: waiting {remaining:.2f}s for the previous "
            "action to finish rendering before interacting."
        )
        # Waiting on the stop event keeps the delay interruptible.
        waiter = getattr(self._stop, "wait", None)
        if callable(waiter):
            waiter(remaining)
        else:  # pragma: no cover - defensive for stub stop events
            time.sleep(remaining)
        return remaining

    # ----------------------------------------------- obstruction protocol
    def _obstruction_reason(
        self, step: PlanStep, point: tuple[int, int], scene: SceneObservation
    ) -> str:
        """Describe the window between the agent and ``point``, if any.

        Returns "" when the point belongs to the window the step means to act
        on (or to nothing at all). A non-empty reason means something has to
        happen *before* this step: dismiss the dialog, or bring the intended
        window back to the front.
        """
        if not bool(getattr(self._settings, "dismiss_obstructions", True)):
            return ""
        cover = window_at(point[0], point[1])
        if cover is None:
            # None off Windows, over the desktop, or on a point no window owns:
            # in all three cases there is no overlay to reason about.
            return ""
        cover_hwnd, cover_title = cover
        intended = self._resolve_window_title(step)
        intended_hwnd = find_window(intended) if intended else None

        if intended_hwnd is not None and cover_hwnd == intended_hwnd:
            return ""
        if is_furti_window(cover_title):
            return (
                f"Furti's own window ({cover_title!r}) covers the target at "
                f"({point[0]}, {point[1]})"
            )
        if intended_hwnd is None:
            # No intended window is named, so the only obstruction we can prove
            # is our own overlay (handled above). Guessing here would press ESC
            # in unrelated applications.
            return ""
        small = self._is_dialog_sized(cover_hwnd, scene)
        kind = "dialog/popup" if small else "window"
        return (
            f"another {kind} ({cover_title!r}) covers the target at "
            f"({point[0]}, {point[1]}); expected {intended!r}"
        )

    @staticmethod
    def _is_dialog_sized(hwnd: int, scene: SceneObservation) -> bool:
        """True when a covering window is popup-sized, not a full app frame.

        A small window on top of the target is a modal/cookie banner and safe to
        dismiss with ESC. A full-size window is a different application, where
        pressing ESC would do something unrelated -- that case is handled by
        refocusing the intended window instead.
        """
        rect = window_rect(hwnd)
        frame = getattr(scene, "frame", None)
        if rect is None or frame is None or getattr(frame, "size", 0) == 0:
            return True
        screen_h, screen_w = frame.shape[0], frame.shape[1]
        if screen_w <= 0 or screen_h <= 0:
            return True
        left, top, right, bottom = rect
        area = max(0, right - left) * max(0, bottom - top)
        return area <= 0.6 * float(screen_w * screen_h)

    def _clear_obstruction(
        self,
        step: PlanStep,
        point: tuple[int, int],
        scene: SceneObservation,
        budget: int,
    ) -> int:
        """Dismiss whatever blocks ``point``; returns how many actions it took.

        ``0`` means nothing was obstructing the step (the common case) and the
        caller should carry on. A non-zero result means the caller must observe
        again and re-resolve the target, because the screen just changed.
        """
        if budget <= 0:
            return 0
        reason = self._obstruction_reason(step, point, scene)
        if not reason:
            return 0
        self._journal.warn(
            f"Step {step.index}: {reason}. Clearing the obstruction before "
            "touching the target."
        )
        intended = self._resolve_window_title(step)
        if "dialog/popup" in reason:
            if self._dismiss_with_label(scene, step):
                return 1
            self._journal.action(
                f"Step {step.index}: pressing ESC to dismiss the popup.",
                signal="acting",
            )
            self._dispatch(
                step, lambda: self._controller.press_key("esc")
            )
            return 1

        # Our own always-on-top readout, or a full-size window in front: the fix
        # is to put the intended window back in the foreground, never to press
        # ESC into an unrelated application.
        if intended and self._focus_obstruction_target(intended, step):
            return 1
        self._journal.warn(
            f"Step {step.index}: could not clear {reason}; hide or untick the "
            "always-on-top status window, then retry."
        )
        return 0

    def _focus_obstruction_target(self, title: str, step: PlanStep) -> bool:
        """Bring the step's intended window back to the front."""
        self._journal.action(
            f"Step {step.index}: refocusing {title!r} over the obstruction.",
            signal="acting",
        )
        return bool(focus_window(title))

    def _clear_unresolved_obstruction(
        self, step: PlanStep, scene: SceneObservation, budget: int
    ) -> int:
        """Dismiss a modal when the step's target could not be anchored.

        An overlay hides whatever the step was anchored to, so "anchor not
        found" can be the symptom and the modal the cause. Evidence is required
        before anything is pressed: the screen must offer a *close-style*
        affordance that is not the step's own target, or Furti's own overlay
        must be holding the foreground. Only the close group is used here --
        ``OK``/``Accept``/``Allow`` belong to the task, not to the recovery
        path, and a recovered page should not be consented to on Furti's
        initiative.
        """
        if budget <= 0:
            return 0
        if not bool(getattr(self._settings, "dismiss_obstructions", True)):
            return 0

        label = self._best_dismiss_label(scene, DISMISS_LABEL_GROUPS[0])
        if label is not None:
            token = _normalise_ocr_text(label.text)
            if token and token == _normalise_ocr_text(step.target or ""):
                # The step itself wants to click this "Close": dismissing it
                # here would consume the very step being executed.
                label = None

        if label is None:
            # Nothing to click. The only obstruction provable without a close
            # affordance is Furti's own always-on-top readout.
            if not is_furti_window(get_foreground_window_title()):
                return 0
            intended = self._resolve_window_title(step)
            if not intended:
                return 0
            self._journal.warn(
                f"Step {step.index}: Furti's own window is in the foreground "
                f"and {step.target or step.description!r} could not be found; "
                f"refocusing {intended!r}."
            )
            return 1 if self._focus_obstruction_target(intended, step) else 0

        center = label.center
        self._journal.warn(
            f"Step {step.index}: {step.target or step.description!r} could not "
            f"be found and the screen offers a {label.text!r} affordance; "
            "dismissing the overlay before re-planning."
        )
        self._dispatch(step, lambda: self._controller.click(center[0], center[1]))
        return 1

    def _dismiss_with_label(self, scene: SceneObservation, step: PlanStep) -> bool:
        """Click a close/accept affordance of the blocking dialog, if visible.

        An OCR label from the dismiss groups is preferred over ESC because ESC
        is page-wide (it also stops page loads and exits fullscreen), while
        clicking "Close" is exactly what a person would do.
        """
        for group in DISMISS_LABEL_GROUPS:
            match = self._best_dismiss_label(scene, group)
            if match is None:
                continue
            center = match.center
            self._journal.action(
                f"Step {step.index}: dismissing the obstruction via "
                f"{match.text!r} at {center}.",
                signal="acting",
            )
            self._dispatch(
                step,
                lambda: self._controller.click(center[0], center[1]),
            )
            return True
        return False

    @staticmethod
    def _best_dismiss_label(scene: SceneObservation, group: frozenset[str]):
        """Best OCR line matching a dismiss label: shortest text, then topmost."""
        candidates = []
        for line in getattr(scene, "text_lines", []) or []:
            token = _normalise_ocr_text(getattr(line, "text", ""))
            if not token:
                continue
            if token in group:
                rank = 0
            elif token.strip(" .!,") in group:
                rank = 1
            else:
                continue
            candidates.append((rank, len(token), line))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1], item[2].center[1], item[2].center[0]))
        return candidates[0][2]

    def _dispatch(self, step: PlanStep, action: Any) -> None:
        """Run one input call and record that the UI may now be changing.

        Used by the obstruction protocol (ESC / dismiss clicks); step actions
        themselves record the same state inside :meth:`_perform_action`.
        """
        action()
        self._last_dispatch_at = time.monotonic()

    def _focus_intended_window(self, step: PlanStep) -> None:
        """Force the step's destination window into the foreground."""
        if not self._settings.focus_intended_window:
            return
        title = self._resolve_window_title(step)
        if not title:
            return
        self._journal.thought(
            f"Step {step.index}: focusing window {title!r} before input."
        )
        if focus_window(title):
            self._journal.thought(
                f"Step {step.index}: window {title!r} is now in the foreground."
            )
        else:
            self._journal.warn(
                f"Step {step.index}: could not focus window {title!r}; "
                "input may land in the wrong application."
            )
        # Give the window manager a moment to finish the foreground switch so
        # the immediately-following keyboard/scroll event lands in the window.
        time.sleep(max(0.05, float(self._settings.input_pause)))

    def _perform_action(
        self,
        step: PlanStep,
        center: tuple[int, int],
        drop_point: Optional[tuple[int, int]] = None,
        *,
        focus_click: bool = False,
    ) -> Optional[ToolResult]:
        self._check_stop()
        cx, cy = int(center[0]), int(center[1])
        action = step.action
        params = step.params or {}

        # Critical steps need the independent verifier's approval first; a
        # rejection surfaces as StepBlockedByVerifier, which the step loop turns
        # into a failure note that feeds the re-align/re-plan path.
        self._verify_or_raise(step)

        if is_direct_tool(action):
            # Direct tools bypass the input device entirely. A failure is
            # raised so the generic step loop can retry/re-plan with the
            # tool's own reason attached.
            result = self._tool_runner.run(step)
            if not result.ok:
                raise RuntimeError(result.detail or f"{action.value} failed")
            return result

        if action in (ActionType.KEY_PRESS, ActionType.TYPE, ActionType.SCROLL):
            self._focus_intended_window(step)

        if action == ActionType.MOVE:
            # Reposition only. A separate click step (or params.clicks on the
            # following click) performs the button press.
            self._controller.move_to(cx, cy)
        elif action == ActionType.CLICK:
            self._controller.click(
                cx,
                cy,
                button=str(params.get("button") or "left"),
                clicks=max(1, int(_as_float(params.get("clicks")) or 1)),
            )
        elif action == ActionType.DOUBLE_CLICK:
            self._controller.double_click(cx, cy)
        elif action == ActionType.RIGHT_CLICK:
            self._controller.right_click(cx, cy)
        elif action == ActionType.DRAG:
            if drop_point is None:
                raise ValueError("drag action has no drop point")
            hold_keys = params.get("hold_keys")
            if isinstance(hold_keys, str):
                hold_keys = [hold_keys]
            self._controller.drag(
                cx,
                cy,
                int(drop_point[0]),
                int(drop_point[1]),
                button=str(params.get("button") or "left"),
                duration=_as_float(params.get("duration")),
                hold_keys=list(hold_keys) if hold_keys else None,
            )
        elif action == ActionType.TYPE:
            if not step.text:
                raise ValueError("type action has no text payload")
            if focus_click:
                # Clicking the field both focuses it and drops the caret where
                # the text should appear; typing without this is the classic
                # "keystrokes went to the wrong window" failure.
                self._journal.thought(
                    f"Step {step.index}: clicking ({cx}, {cy}) to focus the "
                    "field before typing."
                )
                self._controller.click(cx, cy)
            self._controller.type_text(str(step.text))
        elif action == ActionType.SCROLL:
            self._controller.scroll(int(params.get("scroll_clicks", 3)))
        elif action == ActionType.KEY_PRESS:
            key = (
                str(params.get("key") or "").strip()
                or str(step.text or "").strip()
                or self._key_from_target(step.target)
                or "enter"
            )
            self._controller.press_key(
                key,
                presses=max(1, int(_as_float(params.get("presses")) or 1)),
            )
        else:
            raise ValueError(f"Unsupported action type: {action}")

        # Record what just happened so the next step can pace itself (render
        # delay) and so a following type step knows whether the caret was just
        # placed by the step before it.
        self._last_dispatch_at = time.monotonic()
        if action in (
            ActionType.CLICK,
            ActionType.DOUBLE_CLICK,
            ActionType.RIGHT_CLICK,
        ):
            self._last_click_point = (cx, cy)
            self._last_click_index = int(step.index)
        return None

    #: Model calls that may run at once for one dispatched step (the primary
    #: step review and the independent route monitor).
    REVIEW_WORKERS = 2

    def _parallel_review(
        self,
        instruction: str,
        step: PlanStep,
        next_step: Optional[PlanStep],
    ) -> StepReview:
        """Verify the step and audit the route at the same time.

        The primary model answers "did this step work, and is the next one
        ready?" while the independent secondary model separately answers "is the
        route still grounded?". With two DeepSeek keys they are different clients,
        so the pair costs the slower call instead of the sum: verification stops
        adding its full latency to every dispatched action.

        Fails open in every direction: an unavailable or erroring monitor leaves
        the primary verdict untouched, and a monitor rejection turns into an
        ordinary step failure so the existing re-align/re-plan path handles it.
        """
        monitor = None
        verifier = getattr(self, "_verifier", None)
        if verifier is not None and getattr(verifier, "enabled", False):
            candidate = getattr(verifier, "review_progress", None)
            if callable(candidate) and bool(
                getattr(self._settings, "parallel_verify", True)
            ):
                monitor = candidate
        if monitor is None:
            return self._review_step_and_next(instruction, step, next_step)

        started = time.perf_counter()
        with ThreadPoolExecutor(
            max_workers=self.REVIEW_WORKERS, thread_name_prefix="furti-verify"
        ) as pool:
            primary_future = pool.submit(
                self._review_step_and_next, instruction, step, next_step
            )
            monitor_future = pool.submit(
                monitor,
                instruction,
                step,
                next_step=next_step,
                execution_context=self._recent_context(),
            )
            try:
                primary = primary_future.result()
            except Exception as exc:  # noqa: BLE001 - review is advisory
                self._journal.warn(f"Step review failed ({exc}); continuing.")
                primary = StepReview(
                    step_ok=True,
                    step_reason=f"step review unavailable: {exc}",
                    attempted=False,
                )
            try:
                progress = monitor_future.result()
            except Exception as exc:  # noqa: BLE001 - monitoring is advisory
                self._journal.warn(f"Route monitor failed ({exc}); continuing.")
                return primary
        self._journal.thought(
            f"Step {step.index}: step review and route monitor ran in parallel "
            f"({time.perf_counter() - started:.2f}s total)."
        )
        if not progress.attempted:
            self._journal.thought(
                f"Route monitor skipped for step {step.index}: {progress.reason}"
            )
            return primary
        if progress.approved:
            return primary
        reason = progress.reason or "the route is no longer grounded"
        self._journal.warn(
            f"Step {step.index}: the independent route monitor rejected the "
            f"route: {reason}"
        )
        return StepReview(
            step_ok=False,
            step_reason=(
                f"independent route monitor "
                f"({progress.provider or 'secondary'}): {reason}"
            ),
            evidence_mode=primary.evidence_mode,
            visual_required=primary.visual_required,
            next_step_ready=False,
            next_step_reason="independent monitor rejected the route",
            attempted=True,
        )

    # ----------------------------------------------------------- verification
    def _review_step_and_next(
        self,
        instruction: str,
        step: PlanStep,
        next_step: Optional[PlanStep],
    ) -> StepReview:
        """Review the current and next step with one model request."""
        if not self._settings.verify_steps:
            return StepReview(
                step_ok=True,
                step_reason="post-step review disabled",
            )
        self._check_stop()
        # The review must look at a frame captured *after* the action: a
        # pre-action frame cannot prove the step worked. observe_for_review
        # waits out the capture floor (at most one second) to guarantee that.
        observe_for_review = getattr(self._context, "observe_for_review", None)
        if callable(observe_for_review):
            scene = observe_for_review(instruction, force_fresh=True)
        else:
            scene = self._context.observe(instruction, force_fresh=True)
        self._journal.thought(
            f"Reviewing step {step.index} and the next step together "
            f"(fresh screen={scene.fresh})..."
        )
        if not scene.fresh:
            reason = "post-step review could not capture a fresh frame"
            self._journal.warn(
                f"Step {step.index}: {reason}; the step result is unverified."
            )
            return StepReview(
                step_ok=True,
                step_reason=reason,
                next_step_ready=None,
                attempted=False,
            )

        next_block = (
            "No next step; this was the final planned action."
            if next_step is None
            else (
                f"Next planned step {next_step.index}: "
                f"{next_step.description} [{next_step.action.value}], "
                f"target={next_step.target or '(none)'}"
            )
        )
        user_prompt = (
            f"Task: {instruction}\n\n"
            f"Executed step {step.index}: {step.description} "
            f"[{step.action.value}]\n"
            f"Executed target: {step.target or '(focused window or bbox)'}\n\n"
            f"{next_block}\n\n"
            f"{scene.prompt_block()}\n\n"
            "Verify the executed step and assess whether the next planned "
            "step is ready. Select the most reliable evidence mode."
        )
        try:
            self._journal.thought(
                f"AI is reviewing step {step.index} and next-step readiness."
            )
            self._journal.waiting(
                f"Waiting for AI response (review step {step.index} + next)..."
            )
            model = self._planner._fast
            if scene.image_b64 and callable(getattr(model, "chat_vision", None)):
                raw = model.chat_vision(
                    STEP_REVIEW_SYSTEM_PROMPT,
                    user_prompt,
                    scene.image_b64,
                    purpose="step_review",
                )
            else:
                raw = model.chat_text(
                    STEP_REVIEW_SYSTEM_PROMPT,
                    user_prompt,
                    purpose="step_review",
                )
            model_name = getattr(model, "_model", type(model).__name__)
            self._journal.thought(
                f"AI response received (step {step.index} + next-step review)."
            )
            self._journal.ai_output(
                raw,
                model_name,
                f"step_review_{step.index}",
            )
            payload = extract_json_object(raw)
            step_payload = payload.get("step")
            if not isinstance(step_payload, dict):
                step_payload = payload.get("verification")
            if not isinstance(step_payload, dict):
                step_payload = payload
            next_payload = payload.get("next_step")
            if not isinstance(next_payload, dict):
                next_payload = {}

            ok = bool(
                step_payload.get(
                    "step_ok",
                    step_payload.get("ok", payload.get("step_ok", True)),
                )
            )
            reason = str(
                step_payload.get(
                    "step_reason",
                    step_payload.get("reason", payload.get("reason", "")),
                )
            ).strip()
            evidence_mode = str(
                payload.get(
                    "evidence_mode",
                    payload.get("input_mode", "visual" if scene.image_b64 else "text"),
                )
            ).lower().strip()
            if evidence_mode not in {"text", "visual", "both"}:
                evidence_mode = "text"
            visual_required = bool(
                payload.get("visual_required", evidence_mode in {"visual", "both"})
            )
            next_ready_raw = next_payload.get(
                "ready",
                payload.get("next_step_ready"),
            )
            next_ready = (
                None
                if next_step is None or next_ready_raw is None
                else bool(next_ready_raw)
            )
            next_reason = str(next_payload.get("reason", "")).strip()
            next_guidance = str(next_payload.get("guidance", "")).strip()
            if not ok and step.action is ActionType.MOVE:
                # A cursor reposition has no screen side effect to observe, so
                # a "failed" verdict here would only cause a retry/replan loop.
                self._journal.thought(
                    f"Step {step.index}: cursor move cannot be judged visually; "
                    "treating the dispatch as successful."
                )
                ok = True
                reason = (
                    "cursor move dispatched; no screen change expected"
                    + (f" ({reason})" if reason else "")
                )
            self._journal.thought(
                f"Review for step {step.index}: "
                f"{'OK' if ok else 'FAIL'}; evidence={evidence_mode}; "
                f"next_ready={next_ready if next_step is not None else 'n/a'}"
            )
            if next_step is not None and next_ready is False:
                self._journal.warn(
                    f"Next step {next_step.index} is not ready"
                    + (f": {next_reason}" if next_reason else ".")
                )
            return StepReview(
                step_ok=ok,
                step_reason=reason,
                evidence_mode=evidence_mode,
                visual_required=visual_required,
                next_step_ready=next_ready,
                next_step_reason=next_reason,
                next_step_guidance=next_guidance,
                attempted=True,
            )
        except Exception as exc:
            reason = f"post-step review unavailable: {exc}"
            self._journal.warn(f"Step {step.index}: {reason}")
            return StepReview(
                step_ok=True,
                step_reason=reason,
                evidence_mode="text" if not scene.image_b64 else "visual",
                attempted=False,
            )

    def _verify_step(self, instruction: str, step: PlanStep) -> tuple[bool, str]:
        """Compatibility wrapper for callers using the old verification API."""
        review = self._review_step_and_next(instruction, step, None)
        return review.step_ok, review.step_reason

    # ------------------------------------------------------ reflex compiling
    def _reflex_skip_reason(
        self,
        step: PlanStep,
        template_path: Optional[Path],
        scene: SceneObservation,
        anchor_note: str,
    ) -> Optional[str]:
        """Why this step must **not** be cached as a reflex (``None`` = compile).

        A reflex is only worth storing when it is both successful and plausibly
        reusable. A blurry crop, a whole-screen grab or a one-off typed payload
        costs a template file *and* a false cache hit on every later run, which
        is worse than having no reflex at all. Every rejection is journalled, so
        "why was nothing cached?" is always answerable.
        """
        settings = self._settings
        if not getattr(settings, "reflex_enabled", True):
            return "reflex compilation is disabled (FURTI_REFLEX=false)"
        if step.action not in REFLEX_ACTIONS:
            return f"{step.action.value} cannot be replayed by a reflex"
        if template_path is None or not Path(template_path).exists():
            return "the step has no visual template to replay"

        note = str(anchor_note or "")
        match = re.search(r"conf=([0-9]*\.?[0-9]+)", note)
        if "conf=" not in note and not note.startswith("OCR text"):
            # Nothing measurable to trust: "explicit bbox centre" and
            # "focused window" anchors are guesses, not observations.
            return f"the anchor ({note or 'unknown'}) is not a verified visual anchor"
        if match is not None:
            confidence = float(match.group(1))
            threshold = float(
                getattr(settings, "reflex_min_anchor_confidence", 0.75)
            )
            if confidence < threshold:
                return (
                    f"anchor confidence {confidence:.2f} is below "
                    f"reflex_min_anchor_confidence={threshold:.2f}"
                )

        name = self._memory.normalize_name(step.description)
        min_chars = int(getattr(settings, "reflex_min_description_chars", 8))
        if len(name) < min_chars:
            return (
                f"the description {step.description!r} is too generic to key a "
                "reusable reflex on"
            )

        if step.action is ActionType.TYPE:
            typed = str(step.text or "")
            limit = int(getattr(settings, "reflex_max_typed_chars", 160))
            if len(typed) > limit:
                return (
                    f"the typed payload is {len(typed)} characters, which is a "
                    "one-off rather than a reusable reflex"
                )

        image = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
        if image is None:
            return f"the template {Path(template_path).name} could not be read back"
        min_side = int(getattr(settings, "reflex_min_template_side", 6))
        if min(image.shape[0], image.shape[1]) < min_side:
            return (
                f"the template is too small ({image.shape[1]}x{image.shape[0]} px) "
                "to match reliably"
            )
        if scene is not None and scene.frame is not None:
            frame_area = int(scene.frame.shape[0]) * int(scene.frame.shape[1])
            ratio_limit = float(
                getattr(settings, "reflex_max_template_area_ratio", 0.6)
            )
            if frame_area > 0:
                ratio = (image.shape[0] * image.shape[1]) / frame_area
                if ratio > ratio_limit:
                    return (
                        f"the template covers {ratio:.0%} of the screen, which "
                        "would match anything and anchor nothing"
                    )
        return None

    def _compile_reflex(
        self,
        step: PlanStep,
        template_path: Optional[Path],
        scene: SceneObservation,
        anchor_note: str,
        drag_delta: Optional[tuple[int, int]] = None,
    ) -> Optional[str]:
        """Turn a successfully executed step into a reusable cached reflex.

        Only steps that are both successful *and* plausibly reusable are
        cached; see :meth:`_reflex_skip_reason`.
        """
        skip_reason = self._reflex_skip_reason(step, template_path, scene, anchor_note)
        if skip_reason is not None:
            self._journal.thought(
                f"Step {step.index}: not compiling a reflex -- {skip_reason}."
            )
            return None
        name = self._memory.normalize_name(step.description)
        params = {
            key: value for key, value in (step.params or {}).items() if _is_jsonable(value)
        }
        metadata: dict[str, Any] = {
            "description": step.description,
            "action": step.action.value,
            "text": step.text,
            "target": step.target,
            "params": params,
            "anchor": anchor_note,
            "screen_size": (
                [scene.frame.shape[1], scene.frame.shape[0]]
                if scene.frame is not None
                else None
            ),
            "compiled_by": "multi_step_executor",
        }
        if step.action is ActionType.DRAG and drag_delta is not None:
            # Replayed in input (mouse) space: VisionReflex maps the anchor
            # center itself, then applies this offset.
            metadata["drag_delta"] = [int(drag_delta[0]), int(drag_delta[1])]
            metadata["drag_button"] = str(params.get("button") or "left")
            if params.get("hold_keys"):
                hold_keys = params["hold_keys"]
                metadata["drag_hold_keys"] = (
                    [hold_keys] if isinstance(hold_keys, str) else list(hold_keys)
                )
        skill = Skill(
            name=name,
            template_path=str(template_path),
            action=step.action,
            metadata=metadata,
        )
        self._memory.save_skill(skill)
        self._journal.reflex(
            f"Compiled reusable reflex {name!r} from step {step.index} "
            f"(template: {template_path.name}); it starts disabled - enable it "
            "in the Reflexes tab to let it replay without the planner."
        )
        return name

    # ---------------------------------------------------------------- utils
    def _check_stop(self) -> None:
        if self._stop is not None and self._stop.is_set():
            raise TaskAborted("stop hotkey pressed during execution")
