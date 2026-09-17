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
from .jsoncontract import (
    LLMJsonError,
    as_mapping,
    as_pixel,
    as_text,
    extract_json_object,
)
from .models import ActionType, BoundingBox, coerce_action
from .tasklog import TaskJournal
from .tools import describe_tool_step, is_direct_tool

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
    '"action": "click|double_click|right_click|drag|move|type|scroll|key_press|'
    'launch_app|open_path|run_command|write_file|read_file|set_clipboard|'
    'get_clipboard|focus_window|list_windows|close_window|minimize_window|'
    'maximize_window|wait|screenshot|create_folder|list_dir|copy_path|'
    'move_path|delete_path|find_files|path_info|ask_user", '
    '"target": "<icon:name | exact OCR text | element description | null>", '
    '"x": 0, "y": 0, '
    '"bbox": {"x":0,"y":0,"width":0,"height":0} | null, '
    '"bbox_coordinate_space": "full_capture" | "attached_image", '
    '"text": "<text to type when action is type>", '
    '"params": {"key": "enter", "scroll_clicks": 3, "window": "Notepad"}, '
    '"thought": "<why this step, one line>"}]}\n'
    "Output contract (strict, verified before anything moves): respond with "
    "exactly ONE JSON object and nothing else -- no prose, no markdown fences, "
    "no comments, no trailing commas, no NaN or Infinity. Every pixel must be "
    "an integer in full_capture coordinates measured from the top-left of the "
    "screenshot, never negative. Omit a key (or send null) rather than "
    "inventing a value: a missing anchor is planned around, while a wrong "
    "number moves the mouse to the wrong place. A step that breaks the contract "
    "is rejected and never executed, so never guess a coordinate, an action "
    "name or a bbox size you cannot see.\n"
    "Rules: prefer targets anchored to OCR text or icon templates and give "
    "exact pixel coordinates only when the screenshot clearly shows them; "
    "use full_capture coordinates by default and set "
    "bbox_coordinate_space=attached_image only when the bbox is measured "
    "directly from the attached image dimensions; "
    "never invent coordinates you cannot see; keep the step count minimal; "
    "set requires_smart_model=true only when the task genuinely needs complex "
        "multi-condition reasoning; the deep pass re-runs the model in thinking "
        "mode and is slower, so use this flag sparingly. If the user's intent is genuinely ambiguous "
        "and you cannot safely choose, emit one action=ask_user step with "
        'params.question and params.options (an array of 2-6 concise choices); '
        "do not guess. The application will ask the user and send the answer back "
        "in a fresh planning request.\n"
    "Direct tools -- prefer these, because each one replaces a whole chain of "
    "GUI steps and cannot miss its target: launch_app (params.app, e.g. "
    '"notepad", "chrome", "code", "calculator"), open_path (params.path: a '
    'file, a folder or a URL such as "https://example.com"), run_command '
    "(params.command: a shell command, params.background=true to not wait for "
    "it), write_file (params.path + params.content, optional params.append, so "
    "text is saved without opening Notepad at all), read_file (params.path), "
    "screenshot (params.region {x,y,width,height} or params.full=true, "
    "optional params.label and params.path: saves the image and returns the "
    "path in the step result), set_clipboard (params.content) and "
    "get_clipboard, focus_window / minimize_window / maximize_window / "
    "close_window (params.window, or omit it to act on the active window), "
    "list_windows, and wait (params.seconds). These actions take no target or "
    "bbox: they are executed by the operating system. Use a tool whenever one "
    "can do the job -- starting an app, saving a file, copying text, opening a "
    "URL, running a command -- and fall back to click/type/key_press only for "
    "things that live purely inside a GUI (a canvas, a custom dialog, a web "
    "page widget). Never spend three GUI steps on something one tool does.\n"
    "File and folder work is NEVER a GUI task: create_folder (params.path), "
    "list_dir (params.path, optional params.pattern), copy_path / move_path "
    "(params.source + params.destination, move_path also renames), "
    "delete_path (params.path + params.confirm=true; it goes to the Recycle "
    "Bin), find_files (params.root, params.pattern, optional params.limit) and "
    "path_info (params.path) exist precisely so you never open File Explorer "
    "to look around, drag a file, or right-click -> rename. Check with "
    "path_info or find_files instead of navigating, and prefer absolute paths "
    "from the system context below.\n"

    "Window focus: for every action=key_press, action=type and action=scroll "
    "step, put the title of the application window that must receive the "
    'input into params.window (e.g. "Notepad", "Google Chrome", "File '
    "Explorer\"). The agent brings that window to the foreground before "
    "sending the keys; without it the keys may land in the wrong window.\n"
    "Interaction rules -- follow them exactly.\n"
    "1. Focus before typing: never emit a type step (or any keyboard step that "
    "enters text) without first clicking the input field it targets. Give the "
    "step that field as target (icon:name, or the exact OCR text of its label "
    'or placeholder) or as top-level "x"/"y" pixels whenever you can see it: '
    "the agent clicks that spot first so the field really has focus and the "
    "caret sits where the text will appear. A type step with no target types "
    "into whatever already has focus, which is only right when you mean that "
    "(right after opening a new document, or after focusing the window "
    "yourself). A step with params.focus_click=false types without clicking.\n"
    "2. Obstructed targets: before touching page or window content, look at the "
    "grounding for a popup, cookie banner, dialog or modal. If something covers "
    "your target -- or the target text is missing while the screen offers a "
    "Close/X/Accept affordance -- then dismissing it IS the next step (click "
    "'Close', 'X', 'Accept' or press ESC, whichever the grounding shows). Never "
    "click or read elements underneath an overlay, and never mistake a dialog's "
    "own text for the page it is covering.\n"
    "3. Timing guards: do not chain state-changing actions blindly. A UI needs "
    "roughly 200-500 ms to render, so after a click that navigates, opens a "
    "popup or re-renders a list, insert a wait step (params.seconds 0.3-0.5) "
    "before the next click; the agent also paces its own actions. If an action "
    "produced no screen or state change, do not repeat it -- rule out a modal "
    "or an unfocused window first, then take a different route.\n"
    "Keyboard shortcuts: for action=key_press put the whole chord in "
    'params.key, e.g. "ctrl+c" (copy), "ctrl+v" (paste), "ctrl+shift+t" '
    '(reopen tab), "alt+tab" (switch window), "win+r" (run dialog), "enter", '
    '"esc", "f5". Write the chord with "+" between the keys, never as separate '
    "key_press steps.\n"
    "Explicit screen coordinates: when you can see the exact pixel, put it in "
    'the top-level "x" and "y" (full_capture pixels) instead of an anchor. '
    "Then action=move puts the cursor there without clicking, and "
    "action=click/double_click/right_click presses the button there. Use this "
    "for elements with no readable text or icon template. Prefer a "
    "target/bbox anchor whenever the grounding gives you one.\n"
    "Drag and drop: use action=drag. target/bbox mark where the item is "
    "grabbed; params describe where to drop it, using exactly one of "
    '"to_target" ("<icon:name | exact OCR text>", resolved on the live '
    'screen), "to_bbox" ({"x":0,"y":0,"width":0,"height":0}, in '
    'params.to_coordinate_space or the step bbox_coordinate_space), "to_x" '
    'and "to_y" (absolute full_capture pixels) or "dx" and "dy" (pixels to '
    'move from the grab point). To drag something you are already hovering '
    '(no visible grab anchor), give the start as params "from_x"/"from_y" or '
    "omit the grab point entirely to start from the current cursor position. "
    'Optional: "button" ("left"|"right"|"middle", default "left"), '
    '"hold_keys" (e.g. "shift" for a shift-drag), "duration" (seconds for '
    "the movement).\n"
    "Drop then click: a drag ends with the button released. When the task also "
    "needs a click at the drop position, emit a separate "
    "action=click step with the same top-level x/y rather than expecting the "
    "drag to click for you.\n"
    "Clicking a second time: action=click accepts params.clicks (a count) for "
    "repeated presses without re-aiming."
)

REPLAN_SYSTEM_PROMPT = (
    "You are the re-planning component of Furti AI. A step of a larger task "
    "failed. Given the current screen grounding, the original step, and the "
    "failure reason, produce exactly ONE replacement action. "
    "Return JSON only (no markdown) in this shape:\n"
    '{"description": "...", "action": "click|double_click|right_click|drag|move|type|scroll|key_press|'
    'launch_app|open_path|run_command|write_file|read_file|set_clipboard|'
    'get_clipboard|focus_window|list_windows|close_window|minimize_window|'
    'maximize_window|wait", '
    '"target": "<icon:name | exact OCR text | element description | null>", '
    '"x": 0, "y": 0, "bbox": null, '
    '"bbox_coordinate_space": "full_capture", '
    '"text": null, "params": {}, "thought": "..."}'
    "\nReminder: params.key holds a full chord such as \"ctrl+c\"; a drag step "
    "grabs at target/bbox and drops at params.to_target/to_bbox/to_x+to_y/dx+dy; "
    "a step with top-level x/y acts on that exact pixel (action=move just moves "
    "the cursor, the click actions press the button there). For "
    "key_press/type/scroll steps set params.window to the window title that "
    "must receive the input. When a direct tool can do the job, prefer it over "
    "a mouse route: launch_app (params.app), open_path (params.path), "
    "run_command (params.command), write_file (params.path + params.content), "
    "focus_window/close_window (params.window), list_windows, wait "
    "(params.seconds). Those actions need no target or bbox."
)

REALIGN_SYSTEM_PROMPT = (
    "You are the re-alignment component of Furti AI. One planned step could not "
    "be anchored on the live screen, which means the target moved, the window "
    "scrolled, or the stored reflex for it is stale. Keep the SAME intent and, "
    "wherever it is still possible, the SAME action -- your job is a corrected "
    "anchor, not a new route. Read the current screen grounding and return the "
    "step aimed at where the element is NOW.\n"
    "Return JSON only (no markdown) in exactly this shape:\n"
    '{"description": "<same intent, corrected wording>", '
    '"action": "<the same action, unless a direct tool now does the job better>", '
    '"target": "<icon:name | exact OCR text from the grounding below '
    '| element description | null>", "x": 0, "y": 0, '
    '"bbox": {"x":0,"y":0,"width":0,"height":0} | null, '
    '"bbox_coordinate_space": "full_capture", '
    '"text": null, "params": {}, "thought": "<what changed and why>"}\n'
    "Prefer an exact OCR label or icon name that appears in the grounding "
    "below; only use an explicit bbox or x/y when the element has no readable "
    "text. Do not invent coordinates you cannot see. If the element genuinely "
    "is not on screen, return the best available anchor for the step's "
    "intent (for example the window that must be focused first) instead of "
    "leaving the anchor empty."
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
    "steps list.\n"
    "Every step must carry its full payload: key_press steps need "
    'params.key (the whole chord, e.g. "win", "enter", "ctrl+l", '
    '"ctrl+shift+t"), type steps need text and, to type into a specific '
    'field, that field as target (the agent clicks it first), and click steps need a target '
    "and, when an explicit pixel is intended, a bbox dict "
    '{"x":0,"y":0,"width":0,"height":0} or top-level "x"/"y" pixels. '
    'action=move only repositions the cursor (needs "x"/"y" or a target) and '
    "never clicks, so follow it with a click action when a button press is "
    "required. Key_press, type and scroll steps must also name the destination "
    "window in params.window so the agent can focus it before sending input. "
    "Prefer the direct tools over fresh GUI routes whenever they can reach the "
    "result: launch_app (params.app), open_path (params.path), run_command "
    "(params.command), write_file (params.path + params.content), read_file "
    "(params.path), set_clipboard/get_clipboard, focus_window / "
    "minimize_window / maximize_window / close_window (params.window), "
    "list_windows and wait (params.seconds). Those need no target or bbox."
)


class BudgetExceeded(Exception):
    """Raised when the per-task LLM call budget is exhausted."""


def _point_from_payload(value: Any) -> Optional[tuple[int, int]]:
    """Read an ``(x, y)`` pixel pair from the shapes a model emits.

    Accepts ``{"x": 1, "y": 2}``, ``[1, 2]`` and ``[left, top, right, bottom]``
    (corner form, whose centre is used) so a step that names an explicit screen
    position is never silently ignored. Numbers are validated as pixels: a
    non-numeric, negative, non-finite or absurdly large coordinate raises
    :class:`~furti_ai.jsoncontract.LLMJsonError` instead of becoming a cursor
    move, because acting on a broken payload is worse than failing the step.
    """
    if isinstance(value, dict):
        x, y = value.get("x"), value.get("y")
        if x is None and y is None:
            return None
        if x is None or y is None:
            raise LLMJsonError(
                "a point needs both x and y; only one of them was supplied"
            )
        pixel_x = as_pixel(x, field="x")
        pixel_y = as_pixel(y, field="y")
        return int(pixel_x), int(pixel_y)
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        if len(value) != 2 and len(value) != 4:
            raise LLMJsonError(
                f"a point/box must have 2 or 4 numbers, got {len(value)}"
            )
        numbers = [
            as_pixel(item, field=f"point[{position}]")
            for position, item in enumerate(value)
        ]
        if len(numbers) == 2:
            return int(numbers[0]), int(numbers[1])
        # Corner form [left, top, right, bottom].
        return (int(numbers[0]) + int(numbers[2])) // 2, (
            int(numbers[1]) + int(numbers[3])
        ) // 2
    if isinstance(value, (int, float)):
        raise LLMJsonError(f"a point must be an object or array, got {value!r}")
    return None


def _normalise_step_point(
    data: dict[str, Any], params: dict[str, Any]
) -> Optional[tuple[int, int]]:
    """Resolve the explicit pixel a step acts on, normalised into ``params``.

    Models put the coordinate in different places depending on how strictly
    they follow the schema (top level ``x``/``y``, a ``point`` object, or inside
    ``params``). Whichever spelling arrives, the executor reads it back from
    ``params['x']``/``params['y']``, so the payload is copied there.
    """
    candidates = (
        {"x": data.get("x"), "y": data.get("y")},
        data.get("point"),
        {"x": params.get("x"), "y": params.get("y")},
        params.get("point"),
    )
    for candidate in candidates:
        point = _point_from_payload(candidate)
        if point is not None:
            params["x"], params["y"] = point
            return point
    return None


def _describe_drop(params: dict[str, Any]) -> str:
    """One-line summary of where a drag step drops its payload."""
    if params.get("to_target"):
        return str(params["to_target"])
    if isinstance(params.get("to_bbox"), dict):
        box = params["to_bbox"]
        return f"bbox {box.get('x')},{box.get('y')} {box.get('width')}x{box.get('height')}"
    if params.get("to_x") is not None and params.get("to_y") is not None:
        return f"({params['to_x']}, {params['to_y']})"
    if params.get("dx") is not None or params.get("dy") is not None:
        return f"offset ({params.get('dx', 0)}, {params.get('dy', 0)})"
    return "?"


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
    window: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], index: int) -> "PlanStep":
        """Build one step from a model payload, refusing anything unusable.

        The contract is enforced here rather than at dispatch time because this
        is the only place a control payload enters the system: an unknown action
        name, a non-numeric or negative coordinate, a negative bbox size or a
        ``type`` step with no text raises :class:`LLMJsonError`, and the planner
        then re-asks (escalating to the smarter model) instead of the executor
        moving the cursor somewhere plausible but wrong.
        """
        raw_action = data.get("action")
        action = coerce_action(raw_action, None)
        if action is None:
            raise LLMJsonError(
                f"step {index}: unknown action {raw_action!r}; use one of the "
                "documented action names"
            )

        bbox: Optional[BoundingBox] = None
        raw_bbox = data.get("bbox")
        if isinstance(raw_bbox, dict):
            try:
                bbox = BoundingBox.from_dict(raw_bbox)
            except (TypeError, ValueError) as exc:
                raise LLMJsonError(
                    f"step {index}: bbox must contain integer x, y, width and "
                    f"height ({exc})"
                ) from exc
            if bbox.x < 0 or bbox.y < 0:
                raise LLMJsonError(
                    f"step {index}: bbox origin {bbox.x},{bbox.y} is negative; "
                    "coordinates start at the top-left of the screenshot"
                )
            if bbox.width < 0 or bbox.height < 0:
                raise LLMJsonError(
                    f"step {index}: bbox size {bbox.width}x{bbox.height} is "
                    "negative"
                )
            if bbox.area <= 0:
                # The documented placeholder for "no bbox in this step".
                bbox = None
        elif isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
            # Vision models often ignore the dict schema and return
            # [left, top, right, bottom] corner pixels. Treating the array as
            # x/y/width/height would push the click far off screen, so the
            # model-output path reads lists as corners (same rule as
            # BrainPlanner._normalize_model_bbox).
            left, top, right, bottom = (
                as_pixel(value, field=f"bbox[{position}]")
                for position, value in enumerate(raw_bbox)
            )
            if right > left and bottom > top:
                bbox = BoundingBox.from_corners(left, top, right, bottom)
        elif raw_bbox not in (None, {}, []):
            raise LLMJsonError(
                f"step {index}: bbox must be an object with x, y, width and "
                f"height, got {type(raw_bbox).__name__}"
            )

        params = as_mapping(data.get("params"), field="params")
        params = dict(params)
        _normalise_step_point(data, params)
        bbox_space = str(
            data.get("bbox_coordinate_space")
            or data.get("coordinate_space")
            or params.get("bbox_coordinate_space")
            or ""
        ).strip().lower()
        text = as_text(data.get("text"), field="text") or as_text(
            params.get("text"), field="params.text"
        )
        if action is ActionType.TYPE and not (text or "").strip():
            raise LLMJsonError(
                f"step {index}: a type action must carry the text to type in "
                '"text"'
            )
        return cls(
            index=index,
            description=str(data.get("description", "")).strip(),
            action=action,
            bbox=bbox,
            text=text or None,
            target=data.get("target") or None,
            params=params,
            thought=str(data.get("thought", "")).strip(),
            bbox_space=bbox_space,
            window=data.get("window") or params.get("window") or None,
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
            if step.action is ActionType.ASK_USER:
                options = step.params.get("options") or []
                choice_text = ", ".join(str(option) for option in options)
                suffix = f" [{choice_text}]" if choice_text else ""
                lines.append(
                    f"  {step.index}. {step.description} "
                    f"[ask_user]{suffix}"
                )
                continue
            if is_direct_tool(step.action):
                # Tools have no screen anchor: show their arguments instead.
                lines.append(
                    f"  {step.index}. {step.description} "
                    f"[{step.action.value}] -> {describe_tool_step(step)}"
                )
                continue
            anchor = step.target or (
                f"bbox {step.bbox.x},{step.bbox.y} {step.bbox.width}x{step.bbox.height}"
                if step.bbox
                else "current cursor"
            )
            suffix = ""
            if step.action is ActionType.KEY_PRESS:
                suffix = f" key={step.params.get('key') or step.text or '?'}"
            elif step.action is ActionType.DRAG:
                suffix = f" drop={_describe_drop(step.params)}"
            elif step.action is ActionType.MOVE:
                suffix = (
                    f" to ({step.params.get('x')}, {step.params.get('y')})"
                    if step.params.get("x") is not None
                    else " to current cursor"
                )
            window = step.window or step.params.get("window")
            if step.action in (
                ActionType.KEY_PRESS,
                ActionType.TYPE,
                ActionType.SCROLL,
            ) and window:
                suffix += f" window={window}"
            lines.append(
                f"  {step.index}. {step.description} "
                f"[{step.action.value}] -> {anchor}{suffix}"
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
        profile: Any = None,
    ) -> None:
        self._fast = fast_llm
        self._smart = smart_llm
        self._settings = settings
        self._journal = journal
        self._context = visual_context
        self._stop = stop_event
        #: The user/system "context file" (see ``profile.py``). Optional so the
        #: planner keeps working in tests and minimal embeddings.
        self._profile = profile

    # ------------------------------------------------------------ context
    def _profile_block(self) -> str:
        """Compact user/system context injected into every prompt."""
        profile = self._profile
        if profile is None:
            return ""
        block = getattr(profile, "prompt_block", None)
        if not callable(block):
            return ""
        try:
            return block() or ""
        except Exception as exc:  # noqa: BLE001 - context is a bonus, never fatal
            logger.debug("profile prompt block failed: %s", exc)
            return ""

    # ------------------------------------------------------------ public API
    def plan(self, instruction: str) -> TaskPlan:
        """Produce the multi-step plan, escalating models as needed."""
        self._journal.thought(f"Decomposing instruction into steps: {instruction!r}")
        scene = self._context.observe(instruction)

        system = PLAN_SYSTEM_PROMPT
        profile_block = self._profile_block()
        user = (
            f"User instruction: {instruction}\n\n"
            + (f"{profile_block}\n\n" if profile_block else "")
            + f"{scene.prompt_block()}\n\n"
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

    def realign_step(
        self,
        instruction: str,
        failed: PlanStep,
        failure_reason: str,
        scene: SceneObservation,
        attempt: int = 1,
    ) -> PlanStep:
        """Re-anchor one step on the live screen instead of replacing it.

        This is the *reflex re-alignment* path: when a step cannot be grounded,
        the intent is usually still right and only the anchor is stale (the
        window scrolled, the dialog moved, the stored template no longer
        matches). Asking for a corrected anchor keeps the route and lets the
        freshly-anchored action overwrite the stale reflex, whereas
        :meth:`replan_step` deliberately routes around the step with a
        different action.
        """
        self._check_stop()
        self._assert_budget("realign")
        model = (
            self._smart
            if self._smart is not None
            and attempt >= self._settings.max_step_retries
            else self._fast
        )
        self._journal.thought(
            f"AI is re-aligning step {failed.index} on the current screen "
            f"(attempt {attempt})."
        )
        self._journal.waiting(
            f"Waiting for {self._model_label(model)} response "
            f"(re-align step {failed.index})..."
        )
        profile_block = self._profile_block()
        anchor = failed.target or (
            f"bbox {failed.bbox.x},{failed.bbox.y} "
            f"{failed.bbox.width}x{failed.bbox.height}"
            if failed.bbox
            else "(none)"
        )
        user = (
            f"Whole task: {instruction}\n\n"
            f"Step {failed.index} could not be anchored on the live screen.\n"
            f"Step description: {failed.description}\n"
            f"Step action: {failed.action.value}\n"
            f"Previous anchor: {anchor}\n"
            f"Parameters: {json.dumps(failed.params or {}, default=str)}\n"
            f"Failure reason: {failure_reason}\n\n"
            + (f"{profile_block}\n\n" if profile_block else "")
            + f"{scene.prompt_block()}\n\n"
            "Return the same step re-anchored to what the current screen "
            "actually shows."
        )
        raw = (
            model.chat_vision(
                REALIGN_SYSTEM_PROMPT, user, scene.image_b64, purpose="realign"
            )
            if scene.vision_used and scene.image_b64
            else model.chat_text(REALIGN_SYSTEM_PROMPT, user, purpose="realign")
        )
        self._journal.thought(
            f"AI response received from {self._model_label(model)} "
            f"(re-align step {failed.index})."
        )
        self._journal.ai_output(raw, self._model_name(model), f"realign_{failed.index}")
        payload = self._parse_json(raw)
        realigned = PlanStep.from_dict(payload, failed.index)
        self._map_step_bbox_to_frame(realigned, scene)
        if not realigned.description:
            realigned.description = failed.description
        if realigned.action is not failed.action:
            self._journal.warn(
                f"Re-alignment for step {failed.index} changed the action from "
                f"{failed.action.value} to {realigned.action.value}; using the "
                "new action."
            )
        self._journal.thought(
            f"Re-aligned step {failed.index}: {realigned.description} "
            f"[{realigned.action.value}] -> "
            f"{realigned.target or 'explicit coordinates'}"
        )
        return realigned

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

        profile_block = self._profile_block()
        user = (
            f"Whole task: {instruction}\n\n"
            f"Failed step: {failed.description} [{failed.action.value}]\n"
            f"Failure reason: {failure_reason}\n\n"
            + (f"{profile_block}\n\n" if profile_block else "")
            + f"{scene.prompt_block()}\n\n"
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
        profile_block = self._profile_block()
        user = (
            f"Whole task: {instruction}\n\n"
            f"Completed steps:\n{completed}\n\n"
            f"Failed step: {failed_step.description} [{failed_step.action.value}]\n"
            f"Failure reason: {failure_reason}\n\n"
            f"Original remaining plan:\n{remaining}\n\n"
            + (f"{profile_block}\n\n" if profile_block else "")
            + f"{scene.prompt_block()}\n\n"
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
    def _convert_bbox(
        bbox: BoundingBox,
        space: str,
        image_size: tuple[int, int],
        frame_size: tuple[int, int],
    ) -> tuple[BoundingBox, str]:
        """Scale a bbox from attached-image pixels to full-capture pixels."""
        image_width, image_height = image_size
        frame_width, frame_height = frame_size
        if space in {"full", "full_capture", "screen", "frame"}:
            return bbox, "full_capture"
        if not (
            0 <= bbox.x
            and 0 <= bbox.y
            and bbox.x + bbox.width <= image_width
            and bbox.y + bbox.height <= image_height
        ):
            # Coordinates outside the attached image are already in the
            # full-capture space (the model had OCR dimensions in view).
            return bbox, "full_capture"
        scale_x = frame_width / image_width
        scale_y = frame_height / image_height
        return (
            BoundingBox(
                x=round(bbox.x * scale_x),
                y=round(bbox.y * scale_y),
                width=max(1, round(bbox.width * scale_x)),
                height=max(1, round(bbox.height * scale_y)),
            ).clamp(frame_width, frame_height),
            "full_capture",
        )

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

        image_size = (image_width, image_height)
        frame_size = (frame_width, frame_height)
        for step in plan.steps:
            if step.bbox is not None:
                step.bbox, step.bbox_space = TaskPlanner._convert_bbox(
                    step.bbox, step.bbox_space, image_size, frame_size
                )
            TaskPlanner._map_step_point_to_frame(
                step, image_size, frame_size
            )
            to_bbox = step.params.get("to_bbox")
            if not isinstance(to_bbox, dict):
                continue
            try:
                parsed = BoundingBox.from_dict(to_bbox)
            except (TypeError, ValueError):
                continue
            if parsed.area <= 0:
                continue
            space = str(
                step.params.get("to_coordinate_space") or step.bbox_space or ""
            ).strip().lower()
            converted, _ = TaskPlanner._convert_bbox(
                parsed, space, image_size, frame_size
            )
            # Params stay JSON-friendly: they end up in journals and skills.
            step.params["to_bbox"] = {
                "x": converted.x,
                "y": converted.y,
                "width": converted.width,
                "height": converted.height,
            }
            step.params["to_coordinate_space"] = "full_capture"

    @staticmethod
    def _map_step_point_to_frame(
        step: PlanStep,
        image_size: tuple[int, int],
        frame_size: tuple[int, int],
    ) -> None:
        """Convert an explicit step point from attached-image to frame pixels."""
        x = step.params.get("x")
        y = step.params.get("y")
        if x is None or y is None:
            return
        space = str(
            step.params.get("point_coordinate_space") or step.bbox_space or ""
        ).strip().lower()
        try:
            point = BoundingBox(int(x), int(y), 1, 1)
        except (TypeError, ValueError):
            return
        converted, _ = TaskPlanner._convert_bbox(
            point, space, image_size, frame_size
        )
        step.params["x"], step.params["y"] = converted.x, converted.y
        step.params["point_coordinate_space"] = "full_capture"

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
        # Truncate first, then enforce: an inserted focus click is a hard
        # requirement, while the cap is a soft budget, so the click is never the
        # step that gets dropped.
        self._enforce_focus_before_typing(plan)
        self._journal.plan(f"Plan ready: {len(plan.steps)} step(s).\n{plan.describe()}")
        for step in plan.steps:
            if step.thought:
                self._journal.thought(f"Step {step.index} reasoning: {step.thought}")
        return plan

    def _enforce_focus_before_typing(self, plan: TaskPlan) -> None:
        """Make the "click the field first" step explicit for every typing step.

        The executor clicks a resolved field before typing anyway, but doing it
        as its own plan step is what turns the required sequence into
        click -> verify -> type: the click is visible in the plan preview, it
        gets its own post-step review (so a stray dialog is caught before any
        text is sent), and a re-plan can drop just that step. The type step
        keeps its target so the anchor is still re-resolved on the live screen,
        and the executor skips its own click when the previous step already
        pressed that exact point.
        """
        enforced: list[PlanStep] = []
        inserted = 0
        for step in plan.steps:
            if self._needs_focus_step(step, enforced):
                enforced.append(self._focus_step_for(step))
                inserted += 1
            enforced.append(step)
        if not inserted:
            return
        for index, step in enumerate(enforced, start=1):
            step.index = index
        plan.steps = enforced
        self._journal.thought(
            f"Focus-before-typing rule: inserted {inserted} explicit click "
            "step(s) so no text is typed before its field is focused."
        )

    @staticmethod
    def _field_anchor(step: PlanStep) -> bool:
        """True when a step names the field it acts on (anchor or pixel)."""
        if str(step.target or "").strip():
            return True
        if step.bbox is not None:
            return True
        params = step.params or {}
        return params.get("x") is not None and params.get("y") is not None

    def _needs_focus_step(
        self, step: PlanStep, emitted: list[PlanStep]
    ) -> bool:
        """Whether this typing step still needs its own focus click."""
        if step.action is not ActionType.TYPE:
            return False
        if not self._field_anchor(step):
            # Nothing to click: the plan explicitly types into the focused
            # control, which the prompt allows for a deliberately focused window.
            return False
        previous = emitted[-1] if emitted else None
        if previous is not None and previous.action in (
            ActionType.CLICK,
            ActionType.DOUBLE_CLICK,
            ActionType.RIGHT_CLICK,
        ):
            # The model already planned the click; do not press the field twice.
            return False
        return True

    @staticmethod
    def _focus_step_for(step: PlanStep) -> PlanStep:
        """Build the click step that focuses ``step``'s field."""
        params = {
            key: value
            for key, value in (step.params or {}).items()
            if key in {"x", "y", "point_coordinate_space", "bbox_coordinate_space"}
        }
        label = str(step.target or step.description or "the field").strip()
        return PlanStep(
            index=step.index,
            description=f"Click {label} to focus it",
            action=ActionType.CLICK,
            bbox=step.bbox,
            target=step.target,
            params=params,
            thought=(
                "Focus-before-typing rule: the field must be clicked so it owns "
                "the caret before any text is sent."
            ),
            bbox_space=step.bbox_space,
            window=step.window,
        )

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
        """Read the one JSON object the planner asked for.

        Delegates to the shared contract reader so fences/prose are tolerated
        while NaN, truncated and non-object answers are refused.
        """
        return extract_json_object(text)
