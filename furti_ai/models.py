"""Shared data structures for Furti AI.

These plain dataclasses flow between the four core modules:

* ``BoundingBox``  -- pixel geometry of a UI element.
* ``ActionPlan``   -- the structured plan returned by the reasoning brain.
* ``ActionType``   -- the low-level action vocabulary a reflex can perform.
* ``Skill``        -- a *compiled* reflex: template image + metadata + stats.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UserChoiceRequest:
    """A model-authored question that needs a human answer before continuing."""

    question: str
    options: tuple[str, ...] = ()


class ActionType(str, Enum):
    """The set of actions a plan step can perform.

    The first group drives the screen through the mouse/keyboard layer and is
    the only group a compiled reflex (see ``vision.py``) can replay. The second
    group -- the *direct tools* -- asks the operating system to do the work
    instead: launching Notepad is one process spawn, not a Run-dialog chord, a
    window wait, an OCR round trip and a verification call.
    """

    # ---------------------------------------------------------- screen input
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    DRAG = "drag"
    TYPE = "type"
    SCROLL = "scroll"
    KEY_PRESS = "key_press"
    #: Reposition the cursor at an explicit pixel without pressing a button.
    MOVE = "move"

    # --------------------------------------------------------- direct tools
    #: Start an application by name/path instead of hunting its icon.
    LAUNCH_APP = "launch_app"
    #: Open a file, folder or URI with the OS's registered handler.
    OPEN_PATH = "open_path"
    #: Run a shell command and read its output.
    RUN_COMMAND = "run_command"
    #: Write text to a file on disk.
    WRITE_FILE = "write_file"
    #: Read a text file so the model can reason about its contents.
    READ_FILE = "read_file"
    #: Put text on the clipboard (no Ctrl+C dance needed).
    SET_CLIPBOARD = "set_clipboard"
    #: Read the clipboard as text.
    GET_CLIPBOARD = "get_clipboard"
    #: Bring an existing window to the foreground by title.
    FOCUS_WINDOW = "focus_window"
    #: Enumerate the visible top-level windows and their titles.
    LIST_WINDOWS = "list_windows"
    CLOSE_WINDOW = "close_window"
    MINIMIZE_WINDOW = "minimize_window"
    MAXIMIZE_WINDOW = "maximize_window"
    #: Pause for a bounded number of seconds (app startup, page load).
    WAIT = "wait"
    #: Capture the screen (or one region of it) and save the image locally.
    SCREENSHOT = "screenshot"
    #: File-system work. Doing this through Explorer is dozens of clicks;
    #: through the OS it is one call.
    CREATE_FOLDER = "create_folder"
    LIST_DIR = "list_dir"
    COPY_PATH = "copy_path"
    MOVE_PATH = "move_path"
    DELETE_PATH = "delete_path"
    FIND_FILES = "find_files"
    PATH_INFO = "path_info"
    #: Fetch readable web content without opening a browser or screenshot.
    SCRAPE_URL = "scrape_url"
    #: Pause planning and ask the user to resolve an ambiguity.
    ASK_USER = "ask_user"


#: The screen-driving actions a compiled reflex can replay. Kept separate from
#: :class:`ActionType` so the legacy single-action planner never asks the model
#: for a tool that has no template to compile.
REFLEX_ACTIONS: tuple[ActionType, ...] = (
    ActionType.CLICK,
    ActionType.DOUBLE_CLICK,
    ActionType.RIGHT_CLICK,
    ActionType.DRAG,
    ActionType.TYPE,
    ActionType.SCROLL,
    ActionType.KEY_PRESS,
    ActionType.MOVE,
)


#: Alternate spellings models emit for the actions above. Vision models rarely
#: use the exact enum name, and an unrecognised name silently degrades to a
#: plain click that then fails anchor resolution ("no target anchor found"),
#: which looks like "the agent ignored my drag/click".
_ACTION_ALIASES = {
    "click_at": "click",
    "clickat": "click",
    "left_click": "click",
    "leftclick": "click",
    "mouse_click": "click",
    "move_to": "move",
    "moveto": "move",
    "mouse_move": "move",
    "move_mouse": "move",
    "move_pointer": "move",
    "position_cursor": "move",
    "move_cursor": "move",
    "goto": "move",
    "go_to": "move",
    "hover": "move",
    "hover_over": "move",
    "mouse_over": "move",
    "mouseover": "move",
    "drag_to": "drag",
    "dragto": "drag",
    "drag_and_drop": "drag",
    "doubleclick": "double_click",
    "rightclick": "right_click",
    "type_text": "type",
    "typewrite": "type",
    "write": "type",
    "keypress": "key_press",
    "key": "key_press",
    "press_key": "key_press",
    "send_keys": "key_press",
    "key_combo": "key_press",
    "hotkey": "key_press",
    "hot_key": "key_press",
    "shortcut": "key_press",
    "wheel": "scroll",
    "mouse_wheel": "scroll",
    "scroll_wheel": "scroll",
    # --- direct tools -------------------------------------------------
    "open_app": "launch_app",
    "openapp": "launch_app",
    "start_app": "launch_app",
    "startapp": "launch_app",
    "launch": "launch_app",
    "launch_application": "launch_app",
    "run_app": "launch_app",
    "run_program": "launch_app",
    "start_program": "launch_app",
    "open_program": "launch_app",
    "open_application": "launch_app",
    "open_file": "open_path",
    "open_folder": "open_path",
    "open_directory": "open_path",
    "open_document": "open_path",
    "open_url": "open_path",
    "open_website": "open_path",
    "browse": "open_path",
    "browse_to": "open_path",
    "navigate": "open_path",
    "navigate_to": "open_path",
    "visit": "open_path",
    "run": "run_command",
    "run_shell": "run_command",
    "shell": "run_command",
    "shell_command": "run_command",
    "exec": "run_command",
    "execute": "run_command",
    "execute_command": "run_command",
    "run_command_line": "run_command",
    "save_file": "write_file",
    "create_file": "write_file",
    "write_text_file": "write_file",
    "append_file": "write_file",
    "write_to_file": "write_file",
    "read_text_file": "read_file",
    "open_and_read_file": "read_file",
    "copy_to_clipboard": "set_clipboard",
    "clipboard_set": "set_clipboard",
    "set_clipboard_text": "set_clipboard",
    "copy_text": "set_clipboard",
    "paste_from_clipboard": "get_clipboard",
    "clipboard_get": "get_clipboard",
    "read_clipboard": "get_clipboard",
    "get_clipboard_text": "get_clipboard",
    "activate_window": "focus_window",
    "switch_to_window": "focus_window",
    "bring_to_front": "focus_window",
    "bring_window_to_front": "focus_window",
    "raise_window": "focus_window",
    "enumerate_windows": "list_windows",
    "get_windows": "list_windows",
    "list_open_windows": "list_windows",
    "quit_window": "close_window",
    "exit_window": "close_window",
    "minimize": "minimize_window",
    "maximize": "maximize_window",
    "sleep": "wait",
    "delay": "wait",
    "pause": "wait",
    "wait_for": "wait",
    # --- screenshots --------------------------------------------------
    "take_screenshot": "screenshot",
    "takescreenshot": "screenshot",
    "capture_screenshot": "screenshot",
    "screenshot_region": "screenshot",
    "region_screenshot": "screenshot",
    "screen_capture": "screenshot",
    "capture_region": "screenshot",
    "save_screenshot": "screenshot",
    "crop_screenshot": "screenshot",
    "snapshot": "screenshot",
    # --- file system --------------------------------------------------
    "mkdir": "create_folder",
    "make_folder": "create_folder",
    "make_directory": "create_folder",
    "new_folder": "create_folder",
    "create_directory": "create_folder",
    "list_directory": "list_dir",
    "list_folder": "list_dir",
    "list_files": "list_dir",
    "browse_folder": "list_dir",
    "dir_listing": "list_dir",
    "copy_file": "copy_path",
    "copy_folder": "copy_path",
    "copy_directory": "copy_path",
    "duplicate_file": "copy_path",
    "move_file": "move_path",
    "move_folder": "move_path",
    "rename_file": "move_path",
    "rename_path": "move_path",
    "rename": "move_path",
    "delete_file": "delete_path",
    "delete_folder": "delete_path",
    "remove_file": "delete_path",
    "remove_path": "delete_path",
    "search_files": "find_files",
    "find_file": "find_files",
    "glob_files": "find_files",
    "locate_file": "find_files",
    "file_info": "path_info",
    "path_exists": "path_info",
    "stat_path": "path_info",
    "check_path": "path_info",
}

#: Decorative tokens models bolt onto an action name. They carry no meaning:
#: "mouse_move" is a move and "double_click_at" is a double click. Stripping
#: them keeps an unknown spelling from degrading to a plain click.
_ACTION_PREFIXES = ("mouse_", "cursor_", "pointer_")
_ACTION_SUFFIXES = ("_at", "_to", "_cursor", "_mouse", "_pointer")


def _resolve_action_name(name: str) -> Optional["ActionType"]:
    """Look one spelling up in the alias map and then the enum."""
    for _ in range(3):
        alias = _ACTION_ALIASES.get(name)
        if alias is None:
            break
        name = alias
    try:
        return ActionType(name)
    except ValueError:
        return None


def _action_candidates(name: str) -> list[str]:
    """Expand a spelling into alias-free variants, most literal first."""
    candidates = [name]
    known = {name}
    for _ in range(3):
        additions: set[str] = set()
        for candidate in candidates:
            for prefix in _ACTION_PREFIXES:
                if candidate.startswith(prefix) and len(candidate) > len(prefix):
                    additions.add(candidate[len(prefix) :])
            for suffix in _ACTION_SUFFIXES:
                if candidate.endswith(suffix) and len(candidate) > len(suffix):
                    additions.add(candidate[: -len(suffix)])
        additions -= known
        if not additions:
            break
        candidates.extend(sorted(additions))
        known |= additions
    # Natural-language spellings such as "move_the_mouse" are resolved by their
    # leading verb, which stays the least trusted candidate.
    head = name.split("_", 1)[0]
    if head and head not in candidates:
        candidates.append(head)
    return candidates


def coerce_action(value: Any, default: Optional["ActionType"] = None) -> Optional["ActionType"]:
    """Map a model-supplied action string onto an :class:`ActionType`.

    Returns ``default`` when the name is missing or unknown so callers stay in
    control of the fallback instead of the parser guessing.
    """
    raw = str(value or "").strip()
    if "+" in raw:
        # The model occasionally emits the chord itself ("ctrl+shift+t")
        # instead of the action name.
        return ActionType.KEY_PRESS
    name = raw.lower().replace("-", "_").replace(" ", "_")
    if not name:
        return default
    for candidate in _action_candidates(name):
        resolved = _resolve_action_name(candidate)
        if resolved is not None:
            return resolved
    return default


@dataclass
class BoundingBox:
    """Axis-aligned rectangle in screen pixel coordinates."""

    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> tuple[int, int]:
        """The pixel at the middle of the box (click target)."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        """Number of pixels covered by the box."""
        return self.width * self.height

    def clamp(self, width: int, height: int) -> "BoundingBox":
        """Return a copy clipped to a screen/image of the given size."""
        x1 = max(0, self.x)
        y1 = max(0, self.y)
        x2 = min(self.x + self.width, width)
        y2 = min(self.y + self.height, height)
        return BoundingBox(x1, y1, max(0, x2 - x1), max(0, y2 - y1))

    @classmethod
    def from_corners(
        cls, left: int, top: int, right: int, bottom: int
    ) -> "BoundingBox":
        """Build a box from ``[left, top, right, bottom]`` corner pixels.

        Vision models commonly emit corner coordinates despite a schema that
        asks for ``x/y/width/height``. This is kept separate from
        :meth:`from_dict` (whose list form means ``[x, y, width, height]`` for
        callers that construct plans directly) so each shape stays unambiguous.
        """
        left, top, right, bottom = (
            int(left),
            int(top),
            int(right),
            int(bottom),
        )
        return cls(left, top, max(0, right - left), max(0, bottom - top))

    @classmethod
    def from_dict(cls, data: dict[str, Any] | list[Any] | tuple[Any, ...] | None) -> "BoundingBox":
        """Build a box from one of the common shapes the planner sees.

        Raw model output is noisy. Some adapters emit a mapping with keys such
        as ``x``, ``y``, ``width`` and ``height``. Others flatten the payload as
        ``[x, y, width, height]`` or ``(x, y, width, height)``. If the payload is
        absent or malformed, fail safely with a readable default instead of
        crashing the action-plan parser.
        """
        logger.debug("Parsing BoundingBox payload shape=%s value=%r", type(data).__name__, data)
        if not data:
            raise ValueError("bounding box payload is empty")

        if isinstance(data, dict):
            box = cls(
                x=int(data.get("x", 0)),
                y=int(data.get("y", 0)),
                width=int(data.get("width", 0)),
                height=int(data.get("height", 0)),
            )
            logger.debug("BoundingBox parsed from dict: %s", box)
            return box

        if isinstance(data, (list, tuple)) and len(data) == 4:
            box = cls(
                x=int(data[0]),
                y=int(data[1]),
                width=int(data[2]),
                height=int(data[3]),
            )
            logger.debug("BoundingBox parsed from list/tuple: %s", box)
            return box

        logger.warning("Unsupported bbox payload shape=%s value=%r", type(data).__name__, data)
        raise ValueError("bounding box payload must be a dict or a 4-item list/tuple")


@dataclass
class ActionPlan:
    """The structured plan produced by the reasoning layer.

    ``params`` is a catch-all for action-specific arguments (e.g. ``key`` for
    :attr:`ActionType.KEY_PRESS`, ``scroll_clicks`` for
    :attr:`ActionType.SCROLL`).
    """

    action: ActionType
    bbox: Optional[BoundingBox] = None
    text: Optional[str] = None
    confidence: Optional[float] = None
    description: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionPlan":
        """Parse a raw plan dict (e.g. tool-call arguments) into a plan.

        Older or mis-specified adapters sometimes surface the function name
        (for example ``plan_action``) in the ``action`` slot. That is not a
        valid UI action, so we normalize it down to a safe default instead of
        crashing the pipeline.
        """
        logger.debug("ActionPlan.from_dict received: %r", data)
        bbox = None
        raw_bbox = data.get("bbox")
        if raw_bbox:
            try:
                bbox = BoundingBox.from_dict(raw_bbox)
                logger.debug("BBox converted to BoundingBox model: %s", bbox)
            except (TypeError, KeyError, ValueError) as exc:
                logger.warning("Could not convert bbox payload %r; dropping bbox safely: %s", raw_bbox, exc)
                bbox = None
        else:
            logger.debug("ActionPlan.from_dict saw no bbox; leaving bbox=None")

        known = {"action", "bbox", "text", "confidence", "description", "params"}
        params = {k: v for k, v in data.items() if k not in known}
        if isinstance(data.get("params"), dict):
            params.update(data["params"])

        raw_action = str(data.get("action", "click")).lower()
        if raw_action == "plan_action":
            logger.warning("Model leaked 'plan_action' into the action field; normalizing to 'click'.")
            raw_action = "click"

        # Convert arbitrary invalid model strings into the safest legal action
        # rather than aborting the whole command loop.
        action = coerce_action(raw_action)
        if action is None:
            logger.warning("Unknown action %r from model; defaulting to click.", raw_action)
            action = ActionType.CLICK

        plan = cls(
            action=action,
            bbox=bbox,
            text=data.get("text"),
            confidence=data.get("confidence"),
            description=str(data.get("description", "")),
            params=params,
        )
        logger.debug("ActionPlan.from_dict produced: %s", plan)
        return plan


@dataclass
class Skill:
    """A compiled reflex stored in the local cache.

    ``template_path`` points at a cropped PNG of the target UI element;
    ``metadata`` records the expected state (bounding box, screen size, text)
    so the reflex can be replayed and audited later.
    """

    name: str
    template_path: str
    action: ActionType = ActionType.CLICK
    metadata: dict[str, Any] = field(default_factory=dict)
    success_count: int = 0
    #: Replays that failed to anchor. Feeds the "retire an unusable reflex"
    #: rule: a stored reflex that keeps missing is worse than no reflex at all,
    #: because every miss costs a wasted capture before the planner is asked.
    failure_count: int = 0
    #: Times the LLM re-anchored this reflex on the live screen.
    realign_count: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_used: Optional[str] = None
    #: New reflexes start **disabled**. A freshly compiled crop has never been
    #: replayed against a live screen, so it must be opted in from the GUI's
    #: Reflexes tab before it may take over from the planner.
    enabled: bool = False

    def record_success(self) -> None:
        """Bump usage counters after a reflex fires successfully."""
        self.success_count += 1
        self.last_used = datetime.now(timezone.utc).isoformat()

    def record_failure(self) -> None:
        """Count a replay that failed to anchor its target."""
        self.failure_count += 1
        self.last_used = datetime.now(timezone.utc).isoformat()

    def record_realign(self, template_path: Optional[str] = None) -> None:
        """Count a successful LLM re-anchoring of this reflex."""
        self.realign_count += 1
        self.failure_count = 0  # the stale anchor has been replaced
        if template_path:
            self.template_path = template_path
        self.metadata["realigned_at"] = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        data = asdict(self)
        data["action"] = self.action.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Skill":
        """Rehydrate a skill from a JSON-safe dictionary."""
        return cls(
            name=data["name"],
            template_path=data["template_path"],
            action=ActionType(str(data.get("action", "click")).lower()),
            metadata=data.get("metadata", {}),
            success_count=int(data.get("success_count", 0)),
            failure_count=int(data.get("failure_count", 0)),
            realign_count=int(data.get("realign_count", 0)),
            created_at=data.get("created_at", ""),
            last_used=data.get("last_used"),
            enabled=bool(data.get("enabled", False)),
        )
