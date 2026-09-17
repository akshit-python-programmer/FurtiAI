"""Direct (non-GUI) tools the planner can call to finish a step instantly.

Most desktop tasks can be completed by asking the operating system to do the
work instead of replaying a human's mouse and keyboard gestures. Starting
Notepad through :func:`DirectToolRunner._launch_app` is a single process spawn;
doing it with the GUI costs a ``win+r`` chord, a window wait, an OCR round trip
and a model review -- seconds and tokens for something the OS does in
milliseconds.

Every tool returns a :class:`ToolResult`. ``output`` carries the text the
journal should surface (command stdout, clipboard contents, window list, file
excerpt) so the model and the user can see what actually happened.

Safety: :func:`_destructive_reason` refuses irreversible commands (disk format,
recursive force-delete, ``DROP TABLE``, ...) unless the plan explicitly opts in
with ``params.confirm: true`` or the settings allow it. A plan preview is not
consent for destroying data.
"""

from __future__ import annotations

import ctypes
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from html.parser import HTMLParser
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from . import windows as windows_module
from .context import TaskAborted
from .models import ActionType

logger = logging.getLogger(__name__)

__all__ = [
    "DIRECT_ACTIONS",
    "DirectToolRunner",
    "ToolError",
    "ToolResult",
    "describe_tool_step",
    "is_direct_tool",
]


class ToolError(Exception):
    """Raised when a tool is asked for something it cannot do."""


@dataclass(frozen=True)
class ToolResult:
    """Outcome of one direct tool call."""

    tool: str
    ok: bool
    #: One-line explanation of what happened (or why it failed).
    detail: str = ""
    #: Text worth showing the model/user (command output, file excerpt, ...).
    output: str = ""
    #: Machine-readable extras (e.g. the window list).
    data: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Compact one-line rendering for the journal and step notes."""
        parts = [f"{self.tool}: {'ok' if self.ok else 'failed'}"]
        if self.detail:
            parts.append(self.detail)
        if self.output:
            parts.append(self.output)
        return " | ".join(part for part in parts if part)


# ---------------------------------------------------------------------
# Friendly application names -> launchable executables.
# ---------------------------------------------------------------------
_APP_ALIASES: dict[str, str] = {
    "notepad": "notepad.exe",
    "text editor": "notepad.exe",
    "wordpad": "write.exe",
    "calculator": "calc.exe",
    "calc": "calc.exe",
    "paint": "mspaint.exe",
    "mspaint": "mspaint.exe",
    "explorer": "explorer.exe",
    "file explorer": "explorer.exe",
    "files": "explorer.exe",
    "cmd": "cmd.exe",
    "command prompt": "cmd.exe",
    "command line": "cmd.exe",
    "powershell": "powershell.exe",
    "windows powershell": "powershell.exe",
    "pwsh": "pwsh.exe",
    "terminal": "wt.exe",
    "windows terminal": "wt.exe",
    "task manager": "taskmgr.exe",
    "control panel": "control.exe",
    "settings": "ms-settings:",
    "windows settings": "ms-settings:",
    "snipping tool": "snippingtool.exe",
    "snip": "snippingtool.exe",
    "chrome": "chrome.exe",
    "google chrome": "chrome.exe",
    "edge": "msedge.exe",
    "microsoft edge": "msedge.exe",
    "firefox": "firefox.exe",
    "brave": "brave.exe",
    "word": "winword.exe",
    "microsoft word": "winword.exe",
    "excel": "excel.exe",
    "microsoft excel": "excel.exe",
    "powerpoint": "powerpnt.exe",
    "outlook": "outlook.exe",
    "vs code": "code.cmd",
    "vscode": "code.cmd",
    "visual studio code": "code.cmd",
    "code": "code.cmd",
    "spotify": "Spotify.exe",
    "discord": "Discord.exe",
    "slack": "slack.exe",
    "zoom": "Zoom.exe",
    "steam": "steam.exe",
    "vlc": "vlc.exe",
    "photos": "ms-photos:",
    "camera": "microsoft.windows.camera:",
}

#: Commands that destroy data irreversibly. Any match makes the tool refuse
#: unless the plan says ``params.confirm: true`` or the settings opt in.
_DESTRUCTIVE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bformat\s+[a-z]:", "disk format"),
    (r"\bdiskpart\b", "disk partitioning"),
    (r"\bcipher\s+/w", "secure disk wipe"),
    (r"\bvssadmin\s+delete", "shadow-copy deletion"),
    (r"\bbcdedit\b", "boot configuration edit"),
    (r"\bmkfs(\.\w+)?\b", "filesystem creation"),
    (r"\bdd\s+if=", "raw disk write"),
    (r"\brm\s+-[a-z]*r[a-z]*f|\brm\s+-[a-z]*f[a-z]*r", "recursive force delete"),
    (r"\brm\s+-rf\s+/(?!\S)", "root-directory delete"),
    (r"\bdel\s+/[fsq]\b", "force delete"),
    (r"\berase\s+/[fsq]\b", "force delete"),
    (r"\brmdir\s+/s\b", "recursive directory delete"),
    (r"remove-item\b[^\n]*-recurse\b[^\n]*-force", "recursive force delete"),
    (r"\breg\s+delete\s+hklm", "machine-wide registry delete"),
    (r"\bdrop\s+(database|table|schema)\b", "SQL drop"),
    (r"\btruncate\s+table\b", "SQL truncate"),
    (r"\bgsutil\s+rm\b|\bgcloud\s+storage\s+rm\b", "bucket object delete"),
    (r"\bgit\s+push\b[^\n]*--force", "force push"),
    (r"\bgit\s+reset\s+--hard\b", "discarding uncommitted work"),
)


def is_direct_tool(action: Any) -> bool:
    """True when ``action`` completes without touching the mouse or keyboard."""
    try:
        return ActionType(action) in DIRECT_ACTIONS
    except (ValueError, TypeError):
        return False


#: Every action handled by :class:`DirectToolRunner`.
DIRECT_ACTIONS: frozenset[ActionType] = frozenset(
    {
        ActionType.LAUNCH_APP,
        ActionType.OPEN_PATH,
        ActionType.RUN_COMMAND,
        ActionType.WRITE_FILE,
        ActionType.READ_FILE,
        ActionType.SET_CLIPBOARD,
        ActionType.GET_CLIPBOARD,
        ActionType.FOCUS_WINDOW,
        ActionType.LIST_WINDOWS,
        ActionType.CLOSE_WINDOW,
        ActionType.MINIMIZE_WINDOW,
        ActionType.MAXIMIZE_WINDOW,
        ActionType.WAIT,
        ActionType.SCREENSHOT,
        ActionType.CREATE_FOLDER,
        ActionType.LIST_DIR,
        ActionType.COPY_PATH,
        ActionType.MOVE_PATH,
        ActionType.DELETE_PATH,
        ActionType.FIND_FILES,
        ActionType.PATH_INFO,
        ActionType.SCRAPE_URL,
    }
)

#: Parameter names each tool reads its primary argument from, in priority
#: order. Used for plan previews and for the "missing argument" error text.
_PRIMARY_PARAMS: dict[ActionType, tuple[str, ...]] = {
    ActionType.LAUNCH_APP: ("app", "application", "program", "name"),
    ActionType.OPEN_PATH: ("path", "target", "uri", "url", "file"),
    ActionType.RUN_COMMAND: ("command", "cmd", "shell_command", "script"),
    ActionType.WRITE_FILE: ("path", "file", "filename"),
    ActionType.READ_FILE: ("path", "file", "filename"),
    ActionType.SET_CLIPBOARD: ("content", "text", "value"),
    ActionType.FOCUS_WINDOW: ("window", "title", "app"),
    ActionType.CLOSE_WINDOW: ("window", "title", "app"),
    ActionType.MINIMIZE_WINDOW: ("window", "title", "app"),
    ActionType.MAXIMIZE_WINDOW: ("window", "title", "app"),
    ActionType.WAIT: ("seconds", "duration", "wait"),
    ActionType.SCREENSHOT: ("region", "path", "filename", "label"),
    ActionType.CREATE_FOLDER: ("path", "folder", "directory", "name"),
    ActionType.LIST_DIR: ("path", "folder", "directory"),
    ActionType.COPY_PATH: ("source", "from", "src", "path"),
    ActionType.MOVE_PATH: ("source", "from", "src", "path"),
    ActionType.DELETE_PATH: ("path", "target", "file", "folder"),
    ActionType.FIND_FILES: ("root", "path", "folder", "directory"),
    ActionType.PATH_INFO: ("path", "target", "file", "folder"),
    ActionType.SCRAPE_URL: ("url", "uri", "target", "path"),
}


def _clean(value: Any) -> str:
    """Trim a model-supplied scalar into a usable string."""
    if value is None or isinstance(value, bool):
        return ""
    return str(value).strip()


def _truthy(value: Any) -> bool:
    """Lenient boolean read for flags a model writes as text."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y", "full"}


def _as_number(value: Any) -> Optional[float]:
    """Best-effort numeric read ("300", 300, "300.5") or ``None``."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ints(values: Any) -> Optional[tuple[int, int, int, int]]:
    """Convert four model-supplied numbers to ints, or ``None`` if unusable."""
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        return None
    numbers = [_as_number(item) for item in values]
    if any(number is None for number in numbers):
        return None
    return tuple(int(round(number)) for number in numbers)  # type: ignore[return-value]


def _region_payload(params: dict[str, Any]) -> Optional[tuple[int, int, int, int]]:
    """Read a screen region ``(x, y, width, height)`` from a step payload.

    Accepts a nested object under ``region``/``bbox``/``crop`` (dict with
    ``x``/``y``/``width``/``height``, corner keys, or a four-item list) and the
    flat ``x``/``y``/``width``/``height`` spellings. ``None`` means "the whole
    screen".
    """
    for key in ("region", "bbox", "crop", "area", "rectangle"):
        value = params.get(key)
        if isinstance(value, dict):
            if all(k in value for k in ("x", "y", "width", "height")):
                numbers = _ints(
                    [value["x"], value["y"], value["width"], value["height"]]
                )
                if numbers:
                    return numbers
            if all(k in value for k in ("left", "top", "right", "bottom")):
                corners = _ints(
                    [value["left"], value["top"], value["right"], value["bottom"]]
                )
                if corners:
                    left, top, right, bottom = corners
                    return left, top, right - left, bottom - top
        else:
            numbers = _ints(value)
            if numbers:
                return numbers

    flat = _ints(
        [
            params.get("x"),
            params.get("y"),
            params.get("width"),
            params.get("height"),
        ]
    )
    return flat


def _resolve_path(raw: str, settings: Any) -> Path:
    """Turn a model-supplied path into a concrete one.

    ``~`` and environment variables are expanded (so ``%USERPROFILE%\\notes``
    and ``~/Desktop`` both work). A relative path is resolved against the
    agent's workspace rather than the process CWD, so the same plan means the
    same file no matter where Furti was started from.
    """
    text = os.path.expandvars(os.path.expanduser(str(raw or "").strip().strip('"')))
    path = Path(text)
    if not path.is_absolute():
        base = Path(getattr(settings, "workspace", Path.cwd()))
        path = base / path
    return path


class _ReadableHTMLParser(HTMLParser):
    """Collect visible page text while skipping scripts and styling markup."""

    _IGNORED_TAGS = frozenset({"script", "style", "noscript", "template", "svg"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._ignored_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.lower()
        if lowered in self._IGNORED_TAGS:
            self._ignored_depth += 1
        elif lowered == "title" and not self._ignored_depth:
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in self._IGNORED_TAGS and self._ignored_depth:
            self._ignored_depth -= 1
        elif lowered == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        cleaned = " ".join(data.split())
        if not cleaned:
            return
        if self._in_title:
            self.title = f"{self.title} {cleaned}".strip()
        self.parts.append(cleaned)

    def text(self) -> str:
        return "\n".join(self.parts)


def _protected_roots(settings: Any) -> set[Path]:
    """Paths the agent must never delete, even with confirmation."""
    protected: set[Path] = set()
    for candidate in (
        getattr(settings, "workspace", None),
        getattr(settings, "reports_dir", None),
        getattr(settings, "templates_dir", None),
        getattr(settings, "memory_file", None),
    ):
        if candidate is None:
            continue
        try:
            protected.add(Path(candidate).resolve())
        except OSError:
            continue
    return protected


def _sanitize_label(value: str) -> str:
    """Make a model-supplied label safe for a file name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "")).strip("-._")[:40]


def _screenshot_path(settings: Any, params: dict[str, Any]) -> Path:
    """Where a screenshot step should write its image.

    An explicit ``params.path`` wins (relative paths resolve against the
    workspace); otherwise the file lands in the workspace's ``screenshots``
    directory, timestamped and optionally labelled.
    """
    allowed = {"png", "jpg", "jpeg", "webp", "bmp"}
    fmt = _clean(
        params.get("format") or getattr(settings, "screenshot_format", "png")
    ).lower().lstrip(".")
    if fmt not in allowed:
        fmt = "png"
    raw = _clean(params.get("path") or params.get("filename") or params.get("file"))
    if raw:
        candidate = _resolve_path(raw, settings)
        if not candidate.suffix:
            candidate = candidate.with_suffix(f".{fmt}")
        return candidate
    label = _sanitize_label(_clean(params.get("label")) or _clean(params.get("name")))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = getattr(settings, "screenshots_dir", None)
    if base is None:
        # Settings-like objects without the derived property still get a sane
        # home: <workspace>/screenshots rather than the process CWD.
        workspace = getattr(settings, "workspace", None)
        base = (Path(workspace) / "screenshots") if workspace else (Path.cwd() / "screenshots")
    return Path(base) / (f"{stamp}_{label}.{fmt}" if label else f"{stamp}.{fmt}")


def _first_param(
    params: dict[str, Any], step: Any, names: tuple[str, ...]
) -> str:
    """Read the first non-empty named parameter, falling back to step text."""
    for name in names:
        value = _clean(params.get(name))
        if value:
            return value
    return _clean(getattr(step, "text", None)) or _clean(
        getattr(step, "target", None)
    )


def _truncate(text: str, limit: int) -> str:
    """Clamp long tool output so a single step cannot flood the report."""
    text = (text or "").strip()
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n[...{len(text) - limit} more characters]"


def _describe_tool_params(step: Any) -> str:
    """Short ``key=value`` summary of a tool step's arguments."""
    params = getattr(step, "params", None) or {}
    action = getattr(step, "action", None)
    names = _PRIMARY_PARAMS.get(action, ())
    value = _first_param(params, step, names) if names else ""
    if action is ActionType.WAIT and not value:
        value = _clean(params.get("seconds"))
    if action is ActionType.WRITE_FILE and value:
        content = params.get("content")
        if content is None:
            content = getattr(step, "text", None)
        return f"{value} ({len(str(content or ''))} chars)"
    return value or "(no arguments)"


def describe_tool_step(step: Any) -> str:
    """Human-readable arguments of a direct tool step, for plan previews."""
    return _describe_tool_params(step)


def _destructive_reason(command: str) -> str:
    """Return a label when ``command`` would irreversibly destroy data."""
    lowered = str(command or "").lower()
    for pattern, label in _DESTRUCTIVE_PATTERNS:
        if re.search(pattern, lowered):
            return label
    return ""


def _resolve_app_command(name: str) -> list[str]:
    """Turn an application name the model wrote into a launchable command.

    Resolution order: known friendly alias, an existing path, ``PATH`` lookup,
    and finally the raw name so the OS shell can apply its own associations.
    """
    raw = str(name or "").strip().strip('"')
    if not raw:
        raise ToolError("launch_app needs an application name")
    key = re.sub(r"\s+", " ", raw.lower())
    alias = _APP_ALIASES.get(key) or _APP_ALIASES.get(key.removesuffix(".exe"))
    if alias:
        return [alias]
    expanded = os.path.expandvars(os.path.expanduser(raw))
    if Path(expanded).exists():
        return [expanded]
    for probe in (expanded, f"{expanded}.exe"):
        found = shutil.which(probe)
        if found:
            return [found]
    return [expanded]


def _looks_like_uri(target: str) -> bool:
    """True when ``target`` is a URI/protocol alias rather than a file path.

    The scheme must be at least two characters so a Windows drive letter
    (``C:\\data``) is never mistaken for a scheme called ``c``. Protocol
    aliases such as ``ms-settings:`` and real URLs still match.
    """
    return re.match(r"^[a-z][a-z0-9+.\-]+:", str(target or ""), re.IGNORECASE) is not None


def _open_externally(target: str) -> bool:
    """Hand ``target`` (path, URI or protocol alias) to the OS opener."""
    if os.name == "nt":
        try:
            os.startfile(target)  # type: ignore[attr-defined]
            return True
        except OSError:
            return False
    if target.startswith(("http://", "https://", "mailto:", "ftp://")):
        return webbrowser.open(target)
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    executable = shutil.which(opener)
    if not executable:
        return False
    subprocess.Popen(
        [executable, target],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return True


# ---------------------------------------------------------------------
# Clipboard (pure ctypes on Windows, so no extra dependency).
# ---------------------------------------------------------------------
_CF_UNICODETEXT = 13
_GMEM_MOVEABLE = 0x0002


def _open_clipboard(user32: Any, attempts: int = 10, delay: float = 0.05) -> bool:
    """Open the clipboard, retrying while another process holds it."""
    for _ in range(attempts):
        if user32.OpenClipboard(None):
            return True
        time.sleep(delay)
    return False


def _bind_clipboard_api() -> tuple[Any, Any]:
    """Declare ctypes signatures once so 64-bit handles are not truncated."""
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    user32.OpenClipboard.argtypes = [ctypes.c_void_p]
    user32.OpenClipboard.restype = ctypes.c_bool
    user32.GetClipboardData.argtypes = [ctypes.c_uint]
    user32.GetClipboardData.restype = ctypes.c_void_p
    user32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]
    user32.SetClipboardData.restype = ctypes.c_void_p
    kernel32.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = ctypes.c_void_p
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalUnlock.restype = ctypes.c_bool
    return user32, kernel32


def _clipboard_write(text: str) -> None:
    """Replace the clipboard contents with ``text``."""
    if os.name != "nt":
        raise ToolError("set_clipboard is only implemented on Windows")
    user32, kernel32 = _bind_clipboard_api()
    buffer = ctypes.create_unicode_buffer(str(text))
    size = ctypes.sizeof(buffer)
    if not _open_clipboard(user32):
        raise ToolError("could not open the clipboard (in use by another process)")
    try:
        user32.EmptyClipboard()
        handle = kernel32.GlobalAlloc(_GMEM_MOVEABLE, size)
        if not handle:
            raise ToolError("clipboard allocation failed")
        locked = kernel32.GlobalLock(handle)
        if not locked:
            raise ToolError("clipboard lock failed")
        try:
            ctypes.memmove(locked, buffer, size)
        finally:
            kernel32.GlobalUnlock(handle)
        if not user32.SetClipboardData(_CF_UNICODETEXT, handle):
            raise ToolError("SetClipboardData failed")
        # Ownership of the memory block moved to the clipboard: intentionally
        # not freed here, or the paste would read released memory.
    finally:
        user32.CloseClipboard()


def _clipboard_read() -> str:
    """Return the clipboard's text (``""`` when it holds no text)."""
    if os.name != "nt":
        raise ToolError("get_clipboard is only implemented on Windows")
    user32, kernel32 = _bind_clipboard_api()
    if not _open_clipboard(user32):
        raise ToolError("could not open the clipboard (in use by another process)")
    try:
        handle = user32.GetClipboardData(_CF_UNICODETEXT)
        if not handle:
            return ""
        locked = kernel32.GlobalLock(handle)
        if not locked:
            return ""
        try:
            return ctypes.c_wchar_p(locked).value or ""
        finally:
            kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()


class DirectToolRunner:
    """Executes the :data:`DIRECT_ACTIONS` vocabulary without screen input."""

    _HANDLERS: dict[ActionType, str] = {
        ActionType.LAUNCH_APP: "_launch_app",
        ActionType.OPEN_PATH: "_open_path",
        ActionType.RUN_COMMAND: "_run_command",
        ActionType.WRITE_FILE: "_write_file",
        ActionType.READ_FILE: "_read_file",
        ActionType.SET_CLIPBOARD: "_set_clipboard",
        ActionType.GET_CLIPBOARD: "_get_clipboard",
        ActionType.FOCUS_WINDOW: "_focus_window",
        ActionType.LIST_WINDOWS: "_list_windows",
        ActionType.CLOSE_WINDOW: "_window_state",
        ActionType.MINIMIZE_WINDOW: "_window_state",
        ActionType.MAXIMIZE_WINDOW: "_window_state",
        ActionType.WAIT: "_wait",
        ActionType.SCREENSHOT: "_screenshot",
        ActionType.CREATE_FOLDER: "_create_folder",
        ActionType.LIST_DIR: "_list_dir",
        ActionType.COPY_PATH: "_copy_path",
        ActionType.MOVE_PATH: "_move_path",
        ActionType.DELETE_PATH: "_delete_path",
        ActionType.FIND_FILES: "_find_files",
        ActionType.PATH_INFO: "_path_info",
        ActionType.SCRAPE_URL: "_scrape_url",
    }

    def __init__(
        self,
        settings: Any,
        journal: Any = None,
        stop_event: Any = None,
        screen: Any = None,
    ) -> None:
        self._settings = settings
        self._journal = journal
        self._stop = stop_event
        #: Screen capture backend for the screenshot tool. Injected by the
        #: orchestrator (and by tests); built lazily on first use otherwise.
        self._screen = screen

    # ------------------------------------------------------------ public API
    def run(self, step: Any) -> ToolResult:
        """Run the tool named by ``step.action`` and report the outcome.

        Failures are returned as ``ToolResult(ok=False)`` so the executor can
        re-plan from a readable reason instead of dying on an exception.
        """
        action = getattr(step, "action", None)
        handler_name = self._HANDLERS.get(action) if isinstance(action, ActionType) else None
        if handler_name is None:
            return ToolResult(
                tool=str(getattr(action, "value", action)),
                ok=False,
                detail=f"not a direct tool: {getattr(action, 'value', action)}",
            )
        handler: Callable[[Any], ToolResult] = getattr(self, handler_name)
        try:
            return handler(step)
        except TaskAborted:
            # The kill hotkey outranks the tool: let the run unwind.
            raise
        except ToolError as exc:
            return ToolResult(tool=action.value, ok=False, detail=str(exc))
        except Exception as exc:  # noqa: BLE001 - report, never crash the run
            logger.exception("direct tool %s failed", action.value)
            return ToolResult(
                tool=action.value,
                ok=False,
                detail=f"{type(exc).__name__}: {exc}",
            )

    # -------------------------------------------------------------- helpers
    @property
    def _timeout(self) -> float:
        return max(1.0, float(getattr(self._settings, "tool_timeout", 60.0)))

    @property
    def _max_output(self) -> int:
        return int(getattr(self._settings, "tool_max_output_chars", 4000))

    def _check_stop(self) -> None:
        """Raise :class:`TaskAborted` when the kill hotkey was pressed."""
        stop = self._stop
        if stop is not None and stop.is_set():
            raise TaskAborted("task aborted by the user")

    def _settle(self, seconds: float) -> None:
        """Sleep in slices so the kill hotkey stays responsive."""
        deadline = time.perf_counter() + max(0.0, float(seconds))
        while True:
            self._check_stop()
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return
            time.sleep(min(0.1, remaining))

    def _journal_note(self, message: str) -> None:
        action = getattr(self._journal, "action", None)
        if callable(action):
            action(message)

    # ---------------------------------------------------------------- tools
    def _launch_app(self, step: Any) -> ToolResult:
        params = getattr(step, "params", None) or {}
        name = _first_param(params, step, _PRIMARY_PARAMS[ActionType.LAUNCH_APP])
        if not name:
            raise ToolError(
                'launch_app needs params.app (for example "notepad")'
            )
        command = _resolve_app_command(name)
        extra = params.get("args")
        args: list[str] = []
        if isinstance(extra, (list, tuple)):
            args = [str(item) for item in extra]
        elif _clean(extra):
            args = [_clean(extra)]
        cwd = _clean(params.get("cwd")) or None

        executable = command[0]
        on_path = shutil.which(executable) is not None
        if not on_path and not Path(executable).exists():
            # Not a real executable: let the shell resolve protocols and
            # Store-app aliases ("ms-settings:", "wt").
            if not _open_externally(executable):
                raise ToolError(f"could not find an application named {name!r}")
            self._settle(float(params.get("settle", 0.6) or 0))
            return ToolResult(
                tool=ActionType.LAUNCH_APP.value,
                ok=True,
                detail=f"launched {name!r} through the shell",
            )

        popen_kwargs: dict[str, Any] = {}
        if cwd:
            popen_kwargs["cwd"] = cwd
        if os.name == "nt":
            popen_kwargs["creationflags"] = getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
        process = subprocess.Popen([executable, *args], **popen_kwargs)
        self._settle(float(params.get("settle", 0.6) or 0))
        return ToolResult(
            tool=ActionType.LAUNCH_APP.value,
            ok=True,
            detail=f"started {executable} (pid {process.pid})",
            data={"pid": process.pid, "command": executable},
        )

    def _open_path(self, step: Any) -> ToolResult:
        params = getattr(step, "params", None) or {}
        target = _first_param(params, step, _PRIMARY_PARAMS[ActionType.OPEN_PATH])
        if not target:
            raise ToolError("open_path needs params.path (a file, folder or URL)")
        expanded = target
        if not _looks_like_uri(target):
            expanded = os.path.expandvars(os.path.expanduser(target))
            if not Path(expanded).exists():
                raise ToolError(f"path does not exist: {expanded}")
        if not _open_externally(expanded):
            raise ToolError(f"no handler would open {expanded!r}")
        self._settle(float(params.get("settle", 0.5) or 0))
        return ToolResult(
            tool=ActionType.OPEN_PATH.value,
            ok=True,
            detail=f"opened {expanded}",
            data={"path": expanded},
        )

    def _run_command(self, step: Any) -> ToolResult:
        params = getattr(step, "params", None) or {}
        command = _first_param(params, step, _PRIMARY_PARAMS[ActionType.RUN_COMMAND])
        if not command:
            raise ToolError("run_command needs params.command")
        if not getattr(self._settings, "allow_shell_commands", True):
            return ToolResult(
                tool=ActionType.RUN_COMMAND.value,
                ok=False,
                detail="shell commands are disabled (FURTI_ALLOW_SHELL_COMMANDS=false)",
            )
        reason = _destructive_reason(command)
        confirmed = bool(params.get("confirm")) or bool(
            getattr(self._settings, "allow_destructive_commands", False)
        )
        if reason and not confirmed:
            return ToolResult(
                tool=ActionType.RUN_COMMAND.value,
                ok=False,
                detail=(
                    f"refused a potentially destructive command ({reason}); "
                    "re-issue it with params.confirm=true if this is intended"
                ),
            )
        cwd = _clean(params.get("cwd")) or None
        timeout = float(params.get("timeout") or self._timeout)
        self._journal_note(f"Running command: {command}")

        if params.get("background"):
            popen_kwargs: dict[str, Any] = {}
            if cwd:
                popen_kwargs["cwd"] = cwd
            if os.name == "nt":
                popen_kwargs["creationflags"] = getattr(
                    subprocess, "CREATE_NEW_PROCESS_GROUP", 0
                )
            process = subprocess.Popen(command, shell=True, **popen_kwargs)
            return ToolResult(
                tool=ActionType.RUN_COMMAND.value,
                ok=True,
                detail=f"started in the background (pid {process.pid})",
                data={"pid": process.pid},
            )

        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                tool=ActionType.RUN_COMMAND.value,
                ok=False,
                detail=f"command timed out after {timeout:.0f}s",
            )
        output = "\n".join(
            part.strip()
            for part in (completed.stdout or "", completed.stderr or "")
            if part and part.strip()
        )
        return ToolResult(
            tool=ActionType.RUN_COMMAND.value,
            ok=completed.returncode == 0,
            detail=f"exit code {completed.returncode}",
            output=_truncate(output, self._max_output),
            data={"returncode": completed.returncode},
        )

    def _write_file(self, step: Any) -> ToolResult:
        params = getattr(step, "params", None) or {}
        path = _clean(params.get("path")) or _clean(params.get("file")) or _clean(
            params.get("filename")
        ) or _clean(params.get("target")) or _clean(getattr(step, "target", None))
        if not path:
            raise ToolError("write_file needs params.path")
        content = params.get("content")
        if content is None:
            content = params.get("text")
        if content is None:
            content = getattr(step, "text", None)
        content = "" if content is None else str(content)

        target = Path(os.path.expandvars(os.path.expanduser(path)))
        if target.is_dir():
            raise ToolError(f"{target} is a directory, not a file")
        if target.parent and not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
        append = bool(params.get("append"))
        with target.open("a" if append else "w", encoding="utf-8") as handle:
            handle.write(content)
        mode = "appended to" if append else "wrote"
        return ToolResult(
            tool=ActionType.WRITE_FILE.value,
            ok=True,
            detail=f"{mode} {target} ({len(content)} chars)",
            data={"path": str(target), "bytes": len(content)},
        )

    def _read_file(self, step: Any) -> ToolResult:
        params = getattr(step, "params", None) or {}
        path = _clean(params.get("path")) or _clean(params.get("file")) or _clean(
            params.get("filename")
        ) or _clean(params.get("target")) or _clean(getattr(step, "target", None))
        if not path:
            raise ToolError("read_file needs params.path")
        target = Path(os.path.expandvars(os.path.expanduser(path)))
        if not target.exists():
            raise ToolError(f"file does not exist: {target}")
        if target.is_dir():
            entries = sorted(item.name for item in target.iterdir())
            listing = _truncate("\n".join(entries), self._max_output)
            return ToolResult(
                tool=ActionType.READ_FILE.value,
                ok=True,
                detail=f"listed {target} ({len(entries)} entries)",
                output=listing,
                data={"path": str(target), "entries": entries},
            )
        limit = int(params.get("max_chars") or self._max_output)
        text = target.read_text(encoding=str(params.get("encoding") or "utf-8"), errors="replace")
        return ToolResult(
            tool=ActionType.READ_FILE.value,
            ok=True,
            detail=f"read {target} ({len(text)} chars)",
            output=_truncate(text, limit),
            data={"path": str(target), "chars": len(text)},
        )

    def _scrape_url(self, step: Any) -> ToolResult:
        """Fetch a URL and return bounded readable text without browser UI."""
        params = getattr(step, "params", None) or {}
        url = _first_param(params, step, _PRIMARY_PARAMS[ActionType.SCRAPE_URL])
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ToolError("scrape_url requires an http:// or https:// URL")
        max_chars = _as_number(params.get("max_chars"))
        limit = max(1, int(max_chars or self._max_output))
        max_bytes = min(5_000_000, max(1_024, limit * 12))
        request = Request(
            url,
            headers={"User-Agent": "FurtiAI/1.0 (readable-content-fetcher)"},
        )
        self._check_stop()
        with urlopen(request, timeout=self._timeout) as response:
            content_type = str(response.headers.get("Content-Type", "")).lower()
            body = response.read(max_bytes + 1)
        self._check_stop()
        truncated = len(body) > max_bytes
        body = body[:max_bytes]
        charset = "utf-8"
        match = re.search(r"charset=([\w.-]+)", content_type)
        if match:
            charset = match.group(1)
        decoded = body.decode(charset, errors="replace")
        parser = _ReadableHTMLParser()
        if "html" in content_type or "<html" in decoded.lower():
            parser.feed(decoded)
            text = parser.text()
            title = parser.title
        else:
            text = decoded.strip()
            title = ""
        text = _truncate(text, limit)
        if not text:
            raise ToolError("scrape_url found no readable text")
        detail = f"scraped {url} ({len(text)} chars)"
        if title:
            detail += f" title={title!r}"
        if truncated or len(text) >= limit:
            detail += f"; output limited to {limit} chars"
        return ToolResult(
            tool=ActionType.SCRAPE_URL.value,
            ok=True,
            detail=detail,
            output=text,
            data={"url": url, "title": title, "chars": len(text)},
        )

    def _set_clipboard(self, step: Any) -> ToolResult:
        params = getattr(step, "params", None) or {}
        content = params.get("content")
        if content is None:
            content = params.get("text")
        if content is None:
            content = getattr(step, "text", None)
        if content is None:
            raise ToolError("set_clipboard needs params.content")
        _clipboard_write(str(content))
        return ToolResult(
            tool=ActionType.SET_CLIPBOARD.value,
            ok=True,
            detail=f"clipboard now holds {len(str(content))} chars",
        )

    def _get_clipboard(self, step: Any) -> ToolResult:
        params = getattr(step, "params", None) or {}
        text = _clipboard_read()
        return ToolResult(
            tool=ActionType.GET_CLIPBOARD.value,
            ok=True,
            detail=f"clipboard held {len(text)} chars",
            output=_truncate(text, int(params.get("max_chars") or self._max_output)),
        )

    def _focus_window(self, step: Any) -> ToolResult:
        params = getattr(step, "params", None) or {}
        title = _first_param(params, step, _PRIMARY_PARAMS[ActionType.FOCUS_WINDOW])
        if not title:
            raise ToolError(
                'focus_window needs params.window (for example "Notepad")'
            )
        if not windows_module.focus_window(title):
            raise ToolError(f"no window matched {title!r}")
        self._settle(float(params.get("settle", 0.2) or 0))
        return ToolResult(
            tool=ActionType.FOCUS_WINDOW.value,
            ok=True,
            detail=f"focused {title!r}",
        )

    def _list_windows(self, step: Any) -> ToolResult:
        params = getattr(step, "params", None) or {}
        titles = [title for _hwnd, title in windows_module.list_windows()]
        return ToolResult(
            tool=ActionType.LIST_WINDOWS.value,
            ok=True,
            detail=f"{len(titles)} visible window(s)",
            output=_truncate(
                "\n".join(titles), int(params.get("max_chars") or self._max_output)
            ),
            data={"windows": titles},
        )

    def _window_state(self, step: Any) -> ToolResult:
        params = getattr(step, "params", None) or {}
        action = step.action
        title = _clean(params.get("window")) or _clean(params.get("title")) or _clean(
            params.get("app")
        ) or _clean(getattr(step, "target", None))
        # No title means "the window the user is looking at".
        title_or_hwnd: Any = title or None

        if action is ActionType.CLOSE_WINDOW:
            if not windows_module.close_window(title_or_hwnd):
                raise ToolError(f"could not close {title or 'the active window'}")
            return ToolResult(
                tool=action.value,
                ok=True,
                detail=f"closed {title or 'the active window'}",
            )
        if action is ActionType.MINIMIZE_WINDOW:
            if not windows_module.minimize_window(title_or_hwnd):
                raise ToolError(f"could not minimize {title or 'the active window'}")
            return ToolResult(
                tool=action.value,
                ok=True,
                detail=f"minimized {title or 'the active window'}",
            )
        if not windows_module.maximize_window(title_or_hwnd):
            raise ToolError(f"could not maximize {title or 'the active window'}")
        self._settle(float(params.get("settle", 0.2) or 0))
        return ToolResult(
            tool=action.value,
            ok=True,
            detail=f"maximized {title or 'the active window'}",
        )

    def _wait(self, step: Any) -> ToolResult:
        params = getattr(step, "params", None) or {}
        raw = params.get("seconds")
        if raw is None:
            raw = params.get("duration")
        if raw is None:
            raw = getattr(step, "text", None)
        try:
            seconds = float(str(raw).strip())
        except (TypeError, ValueError):
            raise ToolError("wait needs params.seconds (a number)") from None
        seconds = min(max(0.0, seconds), float(params.get("max_seconds") or 120))
        self._settle(seconds)
        return ToolResult(
            tool=ActionType.WAIT.value,
            ok=True,
            detail=f"waited {seconds:.2f}s",
        )

    # --------------------------------------------------------- screenshots
    def _capture_screen(self) -> Any:
        """Capture the screen (building the default backend on first use)."""
        screen = self._screen
        if screen is None:
            from .screen import PyAutoGuiScreen

            screen = PyAutoGuiScreen()
            self._screen = screen
        frame = screen.capture()
        if frame is None or getattr(frame, "size", 0) == 0:
            raise ToolError("screen capture returned an empty frame")
        return frame

    def _screenshot(self, step: Any) -> ToolResult:
        """Capture the whole screen or one region and save it to disk.

        The saved path is returned as the tool output, so a later step (or the
        user) can open the exact image the model was reasoning about.
        """
        import cv2  # local import: keeps the file tools import-light

        params = getattr(step, "params", None) or {}
        frame = self._capture_screen()
        frame_height, frame_width = int(frame.shape[0]), int(frame.shape[1])

        region = _region_payload(params)
        note = f"full screen {frame_width}x{frame_height}"
        region_data: Optional[dict[str, int]] = None
        if region is not None:
            x, y, w, h = region
            x = min(max(0, x), max(0, frame_width - 1))
            y = min(max(0, y), max(0, frame_height - 1))
            w = min(max(1, w), frame_width - x)
            h = min(max(1, h), frame_height - y)
            frame = frame[y : y + h, x : x + w]
            note = f"region {w}x{h} at ({x}, {y})"
            region_data = {"x": x, "y": y, "width": w, "height": h}

        scale = _as_number(params.get("scale"))
        if scale is not None and 0 < scale < 1.0:
            new_size = (
                max(1, int(round(frame.shape[1] * scale))),
                max(1, int(round(frame.shape[0] * scale))),
            )
            frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
            note += f", scaled x{scale}"

        path = _screenshot_path(self._settings, params)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), frame):
            raise ToolError(f"could not write the screenshot to {path}")
        size = path.stat().st_size
        height, width = int(frame.shape[0]), int(frame.shape[1])
        output = f"screenshot saved: {path}"
        if path.suffix.lower() in {".jpg", ".jpeg"}:
            output += " (JPEG: lossy, use png when detail matters)"
        return ToolResult(
            tool=ActionType.SCREENSHOT.value,
            ok=True,
            detail=f"saved {path.name} ({note}, {size} bytes)",
            output=output,
            data={
                "path": str(path),
                "width": width,
                "height": height,
                "bytes": size,
                "region": region_data,
            },
        )

    # ------------------------------------------------------- file system
    def _create_folder(self, step: Any) -> ToolResult:
        params = getattr(step, "params", None) or {}
        raw = _first_param(params, step, _PRIMARY_PARAMS[ActionType.CREATE_FOLDER])
        if not raw:
            raise ToolError("create_folder needs params.path")
        folder = _resolve_path(raw, self._settings)
        existed = folder.exists()
        if existed and not folder.is_dir():
            raise ToolError(f"{folder} exists and is not a folder")
        folder.mkdir(parents=True, exist_ok=True)
        return ToolResult(
            tool=ActionType.CREATE_FOLDER.value,
            ok=True,
            detail=(
                f"{'already existed' if existed else 'created'} {folder}"
                + ("" if existed else f" (with {len(folder.parents)} parent levels ensured)")
            ),
            data={"path": str(folder), "created": not existed},
        )

    def _list_dir(self, step: Any) -> ToolResult:
        params = getattr(step, "params", None) or {}
        raw = _first_param(params, step, _PRIMARY_PARAMS[ActionType.LIST_DIR])
        folder = _resolve_path(raw, self._settings) if raw else Path.home()
        if not folder.exists():
            raise ToolError(f"folder does not exist: {folder}")
        if not folder.is_dir():
            raise ToolError(f"{folder} is not a folder")

        include_hidden = _truthy(params.get("include_hidden"))
        pattern = _clean(params.get("pattern")) or "*"
        directories: list[str] = []
        files: list[str] = []
        for entry in sorted(folder.glob(pattern), key=lambda item: item.name.lower()):
            if not include_hidden and entry.name.startswith("."):
                continue
            try:
                if entry.is_dir():
                    directories.append(f"[dir]  {entry.name}")
                else:
                    files.append(f"[file] {entry.name} ({entry.stat().st_size:,} bytes)")
            except OSError as exc:  # locked/broken entries must not kill the listing
                files.append(f"[????] {entry.name} ({exc})")
        listing = directories + files
        limit = int(params.get("limit") or 200)
        truncated = len(listing) > limit
        shown = listing[:limit]
        body = "\n".join(shown) or "(empty folder)"
        if truncated:
            body += f"\n[...{len(listing) - limit} more entries]"
        return ToolResult(
            tool=ActionType.LIST_DIR.value,
            ok=True,
            detail=(
                f"{folder} holds {len(directories)} folder(s) and "
                f"{len(files)} file(s)"
            ),
            output=_truncate(f"{folder}:\n{body}", self._max_output),
            data={"path": str(folder), "count": len(listing)},
        )

    def _copy_path(self, step: Any) -> ToolResult:
        return self._transfer_path(step, move=False)

    def _move_path(self, step: Any) -> ToolResult:
        return self._transfer_path(step, move=True)

    def _transfer_path(self, step: Any, *, move: bool) -> ToolResult:
        """Copy or move a file/folder (``shutil`` handles both kinds)."""
        import shutil

        params = getattr(step, "params", None) or {}
        raw_source = _first_param(
            params, step, _PRIMARY_PARAMS[ActionType.COPY_PATH]
        )
        raw_dest = _clean(params.get("destination") or params.get("dest")
                          or params.get("to") or params.get("target"))
        if not raw_source:
            raise ToolError(f"{step.action.value} needs params.source")
        if not raw_dest:
            raise ToolError(f"{step.action.value} needs params.destination")
        source = _resolve_path(raw_source, self._settings)
        destination = _resolve_path(raw_dest, self._settings)
        if not source.exists():
            raise ToolError(f"source does not exist: {source}")
        if source == destination:
            raise ToolError("source and destination are the same path")
        destination.parent.mkdir(parents=True, exist_ok=True)

        if move:
            final = Path(shutil.move(str(source), str(destination)))
            verb = "moved"
        elif source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
            final = destination
            verb = "copied folder"
        else:
            final = Path(shutil.copy2(str(source), str(destination)))
            verb = "copied"
        return ToolResult(
            tool=step.action.value,
            ok=True,
            detail=f"{verb} {source.name} -> {final}",
            output=str(final),
            data={"source": str(source), "destination": str(final)},
        )

    def _delete_path(self, step: Any) -> ToolResult:
        """Delete a file or folder -- cautiously.

        Refuses without ``params.confirm: true`` (or
        ``FURTI_ALLOW_DESTRUCTIVE_COMMANDS``), prefers the Recycle Bin so the
        user can undo it, and never deletes a drive root, the home folder or the
        agent's own workspace.
        """
        import shutil

        params = getattr(step, "params", None) or {}
        raw = _first_param(params, step, _PRIMARY_PARAMS[ActionType.DELETE_PATH])
        if not raw:
            raise ToolError("delete_path needs params.path")
        target = _resolve_path(raw, self._settings)

        protected = {Path.home().resolve()} | _protected_roots(self._settings)
        if not target.exists():
            raise ToolError(f"path does not exist: {target}")
        if target.parent == target:
            raise ToolError("refusing to delete a drive root")
        if target.resolve() in protected:
            raise ToolError(f"refusing to delete the protected path {target}")

        confirmed = bool(params.get("confirm")) or bool(
            getattr(self._settings, "allow_destructive_commands", False)
        )
        if not confirmed:
            return ToolResult(
                tool=ActionType.DELETE_PATH.value,
                ok=False,
                detail=(
                    f"refused to delete {target} without confirmation; "
                    "re-issue the step with params.confirm=true if this is intended"
                ),
            )

        kind = "folder" if target.is_dir() else "file"
        try:
            from send2trash import send2trash  # optional dependency

            send2trash(str(target))
            where = "to the Recycle Bin"
        except ImportError:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            where = "permanently (send2trash is not installed)"
        except OSError as exc:
            raise ToolError(f"could not recycle {target}: {exc}") from exc
        return ToolResult(
            tool=ActionType.DELETE_PATH.value,
            ok=True,
            detail=f"deleted {kind} {target.name} {where}",
            output=f"deleted {target}",
            data={"path": str(target), "kind": kind},
        )

    def _find_files(self, step: Any) -> ToolResult:
        params = getattr(step, "params", None) or {}
        raw_root = _first_param(params, step, _PRIMARY_PARAMS[ActionType.FIND_FILES])
        root = _resolve_path(raw_root, self._settings) if raw_root else Path.home()
        if not root.exists():
            raise ToolError(f"search root does not exist: {root}")
        if not root.is_dir():
            raise ToolError(f"search root is not a folder: {root}")

        pattern = _clean(params.get("pattern") or params.get("name")) or "*"
        limit = int(params.get("limit") or 50)
        max_depth = int(params.get("max_depth") or 6)
        matches: list[str] = []
        for candidate in root.rglob(pattern):
            try:
                depth = len(candidate.relative_to(root).parts)
            except ValueError:
                continue
            if depth > max_depth:
                continue
            if not _truthy(params.get("include_hidden")) and any(
                part.startswith(".") for part in candidate.relative_to(root).parts
            ):
                continue
            matches.append(str(candidate))
            if len(matches) >= limit:
                break
        return ToolResult(
            tool=ActionType.FIND_FILES.value,
            ok=True,
            detail=f"{len(matches)} match(es) for {pattern!r} under {root}",
            output=_truncate("\n".join(matches) or "(no matches)", self._max_output),
            data={"root": str(root), "pattern": pattern, "matches": matches},
        )

    def _path_info(self, step: Any) -> ToolResult:
        params = getattr(step, "params", None) or {}
        raw = _first_param(params, step, _PRIMARY_PARAMS[ActionType.PATH_INFO])
        if not raw:
            raise ToolError("path_info needs params.path")
        target = _resolve_path(raw, self._settings)
        if not target.exists():
            return ToolResult(
                tool=ActionType.PATH_INFO.value,
                ok=True,
                detail=f"{target} does not exist",
                output=f"exists: false\npath: {target}",
                data={"path": str(target), "exists": False},
            )
        stat = target.stat()
        facts = {
            "path": str(target),
            "exists": True,
            "kind": "folder" if target.is_dir() else "file",
            "size_bytes": 0 if target.is_dir() else stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            "parent": str(target.parent),
        }
        if target.is_dir():
            try:
                facts["children"] = len(list(target.iterdir()))
            except OSError:
                facts["children"] = None
        body = "\n".join(f"{key}: {value}" for key, value in facts.items())
        return ToolResult(
            tool=ActionType.PATH_INFO.value,
            ok=True,
            detail=f"{target} is a {facts['kind']}",
            output=body,
            data=facts,
        )
