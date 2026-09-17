"""OS window-management helpers for Windows (pure ``ctypes``, no new deps).

The automation pipeline drives the mouse and keyboard through pyautogui, but
pyautogui sends input to **whatever window currently has the foreground focus**.
When Furti's own status window is always-on-top, keyboard and scroll actions
can end up going to the status window instead of the application the user asked
the agent to drive. These helpers let the executor:

* find a target window by (substring) title,
* read the current foreground window title, and
* force a window into the foreground so the next keyboard/scroll action lands
  in the right application.

They also expose :func:`set_process_dpi_aware`, which pins the process to a
single DPI-awareness mode *before* pyautogui/Pillow is imported. This keeps
``pyautogui.screenshot()`` (physical pixels), ``pyautogui.size()`` (mouse
coordinates) and ``pyautogui.moveTo`` all speaking the same coordinate system
on scaled Windows displays -- the root cause of "the cursor lands somewhere
other than where the model aimed".

Everything degrades to a no-op on non-Windows platforms so the rest of the
codebase can import this module unconditionally.
"""

from __future__ import annotations

import ctypes
import logging
import sys
from ctypes import wintypes
from typing import Optional

logger = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform == "win32"

# Window titles that belong to Furti itself. These must never be picked as a
# focus target, otherwise the agent would "focus" its own status window and
# keep typing into itself. Both Furti windows ("Furti AI - Desktop
# Automation" and "Furti AI — live status") start with "Furti AI", so we key
# on that prefix rather than the bare word "furti" -- which would wrongly
# exclude the user's own windows (e.g. an editor or Explorer whose title
# contains the workspace folder name "furti").
_FURTI_TITLE_PREFIX = "furti ai"


def _is_furti_title(title: str) -> bool:
    lowered = title.lower()
    return lowered.startswith(_FURTI_TITLE_PREFIX) or "live status" in lowered


def is_furti_window(title: str | None) -> bool:
    """True when ``title`` belongs to one of Furti's own windows.

    Public counterpart of the title test above: the executor needs it to tell
    "my own always-on-top readout is covering the target" apart from "another
    application is in front".
    """
    return _is_furti_title(str(title or ""))

# show window commands
_SW_RESTORE = 9
#: ShowWindow command that collapses a window to the taskbar.
_SW_MINIMIZE = 6
#: ShowWindow command that expands a window to the whole screen.
_SW_MAXIMIZE = 3
#: Posted to a window to ask it to close (the same message the X button sends).
_WM_CLOSE = 0x0010

# keybd_event constants for the SetForegroundWindow ALT trick.
_VK_MENU = 0x12
_KEYEVENTF_KEYUP = 0x0002

# AllowSetForegroundWindow(ASFW_ANY): let any process take the foreground.
_ASFW_ANY = 0xFFFFFFFF

#: GetAncestor flag: walk up to the top-level (root) owner of a window.
_GA_ROOT = 2

# GetSystemMetrics indices for the virtual desktop rectangle (all monitors).
_SM_XVIRTUALSCREEN = 76
_SM_YVIRTUALSCREEN = 77
_SM_CXVIRTUALSCREEN = 78
_SM_CYVIRTUALSCREEN = 79


def set_process_dpi_aware() -> bool:
    """Pin the process to a DPI-aware mode.

    Must be called before the first ``import pyautogui`` (and ideally before
    any Tk window is created). Once set, the mode cannot be changed, so this is
    intentionally a no-op if awareness is already configured.
    """
    if not _IS_WINDOWS:
        return False
    user32 = ctypes.windll.user32
    try:
        # System-aware is what pyautogui/Pillow and Tk both expect and gives a
        # single consistent physical-pixel coordinate space on scaled displays.
        set_dpi = user32.SetProcessDPIAware
        set_dpi.argtypes = []
        set_dpi.restype = wintypes.BOOL
        if set_dpi():
            return True
    except (AttributeError, OSError):
        pass
    try:
        set_awareness = user32.SetProcessDpiAwareness
        set_awareness.argtypes = [ctypes.c_int]
        set_awareness.restype = ctypes.c_int
        # PROCESS_PER_MONITOR_DPI_AWARE == 2
        if set_awareness(2) == 0:
            return True
    except (AttributeError, OSError):
        pass
    try:
        set_context = user32.SetProcessDpiAwarenessContext
        set_context.argtypes = [ctypes.c_void_p]
        set_context.restype = wintypes.BOOL
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 == -4
        if set_context(ctypes.c_void_p(-4)):
            return True
    except (AttributeError, OSError):
        pass
    return False


def _window_title(hwnd: int) -> str:
    """Best-effort window title for ``hwnd``, or ``""`` when it has none."""
    if not _IS_WINDOWS or not hwnd:
        return ""
    user32 = ctypes.windll.user32
    length = user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, length + 1)
    return buffer.value


def _is_visible(hwnd: int) -> bool:
    if not _IS_WINDOWS or not hwnd:
        return False
    return bool(ctypes.windll.user32.IsWindowVisible(hwnd))


def list_windows() -> list[tuple[int, str]]:
    """Return ``(hwnd, title)`` for every visible top-level window with a title."""
    if not _IS_WINDOWS:
        return []
    user32 = ctypes.windll.user32
    windows: list[tuple[int, str]] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _enum(hwnd: int, _lparam: int) -> bool:
        if _is_visible(hwnd):
            title = _window_title(hwnd)
            if title:
                windows.append((hwnd, title))
        return True

    user32.EnumWindows(_enum, 0)
    return windows


def find_window(title: str, substring: bool = True) -> Optional[int]:
    """Find a visible window by title; ``None`` when nothing matches.

    Matching is case-insensitive. Furti's own windows are always excluded so
    the agent cannot target its status window.
    """
    needle = str(title or "").strip().lower()
    if not needle:
        return None
    for hwnd, candidate in list_windows():
        if _is_furti_title(candidate):
            continue
        lowered = candidate.lower()
        if (substring and needle in lowered) or lowered == needle:
            return hwnd
    return None


def get_foreground_window_title() -> str:
    """Title of the window that currently owns the keyboard focus."""
    if not _IS_WINDOWS:
        return ""
    hwnd = int(ctypes.windll.user32.GetForegroundWindow())
    return _window_title(hwnd)


def window_rect(hwnd: int) -> Optional[tuple[int, int, int, int]]:
    """``(left, top, right, bottom)`` of a window, or ``None``."""
    if not _IS_WINDOWS or not hwnd:
        return None
    rect = wintypes.RECT()
    if not ctypes.windll.user32.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(rect)):
        return None
    return (int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))


def virtual_screen_rect() -> Optional[tuple[int, int, int, int]]:
    """``(left, top, width, height)`` of the whole virtual desktop.

    The executor uses this to refuse a resolved point that is not on any
    monitor: a mis-scaled or hallucinated pixel would otherwise throw the
    cursor off-screen, where every later click lands somewhere unintended. The
    virtual rectangle (not the primary monitor size) is what makes the check
    safe on multi-monitor layouts, where a second display legitimately sits at
    negative coordinates.
    """
    if not _IS_WINDOWS:
        return None
    user32 = ctypes.windll.user32
    try:
        left = int(user32.GetSystemMetrics(_SM_XVIRTUALSCREEN))
        top = int(user32.GetSystemMetrics(_SM_YVIRTUALSCREEN))
        width = int(user32.GetSystemMetrics(_SM_CXVIRTUALSCREEN))
        height = int(user32.GetSystemMetrics(_SM_CYVIRTUALSCREEN))
    except (AttributeError, OSError):
        return None
    if width <= 0 or height <= 0:
        return None
    return left, top, width, height


def window_at(x: int, y: int) -> Optional[tuple[int, str]]:
    """Top-level ``(hwnd, title)`` of the window under a screen point.

    This is what tells an obstruction apart from a miss: once a target has been
    resolved to a pixel, the executor asks which window actually owns that
    pixel, so a modal dialog or Furti's own always-on-top readout can be
    distinguished from "the element moved". ``None`` off Windows, over the
    desktop, or when the point belongs to no window.
    """
    if not _IS_WINDOWS:
        return None
    user32 = ctypes.windll.user32
    hwnd = int(user32.WindowFromPoint(wintypes.POINT(int(x), int(y))))
    if not hwnd:
        return None
    # WindowFromPoint answers with the deepest child; the popup/overlay decision
    # has to be made on the top-level window that owns it.
    root = int(user32.GetAncestor(hwnd, _GA_ROOT))
    if root:
        hwnd = root
    return hwnd, _window_title(hwnd)


def focus_window(title_or_hwnd: str | int) -> bool:
    """Bring a window to the foreground and give it keyboard focus.

    Accepts either a window title (resolved via :func:`find_window`) or a raw
    window handle. Returns ``True`` when the OS reported the window was moved
    to the foreground.

    Windows restricts ``SetForegroundWindow`` to the process that currently
    owns the foreground (or one that just received input). The ALT
    ``keybd_event`` trick and ``AttachThreadInput`` relax that restriction so
    the agent can hand focus back to a target application even when Furti's
    status window is on top.
    """
    if not _IS_WINDOWS:
        return False

    hwnd: int
    if isinstance(title_or_hwnd, str):
        found = find_window(title_or_hwnd)
        if found is None:
            logger.warning("focus_window: no window matched %r", title_or_hwnd)
            return False
        hwnd = found
    else:
        hwnd = int(title_or_hwnd)
    if not hwnd:
        return False

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    # A minimized window must be restored before it can take the foreground.
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, _SW_RESTORE)

    # Let any process hand over the foreground (ASFW_ANY).
    try:
        user32.AllowSetForegroundWindow(_ASFW_ANY)
    except (AttributeError, OSError):
        pass

    # Attach the calling thread's input queue to the target window's thread so
    # SetForegroundWindow/SetFocus are permitted across threads.
    target_pid = wintypes.DWORD(0)
    target_thread = user32.GetWindowThreadProcessId(hwnd, ctypes.byref(target_pid))
    current_thread = kernel32.GetCurrentThreadId()
    attached = False
    if target_thread and target_thread != current_thread:
        attached = bool(user32.AttachThreadInput(current_thread, target_thread, True))

    try:
        user32.BringWindowToTop(hwnd)
        foreground = bool(user32.SetForegroundWindow(hwnd))
        user32.SetActiveWindow(hwnd)
        user32.SetFocus(hwnd)

        if not foreground:
            # The classic workaround: a synthetic ALT press/release grants the
            # calling process permission to change the foreground window.
            user32.keybd_event(_VK_MENU, 0, 0, 0)
            user32.keybd_event(_VK_MENU, 0, _KEYEVENTF_KEYUP, 0)
            foreground = bool(user32.SetForegroundWindow(hwnd))
        return foreground
    finally:
        if attached:
            user32.AttachThreadInput(current_thread, target_thread, False)


# --------------------------------------------------------------- window state
#: Titles that mean "whatever the user is looking at right now".
_ACTIVE_TITLES = frozenset({"active", "current", "foreground", "focused", "top"})


def _foreground_hwnd() -> int:
    """Handle of the window that currently owns the foreground."""
    if not _IS_WINDOWS:
        return 0
    return int(ctypes.windll.user32.GetForegroundWindow())


def resolve_hwnd(title_or_hwnd: str | int | None) -> Optional[int]:
    """Resolve a title / handle / "active" marker into a window handle.

    ``None``, an empty string and the words ``active``/``current``/
    ``foreground``/``focused`` all mean the foreground window, so a plan can
    say "close this window" without knowing its title.
    """
    if title_or_hwnd is None:
        return _foreground_hwnd() or None
    if isinstance(title_or_hwnd, str):
        needle = title_or_hwnd.strip()
        if not needle or needle.lower() in _ACTIVE_TITLES:
            return _foreground_hwnd() or None
        return find_window(needle)
    return int(title_or_hwnd) or None


def _show_window(hwnd: int, command: int) -> bool:
    """Apply a ``ShowWindow`` command, reporting whether it succeeded."""
    if not _IS_WINDOWS or not hwnd:
        return False
    user32 = ctypes.windll.user32
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.ShowWindow.restype = wintypes.BOOL
    return bool(user32.ShowWindow(hwnd, command))


def minimize_window(title_or_hwnd: str | int | None = None) -> bool:
    """Collapse a window to the taskbar (foreground window when unspecified)."""
    hwnd = resolve_hwnd(title_or_hwnd)
    if hwnd is None:
        logger.warning("minimize_window: no window matched %r", title_or_hwnd)
        return False
    return _show_window(hwnd, _SW_MINIMIZE)


def maximize_window(title_or_hwnd: str | int | None = None) -> bool:
    """Expand a window to fill the screen (foreground window when unspecified)."""
    hwnd = resolve_hwnd(title_or_hwnd)
    if hwnd is None:
        logger.warning("maximize_window: no window matched %r", title_or_hwnd)
        return False
    return _show_window(hwnd, _SW_MAXIMIZE)


def restore_window(title_or_hwnd: str | int | None = None) -> bool:
    """Undo a minimize/maximize, returning the window to its normal size."""
    hwnd = resolve_hwnd(title_or_hwnd)
    if hwnd is None:
        logger.warning("restore_window: no window matched %r", title_or_hwnd)
        return False
    return _show_window(hwnd, _SW_RESTORE)


def close_window(title_or_hwnd: str | int | None = None) -> bool:
    """Ask a window to close, like clicking its X button.

    ``WM_CLOSE`` is deliberate rather than ``TerminateProcess``: the
    application still gets to prompt about unsaved work, which is what a user
    clicking the X would expect.
    """
    hwnd = resolve_hwnd(title_or_hwnd)
    if hwnd is None:
        logger.warning("close_window: no window matched %r", title_or_hwnd)
        return False
    user32 = ctypes.windll.user32
    user32.PostMessageW.argtypes = [wintypes.HWND, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
    user32.PostMessageW.restype = wintypes.BOOL
    return bool(user32.PostMessageW(hwnd, _WM_CLOSE, 0, 0))


__all__ = [
    "close_window",
    "find_window",
    "focus_window",
    "get_foreground_window_title",
    "list_windows",
    "maximize_window",
    "minimize_window",
    "restore_window",
    "set_process_dpi_aware",
]
