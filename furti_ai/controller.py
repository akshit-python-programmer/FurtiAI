"""Virtual input abstraction (mouse / keyboard).

Input is wrapped behind :class:`InputController` so the execution engine is
agnostic to the concrete automation library. The default implementation uses
``pyautogui``; swapping in ``pynput`` only means writing one new class.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional, Protocol, Sequence

from .keyboard import ChordError, KeyboardController, describe_chord

__all__ = ["ChordError", "InputController", "PyAutoGuiInput"]


class InputController(Protocol):
    """Structural interface for the input device."""

    def move_to(self, x: int, y: int) -> None:
        """Move the cursor to (x, y) without pressing a button."""
        ...

    def click(self, x: int, y: int, button: str = "left") -> None:
        """Press and release the given mouse button at (x, y)."""
        ...

    def double_click(self, x: int, y: int) -> None:
        """Double-click the left mouse button at (x, y)."""
        ...

    def right_click(self, x: int, y: int) -> None:
        """Right-click at (x, y)."""
        ...

    def drag(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        button: str = "left",
        duration: Optional[float] = None,
        hold_keys: str | Sequence[str] | None = None,
    ) -> None:
        """Press a mouse button at the start point and release it at the end."""
        ...

    def type_text(self, text: str) -> None:
        """Type a string into the focused element."""
        ...

    def press_key(self, key: str, presses: int = 1) -> None:
        """Press a key or key chord (e.g. ``"enter"``, ``"ctrl+c"``)."""
        ...

    def hotkey(self, chord: str | Sequence[str]) -> None:
        """Send a chord with every key held down at once (e.g. ``"alt+tab"``)."""
        ...

    def key_down(self, chord: str | Sequence[str]) -> None:
        """Hold a key/chord down until ``key_up`` is called."""
        ...

    def key_up(self, chord: str | Sequence[str]) -> None:
        """Release a key/chord held by ``key_down``."""
        ...

    def scroll(self, clicks: int) -> None:
        """Scroll the mouse wheel; positive scrolls up."""
        ...


class PyAutoGuiInput:
    """Concrete implementation backed by pyautogui."""

    #: Mouse buttons pyautogui accepts.
    BUTTONS = ("left", "right", "middle")

    #: Pause between the cursor arriving and the button going down. Windows
    #: applications process WM_MOUSEMOVE (hover, focus, tooltips) before
    #: WM_LBUTTONDOWN; pressing the button in the same instant makes the click
    #: land on the previously-hovered control or get dropped entirely, which is
    #: the classic "the cursor moved but nothing was clicked" failure.
    CLICK_SETTLE = 0.12

    def __init__(
        self,
        pause: float = 0.03,
        failsafe: bool = True,
        teleport_cursor: bool = False,
        move_duration: float = 0.25,
        typing_interval: float = 0.02,
        drag_duration: float = 0.4,
        click_settle: float = CLICK_SETTLE,
        warn: Callable[[str], None] | None = None,
    ) -> None:
        # Ensure the process is DPI-aware *before* pyautogui is imported, so
        # pyautogui.size()/moveTo and pyautogui.screenshot() share one
        # coordinate system on scaled Windows displays.
        from .windows import set_process_dpi_aware

        set_process_dpi_aware()

        import pyautogui  # lazy import keeps the module import-light

        self._pyautogui = pyautogui
        self._pyautogui.PAUSE = max(0.0, float(pause))
        # Move the mouse to a screen corner to abort any automation.
        self._pyautogui.FAILSAFE = failsafe
        #: Optional sink for one-off warnings (corner recoveries). The
        #: orchestrator points this at the journal.
        self.warn = warn
        self._corner_warned = False
        # When False (default) the cursor visibly *moves* to the target over
        # ``move_duration`` seconds; when True it teleports instantly.
        self.teleport_cursor = teleport_cursor
        self.move_duration = max(0.0, float(move_duration))
        self.drag_duration = max(0.0, float(drag_duration))
        self.typing_interval = max(0.0, float(typing_interval))
        self.click_settle = max(0.0, float(click_settle))
        # Chords must be dispatched as chords: pyautogui.press("ctrl+c") is a
        # silent no-op, which is why shortcuts used to never fire.
        self._keyboard = KeyboardController(
            pyautogui, interval=max(0.0, float(pause))
        )

    # ---------------------------------------------------- corner guard rails
    # pyautogui aborts automation while the pointer sits exactly on a screen
    # corner -- its "slam the mouse into a corner" kill switch. That is a
    # valuable human escape hatch, but it becomes a total freeze the moment a
    # stray coordinate parks the cursor at (0, 0): every later moveTo/click
    # raises before it can move anything, which looks exactly like "the mouse
    # stopped responding".
    def _failsafe_points(self) -> set[tuple[int, int]]:
        """The exact pixels pyautogui treats as its abort corners."""
        points = getattr(self._pyautogui, "FAILSAFE_POINTS", None)
        if points:
            return {(int(px), int(py)) for px, py in points}
        try:
            width, height = self._pyautogui.size()
        except Exception:  # pragma: no cover - backend without geometry
            return set()
        return {(0, 0), (0, height - 1), (width - 1, 0), (width - 1, height - 1)}

    def _is_abort_corner(self, point: Any) -> bool:
        """Whether ``point`` is one of pyautogui's abort corners."""
        try:
            return (int(point[0]), int(point[1])) in self._failsafe_points()
        except (TypeError, ValueError, IndexError):
            return False

    @contextmanager
    def _failsafe_suspended(self) -> Iterator[None]:
        """Run calls with pyautogui's corner abort temporarily disabled."""
        previous = self._pyautogui.FAILSAFE
        self._pyautogui.FAILSAFE = False
        try:
            yield
        finally:
            self._pyautogui.FAILSAFE = previous

    def _release_corner_park(self) -> None:
        """Nudge a corner-parked pointer back inward so input works again."""
        if not self._pyautogui.FAILSAFE:
            return
        try:
            current = self._pyautogui.position()
            width, height = self._pyautogui.size()
        except Exception:  # pragma: no cover - backend without geometry
            return
        if not self._is_abort_corner(current):
            return
        x = min(int(width) - 1, max(1, int(current[0])))
        y = min(int(height) - 1, max(1, int(current[1])))
        with self._failsafe_suspended():
            self._pyautogui.moveTo(x, y)
        if not self._corner_warned and self.warn is not None:
            self._corner_warned = True
            self.warn(
                f"The pointer was parked at {tuple(int(v) for v in current)}, a "
                "pyautogui abort corner, which blocks every input call; it was "
                "moved back onto the screen. Use the kill hotkey or the STOP "
                "button to abort a task instead."
            )

    @contextmanager
    def _pointer_ready(self, point: Any) -> Iterator[None]:
        """Make the pointer able to act on ``point``.

        A target that *is* an abort corner is legitimate (the Start button
        lives in one), so the abort is suspended for that one gesture. Any
        other corner park is treated as an artifact and released first.
        """
        if self._pyautogui.FAILSAFE and self._is_abort_corner(point):
            with self._failsafe_suspended():
                yield
            return
        self._release_corner_park()
        yield

    def _goto(self, x: int, y: int) -> None:
        """Position the cursor at (x, y), moving smoothly unless teleporting.

        Duration is scaled by distance so short hops stay quick while long
        traversals remain visible for the demo.
        """
        if self.teleport_cursor:
            self._pyautogui.moveTo(x, y)
            return
        current = self._pyautogui.position()
        distance = ((current[0] - x) ** 2 + (current[1] - y) ** 2) ** 0.5
        duration = min(self.move_duration, 0.05 + distance * 0.0008)
        self._pyautogui.moveTo(x, y, duration=duration)

    def _settle(self) -> None:
        """Let the target application process the pointer move before a press."""
        if self.click_settle > 0:
            time.sleep(self.click_settle)

    def move_to(self, x: int, y: int) -> None:
        """Move the cursor to (x, y) without clicking."""
        with self._pointer_ready((x, y)):
            self._goto(x, y)

    def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> None:
        with self._pointer_ready((x, y)):
            self._goto(x, y)
            self._settle()
            count = max(1, int(clicks))
            print(f"Clicking at ({x}, {y}) with button '{button}' x{count}")
            self._pyautogui.click(
                button=button,
                clicks=count,
                interval=self.typing_interval,
            )

    def double_click(self, x: int, y: int) -> None:
        with self._pointer_ready((x, y)):
            self._goto(x, y)
            self._settle()
            print(f"Double-clicking at ({x}, {y})")
            self._pyautogui.doubleClick()

    def right_click(self, x: int, y: int) -> None:
        with self._pointer_ready((x, y)):
            self._goto(x, y)
            self._settle()
            print(f"Right-clicking at ({x}, {y})")
            self._pyautogui.rightClick()

    def _resolve_button(self, button: str) -> str:
        resolved = str(button or "left").strip().lower()
        if resolved not in self.BUTTONS:
            raise ValueError(
                f"unsupported mouse button {button!r}; "
                f"expected one of {', '.join(self.BUTTONS)}"
            )
        return resolved

    def drag(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        button: str = "left",
        duration: Optional[float] = None,
        hold_keys: str | Sequence[str] | None = None,
    ) -> None:
        """Drag with ``button`` from the start point to the end point.

        ``hold_keys`` keeps extra keys held for the whole gesture (shift-drag
        to extend a selection, ctrl-drag to copy). The button is released in a
        ``finally`` block so a failed move can never leave the mouse stuck down.
        """
        button = self._resolve_button(button)
        # A drag that grabs or drops on an abort corner (an edge slider) must
        # not trip the corner abort mid-gesture: the mouseUp would raise and
        # leave the button held down.
        self._release_corner_park()
        corner_gesture = bool(self._pyautogui.FAILSAFE) and (
            self._is_abort_corner((int(start_x), int(start_y)))
            or self._is_abort_corner((int(end_x), int(end_y)))
        )
        if corner_gesture:
            self._pyautogui.FAILSAFE = False
        if hold_keys:
            self._keyboard.key_down(hold_keys)
        try:
            self._goto(int(start_x), int(start_y))
            move_duration = self._drag_duration(
                (int(start_x), int(start_y)), (int(end_x), int(end_y)), duration
            )
            print(
                f"Dragging from ({start_x}, {start_y}) to ({end_x}, {end_y}) "
                f"with button '{button}'"
            )
            # The grab point must already be under the cursor when the button
            # goes down, otherwise the app drags whatever it was hovering.
            self._settle()
            self._pyautogui.mouseDown(button=button)
            try:
                if self.teleport_cursor:
                    self._pyautogui.moveTo(end_x, end_y)
                else:
                    self._pyautogui.moveTo(end_x, end_y, duration=move_duration)
            finally:
                # Applications only complete a drop once the button is
                # released, so this must run even if the move raised.
                self._pyautogui.mouseUp(button=button)
            # Give the drop a moment to complete so a click issued straight
            # afterwards is not swallowed by the drag-and-drop machinery.
            self._settle()
        finally:
            if corner_gesture:
                self._pyautogui.FAILSAFE = True
            if hold_keys:
                self._keyboard.key_up(hold_keys)

    def _drag_duration(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        duration: Optional[float],
    ) -> float:
        if duration is not None:
            return max(0.0, float(duration))
        distance = ((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2) ** 0.5
        return min(self.drag_duration, 0.2 + distance * 0.001)

    def type_text(self, text: str) -> None:
        if not text:
            return
        # A tiny interval is intentional: sending the whole string in one
        # burst can drop characters in Windows controls with busy event loops.
        self._pyautogui.write(text, interval=self.typing_interval)

    def press_key(self, key: str, presses: int = 1) -> None:
        """Press a single key or a ``+``-joined chord such as ``ctrl+shift+t``."""
        keys = self._keyboard.parse(key)
        print(
            f"Pressing {'keys' if len(keys) > 1 else 'key'}: "
            f"{'+'.join(keys)}" + (f" x{presses}" if presses > 1 else "")
        )
        self._keyboard.press(keys, presses=presses)

    def hotkey(self, chord: str | Sequence[str]) -> None:
        """Hold every key of ``chord`` down together, then release them."""
        keys = self._keyboard.parse(chord)
        print(f"Hotkey: {'+'.join(keys)}")
        self._keyboard.hotkey(keys)

    def key_down(self, chord: str | Sequence[str]) -> None:
        print(f"Holding: {describe_chord(chord)}")
        self._keyboard.key_down(chord)

    def key_up(self, chord: str | Sequence[str]) -> None:
        self._keyboard.key_up(chord)

    def scroll(self, clicks: int) -> None:
        self._pyautogui.scroll(clicks)
