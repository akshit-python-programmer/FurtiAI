"""Virtual input abstraction (mouse / keyboard).

Input is wrapped behind :class:`InputController` so the execution engine is
agnostic to the concrete automation library. The default implementation uses
``pyautogui``; swapping in ``pynput`` only means writing one new class.
"""

from __future__ import annotations

from typing import Protocol


class InputController(Protocol):
    """Structural interface for the input device."""

    def click(self, x: int, y: int, button: str = "left") -> None:
        """Press and release the given mouse button at (x, y)."""
        ...

    def double_click(self, x: int, y: int) -> None:
        """Double-click the left mouse button at (x, y)."""
        ...

    def right_click(self, x: int, y: int) -> None:
        """Right-click at (x, y)."""
        ...

    def type_text(self, text: str) -> None:
        """Type a string into the focused element."""
        ...

    def press_key(self, key: str) -> None:
        """Press a single key (e.g. ``"enter"``, ``"ctrl+c"``)."""
        ...

    def scroll(self, clicks: int) -> None:
        """Scroll the mouse wheel; positive scrolls up."""
        ...


class PyAutoGuiInput:
    """Concrete implementation backed by pyautogui."""

    def __init__(
        self,
        pause: float = 0.03,
        failsafe: bool = True,
        teleport_cursor: bool = False,
        move_duration: float = 0.25,
        typing_interval: float = 0.02,
    ) -> None:
        import pyautogui  # lazy import keeps the module import-light

        self._pyautogui = pyautogui
        self._pyautogui.PAUSE = max(0.0, float(pause))
        # Move the mouse to a screen corner to abort any automation.
        self._pyautogui.FAILSAFE = failsafe
        # When False (default) the cursor visibly *moves* to the target over
        # ``move_duration`` seconds; when True it teleports instantly.
        self.teleport_cursor = teleport_cursor
        self.move_duration = max(0.0, float(move_duration))
        self.typing_interval = max(0.0, float(typing_interval))

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

    def move_to(self, x: int, y: int) -> None:
        """Move the cursor to (x, y) without clicking."""
        self._goto(x, y)

    def click(self, x: int, y: int, button: str = "left") -> None:
        self._goto(x, y)
        self._pyautogui.click(button=button)

    def double_click(self, x: int, y: int) -> None:
        self._goto(x, y)
        self._pyautogui.doubleClick()

    def right_click(self, x: int, y: int) -> None:
        self._goto(x, y)
        self._pyautogui.rightClick()

    def type_text(self, text: str) -> None:
        if not text:
            return
        # A tiny interval is intentional: sending the whole string in one
        # burst can drop characters in Windows controls with busy event loops.
        self._pyautogui.write(text, interval=self.typing_interval)

    def press_key(self, key: str) -> None:
        self._pyautogui.press(key)

    def scroll(self, clicks: int) -> None:
        self._pyautogui.scroll(clicks)
