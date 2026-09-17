"""Screen capture abstraction.

Capturing is isolated behind :class:`ScreenCapture` so the pipeline can be
tested with a synthetic screen (see ``__main__.py``) and so faster backends
(e.g. ``mss``) can be dropped in without touching the rest of the code.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class ScreenCapture(Protocol):
    """Anything that can produce the current screen as a BGR numpy array."""

    def capture(self) -> np.ndarray:
        """Return the screen as an HxWx3 BGR ``np.ndarray`` of ``uint8``."""
        ...


def map_capture_point_to_input(
    screen: Any,
    point: tuple[int, int],
    frame_shape: tuple[int, ...],
) -> tuple[int, int]:
    """Map a captured-image pixel to the coordinate system used for input."""
    mapper = getattr(screen, "to_input_point", None)
    if callable(mapper):
        return mapper(point, frame_shape)
    return int(point[0]), int(point[1])


class PyAutoGuiScreen:
    """Concrete capture using pyautogui (Pillow under the hood)."""

    def __init__(self) -> None:
        # Pin DPI awareness before pyautogui/Pillow is first imported so the
        # screenshot (physical pixels) and pyautogui.size() (mouse coords)
        # always agree. Safe to call more than once: the OS ignores it once
        # awareness is already set.
        from .windows import set_process_dpi_aware

        set_process_dpi_aware()

    def capture(self) -> np.ndarray:
        import cv2
        import pyautogui

        pil_image = pyautogui.screenshot()  # PIL Image, RGB byte order
        frame = np.array(pil_image)
        # cv2.imshow("Furti AI: screen capture", frame)
        # cv2.waitKey(0)  # Wait for a key press to close the window
        # OpenCV works in BGR; convert so matching is consistent everywhere.
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    @staticmethod
    def input_size() -> tuple[int, int]:
        """Return the screen size expected by PyAutoGUI mouse methods."""
        import pyautogui

        size = pyautogui.size()
        width = getattr(size, "width", None)
        height = getattr(size, "height", None)
        if width is None or height is None:
            width, height = size
        return int(width), int(height)

    def to_input_point(
        self,
        point: tuple[int, int],
        frame_shape: tuple[int, ...],
    ) -> tuple[int, int]:
        """Convert capture pixels to PyAutoGUI coordinates.

        On a scaled Windows display, the screenshot can use physical pixels
        while ``pyautogui.moveTo`` consumes logical pixels.
        """
        if len(frame_shape) < 2:
            return int(point[0]), int(point[1])
        frame_height, frame_width = int(frame_shape[0]), int(frame_shape[1])
        input_width, input_height = self.input_size()
        if frame_width <= 0 or frame_height <= 0:
            return int(point[0]), int(point[1])

        x = round(float(point[0]) * input_width / frame_width)
        y = round(float(point[1]) * input_height / frame_height)
        return (
            min(max(0, x), max(0, input_width - 1)),
            min(max(0, y), max(0, input_height - 1)),
        )
