"""The local execution layer.

``VisionReflex`` replays a compiled skill in milliseconds using OpenCV template
matching -- no network, no tokens. It captures the screen, locates the stored
template with ``cv2.matchTemplate``, and, only if the match confidence clears
the threshold, performs the action. On failure it returns ``False``, which the
orchestrator uses to trigger the ``BrainPlanner`` fallback.

Future work: the same ``execute()`` interface can dispatch to a YOLOv8 detector
for the object classes that template matching handles poorly (free-form shapes).
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from .controller import InputController
from .models import ActionType, Skill
from .screen import ScreenCapture, map_capture_point_to_input

logger = logging.getLogger(__name__)

# (top-left x, top-left y), confidence, (template width, template height)
MatchResult = tuple[tuple[int, int], float, tuple[int, int]]


class VisionReflex:
    """Low-latency template-matching executor."""

    def __init__(
        self,
        screen: ScreenCapture,
        controller: InputController,
        confidence_threshold: float = 0.9,
        multi_scale: bool = True,
        scale_range: tuple[float, float] = (0.5, 1.5),
        scale_steps: int = 11,
    ) -> None:
        self._screen = screen
        self._input = controller
        self.confidence_threshold = confidence_threshold
        self.multi_scale = multi_scale
        self.scale_range = scale_range
        self.scale_steps = scale_steps

    # --------------------------------------------------------------- public
    def execute(self, skill: Skill) -> bool:
        """Replay a skill against the live screen.

        Returns True if the target was found and the action fired, False
        otherwise (UI moved / changed / threshold not met).
        """
        template = cv2.imread(skill.template_path, cv2.IMREAD_COLOR)
        if template is None:
            logger.error(
                "Template image could not be read for skill %r: %s",
                skill.name,
                skill.template_path,
            )
            return False

        screen_img = self._screen.capture()
        match = self._match(screen_img, template)
        if match is None:
            logger.info("No viable template match for skill %r.", skill.name)
            return False

        (x, y), confidence, (tw, th) = match
        if confidence < self.confidence_threshold:
            logger.info(
                "Confidence %.3f below threshold %.2f for skill %r.",
                confidence,
                self.confidence_threshold,
                skill.name,
            )
            return False

        frame_center = (x + tw // 2, y + th // 2)
        cx, cy = self.to_input_point(frame_center, screen_img.shape)
        logger.info(
            "Executing skill %r at frame=(%d, %d), input=(%d, %d) "
            "with confidence %.3f.",
            skill.name,
            frame_center[0],
            frame_center[1],
            cx,
            cy,
            confidence,
        )
        self._perform_action(skill, (cx, cy))
        return True

    def to_input_point(
        self,
        point: tuple[int, int],
        frame_shape: tuple[int, ...],
    ) -> tuple[int, int]:
        """Map a screenshot-space point to the input backend's coordinates."""
        return map_capture_point_to_input(self._screen, point, frame_shape)

    # ------------------------------------------------------------- matching
    def locate_on(
        self, screen_img: np.ndarray, template: np.ndarray
    ) -> Optional[MatchResult]:
        """Best multi-scale match of ``template`` inside ``screen_img``.

        Returns ``(top_left, confidence, (w, h))`` or ``None`` when the
        template is unusable. Exposed for the multi-step executor, which
        re-anchors planned crops on the live screen before acting.
        """
        if (
            template is None
            or template.size == 0
            or screen_img is None
            or template.shape[0] > screen_img.shape[0]
            or template.shape[1] > screen_img.shape[1]
        ):
            return None
        return self._match(screen_img, template)

    def _match(
        self, screen_img: np.ndarray, template: np.ndarray
    ) -> Optional[MatchResult]:
        """Find the best template match, optionally across multiple scales."""
        if (
            template is None
            or template.size == 0
            or screen_img is None
            or template.shape[0] > screen_img.shape[0]
            or template.shape[1] > screen_img.shape[1]
        ):
            return None
        exact = self._match_once(screen_img, template)
        if not self.multi_scale or exact[1] >= self.confidence_threshold:
            return exact

        best: Optional[MatchResult] = exact
        low, high = self.scale_range
        for scale in np.linspace(low, high, self.scale_steps):
            if abs(float(scale) - 1.0) < 1e-9:
                continue
            w = int(round(template.shape[1] * scale))
            h = int(round(template.shape[0] * scale))
            if w < 4 or h < 4 or w > screen_img.shape[1] or h > screen_img.shape[0]:
                continue
            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            resized = cv2.resize(template, (w, h), interpolation=interpolation)
            result = self._match_once(screen_img, resized)
            if result is None:
                continue
            if best is None or result[1] > best[1]:
                best = result
        return best

    def _match_once(
        self, screen_img: np.ndarray, template: np.ndarray
    ) -> MatchResult:
        """Single-scale match; returns location, confidence, and template size."""
        result = cv2.matchTemplate(screen_img, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        return (max_loc, float(max_val), (template.shape[1], template.shape[0]))

    # ------------------------------------------------------------ actuation
    def _perform_action(self, skill: Skill, center: tuple[int, int]) -> None:
        """Dispatch the skill's action to the input controller."""
        cx, cy = center
        action = skill.action
        metadata = skill.metadata
        params = metadata.get("params")
        params = params if isinstance(params, dict) else {}

        if action == ActionType.CLICK:
            self._input.click(cx, cy)
        elif action == ActionType.DOUBLE_CLICK:
            self._input.double_click(cx, cy)
        elif action == ActionType.RIGHT_CLICK:
            self._input.right_click(cx, cy)
        elif action == ActionType.DRAG:
            delta = metadata.get("drag_delta")
            if not isinstance(delta, (list, tuple)) or len(delta) != 2:
                raise ValueError("drag reflex has no recorded drag_delta")
            hold_keys = metadata.get("drag_hold_keys")
            if isinstance(hold_keys, str):
                hold_keys = [hold_keys]
            self._input.drag(
                cx,
                cy,
                int(cx + delta[0]),
                int(cy + delta[1]),
                button=str(metadata.get("drag_button") or "left"),
                duration=metadata.get("duration"),
                hold_keys=list(hold_keys) if hold_keys else None,
            )
        elif action == ActionType.TYPE:
            self._input.click(cx, cy)  # focus the field first
            text = str(metadata.get("text") or "")
            if text:
                self._input.type_text(text)
        elif action == ActionType.SCROLL:
            self._input.click(cx, cy)  # focus the pane first
            self._input.scroll(int(metadata.get("scroll_clicks", 3)))
        elif action == ActionType.KEY_PRESS:
            key = str(metadata.get("key") or params.get("key") or "enter")
            presses = params.get("presses")
            self._input.press_key(
                key, presses=int(presses) if isinstance(presses, (int, float)) else 1
            )
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported action type: {action}")
