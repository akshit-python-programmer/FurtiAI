"""Rate-limited visual observation pipeline.

:class:`VisualContextManager` is the only place that touches the screen. It

1. throttles captures to at least ``screenshot_min_interval`` seconds apart
   (an absolute guard against screenshot-per-second token burn),
2. grounds the frame with fast RapidOCR text + cached saved-template matching,
3. asks the LLM -- cheaply, text-only -- whether the *raw screenshot* is
   actually important for this instruction, and only then encodes (downscaled)
   pixels, and
4. returns a :class:`SceneObservation` the planner can consume.

The gate keeps most calls text-only: OCR + icons already carry the
coordinates the model needs to click things.
"""

from __future__ import annotations

import base64
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import cv2
import numpy as np

from .config import Settings
from .jsoncontract import extract_json_object
from .ocr import IconMatch, IconMatcher, TextDetector, TextLine, describe_scene
from .screen import ScreenCapture

logger = logging.getLogger(__name__)

# Heuristic fallback when the gate model is unavailable or errors out: words
# that strongly suggest the raw pixels matter.
VISUAL_HINT_WORDS = (
    "click",
    "button",
    "icon",
    "image",
    "photo",
    "color",
    "colour",
    "logo",
    "picture",
    "look",
    "visual",
    "appearance",
    "screenshot",
    "chart",
    "graph",
    "diagram",
)

GATE_SYSTEM_PROMPT = (
    "You are the visual-gating component of a desktop automation agent. "
    "Given the user instruction and a text summary of the current screen "
    "(OCR text plus recognised icon templates with coordinates), decide "
    "whether the agent must also receive the raw screenshot image, or whether "
    "the text summary alone contains enough information to locate the target. "
    "Answer with JSON only: {\"use_screenshot\": true|false, \"reason\": \"...\"}. "
    "Prefer false whenever the text summary contains the target coordinates."
)


@dataclass
class SceneObservation:
    """Everything the planner may need to reason about the current screen."""

    frame: Optional[np.ndarray]
    text_lines: list[TextLine] = field(default_factory=list)
    icons: list[IconMatch] = field(default_factory=list)
    scene_text: str = ""
    image_b64: Optional[str] = None
    vision_used: bool = False
    vision_reason: str = ""
    fresh: bool = False
    captured_at: str = ""
    frame_size: tuple[int, int] = field(default_factory=tuple)
    vision_size: tuple[int, int] = field(default_factory=tuple)

    def prompt_block(self) -> str:
        """The text block appended to every planner/executor LLM request."""
        captured = (
            f" (captured at {self.captured_at})" if self.captured_at else ""
        )
        if self.frame_size:
            frame_dimensions = f"{self.frame_size[0]}x{self.frame_size[1]}"
        elif self.frame is not None:
            frame_dimensions = f"{self.frame.shape[1]}x{self.frame.shape[0]}"
        else:
            frame_dimensions = "unknown"
        attached_dimensions = (
            f"{self.vision_size[0]}x{self.vision_size[1]}"
            if self.vision_size
            else frame_dimensions
        )
        coordinate_note = (
            f"Coordinate reference: full screen capture is {frame_dimensions} "
            f"pixels. OCR/icon coordinates use that full-capture space. "
            f"The attached screenshot is {attached_dimensions} pixels; if you "
            "use coordinates read from the attached image, scale them back "
            "to the full capture dimensions."
        )
        return (
            f"Current screen grounding{captured}:\n"
            + coordinate_note
            + "\n"
            + (self.scene_text or "(no screen data)")
        )


# Minimum wall-clock gap between any two captures, even when a caller
# explicitly forces a fresh frame. This is the absolute "never screenshot
# every second" guard.
MIN_FORCE_INTERVAL = 1.0


class VisualContextManager:
    """Throttled capture + parallel fast grounding + screenshot gate."""

    def __init__(
        self,
        screen: ScreenCapture,
        text_detector: Optional[TextDetector],
        icon_matcher: Optional[IconMatcher],
        settings: Settings,
        llm: Any = None,  # any client exposing chat_text()
        stop_event: Any = None,
        journal: Any = None,
    ) -> None:
        self._screen = screen
        self._text_detector = text_detector
        self._icon_matcher = icon_matcher
        self._settings = settings
        self._llm = llm
        self._stop_event = stop_event
        self._journal = journal
        self._last_capture_ts = 0.0
        self._last_capture_at = ""
        self._last_capture_epoch = 0.0
        self._cached_frame: Optional[np.ndarray] = None
        self._cached_text: list[TextLine] = []
        self._cached_icons: list[IconMatch] = []
        self._gate_cache: dict[str, tuple[bool, str]] = {}
        self._grounding_failures: set[str] = set()
        self._last_encoded_size: tuple[int, int] = ()
        self._grounding_pool = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="furti-grounding",
        )
        self._closed = False

    @property
    def last_capture_ts(self) -> float:
        """Monotonic timestamp of the most recent screen capture."""
        return self._last_capture_ts

    @property
    def last_capture_at(self) -> str:
        """Local wall-clock time of the most recent screen capture."""
        return self._last_capture_at

    def _capture_due(self, force_fresh: bool) -> bool:
        """Whether a capture is allowed right now under the throttles."""
        if self._cached_frame is None:
            return True
        elapsed = time.monotonic() - self._last_capture_ts
        if force_fresh:
            return elapsed >= MIN_FORCE_INTERVAL
        return elapsed >= self._settings.screenshot_min_interval

    # --------------------------------------------------------------- public
    def observe(
        self,
        instruction: str,
        force_fresh: bool = False,
        include_screenshot: Optional[bool] = None,
        wait_for_fresh: bool = False,
    ) -> SceneObservation:
        """Ground the current screen for ``instruction``.

        ``force_fresh`` asks for a new capture (used right before acting) but
        still respects the 1-second absolute floor, so the agent can never
        take screenshots faster than once per second.

        ``wait_for_fresh`` waits out that floor instead of returning the
        cached frame. Post-action review uses it: a frame captured *before*
        the action cannot prove that the action worked, and silently skipping
        the review would report an unverified step as a success.
        """
        if self._closed:
            raise RuntimeError("visual context manager is closed")
        if self._stop_event is not None and self._stop_event.is_set():
            raise TaskAborted("stop hotkey pressed before screen capture")

        fresh = False
        should_capture = self._capture_due(force_fresh)
        if (
            not should_capture
            and force_fresh
            and wait_for_fresh
            and self._cached_frame is not None
        ):
            # Wait out the absolute floor (at most MIN_FORCE_INTERVAL) instead
            # of handing back the pre-action frame.
            deadline = self._last_capture_ts + MIN_FORCE_INTERVAL + 0.5
            while not should_capture and time.monotonic() < deadline:
                if self._stop_event is not None and self._stop_event.is_set():
                    raise TaskAborted(
                        "stop hotkey pressed while waiting for a capture slot"
                    )
                time.sleep(0.02)
                should_capture = self._capture_due(force_fresh)

        if should_capture:
            capture_started = time.perf_counter()
            self._cached_frame = self._screen.capture()
            self._last_capture_ts = time.monotonic()
            self._last_capture_epoch = time.time()
            self._last_capture_at = datetime.now().astimezone().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            fresh = True
            if self._journal is not None:
                self._journal.thought(
                    f"Raw screenshot captured in "
                    f"{time.perf_counter() - capture_started:.2f}s; "
                    "grounding it with OCR and icon matching."
                )
            grounding_started = time.perf_counter()
            text_future = self._grounding_pool.submit(
                self._text_detector.detect,
                self._cached_frame,
            ) if (
                self._text_detector is not None
                and self._text_detector.available
            ) else None
            icon_future = self._grounding_pool.submit(
                self._icon_matcher.find_icons,
                self._cached_frame,
            ) if self._icon_matcher is not None else None
            self._cached_text = self._read_grounding_result(
                text_future,
                "OCR",
            )
            self._note_text_detection_failure()
            self._cached_icons = self._read_grounding_result(
                icon_future,
                "icon matching",
            )
            logger.debug(
                "Fresh capture: %s, %d OCR lines, %d icon matches.",
                self._cached_frame.shape,
                len(self._cached_text),
                len(self._cached_icons),
            )
            if self._journal is not None:
                self._journal.thought(
                    f"Screen grounding completed in "
                    f"{time.perf_counter() - grounding_started:.2f}s "
                    f"(OCR={len(self._cached_text)}, "
                    f"icons={len(self._cached_icons)})."
                )
            if self._journal is not None:
                self._journal.screenshot(
                    self._last_capture_at,
                    self._last_capture_epoch,
                    self._cached_frame.shape,
                    len(self._cached_text),
                    len(self._cached_icons),
                )

        if include_screenshot is True:
            # Post-step review explicitly requests the image, but only attach
            # a frame captured after the action. Never resend a pre-action
            # cached frame as if it were current.
            use_screenshot = fresh
            reason = (
                "post-step review requested visual evidence"
                if fresh
                else "post-step review has no fresh frame; use text grounding"
            )
        elif include_screenshot is False:
            use_screenshot, reason = False, "caller requested text grounding only"
        else:
            use_screenshot, reason = self._decide_vision(instruction)
        image_b64: Optional[str] = None
        frame_size = (
            (int(self._cached_frame.shape[1]), int(self._cached_frame.shape[0]))
            if self._cached_frame is not None
            else ()
        )
        vision_size = frame_size
        if use_screenshot and self._cached_frame is not None:
            image_b64 = self._encode_screen(self._cached_frame)
            vision_size = self._last_encoded_size or frame_size

        return SceneObservation(
            frame=self._cached_frame,
            text_lines=list(self._cached_text),
            icons=list(self._cached_icons),
            scene_text=describe_scene(self._cached_text, self._cached_icons),
            image_b64=image_b64,
            vision_used=use_screenshot,
            vision_reason=reason,
            fresh=fresh,
            captured_at=self._last_capture_at,
            frame_size=frame_size,
            vision_size=vision_size,
        )

    def _read_grounding_result(self, future: Any, label: str) -> list[Any]:
        """Return optional grounding output without hiding backend failures."""
        if future is None:
            return []
        try:
            result = future.result()
        except Exception as exc:
            logger.warning("%s failed for this frame: %s", label, exc)
            if label not in self._grounding_failures:
                self._grounding_failures.add(label)
                if self._journal is not None:
                    # A broken grounding backend removes anchors silently
                    # otherwise; report it once per task instead of per frame.
                    self._journal.warn(
                        f"{label} is failing on every frame ({exc}); "
                        "target anchoring is degraded for this task."
                    )
            return []
        return list(result or [])

    def _note_text_detection_failure(self) -> None:
        """Report an OCR backend that died mid-task, once per task.

        A backend that fails on every frame would otherwise leave the agent
        with no text anchors and no explanation in the report.
        """
        detector = self._text_detector
        if (
            self._journal is None
            or detector is None
            or not getattr(detector, "runtime_failure", "")
            or "text detection" in self._grounding_failures
        ):
            return
        self._grounding_failures.add("text detection")
        self._journal.warn(
            f"text detection failed on every frame ({detector.runtime_failure}); "
            "OCR is disabled and target anchoring is degraded for this task."
        )

    def observe_for_review(
        self,
        instruction: str,
        force_fresh: bool = True,
    ) -> SceneObservation:
        """Observe a post-action state and attach pixels when freshly captured.

        The capture floor is awaited rather than skipped: a step can only be
        verified against a frame that was taken after its action.
        """
        return self.observe(
            instruction,
            force_fresh=force_fresh,
            include_screenshot=True,
            wait_for_fresh=True,
        )

    def last_scene_text(self) -> str:
        return describe_scene(self._cached_text, self._cached_icons)

    def close(self) -> None:
        """Stop the short-lived grounding workers after a task completes."""
        if self._closed:
            return
        self._closed = True
        self._grounding_pool.shutdown(wait=True, cancel_futures=True)

    # ---------------------------------------------------------------- gating
    def _decide_vision(self, instruction: str) -> tuple[bool, str]:
        cached = self._gate_cache.get(instruction)
        if cached is not None:
            return cached

        if self._llm is not None:
            try:
                if self._journal is not None:
                    self._journal.thought(
                        "AI is deciding whether the current task needs "
                        "visual input."
                    )
                    self._journal.waiting(
                        "Waiting for AI response (visual-input decision)..."
                    )
                raw = self._llm.chat_text(
                    GATE_SYSTEM_PROMPT,
                    f"User instruction: {instruction}\n\n{self.last_scene_text()}",
                    purpose="visual_gate",
                )
                if self._journal is not None:
                    self._journal.thought(
                        "AI response received (visual-input decision)."
                    )
                    self._journal.ai_output(
                        raw,
                        getattr(self._llm, "_model", type(self._llm).__name__),
                        "visual_gate",
                    )
                payload = extract_json_object(raw)
                use = bool(payload.get("use_screenshot", False))
                reason = str(payload.get("reason", ""))
                result = (use, reason)
                self._gate_cache[instruction] = result
                logger.debug(
                    "Visual gate for %r -> use_screenshot=%s (%s)",
                    instruction,
                    use,
                    reason,
                )
                return result
            except Exception as exc:
                logger.warning("Visual gate model failed (%s); using heuristic.", exc)

        result = self._heuristic_gate(instruction)
        self._gate_cache[instruction] = result
        return result

    def _heuristic_gate(self, instruction: str) -> tuple[bool, str]:
        lowered = instruction.lower()
        has_visual_words = any(word in lowered for word in VISUAL_HINT_WORDS)
        has_grounding = len(self._cached_text) >= 2 or bool(self._cached_icons)
        if has_visual_words or not has_grounding:
            return True, "heuristic: instruction mentions visual targets or screen grounding is thin"
        return False, "heuristic: OCR/icon grounding is sufficient"

    # -------------------------------------------------------------- encoding
    def _encode_screen(self, frame: np.ndarray) -> str:
        image = frame
        height, width = image.shape[:2]
        max_dim = self._settings.max_image_dim
        if max(width, height) > max_dim:
            scale = max_dim / max(width, height)
            image = cv2.resize(
                image,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        ok, buffer = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("Failed to encode screenshot as PNG.")
        self._last_encoded_size = (int(image.shape[1]), int(image.shape[0]))
        return base64.b64encode(buffer.tobytes()).decode("ascii")


class TaskAborted(Exception):
    """Raised when the kill hotkey / stop button aborts a running task."""
