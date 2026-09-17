"""Fast grounding layer: RapidOCR text detection + cached CV icon matching.

Before the LLM is consulted the screen is reduced to a compact, cheap textual
scene description:

* :class:`TextDetector` uses lightweight RapidOCR ONNX inference by default
  and returns every text line with its pixel box and recognition confidence.
* :class:`IconMatcher` caches grayscale templates, downsamples large frames,
  and performs one exact-scale ``cv2.matchTemplate`` pass by default.

PaddleOCR is normally the legacy compatibility backend, but it is loaded
automatically when RapidOCR cannot be imported: without a text backend every
capture would report zero text lines and the executor would have no text
anchors at all.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .models import BoundingBox

logger = logging.getLogger(__name__)

# Anchor crops saved by the executor (planned bboxes and OCR boxes) share the
# template directory with user templates, but they are single-run snapshots
# of whatever happened to be on screen. Advertising them as ``icon:<name>``
# lets a stale crop from an unrelated task become the anchor the model
# clicks, so they are kept out of the icon library.
AUTO_CROP_SUFFIX = ".auto"
_LEGACY_AUTO_CROP_RE = re.compile(r"^task(?:_\d+|_ocr_\d+)$")


def is_auto_crop_template(stem: str) -> bool:
    """Return True for executor-saved anchor crops (not user icons)."""
    return stem.endswith(AUTO_CROP_SUFFIX) or bool(_LEGACY_AUTO_CROP_RE.match(stem))


@dataclass
class TextLine:
    """One OCR-detected line of text with its screen location."""

    text: str
    bbox: BoundingBox
    confidence: float = 0.0

    @property
    def center(self) -> tuple[int, int]:
        return self.bbox.center


@dataclass
class IconMatch:
    """A saved template recognized on the screen."""

    name: str
    bbox: BoundingBox
    confidence: float

    @property
    def center(self) -> tuple[int, int]:
        return self.bbox.center


def _points_to_bbox(points: Any, image_shape: tuple[int, int]) -> BoundingBox:
    """Convert an Nx2 array of corner points to a clamped BoundingBox."""
    try:
        arr = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    except (TypeError, ValueError):
        return BoundingBox(0, 0, 0, 0)
    if arr.shape[0] == 0:
        return BoundingBox(0, 0, 0, 0)
    height, width = image_shape[:2]
    x0 = max(0, int(arr[:, 0].min()))
    y0 = max(0, int(arr[:, 1].min()))
    x1 = min(width, int(arr[:, 0].max()))
    y1 = min(height, int(arr[:, 1].max()))
    return BoundingBox(x0, y0, max(0, x1 - x0), max(0, y1 - y0))


class TextDetector:
    """Fast OCR wrapper with RapidOCR-first and PaddleOCR compatibility paths."""

    # Consecutive failed frames tolerated before text detection is switched
    # off for the task. A single failure is survivable; a persistent one must
    # not stay silent, because it removes every text anchor for the rest of
    # the task without the journal saying so.
    MAX_CONSECUTIVE_FAILURES = 3

    def __init__(
        self,
        lang: str = "en",
        enabled: bool = True,
        enable_mkldnn: bool = False,
        max_dim: int = 640,
        backend: str = "rapidocr",
        min_confidence: float = 0.35,
        max_lines: int = 80,
    ) -> None:
        self.lang = lang
        self.enabled = enabled
        self.enable_mkldnn = enable_mkldnn
        self.max_dim = max(320, int(max_dim))
        self.backend = str(backend or "rapidocr").lower()
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))
        self.max_lines = max(1, int(max_lines))
        self.available = False
        self._ocr: Any = None
        self._active_backend = ""
        # Why the active backend differs from the requested one, if it does.
        self.backend_note = ""
        # Inference error that disabled OCR after the model loaded, if any.
        self.runtime_failure = ""
        self._consecutive_failures = 0
        if enabled:
            self._init_ocr()

    def _init_ocr(self) -> None:
        if self.backend in {"rapidocr", "auto"}:
            try:
                from rapidocr_onnxruntime import RapidOCR

                self._ocr = RapidOCR()
                self._active_backend = "rapidocr"
                self.available = True
                logger.info(
                    "RapidOCR ready (lang=%s, max_dim=%d).",
                    self.lang,
                    self.max_dim,
                )
                return
            except Exception as exc:
                # RapidOCR is the fast path, never a hard requirement: without
                # any text backend the executor loses every text anchor, so
                # always try the compatibility backend before giving up.
                self.backend_note = f"RapidOCR unavailable ({exc})"
                logger.warning(
                    "%s. Trying the PaddleOCR compatibility backend. "
                    "Install requirements-ocr.txt for the faster backend.",
                    self.backend_note,
                )

        self._init_paddle()

    def _init_paddle(self) -> None:
        try:
            from paddleocr import PaddleOCR  # heavy import; lazy on purpose

            kwargs: dict[str, Any] = {
                "lang": self.lang,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
                "enable_mkldnn": self.enable_mkldnn,
            }
            if self.lang.lower() == "en":
                # The default v6 medium models are unnecessarily slow for
                # locating desktop labels. The v5 mobile pair is sufficient
                # for UI text and keeps each grounding pass responsive.
                kwargs.update(
                    text_detection_model_name="PP-OCRv5_mobile_det",
                    text_recognition_model_name="en_PP-OCRv5_mobile_rec",
                )
            self._ocr = PaddleOCR(
                **kwargs,
            )
            self._active_backend = "paddle"
            self.available = True
            if self.backend_note:
                self.backend_note += "; using the PaddleOCR fallback"
            logger.info("PaddleOCR ready (lang=%s).", self.lang)
        except Exception as exc:
            self.available = False
            self.backend_note = (
                f"{self.backend_note}; PaddleOCR unavailable ({exc})"
                if self.backend_note
                else f"PaddleOCR unavailable ({exc})"
            )
            logger.warning(
                "OCR backend could not be initialised (%s). Text grounding "
                "is disabled; the agent falls back to icon matching and "
                "screenshot reasoning.",
                exc,
            )

    # --------------------------------------------------------------- public
    def describe(self) -> str:
        """Report the *actual* OCR state for the journal and status window."""
        if not self.enabled:
            return "OCR disabled by configuration (FURTI_OCR_ENABLED=false)"
        if not self.available:
            reason = (
                self.runtime_failure
                or self.backend_note
                or "no backend could be initialised"
            )
            return (
                "OCR UNAVAILABLE - text anchoring is off, so targets can "
                f"only come from icon templates or the screenshot ({reason})"
            )
        state = (
            f"OCR active (requested={self.backend}, "
            f"active={self._active_backend})"
        )
        return f"{state}; {self.backend_note}" if self.backend_note else state

    def detect(self, image: np.ndarray) -> list[TextLine]:
        """Return all text lines found on ``image`` (BGR, uint8)."""
        if not self.available or self._ocr is None:
            return []
        if image.size == 0:
            return []
        # Keep OCR bounded on large/high-DPI displays. Coordinates are mapped
        # back to the original capture space before they reach the executor.
        source_height, source_width = image.shape[:2]
        ocr_image = image
        if max(source_width, source_height) > self.max_dim:
            scale = self.max_dim / max(source_width, source_height)
            ocr_image = cv2.resize(
                image,
                (
                    max(1, round(source_width * scale)),
                    max(1, round(source_height * scale)),
                ),
                interpolation=cv2.INTER_AREA,
            )
        # RapidOCR consumes the OpenCV BGR array directly. PaddleOCR's
        # compatibility path expects RGB.
        ocr_input = (
            ocr_image
            if self._active_backend == "rapidocr"
            else cv2.cvtColor(ocr_image, cv2.COLOR_BGR2RGB)
        )
        try:
            raw = self._run_ocr(ocr_input)
        except Exception as exc:
            self._notice_inference_failure(exc)
            return []
        self._consecutive_failures = 0
        lines = [
            line
            for line in self._parse(raw, ocr_image.shape[:2])
            if line.confidence >= self.min_confidence
        ]
        if len(lines) > self.max_lines:
            lines = sorted(lines, key=lambda line: line.confidence, reverse=True)[
                : self.max_lines
            ]
        if ocr_image.shape[:2] != image.shape[:2]:
            scale_x = source_width / ocr_image.shape[1]
            scale_y = source_height / ocr_image.shape[0]
            lines = [
                TextLine(
                    text=line.text,
                    bbox=BoundingBox(
                        x=round(line.bbox.x * scale_x),
                        y=round(line.bbox.y * scale_y),
                        width=max(1, round(line.bbox.width * scale_x)),
                        height=max(1, round(line.bbox.height * scale_y)),
                    ).clamp(source_width, source_height),
                    confidence=line.confidence,
                )
                for line in lines
            ]
        return lines

    def _notice_inference_failure(self, exc: Exception) -> None:
        """Log a failed frame and disable OCR once it stops being transient."""
        self._consecutive_failures += 1
        logger.warning(
            "%s text detection failed; continuing without OCR for this "
            "frame: %s",
            self._active_backend or "OCR",
            exc,
        )
        if self._consecutive_failures < self.MAX_CONSECUTIVE_FAILURES:
            return
        if not self.runtime_failure:
            self.runtime_failure = str(exc)
            logger.error(
                "Text detection failed on %d consecutive frames (%s); "
                "disabling OCR for this task.",
                self._consecutive_failures,
                exc,
            )
        self.available = False

    def _run_ocr(self, rgb: np.ndarray) -> Any:
        """Call RapidOCR/PaddleOCR without version-specific keyword arguments."""
        predict = getattr(self._ocr, "predict", None)
        legacy_ocr = getattr(self._ocr, "ocr", None)

        if callable(predict):
            try:
                return predict(rgb)
            except Exception as predict_exc:
                if not callable(legacy_ocr):
                    raise
                logger.debug(
                    "%s.predict failed (%s); trying compatibility "
                    "ocr(img) call.",
                    self._active_backend or "OCR",
                    predict_exc,
                )
                try:
                    # PaddleOCR 3.x implements ocr(img, **kwargs) by
                    # forwarding kwargs to predict(); passing cls=True here
                    # raises the reported TypeError. PaddleOCR 2.x defaults
                    # cls to True, so no explicit keyword is needed there.
                    return legacy_ocr(rgb)
                except Exception as legacy_exc:
                    raise RuntimeError(
                        f"{self._active_backend or 'OCR'} predict and "
                        "compatibility calls failed: "
                        f"{predict_exc}; {legacy_exc}"
                    ) from legacy_exc

        if callable(legacy_ocr):
            return legacy_ocr(rgb)
        if callable(self._ocr):
            return self._ocr(rgb)
        raise AttributeError("OCR backend exposes no supported inference method")

    @staticmethod
    def _parse(raw: Any, shape: tuple[int, int]) -> list[TextLine]:
        lines: list[TextLine] = []
        if raw is None:
            return lines
        # rapidocr_onnxruntime returns (items, elapsed) in current releases.
        if isinstance(raw, tuple) and raw:
            raw = raw[0]

        # Some RapidOCR releases expose an object with parallel arrays.
        if not isinstance(raw, (list, tuple, dict)):
            boxes = getattr(raw, "boxes", None)
            texts = getattr(raw, "txts", None)
            if texts is None:
                texts = getattr(raw, "texts", None)
            scores = getattr(raw, "scores", None)
            if boxes is not None and texts is not None:
                try:
                    count = min(len(boxes), len(texts))
                except TypeError:
                    count = 0
                raw = []
                for index in range(count):
                    score = (
                        scores[index]
                        if scores is not None and index < len(scores)
                        else 0.0
                    )
                    raw.append([boxes[index], texts[index], score])

        # RapidOCR's compact result shape is:
        # [polygon, recognized_text, confidence].
        if isinstance(raw, list) and raw and all(
            isinstance(item, (list, tuple))
            and len(item) >= 3
            and isinstance(item[1], (str, np.str_))
            for item in raw
        ):
            for item in raw:
                text = str(item[1]).strip()
                if not text:
                    continue
                try:
                    score = float(item[2])
                except (TypeError, ValueError):
                    score = 0.0
                lines.append(
                    TextLine(
                        text=text,
                        bbox=_points_to_bbox(item[0], shape),
                        confidence=score,
                    )
                )
            return lines

        # PaddleOCR 3.x: list of dicts with rec_texts / rec_scores / rec_polys.
        if isinstance(raw, list):
            for page in raw:
                if isinstance(page, dict):
                    lines.extend(TextDetector._parse_v3_page(page, shape))
                elif isinstance(page, list):
                    lines.extend(TextDetector._parse_v2_page(page, shape))
        elif isinstance(raw, dict):
            lines.extend(TextDetector._parse_v3_page(raw, shape))
        return lines

    @staticmethod
    def _parse_v3_page(page: dict[str, Any], shape: tuple[int, int]) -> list[TextLine]:
        texts = page.get("rec_texts")
        if texts is None:
            texts = []
        scores = page.get("rec_scores")
        if scores is None:
            scores = []
        polys = page.get("rec_polys")
        if polys is None:
            polys = page.get("dt_polys")
        if polys is None:
            polys = []
        lines: list[TextLine] = []
        for index, text in enumerate(texts):
            if not text or not str(text).strip():
                continue
            poly = polys[index] if index < len(polys) else None
            bbox = _points_to_bbox(poly, shape) if poly is not None else BoundingBox(0, 0, 0, 0)
            score = float(scores[index]) if index < len(scores) else 0.0
            lines.append(TextLine(text=str(text), bbox=bbox, confidence=score))
        return lines

    @staticmethod
    def _parse_v2_page(page: list[Any], shape: tuple[int, int]) -> list[TextLine]:
        lines: list[TextLine] = []
        for item in page or []:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            poly, payload = item
            text, score = "", 0.0
            if isinstance(payload, (list, tuple)) and payload:
                text = str(payload[0])
                if len(payload) > 1:
                    try:
                        score = float(payload[1])
                    except (TypeError, ValueError):
                        score = 0.0
            if not text.strip():
                continue
            lines.append(
                TextLine(
                    text=text,
                    bbox=_points_to_bbox(poly, shape),
                    confidence=score,
                )
            )
        return lines


class IconMatcher:
    """Pattern-match the saved template library against the screen.

    Named user templates double as icons. Recognising icons this way is free
    (no LLM tokens) and gives the planner stable named anchors. Executor-saved
    anchor crops are skipped: see :func:`is_auto_crop_template`.
    """

    def __init__(
        self,
        templates_dir: Path,
        threshold: float = 0.85,
        max_templates: int = 24,
        max_screen_dim: int = 1280,
        multi_scale: bool = False,
    ) -> None:
        self.templates_dir = Path(templates_dir)
        self.threshold = threshold
        self.max_templates = max(1, int(max_templates))
        self.max_screen_dim = max(320, int(max_screen_dim))
        self.multi_scale = bool(multi_scale)
        self._template_cache: dict[str, tuple[int, np.ndarray]] = {}

    def _template_files(self) -> list[Path]:
        if not self.templates_dir.exists():
            return []
        files = [
            path
            for path in sorted(self.templates_dir.glob("*.png"))
            if not is_auto_crop_template(path.stem)
        ]
        return files[: self.max_templates]

    # --------------------------------------------------------------- public
    def find_icons(self, screen: np.ndarray) -> list[IconMatch]:
        """Return every saved template whose best match clears the threshold."""
        matches: list[IconMatch] = []
        if screen.size == 0:
            return matches
        gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        scale = 1.0
        if max(gray_screen.shape[:2]) > self.max_screen_dim:
            scale = self.max_screen_dim / max(gray_screen.shape[:2])
            gray_screen = cv2.resize(
                gray_screen,
                (
                    max(1, round(gray_screen.shape[1] * scale)),
                    max(1, round(gray_screen.shape[0] * scale)),
                ),
                interpolation=cv2.INTER_AREA,
            )

        for path in self._template_files():
            template = self._load_template(path)
            if template is None or template.size == 0:
                continue
            if scale != 1.0:
                template = cv2.resize(
                    template,
                    (
                        max(1, round(template.shape[1] * scale)),
                        max(1, round(template.shape[0] * scale)),
                    ),
                    interpolation=cv2.INTER_AREA,
                )
            result = self._match_multi_scale(
                gray_screen,
                template,
                multi_scale=self.multi_scale,
            )
            if result is None:
                continue
            (x, y), confidence, (tw, th) = result
            if confidence < self.threshold:
                continue
            if scale != 1.0:
                x = round(x / scale)
                y = round(y / scale)
                tw = max(1, round(tw / scale))
                th = max(1, round(th / scale))
            matches.append(
                IconMatch(
                    name=path.stem,
                    bbox=BoundingBox(x, y, tw, th),
                    confidence=confidence,
                )
            )
        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches

    def _load_template(self, path: Path) -> Optional[np.ndarray]:
        """Load each template once and invalidate it when the file changes."""
        try:
            stat = path.stat()
        except OSError:
            return None
        key = str(path)
        cached = self._template_cache.get(key)
        if cached is not None and cached[0] == stat.st_mtime_ns:
            return cached[1]
        template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if template is None or template.size == 0:
            return None
        self._template_cache[key] = (stat.st_mtime_ns, template)
        return template

    def _match_multi_scale(
        self,
        screen: np.ndarray,
        template: np.ndarray,
        *,
        multi_scale: Optional[bool] = None,
    ) -> Optional[tuple[tuple[int, int], float, tuple[int, int]]]:
        if (
            template.shape[0] > screen.shape[0]
            or template.shape[1] > screen.shape[1]
        ):
            return None
        exact = self._match_once(screen, template)
        if exact[1] >= self.threshold:
            return exact
        if multi_scale is None:
            multi_scale = self.multi_scale
        if not multi_scale:
            return exact
        best: Optional[tuple[tuple[int, int], float, tuple[int, int]]] = exact
        for scale in np.linspace(0.6, 1.4, 9):
            if abs(float(scale) - 1.0) < 1e-9:
                continue
            w = int(round(template.shape[1] * scale))
            h = int(round(template.shape[0] * scale))
            if w < 4 or h < 4 or w > screen.shape[1] or h > screen.shape[0]:
                continue
            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            resized = cv2.resize(template, (w, h), interpolation=interpolation)
            response = cv2.matchTemplate(screen, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(response)
            candidate = (max_loc, float(max_val), (w, h))
            if best is None or candidate[1] > best[1]:
                best = candidate
        return best

    @staticmethod
    def _match_once(
        screen: np.ndarray, template: np.ndarray
    ) -> tuple[tuple[int, int], float, tuple[int, int]]:
        response = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(response)
        return (max_loc, float(max_val), (template.shape[1], template.shape[0]))


# --------------------------------------------------------------------------
# Scene formatting shared by the visual-context pipeline.
# --------------------------------------------------------------------------
def describe_scene(text_lines: list[TextLine], icons: list[IconMatch]) -> str:
    """Render the grounding output as a compact text block for the LLM."""
    parts: list[str] = []
    if text_lines:
        parts.append("OCR text on screen (text, center, box):")
        for line in text_lines[:40]:
            parts.append(
                f"- {line.text!r} center=({line.center[0]},{line.center[1]}) "
                f"box=({line.bbox.x},{line.bbox.y},{line.bbox.width},{line.bbox.height}) "
                f"conf={line.confidence:.2f}"
            )
    else:
        parts.append("OCR text on screen: none detected.")
    if icons:
        parts.append("Recognised icons from the saved template library:")
        for icon in icons[:20]:
            parts.append(
                f"- icon:{icon.name} center=({icon.center[0]},{icon.center[1]}) "
                f"conf={icon.confidence:.2f}"
            )
    else:
        parts.append("Recognised icons: none of the saved templates matched.")
    return "\n".join(parts)
