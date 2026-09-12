"""Grounding layer: PaddleOCR text detection + saved-template icon matching.

Before the LLM is consulted the screen is reduced to a compact, cheap textual
scene description:

* :class:`TextDetector` runs PaddleOCR and returns every text line with its
  pixel box and recognition confidence.
* :class:`IconMatcher` pattern-matches every saved reflex template (the icon
  library the agent has already learned) against the current screen using
  multi-scale ``cv2.matchTemplate``.

The LLM then works with coordinates/text *and* may still request the raw
screenshot when it decides the visual detail matters -- so most calls can run
text-only and cheap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .models import BoundingBox

logger = logging.getLogger(__name__)


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
    """PaddleOCR wrapper tolerant of both the 2.x and 3.x APIs."""

    def __init__(
        self,
        lang: str = "en",
        enabled: bool = True,
        enable_mkldnn: bool = False,
        max_dim: int = 960,
    ) -> None:
        self.lang = lang
        self.enabled = enabled
        self.enable_mkldnn = enable_mkldnn
        self.max_dim = max(320, int(max_dim))
        self.available = False
        self._ocr: Any = None
        if enabled:
            self._init_ocr()

    def _init_ocr(self) -> None:
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
            self.available = True
            logger.info("PaddleOCR ready (lang=%s).", self.lang)
        except Exception as exc:
            self.available = False
            logger.warning(
                "PaddleOCR could not be initialised (%s). "
                "Text grounding is disabled; the agent falls back to icon "
                "matching + screenshot-only reasoning.",
                exc,
            )

    # --------------------------------------------------------------- public
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
        # PaddleOCR is RGB-first; cv2 gives us BGR.
        rgb = cv2.cvtColor(ocr_image, cv2.COLOR_BGR2RGB)
        try:
            raw = self._run_ocr(rgb)
        except Exception as exc:
            # OCR is an optional grounding aid; do not abort the task when a
            # model/API mismatch prevents text detection for one frame.
            logger.warning(
                "PaddleOCR text detection failed; continuing without OCR "
                "for this frame: %s",
                exc,
            )
            return []
        lines = self._parse(raw, ocr_image.shape[:2])
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

    def _run_ocr(self, rgb: np.ndarray) -> Any:
        """Call PaddleOCR without passing 2.x-only ``cls`` to 3.x wrappers."""
        predict = getattr(self._ocr, "predict", None)
        legacy_ocr = getattr(self._ocr, "ocr", None)

        if callable(predict):
            try:
                return predict(rgb)
            except Exception as predict_exc:
                if not callable(legacy_ocr):
                    raise
                logger.debug(
                    "PaddleOCR.predict failed (%s); trying compatibility "
                    "ocr(img) call.",
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
                        "PaddleOCR predict and compatibility calls failed: "
                        f"{predict_exc}; {legacy_exc}"
                    ) from legacy_exc

        if callable(legacy_ocr):
            return legacy_ocr(rgb)
        raise AttributeError("PaddleOCR exposes neither predict() nor ocr()")

    @staticmethod
    def _parse(raw: Any, shape: tuple[int, int]) -> list[TextLine]:
        lines: list[TextLine] = []
        if raw is None:
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
        texts = page.get("rec_texts") or []
        scores = page.get("rec_scores") or []
        polys = page.get("rec_polys") or page.get("dt_polys") or []
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

    Every compiled reflex template doubles as an icon. Recognising icons this
    way is free (no LLM tokens) and gives the planner stable named anchors.
    """

    def __init__(
        self,
        templates_dir: Path,
        threshold: float = 0.85,
        max_templates: int = 60,
    ) -> None:
        self.templates_dir = Path(templates_dir)
        self.threshold = threshold
        self.max_templates = max_templates

    def _template_files(self) -> list[Path]:
        files = sorted(self.templates_dir.glob("*.png")) if self.templates_dir.exists() else []
        return files[: self.max_templates]

    # --------------------------------------------------------------- public
    def find_icons(self, screen: np.ndarray) -> list[IconMatch]:
        """Return every saved template whose best match clears the threshold."""
        matches: list[IconMatch] = []
        if screen.size == 0:
            return matches
        gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

        for path in self._template_files():
            template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if template is None or template.size == 0:
                continue
            result = self._match_multi_scale(gray_screen, template)
            if result is None:
                continue
            (x, y), confidence, (tw, th) = result
            if confidence < self.threshold:
                continue
            matches.append(
                IconMatch(
                    name=path.stem,
                    bbox=BoundingBox(x, y, tw, th),
                    confidence=confidence,
                )
            )
        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches

    def _match_multi_scale(
        self, screen: np.ndarray, template: np.ndarray
    ) -> Optional[tuple[tuple[int, int], float, tuple[int, int]]]:
        if (
            template.shape[0] > screen.shape[0]
            or template.shape[1] > screen.shape[1]
        ):
            return None
        exact = self._match_once(screen, template)
        if exact[1] >= self.threshold:
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
