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


class ActionType(str, Enum):
    """The set of low-level actions a compiled reflex can perform."""

    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    TYPE = "type"
    SCROLL = "scroll"
    KEY_PRESS = "key_press"


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

        try:
            action = ActionType(raw_action)
        except ValueError:
            # Convert arbitrary invalid model strings into the safest legal
            # action rather than aborting the whole command loop.
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
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_used: Optional[str] = None

    def record_success(self) -> None:
        """Bump usage counters after a reflex fires successfully."""
        self.success_count += 1
        self.last_used = datetime.now(timezone.utc).isoformat()

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
            created_at=data.get("created_at", ""),
            last_used=data.get("last_used"),
        )
