"""Token usage tracking and approximate API cost estimation.

Every LLM call reports its model and its prompt/completion token counts here.
At the end of a task the accumulated usage is converted to a rough USD cost
using a standard per-million-token price table (approximate list prices, not
a live quote -- see :data:`MODEL_PRICES`).

Prices are *approximate* and model-dependent. They are matched by substring
against the model name (``flash``, ``pro``, ``deepseek``, ...) so upgrading a
model version keeps the estimate roughly right. Override entries through the
``FURTI_MODEL_PRICES`` environment variable:

.. code-block:: powershell

    $env:FURTI_MODEL_PRICES = '{"gemini-3.6-flash": [0.10, 0.40]}'  # [in, out] USD per 1M tokens
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Approximate standard API list prices in USD per 1M tokens: [input, output].
# Google Gemini flash-tier / pro-tier and DeepSeek chat/reasoner entries are
# kept as representative defaults; unknown models fall back to a modest
# flash-tier estimate.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "gemini": (0.30, 2.50),          # generic Gemini fallback
    "flash": (0.30, 2.50),           # Gemini flash tier (approximate)
    "pro": (1.25, 10.00),            # Gemini pro tier (approximate)
    "ultra": (2.00, 12.00),          # Gemini ultra tier (approximate)
    "deepseek-reasoner": (0.55, 2.19),
    "deepseek-chat": (0.27, 1.10),
    "deepseek-flash": (0.10, 0.40),
}
DEFAULT_PRICE: tuple[float, float] = MODEL_PRICES["flash"]


def _env_price_overrides() -> dict[str, tuple[float, float]]:
    raw = os.getenv("FURTI_MODEL_PRICES", "")
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Ignoring malformed FURTI_MODEL_PRICES JSON: %r", raw)
        return {}
    out: dict[str, tuple[float, float]] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            try:
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    out[str(key)] = (float(value[0]), float(value[1]))
            except (TypeError, ValueError):
                continue
    return out


def price_for_model(model: str) -> tuple[float, float]:
    """Return the (input, output) USD-per-1M-token price for a model name."""
    merged = dict(MODEL_PRICES)
    merged.update(_env_price_overrides())

    lowered = (model or "").lower()
    for key in sorted(merged, key=len, reverse=True):
        if key in lowered:
            return merged[key]
    return DEFAULT_PRICE


@dataclass
class UsageEntry:
    """One recorded LLM call."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    purpose: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class CostSummary:
    """Aggregate usage and approximate spend for one task run."""

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_usd: float = 0.0
    calls: int = 0
    per_model: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    def lines(self) -> list[str]:
        """Human-readable breakdown lines for console/report output."""
        lines = [
            f"LLM calls: {self.calls}",
            f"Tokens: {self.total_prompt_tokens} prompt + "
            f"{self.total_completion_tokens} completion = {self.total_tokens}",
        ]
        for model, stats in self.per_model.items():
            lines.append(
                f"  {model}: {stats['calls']} call(s), {stats['prompt']} prompt + "
                f"{stats['completion']} completion tokens ~ ${stats['cost']:.4f}"
            )
        lines.append(f"Approximate total API cost: ${self.total_cost_usd:.4f} USD")
        return lines


class UsageTracker:
    """Thread-safe collector for per-call token usage."""

    def __init__(self) -> None:
        self._entries: list[UsageEntry] = []
        self._lock = threading.Lock()

    def record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        purpose: str = "",
    ) -> None:
        with self._lock:
            self._entries.append(
                UsageEntry(
                    model=model,
                    prompt_tokens=max(0, int(prompt_tokens)),
                    completion_tokens=max(0, int(completion_tokens)),
                    purpose=purpose,
                )
            )

    def summary(self) -> CostSummary:
        with self._lock:
            entries = list(self._entries)

        summary = CostSummary(calls=len(entries))
        for entry in entries:
            in_price, out_price = price_for_model(entry.model)
            cost = (
                entry.prompt_tokens * in_price + entry.completion_tokens * out_price
            ) / 1_000_000
            summary.total_prompt_tokens += entry.prompt_tokens
            summary.total_completion_tokens += entry.completion_tokens
            summary.total_cost_usd += cost
            stats = summary.per_model.setdefault(
                entry.model,
                {"calls": 0, "prompt": 0, "completion": 0, "cost": 0.0},
            )
            stats["calls"] += 1
            stats["prompt"] += entry.prompt_tokens
            stats["completion"] += entry.completion_tokens
            stats["cost"] += cost
        return summary

    def call_count(self) -> int:
        with self._lock:
            return len(self._entries)


def make_usage_callback(tracker: Optional[UsageTracker]):
    """Build an ``LLMClient.usage_callback`` compatible with our clients.

    Returns ``None`` when no tracker is configured so the clients can skip
    the bookkeeping entirely.
    """
    if tracker is None:
        return None

    def _record(model: str, prompt_tokens: int, completion_tokens: int, purpose: str = "") -> None:
        tracker.record(model, prompt_tokens, completion_tokens, purpose)

    return _record
