"""Transparency layer: console logging + per-task markdown report.

Every thought the LLM produces and every action the agent takes flows through
:class:`TaskJournal`. Each event is:

* printed to the console with a ``[KIND]`` prefix and a timestamp,
* forwarded to the always-on-top status window (when enabled), and
* appended to a chronological list that is written out as ``<task_name>.md``
  in the workspace ``reports`` directory when the task finishes.

This is the single source of truth for "what is the agent thinking/doing
right now?" -- the user can watch the console or the Tk window and read the
full replay afterwards in the markdown file.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from .models import ActionPlan, ActionType


@dataclass
class LogEvent:
    """A single recorded line of the agent's internal state."""

    kind: str
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%H:%M:%S")
    )


class TaskJournal:
    """Thread-safe console + file journal for one task run."""

    def __init__(self, task_name: str, reports_dir: Path) -> None:
        self.task_name = task_name
        self.reports_dir = Path(reports_dir)
        self.instruction: str = ""
        self.plan_snapshot: Optional[dict[str, Any]] = None
        self.plan_revisions: list[dict[str, Any]] = []
        self.step_results: list[dict[str, Any]] = []
        self.reflexes_compiled: list[str] = []
        self.cost_lines: list[str] = []
        self.status = "not started"
        self.finished_at: str = ""
        self._events: list[LogEvent] = []
        self._lock = threading.Lock()
        # Optional sink for the Tk status window (set once at wiring time).
        self.status_sink: Optional[Callable[[dict[str, Any]], None]] = None
        self._last_message = ""

    # ------------------------------------------------------------- recording
    def record(
        self,
        kind: str,
        message: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log one event everywhere: console, status window, file buffer."""
        event = LogEvent(
            kind=kind,
            message=message,
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._events.append(event)
            self._last_message = message
        print(f"[{event.timestamp}] [{kind}] {message}", flush=True)
        if self.status_sink is not None:
            try:
                phase = {
                    "THOUGHT": "AI thinking",
                    "PLAN": "planning",
                    "MODEL": "AI thinking",
                    "WAIT": "waiting for AI response",
                    "AI_OUTPUT": "AI response received",
                    "SCREENSHOT": "screen captured",
                    "STEP": "executing",
                    "ACTION": "executing",
                    "CONFIRM": "action confirmed",
                    "WARN": "recovering",
                    "ERROR": "error",
                }.get(kind)
                self.status_sink(
                    {
                        "event_kind": kind,
                        "message": message,
                        "timestamp": event.timestamp,
                        "task": self.instruction or self.task_name,
                        "phase": phase,
                        **event.metadata,
                    }
                )
            except Exception:  # status UI must never break the agent
                pass

    # Convenience shorthands used across the pipeline.
    def thought(self, message: str) -> None:
        self.record("THOUGHT", message)

    def plan(self, message: str) -> None:
        self.record("PLAN", message)

    def step(self, message: str) -> None:
        self.record("STEP", message)

    def action(self, message: str, signal: str = "trying") -> None:
        self.record(
            "ACTION",
            message,
            metadata={"current_action": message, "action_signal": signal},
        )

    def confirm(self, message: str) -> None:
        """Record that an input action returned without an execution error."""
        self.record(
            "CONFIRM",
            message,
            metadata={"action_signal": "confirmed"},
        )

    def waiting(self, message: str) -> None:
        """Record that execution is blocked on an external response."""
        self.record("WAIT", message)

    def screenshot(
        self,
        captured_at: str,
        captured_epoch: float,
        frame_shape: tuple[int, ...],
        ocr_count: int,
        icon_count: int,
    ) -> None:
        """Publish a visible signal whenever a new screen frame is captured."""
        height = int(frame_shape[0]) if frame_shape else 0
        width = int(frame_shape[1]) if len(frame_shape) > 1 else 0
        message = (
            f"Captured screenshot at {captured_at} "
            f"({width}x{height}; OCR={ocr_count}, icons={icon_count})"
        )
        self.record(
            "SCREENSHOT",
            message,
            metadata={
                "last_screenshot_at": captured_at,
                "last_screenshot_epoch": float(captured_epoch),
                "screenshot_signal": "fresh",
            },
        )

    def ai_output(
        self,
        output: Any,
        model: str,
        purpose: str,
        max_chars: int = 4000,
    ) -> None:
        """Show the latest raw model response without flooding the UI."""
        text = str(output).strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...[output truncated for display]"
        label = f"{model or 'AI'} [{purpose or 'response'}]"
        self.record(
            "AI_OUTPUT",
            f"{label}:\n{text or '(empty response)'}",
            metadata={
                "ai_output": text or "(empty response)",
                "ai_model": model or "AI",
                "ai_purpose": purpose or "response",
            },
        )

    def reflex(self, message: str) -> None:
        self.record("REFLEX", message)

    def model(self, message: str) -> None:
        self.record("MODEL", message)

    def cost(self, message: str) -> None:
        self.record("COST", message)

    def warn(self, message: str) -> None:
        self.record("WARN", message)

    def error(self, message: str) -> None:
        self.record("ERROR", message)

    def system(self, message: str) -> None:
        self.record("SYSTEM", message)

    @property
    def last_message(self) -> str:
        with self._lock:
            return self._last_message

    # ---------------------------------------------------------------- report
    def report_path(self) -> Path:
        safe = (
            self.task_name.replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
            .strip("_")
            or "task"
        )
        return self.reports_dir / f"{safe}.md"

    def write_report(self, cost_summary: Any) -> Path:
        """Write ``<task_name>.md`` with the plan, steps, logs and cost."""
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        path = self.report_path()
        with self._lock:
            events = list(self._events)

        lines: list[str] = []
        lines.append(f"# Task report: {self.task_name}")
        lines.append("")
        lines.append(f"- **Status:** {self.status}")
        lines.append(f"- **Started:** {events[0].timestamp if events else 'n/a'}")
        lines.append(f"- **Finished:** {self.finished_at or 'n/a'}")
        lines.append(f"- **Instruction:** {self.instruction}")
        lines.append("")

        if self.plan_snapshot:
            lines.append("## Plan")
            lines.append("")
            lines.append(f"> Goal: {self.plan_snapshot.get('goal', self.instruction)}")
            lines.append("")
            for step in self.plan_snapshot.get("steps", []):
                action = step.get("action", "?")
                desc = step.get("description", "")
                target = step.get("target") or (
                    step.get("bbox") and _bbox_text(step["bbox"])
                )
                lines.append(f"- **Step {step.get('step', '?')}:** {desc}")
                lines.append(f"  - action: `{action}`"
                             + (f", target: {target}" if target else ""))
            lines.append("")

        if self.plan_revisions:
            lines.append("## Adaptive plan revisions")
            lines.append("")
            for revision in self.plan_revisions:
                lines.append(
                    f"- After step {revision.get('after_step')}: "
                    f"{revision.get('reason', 'route changed')}"
                )
                for step in revision.get("steps", []):
                    lines.append(
                        f"  - **Step {step.get('step', '?')}:** "
                        f"{step.get('description', '')} "
                        f"[{step.get('action', '?')}]"
                    )
            lines.append("")

        lines.append("## Step results")
        lines.append("")
        for result in self.step_results:
            if result.get("superseded"):
                mark = "RECOVERED"
            else:
                mark = "SUCCESS" if result.get("success") else "FAILED"
            lines.append(
                f"- **Step {result.get('step')}** [{mark}] {result.get('description', '')}"
            )
            if "action_dispatched" in result:
                lines.append(
                    "  - action dispatched: "
                    f"`{bool(result.get('action_dispatched'))}`"
                )
            if result.get("visually_verified") is not None:
                lines.append(
                    "  - visually verified: "
                    f"`{bool(result.get('visually_verified'))}`"
                )
            for note in result.get("notes", []):
                lines.append(f"  - {note}")
        lines.append("")

        if self.reflexes_compiled:
            lines.append("## Reflexes compiled (reusable skills)")
            lines.append("")
            for reflex in self.reflexes_compiled:
                lines.append(f"- `{reflex}`")
            lines.append("")

        lines.append("## Execution log")
        lines.append("")
        for event in events:
            lines.append(f"`[{event.timestamp}]` **[{event.kind}]** {event.message}")
        lines.append("")

        lines.append("## Usage & approximate cost")
        lines.append("")
        for cost_line in cost_summary.lines():
            lines.append(f"- {cost_line}")
        lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")
        return path


def _bbox_text(bbox: dict[str, Any]) -> str:
    return (
        f"bbox ({bbox.get('x')}, {bbox.get('y')}, "
        f"{bbox.get('width')}x{bbox.get('height')})"
    )


def plan_to_dict(
    task_name: str, goal: str, steps: list[ActionPlan], notes: str = ""
) -> dict[str, Any]:
    """Snapshot a :class:`TaskPlan` into the JSON the report renders."""
    step_dicts: list[dict[str, Any]] = []
    for index, step in enumerate(steps, start=1):
        entry: dict[str, Any] = {
            "step": index,
            "description": step.description,
            "action": step.action.value,
        }
        if step.bbox is not None:
            entry["bbox"] = {
                "x": step.bbox.x,
                "y": step.bbox.y,
                "width": step.bbox.width,
                "height": step.bbox.height,
            }
            if getattr(step, "bbox_space", ""):
                entry["bbox_coordinate_space"] = step.bbox_space
        if step.text:
            entry["text"] = step.text
        if step.target:
            entry["target"] = step.target
        if step.thought:
            entry["thought"] = step.thought
        if step.params:
            entry["params"] = step.params
        step_dicts.append(entry)
    payload: dict[str, Any] = {
        "task": task_name,
        "goal": goal,
        "notes": notes,
        "steps": step_dicts,
    }
    return payload
