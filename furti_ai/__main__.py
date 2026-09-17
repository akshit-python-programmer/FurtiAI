"""Offline demo of the compiled-reflex, fallback-loop architecture.

Run with::

    python -m furti_ai

The demo uses a synthetic screen and a mock LLM, so it performs zero real
clicks and needs no API key. It walks through:

  1. Novel task  -> planner compiles a reflex -> reflex executes.
  2. Cached task -> reflex executes with no LLM call.
  3. UI changed  -> cached reflex fails -> fallback re-plans -> retry wins.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Optional

import cv2
import numpy as np

from .brain import BrainPlanner, MockLLMClient
from .config import Settings
from .memory import MemoryManager
from .orchestrator import AgentOrchestrator, build_task_agent
from .vision import VisionReflex

BLUE = (180, 90, 30)  # BGR
GREEN = (70, 170, 60)  # BGR
WHITE = (255, 255, 255)


class FakeScreen:
    """Deterministic synthetic screen so the demo performs no real capture."""

    def __init__(self, size: tuple[int, int] = (900, 700)) -> None:
        self.size = size
        self.buttons: list[dict[str, Any]] = []

    def add_button(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        color: tuple[int, int, int],
        label: str,
    ) -> None:
        self.buttons.append(
            {"x": x, "y": y, "w": w, "h": h, "color": color, "label": label}
        )

    def clear(self) -> None:
        self.buttons.clear()

    def capture(self) -> np.ndarray:
        width, height = self.size
        image = np.full((height, width, 3), 245, dtype=np.uint8)
        for b in self.buttons:
            x, y, w, h = b["x"], b["y"], b["w"], b["h"]
            cv2.rectangle(image, (x, y), (x + w, y + h), b["color"], -1)
            cv2.putText(
                image,
                b["label"],
                (x + 20, y + h // 2 + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                WHITE,
                2,
            )
        return image


class FakeInput:
    """Records clicks instead of moving the real mouse."""

    def __init__(self) -> None:
        self.clicks: list[tuple[int, int]] = []
        self.drags: list[tuple[tuple[int, int], tuple[int, int]]] = []

    def click(self, x: int, y: int, button: str = "left") -> None:
        self.clicks.append((x, y))

    def double_click(self, x: int, y: int) -> None:
        self.clicks.append((x, y))

    def right_click(self, x: int, y: int) -> None:
        self.clicks.append((x, y))

    def drag(
        self,
        x: int,
        y: int,
        end_x: int,
        end_y: int,
        button: str = "left",
        duration: Optional[float] = None,
        hold_keys: Optional[list[str]] = None,
    ) -> None:
        self.drags.append(((x, y), (end_x, end_y)))

    def type_text(self, text: str) -> None:
        pass

    def press_key(self, key: str, presses: int = 1) -> None:
        pass

    def scroll(self, clicks: int) -> None:
        pass


def scene_plan(screen: FakeScreen) -> Callable[[str], dict[str, Any]]:
    """Mock planner that always targets the (single) button on the screen."""

    def _plan(_prompt: str) -> dict[str, Any]:
        b = screen.buttons[0]
        return {
            "action": "click",
            "bbox": {"x": b["x"], "y": b["y"], "width": b["w"], "height": b["h"]},
            "description": f"Click the {b['label']} button",
        }

    return _plan


def simulate() -> None:
    logging.basicConfig(
        level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s"
    )

    with TemporaryDirectory() as tmp:
        settings = Settings(
            workspace=Path(tmp),
            # Tightened so the "UI changed" failure case is deterministic:
            # only a near-identical crop (1.0) passes.
            confidence_threshold=0.99,
        )
        settings.ensure_dirs()

        screen = FakeScreen()
        controller = FakeInput()
        memory = MemoryManager(settings.memory_file)
        llm = MockLLMClient(plan_fn=scene_plan(screen))
        vision = VisionReflex(screen, controller, settings.confidence_threshold)
        brain = BrainPlanner(llm, screen, memory, settings)
        agent = AgentOrchestrator(memory, vision, brain)

        command = "Click the Export button"

        print("\n=== Run 1: novel task (planner compiles a reflex) ===")
        screen.add_button(320, 260, 160, 80, BLUE, "Export")
        ok = agent.run(command)
        print(f"success={ok}  llm_calls={llm.call_count}  clicks={controller.clicks}")

        print("\n=== Run 2: cached skill (reflex only, zero LLM calls) ===")
        ok = agent.run(command)
        print(f"success={ok}  llm_calls={llm.call_count}  clicks={controller.clicks}")

        print("\n=== Run 3: UI changed (reflex fails -> fallback re-plans -> retry) ===")
        screen.clear()
        screen.add_button(500, 150, 200, 100, GREEN, "Export")
        ok = agent.run(command)
        print(f"success={ok}  llm_calls={llm.call_count}  clicks={controller.clicks}")

        cached = memory.get_skill(command)
        print(f"\nSkills in memory: {[s.name for s in memory.list_skills()]}")
        print(f"Cached skill metadata: {cached.metadata if cached else None}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python -m furti_ai",
        description="Furti AI desktop automation agent.",
    )
    parser.add_argument(
        "--task",
        metavar="INSTRUCTION",
        help=(
            "Run the multi-step task pipeline for INSTRUCTION: transparent "
            "planning, console confirmation, execution with the status "
            "window, and a <task_name>.md report with token/cost tracking."
        ),
    )
    args = parser.parse_args()

    if args.task:
        try:
            agent = build_task_agent()
        except ValueError as exc:
            print(f"fatal: {exc}", file=sys.stderr)
            raise SystemExit(1)
        ok = agent.run_task(args.task)
        raise SystemExit(0 if ok else 1)
    else:
        # No arguments: keep the original offline reflex demo (no API key,
        # no real clicks).
        simulate()
