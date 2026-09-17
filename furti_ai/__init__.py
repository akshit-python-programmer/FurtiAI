"""Furti AI -- a desktop automation agent built on compiled reflexes.

The system separates high-latency reasoning (a model-backed "brain") from
low-latency execution (an OpenCV "reflex"). LLM plans are compiled once into
local template skills, cached by :class:`MemoryManager`, and replayed in
milliseconds by :class:`VisionReflex`. The :class:`AgentOrchestrator` wires the
fallback loop together.

The multi-step task pipeline (:class:`TaskAgent`) extends this with
transparent planning, RapidOCR + cached icon grounding, a status window, a kill
hotkey, token/cost tracking and per-task ``<task_name>.md`` reports.
"""

from .agent import TaskAgent
from .brain import BrainPlanner, DeepSeekClient, GeminiClient, LLMClient, MockLLMClient
from .config import Settings
from .context import SceneObservation, TaskAborted, VisualContextManager
from .controller import InputController, PyAutoGuiInput
from .cost import CostSummary, UsageTracker
from .executor import ExecutionReport, PlanExecutor, StepResult
from .memory import MemoryManager
from .models import ActionPlan, ActionType, BoundingBox, Skill, coerce_action
from .ocr import IconMatch, IconMatcher, TextDetector, TextLine
from .orchestrator import AgentOrchestrator, build_agent, build_task_agent
from .planner import PlanStep, TaskPlan, TaskPlanner
from .screen import PyAutoGuiScreen, ScreenCapture
from .status import KillSwitch, StatusWindow
from .tasklog import TaskJournal
from .vision import VisionReflex
from .windows import find_window, focus_window, list_windows, set_process_dpi_aware

__all__ = [
    "ActionPlan",
    "ActionType",
    "AgentOrchestrator",
    "BoundingBox",
    "BrainPlanner",
    "coerce_action",
    "CostSummary",
    "DeepSeekClient",
    "ExecutionReport",
    "GeminiClient",
    "IconMatch",
    "IconMatcher",
    "InputController",
    "KillSwitch",
    "LLMClient",
    "MemoryManager",
    "MockLLMClient",
    "PlanExecutor",
    "PlanStep",
    "PyAutoGuiInput",
    "PyAutoGuiScreen",
    "SceneObservation",
    "ScreenCapture",
    "Settings",
    "Skill",
    "StatusWindow",
    "StepResult",
    "TaskAborted",
    "TaskAgent",
    "TaskJournal",
    "TaskPlan",
    "TaskPlanner",
    "TextDetector",
    "TextLine",
    "UsageTracker",
    "VisionReflex",
    "VisualContextManager",
    "build_agent",
    "build_task_agent",
    "find_window",
    "focus_window",
    "list_windows",
    "set_process_dpi_aware",
]
