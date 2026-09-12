"""The central coordinator: routes commands and owns the fallback loop."""

from __future__ import annotations

import logging
import threading

from .agent import TaskAgent
from .brain import BrainPlanner, DeepSeekClient, GeminiClient
from .config import Settings
from .context import VisualContextManager
from .controller import PyAutoGuiInput
from .cost import UsageTracker, make_usage_callback
from .executor import PlanExecutor
from .memory import MemoryManager
from .ocr import IconMatcher, TextDetector
from .planner import TaskPlanner
from .screen import PyAutoGuiScreen
from .status import KillSwitch, StatusWindow
from .tasklog import TaskJournal
from .vision import VisionReflex

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class AgentOrchestrator:
    """Main loop implementing the compiled-reflex, fallback-loop pattern.

    Decision flow for a user command:

        1. Normalize the command into a cache key.
        2. Cache hit? Replay the stored reflex (milliseconds, zero tokens).
        3. Reflex failed, or the command is novel? Ask the brain to compile a
           new reflex from a screenshot, cache it, and retry once.
    """

    def __init__(
        self,
        memory: MemoryManager,
        vision: VisionReflex,
        brain: BrainPlanner,
    ) -> None:
        self._memory = memory
        self._vision = vision
        self._brain = brain

    def run(self, command: str) -> bool:
        """Execute a natural-language command. Returns True on success."""
        name = self._memory.normalize_name(command)
        skill = self._memory.get_skill(name)

        if skill is not None:
            logger.info("Cache hit for %r; replaying reflex.", name)
            print("Cache hit for %r; replaying reflex.", name)
            if self._vision.execute(skill):
                self._memory.record_success(name)
                print("True")
                return True
            logger.warning("Reflex failed for %r; triggering planner fallback.", name)
        else:
            print("Cache miss for %r; consulting planner.", name)
            
            logger.info("Cache miss for %r; consulting planner.", name)

        return self._plan_and_run(command, name)

    def _plan_and_run(self, command: str, name: str) -> bool:
        """Compile a fresh reflex via the brain and retry execution once."""
        try:
            skill = self._brain.plan(command, name)
        except Exception as exc:
            logger.exception("Planner failed for %r: %s", command, exc)
            return False

        if skill is None:
            print("no skill found")
            logger.warning(
                "Planner returned no skill for %r; no template was compiled and no reflex can be replayed.",
                command,
            )
            return False

        return self._vision.execute(skill)


def build_agent(settings: Settings | None = None) -> AgentOrchestrator:
    """Wire together the real (screen/input/model) components.

    ``agent = build_agent(); agent.run("Click the Export button")`` is the
    entire public API for production use.
    """
    settings = settings or Settings()
    settings.ensure_dirs()

    screen = PyAutoGuiScreen()
    controller = PyAutoGuiInput(
        pause=settings.input_pause,
        typing_interval=settings.typing_interval,
        teleport_cursor=settings.cursor_teleport,
        move_duration=settings.cursor_move_duration,
    )
    memory = MemoryManager(settings.memory_file)
    vision = VisionReflex(screen, controller, settings.confidence_threshold)

    llm = _make_llm(settings, None, None)
    brain = BrainPlanner(llm, screen, memory, settings)

    return AgentOrchestrator(memory, vision, brain)


def _make_llm(settings: Settings, model: str | None, usage_callback):
    """Build the requested provider, using DeepSeek-flash for vision by default."""
    provider = settings.llm_provider
    requested_model = (model or "").lower()
    if provider not in {"auto", "deepseek", "gemini"}:
        raise ValueError(
            "FURTI_LLM_PROVIDER must be one of: auto, deepseek, gemini"
        )
    if provider == "deepseek":
        use_deepseek = True
    elif provider == "gemini":
        use_deepseek = False
    elif requested_model.startswith("deepseek"):
        use_deepseek = True
    elif requested_model.startswith("gemini"):
        use_deepseek = False
    else:
        # In auto mode prefer DeepSeek when its key is available because the
        # default deepseek-flash path accepts the image_url payload.
        use_deepseek = bool(settings.deepseek_api_key) or not settings.google_api_key
    if use_deepseek:
        if not settings.deepseek_api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY is required for the selected DeepSeek provider."
            )
        return DeepSeekClient(
            settings,
            model=model or None,
            usage_callback=usage_callback,
        )
    if not settings.google_api_key:
        raise ValueError(
            "GOOGLE_API_KEY is required for the selected Gemini provider."
        )
    return GeminiClient(
        settings,
        model=model or None,
        usage_callback=usage_callback,
    )


def build_task_agent(settings: Settings | None = None) -> TaskAgent:
    """Wire the full multi-step, transparent task pipeline.

    ``agent = build_task_agent(); agent.run_task("Open the notes app and ...")``
    plans in steps, asks for console confirmation, executes with the live
    status window, and writes a ``<task_name>.md`` report with token/cost data.

    The legacy :func:`build_agent` single-command reflex path is unchanged.
    """
    settings = settings or Settings()
    settings.ensure_dirs()

    stop_event = threading.Event()
    usage = UsageTracker()
    usage_cb = make_usage_callback(usage)

    fast_model = settings.fast_model or None
    smart_model = settings.smart_model or None

    # Build the fast client first: it feeds the visual gate, the planner and
    # the step verifier. The smart client is only consulted on escalation.
    fast = _make_llm(settings, fast_model, usage_cb)
    smart = _make_llm(settings, smart_model, usage_cb) if smart_model else None

    screen = PyAutoGuiScreen()
    controller = PyAutoGuiInput(
        pause=settings.input_pause,
        teleport_cursor=settings.cursor_teleport,
        move_duration=settings.cursor_move_duration,
        typing_interval=settings.typing_interval,
    )
    memory = MemoryManager(settings.memory_file)
    vision = VisionReflex(screen, controller, settings.confidence_threshold)

    journal = TaskJournal("task", settings.reports_dir)
    window = StatusWindow()
    journal.status_sink = window.post  # console transparency -> Tk mirror
    journal.system(
        f"OCR setup: {'enabled' if settings.ocr_enabled else 'disabled'} "
        f"(max_dim={settings.ocr_max_dim}, mkldnn={settings.ocr_enable_mkldnn})."
    )
    text_detector = TextDetector(
        lang=settings.ocr_lang,
        enabled=settings.ocr_enabled,
        enable_mkldnn=settings.ocr_enable_mkldnn,
        max_dim=settings.ocr_max_dim,
    )
    icon_matcher = IconMatcher(
        settings.templates_dir, threshold=settings.icon_match_threshold
    )
    context = VisualContextManager(
        screen,
        text_detector,
        icon_matcher,
        settings,
        llm=fast,
        stop_event=stop_event,
        journal=journal,
    )

    planner = TaskPlanner(fast, settings, journal, context, stop_event, smart)
    executor = PlanExecutor(
        settings, journal, vision, controller, memory, planner, context,
        stop_event, task_slug="task",
    )
    kill_switch = KillSwitch(settings.kill_hotkey, stop_event)

    return TaskAgent(
        settings, journal, planner, executor, usage, window, kill_switch, stop_event
    )
