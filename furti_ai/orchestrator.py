"""The central coordinator: routes commands and owns the fallback loop."""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

from .agent import TaskAgent, UserChoiceCallback
from .brain import BrainPlanner, DeepSeekClient, GeminiClient
from .config import Settings
from .context import VisualContextManager
from .controller import PyAutoGuiInput
from .cost import UsageTracker, make_usage_callback
from .executor import PlanExecutor
from .memory import MemoryManager
from .ocr import IconMatcher, TextDetector
from .planner import TaskPlan, TaskPlanner
from .profile import build_user_context
from .reflex import ReflexRealigner
from .screen import PyAutoGuiScreen
from .status import KillSwitch, StatusWindow
from .tasklog import TaskJournal
from .verifier import build_cross_verifier
from .vision import VisionReflex

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

#: Provider label used when the reviewer runs on the *second* DeepSeek key.
SECONDARY_DEEPSEEK_PROVIDER = "deepseek-secondary"


class AgentOrchestrator:
    """Main loop implementing the compiled-reflex, fallback-loop pattern.

    Decision flow for a user command:

        1. Normalize the command into a cache key.
        2. Cache hit? Replay the stored reflex (milliseconds, zero tokens).
        3. Reflex failed? Hand it back to the LLM to be *re-aligned* on the live
           screen and replayed once more; retire it if it has become useless.
        4. Still failing, or the command is novel? Ask the brain to compile a
           new reflex from a screenshot, cache it, and retry once.
    """

    def __init__(
        self,
        memory: MemoryManager,
        vision: VisionReflex,
        brain: BrainPlanner,
        realigner: ReflexRealigner | None = None,
        journal: TaskJournal | None = None,
    ) -> None:
        self._memory = memory
        self._vision = vision
        self._brain = brain
        self._realigner = realigner
        self._journal = journal

    def run(self, command: str) -> bool:
        """Execute a natural-language command. Returns True on success."""
        name = self._memory.normalize_name(command)
        # Reflexes are compiled disabled, so replay must ask for an enabled one
        # explicitly; a stored-but-disabled reflex falls through to the planner.
        skill = self._memory.get_skill(name, require_enabled=True)

        if skill is None and self._memory.has_skill(name):
            logger.info(
                "Reflex %r is stored but disabled; consulting the planner.", name
            )
            print("Reflex %r is stored but disabled; consulting the planner." % name)

        if skill is not None:
            logger.info("Cache hit for %r; replaying reflex.", name)
            print("Cache hit for %r; replaying reflex.", name)
            if self._vision.execute(skill):
                self._memory.record_success(name)
                print("True")
                return True
            logger.warning("Reflex failed for %r; re-aligning it.", name)
            if self._handle_reflex_failure(skill) is not None:
                print("True")
                return True

        else:
            print("Cache miss for %r; consulting planner.", name)

            logger.info("Cache miss for %r; consulting planner.", name)

        return self._plan_and_run(command, name)

    def _handle_reflex_failure(self, skill) -> Any:
        """Count the miss, retire an unusable reflex, else re-align it.

        Returns the repaired skill when the re-aligned reflex replays
        successfully, and ``None`` when the caller should re-plan instead.
        """
        record_failure = getattr(skill, "record_failure", None)
        if callable(record_failure):
            skill.record_failure()
        save = getattr(self._memory, "save_skill", None)
        if callable(save):
            save(skill)

        realigner = self._realigner
        if realigner is None or not getattr(realigner, "enabled", False):
            logger.info(
                "No reflex re-aligner available; falling back to the planner."
            )
            return None
        if realigner.is_useless(skill):
            realigner.retire(skill, "it kept failing to anchor")
            return None

        result = realigner.realign(skill, "the stored template did not match")
        if not result.ok or result.skill is None:
            logger.warning("Re-alignment did not help for %r.", skill.name)
            return None
        # The template was rewritten in place: replay it once.
        if self._vision.execute(result.skill):
            self._memory.record_success(skill.name)
            return result.skill
        return None

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
        drag_duration=settings.drag_duration,
        click_settle=settings.click_settle,
        failsafe=settings.failsafe,
    )
    memory = MemoryManager(settings.memory_file)
    vision = VisionReflex(screen, controller, settings.confidence_threshold)

    llm = _make_llm(settings, None, None)
    brain = BrainPlanner(llm, screen, memory, settings)
    realigner = ReflexRealigner(settings, llm, memory, screen)

    return AgentOrchestrator(memory, vision, brain, realigner)


def build_secondary_llm(
    settings: Settings,
    primary_model: str = "",
    journal: TaskJournal | None = None,
) -> tuple[Any, str]:
    """Build the *other* provider's client for cross-verification.

    Independence is the whole point: a reviewer running on the same provider (or
    worse, the same model) shares the planner's failure modes. This returns the
    provider that the primary model is *not*, with its own API key, and an empty
    client when that key is missing so verification simply stays off.

    Two DeepSeek keys (``deepseek_api_key`` + ``deepseek_api_key_2``) are also
    supported: the primary key keeps planning while the reviewer gets its own
    client and its own rate limit, which is what makes the parallel
    step-review/route-monitor pair possible on one provider. The reviewer stays
    on the fast model because it runs after every dispatched step; the deeper
    thinking pass is the escalation client, used only after a failure.
    """
    requested = str(getattr(settings, "cross_verify_provider", "auto") or "auto").lower()
    model = str(getattr(settings, "cross_verify_model", "") or "") or None
    secondary_key = str(getattr(settings, "deepseek_api_key_2", "") or "").strip()

    if requested in {"auto", ""}:
        primary = detect_primary_provider(settings, primary_model)
        if primary == "deepseek" and secondary_key:
            if journal is not None:
                journal.system(
                    "Parallel verification is on: the reviewer uses the second "
                    "DeepSeek key while the primary key plans."
                )
            return (
                DeepSeekClient(
                    settings,
                    # The reviewer runs on *every* dispatched step, so it stays on
                    # the fast model: the deep thinking brain is reserved for the
                    # escalation path after a failure.
                    model=model or settings.deepseek_model or None,
                    api_key=secondary_key,
                    thinking=False,
                ),
                SECONDARY_DEEPSEEK_PROVIDER,
            )
        requested = "gemini" if primary == "deepseek" else "deepseek"

    if requested == "gemini":
        if not settings.google_api_key:
            if journal is not None:
                journal.system(
                    "Cross-verification is off: no GOOGLE_API_KEY for the "
                    "secondary (Gemini) provider."
                )
            return None, ""
        return GeminiClient(settings, model=model), "gemini"

    key = secondary_key or settings.deepseek_api_key
    if not key:
        if journal is not None:
            journal.system(
                "Cross-verification is off: no DEEPSEEK_API_KEY for the "
                "secondary (DeepSeek) provider."
            )
        return None, ""
    return DeepSeekClient(settings, model=model, api_key=key), "deepseek"


def detect_primary_provider(settings: Settings, requested_model: str = "") -> str:
    """Which provider the planning model belongs to (``deepseek``/``gemini``)."""
    provider = str(getattr(settings, "llm_provider", "auto") or "auto").lower()
    if provider in {"deepseek", "gemini"}:
        return provider
    requested = str(requested_model or "").lower()
    if requested.startswith("gemini"):
        return "gemini"
    if requested.startswith("deepseek"):
        return "deepseek"
    # Mirrors _make_llm's auto rule: DeepSeek wins when its key is present.
    if settings.deepseek_api_key:
        return "deepseek"
    if settings.google_api_key:
        return "gemini"
    return "deepseek"


def _deepseek_client_kwargs(
    model: str | None,
    usage_callback: Any,
    thinking: bool | None,
) -> dict[str, Any]:
    """Keyword arguments for :class:`DeepSeekClient`.

    ``thinking`` is only forwarded when it is explicitly set: the default means
    "follow the settings", and leaving it out keeps minimal duck-typed clients
    (used in tests and by embedders) working unchanged.
    """
    kwargs: dict[str, Any] = {
        "model": model or None,
        "usage_callback": usage_callback,
    }
    if thinking is not None:
        kwargs["thinking"] = thinking
    return kwargs


def _make_llm(
    settings: Settings,
    model: str | None,
    usage_callback,
    thinking: bool | None = None,
):
    """Build the requested provider, using DeepSeek-flash for vision by default.

    ``thinking`` overrides thinking mode for this client alone, which is how the
    escalation brain is built: same flash model, deeper reasoning, tools omitted.
    """
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
            **_deepseek_client_kwargs(model, usage_callback, thinking),
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


def build_task_agent(
    settings: Settings | None = None,
    *,
    stop_event: threading.Event | None = None,
    status_sink: Callable[[dict[str, Any]], None] | None = None,
    confirmation_callback: Callable[[TaskPlan], bool | str | None] | None = None,
    user_choice_callback: UserChoiceCallback | None = None,
    manage_status_window: bool = True,
) -> TaskAgent:
    """Wire the full multi-step, transparent task pipeline.

    ``agent = build_task_agent(); agent.run_task("Open the notes app and ...")``
    plans in steps, asks for console confirmation by default (or a supplied
    UI callback), executes with the live status window, and writes a
    ``<task_name>.md`` report with token/cost data.

    The legacy :func:`build_agent` single-command reflex path is unchanged.
    """
    settings = settings or Settings()
    settings.ensure_dirs()

    task_stop_event = stop_event if stop_event is not None else threading.Event()
    usage = UsageTracker()
    usage_cb = make_usage_callback(usage)

    fast_model = settings.fast_model or None
    smart_model = settings.smart_model or None
    if not smart_model and detect_primary_provider(settings, fast_model or "") == "deepseek":
        # Escalation is DeepSeek flash in thinking mode. It is consulted only
        # after a failure (unparsable plan, exhausted step retries, re-planning),
        # so the deep pass never slows down the routine path.
        smart_model = settings.deepseek_reasoning_model or settings.deepseek_model

    # Build the fast client first: it feeds the visual gate, the planner and
    # the step verifier. The smart client is only consulted on escalation.
    fast = _make_llm(settings, fast_model, usage_cb)
    smart = (
        _make_llm(
            settings,
            smart_model,
            usage_cb,
            thinking=settings.deepseek_escalation_thinking,
        )
        if smart_model
        else None
    )

    screen = PyAutoGuiScreen()
    controller = PyAutoGuiInput(
        pause=settings.input_pause,
        teleport_cursor=settings.cursor_teleport,
        move_duration=settings.cursor_move_duration,
        typing_interval=settings.typing_interval,
        drag_duration=settings.drag_duration,
        click_settle=settings.click_settle,
        failsafe=settings.failsafe,
    )
    memory = MemoryManager(settings.memory_file)
    vision = VisionReflex(screen, controller, settings.confidence_threshold)

    journal = TaskJournal("task", settings.reports_dir)
    # Corner recoveries must reach the user, not just the console.
    controller.warn = journal.warn
    window = StatusWindow()
    journal.status_sink = status_sink or window.post

    # The user/system context file: detected once, refreshed when stale, and
    # injected into every planning prompt so the model stops guessing at paths,
    # folder locations and installed applications.
    profile = build_user_context(settings, journal)

    # Independent second opinion on critical steps, from the other provider.
    verifier = build_cross_verifier(
        settings,
        primary_model=fast_model or settings.deepseek_model or "",
        journal=journal,
        profile=profile,
    )
    if verifier.enabled:
        journal.system(
            f"Cross-verification is on: critical steps are reviewed by "
            f"{verifier.provider}/{verifier.model or 'default'} before dispatch "
            f"(budget {settings.cross_verify_max_calls} call(s) per task)."
        )

    text_detector = TextDetector(
        lang=settings.ocr_lang,
        enabled=settings.ocr_enabled,
        enable_mkldnn=settings.ocr_enable_mkldnn,
        max_dim=settings.ocr_max_dim,
        backend=settings.ocr_backend,
        min_confidence=settings.ocr_min_confidence,
        max_lines=settings.ocr_max_lines,
    )
    # Report the backend that actually loaded, not the one that was requested:
    # a request that silently fails leaves the executor without text anchors.
    journal.system(
        f"{text_detector.describe()} "
        f"(max_dim={settings.ocr_max_dim}, "
        f"min_conf={settings.ocr_min_confidence:.2f}, "
        f"max_lines={settings.ocr_max_lines}); "
        f"icons(max_templates={settings.icon_max_templates}, "
        f"max_dim={settings.icon_match_max_dim}, "
        f"multiscale={settings.icon_match_multiscale})."
    )
    icon_matcher = IconMatcher(
        settings.templates_dir,
        threshold=settings.icon_match_threshold,
        max_templates=settings.icon_max_templates,
        max_screen_dim=settings.icon_match_max_dim,
        multi_scale=settings.icon_match_multiscale,
    )
    context = VisualContextManager(
        screen,
        text_detector,
        icon_matcher,
        settings,
        llm=fast,
        stop_event=task_stop_event,
        journal=journal,
    )

    planner = TaskPlanner(
        fast, settings, journal, context, task_stop_event, smart, profile=profile
    )
    executor = PlanExecutor(
        settings, journal, vision, controller, memory, planner, context,
        task_stop_event, task_slug="task", verifier=verifier,
    )
    kill_switch = KillSwitch(settings.kill_hotkey, task_stop_event)

    return TaskAgent(
        settings,
        journal,
        planner,
        executor,
        usage,
        window,
        kill_switch,
        task_stop_event,
        confirmation_callback=confirmation_callback,
        user_choice_callback=user_choice_callback,
        manage_status_window=manage_status_window,
        profile=profile,
        verifier=verifier,
    )
