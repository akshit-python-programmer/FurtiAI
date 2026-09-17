"""Central configuration for Furti AI.

Everything a module needs to know (paths, thresholds, model endpoint) lives
here, so swapping models or storage locations is a one-line change.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from pathlib import Path


def _local_key(name: str) -> str:
    """Read one API key from the ignored, repository-local ``keys.json``.

    ``FURTI_KEYS_FILE`` is authoritative when it is set: pointing it at a
    missing or unreadable file must never silently fall back to a different key
    file, because the caller asked for that specific one.
    """
    configured_path = os.getenv("FURTI_KEYS_FILE", "").strip()
    path = Path(configured_path) if configured_path else Path.cwd() / "keys.json"
    if not path.is_file():
        return ""
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    value = payload.get(name, "")
    return str(value).strip() if value else ""


def _secret(name: str, environment_name: str) -> str:
    """Prefer environment secrets, then the ignored local key file."""
    return os.getenv(environment_name, "").strip() or _local_key(name)


@dataclass
class Settings:
    """Runtime settings for the agent.

    Attributes:
        confidence_threshold: Minimum ``cv2.matchTemplate`` confidence before a
            reflex is allowed to act. Anything lower triggers the fallback.
        workspace: Root directory for skills, templates, and the memory file.
        deepseek_api_key / deepseek_base_url / deepseek_model: Connection
            details for the OpenAI-compatible reasoning endpoint.
    """

    confidence_threshold: float = field(
        default_factory=lambda: float(
            os.getenv("FURTI_CONFIDENCE_THRESHOLD", "0.9")
        )
    )

    workspace: Path = field(
        default_factory=lambda: Path(
            os.getenv("FURTI_WORKSPACE", str(Path.home() / ".furti_ai"))
        )
    )

    # Cursor behaviour: when False (default) the cursor *moves* to the target
    # over ``cursor_move_duration`` seconds (visible, demo-friendly) instead of
    # teleporting instantly. Set True for the old instant behaviour.
    cursor_teleport: bool = field(
        default_factory=lambda: os.getenv("FURTI_CURSOR_TELEPORT", "false").lower()
        in {"1", "true", "yes", "on"}
    )
    cursor_move_duration: float = field(
        default_factory=lambda: float(os.getenv("FURTI_CURSOR_MOVE_DURATION", "0.5"))
    )
    # PyAutoGUI's pause is applied after each low-level input call. Keep it
    # short so steps do not feel artificially serialized.
    input_pause: float = field(
        default_factory=lambda: float(os.getenv("FURTI_INPUT_PAUSE", "0.03"))
    )
    # A small per-character interval lets Windows applications consume the
    # keyboard event queue without making normal text entry feel sluggish.
    typing_interval: float = field(
        default_factory=lambda: float(os.getenv("FURTI_TYPING_INTERVAL", "0.1"))
    )
    # A drag must be slow enough for the OS drag-and-drop machinery to see the
    # intermediate mouse moves; long distances are given a little more time.
    drag_duration: float = field(
        default_factory=lambda: float(os.getenv("FURTI_DRAG_DURATION", "0.4"))
    )
    # Pause between the cursor arriving at a point and the button press. Without
    # it the click races the application's WM_MOUSEMOVE handling and is either
    # dropped or delivered to the previously-hovered control.
    click_settle: float = field(
        default_factory=lambda: float(os.getenv("FURTI_CLICK_SETTLE", "0.12"))
    )
    # pyautogui aborts automation while the pointer sits exactly on a screen
    # corner. That is a deliberate human escape hatch, but the agent also
    # targets corners (the Start button) and a stray coordinate can park the
    # cursor there, which used to freeze all input. The controller suspends the
    # trip for a deliberate corner target and releases an accidental park;
    # turning this off removes the corner trip entirely so the kill hotkey is
    # the only abort path.
    failsafe: bool = field(
        default_factory=lambda: os.getenv("FURTI_FAILSAFE", "true").lower()
        not in {"0", "false", "no", "off"}
    )
    # Before keyboard/scroll actions, force the intended window into the
    # foreground via the OS (SetForegroundWindow) so keys land in the right
    # application instead of Furti's own status window.
    focus_intended_window: bool = field(
        default_factory=lambda: os.getenv("FURTI_FOCUS_WINDOW", "true").lower()
        not in {"0", "false", "no", "off"}
    )
    # Ask the model endpoint for JSON-only output (DeepSeek's
    # ``response_format``, Gemini's ``response_mime_type``). The control payload
    # is parsed strictly either way; this only stops the answer from drifting
    # into prose, and endpoints that reject the hint are retried without it.
    llm_json_mode: bool = field(
        default_factory=lambda: os.getenv("FURTI_LLM_JSON_MODE", "true").lower()
        not in {"0", "false", "no", "off"}
    )
    # High-level actions are never chained blindly: after an action that can
    # change the UI (a click, drag or keystroke) the next interaction waits out
    # this long so the application can finish rendering. Windows, menus and web
    # pages routinely need 200-500 ms before the next target even exists, and a
    # click that arrives mid-render is silently dropped.
    render_delay: float = field(
        default_factory=lambda: float(os.getenv("FURTI_RENDER_DELAY", "0.35"))
    )
    # Before acting on a resolved target, ask which window owns that pixel. If a
    # dialog/popup (or Furti's own always-on-top readout) covers it, dismiss the
    # obstruction first instead of clicking through it.
    dismiss_obstructions: bool = field(
        default_factory=lambda: os.getenv(
            "FURTI_DISMISS_OBSTRUCTIONS", "true"
        ).lower()
        not in {"0", "false", "no", "off"}
    )
    max_obstruction_dismissals: int = field(
        default_factory=lambda: int(os.getenv("FURTI_MAX_OBSTRUCTION_DISMISSALS", "2"))
    )

    # Tiered reasoning: fast model for routine calls, smart model escalation
    # for complex/failed reasoning. Empty smart_model means "no escalation".
    fast_model: str = field(
        default_factory=lambda: os.getenv("FURTI_FAST_MODEL", "")
    )
    smart_model: str = field(
        default_factory=lambda: os.getenv("FURTI_SMART_MODEL", "")
    )
    # DeepSeek-only escalation target, consulted only after the cheap flash
    # model *failed* (unparsable plan, exhausted step retries, adaptive
    # re-planning). It defaults to the same flash model with thinking mode on:
    # the deeper reasoning is available when something went wrong without paying
    # a slower model on every routine call. Point this at a distinct
    # thinking-capable model name if the endpoint exposes one.
    deepseek_reasoning_model: str = field(
        default_factory=lambda: os.getenv(
            "DEEPSEEK_REASONING_MODEL", "deepseek-flash"
        )
    )
    # Thinking mode for that escalation client only. The routine clients keep
    # tools/function calling, which thinking endpoints reject.
    deepseek_escalation_thinking: bool = field(
        default_factory=lambda: os.getenv(
            "FURTI_DEEPSEEK_ESCALATION_THINKING", "true"
        ).lower()
        not in {"0", "false", "no", "off"}
    )

    # ------------------------------------------------------------------
    # Multi-step task guardrails (anti-endless-loop / token budget).
    # ------------------------------------------------------------------
    max_plan_steps: int = field(
        default_factory=lambda: int(os.getenv("FURTI_MAX_PLAN_STEPS", "10"))
    )
    max_step_retries: int = field(
        default_factory=lambda: int(os.getenv("FURTI_MAX_STEP_RETRIES", "2"))
    )
    max_plan_replans: int = field(
        default_factory=lambda: int(os.getenv("FURTI_MAX_PLAN_REPLANS", "3"))
    )
    max_llm_calls_per_task: int = field(
        default_factory=lambda: int(os.getenv("FURTI_MAX_LLM_CALLS", "100"))
    )
    max_consecutive_failures: int = field(
        default_factory=lambda: int(os.getenv("FURTI_MAX_FAILURES", "3"))
    )
    # Minimum seconds between two screen captures. A hard throttle so the
    # agent can never spam screenshots every second and burn tokens.
    screenshot_min_interval: float = field(
        default_factory=lambda: float(os.getenv("FURTI_SCREENSHOT_INTERVAL", "4.0"))
    )
    # Screenshots sent to the model are downscaled to this max dimension to
    # keep image token cost low.
    max_image_dim: int = field(
        default_factory=lambda: int(os.getenv("FURTI_MAX_IMAGE_DIM", "1280"))
    )

    # ------------------------------------------------------------------
    # Fast visual pipeline: RapidOCR text + cached saved-template icons.
    # ------------------------------------------------------------------
    ocr_backend: str = field(
        default_factory=lambda: os.getenv("FURTI_OCR_BACKEND", "rapidocr").lower()
    )
    ocr_enabled: bool = field(
        default_factory=lambda: os.getenv("FURTI_OCR_ENABLED", "true").lower()
        not in {"0", "false", "no", "off"}
    )
    ocr_lang: str = field(default_factory=lambda: os.getenv("FURTI_OCR_LANG", "en"))
    ocr_max_dim: int = field(
        default_factory=lambda: int(os.getenv("FURTI_OCR_MAX_DIM", "640"))
    )
    ocr_min_confidence: float = field(
        default_factory=lambda: float(os.getenv("FURTI_OCR_MIN_CONFIDENCE", "0.35"))
    )
    ocr_max_lines: int = field(
        default_factory=lambda: int(os.getenv("FURTI_OCR_MAX_LINES", "80"))
    )
    # PaddlePaddle's oneDNN path can fail on some CPU/model combinations.
    # It is only relevant when the legacy Paddle backend is selected.
    ocr_enable_mkldnn: bool = field(
        default_factory=lambda: os.getenv("FURTI_OCR_MKLDNN", "false").lower()
        in {"1", "true", "yes", "on"}
    )
    icon_match_threshold: float = field(
        default_factory=lambda: float(os.getenv("FURTI_ICON_THRESHOLD", "0.85"))
    )
    icon_max_templates: int = field(
        default_factory=lambda: int(os.getenv("FURTI_ICON_MAX_TEMPLATES", "24"))
    )
    icon_match_max_dim: int = field(
        default_factory=lambda: int(os.getenv("FURTI_ICON_MAX_DIM", "1280"))
    )
    icon_match_multiscale: bool = field(
        default_factory=lambda: os.getenv("FURTI_ICON_MULTISCALE", "false").lower()
        in {"1", "true", "yes", "on"}
    )

    # ------------------------------------------------------------------
    # Live feedback / control.
    # ------------------------------------------------------------------
    enable_status_window: bool = field(
        default_factory=lambda: os.getenv("FURTI_STATUS_WINDOW", "true").lower()
        not in {"0", "false", "no", "off"}
    )
    # Global hotkey that aborts the running task (pynput format).
    kill_hotkey: str = field(
        default_factory=lambda: os.getenv("FURTI_KILL_HOTKEY", "<ctrl>+<alt>+k")
    )
    # One combined LLM review after each dispatched step verifies the current
    # result and whether the next step is ready. The model also selects
    # text/visual evidence for that review.
    verify_steps: bool = field(
        default_factory=lambda: os.getenv("FURTI_VERIFY_STEPS", "true").lower()
        not in {"0", "false", "no", "off"}
    )
    # A second, independent model call audits the *route* after each dispatched
    # step ("is the agent still on track?"). It is what catches a popup being
    # mistaken for page content or an action landing on the wrong control.
    verify_progress: bool = field(
        default_factory=lambda: os.getenv("FURTI_VERIFY_PROGRESS", "true").lower()
        not in {"0", "false", "no", "off"}
    )
    # Run the step review and the route monitor at the same time. With two
    # DeepSeek keys they use separate clients, so the pair costs the slower call
    # instead of the sum of both.
    parallel_verify: bool = field(
        default_factory=lambda: os.getenv("FURTI_PARALLEL_VERIFY", "true").lower()
        not in {"0", "false", "no", "off"}
    )

    # ------------------------------------------------------------------
    # Direct tools: high-level actions that call the OS instead of driving
    # the mouse (launch_app, run_command, write_file, clipboard, window
    # state, ...). They skip screen capture, OCR and the verification call,
    # so a task that used to take a dozen GUI steps finishes in one.
    # ------------------------------------------------------------------
    direct_tools: bool = field(
        default_factory=lambda: os.getenv("FURTI_DIRECT_TOOLS", "true").lower()
        not in {"0", "false", "no", "off"}
    )
    # Shell execution is the most powerful tool, so it can be switched off
    # independently of the harmless ones (clipboard, window focus, ...).
    allow_shell_commands: bool = field(
        default_factory=lambda: os.getenv(
            "FURTI_ALLOW_SHELL_COMMANDS", "true"
        ).lower()
        not in {"0", "false", "no", "off"}
    )
    # Irreversible commands (disk format, force delete, DROP TABLE, ...) are
    # refused unless this is enabled or the step passes params.confirm=true.
    allow_destructive_commands: bool = field(
        default_factory=lambda: os.getenv(
            "FURTI_ALLOW_DESTRUCTIVE_COMMANDS", "false"
        ).lower()
        in {"1", "true", "yes", "on"}
    )
    # Ceiling for run_command, so a hung command cannot stall the task.
    tool_timeout: float = field(
        default_factory=lambda: float(os.getenv("FURTI_TOOL_TIMEOUT", "60"))
    )
    # How much tool output (stdout, file excerpt, window list) is surfaced.
    tool_max_output_chars: int = field(
        default_factory=lambda: int(os.getenv("FURTI_TOOL_MAX_OUTPUT", "4000"))
    )
    # Default image format for the screenshot tool.
    screenshot_format: str = field(
        default_factory=lambda: os.getenv("FURTI_SCREENSHOT_FORMAT", "png").lower()
    )

    # ------------------------------------------------------------------
    # User / system profile ("context file").
    # A persisted description of the machine and its owner: folders, drives,
    # installed applications and CLI tools, plus notes the user wants the
    # planner to know. Injected into planning prompts so the model stops
    # guessing at paths and app names.
    # ------------------------------------------------------------------
    profile_enabled: bool = field(
        default_factory=lambda: os.getenv("FURTI_PROFILE", "true").lower()
        not in {"0", "false", "no", "off"}
    )
    # Re-detect the environment when the stored profile is older than this.
    profile_max_age_days: float = field(
        default_factory=lambda: float(os.getenv("FURTI_PROFILE_MAX_AGE_DAYS", "7"))
    )
    # Bound the advertised application list so the prompt stays small.
    profile_max_apps: int = field(
        default_factory=lambda: int(os.getenv("FURTI_PROFILE_MAX_APPS", "40"))
    )
    # Free-form facts about the user, e.g.
    # FURTI_USER_NOTES="invoices live in D:\invoices; always save drafts to Desktop".
    user_notes: str = field(
        default_factory=lambda: os.getenv("FURTI_USER_NOTES", "")
    )

    # ------------------------------------------------------------------
    # Reflex quality gate. A reflex is only compiled when it is both
    # successful *and* plausibly reusable; a blurry or one-off crop costs a
    # template file and future false cache hits.
    # ------------------------------------------------------------------
    reflex_enabled: bool = field(
        default_factory=lambda: os.getenv("FURTI_REFLEX", "true").lower()
        not in {"0", "false", "no", "off"}
    )
    # Minimum anchor confidence reported by OCR/icon/template matching.
    reflex_min_anchor_confidence: float = field(
        default_factory=lambda: float(os.getenv("FURTI_REFLEX_MIN_CONFIDENCE", "0.75"))
    )
    # Templates thinner than this in either dimension are pixel noise.
    reflex_min_template_side: int = field(
        default_factory=lambda: int(os.getenv("FURTI_REFLEX_MIN_TEMPLATE_SIDE", "6"))
    )
    # A crop covering most of the screen can never anchor anything useful.
    reflex_max_template_area_ratio: float = field(
        default_factory=lambda: float(os.getenv("FURTI_REFLEX_MAX_AREA_RATIO", "0.6"))
    )
    # The description is the cache key, so it must carry real meaning.
    reflex_min_description_chars: int = field(
        default_factory=lambda: int(os.getenv("FURTI_REFLEX_MIN_DESCRIPTION", "8"))
    )
    # Replaying a reflex re-types the same text; a long unique payload (an
    # email body, a paragraph) is a one-off, not a reusable reflex.
    reflex_max_typed_chars: int = field(
        default_factory=lambda: int(os.getenv("FURTI_REFLEX_MAX_TYPED", "160"))
    )
    # A reflex that fails this many replays is retired instead of retried.
    reflex_retire_failures: int = field(
        default_factory=lambda: int(os.getenv("FURTI_REFLEX_RETIRE_FAILURES", "3"))
    )
    # Let the model re-align a stale reflex on the live screen before the
    # full planner is asked to route around it.
    reflex_realign: bool = field(
        default_factory=lambda: os.getenv("FURTI_REFLEX_REALIGN", "true").lower()
        not in {"0", "false", "no", "off"}
    )

    # ------------------------------------------------------------------
    # Cross-provider verification ("second opinion").
    # Critical steps are reviewed by a *different* provider with its own API
    # key before dispatch, so a single model deviating cannot act alone.
    # ------------------------------------------------------------------
    cross_verify: bool = field(
        default_factory=lambda: os.getenv("FURTI_CROSS_VERIFY", "true").lower()
        not in {"0", "false", "no", "off"}
    )
    # `auto` picks the provider that is *not* the primary one.
    cross_verify_provider: str = field(
        default_factory=lambda: os.getenv("FURTI_CROSS_VERIFY_PROVIDER", "auto").lower()
    )
    cross_verify_model: str = field(
        default_factory=lambda: os.getenv("FURTI_CROSS_VERIFY_MODEL", "")
    )
    # Hard ceiling so verification can never dominate the token budget.
    cross_verify_max_calls: int = field(
        default_factory=lambda: int(os.getenv("FURTI_CROSS_VERIFY_MAX_CALLS", "12"))
    )
    # Also review the finished plan before the user approves it.
    cross_verify_plans: bool = field(
        default_factory=lambda: os.getenv("FURTI_CROSS_VERIFY_PLANS", "true").lower()
        not in {"0", "false", "no", "off"}
    )
    # When the verifier rejects a step, retry it once with the verifier's
    # suggested safer alternative before failing the step.
    cross_verify_apply_alternative: bool = field(
        default_factory=lambda: os.getenv(
            "FURTI_CROSS_VERIFY_APPLY_ALTERNATIVE", "true"
        ).lower()
        not in {"0", "false", "no", "off"}
    )

    deepseek_api_key: str = field(
        default_factory=lambda: _secret("deepseek_api_key", "DEEPSEEK_API_KEY")
    )
    deepseek_api_key_2: str = field(
        default_factory=lambda: _secret("deepseek_api_key_2", "DEEPSEEK_API_KEY_2")
    )
    deepseek_base_url: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )
    # DeepSeek-flash is used by default for the task pipeline because this
    # OpenAI-compatible path accepts the image_url payload used by chat_vision.
    deepseek_model: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
    )
    llm_provider: str = field(
        default_factory=lambda: os.getenv("FURTI_LLM_PROVIDER", "auto").lower()
    )

    google_api_key: str = field(
        default_factory=lambda: _secret("google_api_key", "GOOGLE_API_KEY")
    )
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    )
    

    # Thinking-mode endpoints (for example DeepSeek reasoner models) do not
    # support the same tool_choice/function-calling contract as standard
    # chat-completion endpoints. The default is False so the code remains
    # compatible with older OpenAI-style models.
    deepseek_thinking_mode: bool = field(
        default_factory=lambda: os.getenv("DEEPSEEK_THINKING_MODE", "false").lower()
        in {"1", "true", "yes", "on"}
    )

    # When turned off, the client will simply ask the model for a JSON plan in
    # the user content, then parse the model's answer as structured JSON.
    deepseek_use_function_calling: bool = field(
        default_factory=lambda: os.getenv("DEEPSEEK_USE_FUNCTION_CALLING", "true").lower()
        not in {"0", "false", "no", "off"}
    )

    # Gemini function calling: when enabled, chat_with_vision requests the
    # `plan_action` function the same way the DeepSeek client does (mode=ANY
    # with an allowed function name), instead of accepting free-form JSON.
    gemini_use_function_calling: bool = field(
        default_factory=lambda: os.getenv("GEMINI_USE_FUNCTION_CALLING", "true").lower()
        not in {"0", "false", "no", "off"}
    )

    # ------------------------------------------------------------------
    # Derived on-disk layout.
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        # Tolerate plain strings (e.g. Settings(workspace="C:\\foo")).
        self.workspace = Path(self.workspace)

    @property
    def skills_dir(self) -> Path:
        """Reserved for future per-skill sidecar files (e.g. YAML bundles)."""
        return self.workspace / "skills"

    @property
    def templates_dir(self) -> Path:
        """Directory holding the cropped template images for each skill.

        These crops double as the icon library that the visual pipeline
        pattern-matches against the screen before consulting the LLM.
        """
        return self.workspace / "templates"

    @property
    def reports_dir(self) -> Path:
        """Directory where per-task ``<task_name>.md`` reports are written."""
        return self.workspace / "reports"

    @property
    def memory_file(self) -> Path:
        """JSON file backing the :class:`MemoryManager` cache."""
        return self.workspace / "memory.json"

    @property
    def screenshots_dir(self) -> Path:
        """Directory for screenshots the agent takes (full screen or a region)."""
        return self.workspace / "screenshots"

    @property
    def context_dir(self) -> Path:
        """Directory holding the persisted user/system profile ("context file")."""
        return self.workspace / "context"

    @property
    def profile_file(self) -> Path:
        """JSON file with the detected user/system facts."""
        return self.context_dir / "profile.json"

    @property
    def profile_report(self) -> Path:
        """Human-readable mirror of the profile, for the user to edit."""
        return self.context_dir / "profile.md"

    def ensure_dirs(self) -> None:
        """Create the on-disk layout if it does not exist yet."""
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        self.context_dir.mkdir(parents=True, exist_ok=True)
