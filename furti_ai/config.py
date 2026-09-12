"""Central configuration for Furti AI.

Everything a module needs to know (paths, thresholds, model endpoint) lives
here, so swapping models or storage locations is a one-line change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


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

    confidence_threshold: float = 0.9

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
        default_factory=lambda: float(os.getenv("FURTI_CURSOR_MOVE_DURATION", "0.25"))
    )
    # PyAutoGUI's pause is applied after each low-level input call. Keep it
    # short so steps do not feel artificially serialized.
    input_pause: float = field(
        default_factory=lambda: float(os.getenv("FURTI_INPUT_PAUSE", "0.03"))
    )
    # A small per-character interval lets Windows applications consume the
    # keyboard event queue without making normal text entry feel sluggish.
    typing_interval: float = field(
        default_factory=lambda: float(os.getenv("FURTI_TYPING_INTERVAL", "0.02"))
    )

    # Tiered reasoning: fast model for routine calls, smart model escalation
    # for complex/failed reasoning. Empty smart_model means "no escalation".
    fast_model: str = field(
        default_factory=lambda: os.getenv("FURTI_FAST_MODEL", "")
    )
    smart_model: str = field(
        default_factory=lambda: os.getenv("FURTI_SMART_MODEL", "")
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
        default_factory=lambda: int(os.getenv("FURTI_MAX_LLM_CALLS", "30"))
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
    # Visual pipeline: PaddleOCR text + saved-template icon recognition.
    # ------------------------------------------------------------------
    ocr_enabled: bool = field(
        default_factory=lambda: os.getenv("FURTI_OCR_ENABLED", "true").lower()
        not in {"0", "false", "no", "off"}
    )
    ocr_lang: str = field(default_factory=lambda: os.getenv("FURTI_OCR_LANG", "en"))
    ocr_max_dim: int = field(
        default_factory=lambda: int(os.getenv("FURTI_OCR_MAX_DIM", "960"))
    )
    # PaddlePaddle's oneDNN path can fail on some CPU/model combinations.
    # Keep it opt-in so OCR remains reliable on the supported desktop setup.
    ocr_enable_mkldnn: bool = field(
        default_factory=lambda: os.getenv("FURTI_OCR_MKLDNN", "false").lower()
        in {"1", "true", "yes", "on"}
    )
    icon_match_threshold: float = field(
        default_factory=lambda: float(os.getenv("FURTI_ICON_THRESHOLD", "0.85"))
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
    # Optional LLM self-check after each step. Off by default to save tokens.
    verify_steps: bool = field(
        default_factory=lambda: os.getenv("FURTI_VERIFY_STEPS", "false").lower()
        in {"1", "true", "yes", "on"}
    )

    deepseek_api_key: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", "")
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
        default_factory=lambda: os.getenv("GOOGLE_API_KEY", "")
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

    def ensure_dirs(self) -> None:
        """Create the on-disk layout if it does not exist yet."""
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
