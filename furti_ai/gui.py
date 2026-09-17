"""Tkinter application for running Furti AI tasks.

The application owns the Tk event loop on the main thread. Planning and
execution run in a worker thread, while journal events and plan-confirmation
requests cross the thread boundary through a queue.
"""

from __future__ import annotations

import queue
import threading
import sys
import tkinter.font as tkfont
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from .config import Settings
from .models import UserChoiceRequest
from .memory import MemoryManager
from .orchestrator import build_agent, build_task_agent
from .planner import TaskPlan
from .status import StatusWindow


BG = "#11161d"
PANEL = "#1a222c"
PANEL_LIGHT = "#222d39"
TEXT = "#e7edf2"
MUTED = "#9aa9b5"
ACCENT = "#7fd1ff"
GREEN = "#9ee6a8"
AMBER = "#ffd479"
RED = "#ff9c9c"


@dataclass
class ConfirmationRequest:
    """A plan waiting for a decision from the Tk thread."""

    plan: TaskPlan
    event: threading.Event
    decision: bool | str | None = None

    @classmethod
    def for_plan(cls, plan: TaskPlan) -> "ConfirmationRequest":
        return cls(plan=plan, event=threading.Event())

    def resolve(self, decision: bool | str) -> None:
        """Resolve the request once; duplicate button clicks are ignored."""
        if self.event.is_set():
            return
        self.decision = decision
        self.event.set()


@dataclass
class UserChoiceResponse:
    """A model question waiting for a response from the Tk thread."""

    request: UserChoiceRequest
    event: threading.Event
    answer: str | None = None

    def resolve(self, answer: str | None) -> None:
        if self.event.is_set():
            return
        self.answer = answer.strip() if isinstance(answer, str) else None
        self.event.set()


class FurtiApp(tk.Tk):
    """Full desktop application for planning and executing Furti tasks."""

    POLL_MS = 100
    MAX_LOG_LINES = 800
    STATUS_LINGER_MS = 1500

    _BASIC_FIELDS = (
        "llm_provider",
        "deepseek_api_key",
        "deepseek_api_key_2",
        "google_api_key",
        "deepseek_model",
        "gemini_model",
        "fast_model",
        "smart_model",
        "deepseek_reasoning_model",
        "deepseek_escalation_thinking",
        "llm_json_mode",
        "workspace",
        "cursor_teleport",
        "cursor_move_duration",
        "typing_interval",
        "verify_steps",
        "verify_progress",
        "parallel_verify",
        "direct_tools",
        "allow_shell_commands",
        "reflex_enabled",
        "cross_verify",
    )
    _ADVANCED_FIELDS = (
        "confidence_threshold",
        "input_pause",
        "focus_intended_window",
        "render_delay",
        "dismiss_obstructions",
        "max_obstruction_dismissals",
        "drag_duration",
        "click_settle",
        "failsafe",
        "allow_destructive_commands",
        "tool_timeout",
        "tool_max_output_chars",
        "screenshot_format",
        "profile_enabled",
        "profile_max_age_days",
        "profile_max_apps",
        "user_notes",
        "reflex_min_anchor_confidence",
        "reflex_min_template_side",
        "reflex_max_template_area_ratio",
        "reflex_min_description_chars",
        "reflex_max_typed_chars",
        "reflex_retire_failures",
        "reflex_realign",
        "cross_verify_provider",
        "cross_verify_model",
        "cross_verify_max_calls",
        "cross_verify_plans",
        "cross_verify_apply_alternative",
        "gemini_use_function_calling",
        "max_plan_steps",
        "max_step_retries",
        "max_plan_replans",
        "max_llm_calls_per_task",
        "max_consecutive_failures",
        "screenshot_min_interval",
        "max_image_dim",
        "ocr_enabled",
        "ocr_backend",
        "ocr_lang",
        "ocr_max_dim",
        "ocr_min_confidence",
        "ocr_max_lines",
        "ocr_enable_mkldnn",
        "icon_match_threshold",
        "icon_max_templates",
        "icon_match_max_dim",
        "icon_match_multiscale",
        "enable_status_window",
        "kill_hotkey",
        "deepseek_base_url",
        "deepseek_thinking_mode",
        "deepseek_use_function_calling",
    )

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        initial_task: str = "",
        initial_mode: str = "task",
    ) -> None:
        super().__init__()
        self.title("Furti AI - Desktop Automation")
        self._set_window_icon(self)
        self.geometry("1360x900")
        self.minsize(1120, 760)
        self.configure(bg=BG)

        self._initial_settings = settings or Settings()
        self._setting_vars: dict[str, tk.Variable] = {}
        self._queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pending_confirmation: ConfirmationRequest | None = None
        self._pending_choice: UserChoiceResponse | None = None
        self._choice_dialog: tk.Toplevel | None = None
        self._running = False
        self._closing = False
        self._last_capture_epoch = 0.0
        self._last_capture_at = ""
        self._status_window: StatusWindow | None = None
        self._status_hide_id: Any = None
        self._icon_images: dict[str, tk.PhotoImage] = {}
        self._ui_font_family = "Segoe UI"
        self._mono_font_family = "Consolas"

        self._load_custom_fonts()
        self._configure_style()
        self._load_icons()
        self._create_setting_vars()
        self._build_widgets(initial_task, initial_mode)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(self.POLL_MS, self._poll_events)
        self.after(250, self._refresh_screenshot_age)

    # ------------------------------------------------------------ construction
    @staticmethod
    def _set_window_icon(window: Any) -> None:
        """Apply the repository ICO when the current Tk platform supports it."""
        icon_path = Path(__file__).resolve().parent.parent / "icon.ico"
        if not icon_path.is_file():
            return
        try:
            window.iconbitmap(default=str(icon_path))
        except (AttributeError, tk.TclError, OSError):
            pass

    def _load_custom_fonts(self) -> None:
        """Register local TTF/OTF fonts and select their Tk family names."""
        roots = (Path.cwd() / "fonts", Path(__file__).resolve().parent / "fonts")
        font_files = [
            path
            for root in roots
            if root.is_dir()
            for path in root.iterdir()
            if path.suffix.lower() in {".ttf", ".otf"}
        ]
        if not font_files:
            return
        before = set(tkfont.families(self))
        if sys.platform == "win32":
            try:
                import ctypes

                add_font = ctypes.windll.gdi32.AddFontResourceExW
                add_font.argtypes = [ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_void_p]
                add_font.restype = ctypes.c_int
                for path in font_files:
                    add_font(str(path), 0x10, None)  # FR_PRIVATE
            except (AttributeError, OSError):
                return
        after = set(tkfont.families(self))
        discovered = sorted(after - before)
        if not discovered:
            return
        # Prefer a family whose name resembles a font filename; otherwise use
        # the first newly registered family as the application typeface.
        stem_words = {
            word.lower()
            for path in font_files
            for word in path.stem.replace("-", " ").replace("_", " ").split()
        }
        preferred = next(
            (
                family
                for family in discovered
                if stem_words.intersection(family.lower().split())
            ),
            discovered[0],
        )
        self._ui_font_family = preferred
        self._mono_font_family = preferred

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "TFrame",
            background=BG,
        )
        style.configure(
            "Panel.TFrame",
            background=PANEL,
        )
        style.configure(
            "TLabel",
            background=BG,
            foreground=TEXT,
        )
        style.configure(
            "Panel.TLabel",
            background=PANEL,
            foreground=TEXT,
        )
        style.configure(
            "Muted.Panel.TLabel",
            background=PANEL,
            foreground=MUTED,
        )
        style.configure(
            "TLabelframe",
            background=PANEL,
            foreground=TEXT,
        )
        style.configure(
            "TLabelframe.Label",
            background=PANEL,
            foreground=ACCENT,
        )
        style.configure(
            "TButton",
            padding=(12, 8),
            font=(self._ui_font_family, 11),
        )
        #: Narrow variant for buttons that share a row with a full-width one.
        style.configure(
            "Compact.TButton",
            padding=(2, 8),
            font=(self._ui_font_family, 11),
        )
        style.configure(
            "TNotebook.Tab",
            padding=(14, 9),
            font=(self._ui_font_family, 11, "bold"),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#263f50"), ("active", PANEL_LIGHT)],
            foreground=[("selected", "#ffffff"), ("active", TEXT)],
        )
        style.configure("TLabel", font=(self._ui_font_family, 11))
        style.configure("Panel.TLabel", font=(self._ui_font_family, 11))
        style.configure("Muted.Panel.TLabel", font=(self._ui_font_family, 10))
        style.configure(
            "TCheckbutton",
            font=(self._ui_font_family, 11),
        )
        style.configure(
            "Accent.TButton",
            background="#1e6b91",
            foreground="#ffffff",
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#2b88b5"), ("disabled", "#38434c")],
        )
        style.configure(
            "Danger.TButton",
            background="#7b2f35",
            foreground="#ffffff",
        )
        style.map(
            "Danger.TButton",
            background=[("active", "#a64249"), ("disabled", "#38434c")],
        )
        style.configure(
            "TCheckbutton",
            background=PANEL,
            foreground=TEXT,
        )
        style.map(
            "TCheckbutton",
            background=[("active", PANEL_LIGHT)],
        )
        style.configure(
            "TEntry",
            fieldbackground="#0d1218",
            foreground=TEXT,
            padding=6,
            font=(self._ui_font_family, 11),
        )
        style.configure(
            "TCombobox",
            fieldbackground="#0d1218",
            foreground=TEXT,
            padding=5,
            font=(self._ui_font_family, 11),
        )

    def _load_icons(self) -> None:
        """Load the supplied PNG assets without making them a startup trap."""
        roots = (Path.cwd() / "icons", Path(__file__).resolve().parent / "icons")
        files: dict[str, Path] = {}
        for root in roots:
            if root.is_dir():
                files.update({path.stem.lower(): path for path in root.glob("*.png")})
        if not files:
            return
        try:
            from PIL import Image, ImageTk
        except ImportError:
            return
        aliases = {
            "stop": "power-button",
            "play": "monitor-flash",
            "plan": "design-file-ai",
            "clear": "delete-2",
            "trash": "delete-2",
            "models": "design-file-ai",
            "brain": "design-file-ai",
            "input": "cursor-hand",
            "cursor": "cursor-hand",
            "safety": "monitor-warning",
            "shield": "check-badge",
            "vision": "picture-sun",
            "eye": "view-off",
            "advanced": "synchronize-arrow",
            "settings": "synchronize-arrow",
            "gear": "synchronize-arrow",
            "check": "check-square",
            "approve": "check-badge",
            "edit": "file-code-edit",
            "pencil": "file-code-edit",
            "x": "remove-bold",
            "close": "remove-bold",
            "decline": "remove-bold",
            "ai": "design-file-ai",
            "log": "analytics-graph-lines",
            "activity": "analytics-graph-lines",
        }
        for name, keyword in aliases.items():
            path = next((path for stem, path in files.items() if keyword in stem), None)
            if path is None:
                continue
            try:
                image = Image.open(path).convert("RGBA")
                image.thumbnail((22, 22), Image.Resampling.LANCZOS)
                self._icon_images[name] = ImageTk.PhotoImage(image, master=self)
            except Exception:
                continue

    def _icon(self, *names: str) -> tk.PhotoImage | None:
        for name in names:
            image = self._icon_images.get(name.lower())
            if image is not None:
                return image
        return None

    def _button_options(self, *icon_names: str) -> dict[str, Any]:
        image = self._icon(*icon_names)
        return {"image": image, "compound": "left"} if image is not None else {}

    @staticmethod
    def _add_tab(
        notebook: ttk.Notebook,
        child: ttk.Frame,
        title: str,
        image: tk.PhotoImage | None = None,
    ) -> None:
        """Add a tab using only options supported by the current Tk build."""
        options: dict[str, Any] = {"text": title}
        if image is not None:
            options.update(image=image, compound="left")
        notebook.add(child, **options)

    def _create_setting_vars(self) -> None:
        for setting in fields(Settings):
            value = getattr(self._initial_settings, setting.name)
            if isinstance(value, bool):
                variable: tk.Variable = tk.BooleanVar(self, value=value)
            else:
                variable = tk.StringVar(self, value=str(value))
            self._setting_vars[setting.name] = variable

    def _build_widgets(self, initial_task: str, initial_mode: str) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        header = tk.Frame(self, bg=BG)
        header.grid(row=0, column=0, sticky="ew", padx=18, pady=(14, 8))
        header.columnconfigure(1, weight=1)
        tk.Label(
            header,
            text="Furti AI",
            bg=BG,
            fg=ACCENT,
            font=(self._ui_font_family, 22, "bold"),
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            header,
            text="Transparent desktop automation with plan approval",
            bg=BG,
            fg=MUTED,
            font=(self._ui_font_family, 12),
        ).grid(row=1, column=0, sticky="w")
        self.status_badge = tk.Label(
            header,
            text="READY",
            bg="#214b35",
            fg=GREEN,
            font=(self._ui_font_family, 11, "bold"),
            padx=12,
            pady=5,
        )
        self.status_badge.grid(row=0, column=2, rowspan=2, padx=(12, 0))
        self.stop_button = ttk.Button(
            header,
            text="Stop",
            style="Danger.TButton",
            command=self._stop_task,
            state="disabled",
            **self._button_options("stop", "square", "x"),
        )
        self.stop_button.grid(row=0, column=3, rowspan=2, padx=(8, 0))

        body = tk.Frame(self, bg=BG)
        body.grid(row=1, column=0, sticky="nsew", padx=18)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = tk.Frame(body, bg=PANEL, width=365)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        self._build_task_panel(left, initial_task, initial_mode)
        self._build_plan_panel(right)
        # Live status, AI output and the event log share one tabbed pane, so the
        # proposed plan keeps the full height of the column.
        self._build_output_panels(right)

        footer = tk.Frame(self, bg=BG)
        footer.grid(row=2, column=0, sticky="ew", padx=18, pady=(8, 14))
        footer.columnconfigure(0, weight=1)
        self.report_var = tk.StringVar(value="Report: not started")
        tk.Label(
            footer,
            textvariable=self.report_var,
            bg=BG,
            fg=MUTED,
            anchor="w",
            font=(self._mono_font_family, 10),
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            footer,
            text=f"Kill hotkey: {self._initial_settings.kill_hotkey}",
            bg=BG,
            fg=MUTED,
            anchor="e",
            font=(self._mono_font_family, 10),
        ).grid(row=0, column=1, sticky="e")

    def _build_task_panel(
        self,
        parent: tk.Frame,
        initial_task: str,
        initial_mode: str,
    ) -> None:
        task_frame = ttk.LabelFrame(parent, text="Task", padding=10)
        task_frame.pack(fill="x", padx=10, pady=(10, 8))
        task_editor = ttk.Frame(task_frame, style="Panel.TFrame")
        task_editor.pack(fill="both", expand=True)
        task_editor.columnconfigure(0, weight=1)
        task_editor.rowconfigure(0, weight=1)
        self.task_input = tk.Text(
            task_editor,
            height=5,
            width=38,
            wrap="word",
            bg="#0d1218",
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            padx=8,
            pady=8,
            font=(self._ui_font_family, 12),
        )
        self.task_input.grid(row=0, column=0, sticky="nsew")
        task_scrollbar = ttk.Scrollbar(
            task_editor, orient="vertical", command=self.task_input.yview
        )
        task_scrollbar.grid(row=0, column=1, sticky="ns")
        self.task_input.configure(yscrollcommand=task_scrollbar.set)
        if initial_task:
            self.task_input.insert("1.0", initial_task)

        mode_row = ttk.Frame(task_frame, style="Panel.TFrame")
        mode_row.pack(fill="x", pady=(8, 0))
        ttk.Label(mode_row, text="Mode", style="Panel.TLabel").pack(side="left")
        self.mode_var = tk.StringVar(value=initial_mode if initial_mode in {"task", "reflex"} else "task")
        ttk.Combobox(
            mode_row,
            textvariable=self.mode_var,
            state="readonly",
            values=("task", "reflex"),
            width=13,
        ).pack(side="right")

        self.start_button = ttk.Button(
            task_frame,
            text="Plan task",
            style="Accent.TButton",
            command=self._start_task,
            **self._button_options("play", "plan", "sparkles"),
        )
        self.start_button.pack(fill="x", pady=(10, 0))

        # Clear and "clear the reflex cache" share one row at a 10:90 split.
        # The reflex button is deliberately compact (icon only when one is
        # available) because 10% of the panel is roughly one glyph wide; the
        # confirmation dialog spells out what it removes.
        actions_row = ttk.Frame(task_frame, style="Panel.TFrame")
        actions_row.pack(fill="x", pady=(6, 0))
        # A uniform group is what makes the split exact: plain weights only
        # share the space left over after each widget's requested width.
        actions_row.columnconfigure(0, weight=1, uniform="actions")
        actions_row.columnconfigure(1, weight=9, uniform="actions")

        reflex_icon = self._icon("trash", "clear", "x")
        reflex_kwargs: dict[str, Any] = (
            {"image": reflex_icon} if reflex_icon is not None else {"text": "Reflex"}
        )
        self.clear_cache_button = ttk.Button(
            actions_row,
            style="Compact.TButton",
            command=self._clear_cache,
            **reflex_kwargs,
        )
        self.clear_cache_button.grid(row=0, column=0, sticky="ew")
        self.clear_button = ttk.Button(
            actions_row,
            text="Clear",
            command=self._clear_task,
            **self._button_options("trash", "clear", "x"),
        )
        self.clear_button.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        notebook = ttk.Notebook(parent)
        notebook.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.settings_notebook = notebook

        model_tab = ttk.Frame(notebook, style="Panel.TFrame", padding=8)
        input_tab = ttk.Frame(notebook, style="Panel.TFrame", padding=8)
        safety_tab = ttk.Frame(notebook, style="Panel.TFrame", padding=8)
        vision_tab = ttk.Frame(notebook, style="Panel.TFrame", padding=8)
        advanced_tab = ttk.Frame(notebook, style="Panel.TFrame", padding=8)
        context_tab = ttk.Frame(notebook, style="Panel.TFrame", padding=8)
        reflex_tab = ttk.Frame(notebook, style="Panel.TFrame", padding=8)
        self._tabbed_setting_names = {
            "llm_provider", "deepseek_api_key", "google_api_key",
            "deepseek_api_key_2",
            "deepseek_model", "gemini_model", "fast_model", "smart_model",
            "deepseek_reasoning_model", "deepseek_escalation_thinking",
            "llm_json_mode",
            "user_notes",
            "workspace", "cursor_teleport", "cursor_move_duration",
            "typing_interval", "input_pause", "focus_intended_window",
            "drag_duration", "click_settle", "verify_steps", "direct_tools",
            "render_delay", "dismiss_obstructions", "max_obstruction_dismissals",
            "allow_shell_commands", "allow_destructive_commands", "cross_verify",
            "reflex_enabled", "cross_verify_provider", "cross_verify_model",
            "confidence_threshold", "screenshot_format", "max_image_dim",
            "ocr_enabled", "ocr_backend", "ocr_lang", "ocr_max_dim",
            "ocr_min_confidence", "ocr_max_lines", "icon_match_threshold",
            "icon_max_templates", "icon_match_max_dim", "icon_match_multiscale",
        }
        for tab in (model_tab, input_tab, safety_tab, vision_tab, advanced_tab, context_tab, reflex_tab):
            tab.columnconfigure(0, weight=1)
            tab.rowconfigure(0, weight=1)
        self._add_tab(notebook, model_tab, "Models", self._icon("models", "brain"))
        self._add_tab(notebook, input_tab, "Input", self._icon("input", "cursor"))
        self._add_tab(notebook, safety_tab, "Safety", self._icon("safety", "shield"))
        self._add_tab(notebook, vision_tab, "Vision", self._icon("vision", "eye"))
        self._add_tab(
            notebook,
            advanced_tab,
            "Advanced",
            self._icon("advanced", "settings", "gear"),
        )
        self._add_tab(notebook, context_tab, "AI Context", self._icon("ai", "brain"))
        self._add_tab(notebook, reflex_tab, "Reflexes", self._icon("check", "activity"))

        self._build_settings_group(
            model_tab,
            "Providers and models",
            (
                "llm_provider", "deepseek_api_key", "google_api_key",
                "deepseek_model", "gemini_model", "fast_model", "smart_model",
                "deepseek_reasoning_model", "deepseek_escalation_thinking",
                "llm_json_mode",
            ),
        )
        self._build_settings_group(
            input_tab,
            "Workspace and input behavior",
            (
                "workspace", "cursor_teleport", "cursor_move_duration",
                "typing_interval", "input_pause", "focus_intended_window",
                "render_delay", "drag_duration", "click_settle", "failsafe",
            ),
        )
        self._build_settings_group(
            safety_tab,
            "Execution guardrails",
            (
                "verify_steps", "direct_tools", "allow_shell_commands",
                "verify_progress", "parallel_verify",
                "dismiss_obstructions", "max_obstruction_dismissals",
                "allow_destructive_commands", "cross_verify",
                "reflex_enabled",
                "cross_verify_provider", "cross_verify_model",
            ),
        )
        self._build_settings_group(
            vision_tab,
            "Screen grounding and local matching",
            (
                "confidence_threshold", "screenshot_format", "max_image_dim",
                "ocr_enabled", "ocr_backend", "ocr_lang", "ocr_max_dim",
                "ocr_min_confidence", "ocr_max_lines", "icon_match_threshold",
                "icon_max_templates", "icon_match_max_dim", "icon_match_multiscale",
            ),
        )
        self._build_advanced_panel(advanced_tab)
        self._build_context_panel(context_tab)
        self._build_reflex_panel(reflex_tab)

    def _build_settings_group(
        self,
        parent: ttk.Frame,
        title: str,
        names: tuple[str, ...],
    ) -> None:
        canvas = tk.Canvas(parent, bg=PANEL, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas, style="Panel.TFrame")
        inner.columnconfigure(1, weight=1)
        inner.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(window_id, width=event.width),
        )
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        frame = ttk.LabelFrame(inner, text=title, padding=8)
        frame.grid(row=0, column=0, sticky="new")
        frame.columnconfigure(1, weight=1)
        for row, name in enumerate(names):
            self._add_setting_control(frame, row, name)

    def _build_reflex_panel(self, parent: ttk.Frame) -> None:
        """List the compiled reflexes so they can be inspected and toggled.

        New reflexes are compiled disabled, so this tab is where one gets opted
        in; the master switch is the same setting as the Safety tab's
        ``reflex_enabled`` (both widgets share one variable and stay in sync).
        """
        header = ttk.LabelFrame(parent, text="Reflex cache", padding=10)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        self.reflex_summary_var = tk.StringVar(value="No reflexes compiled yet.")
        ttk.Label(
            header,
            textvariable=self.reflex_summary_var,
            style="Panel.TLabel",
            wraplength=300,
            justify="left",
        ).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Checkbutton(
            header,
            text="Use reflexes in operation",
            variable=self._setting_vars["reflex_enabled"],
            command=self._refresh_reflex_list,
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 2))

        controls = ttk.Frame(header, style="Panel.TFrame")
        controls.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        for column in range(3):
            controls.columnconfigure(column, weight=1, uniform="reflexactions")
        ttk.Button(
            controls,
            text="Refresh",
            style="Compact.TButton",
            command=self._refresh_reflex_list,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 3))
        ttk.Button(
            controls,
            text="Enable all",
            style="Compact.TButton",
            command=lambda: self._set_all_reflexes(True),
        ).grid(row=0, column=1, sticky="ew", padx=3)
        ttk.Button(
            controls,
            text="Disable all",
            style="Compact.TButton",
            command=lambda: self._set_all_reflexes(False),
        ).grid(row=0, column=2, sticky="ew", padx=(3, 0))

        list_frame = ttk.LabelFrame(parent, text="Compiled reflexes", padding=8)
        list_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        canvas = tk.Canvas(list_frame, bg=PANEL, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas, style="Panel.TFrame")
        inner.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(window_id, width=event.width),
        )
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        inner.columnconfigure(1, weight=1)
        self._reflex_rows = inner
        self._refresh_reflex_list()

    def _load_reflexes(self) -> list[Any]:
        """Read the compiled reflexes using the settings currently in the form."""
        try:
            settings = self._collect_settings()
        except (TypeError, ValueError):
            settings = self._initial_settings
        try:
            return MemoryManager(settings.memory_file).list_skills()
        except OSError:
            return []

    def _refresh_reflex_list(self) -> None:
        inner = getattr(self, "_reflex_rows", None)
        if inner is None:
            return
        for child in inner.winfo_children():
            child.destroy()
        skills = sorted(self._load_reflexes(), key=lambda skill: skill.name)
        enabled = sum(1 for skill in skills if getattr(skill, "enabled", False))
        self.reflex_summary_var.set(
            f"{len(skills)} reflex(es) cached, {enabled} enabled. "
            "New reflexes are stored disabled until you enable them here."
        )
        if not skills:
            ttk.Label(
                inner,
                text="Nothing compiled yet.",
                style="Muted.Panel.TLabel",
            ).grid(row=0, column=0, columnspan=4, sticky="w")
            return
        for row, skill in enumerate(skills):
            variable = tk.BooleanVar(self, value=bool(getattr(skill, "enabled", False)))
            ttk.Checkbutton(
                inner,
                variable=variable,
                command=lambda name=skill.name, var=variable: self._set_reflex_enabled(
                    name, var.get()
                ),
            ).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Label(
                inner,
                text=skill.name.replace("_", " "),
                style="Panel.TLabel",
                wraplength=170,
                justify="left",
            ).grid(row=row, column=1, sticky="w", padx=(4, 6), pady=2)
            ttk.Label(
                inner,
                text=(
                    f"{getattr(skill.action, 'value', skill.action)}"
                    f" · ok {skill.success_count}"
                    f" · miss {skill.failure_count}"
                ),
                style="Muted.Panel.TLabel",
            ).grid(row=row, column=2, sticky="w", pady=2)
            ttk.Button(
                inner,
                text="×",
                style="Compact.TButton",
                command=lambda name=skill.name: self._delete_reflex(name),
            ).grid(row=row, column=3, sticky="e", pady=2)

    def _set_reflex_enabled(self, name: str, enabled: bool) -> None:
        if self._update_reflexes(lambda memory: memory.set_enabled(name, enabled)):
            state = "enabled" if enabled else "disabled"
            self._append_log("SYSTEM", f"Reflex {name!r} {state}.")
            self._refresh_reflex_list()

    def _set_all_reflexes(self, enabled: bool) -> None:
        changed: list[int] = []

        def apply(memory: MemoryManager) -> bool:
            changed.append(memory.set_all_enabled(enabled))
            return True

        if self._update_reflexes(apply):
            state = "enabled" if enabled else "disabled"
            self._append_log(
                "SYSTEM", f"{changed[0] if changed else 0} reflex(es) {state}."
            )
            self._refresh_reflex_list()

    def _delete_reflex(self, name: str) -> None:
        if not messagebox.askyesno(
            "Delete reflex",
            f"Delete the compiled reflex {name!r} and its template?",
        ):
            return

        def apply(memory: MemoryManager) -> bool:
            self._remove_reflex_template(memory, name)
            return memory.forget(name)

        if self._update_reflexes(apply):
            self._append_log("SYSTEM", f"Deleted reflex {name!r}.")
            self._refresh_reflex_list()

    def _remove_reflex_template(self, memory: MemoryManager, name: str) -> None:
        """Remove the reflex's own crop, leaving user icon templates alone."""
        skill = memory.get_skill(name)
        if skill is None:
            return
        template = Path(str(skill.template_path or ""))
        try:
            if template.is_file() and self._initial_settings.templates_dir in template.parents:
                template.unlink()
        except OSError:
            pass

    def _update_reflexes(self, action: Any) -> bool:
        """Apply a change to the reflex cache, reporting failures in the log."""
        try:
            settings = self._collect_settings()
        except (TypeError, ValueError) as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return False
        try:
            return bool(action(MemoryManager(settings.memory_file)))
        except OSError as exc:
            messagebox.showerror("Reflex cache error", str(exc))
            return False

    def _build_context_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Context for the AI", padding=10)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)
        ttk.Label(
            frame,
            text="Add durable facts, preferences, paths, or constraints Furti should use while planning.",
            style="Panel.TLabel",
            wraplength=330,
            justify="left",
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.context_input = tk.Text(
            frame,
            height=12,
            wrap="word",
            bg="#0d1218",
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            font=(self._ui_font_family, 12),
            padx=8,
            pady=8,
        )
        self.context_input.grid(row=1, column=0, sticky="nsew")
        context_scrollbar = ttk.Scrollbar(
            frame, orient="vertical", command=self.context_input.yview
        )
        context_scrollbar.grid(row=1, column=1, sticky="ns")
        self.context_input.configure(yscrollcommand=context_scrollbar.set)
        notes = str(self._initial_settings.user_notes or "")
        if notes:
            self.context_input.insert("1.0", notes)

    def _build_advanced_panel(self, parent: tk.Frame) -> None:
        canvas = tk.Canvas(
            parent,
            bg=PANEL,
            highlightthickness=0,
            borderwidth=0,
        )
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas, style="Panel.TFrame")
        inner.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(window_id, width=event.width),
        )
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        names = tuple(
            name
            for name in self._ADVANCED_FIELDS
            if name not in getattr(self, "_tabbed_setting_names", set())
        )
        for row, name in enumerate(names):
            self._add_setting_control(inner, row, name)

        note = ttk.Label(
            inner,
            text=(
                "The main Furti window is the live status UI; the secondary "
                "overlay is not opened from this embedded app."
            ),
            style="Muted.Panel.TLabel",
            wraplength=315,
            justify="left",
        )
        note.grid(
            row=len(names),
            column=0,
            columnspan=2,
            sticky="w",
            pady=(10, 0),
        )

    def _add_setting_control(self, parent: ttk.Frame, row: int, name: str) -> None:
        label = name.replace("_", " ").title()
        variable = self._setting_vars[name]
        ttk.Label(parent, text=label, style="Panel.TLabel").grid(
            row=row,
            column=0,
            sticky="w",
            padx=(0, 6),
            pady=3,
        )
        parent.grid_columnconfigure(1, weight=1)
        if isinstance(variable, tk.BooleanVar):
            check = ttk.Checkbutton(parent, variable=variable)
            check.grid(row=row, column=1, sticky="e", pady=3)
            if name == "enable_status_window":
                check.state(["disabled"])
        elif name == "llm_provider":
            ttk.Combobox(
                parent,
                textvariable=variable,
                values=("auto", "deepseek", "gemini"),
                state="readonly",
                width=18,
            ).grid(row=row, column=1, sticky="ew", pady=3)
        else:
            entry = ttk.Entry(parent, textvariable=variable, width=20)
            if name.endswith("_api_key"):
                entry.configure(show="*")
            entry.grid(row=row, column=1, sticky="ew", pady=3)

    def _build_plan_panel(self, parent: tk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Proposed plan", padding=8)
        frame.grid(row=0, column=0, sticky="nsew", pady=(10, 8))
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        self.plan_text = ScrolledText(
            frame,
            height=9,
            wrap="word",
            bg="#0d1218",
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            font=(self._mono_font_family, 10),
        )
        self.plan_text.grid(row=0, column=0, sticky="nsew")
        self._set_readonly_text(self.plan_text, "No plan has been generated yet.")

        controls = ttk.Frame(frame, style="Panel.TFrame")
        controls.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(2, weight=1)
        self.approve_button = ttk.Button(
            controls,
            text="Approve and run",
            style="Accent.TButton",
            command=self._approve_plan,
            state="disabled",
            **self._button_options("check", "approve", "play"),
        )
        self.approve_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.edit_button = ttk.Button(
            controls,
            text="Edit and re-plan",
            command=self._edit_plan,
            state="disabled",
            **self._button_options("edit", "pencil"),
        )
        self.edit_button.grid(row=0, column=1, sticky="ew", padx=4)
        self.decline_button = ttk.Button(
            controls,
            text="Decline",
            style="Danger.TButton",
            command=self._decline_plan,
            state="disabled",
            **self._button_options("x", "close", "decline"),
        )
        self.decline_button.grid(row=0, column=2, sticky="ew", padx=(4, 0))

    def _build_status_panel(self, parent: tk.Frame) -> None:
        """Live status cards, built into the status tab of the output notebook."""
        frame = ttk.LabelFrame(parent, text="Live status", padding=8)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        self.phase_var = tk.StringVar(value="Idle")
        self.step_var = tk.StringVar(value="No active step")
        self.screenshot_var = tk.StringVar(value="Last screenshot: -")
        self.action_var = tk.StringVar(value="Current action: -")
        self.tokens_var = tk.StringVar(value="Tokens: -")
        self.cost_var = tk.StringVar(value="Cost: -")
        rows = (
            ("Phase", self.phase_var),
            ("Step", self.step_var),
            ("Screenshot", self.screenshot_var),
            ("Action", self.action_var),
            ("Usage", self.tokens_var),
            ("Estimate", self.cost_var),
        )
        for row, (label, variable) in enumerate(rows):
            ttk.Label(frame, text=f"{label}:", style="Muted.Panel.TLabel").grid(
                row=row,
                column=0,
                sticky="nw",
                padx=(0, 8),
                pady=2,
            )
            ttk.Label(
                frame,
                textvariable=variable,
                style="Panel.TLabel",
                wraplength=620,
                justify="left",
            ).grid(row=row, column=1, sticky="w", pady=2)

        self.signal_label = tk.Label(
            frame,
            text="READY",
            bg="#214b35",
            fg=GREEN,
            font=(self._mono_font_family, 10, "bold"),
            anchor="w",
            padx=8,
            pady=4,
        )
        self.signal_label.grid(
            row=0,
            column=2,
            rowspan=2,
            sticky="e",
            padx=(12, 0),
        )
        self.overlay_var = tk.BooleanVar(
            self,
            value=bool(self._initial_settings.enable_status_window),
        )
        ttk.Checkbutton(
            frame,
            text="Always-on-top status window",
            variable=self.overlay_var,
            command=self._toggle_status_window,
        ).grid(row=len(rows), column=0, columnspan=3, sticky="w", pady=(6, 0))

    def _build_output_panels(self, parent: tk.Frame) -> None:
        output_tabs = ttk.Notebook(parent)
        output_tabs.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        parent.rowconfigure(1, weight=1)
        self.output_notebook = output_tabs
        status_tab = ttk.Frame(output_tabs, style="Panel.TFrame", padding=8)
        ai_tab = ttk.Frame(output_tabs, style="Panel.TFrame", padding=8)
        log_tab = ttk.Frame(output_tabs, style="Panel.TFrame", padding=8)
        self._add_tab(
            output_tabs, status_tab, "Live Status", self._icon("activity", "log")
        )
        self._add_tab(output_tabs, ai_tab, "AI Output", self._icon("ai", "brain"))
        self._add_tab(output_tabs, log_tab, "Event Log", self._icon("log", "activity"))
        self._build_status_panel(status_tab)

        ai_frame = ttk.LabelFrame(ai_tab, text="Latest AI output", padding=8)
        ai_frame.pack(fill="both", expand=True)
        ai_frame.rowconfigure(0, weight=1)
        ai_frame.columnconfigure(0, weight=1)
        self.ai_output = ScrolledText(
            ai_frame,
            height=5,
            wrap="word",
            bg="#0d1218",
            fg="#d8e8ff",
            relief="flat",
            font=(self._mono_font_family, 10),
        )
        self.ai_output.pack(fill="both", expand=True)
        self._set_readonly_text(self.ai_output, "No model response yet.")

        log_frame = ttk.LabelFrame(log_tab, text="Transparent event log", padding=8)
        log_frame.pack(fill="both", expand=True)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log_text = ScrolledText(
            log_frame,
            height=12,
            wrap="word",
            bg="#0d1218",
            fg="#cbd7df",
            relief="flat",
            font=(self._mono_font_family, 10),
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_text.tag_configure("ERROR", foreground=RED)
        self.log_text.tag_configure("WARN", foreground=AMBER)
        self.log_text.tag_configure("ACTION", foreground=AMBER)
        self.log_text.tag_configure("CONFIRM", foreground=GREEN)
        self.log_text.tag_configure("AI_OUTPUT", foreground="#b5d9ff")
        self._set_readonly_text(
            self.log_text,
            "Logs from planning and execution will appear here.",
        )

    # -------------------------------------------------------------- settings
    def _toggle_advanced(self) -> None:
        self.advanced_visible = not self.advanced_visible
        if self.advanced_visible:
            self.advanced_container.pack(fill="both", expand=True, padx=10, pady=(0, 10))
            self.advanced_button.configure(text="Hide advanced settings")
        else:
            self.advanced_container.pack_forget()
            self.advanced_button.configure(text="Show advanced settings")

    def _collect_settings(self) -> Settings:
        values: dict[str, Any] = {}
        for setting in fields(Settings):
            name = setting.name
            variable = self._setting_vars[name]
            current = getattr(self._initial_settings, name)
            if isinstance(current, bool):
                values[name] = bool(variable.get())
                continue
            raw = str(variable.get()).strip()
            if isinstance(current, Path):
                if not raw:
                    raise ValueError("Workspace cannot be empty.")
                values[name] = Path(raw)
            elif isinstance(current, float):
                values[name] = float(raw)
            elif isinstance(current, int):
                values[name] = int(raw)
            else:
                values[name] = raw
        if hasattr(self, "context_input"):
            values["user_notes"] = self.context_input.get("1.0", "end").strip()
        values["llm_provider"] = str(values["llm_provider"]).lower()
        self._validate_settings(values)
        return Settings(**values)

    def _clear_cache(self) -> None:
        """Clear compiled reflexes without touching user icon templates."""
        if self._running:
            messagebox.showwarning(
                "Task running", "Stop the current task before clearing reflexes."
            )
            return
        if not messagebox.askyesno(
            "Clear reflex cache",
            "Remove all compiled reflexes and their generated templates?",
        ):
            return
        try:
            settings = self._collect_settings()
            removed = MemoryManager(settings.memory_file).clear_all(settings.templates_dir)
        except (OSError, TypeError, ValueError) as exc:
            messagebox.showerror("Cache error", str(exc))
            return
        self._append_log("SYSTEM", f"Cleared reflex cache ({removed} template(s) removed).")
        self._refresh_reflex_list()

    @staticmethod
    def _validate_settings(values: dict[str, Any]) -> None:
        for name in ("confidence_threshold", "icon_match_threshold"):
            value = float(values[name])
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")
        ocr_min_confidence = float(values["ocr_min_confidence"])
        if not 0 <= ocr_min_confidence <= 1:
            raise ValueError("ocr_min_confidence must be between 0 and 1.")
        for name in (
            "cursor_move_duration",
            "input_pause",
            "typing_interval",
            "drag_duration",
            "screenshot_min_interval",
        ):
            if float(values[name]) < 0:
                raise ValueError(f"{name} cannot be negative.")
        for name in (
            "max_plan_steps",
            "max_step_retries",
            "max_plan_replans",
            "max_llm_calls_per_task",
            "max_consecutive_failures",
            "max_image_dim",
            "ocr_max_dim",
            "ocr_max_lines",
            "icon_max_templates",
            "icon_match_max_dim",
        ):
            if int(values[name]) < 1:
                raise ValueError(f"{name} must be at least 1.")

    # ------------------------------------------------------------- lifecycle
    def _start_task(self) -> None:
        if self._running:
            return
        instruction = self.task_input.get("1.0", "end").strip()
        if not instruction:
            messagebox.showwarning("Task required", "Enter a task before planning.")
            self.task_input.focus_set()
            return
        try:
            settings = self._collect_settings()
        except (TypeError, ValueError) as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return

        self._stop_event.clear()
        self._pending_confirmation = None
        self._running = True
        self._set_running_controls(True)
        self._clear_runtime_output()
        self._prepare_status_window(instruction, settings)
        self._enqueue_system(f"Starting {self.mode_var.get()} mode.")
        mode = self.mode_var.get()
        self._worker = threading.Thread(
            target=self._run_worker,
            args=(instruction, mode, settings),
            name="furti-app-worker",
            daemon=True,
        )
        self._worker.start()

    def _run_worker(self, instruction: str, mode: str, settings: Settings) -> None:
        try:
            if mode == "reflex":
                self._enqueue_system("Legacy reflex mode started.")
                success = build_agent(settings).run(instruction)
            else:
                agent = build_task_agent(
                    settings,
                    stop_event=self._stop_event,
                    status_sink=lambda event: self._queue.put(("journal", event)),
                    confirmation_callback=self._request_confirmation,
                    user_choice_callback=self._request_user_choice,
                    manage_status_window=False,
                )
                self._enqueue_system(
                    "Task mode started. The always-on-top status window mirrors "
                    "this view."
                )
                success = agent.run_task(instruction)
            self._queue.put(("finished", bool(success)))
        except Exception as exc:
            self._queue.put(("worker_error", exc))

    # --------------------------------------------------- always-on-top window
    def _ensure_status_window(self) -> StatusWindow | None:
        """Create the overlay as a Toplevel of this window (never a 2nd root)."""
        window = self._status_window
        if window is None:
            window = StatusWindow(master=self)
            window.start(self._stop_event)
            if not window.available:
                return None
            self._status_window = window
        return window

    def _prepare_status_window(self, instruction: str, settings: Settings) -> None:
        if not settings.enable_status_window or not self.overlay_var.get():
            self._hide_status_window()
            return
        window = self._ensure_status_window()
        if window is None:
            return
        self._cancel_status_hide()
        window.post(
            {
                "task": instruction,
                "phase": "Planning",
                "step": "No active step",
                "current_action": "Waiting for the plan",
            }
        )
        window.show()

    def _toggle_status_window(self) -> None:
        if not self.overlay_var.get():
            self._hide_status_window()
            return
        if not self._running:
            self._append_log(
                "SYSTEM",
                "The always-on-top status window appears while a task runs.",
            )
            return
        window = self._ensure_status_window()
        if window is not None:
            self._cancel_status_hide()
            window.show()

    def _hide_status_window(self) -> None:
        self._cancel_status_hide()
        window = self._status_window
        if window is not None:
            window.hide()

    def _cancel_status_hide(self) -> None:
        hide_id = self._status_hide_id
        self._status_hide_id = None
        if hide_id is None:
            return
        try:
            self.after_cancel(hide_id)
        except Exception:
            pass

    def _hide_status_window_later(self) -> None:
        self._cancel_status_hide()
        self._status_hide_id = self.after(
            self.STATUS_LINGER_MS,
            self._hide_status_window,
        )

    def _post_status(self, snapshot: dict[str, Any]) -> None:
        window = self._status_window
        if window is not None:
            window.post(snapshot)

    def _request_confirmation(self, plan: TaskPlan) -> bool | str:
        request = ConfirmationRequest.for_plan(plan)
        self._queue.put(("plan", request))
        while not request.event.wait(0.1):
            if self._stop_event.is_set():
                request.resolve(False)
        return request.decision if request.decision is not None else False

    def _request_user_choice(self, choice: UserChoiceRequest) -> str | None:
        response = UserChoiceResponse(choice, threading.Event())
        self._queue.put(("question", response))
        while not response.event.wait(0.1):
            if self._stop_event.is_set():
                response.resolve(None)
        return response.answer

    def _stop_task(self) -> None:
        if not self._running:
            return
        self._stop_event.set()
        if self._pending_confirmation is not None:
            self._pending_confirmation.resolve(False)
            self._pending_confirmation = None
            self._set_plan_controls(False)
        if self._pending_choice is not None:
            self._pending_choice.resolve(None)
            self._pending_choice = None
        if self._choice_dialog is not None:
            self._choice_dialog.destroy()
            self._choice_dialog = None
        self._set_signal("STOPPING", "#5a1e1e", RED)
        self.phase_var.set("Stopping safely...")
        self.stop_button.configure(state="disabled")
        self._append_log("SYSTEM", "Stop requested; waiting for the current safe boundary.")

    def _on_close(self) -> None:
        if self._running:
            self._closing = True
            self._stop_task()
            self._append_log("SYSTEM", "Close requested; the task must stop safely first.")
            return
        self._closing = True
        self._hide_status_window()
        self.destroy()

    def _finish_run(self, success: bool) -> None:
        self._running = False
        self._pending_confirmation = None
        if self._pending_choice is not None:
            self._pending_choice.resolve(None)
            self._pending_choice = None
        if self._choice_dialog is not None:
            self._choice_dialog.destroy()
            self._choice_dialog = None
        self._set_running_controls(False)
        if success:
            phase, signal = "Completed", ("COMPLETED", "#214b35", GREEN)
        elif self._stop_event.is_set():
            phase, signal = "Stopped", ("STOPPED", "#5a1e1e", RED)
        else:
            phase, signal = (
                "Finished without execution",
                ("DECLINED / FAILED", "#4b3b1d", AMBER),
            )
        self.phase_var.set(phase)
        self._set_signal(*signal)
        self._post_status(
            {
                "phase": phase,
                "current_action": "Task finished",
                "action_signal": "done",
            }
        )
        self._hide_status_window_later()
        if self._closing:
            self.after(50, self._destroy_after_worker)
        else:
            self._worker = None

    def _destroy_after_worker(self) -> None:
        worker = self._worker
        if worker is None or not worker.is_alive():
            self._worker = None
            self._hide_status_window()
            self.destroy()
        else:
            self.after(50, self._destroy_after_worker)

    def _set_running_controls(self, running: bool) -> None:
        self.start_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")
        self.status_badge.configure(
            text="RUNNING" if running else "READY",
            bg="#4b3b1d" if running else "#214b35",
            fg=AMBER if running else GREEN,
        )

    def _clear_task(self) -> None:
        if self._running:
            return
        self.task_input.delete("1.0", "end")
        self.task_input.focus_set()

    def _clear_runtime_output(self) -> None:
        self._set_readonly_text(self.plan_text, "Planning has started...")
        self._set_readonly_text(self.ai_output, "Waiting for the first model response...")
        self._set_readonly_text(self.log_text, "")
        self.phase_var.set("Planning")
        self.step_var.set("No active step")
        self.screenshot_var.set("Last screenshot: -")
        self.action_var.set("Current action: -")
        self.tokens_var.set("Tokens: -")
        self.cost_var.set("Cost: -")
        self.report_var.set("Report: pending")
        self._last_capture_epoch = 0.0
        self._last_capture_at = ""
        self._set_signal("AI THINKING", "#302c57", "#d8c8ff")

    # ------------------------------------------------------------ plan actions
    def _set_plan_controls(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.approve_button.configure(state=state)
        self.edit_button.configure(state=state)
        self.decline_button.configure(state=state)

    def _approve_plan(self) -> None:
        self._resolve_plan(True)

    def _decline_plan(self) -> None:
        self._resolve_plan(False)

    def _edit_plan(self) -> None:
        instruction = self.task_input.get("1.0", "end").strip()
        if not instruction:
            messagebox.showwarning("Task required", "Enter the revised task first.")
            return
        self._resolve_plan(instruction)

    def _resolve_plan(self, decision: bool | str) -> None:
        request = self._pending_confirmation
        if request is None:
            return
        self._pending_confirmation = None
        self._set_plan_controls(False)
        if isinstance(decision, str):
            self.phase_var.set("Re-planning")
            self._set_signal("AI THINKING", "#302c57", "#d8c8ff")
        elif decision:
            self.phase_var.set("Executing")
            self._set_signal("APPROVED", "#214b35", GREEN)
        else:
            self.phase_var.set("Declined")
            self._set_signal("DECLINED", "#4b3b1d", AMBER)
        request.resolve(decision)

    # --------------------------------------------------------------- queue/UI
    def _poll_events(self) -> None:
        try:
            while True:
                event_type, payload = self._queue.get_nowait()
                if event_type == "journal":
                    self._post_status(payload)
                    self._apply_journal_event(payload)
                elif event_type == "plan":
                    self._show_plan(payload)
                elif event_type == "question":
                    self._show_question(payload)
                elif event_type == "finished":
                    self._finish_run(bool(payload))
                elif event_type == "worker_error":
                    self._handle_worker_error(payload)
                elif event_type == "system":
                    self._append_log("SYSTEM", str(payload))
        except queue.Empty:
            pass
        if not self._closing or self._running:
            self.after(self.POLL_MS, self._poll_events)

    def _show_plan(self, request: ConfirmationRequest) -> None:
        self._pending_confirmation = request
        self._set_readonly_text(self.plan_text, request.plan.describe())
        self.phase_var.set("Waiting for your approval")
        self._set_signal("PLAN READY", "#1e3d59", "#9ed8ff")
        self._post_status(
            {
                "phase": "Waiting for your approval",
                "step": "Plan ready for approval",
                "current_action": "Review the plan in the Furti AI window",
            }
        )
        if self._stop_event.is_set():
            request.resolve(False)
            return
        self._set_plan_controls(True)

    def _show_question(self, response: UserChoiceResponse) -> None:
        """Render a model-authored choice dialog on the Tk main thread."""
        self._pending_choice = response
        self.phase_var.set("Waiting for your choice")
        self._set_signal("INPUT NEEDED", "#1e3d59", "#9ed8ff")
        self._post_status(
            {
                "phase": "Waiting for your choice",
                "step": "Model needs clarification",
                "current_action": response.request.question,
            }
        )
        dialog = tk.Toplevel(self)
        self._choice_dialog = dialog
        self._set_window_icon(dialog)
        dialog.title("Furti AI needs a choice")
        dialog.transient(self)
        dialog.attributes("-topmost", True)
        dialog.grab_set()
        dialog.configure(bg=PANEL)
        dialog.columnconfigure(0, weight=1)
        ttk.Label(
            dialog,
            text=response.request.question,
            style="Panel.TLabel",
            wraplength=520,
            justify="left",
        ).grid(row=0, column=0, sticky="w", padx=18, pady=(18, 10))
        answer = tk.StringVar(dialog)
        options = response.request.options
        if options:
            for row, option in enumerate(options, 1):
                ttk.Radiobutton(
                    dialog,
                    text=option,
                    value=option,
                    variable=answer,
                ).grid(row=row, column=0, sticky="w", padx=18, pady=3)
        else:
            ttk.Entry(dialog, textvariable=answer, width=58).grid(
                row=1, column=0, sticky="ew", padx=18, pady=4
            )
        controls = ttk.Frame(dialog, style="Panel.TFrame")
        controls.grid(row=len(options) + 1, column=0, sticky="e", padx=18, pady=14)

        def submit() -> None:
            value = answer.get().strip()
            if not value:
                return
            dialog.grab_release()
            dialog.destroy()
            self._choice_dialog = None
            self._pending_choice = None
            response.resolve(value)

        def cancel() -> None:
            dialog.grab_release()
            dialog.destroy()
            self._choice_dialog = None
            self._pending_choice = None
            response.resolve(None)

        ttk.Button(controls, text="Cancel", command=cancel).pack(
            side="right", padx=(8, 0)
        )
        ttk.Button(
            controls,
            text="Continue",
            style="Accent.TButton",
            command=submit,
        ).pack(side="right")
        dialog.protocol("WM_DELETE_WINDOW", cancel)
        dialog.update_idletasks()
        dialog.focus_force()
        dialog.lift()

    def _apply_journal_event(self, snapshot: dict[str, Any]) -> None:
        kind = str(snapshot.get("event_kind") or "INFO")
        message = str(snapshot.get("message") or "")
        phase = snapshot.get("phase")
        if phase:
            self.phase_var.set(str(phase).title())
        if kind == "STEP":
            self.step_var.set(message)
        if snapshot.get("current_action"):
            self.action_var.set(str(snapshot["current_action"]))
        if snapshot.get("last_screenshot_epoch"):
            self._last_capture_epoch = float(snapshot["last_screenshot_epoch"])
            self._last_capture_at = str(snapshot.get("last_screenshot_at", "-"))
            self._refresh_screenshot_age()
        if snapshot.get("ai_output") is not None:
            self._set_readonly_text(self.ai_output, str(snapshot["ai_output"]))
        if kind == "COST":
            if message.startswith("Tokens:"):
                self.tokens_var.set(message)
            if "Approximate total API cost:" in message:
                self.cost_var.set(message)
        if kind == "SYSTEM" and message.startswith("Report written:"):
            self.report_var.set(message)

        signal = {
            "THOUGHT": ("AI THINKING", "#302c57", "#d8c8ff"),
            "PLAN": ("PLANNING", "#1e3d59", "#9ed8ff"),
            "MODEL": ("AI THINKING", "#302c57", "#d8c8ff"),
            "WAIT": ("WAITING FOR AI", "#302c57", "#d8c8ff"),
            "AI_OUTPUT": ("AI RESPONSE RECEIVED", "#1e3d59", "#9ed8ff"),
            "SCREENSHOT": ("SCREEN CAPTURED", "#214b35", GREEN),
            "STEP": ("EXECUTING STEP", "#4b3b1d", AMBER),
            "ACTION": ("ACTING", "#4b3b1d", AMBER),
            "CONFIRM": ("ACTION CONFIRMED", "#214b35", GREEN),
            "WARN": ("RECOVERING", "#4b3b1d", AMBER),
            "ERROR": ("ERROR", "#5a1e1e", RED),
        }.get(kind)
        if signal:
            self._set_signal(*signal)
        if message:
            timestamp = str(snapshot.get("timestamp") or "--:--:--")
            self._append_log(kind, f"[{timestamp}] [{kind}] {message}")

    def _handle_worker_error(self, error: Exception) -> None:
        self._append_log("ERROR", f"Worker failed: {error}")
        self.report_var.set("Report: worker failed; see event log")
        self._finish_run(False)

    def _enqueue_system(self, message: str) -> None:
        self._queue.put(
            (
                "system",
                f"[{datetime.now().strftime('%H:%M:%S')}] [SYSTEM] {message}",
            )
        )

    def _append_log(self, kind: str, message: str) -> None:
        text = self.log_text
        text.configure(state="normal")
        tag = (
            kind
            if kind in {"ERROR", "WARN", "ACTION", "CONFIRM", "AI_OUTPUT"}
            else ""
        )
        if tag:
            text.insert("end", message + "\n", tag)
        else:
            text.insert("end", message + "\n")
        line_count = int(text.index("end-1c").split(".")[0])
        if line_count > self.MAX_LOG_LINES:
            text.delete("1.0", f"{line_count - self.MAX_LOG_LINES}.0")
        text.see("end")
        text.configure(state="disabled")

    def _set_signal(self, text: str, background: str, foreground: str) -> None:
        self.signal_label.configure(text=text, bg=background, fg=foreground)

    @staticmethod
    def _set_readonly_text(widget: ScrolledText, value: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.see("1.0")
        widget.configure(state="disabled")

    def _refresh_screenshot_age(self) -> None:
        if self._last_capture_epoch <= 0:
            self.screenshot_var.set("Last screenshot: -")
        else:
            age = max(0.0, datetime.now().timestamp() - self._last_capture_epoch)
            self.screenshot_var.set(
                f"Last screenshot: {self._last_capture_at} ({age:.1f}s ago)"
            )
        if not self._closing:
            self.after(250, self._refresh_screenshot_age)


__all__ = ["ConfirmationRequest", "FurtiApp"]
