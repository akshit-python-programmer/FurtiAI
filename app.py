
"""Configurable application entry point for Furti AI.

Running this file without a task opens the full Tkinter application:

    python app.py

The command-line task path remains available for automation and scripting:

    python app.py --task "Open Notepad and type hello"

It also exposes a small Python API so an application can import Furti rather
than invoking ``python -m furti_ai``:

    from app import build_settings, run

    settings = build_settings(
        llm_provider="deepseek",
        deepseek_api_key="...",
        enable_status_window=True,
    )
    run("Open Notepad and type hello", settings=settings)

Every field on :class:`furti_ai.Settings` has a corresponding CLI option.
Values passed explicitly to ``app.py`` override environment variables; values
not passed on the command line keep the existing ``Settings`` environment
defaults.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Sequence

from furti_ai import Settings, build_agent, build_task_agent


SETTING_NAMES = tuple(field.name for field in fields(Settings))



def build_settings(**overrides: Any) -> Settings:
    """Create :class:`Settings` with validated, programmatic overrides.

    This helper is useful when ``app.py`` is imported by another Python
    application. Environment variables remain the fallback for omitted values.
    """
    unknown = sorted(set(overrides) - set(SETTING_NAMES))
    if unknown:
        names = ", ".join(unknown)
        raise TypeError(f"Unknown Furti setting(s): {names}")
    return Settings(**overrides)


def settings_from_args(namespace: argparse.Namespace) -> Settings:
    """Convert parsed CLI values into ``Settings`` without overriding env vars."""
    values = {
        name: getattr(namespace, name)
        for name in SETTING_NAMES
        if getattr(namespace, name, None) is not None
    }
    return build_settings(**values)

 
def run(
    task: str,
    *,
    mode: str = "task",
    settings: Settings | None = None,
) -> bool:
    """Run one task through the selected Furti pipeline.

    Args:
        task: Natural-language desktop instruction.
        mode: ``"task"`` for transparent multi-step execution or ``"reflex"``
            for the legacy single-action cached-reflex path.
        settings: Optional fully configured :class:`Settings` instance.

    Returns:
        ``True`` when the selected agent reports success.
    """
    if not task or not task.strip():
        raise ValueError("task must not be empty")
    if mode == "task":
        return build_task_agent(settings).run_task(task)
    if mode == "reflex":
        return build_agent(settings).run(task)
    raise ValueError("mode must be either 'task' or 'reflex'")


def launch_gui(
    settings: Settings | None = None,
    *,
    initial_task: str = "",
    initial_mode: str = "task",
) -> int:
    """Launch the primary Tkinter application window.

    The GUI owns the Tk event loop and runs Furti's planner/executor in a
    worker thread. This function is intentionally separate from :func:`run`
    so importing ``app.py`` does not create a window.
    """
    try:
        import tkinter  # noqa: F401 - presence check only
    except ImportError as exc:
        raise RuntimeError(
            "Tkinter is required for the graphical app. "
            "Install a Python distribution that includes tkinter."
        ) from exc
    try:
        from furti_ai.gui import FurtiApp
    except ImportError as exc:
        # Any *other* import failure (a missing dependency, a name that moved)
        # must not be reported as "tkinter is missing"; that message sent
        # debugging in completely the wrong direction.
        raise RuntimeError(f"Could not load the graphical app: {exc}") from exc

    application = FurtiApp(
        settings=settings,
        initial_task=initial_task,
        initial_mode=initial_mode,
    )
    application.mainloop()
    return 0


def create_parser() -> argparse.ArgumentParser:
    """Build the complete command-line interface for ``app.py``."""
    parser = argparse.ArgumentParser(
        prog="python app.py",
        description=(
            "Run Furti AI as an importable application. With no --task, the "
            "full Tkinter UI opens; --task runs the command-line wrapper."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    execution = parser.add_argument_group("execution")
    execution.add_argument(
        "--task",
        "--instruction",
        dest="task",
        metavar="TEXT",
        help="Natural-language desktop task to execute.",
    )
    execution.add_argument(
        "--mode",
        choices=("task", "reflex"),
        default=None,
        help=(
            "task = transparent multi-step planner; reflex = legacy "
            "single-action cache path."
        ),
    )
    execution.add_argument(
        "--show-settings",
        action="store_true",
        help="Print resolved settings with API keys redacted before running.",
    )
    execution.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default=os.getenv("FURTI_LOG_LEVEL", "INFO").upper(),
        help="Python logging level for diagnostic output.",
    )

    settings = parser.add_argument_group(
        "Furti settings (each option also has an environment-variable fallback)"
    )

    def value(
        name: str,
        value_type: type,
        help_text: str,
        *,
        metavar: str | None = None,
        choices: Sequence[str] | None = None,
    ) -> None:
        option = f"--{name.replace('_', '-')}"
        settings.add_argument(
            option,
            dest=name,
            type=value_type,
            default=None,
            metavar=metavar,
            choices=choices,
            help=help_text,
        )

    def boolean(name: str, help_text: str) -> None:
        option = f"--{name.replace('_', '-')}"
        settings.add_argument(
            option,
            dest=name,
            action=argparse.BooleanOptionalAction,
            default=None,
            help=help_text,
        )

    value(
        "confidence_threshold",
        float,
        "Minimum template-match confidence before a reflex can act.",
        metavar="0..1",
    )
    value("workspace", Path, "Root directory for memory, templates, and reports.", metavar="PATH")
    boolean("cursor_teleport", "Teleport the cursor instead of moving it visibly.")
    value(
        "cursor_move_duration",
        float,
        "Maximum smooth cursor travel duration in seconds.",
        metavar="SECONDS",
    )
    value(
        "input_pause",
        float,
        "PyAutoGUI pause after each low-level input call in seconds.",
        metavar="SECONDS",
    )
    value(
        "typing_interval",
        float,
        "Delay between typed characters in seconds.",
        metavar="SECONDS",
    )
    value("fast_model", str, "Fast model override used for routine calls.", metavar="MODEL")
    value("smart_model", str, "Optional model used after reasoning escalation.", metavar="MODEL")

    value("max_plan_steps", int, "Maximum steps retained in an initial plan.", metavar="COUNT")
    value("max_step_retries", int, "Retries/re-plans allowed for one step.", metavar="COUNT")
    value(
        "max_plan_replans",
        int,
        "Maximum adaptive replacements of the unfinished route.",
        metavar="COUNT",
    )
    value(
        "max_llm_calls_per_task",
        int,
        "Hard LLM-call budget for one task.",
        metavar="COUNT",
    )
    value(
        "max_consecutive_failures",
        int,
        "Abort after this many consecutive failed steps.",
        metavar="COUNT",
    )
    value(
        "screenshot_min_interval",
        float,
        "Normal minimum interval between screen captures in seconds.",
        metavar="SECONDS",
    )
    value(
        "max_image_dim",
        int,
        "Maximum dimension for screenshots sent to the model.",
        metavar="PIXELS",
    )

    boolean("ocr_enabled", "Enable OCR text grounding.")
    value("ocr_backend", str, "OCR backend (rapidocr, paddle, or auto).", metavar="BACKEND", choices=("rapidocr", "paddle", "auto"))
    value("ocr_lang", str, "OCR language code.", metavar="LANG")
    value(
        "ocr_max_dim",
        int,
        "Maximum OCR input dimension; boxes are restored to capture pixels.",
        metavar="PIXELS",
    )
    value(
        "ocr_min_confidence",
        float,
        "Minimum OCR confidence retained for screen grounding.",
        metavar="0..1",
    )
    value("ocr_max_lines", int, "Maximum OCR lines retained per frame.", metavar="COUNT")
    boolean("ocr_enable_mkldnn", "Enable PaddlePaddle oneDNN CPU acceleration.")
    value(
        "icon_match_threshold",
        float,
        "Minimum saved-template icon-match confidence.",
        metavar="0..1",
    )
    value(
        "icon_max_templates",
        int,
        "Maximum saved templates scanned per frame.",
        metavar="COUNT",
    )
    value(
        "icon_match_max_dim",
        int,
        "Maximum screen dimension used for icon matching.",
        metavar="PIXELS",
    )
    boolean(
        "icon_match_multiscale",
        "Enable slower multi-scale icon matching when exact matching misses.",
    )

    boolean("enable_status_window", "Show the always-on-top Tk status window.")
    value("kill_hotkey", str, "Global abort hotkey in pynput syntax.", metavar="HOTKEY")
    boolean(
        "verify_steps",
        "Review the completed step and next-step readiness in one model call.",
    )

    value(
        "llm_provider",
        str,
        "Reasoning provider selection.",
        choices=("auto", "deepseek", "gemini"),
    )
    value("deepseek_api_key", str, "DeepSeek API key (prefer the environment variable).", metavar="KEY")
    value("deepseek_base_url", str, "OpenAI-compatible DeepSeek base URL.", metavar="URL")
    value("deepseek_model", str, "DeepSeek model name.", metavar="MODEL")
    value("google_api_key", str, "Google API key (prefer the environment variable).", metavar="KEY")
    value("gemini_model", str, "Gemini model name.", metavar="MODEL")
    boolean("deepseek_thinking_mode", "Use thinking-mode request compatibility.")
    boolean(
        "deepseek_use_function_calling",
        "Enable DeepSeek/OpenAI-compatible function calling.",
    )
    return parser


def _safe_settings(settings: Settings) -> dict[str, Any]:
    """Return settings suitable for console display without exposing secrets."""
    values: dict[str, Any] = {}
    for field in fields(settings):
        value = getattr(settings, field.name)
        if field.name.endswith("_api_key"):
            values[field.name] = "<set>" if value else "<unset>"
        elif isinstance(value, Path):
            values[field.name] = str(value)
        else:
            values[field.name] = value
    return values

 
def main(argv: Sequence[str] | None = None) -> int:
    """Application entry point used by ``python app.py``."""
    parser = create_parser()
    args = parser.parse_args(argv)
    mode = args.mode or "task"

    settings = settings_from_args(args)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.show_settings:
        print("Resolved Furti settings:")
        for name, value in _safe_settings(settings).items():
            print(f"  {name} = {value}")

    try:
        if not args.task:
            return launch_gui(
                settings=settings,
                initial_mode=mode,
            )
        ok = run(args.task, mode=mode, settings=settings)
    except (ImportError, OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"fatal: {exc}", file=sys.stderr)
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
