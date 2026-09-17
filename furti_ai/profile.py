"""A persisted description of the user and the machine they are working on.

The planner used to *guess* the environment: it did not know that this user's
Desktop is redirected into OneDrive, that VLC is installed, that ``git`` is on
``PATH``, or where their notes folder lives. Every guess is a chance to take
the slow route (open Explorer and click around) instead of the right one (call
one tool with the correct absolute path).

:class:`UserContext` detects those facts once and stores them in
``<workspace>/context/profile.json``, with a human-readable mirror in
``<workspace>/context/profile.md``. The planner, the re-planner and the
cross-provider verifier all receive a compact rendering of it via
:meth:`UserContext.prompt_block`.

The ``profile.md`` file is meant to be read (and edited) by the user: any
bullet points under its ``## Notes`` heading are loaded back as user facts, so
"I keep invoices in D:\\invoices" becomes part of the agent's context without
touching environment variables.

Everything here degrades silently: a fact that cannot be detected is simply
absent, because a partial profile is still far better than none.
"""

from __future__ import annotations

import getpass
import logging
import os
import platform
import shutil
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

__all__ = ["PROFILE_VERSION", "UserContext"]

#: Bumped when the detected shape changes, forcing a refresh of old files.
PROFILE_VERSION = 2

#: Command-line tools worth advertising when they are installed.
_CLI_PROBES = (
    "git",
    "python",
    "pip",
    "uv",
    "node",
    "npm",
    "code",
    "code-insiders",
    "pwsh",
    "powershell",
    "cmd",
    "winget",
    "choco",
    "scoop",
    "docker",
    "gh",
    "ffmpeg",
    "curl",
    "wget",
    "tar",
    "7z",
    "jq",
    "rg",
    "robocopy",
    "notepad",
)

#: Start-menu shortcuts whose names are documentation, not applications.
_APP_NOISE = (
    "uninstall",
    "readme",
    "release notes",
    "documentation",
    "help",
    "manual",
    "website",
    "web site",
    "license",
    "repair",
    "modify",
    "setup",
    "update",
    "demo",
    "sample",
    "about ",
)

#: Folder names looked for under the home directory (and OneDrive).
_KNOWN_FOLDERS = (
    ("desktop", "Desktop"),
    ("documents", "Documents"),
    ("downloads", "Downloads"),
    ("pictures", "Pictures"),
    ("music", "Music"),
    ("videos", "Videos"),
)


def _safe(callable_obj, default=None):
    """Run a probe, returning ``default`` when the environment refuses."""
    try:
        return callable_obj()
    except Exception as exc:  # noqa: BLE001 - a profile fact is never critical
        logger.debug("profile probe failed: %s", exc)
        return default


class UserContext:
    """Detected user/system facts, persisted as a small JSON "context file"."""

    def __init__(self, settings: Any, journal: Any = None) -> None:
        self._settings = settings
        self._journal = journal
        self._data: dict[str, Any] = {}

    # ------------------------------------------------------------- lifecycle
    @property
    def enabled(self) -> bool:
        return bool(getattr(self._settings, "profile_enabled", True))

    @property
    def profile_file(self) -> Path:
        return Path(getattr(self._settings, "profile_file"))

    @property
    def report_file(self) -> Path:
        return Path(getattr(self._settings, "profile_report"))

    def data(self) -> dict[str, Any]:
        """Return the loaded profile, loading it from disk on first use."""
        if not self._data:
            self._data = self._load()
        return self._data

    def prepare(self, force: bool = False) -> dict[str, Any]:
        """Load the profile, re-detecting it when missing or stale."""
        if not self.enabled:
            return {}
        data = self.data()
        if force or self._is_stale(data):
            data = self.refresh(force=True)
        return data

    # ----------------------------------------------------------------- notes
    def notes(self) -> list[str]:
        """User-supplied facts (from settings, ``remember()`` or profile.md)."""
        typed = str(getattr(self._settings, "user_notes", "") or "").strip()
        stored = [str(item).strip() for item in self.data().get("notes", []) if str(item).strip()]
        if typed:
            stored.extend(
                part.strip() for part in typed.replace(";", "\n").splitlines() if part.strip()
            )
        seen: set[str] = set()
        ordered: list[str] = []
        for note in stored:
            key = note.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(note)
        return ordered

    def remember(self, note: str, save: bool = True) -> None:
        """Store a durable fact about the user (idempotent)."""
        text = str(note or "").strip()
        if not text:
            return
        data = self.data()
        notes = [str(item) for item in data.get("notes", [])]
        if text.lower() not in {item.lower() for item in notes}:
            notes.append(text)
            data["notes"] = notes
            if save:
                self._save()
            self._journal_note(f"Remembered user context: {text}")

    def record_task(self, instruction: str, success: bool, save: bool = True) -> None:
        """Keep a short history of what the agent was asked to do."""
        text = str(instruction or "").strip()
        if not text:
            return
        data = self.data()
        history = [
            entry for entry in data.get("recent_tasks", []) if isinstance(entry, dict)
        ]
        history.append(
            {
                "instruction": text,
                "success": bool(success),
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
        data["recent_tasks"] = history[-20:]
        if save:
            self._save()

    # -------------------------------------------------------------- refresh
    def refresh(self, force: bool = False) -> dict[str, Any]:
        """Re-detect the environment, keeping notes and task history."""
        if not self.enabled and not force:
            return {}
        previous = self.data() if self._data else self._load()
        # Each section is detected independently: an exotic machine (or a
        # missing display server) must cost one fact, not the whole profile.
        detected: dict[str, Any] = {
            "version": PROFILE_VERSION,
            "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "identity": _safe(self._identity, {}) or {},
            "display": _safe(self._display, {}) or {},
            "folders": _safe(self._folders, {}) or {},
            "drives": _safe(self._drives, []) or [],
            "apps": _safe(self._apps, []) or [],
            "cli_tools": _safe(self._cli_tools, []) or [],
            "notes": list(previous.get("notes", [])),
            "recent_tasks": list(previous.get("recent_tasks", [])),
        }
        self._data = detected
        self._save()
        self._journal_note(
            f"System profile refreshed ({len(detected['apps'])} app(s), "
            f"{len(detected['cli_tools'])} CLI tool(s)) -> {self.profile_file}"
        )
        return detected

    # -------------------------------------------------------------- prompts
    def prompt_block(self, max_apps: Optional[int] = None) -> str:
        """Compact context block for the planning / verification prompts."""
        if not self.enabled:
            return ""
        summary = self.summary(max_apps=max_apps)
        if not summary:
            return ""
        return (
            "Known user and system context (trust this over your assumptions; "
            "use the exact paths and application names below):\n" + summary
        )

    def summary(self, max_apps: Optional[int] = None) -> str:
        """Human-readable bullet list of the detected facts."""
        data = self.data()
        if not data:
            return ""
        limit = int(
            max_apps
            if max_apps is not None
            else getattr(self._settings, "profile_max_apps", 40)
        )
        lines: list[str] = []

        identity = data.get("identity") or {}
        who = " on ".join(
            part for part in (identity.get("username"), identity.get("hostname")) if part
        )
        if who or identity.get("os"):
            lines.append(f"- user: {who or 'unknown'} ({identity.get('os', '?')})")

        display = data.get("display") or {}
        if display.get("width"):
            scaling = display.get("scaling_percent")
            suffix = f" at {scaling}% scaling" if scaling else ""
            lines.append(
                f"- screen: {display['width']}x{display['height']}{suffix}"
            )

        folders = data.get("folders") or {}
        if folders:
            lines.append(
                "- folders: " + ", ".join(f"{name}={path}" for name, path in folders.items())
            )

        drives = data.get("drives") or []
        usable = [item for item in drives if item.get("free_gb") is not None]
        if usable:
            lines.append(
                "- drives: "
                + ", ".join(
                    f"{item['path']} {item['free_gb']:.0f}GB free"
                    for item in usable[:4]
                )
            )

        apps = data.get("apps") or []
        if apps:
            shown = apps[:limit]
            more = f" (+{len(apps) - len(shown)} more)" if len(apps) > len(shown) else ""
            lines.append("- installed applications: " + ", ".join(shown) + more)

        tools = data.get("cli_tools") or []
        if tools:
            lines.append("- command-line tools available: " + ", ".join(tools))

        notes = self.notes()
        if notes:
            lines.append("- user notes: " + " | ".join(notes[:8]))

        history = data.get("recent_tasks") or []
        if history:
            recent = "; ".join(
                f"{entry.get('instruction', '')[:60]} "
                f"({'ok' if entry.get('success') else 'failed'})"
                for entry in history[-3:]
            )
            lines.append(f"- recent tasks: {recent}")

        if not lines:
            return ""
        lines.append(
            f"- profile file: {self.profile_file} (edit its Notes section to "
            "teach the agent about your machine)"
        )
        return "\n".join(lines)

    def as_markdown(self) -> str:
        """Full profile document, including the editable Notes section."""
        data = self.data()
        lines = [
            "# Furti AI - user and system context",
            "",
            "This file is generated by the agent. It is safe to edit: anything you",
            "add as a bullet under `## Notes` is loaded back as a user fact.",
            "",
            f"- generated: {data.get('collected_at', 'never')}",
            "",
            "## Detected system",
            "",
            self.summary() or "(nothing detected yet)",
            "",
            "## Notes",
            "",
            "<!-- One fact per line, e.g. 'I keep invoices in D:\\invoices'. -->",
            "",
        ]
        seen: set[str] = set()
        for note in self.notes():
            key = note.lower()
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- {note}")
        lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------- detection
    def _identity(self) -> dict[str, Any]:
        return {
            "username": _safe(getpass.getuser) or "",
            "home": str(Path.home()),
            "hostname": _safe(socket.gethostname) or "",
            "os": f"{platform.system()} {platform.release() or ''}".strip(),
            "os_version": platform.version(),
            "python": platform.python_version(),
            "shell": os.environ.get("COMSPEC") or os.environ.get("SHELL") or "",
            "workspace": str(getattr(self._settings, "workspace", "")),
        }

    def _display(self) -> dict[str, Any]:
        size: Optional[tuple[int, int]] = None
        try:
            from .screen import PyAutoGuiScreen

            size = PyAutoGuiScreen.input_size()
        except Exception as exc:  # noqa: BLE001 - headless / no display
            logger.debug("screen size probe failed: %s", exc)
        return {
            "width": int(size[0]) if size else 0,
            "height": int(size[1]) if size else 0,
            "scaling_percent": self._dpi_scaling(),
        }

    @staticmethod
    def _dpi_scaling() -> Optional[int]:
        """Windows display scaling in percent (150 = 150%), when available."""
        if sys.platform != "win32":
            return None

        def _read() -> int:
            import ctypes

            user32 = ctypes.windll.user32
            try:
                # GetDpiForSystem exists on Windows 10 1607+.
                dpi = int(user32.GetDpiForSystem())
            except (AttributeError, OSError):
                hdc = user32.GetDC(0)
                dpi = int(ctypes.windll.gdi32.GetDeviceCaps(hdc, 88))  # LOGPIXELSX
                user32.ReleaseDC(0, hdc)
            return round(dpi / 96.0 * 100)

        return _safe(_read)

    def _folders(self) -> dict[str, str]:
        """Existing well-known folders, preferring OneDrive redirects."""
        home = Path.home()
        roots = [home / "OneDrive", home / "OneDrive - Personal", home]
        found: dict[str, str] = {}
        for key, name in _KNOWN_FOLDERS:
            for root in roots:
                candidate = root / name
                if candidate.is_dir():
                    found[key] = str(candidate)
                    break
        workspace = getattr(self._settings, "workspace", None)
        if workspace:
            found["workspace"] = str(workspace)
        return found

    def _drives(self) -> list[dict[str, Any]]:
        """Mounted drives with their free space."""
        drives: list[Path] = []
        listdrives = getattr(os, "listdrives", None)
        if callable(listdrives):
            drives = [Path(entry) for entry in _safe(listdrives, []) or []]
        elif sys.platform == "win32":
            for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
                candidate = Path(f"{letter}:\\")
                if candidate.exists():
                    drives.append(candidate)
        else:
            drives = [Path("/")]

        facts: list[dict[str, Any]] = []
        for drive in drives:
            usage = _safe(lambda d=drive: shutil.disk_usage(d))
            facts.append(
                {
                    "path": str(drive),
                    "total_gb": round(usage.total / 1024**3, 1) if usage else None,
                    "free_gb": round(usage.free / 1024**3, 1) if usage else None,
                }
            )
        return facts

    def _apps(self) -> list[str]:
        """Installed applications, most launchable first.

        The order matters: the prompt only shows the first ``profile_max_apps``
        entries, so the ones the agent can start *by name* (the ``launch_app``
        aliases that resolve on this machine) must come before the long tail of
        Start-menu shortcuts.
        """
        launchable: list[str] = []
        try:
            from .tools import _APP_ALIASES, _resolve_app_command

            for friendly in _APP_ALIASES:
                if len(friendly) < 4:
                    continue
                command = _resolve_app_command(friendly)[0]
                if shutil.which(command) or Path(command).exists():
                    launchable.append(friendly.title())
        except Exception as exc:  # noqa: BLE001 - the app list is a bonus
            logger.debug("app alias probe failed: %s", exc)

        known = {name.lower() for name in launchable}
        shortcuts: dict[str, None] = {}
        for folder in self._start_menu_folders():
            for shortcut in _safe(lambda f=folder: list(f.rglob("*.lnk")), []) or []:
                stem = shortcut.stem.strip()
                lowered = stem.lower()
                if not stem or len(stem) > 40:
                    continue
                if any(noise in lowered for noise in _APP_NOISE):
                    continue
                if lowered in known or any(
                    lowered.endswith(f" - {name}") or lowered.startswith(f"{name} ")
                    for name in known
                ):
                    continue
                shortcuts.setdefault(stem, None)

        deduped: dict[str, None] = {}
        for name in launchable:
            deduped.setdefault(name, None)
        for name in sorted(shortcuts, key=str.lower):
            deduped.setdefault(name, None)
        return list(deduped)

    @staticmethod
    def _start_menu_folders() -> list[Path]:
        if sys.platform != "win32":
            return [Path("/usr/share/applications")] if Path("/usr/share/applications").is_dir() else []
        folders: list[Path] = []
        for key, suffix in (
            ("APPDATA", r"Microsoft\Windows\Start Menu\Programs"),
            ("PROGRAMDATA", r"Microsoft\Windows\Start Menu\Programs"),
        ):
            root = os.environ.get(key)
            if not root:
                continue
            candidate = Path(root) / suffix
            if candidate.is_dir():
                folders.append(candidate)
        return folders

    def _cli_tools(self) -> list[str]:
        return [name for name in _CLI_PROBES if _safe(lambda n=name: shutil.which(n))]

    # ------------------------------------------------------------ persistence
    def _is_stale(self, data: dict[str, Any]) -> bool:
        if not data or int(data.get("version", 0)) != PROFILE_VERSION:
            return True
        collected = str(data.get("collected_at") or "")
        if not collected:
            return True
        max_age_days = float(getattr(self._settings, "profile_max_age_days", 7))
        if max_age_days <= 0:
            return False
        try:
            stamp = datetime.fromisoformat(collected)
        except ValueError:
            return True
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - stamp).total_seconds() / 86400
        return age_days > max_age_days

    def _load(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        path = self.profile_file
        if path.exists():
            try:
                import json

                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    data = loaded
            except (OSError, ValueError) as exc:
                logger.warning("Could not read the profile file %s: %s", path, exc)
        if not data:
            data = {
                "version": PROFILE_VERSION,
                "collected_at": "",
                "notes": [],
                "recent_tasks": [],
            }
        # Fold in notes the user typed into the markdown mirror.
        markdown_notes = self._notes_from_markdown()
        if markdown_notes:
            merged = [str(item) for item in data.get("notes", [])]
            known = {item.lower() for item in merged}
            for note in markdown_notes:
                if note.lower() not in known:
                    merged.append(note)
                    known.add(note.lower())
            data["notes"] = merged
        return data

    def _notes_from_markdown(self) -> list[str]:
        """Read bullet points from the ``## Notes`` section of ``profile.md``."""
        path = self.report_file
        if not path.exists():
            return []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        notes: list[str] = []
        in_notes = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("## "):
                in_notes = stripped.lower().startswith("## notes")
                continue
            if not in_notes or not stripped:
                continue
            if stripped.startswith("<!--") or stripped.startswith("#"):
                continue
            if stripped.startswith(("-", "*")):
                text = stripped.lstrip("-* ").strip()
                if text:
                    notes.append(text)
        return notes

    def _save(self) -> None:
        import json

        try:
            self.profile_file.parent.mkdir(parents=True, exist_ok=True)
            self.profile_file.write_text(
                json.dumps(self._data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            self.report_file.write_text(self.as_markdown(), encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not save the profile: %s", exc)

    def _journal_note(self, message: str) -> None:
        system = getattr(self._journal, "system", None)
        if callable(system):
            system(message)


def build_user_context(settings: Any, journal: Any = None) -> UserContext:
    """Create the context file and make sure it exists on disk."""
    user_context = UserContext(settings, journal)
    if user_context.enabled:
        _safe(lambda: settings.ensure_dirs())
        user_context.prepare()
    return user_context
