"""The RAG / cache layer.

``MemoryManager`` stores every *compiled skill* -- a cropped template image
plus metadata (action, expected bounding box, screen size, usage stats).
Before any reasoning happens, the orchestrator asks this layer whether the
task has been seen before, so repeated actions cost zero tokens and run in
single-digit milliseconds.

The current backend is a small JSON file. The interface is deliberately narrow
so a vector store (chromadb) can replace it for fuzzy, semantic recall without
changing any caller.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from .models import Skill

logger = logging.getLogger(__name__)


def _template_exists(skill: Skill) -> bool:
    """True when the skill's stored template image is still on disk."""
    raw = str(skill.template_path or "").strip()
    return bool(raw) and Path(raw).is_file()


class MemoryManager:
    """Persistent key/value skill cache backed by a JSON file."""

    def __init__(self, memory_file: Path) -> None:
        self._memory_file = Path(memory_file)
        self._skills: dict[str, Skill] = {}
        self._load()

    # ------------------------------------------------------------------ util
    @staticmethod
    def normalize_name(raw: str) -> str:
        """Turn a free-text command into a stable cache key, e.g.

        ``"Click the Export button"`` -> ``"click_the_export_button"``.
        """
        slug = re.sub(r"[^a-z0-9]+", "_", raw.strip().lower()).strip("_")
        return slug or "unnamed_task"

    # --------------------------------------------------------------- lookup
    def get_skill(
        self, name: str, *, require_enabled: bool = False
    ) -> Optional[Skill]:
        """Return the cached skill for a command, or ``None`` if it is novel.

        A skill whose template image is gone cannot be replayed, so it is
        reported as a cache miss (the caller re-plans) instead of failing the
        reflex path.

        ``require_enabled`` is how the *replay* path asks: reflexes are compiled
        disabled, so a stored-but-disabled reflex must not take over from the
        planner. Callers that only want stats (``record_success``) leave it off.
        """
        skill = self._skills.get(self.normalize_name(name))
        if skill is None:
            return None
        if require_enabled and not bool(getattr(skill, "enabled", False)):
            logger.info(
                "Skill %r is disabled; treating it as a cache miss.", skill.name
            )
            return None
        if not _template_exists(skill):
            logger.warning(
                "Ignoring cached skill %r: template %s is missing; re-planning.",
                skill.name,
                skill.template_path,
            )
            return None
        return skill

    def set_enabled(self, name: str, enabled: bool) -> bool:
        """Turn one reflex on or off in the cache. Returns False if unknown."""
        skill = self._skills.get(self.normalize_name(name))
        if skill is None:
            return False
        skill.enabled = bool(enabled)
        self._save()
        logger.info("Reflex %r is now %s.", skill.name, "enabled" if enabled else "disabled")
        return True

    def set_all_enabled(self, enabled: bool) -> int:
        """Turn every reflex on or off at once; returns how many changed."""
        changed = 0
        for skill in self._skills.values():
            if bool(getattr(skill, "enabled", False)) != bool(enabled):
                skill.enabled = bool(enabled)
                changed += 1
        if changed:
            self._save()
        return changed

    def has_skill(self, name: str) -> bool:
        """Return True if a compiled skill already exists for the command."""
        return self.normalize_name(name) in self._skills

    def list_skills(self) -> list[Skill]:
        """Return every cached skill (for inspection / debugging)."""
        return list(self._skills.values())

    # -------------------------------------------------------------- storage
    def save_skill(self, skill: Skill) -> None:
        """Insert or overwrite a skill and persist immediately."""
        self._skills[skill.name] = skill
        self._save()
        logger.debug("Saved skill %r (%d total).", skill.name, len(self._skills))

    def record_success(self, name: str) -> None:
        """Bump usage counters after a reflex fires successfully."""
        skill = self.get_skill(name)
        if skill is None:
            return
        skill.record_success()
        self._save()

    def forget(self, name: str) -> bool:
        """Remove a skill from the cache. Returns True if it existed."""
        key = self.normalize_name(name)
        if key in self._skills:
            del self._skills[key]
            self._save()
            return True
        return False

    def clear_all(self, templates_dir: Path | None = None) -> int:
        """Remove every compiled reflex and its generated template files."""
        removed = 0
        for skill in self._skills.values():
            template = Path(str(skill.template_path or ""))
            if template.is_file():
                try:
                    template.unlink()
                    removed += 1
                except OSError:
                    logger.warning("Could not remove reflex template %s", template)
        # Executor-only anchor crops are not referenced by the memory JSON but
        # are still generated cache artifacts.
        generated_dir = Path(templates_dir) if templates_dir is not None else self._memory_file.parent
        for template in generated_dir.glob("*.auto.png"):
            try:
                template.unlink()
                removed += 1
            except OSError:
                logger.warning("Could not remove generated anchor %s", template)
        self._skills.clear()
        self._save()
        return removed

    # ---------------------------------------------------------- persistence
    def _load(self) -> None:
        if not self._memory_file.exists():
            logger.info("No memory file at %s; starting with an empty cache.", self._memory_file)
            return
        try:
            payload = json.loads(self._memory_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read memory file %s: %s", self._memory_file, exc)
            return
        for entry in payload.get("skills", []):
            try:
                skill = Skill.from_dict(entry)
                self._skills[skill.name] = skill
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping corrupt skill entry (%r): %s", entry, exc)

    def _save(self) -> None:
        self._memory_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {"skills": [s.to_dict() for s in self._skills.values()]}
        self._memory_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
