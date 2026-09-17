"""Tests for the reflex cache's replay-eligibility guardrails."""

from pathlib import Path

from furti_ai.memory import MemoryManager
from furti_ai.models import Skill
from furti_ai.vision import VisionReflex


def test_get_skill_reports_miss_when_template_is_missing(tmp_path: Path):
    memory_file = tmp_path / "memory.json"
    memory = MemoryManager(memory_file)
    memory.save_skill(
        Skill(name="click_export", template_path=str(tmp_path / "gone.png"))
    )

    reloaded = MemoryManager(memory_file)

    assert reloaded.get_skill("Click the Export button") is None
    # The stale entry is kept for inspection/cleanup rather than deleted.
    assert [s.name for s in reloaded.list_skills()] == ["click_export"]


def test_get_skill_returns_skill_when_template_exists(tmp_path: Path):
    template = tmp_path / "export.png"
    template.write_bytes(b"not-a-real-png")
    memory_file = tmp_path / "memory.json"
    memory = MemoryManager(memory_file)
    memory.save_skill(Skill(name="click_export", template_path=str(template)))

    reloaded = MemoryManager(memory_file)

    assert reloaded.get_skill("click_export") is not None


def test_reflex_execute_returns_false_for_unreadable_template(tmp_path: Path):
    class ExplodingScreen:
        def capture(self):  # pragma: no cover - must not be reached
            raise AssertionError("a missing template must not capture the screen")

    reflex = VisionReflex(ExplodingScreen(), None)
    skill = Skill(name="click_export", template_path=str(tmp_path / "gone.png"))

    assert reflex.execute(skill) is False

def test_memory_clear_all_removes_reflexes_and_generated_crops(tmp_path):
    memory = MemoryManager(tmp_path / "memory.json")
    templates = tmp_path / "templates"
    templates.mkdir()
    generated = templates / "task_1.auto.png"
    generated.write_bytes(b"crop")
    skill_path = templates / "skill.png"
    skill_path.write_bytes(b"skill")
    skill = Skill(name="open_app", template_path=str(skill_path))
    memory.save_skill(skill)

    removed = memory.clear_all(templates)

    assert removed == 2
    assert memory.list_skills() == []
    assert not generated.exists()
    assert not skill_path.exists()


def _save_reflex(memory: MemoryManager, tmp_path: Path, name: str = "click_export"):
    """Compile a reflex the way the executor does: enabled defaults unset/off."""
    template = tmp_path / f"{name}.png"
    template.write_bytes(b"not-a-real-png")
    # Mirrors executor._compile_reflex, which never passes enabled explicitly.
    memory.save_skill(Skill(name=name, template_path=str(template)))
    return template


def test_newly_compiled_reflex_is_disabled(tmp_path: Path):
    memory = MemoryManager(tmp_path / "memory.json")
    _save_reflex(memory, tmp_path)

    assert memory.list_skills()[0].enabled is False


def test_disabled_reflex_is_not_replay_eligible(tmp_path: Path):
    memory = MemoryManager(tmp_path / "memory.json")
    _save_reflex(memory, tmp_path)

    # The replay path asks for an enabled reflex and gets a miss...
    assert memory.get_skill("click_export", require_enabled=True) is None
    # ...while stats-only lookups still find it, so counters keep working.
    assert memory.get_skill("click_export") is not None


def test_set_enabled_makes_reflex_replay_eligible_and_persists(tmp_path: Path):
    memory_file = tmp_path / "memory.json"
    memory = MemoryManager(memory_file)
    _save_reflex(memory, tmp_path)

    assert memory.set_enabled("click_export", True) is True

    reloaded = MemoryManager(memory_file)
    assert reloaded.get_skill("click_export", require_enabled=True) is not None
    assert reloaded.set_enabled("missing_reflex", True) is False


def test_enabled_flag_survives_serialization(tmp_path: Path):
    skill = Skill(name="click_export", template_path="t.png", enabled=True)

    restored = Skill.from_dict(skill.to_dict())

    assert restored.enabled is True
    # Payloads written before the flag existed stay opted out.
    legacy = skill.to_dict()
    legacy.pop("enabled")
    assert Skill.from_dict(legacy).enabled is False


def test_set_all_enabled_toggles_every_reflex(tmp_path: Path):
    memory = MemoryManager(tmp_path / "memory.json")
    _save_reflex(memory, tmp_path, "click_export")
    _save_reflex(memory, tmp_path, "open_app")

    assert memory.set_all_enabled(True) == 2
    assert all(skill.enabled for skill in memory.list_skills())

    assert memory.set_all_enabled(False) == 2
    assert not any(skill.enabled for skill in memory.list_skills())

    # A no-op flip reports nothing changed.
    assert memory.set_all_enabled(False) == 0


def test_disabled_reflex_falls_back_to_planning(tmp_path: Path):
    """The legacy replay path must not use a stored-but-disabled reflex."""
    from furti_ai.orchestrator import AgentOrchestrator

    memory = MemoryManager(tmp_path / "memory.json")
    _save_reflex(memory, tmp_path)

    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._memory = memory
    planned: list[str] = []
    orchestrator._plan_and_run = lambda command, name: planned.append(command) or True
    orchestrator._journal = None

    assert orchestrator.run("click_export") is True
    assert planned == ["click_export"]
