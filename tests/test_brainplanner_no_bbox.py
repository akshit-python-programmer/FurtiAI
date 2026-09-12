from pathlib import Path
import numpy as np

from furti_ai.brain import BrainPlanner, MockLLMClient
from furti_ai.config import Settings
from furti_ai.memory import MemoryManager
from furti_ai.models import ActionPlan, BoundingBox


class DummyScreen:
    def capture(self):
        return np.zeros((64, 64, 3), dtype=np.uint8)


def test_brainplanner_returns_none_when_plan_lacks_bbox(tmp_path):
    settings = Settings(workspace=Path(tmp_path))
    settings.ensure_dirs()

    memory = MemoryManager(settings.memory_file)
    llm = MockLLMClient(
        plan={
            "action": "click",
            "description": "Click the close button in the top-right corner.",
        }
    )
    brain = BrainPlanner(llm, DummyScreen(), memory, settings)

    skill = brain.plan("Click on the Close button in the top right corner of the window", "close_window")

    assert skill is None
    assert not any((settings.templates_dir).glob("*.png"))


def test_actionplan_accepts_bbox_list_shape():
    plan = ActionPlan.from_dict(
        {
            "action": "click",
            "bbox": [10, 20, 30, 40],
            "description": "Click the top-left area.",
        }
    )

    assert plan.action == "click"
    assert plan.bbox == BoundingBox(10, 20, 30, 40)


def test_brainplanner_normalizes_model_corner_bbox():
    raw_plan = {
        "action": "click",
        "bbox": [17, 442, 33, 468],
        "description": "Click the icon.",
    }

    normalized = BrainPlanner._normalize_model_bbox(
        raw_plan,
        screen_width=1920,
        screen_height=1080,
    )

    assert normalized["bbox"] == {"x": 17, "y": 442, "width": 16, "height": 26}
