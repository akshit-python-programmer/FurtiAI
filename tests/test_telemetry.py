import time

import numpy as np

from furti_ai.config import Settings
from furti_ai.context import VisualContextManager
from furti_ai import orchestrator
from furti_ai.status import StatusWindow
from furti_ai.tasklog import TaskJournal


class FakeScreen:
    def capture(self):
        return np.zeros((40, 80, 3), dtype=np.uint8)


class FakeWidget:
    def __init__(self):
        self.values = {}
        self.text = ""

    def config(self, **kwargs):
        self.values.update(kwargs)

    def delete(self, *_args):
        self.text = ""

    def insert(self, _index, text):
        self.text = text

    def see(self, _index):
        pass


def test_journal_publishes_screenshot_ai_and_action_signals(tmp_path):
    journal = TaskJournal("telemetry", tmp_path)
    snapshots = []
    journal.status_sink = snapshots.append

    journal.screenshot(
        "2026-09-12 15:00:00",
        time.time(),
        (40, 80, 3),
        2,
        1,
    )
    journal.ai_output('{"steps":[{"action":"click"}]}', "deepseek-flash", "plan")
    journal.action("Step 1: click Save at (20, 20)")

    assert snapshots[0]["last_screenshot_at"] == "2026-09-12 15:00:00"
    assert snapshots[1]["ai_output"].startswith('{"steps"')
    assert snapshots[2]["current_action"].startswith("Step 1")
    assert snapshots[2]["action_signal"] == "trying"


def test_context_records_last_screenshot_time(tmp_path):
    journal = TaskJournal("capture", tmp_path)
    settings = Settings(workspace=tmp_path, screenshot_min_interval=60.0)
    context = VisualContextManager(
        FakeScreen(),
        text_detector=None,
        icon_matcher=None,
        settings=settings,
        journal=journal,
    )

    observation = context.observe("inspect the screen")

    assert observation.fresh is True
    assert observation.captured_at
    assert context.last_capture_at == observation.captured_at
    assert journal._events[-1].kind == "SCREENSHOT"


class TextDetectorThatDies:
    """Detector stand-in for a backend that fails after a successful load."""

    runtime_failure = "oneDNN path is unavailable in this build"

    def __init__(self):
        self.available = True

    def detect(self, _frame):
        self.available = False
        return []


def test_context_surfaces_text_detection_failure_once(tmp_path):
    journal = TaskJournal("ocr-failure", tmp_path)
    settings = Settings(workspace=tmp_path, screenshot_min_interval=0.0)
    context = VisualContextManager(
        FakeScreen(),
        text_detector=TextDetectorThatDies(),
        icon_matcher=None,
        settings=settings,
        journal=journal,
    )

    context.observe("inspect the screen", force_fresh=True)
    warnings = [event for event in journal._events if event.kind == "WARN"]
    assert len(warnings) == 1
    assert "text detection failed on every frame" in warnings[0].message

    # A later frame must not repeat the same warning.
    context._last_capture_ts = 0.0
    context.observe("inspect the screen", force_fresh=True)
    assert len([event for event in journal._events if event.kind == "WARN"]) == 1


def test_status_window_applies_visual_signals_without_tk(tmp_path):
    window = StatusWindow()
    window._widgets = {
        "task": FakeWidget(),
        "phase": FakeWidget(),
        "step": FakeWidget(),
        "activity": FakeWidget(),
        "screenshot": FakeWidget(),
        "action": FakeWidget(),
        "ai_output": FakeWidget(),
        "log": FakeWidget(),
        "stats": FakeWidget(),
    }

    window._apply(
        {
            "event_kind": "SCREENSHOT",
            "message": "Captured screenshot",
            "timestamp": "15:00:00",
            "last_screenshot_at": "2026-09-12 15:00:00",
            "last_screenshot_epoch": time.time(),
        }
    )
    window._apply(
        {
            "event_kind": "AI_OUTPUT",
            "message": "model output",
            "ai_output": '{"ok": true}',
        }
    )
    window._apply(
        {
            "event_kind": "ACTION",
            "message": "click Save",
            "current_action": "click Save",
            "action_signal": "trying",
        }
    )

    assert "Last screenshot:" in window._widgets["screenshot"].values["text"]
    assert window._widgets["ai_output"].text == '{"ok": true}'
    assert window._widgets["action"].values["text"] == "Current action: click Save"
    assert "[ACTING]" in window._widgets["activity"].values["text"]


def test_deepseek_flash_is_selectable_for_image_calls(monkeypatch):
    class FakeDeepSeek:
        def __init__(self, settings, model=None, usage_callback=None):
            self.settings = settings
            self.model = model or settings.deepseek_model

    class FakeGemini:
        def __init__(self, settings, model=None, usage_callback=None):
            self.model = model

    monkeypatch.setattr(orchestrator, "DeepSeekClient", FakeDeepSeek)
    monkeypatch.setattr(orchestrator, "GeminiClient", FakeGemini)
    settings = Settings(
        deepseek_api_key="deepseek-key",
        google_api_key="google-key",
        llm_provider="deepseek",
    )

    client = orchestrator._make_llm(settings, None, None)

    assert isinstance(client, FakeDeepSeek)
    assert client.model == "deepseek-flash"


def test_auto_provider_prefers_deepseek_when_both_keys_exist(monkeypatch):
    class FakeDeepSeek:
        def __init__(self, settings, model=None, usage_callback=None):
            self.model = model or settings.deepseek_model

    class FakeGemini:
        def __init__(self, settings, model=None, usage_callback=None):
            self.model = model

    monkeypatch.setattr(orchestrator, "DeepSeekClient", FakeDeepSeek)
    monkeypatch.setattr(orchestrator, "GeminiClient", FakeGemini)
    settings = Settings(
        deepseek_api_key="deepseek-key",
        google_api_key="google-key",
        llm_provider="auto",
    )

    client = orchestrator._make_llm(settings, None, None)

    assert isinstance(client, FakeDeepSeek)
    assert client.model == "deepseek-flash"
