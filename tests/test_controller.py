import sys
from types import SimpleNamespace

from furti_ai.controller import PyAutoGuiInput
from furti_ai.screen import PyAutoGuiScreen


class FakePyAutoGUI(SimpleNamespace):
    def __init__(self):
        super().__init__(
            PAUSE=None,
            FAILSAFE=None,
            writes=[],
        )

    def write(self, text, interval=0.0):
        self.writes.append((text, interval))


def test_typing_uses_windows_friendly_interval(monkeypatch):
    fake = FakePyAutoGUI()
    monkeypatch.setitem(sys.modules, "pyautogui", fake)

    controller = PyAutoGuiInput(
        pause=0.03,
        move_duration=0.25,
        typing_interval=0.02,
    )
    controller.type_text("hello")

    assert fake.PAUSE == 0.03
    assert fake.writes == [("hello", 0.02)]


def test_screen_maps_physical_capture_pixels_to_logical_mouse_pixels(monkeypatch):
    fake = SimpleNamespace(size=lambda: SimpleNamespace(width=1280, height=720))
    monkeypatch.setitem(sys.modules, "pyautogui", fake)

    screen = PyAutoGuiScreen()

    assert screen.to_input_point((960, 540), (1080, 1920, 3)) == (640, 360)
