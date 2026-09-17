import sys
import time
from types import SimpleNamespace

import pytest

from furti_ai.controller import PyAutoGuiInput
from furti_ai.keyboard import ChordError
from furti_ai.screen import PyAutoGuiScreen


class FakePyAutoGUI(SimpleNamespace):
    def __init__(self):
        super().__init__(
            PAUSE=None,
            FAILSAFE=None,
            writes=[],
            events=[],
            cursor=(0, 0),
        )

    def write(self, text, interval=0.0):
        self.writes.append((text, interval))

    def position(self):
        return self.cursor

    def moveTo(self, x, y, duration=0.0):
        self.cursor = (x, y)
        self.events.append(("move", x, y, duration))

    def mouseDown(self, button="left"):
        self.events.append(("mouseDown", button))

    def mouseUp(self, button="left"):
        self.events.append(("mouseUp", button))

    def press(self, key, presses=1, interval=0.0):
        self.events.append(("press", key, presses, interval))

    def hotkey(self, *keys):
        self.events.append(("hotkey", *keys))

    def click(self, button="left", clicks=1, interval=0.0):
        self.events.append(("click", button, clicks))

    def doubleClick(self, button="left"):
        self.events.append(("doubleClick", button))

    def rightClick(self):
        self.events.append(("rightClick",))

    def keyDown(self, key):
        self.events.append(("keyDown", key))

    def keyUp(self, key):
        self.events.append(("keyUp", key))


def make_input(monkeypatch, **kwargs) -> tuple[PyAutoGuiInput, FakePyAutoGUI]:
    fake = FakePyAutoGUI()
    monkeypatch.setitem(sys.modules, "pyautogui", fake)
    return PyAutoGuiInput(**kwargs), fake


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


# ------------------------------------------------------------ keyboard input
def test_press_key_sends_chords_as_hotkeys(monkeypatch):
    controller, fake = make_input(monkeypatch, pause=0.0, teleport_cursor=True)

    controller.press_key("ctrl+c")

    # pyautogui.press("ctrl+c") is a silent no-op, so a chord must be a hotkey.
    assert fake.events == [("hotkey", "ctrl", "c")]


def test_press_key_sends_aliases_and_repeats_single_keys(monkeypatch):
    controller, fake = make_input(monkeypatch, pause=0.0, teleport_cursor=True)

    controller.press_key("Enter", presses=2)

    assert fake.events == [("press", "enter", 2, 0.0)]


def test_press_key_repeats_chords(monkeypatch):
    controller, fake = make_input(monkeypatch, pause=0.0, teleport_cursor=True)

    controller.press_key("alt+tab", presses=2)

    assert fake.events == [
        ("hotkey", "alt", "tab"),
        ("hotkey", "alt", "tab"),
    ]


def test_press_key_rejects_unknown_keys(monkeypatch):
    controller, _fake = make_input(monkeypatch, pause=0.0, teleport_cursor=True)

    with pytest.raises(ChordError):
        controller.press_key("ctrl+banana")


def test_key_down_and_up_hold_then_release_in_reverse(monkeypatch):
    controller, fake = make_input(monkeypatch, pause=0.0, teleport_cursor=True)

    controller.key_down("ctrl+shift+t")
    controller.key_up("ctrl+shift+t")

    assert fake.events == [
        ("keyDown", "ctrl"),
        ("keyDown", "shift"),
        ("keyDown", "t"),
        ("keyUp", "t"),
        ("keyUp", "shift"),
        ("keyUp", "ctrl"),
    ]


# ---------------------------------------------------------------- drag input
def test_drag_presses_and_releases_with_a_move_between(monkeypatch):
    controller, fake = make_input(
        monkeypatch, pause=0.0, teleport_cursor=False, drag_duration=0.5
    )

    controller.drag(10, 20, 210, 220, duration=0.3)

    assert fake.events[0] == ("move", 10, 20, fake.events[0][3])
    assert fake.events[1] == ("mouseDown", "left")
    assert fake.events[2] == ("move", 210, 220, 0.3)
    assert fake.events[3] == ("mouseUp", "left")


def test_drag_holds_and_releases_extra_keys(monkeypatch):
    controller, fake = make_input(monkeypatch, pause=0.0, teleport_cursor=True)

    controller.drag(0, 0, 50, 50, button="right", hold_keys="shift")

    assert fake.events == [
        ("keyDown", "shift"),
        ("move", 0, 0, 0.0),
        ("mouseDown", "right"),
        ("move", 50, 50, 0.0),
        ("mouseUp", "right"),
        ("keyUp", "shift"),
    ]


def test_drag_releases_the_button_when_the_drop_move_fails(monkeypatch):
    controller, fake = make_input(monkeypatch, pause=0.0, teleport_cursor=False)
    calls = {"count": 0}
    real_move = fake.moveTo

    def explode_after_grab(x, y, duration=0.0):
        calls["count"] += 1
        if calls["count"] > 1:
            raise RuntimeError("cursor move failed")
        return real_move(x, y, duration)

    monkeypatch.setattr(fake, "moveTo", explode_after_grab)

    with pytest.raises(RuntimeError):
        controller.drag(0, 0, 80, 80)

    # The button must never stay stuck down after a failed drop move.
    assert [event[0] for event in fake.events] == ["move", "mouseDown", "mouseUp"]
    assert fake.events[1:] == [("mouseDown", "left"), ("mouseUp", "left")]


def test_drag_rejects_unknown_buttons(monkeypatch):
    controller, fake = make_input(monkeypatch, pause=0.0, teleport_cursor=True)

    with pytest.raises(ValueError):
        controller.drag(0, 0, 10, 10, button="thumb")

    assert fake.events == []


# ------------------------------------------------------- click settling
def test_click_settles_between_the_move_and_the_press(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    controller, fake = make_input(monkeypatch, pause=0.0, click_settle=0.25)

    controller.click(30, 40)

    # Without the settle the press races the move and lands on whatever was
    # hovered before, which reads as "the cursor moved but nothing clicked".
    assert sleeps == [0.25]
    assert fake.events[0] == ("move", 30, 40, fake.events[0][3])
    assert fake.events[1] == ("click", "left", 1)


def test_click_forwards_button_and_repeat_count(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    controller, fake = make_input(monkeypatch, pause=0.0, click_settle=0.25)

    controller.click(1, 2, button="right", clicks=2)

    assert fake.events[1] == ("click", "right", 2)


def test_click_settle_can_be_disabled(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    controller, _fake = make_input(monkeypatch, pause=0.0, click_settle=0.0)

    controller.click(1, 2)

    assert sleeps == []


def test_double_click_and_right_click_settle_too(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    controller, fake = make_input(monkeypatch, pause=0.0, click_settle=0.1)

    controller.double_click(5, 6)
    controller.right_click(7, 8)

    assert sleeps == [0.1, 0.1]
    assert ("doubleClick", "left") in fake.events
    assert ("rightClick",) in fake.events


def test_move_to_repositions_without_pressing_a_button(monkeypatch):
    controller, fake = make_input(monkeypatch, pause=0.0, teleport_cursor=True)

    controller.move_to(11, 22)

    assert fake.events == [("move", 11, 22, 0.0)]


def test_drag_settles_before_the_grab_and_after_the_drop(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    controller, fake = make_input(
        monkeypatch, pause=0.0, teleport_cursor=True, click_settle=0.1
    )

    controller.drag(0, 0, 60, 60)

    # The grab point must be hovered before mouseDown, and the drop must be
    # processed before a following click can be issued.
    assert sleeps == [0.1, 0.1]
    assert [event[0] for event in fake.events] == [
        "move",
        "mouseDown",
        "move",
        "mouseUp",
    ]


# ---------------------------------------------------------- corner abort
class CornerPyAutoGUI(FakePyAutoGUI):
    """Backend with geometry, so the abort corners are known."""

    def __init__(self, cursor=(0, 0), size=(100, 80)):
        super().__init__()
        self.FAILSAFE = True
        self.FAILSAFE_POINTS = ((0, 0), (0, 79), (99, 0), (99, 79))
        self.cursor = cursor
        self._size = size

    def size(self):
        return self._size


def make_corner_input(monkeypatch, cursor=(0, 0)):
    fake = CornerPyAutoGUI(cursor=cursor)
    monkeypatch.setitem(sys.modules, "pyautogui", fake)
    warnings: list[str] = []
    return PyAutoGuiInput(warn=warnings.append), fake, warnings


def test_corner_parked_pointer_is_released_before_moving(monkeypatch):
    """A corner park makes pyautogui raise on every call: it must be nudged out."""
    controller, fake, warnings = make_corner_input(monkeypatch)

    controller.move_to(500, 400)

    assert fake.events[0][:3] == ("move", 1, 1)
    assert fake.events[-1][:3] == ("move", 500, 400)
    assert fake.FAILSAFE is True  # restored once the pointer is clear
    assert len(warnings) == 1 and "abort corner" in warnings[0]


def test_corner_recovery_warns_only_once(monkeypatch):
    controller, fake, warnings = make_corner_input(monkeypatch, cursor=(0, 0))

    controller.move_to(0, 0)  # deliberate corner target: allowed
    controller.move_to(50, 50)  # parked again -> released, but warned once

    assert len(warnings) == 1
    assert fake.events[-1][:3] == ("move", 50, 50)


def test_deliberate_corner_target_suspends_the_abort(monkeypatch):
    """The Start button lives in a corner, so a corner target must be allowed."""
    controller, fake, warnings = make_corner_input(monkeypatch, cursor=(50, 40))

    controller.click(0, 0)

    assert ("click", "left", 1) in fake.events
    assert fake.FAILSAFE is True
    assert warnings == []
