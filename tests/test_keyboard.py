import pytest

from furti_ai.keyboard import (
    ChordError,
    KeyboardController,
    canonical_key,
    describe_chord,
    parse_chord,
)


class FakeBackend:
    """Minimal pyautogui stand-in that records keyboard traffic."""

    def __init__(self, keyboard_keys=None):
        self.events = []
        self.KEYBOARD_KEYS = list(keyboard_keys or [])

    def press(self, key, presses=1, interval=0.0):
        self.events.append(("press", key, presses, interval))

    def hotkey(self, *keys):
        self.events.append(("hotkey", *keys))

    def keyDown(self, key):
        self.events.append(("down", key))

    def keyUp(self, key):
        self.events.append(("up", key))


def test_canonical_key_normalises_friendly_names():
    assert canonical_key("Escape") == "esc"
    assert canonical_key("return") == "enter"
    assert canonical_key("cmd") == "win"
    assert canonical_key("Command") == "win"
    assert canonical_key("option") == "alt"
    assert canonical_key("pgup") == "pageup"
    assert canonical_key("page_down") == "pagedown"
    assert canonical_key("del") == "delete"
    assert canonical_key("space bar") == "space"
    assert canonical_key("uparrow") == "up"
    assert canonical_key("Caps_Lock") == "capslock"


def test_canonical_key_keeps_single_characters_and_function_keys():
    assert canonical_key("A") == "a"
    assert canonical_key("7") == "7"
    assert canonical_key("+") == "+"
    assert canonical_key("plus") == "+"
    assert canonical_key("F5") == "f5"
    assert canonical_key("f24") == "f24"
    assert canonical_key("num7") == "num7"


def test_canonical_key_rejects_unknown_names():
    with pytest.raises(ChordError):
        canonical_key("banana")

    with pytest.raises(ChordError):
        canonical_key("f25")


def test_parse_chord_handles_separators_case_and_order():
    assert parse_chord("CTRL + C") == ["ctrl", "c"]
    assert parse_chord("alt+tab") == ["alt", "tab"]
    assert parse_chord("win+r") == ["win", "r"]
    # The base key is always moved behind the modifiers.
    assert parse_chord("shift+ctrl+t") == ["shift", "ctrl", "t"]
    assert parse_chord("ctrl shift t") == ["ctrl", "shift", "t"]
    assert parse_chord(["ctrl", "alt", "Delete"]) == ["ctrl", "alt", "delete"]
    assert parse_chord("ctrl+c,") == ["ctrl", "c"]


def test_parse_chord_keeps_the_plus_key():
    assert parse_chord("ctrl++") == ["ctrl", "+"]
    assert parse_chord("ctrl + +") == ["ctrl", "+"]


def test_parse_chord_rejoins_key_names_split_by_a_space():
    assert parse_chord("page down") == ["pagedown"]
    assert parse_chord("ctrl + page up") == ["ctrl", "pageup"]
    assert parse_chord("caps lock") == ["capslock"]
    assert parse_chord("print screen") == ["printscreen"]
    # Two genuine keys must never be fused into one.
    assert parse_chord("ctrl+c") == ["ctrl", "c"]
    assert parse_chord("alt+shift") == ["alt", "shift"]


def test_parse_chord_allows_a_lone_modifier():
    assert parse_chord("ctrl") == ["ctrl"]


def test_parse_chord_rejects_empty_and_multi_base_chords():
    with pytest.raises(ChordError):
        parse_chord("")

    with pytest.raises(ChordError):
        parse_chord("ctrl+a+b")


def test_describe_chord_is_log_friendly():
    assert describe_chord("Control+Escape") == "ctrl+esc"
    assert describe_chord("cmd + shift + p") == "win+shift+p"


def test_press_dispatches_single_keys_through_press():
    backend = FakeBackend()
    keyboard = KeyboardController(backend, interval=0.05)

    keys = keyboard.press("enter")

    assert keys == ["enter"]
    assert backend.events == [("press", "enter", 1, 0.05)]


def test_press_dispatches_chords_through_hotkey():
    backend = FakeBackend()
    keyboard = KeyboardController(backend)

    keyboard.press("ctrl+c")

    assert backend.events == [("hotkey", "ctrl", "c")]


def test_press_repeats_chords_and_single_keys():
    backend = FakeBackend()
    keyboard = KeyboardController(backend)

    keyboard.press("ctrl+c", presses=2)
    keyboard.press("down", presses=3)

    assert backend.events == [
        ("hotkey", "ctrl", "c"),
        ("hotkey", "ctrl", "c"),
        ("press", "down", 3, 0.0),
    ]


def test_hotkey_holds_all_keys_together():
    backend = FakeBackend()
    keyboard = KeyboardController(backend)

    assert keyboard.hotkey("ctrl+shift+t") == ["ctrl", "shift", "t"]
    assert backend.events == [("hotkey", "ctrl", "shift", "t")]


def test_key_down_and_key_up_use_reverse_order():
    backend = FakeBackend()
    keyboard = KeyboardController(backend)

    keyboard.key_down("ctrl+shift")
    keyboard.key_up("ctrl+shift")

    assert backend.events == [
        ("down", "ctrl"),
        ("down", "shift"),
        ("up", "shift"),
        ("up", "ctrl"),
    ]


def test_parse_validates_against_the_backend_key_table():
    backend = FakeBackend(keyboard_keys=["ctrl", "c", "enter"])
    keyboard = KeyboardController(backend)

    assert keyboard.known_keys == {"ctrl", "c", "enter"}
    with pytest.raises(ChordError):
        keyboard.press("ctrl+q")


def test_backend_without_key_table_skips_table_validation():
    backend = FakeBackend()
    keyboard = KeyboardController(backend)

    assert keyboard.known_keys == set()
    assert keyboard.press("ctrl+q") == ["ctrl", "q"]
