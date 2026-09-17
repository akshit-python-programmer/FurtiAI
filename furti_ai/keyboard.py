"""Keyboard shortcut parsing and dispatch.

``pyautogui.press()`` only understands a **single** key name. Handing it a
chord such as ``"ctrl+c"`` is a *silent* no-op: ``"ctrl+c"`` is not in
``pyautogui.KEYBOARD_KEYS``, so ``keyDown`` returns early and the shortcut
never reaches the application. Model output is also full of friendly
spellings pyautogui does not know (``escape``, ``return``, ``cmd``, ``pgup``,
``del``).

:class:`KeyboardController` closes both gaps: it normalises aliases onto
pyautogui's key names, splits a chord into ``(modifiers, key)``, and dispatches
modifiers through ``hotkey()`` -- which holds every key of the chord down
simultaneously -- so real shortcuts such as ``ctrl+shift+t`` or ``alt+tab``
actually fire. Unknown keys raise :class:`ChordError` instead of doing nothing,
so the executor's retry / re-plan path can react.

Chord syntax (case-insensitive, ``+`` or whitespace separated)::

    "enter"                    -> press("enter")
    "ctrl+c"                   -> hotkey("ctrl", "c")
    "CTRL + Shift + T"         -> hotkey("ctrl", "shift", "t")
    "alt+tab"                  -> hotkey("alt", "tab")
    "win+r"                    -> hotkey("win", "r")
    "ctrl++" / "ctrl+plus"     -> hotkey("ctrl", "+")
    ["ctrl", "alt", "delete"]  -> hotkey("ctrl", "alt", "delete")

A spaced ``+`` acts as a separator, so ``"ctrl + shift + t"`` is read as
``ctrl+shift+t``; write the plus key as ``+`` (glued) or ``plus``. Name
fragments a model split with a space are rejoined too, so ``"page down"`` and
``"caps lock"`` work as well as ``"pagedown"`` and ``"capslock"``.
"""

from __future__ import annotations

import re
import time
from typing import Any, Sequence

__all__ = [
    "ChordError",
    "KeyboardController",
    "CANONICAL_KEYS",
    "MODIFIER_KEYS",
    "canonical_key",
    "describe_chord",
    "parse_chord",
]

#: Canonical pyautogui names of the keys that act as chord modifiers.
MODIFIER_KEYS = frozenset(
    {"ctrl", "alt", "shift", "win", "winleft", "winright", "altleft", "altright"}
)

# Names that pyautogui spells differently from the vocabulary humans (and
# language models) use. pyautogui's own names are handled by CANONICAL_KEYS.
KEY_ALIASES: dict[str, str] = {
    # modifiers
    "control": "ctrl",
    "ctl": "ctrl",
    "option": "alt",
    "opt": "alt",
    "super": "win",
    "meta": "win",
    "cmd": "win",
    "command": "win",
    "windows": "win",
    "os": "win",
    # editing / navigation
    "return": "enter",
    "ret": "enter",
    "escape": "esc",
    "spacebar": "space",
    "space_bar": "space",
    "back_space": "backspace",
    "bksp": "backspace",
    "del": "delete",
    "ins": "insert",
    "pgup": "pageup",
    "page_up": "pageup",
    "pgdn": "pagedown",
    "pgdown": "pagedown",
    "page_down": "pagedown",
    "uparrow": "up",
    "arrowup": "up",
    "downarrow": "down",
    "arrowdown": "down",
    "leftarrow": "left",
    "arrowleft": "left",
    "rightarrow": "right",
    "arrowright": "right",
    "caps": "capslock",
    "caps_lock": "capslock",
    "home_key": "home",
    "end_key": "end",
    "prtsc": "printscreen",
    "prtscr": "printscreen",
    "print_screen": "printscreen",
    "num_lock": "numlock",
    "scroll_lock": "scrolllock",
    "break": "pause",
    # punctuation
    "plus": "+",
    "minus": "-",
    "hyphen": "-",
    "dash": "-",
    "equal": "=",
    "equals": "=",
    "comma": ",",
    "period": ".",
    "dot": ".",
    "slash": "/",
    "forward_slash": "/",
    "backslash": "\\",
    "back_slash": "\\",
    "semicolon": ";",
    "quote": "'",
    "apostrophe": "'",
    "singlequote": "'",
    "single_quote": "'",
    "doublequote": '"',
    "double_quote": '"',
    "backtick": "`",
    "grave": "`",
    "tilde": "~",
    "leftbracket": "[",
    "rightbracket": "]",
    "leftbrace": "{",
    "rightbrace": "}",
    "leftparen": "(",
    "rightparen": ")",
    "star": "*",
    "asterisk": "*",
    "percent": "%",
    "at": "@",
    "hash": "#",
    "dollar": "$",
    "caret": "^",
    "ampersand": "&",
    "underscore": "_",
    "pipe": "|",
    "colon": ":",
    "question": "?",
}

_FUNCTION_KEY = re.compile(r"^f(?:[1-9]|1[0-9]|2[0-4])$")

#: pyautogui's own multi-character key names. They resolve to themselves so
#: that re-parsing an already-normalised chord (and the input controller
#: echoing canonical names back) stays idempotent.
CANONICAL_KEYS = frozenset(
    {
        "accept",
        "add",
        "alt",
        "altleft",
        "altright",
        "apps",
        "backspace",
        "browserback",
        "browserfavorites",
        "browserforward",
        "browserhome",
        "browserrefresh",
        "browsersearch",
        "browserstop",
        "capslock",
        "clear",
        "convert",
        "ctrl",
        "ctrlleft",
        "ctrlright",
        "decimal",
        "delete",
        "divide",
        "down",
        "end",
        "enter",
        "esc",
        "execute",
        "final",
        "find",
        "help",
        "home",
        "insert",
        "junja",
        "kana",
        "kanji",
        "launchapp1",
        "launchapp2",
        "launchmail",
        "launchmediaselect",
        "left",
        "modechange",
        "multiply",
        "nexttrack",
        "nonconvert",
        "numlock",
        "pagedown",
        "pageup",
        "pause",
        "playpause",
        "prevtrack",
        "print",
        "printscreen",
        "process",
        "right",
        "scrolllock",
        "select",
        "separator",
        "shift",
        "shiftleft",
        "shiftright",
        "sleep",
        "space",
        "stop",
        "subtract",
        "tab",
        "up",
        "volumedown",
        "volumemute",
        "volumeup",
        "win",
        "winleft",
        "winright",
        "zoom",
    }
)

_KEY_HINT = (
    "use a single character, a function key (f1-f24), or a name such as "
    "ctrl, alt, shift, win, enter, esc, tab, space, backspace, delete, "
    "insert, home, end, pageup, pagedown, up, down, left, right"
)


class ChordError(ValueError):
    """Raised when a chord cannot be translated into keyboard events."""


def _clean(token: str) -> str:
    """Trim padding and stray JSON punctuation from one chord token."""
    token = token.strip()
    # Models occasionally punctuate a whole chord ("ctrl+c,"): only strip
    # trailing separators from multi-character tokens so the "," and ";"
    # keys themselves survive.
    if len(token) > 1:
        token = token.rstrip(",;").strip()
    return token


def _rejoin_split_key_names(tokens: list[str]) -> list[str]:
    """Rejoin names a model split with a space, e.g. ``"page down"``.

    ``"page up"``/``"caps lock"``/``"print screen"`` are written with a space
    far more often than with pyautogui's spelling, so two adjacent tokens are
    merged when the merged name is a known alias. Nothing else merges: the
    joined form of two real keys (``ctrl_c``) is not an alias.
    """
    merged: list[str] = []
    index = 0
    while index < len(tokens):
        if index + 1 < len(tokens):
            pair = f"{tokens[index]}_{tokens[index + 1]}"
            if pair.lower() in KEY_ALIASES:
                merged.append(pair)
                index += 2
                continue
        merged.append(tokens[index])
        index += 1
    return merged


def _tokenize(chord: str | Sequence[str]) -> list[str]:
    """Split a chord into raw key tokens, tolerating ``+`` and whitespace.

    A ``+`` with whitespace around it is a *separator* (``"ctrl + c"``), while
    a ``+`` glued to the previous token is the plus key itself
    (``"ctrl++"``); ``plus`` spells it unambiguously.
    """
    if isinstance(chord, (list, tuple, set, frozenset)):
        tokens: list[str] = []
        for item in chord:
            tokens.extend(_tokenize(str(item)))
        return tokens

    text = re.sub(r"\s*\+\s*", "+", str(chord or "").strip())
    if not text:
        return []
    tokens = []
    current: list[str] = []
    for char in text:
        if char == "+":
            if current:
                tokens.append("".join(current))
                current = []
            else:
                # A lone "+" between separators is the plus key itself
                # ("ctrl++" / "Ctrl + plus").
                tokens.append("+")
        elif char.isspace():
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(char)
    if current:
        tokens.append("".join(current))
    cleaned = [_clean(token) for token in tokens]
    return _rejoin_split_key_names([token for token in cleaned if token])


def canonical_key(token: str) -> str:
    """Translate one key name into the spelling pyautogui understands."""
    raw = (token or "").strip()
    if not raw:
        raise ChordError(f"empty key name in chord; {_KEY_HINT}")
    stripped = raw.strip("\"'")
    lowered = (stripped or raw).lower()
    alias = KEY_ALIASES.get(lowered)
    if alias is None:
        alias = KEY_ALIASES.get(re.sub(r"[\s\-]+", "_", lowered))
    if alias is not None:
        return alias
    if lowered in CANONICAL_KEYS or len(lowered) == 1 or _FUNCTION_KEY.match(lowered):
        return lowered
    if lowered.startswith("num") and lowered[3:].isdigit() and len(lowered) == 4:
        return lowered
    raise ChordError(f"unknown key name {raw!r}; {_KEY_HINT}")


def parse_chord(chord: str | Sequence[str]) -> list[str]:
    """Normalise a chord into pyautogui key names, modifiers first.

    The modifier order the caller used is preserved; only the base key is
    always moved to the end, because ``hotkey`` presses its arguments in
    order and releases them in reverse.
    """
    tokens = _tokenize(chord)
    if not tokens:
        raise ChordError("empty key chord")
    keys = [canonical_key(token) for token in tokens]
    modifiers: list[str] = []
    for key in keys:
        if key in MODIFIER_KEYS and key not in modifiers:
            modifiers.append(key)
    base = [key for key in keys if key not in MODIFIER_KEYS]
    if len(base) > 1:
        raise ChordError(
            "a chord takes one base key, got "
            + "+".join(base)
            + "; plan one key_press step per keystroke"
        )
    return modifiers + base


def describe_chord(chord: str | Sequence[str]) -> str:
    """Human-readable form of a chord, for logs and journals."""
    return "+".join(parse_chord(chord))


def _load_backend() -> Any:
    """Import pyautogui lazily so the module stays import-light."""
    try:
        import pyautogui
    except Exception as exc:  # pragma: no cover - platform dependent
        raise RuntimeError("pyautogui is required for keyboard input") from exc
    return pyautogui


class KeyboardController:
    """Send single keys and key chords through a pyautogui-like backend."""

    def __init__(self, backend: Any = None, *, interval: float = 0.0) -> None:
        self._backend = backend if backend is not None else _load_backend()
        self.interval = max(0.0, float(interval))
        known = getattr(self._backend, "KEYBOARD_KEYS", None) or ()
        # Empty means "the backend does not advertise its key table", in
        # which case alias/key-shape validation alone has to be enough.
        self._known_keys = {str(key).lower() for key in known}

    @property
    def known_keys(self) -> set[str]:
        """Key names the backend accepts (empty when it does not publish one)."""
        return set(self._known_keys)

    def parse(self, chord: str | Sequence[str]) -> list[str]:
        """Validate a chord and return its canonical key names."""
        keys = parse_chord(chord)
        if self._known_keys:
            unknown = [key for key in keys if key not in self._known_keys]
            if unknown:
                raise ChordError(
                    "the keyboard backend cannot send "
                    + ", ".join(repr(key) for key in unknown)
                )
        return keys

    def press(
        self,
        chord: str | Sequence[str],
        presses: int = 1,
        interval: float | None = None,
    ) -> list[str]:
        """Press a key or a whole chord, optionally repeated.

        A single key goes through ``press``; anything longer is dispatched as a
        chord so modifiers are held for the whole combination.
        """
        keys = self.parse(chord)
        count = max(1, int(presses))
        gap = self.interval if interval is None else max(0.0, float(interval))
        if len(keys) == 1:
            self._backend.press(keys[0], presses=count, interval=gap)
            return keys
        for index in range(count):
            if index and gap:
                time.sleep(gap)
            self._backend.hotkey(*keys)
        return keys

    def hotkey(self, chord: str | Sequence[str]) -> list[str]:
        """Hold every key of the chord down, then release in reverse order."""
        keys = self.parse(chord)
        self._backend.hotkey(*keys)
        return keys

    def key_down(self, chord: str | Sequence[str]) -> list[str]:
        """Hold the chord's keys down (caller must release with ``key_up``)."""
        keys = self.parse(chord)
        for key in keys:
            self._backend.keyDown(key)
        return keys

    def key_up(self, chord: str | Sequence[str]) -> list[str]:
        """Release the chord's keys in reverse order."""
        keys = self.parse(chord)
        for key in reversed(keys):
            self._backend.keyUp(key)
        return keys
