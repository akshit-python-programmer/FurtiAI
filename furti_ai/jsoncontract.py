"""The strict JSON contract for model-authored control payloads.

Every screen action the agent performs arrives as model output, so a loosely
parsed answer is a *movement* risk: a stray integer, a string where a number was
expected, or a NaN becomes a cursor jump, a wrong click or a drag that grabs the
wrong thing. This module is the single place those answers are read:

* :func:`extract_json_object` takes the model's text and returns exactly one
  JSON object -- tolerating the fences and prose a model wraps around it, but
  refusing NaN/Infinity, arrays, and anything it cannot parse cleanly;
* :func:`as_optional_int`, :func:`as_pixel`, :func:`as_optional_bool` and
  :func:`as_text` validate individual control fields, raising
  :class:`LLMJsonError` instead of guessing a value.

Callers are expected to treat a raised :class:`LLMJsonError` as "do not act":
the planner escalates to the smarter model and re-asks, and a step that reaches
the executor with a broken payload is refused rather than dispatched.
"""

from __future__ import annotations

import json
import math
from typing import Any, Optional

#: Largest pixel value the agent will accept from a model. Real coordinates are
#: ``full_capture`` pixels of the attached screenshot, so anything past this is
#: a mis-scaled or hallucinated number rather than a target.
MAX_PIXEL = 20000

#: Longest model excerpt quoted back in an error message.
_EXCERPT = 200


class LLMJsonError(ValueError):
    """The model's answer did not satisfy the control-output contract."""


def _reject_constant(name: str) -> Any:
    """``json.loads`` hook: NaN/Infinity are not JSON and must never be parsed."""
    raise LLMJsonError(f"{name} is not valid JSON; use a plain finite number")


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped[3:]
    if body[:4].lower() == "json":
        body = body[4:]
    end = body.rfind("```")
    if end != -1:
        body = body[:end]
    return body.strip()


def _first_object(text: str) -> Optional[str]:
    """Slice out the first balanced ``{...}`` block, ignoring braces in strings."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def extract_json_object(raw: Any) -> dict[str, Any]:
    """Read the single JSON object a model was asked for, or raise.

    Fenced answers (`````json ... `````) and objects surrounded by prose are
    accepted, because rejecting those would waste a whole model round trip for
    a formatting nit. Anything else -- empty output, a bare array, a truncated
    object, ``NaN``/``Infinity`` -- raises :class:`LLMJsonError`.
    """
    text = _strip_fences(str(raw or ""))
    if not text:
        raise LLMJsonError(
            "the model returned an empty response where a JSON object was required"
        )

    payload: Any = None
    try:
        payload = json.loads(text, parse_constant=_reject_constant)
    except LLMJsonError:
        raise
    except ValueError:
        candidate = _first_object(text)
        if candidate is not None:
            try:
                payload = json.loads(candidate, parse_constant=_reject_constant)
            except LLMJsonError:
                raise
            except ValueError as exc:
                raise LLMJsonError(
                    f"the model's JSON object could not be parsed: {exc}"
                ) from exc

    if payload is None:
        raise LLMJsonError(
            f"no JSON object in the model response: {text[:_EXCERPT]!r}"
        )
    if not isinstance(payload, dict):
        raise LLMJsonError(
            "expected exactly one JSON object, got "
            f"{type(payload).__name__}: {text[:_EXCERPT]!r}"
        )
    return payload


def _to_number(value: Any, *, field: str) -> float:
    """Coerce one numeric field, refusing booleans, NaN and junk."""
    if isinstance(value, bool) or value is None:
        raise LLMJsonError(f"{field} must be a number, got {value!r}")
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        try:
            number = float(str(value).strip())
        except (TypeError, ValueError) as exc:
            raise LLMJsonError(f"{field} must be a number, got {value!r}") from exc
    if math.isnan(number) or math.isinf(number):
        raise LLMJsonError(f"{field} must be a finite number, got {value!r}")
    return number


def as_optional_int(
    value: Any,
    *,
    field: str,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> Optional[int]:
    """Read an optional integer control field; ``None`` when it was not sent."""
    if value is None or value == "":
        return None
    rounded = int(round(_to_number(value, field=field)))
    if minimum is not None and rounded < minimum:
        raise LLMJsonError(
            f"{field}={rounded} is below the allowed minimum {minimum}"
        )
    if maximum is not None and rounded > maximum:
        raise LLMJsonError(
            f"{field}={rounded} is above the allowed maximum {maximum}"
        )
    return rounded


def as_pixel(value: Any, *, field: str, allow_zero: bool = True) -> Optional[int]:
    """Read one coordinate: a non-negative ``full_capture`` pixel."""
    minimum = 0 if allow_zero else 1
    return as_optional_int(value, field=field, minimum=minimum, maximum=MAX_PIXEL)


def as_optional_bool(value: Any) -> Optional[bool]:
    """Read a tri-state flag; ``None`` means the field was not stated.

    Models emit booleans as ``true``, ``"true"``, ``"yes"`` and ``1`` -- but also
    as ``"false"``, which is *truthy* in Python. This is the one place that
    ambiguity is resolved.
    """
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    return None


def as_text(value: Any, *, field: str, required: bool = False) -> Optional[str]:
    """Read a text field, refusing containers where a string was required."""
    if value is None:
        if required:
            raise LLMJsonError(f"{field} is required")
        return None
    if isinstance(value, (dict, list, tuple)):
        raise LLMJsonError(f"{field} must be a string, got {type(value).__name__}")
    text = str(value)
    if required and not text.strip():
        raise LLMJsonError(f"{field} is required and cannot be empty")
    return text


def as_mapping(value: Any, *, field: str) -> dict[str, Any]:
    """Read an object field, refusing a list/string where an object was needed."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise LLMJsonError(f"{field} must be a JSON object, got {type(value).__name__}")
    return value


def as_sequence(value: Any, *, field: str) -> list[Any]:
    """Read an array field, refusing a bare scalar."""
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise LLMJsonError(f"{field} must be a JSON array, got {type(value).__name__}")
    return list(value)
