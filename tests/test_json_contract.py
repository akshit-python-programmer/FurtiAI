"""The strict JSON control contract.

Every screen action the agent takes comes from model output, so a loosely read
payload is a *movement* risk. These tests pin the contract: the reader accepts
the wrappers a model adds (fences, prose) but refuses anything it would have to
guess at, and the per-field readers reject nonsense instead of coercing it.
"""

import pytest

from furti_ai.jsoncontract import (
    LLMJsonError,
    as_mapping,
    as_optional_bool,
    as_optional_int,
    as_pixel,
    as_sequence,
    as_text,
    extract_json_object,
)


# ------------------------------------------------------------------ reading
def test_a_plain_object_is_read():
    assert extract_json_object('{"a": 1}') == {"a": 1}


def test_fenced_json_is_read():
    raw = '```json\n{"action": "click", "x": 10}\n```'
    assert extract_json_object(raw)["action"] == "click"


def test_prose_around_the_object_is_ignored():
    raw = 'Sure! Here is the plan:\n{"goal": "g", "steps": []}\nLet me know.'
    assert extract_json_object(raw) == {"goal": "g", "steps": []}


def test_braces_inside_strings_do_not_end_the_object():
    raw = 'prefix {"note": "use {braces} carefully", "x": 1} suffix'
    assert extract_json_object(raw)["note"] == "use {braces} carefully"


def test_an_empty_response_is_refused():
    with pytest.raises(LLMJsonError, match="empty response"):
        extract_json_object("   ")


def test_an_array_is_refused():
    with pytest.raises(LLMJsonError, match="expected exactly one JSON object"):
        extract_json_object('[{"action": "click"}]')


def test_prose_without_an_object_is_refused():
    with pytest.raises(LLMJsonError, match="no JSON object"):
        extract_json_object("I would click the Export button.")


def test_a_truncated_object_is_refused():
    with pytest.raises(LLMJsonError, match="no JSON object"):
        extract_json_object('{"steps": [{"action": "click"')


def test_a_balanced_but_invalid_object_is_refused():
    # A trailing comma is the classic near-miss: findable as an object, still
    # not parseable, so it must be refused rather than repaired by guesswork.
    with pytest.raises(LLMJsonError, match="could not be parsed"):
        extract_json_object('{"x": 1,}')


def test_nan_and_infinity_are_refused():
    with pytest.raises(LLMJsonError, match="not valid JSON"):
        extract_json_object('{"x": NaN, "y": 1}')
    with pytest.raises(LLMJsonError, match="not valid JSON"):
        extract_json_object('{"confidence": Infinity}')


# -------------------------------------------------------------- field reads
def test_optional_int_accepts_numeric_strings_and_rounds():
    assert as_optional_int(" 300 ", field="x") == 300
    assert as_optional_int(450.6, field="y") == 451


def test_optional_int_reports_absence_as_none():
    assert as_optional_int(None, field="x") is None
    assert as_optional_int("", field="x") is None


def test_optional_int_refuses_junk_and_booleans():
    with pytest.raises(LLMJsonError, match="must be a number"):
        as_optional_int("left", field="x")
    with pytest.raises(LLMJsonError, match="must be a number"):
        as_optional_int(True, field="width")


def test_optional_int_enforces_bounds():
    with pytest.raises(LLMJsonError, match="below the allowed minimum"):
        as_optional_int(-5, field="x", minimum=0)
    with pytest.raises(LLMJsonError, match="above the allowed maximum"):
        as_optional_int(99999, field="x", maximum=20000)


def test_pixels_refuse_negative_and_absurd_values():
    assert as_pixel(0, field="x") == 0
    with pytest.raises(LLMJsonError, match="below the allowed minimum"):
        as_pixel(-1, field="x")
    with pytest.raises(LLMJsonError, match="above the allowed maximum"):
        as_pixel(10**7, field="x")


def test_flags_resolve_the_string_boolean_trap():
    assert as_optional_bool("false") is False
    assert as_optional_bool(False) is False
    assert as_optional_bool("yes") is True
    assert as_optional_bool("maybe") is None  # not stated, so the caller decides


def test_text_refuses_containers():
    assert as_text("hello", field="text") == "hello"
    with pytest.raises(LLMJsonError, match="must be a string"):
        as_text(["hello"], field="text")
    with pytest.raises(LLMJsonError, match="required"):
        as_text("   ", field="text", required=True)


def test_mapping_and_sequence_refuse_the_wrong_shape():
    assert as_mapping(None, field="params") == {}
    assert as_sequence([1, 2], field="steps") == [1, 2]
    with pytest.raises(LLMJsonError, match="must be a JSON object"):
        as_mapping([1], field="params")
    with pytest.raises(LLMJsonError, match="must be a JSON array"):
        as_sequence("steps", field="steps")
