"""Cross-provider verification: what gets audited, and what happens on a veto.

The value of this feature is entirely in the *decisions*: which steps deserve a
second opinion, that a rejection stops dispatch, that the verifier's suggestion
reaches the planner, and that none of it can block a task when the second
provider is unavailable.
"""

import json
from types import SimpleNamespace

import pytest

from furti_ai.executor import PlanExecutor
from furti_ai.models import ActionType
from furti_ai.planner import PlanStep, TaskPlan
from furti_ai.verifier import (
    CRITICAL_ACTIONS,
    CrossReview,
    CrossVerifier,
    criticality,
)


def make_settings(**overrides):
    defaults = {
        "cross_verify": True,
        "cross_verify_max_calls": 5,
        "cross_verify_plans": True,
        "cross_verify_apply_alternative": True,
        "reflex_min_anchor_confidence": 0.75,
        "allow_destructive_commands": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class ScriptedLLM:
    """Secondary provider stub that answers with queued JSON verdicts."""

    def __init__(self, responses=None, error=None):
        self.responses = list(responses or [])
        self.error = error
        self.calls = []
        self._model = "reviewer-model"

    def chat_text(self, system, user, purpose=""):
        self.calls.append({"system": system, "user": user, "purpose": purpose})
        if self.error is not None:
            raise self.error
        return self.responses.pop(0) if self.responses else '{"approve": true}'


class FakeJournal:
    def __init__(self):
        self.warnings = []
        self.thoughts = []
        self.confirms = []
        self.systems = []
        self.ai_outputs = []
        self.actions = []

    def warn(self, message):
        self.warnings.append(message)

    def action(self, message, signal="trying"):
        self.actions.append(message)

    def error(self, message):
        self.warnings.append(message)

    def thought(self, message):
        self.thoughts.append(message)

    def confirm(self, message):
        self.confirms.append(message)

    def system(self, message):
        self.systems.append(message)

    def ai_output(self, output, model, purpose, max_chars=4000):
        self.ai_outputs.append((purpose, str(output)))


def make_verifier(llm=None, tmp_path=None, **settings):
    return CrossVerifier(
        make_settings(**settings),
        llm,
        FakeJournal(),
        provider="fake",
    )


def make_step(action, **kwargs):
    defaults = {
        "index": 1,
        "description": "do the thing",
        "action": action,
        "params": {},
        "target": None,
        "window": None,
    }
    defaults.update(kwargs)
    return PlanStep(**defaults)


# -------------------------------------------------------------- criticality
@pytest.mark.parametrize(
    "action",
    [
        ActionType.RUN_COMMAND,
        ActionType.DELETE_PATH,
        ActionType.MOVE_PATH,
        ActionType.CLOSE_WINDOW,
    ],
)
def test_state_changing_actions_are_critical(action):
    needed, reason = criticality(make_step(action, params={"command": "echo hi"}))
    assert needed is True
    assert reason


def test_irreversible_command_is_flagged_as_such():
    needed, reason = criticality(
        make_step(ActionType.RUN_COMMAND, params={"command": "rm -rf /"})
    )
    assert needed
    assert "irreversible" in reason


def test_harmless_steps_are_not_critical():
    for action, params in (
        (ActionType.CLICK, {}),
        (ActionType.TYPE, {}),
        (ActionType.LAUNCH_APP, {"app": "notepad"}),
        (ActionType.SET_CLIPBOARD, {"content": "hi"}),
        (ActionType.CREATE_FOLDER, {"path": "x"}),
        (ActionType.LIST_DIR, {"path": "x"}),
        (ActionType.FIND_FILES, {"root": "x"}),
        (ActionType.PATH_INFO, {"path": "x"}),
        (ActionType.SCREENSHOT, {}),
        (ActionType.WAIT, {"seconds": 1}),
    ):
        needed, _reason = criticality(make_step(action, params=params))
        assert needed is False, action


def test_write_file_is_critical_only_when_it_overwrites(tmp_path):
    fresh = tmp_path / "new.txt"
    existing = tmp_path / "old.txt"
    existing.write_text("data", encoding="utf-8")

    assert criticality(make_step(ActionType.WRITE_FILE, params={"path": str(fresh)}))[0] is False
    assert criticality(make_step(ActionType.WRITE_FILE, params={"path": str(existing)}))[0] is True
    # Appending cannot destroy anything.
    assert (
        criticality(
            make_step(
                ActionType.WRITE_FILE,
                params={"path": str(existing), "append": True},
            )
        )[0]
        is False
    )


@pytest.mark.parametrize("chord", ["enter", "ctrl+s", "alt+f4", "shift+delete"])
def test_committing_chords_are_critical(chord):
    needed, _reason = criticality(
        make_step(ActionType.KEY_PRESS, params={"key": chord})
    )
    assert needed is True


def test_plain_navigation_keys_are_not_critical():
    for chord in ("esc", "tab", "ctrl+shift+t", "f5"):
        assert criticality(make_step(ActionType.KEY_PRESS, params={"key": chord}))[0] is False


def test_a_step_can_be_marked_critical_explicitly():
    needed, reason = criticality(make_step(ActionType.CLICK, params={"critical": True}))
    assert needed
    assert "marked it critical" in reason


# -------------------------------------------------------------- step review
def test_step_review_is_skipped_for_harmless_steps():
    llm = ScriptedLLM(['{"approve": false}'])
    verifier = make_verifier(llm)

    review = verifier.review_step("open notepad", make_step(ActionType.CLICK))

    assert review.attempted is False
    assert llm.calls == []
    assert "not a critical step" in review.reason


def test_step_review_is_disabled_without_a_second_provider():
    verifier = make_verifier(None)

    review = verifier.review_step("delete it", make_step(ActionType.DELETE_PATH))

    assert verifier.enabled is False
    assert review.attempted is False
    assert review.approved is True  # fail-open


def test_step_review_approves_a_safe_step():
    llm = ScriptedLLM(['{"approve": true, "reason": "the command matches the request", "risk": "low"}'])
    verifier = make_verifier(llm)

    review = verifier.review_step(
        "check the repo status",
        make_step(ActionType.RUN_COMMAND, params={"command": "git status"}),
    )

    assert review.attempted is True
    assert review.approved is True
    assert review.model == "reviewer-model"
    assert "matches the request" in review.reason
    assert llm.calls[0]["purpose"] == "cross_verify_step"


def test_step_review_rejects_with_a_usable_failure_note():
    llm = ScriptedLLM(
        [
            json.dumps(
                {
                    "approve": False,
                    "reason": "this deletes the whole folder, not the one file asked for",
                    "risk": "high",
                    "safer_alternative": "delete only the file report.txt",
                }
            )
        ]
    )
    verifier = make_verifier(llm)

    review = verifier.review_step(
        "tidy the downloads folder",
        make_step(ActionType.DELETE_PATH, params={"path": "C:\\Downloads", "confirm": True}),
    )

    assert review.approved is False
    note = review.failure_note()
    assert "independent verifier" in note
    assert "delete only the file report.txt" in note
    assert any("REJECTED" in warning for warning in verifier._journal.warnings)

    # FURTI_CROSS_VERIFY_APPLY_ALTERNATIVE=false keeps only the refusal.
    without = review.failure_note(include_alternative=False)
    assert "delete only the file report.txt" not in without
    assert "deletes the whole folder" in without


def test_step_review_prompt_carries_the_evidence():
    llm = ScriptedLLM()
    verifier = make_verifier(llm)
    step = make_step(
        ActionType.RUN_COMMAND,
        description="Archive the folder",
        params={"command": "tar -cf backup.tar notes"},
        target="terminal",
        text="tar",
    )

    verifier.review_step("archive my notes", step, execution_context="step 1 ok")

    prompt = llm.calls[0]["user"]
    assert "archive my notes" in prompt
    assert "tar -cf backup.tar notes" in prompt
    assert "step 1 ok" in prompt


def test_verifier_call_failure_fails_open():
    llm = ScriptedLLM(error=RuntimeError("network down"))
    verifier = make_verifier(llm)

    review = verifier.review_step("run it", make_step(ActionType.RUN_COMMAND, params={"command": "echo hi"}))

    assert review.attempted is False
    assert review.approved is True
    assert "network down" in review.reason
    assert any("unavailable" in warning for warning in verifier._journal.warnings)


def test_unparsable_verdict_fails_open():
    llm = ScriptedLLM(["I am not sure about this one, sorry."])
    verifier = make_verifier(llm)

    review = verifier.review_step("run it", make_step(ActionType.RUN_COMMAND, params={"command": "echo hi"}))

    assert review.approved is True
    assert review.attempted is False


def test_budget_is_enforced_and_then_fails_open():
    llm = ScriptedLLM(['{"approve": true}'] * 10)
    verifier = make_verifier(llm, cross_verify_max_calls=2)
    step = make_step(ActionType.RUN_COMMAND, params={"command": "echo hi"})

    assert verifier.review_step("a", step).attempted is True
    assert verifier.review_step("b", step).attempted is True
    third = verifier.review_step("c", step)

    assert third.attempted is False
    assert third.approved is True
    assert "budget" in third.reason
    assert len(llm.calls) == 2
    assert any("budget exhausted" in warning for warning in verifier._journal.warnings)


def test_fenced_json_verdict_is_accepted():
    llm = ScriptedLLM(['```json\n{"approve": false, "reason": "no"}\n```'])
    verifier = make_verifier(llm)
    review = verifier.review_step("x", make_step(ActionType.RUN_COMMAND, params={"command": "echo"}))
    assert review.approved is False


def test_verdict_wrapped_in_prose_is_still_parsed():
    llm = ScriptedLLM(['Here is my answer: {"approve": true, "reason": "fine"} -- done'])
    verifier = make_verifier(llm)
    review = verifier.review_step("x", make_step(ActionType.RUN_COMMAND, params={"command": "echo"}))
    assert review.approved is True
    assert review.reason == "fine"


# -------------------------------------------------------------- plan review
def test_plan_review_skips_plans_without_critical_steps():
    llm = ScriptedLLM()
    verifier = make_verifier(llm)
    plan = TaskPlan(
        task_name="t",
        goal="click around",
        steps=[make_step(ActionType.CLICK), make_step(ActionType.TYPE, index=2)],
    )

    review = verifier.review_plan("click around", plan)

    assert review.attempted is False
    assert llm.calls == []


def test_plan_review_reports_issues():
    llm = ScriptedLLM(
        [
            json.dumps(
                {
                    "approve": False,
                    "issues": ["step 1 deletes a folder the request never mentioned"],
                    "safer_alternative": "ask the user which folder to clean",
                }
            )
        ]
    )
    verifier = make_verifier(llm)
    plan = TaskPlan(
        task_name="t",
        goal="tidy up",
        steps=[
            make_step(
                ActionType.DELETE_PATH,
                params={"path": "C:\\data", "confirm": True},
                description="Delete the data folder",
            )
        ],
    )

    review = verifier.review_plan("tidy up my downloads", plan)

    assert review.attempted is True
    assert review.approved is False
    assert review.issues and "deletes a folder" in review.issues[0]
    assert review.alternative.startswith("ask the user")
    assert any("concern" in warning for warning in verifier._journal.warnings)
    # Only the critical steps are sent for audit.
    assert "Delete the data folder" in llm.calls[0]["user"]


def test_plan_review_can_be_disabled():
    verifier = make_verifier(ScriptedLLM(), cross_verify_plans=False)
    plan = TaskPlan(
        task_name="t",
        goal="g",
        steps=[make_step(ActionType.DELETE_PATH, params={"path": "x"})],
    )
    review = verifier.review_plan("g", plan)
    assert review.attempted is False


def test_profile_context_is_shared_with_the_verifier():
    class FakeProfile:
        def prompt_block(self):
            return "- folders: downloads=C:\\Users\\tester\\Downloads"

    llm = ScriptedLLM()
    verifier = CrossVerifier(
        make_settings(), llm, FakeJournal(), provider="fake", profile=FakeProfile()
    )

    verifier.review_step("run it", make_step(ActionType.RUN_COMMAND, params={"command": "echo"}))

    assert "C:\\Users\\tester\\Downloads" in llm.calls[0]["user"]


def test_cross_review_note_is_human_readable():
    approved = CrossReview(attempted=True, approved=True, reason="fine", provider="gemini", model="m")
    rejected = CrossReview(
        attempted=True,
        approved=False,
        reason="wrong target",
        alternative="use the other button",
        provider="gemini",
    )
    skipped = CrossReview(attempted=False, reason="no second provider configured")

    assert "approved" in approved.note()
    assert "gemini" in approved.note()
    assert "rejected" in rejected.note()
    assert "use the other button" in rejected.note()
    assert "skipped" in skipped.note()


def test_critical_actions_constant_matches_the_flagged_set():
    assert ActionType.RUN_COMMAND in CRITICAL_ACTIONS
    assert ActionType.DELETE_PATH in CRITICAL_ACTIONS
    assert ActionType.CLICK not in CRITICAL_ACTIONS


# ----------------------------------------------------- executor integration
class VerdictLLM:
    def __init__(self, verdict):
        self.verdict = verdict
        self._model = "reviewer-model"

    def chat_text(self, system, user, purpose=""):
        return json.dumps(self.verdict)


def make_executor(tmp_path, verifier):
    executor = PlanExecutor.__new__(PlanExecutor)
    executor._settings = SimpleNamespace(
        direct_tools=True,
        allow_shell_commands=True,
        allow_destructive_commands=False,
        tool_timeout=5.0,
        tool_max_output_chars=1000,
        templates_dir=tmp_path,
        reflex_retire_failures=3,
    )
    executor._stop = None
    executor._journal = FakeJournal()
    executor._context = None
    executor._verifier = verifier
    executor._instruction = "delete my downloads folder"
    executor._cursor_position = lambda: (0, 0)
    return executor


def test_executor_blocks_a_rejected_tool_step(tmp_path):
    """A veto must stop the action itself, not merely complain about it."""
    target = tmp_path / "victim"
    target.mkdir()
    llm = ScriptedLLM(
        ['{"approve": false, "reason": "the request said one file, not the folder"}']
    )
    verifier = make_verifier(llm)
    executor = make_executor(tmp_path, verifier)

    result = executor._execute_tool_step(
        make_step(
            ActionType.DELETE_PATH,
            description="Delete the folder",
            params={"path": str(target), "confirm": True},
        )
    )

    assert result.success is False
    assert result.action_dispatched is False
    assert target.exists(), "the blocked delete must not have run"
    assert any("independent verifier" in note for note in result.notes)


def test_executor_runs_an_approved_tool_step(tmp_path):
    target = tmp_path / "made"
    llm = ScriptedLLM(['{"approve": true, "reason": "matches the request"}'])
    verifier = make_verifier(llm)
    executor = make_executor(tmp_path, verifier)

    result = executor._execute_tool_step(
        make_step(ActionType.CREATE_FOLDER, params={"path": str(target)})
    )

    assert result.success is True
    assert target.is_dir()


def test_executor_does_not_verify_when_no_verifier_is_configured(tmp_path):
    executor = make_executor(tmp_path, None)
    result = executor._execute_tool_step(
        make_step(ActionType.CREATE_FOLDER, params={"path": str(tmp_path / "ok")})
    )
    assert result.success is True


def test_perform_action_raises_for_a_blocked_gui_step(tmp_path):
    from furti_ai.executor import StepBlockedByVerifier

    llm = ScriptedLLM(['{"approve": false, "reason": "alt+f4 would discard work"}'])
    verifier = make_verifier(llm)
    executor = make_executor(tmp_path, verifier)
    executor._controller = SimpleNamespace()

    with pytest.raises(StepBlockedByVerifier):
        executor._perform_action(
            make_step(ActionType.KEY_PRESS, params={"key": "alt+f4"}), (0, 0), None
        )


# ------------------------------------------------------- route monitoring
def make_verifier_with(llm, **settings):
    return CrossVerifier(
        make_settings(**settings), llm, FakeJournal(), provider="deepseek-secondary"
    )


def test_route_monitor_reports_an_off_track_route():
    """The monitor exists to catch a route that drifted, not just a bad step."""
    llm = ScriptedLLM(
        [
            '{"approve": false, "reason": "a popup is blocking the page and its '
            'text was treated as page content", "risk": "medium"}'
        ]
    )
    verifier = make_verifier_with(llm)

    review = verifier.review_progress(
        "read the article",
        make_step(ActionType.CLICK, description="Read the article body"),
        next_step=make_step(ActionType.TYPE, description="Type a summary"),
        execution_context="step 1 click Read more (ok)",
    )

    assert review.attempted is True
    assert review.approved is False
    assert "popup" in review.reason
    assert review.risk == "medium"
    assert llm.calls[0]["purpose"] == "cross_verify_progress"


def test_route_monitor_fails_open_when_the_provider_errors():
    """A monitor that could not answer must never fail the step."""
    verifier = make_verifier_with(ScriptedLLM(error=RuntimeError("rate limited")))

    review = verifier.review_progress(
        "read the article", make_step(ActionType.CLICK)
    )

    assert review.attempted is False
    assert "rate limited" in review.reason


def test_route_monitor_is_skipped_when_disabled():
    verifier = make_verifier_with(ScriptedLLM(), verify_progress=False)

    review = verifier.review_progress("do it", make_step(ActionType.CLICK))

    assert review.attempted is False
    assert "disabled" in review.reason


def test_route_monitor_respects_the_shared_call_budget():
    verifier = make_verifier_with(ScriptedLLM(), cross_verify_max_calls=1)

    first = verifier.review_progress("do it", make_step(ActionType.CLICK))
    second = verifier.review_progress("do it", make_step(ActionType.CLICK))

    assert first.attempted is True
    assert second.attempted is False
    assert "budget" in second.reason
