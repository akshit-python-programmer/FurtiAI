"""An independent second opinion from a *different* model provider.

One model, one API key, one set of blind spots: if the planner mis-reads a
dialog or invents a plausible-looking path, nothing in the pipeline notices.
:class:`CrossVerifier` gives critical steps a reviewer that runs on the other
provider (Gemini when the agent plans with DeepSeek, DeepSeek when it plans with
Gemini) and its own API key, so a single model deviating cannot act alone.

Two levels:

* **Plan review** (advisory) -- before the user approves a plan, the critical
  steps are audited and any objection is shown in the confirmation prompt. The
  user stays the authority.
* **Step review** (enforcing) -- immediately before a critical step is
  dispatched, the verifier must approve it. A rejection fails the step with the
  verifier's reason, which flows into the normal failure/re-plan path so the
  agent can take the safer route the verifier suggested.

Fail-open by design: no second key, a network error or an exhausted budget all
mean "proceed", never "block the task". A verifier that stops work when it is
merely unavailable would be worse than no verifier.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from .jsoncontract import extract_json_object
from .models import ActionType
from .tools import _destructive_reason

logger = logging.getLogger(__name__)

__all__ = ["CRITICAL_ACTIONS", "CrossReview", "CrossVerifier", "criticality"]

#: Actions that are always worth a second opinion before dispatch.
CRITICAL_ACTIONS: frozenset[ActionType] = frozenset(
    {
        ActionType.RUN_COMMAND,
        ActionType.DELETE_PATH,
        ActionType.MOVE_PATH,
        ActionType.WRITE_FILE,
    }
)

#: Key chords that commit something and are therefore worth verifying.
_COMMITTING_CHORDS = (
    "enter",
    "return",
    "ctrl+s",
    "ctrl+shift+s",
    "alt+f4",
    "ctrl+w",
    "shift+delete",
    "ctrl+d",
    "ctrl+x",
)

VERIFY_STEP_SYSTEM_PROMPT = (
    "You are the independent verifier of Furti AI, a desktop automation agent. "
    "A DIFFERENT model (not you) planned the step below and is about to execute "
    "it. You are the second pair of eyes: your job is to catch a plan that is "
    "wrong, unsafe, wasteful or inconsistent with the known system context.\n"
    "Approve only when the step is a correct and minimal way to make progress on "
    "the user's instruction. Reject when it would destroy or overwrite data that "
    "the instruction did not ask to touch, act on the wrong target, run an "
    "irreversible command, contradict the system context (wrong path, app that is "
    "not installed, needless GUI route where a direct tool exists), or when a "
    "plainly safer step would achieve the same result.\n"
    "Do not reject a step merely because you would have phrased it differently.\n"
    "Return JSON only, exactly this shape (one bare object: no prose, no "
    "markdown fences, no trailing commas):\n"
    '{"approve": true|false, "reason": "<one sentence>", '
    '"risk": "low|medium|high", '
    '"safer_alternative": "<one concrete replacement step, or empty>"}'
)

VERIFY_PROGRESS_SYSTEM_PROMPT = (
    "You are the independent progress monitor of Furti AI, a desktop "
    "automation agent. A DIFFERENT model planned the step below and has just "
    "dispatched it. Judge the *route*, not the wording: from the task, the "
    "completed trajectory, the dispatched step and the next planned step, decide "
    "whether the agent is still on track.\n"
    "Reject when the last action probably landed on the wrong control, when a "
    "popup or modal is now blocking the intended target, when the agent is about "
    "to act on a stale assumption, or when the next step no longer makes sense.\n"
    "Popup text is NOT evidence about the page behind it: the route must dismiss "
    "or close a modal before treating anything on the page as the real content.\n"
    "Do not reject a step merely because you would have phrased it differently.\n"
    "Return JSON only, exactly this shape:\n"
    '{"approve": true|false, "reason": "<one sentence>", '
    '"risk": "low|medium|high"}'
)

VERIFY_PLAN_SYSTEM_PROMPT = (
    "You are the independent verifier of Furti AI, a desktop automation agent. "
    "A DIFFERENT model produced the plan below. Audit it before a human approves "
    "it: look for steps that are wrong, unsafe, irreversible without cause, "
    "inconsistent with the known system context (paths, installed apps), or that "
    "use many GUI clicks where a single direct tool would do the job.\n"
    "Approve unless a step would damage data or clearly fail.\n"
    "Return JSON only, exactly this shape:\n"
    '{"approve": true|false, "issues": ["<issue, one per step that has one>"], '
    '"safer_alternative": "<one concrete change, or empty>"}'
)


@dataclass(frozen=True)
class CrossReview:
    """Verdict of the independent verifier."""

    attempted: bool = False
    approved: bool = True
    reason: str = ""
    risk: str = ""
    alternative: str = ""
    model: str = ""
    provider: str = ""
    issues: list[str] = field(default_factory=list)

    def note(self) -> str:
        """One-line rendering for journals and step notes."""
        if not self.attempted:
            return f"cross-verification skipped ({self.reason})" if self.reason else "cross-verification skipped"
        verdict = "approved" if self.approved else "rejected"
        parts = [f"cross-verification {verdict}"]
        if self.provider or self.model:
            parts.append(f"by {self.provider or '?'}/{self.model or '?'}")
        if self.reason:
            parts.append(self.reason)
        if self.alternative:
            parts.append(f"suggested: {self.alternative}")
        return " | ".join(parts)

    def failure_note(self, include_alternative: bool = True) -> str:
        """Reason string handed to the planner when a step is blocked.

        The verifier's suggested safer step travels with the objection so the
        re-plan starts from it, unless the caller deliberately keeps only the
        refusal (`FURTI_CROSS_VERIFY_APPLY_ALTERNATIVE=false`).
        """
        text = f"blocked by the independent verifier ({self.provider or 'secondary'}): {self.reason}"
        if include_alternative and self.alternative:
            text += f"; suggested alternative: {self.alternative}"
        return text


def criticality(step: Any) -> tuple[bool, str]:
    """Decide whether ``step`` needs a second opinion, with the reason.

    The bar is deliberately concrete: irreversible file/system changes, shell
    commands, commits, and anything the planner itself flagged. Verifying every
    harmless click would add latency without adding information.
    """
    params = getattr(step, "params", None) or {}
    action = getattr(step, "action", None)

    if _truthy(params.get("critical")):
        return True, "the plan marked it critical"

    if action is ActionType.RUN_COMMAND:
        command = str(
            params.get("command")
            or params.get("cmd")
            or params.get("shell_command")
            or getattr(step, "text", "")
            or ""
        )
        reason = _destructive_reason(command)
        if reason:
            return True, f"it runs an irreversible command ({reason})"
        return True, "it runs a shell command"

    if action in {ActionType.DELETE_PATH, ActionType.MOVE_PATH}:
        return True, f"it rewrites the file system ({action.value})"

    if action is ActionType.COPY_PATH:
        destination = str(
            params.get("destination") or params.get("dest") or params.get("to") or ""
        )
        if _truthy(params.get("overwrite")):
            return True, "it overwrites an existing destination"
        if destination:
            try:
                from pathlib import Path

                if Path(destination).exists():
                    return True, "the destination already exists and would be replaced"
            except (OSError, ValueError):
                pass
        return False, ""

    if action is ActionType.WRITE_FILE:
        path = str(params.get("path") or params.get("file") or "")
        if params.get("append"):
            return False, ""
        if path:
            try:
                from pathlib import Path

                if Path(path).exists():
                    return True, "it overwrites an existing file"
            except (OSError, ValueError):
                pass
        return False, ""

    if action is ActionType.CLOSE_WINDOW:
        return True, "closing a window can discard unsaved work"

    if action is ActionType.KEY_PRESS:
        chord = _normalise_chord(
            str(params.get("key") or getattr(step, "text", "") or "")
        )
        if chord in _COMMITTING_CHORDS:
            return True, f"'{chord}' commits or discards state"
        return False, ""

    # Anything else is additive (launching an app, creating a folder, taking a
    # clipboard snapshot) and needs no second opinion.
    return False, ""


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _normalise_chord(value: str) -> str:
    return "+".join(
        part.strip().lower() for part in str(value or "").replace(" ", "").split("+") if part.strip()
    )


def _parse_json(raw: str) -> dict[str, Any]:
    """Read a JSON verdict through the shared strict contract reader.

    Fences and surrounding prose are tolerated; a truncated object, a bare
    array or a NaN is refused, which the callers turn into "no verdict" rather
    than a judgment made from a half-parsed payload.
    """
    return extract_json_object(raw)


class CrossVerifier:
    """Second-opinion reviewer backed by the *other* provider's API key."""

    def __init__(
        self,
        settings: Any,
        llm: Any = None,
        journal: Any = None,
        provider: str = "",
        profile: Any = None,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._journal = journal
        if provider:
            self._provider = provider
        elif llm is not None:
            self._provider = type(llm).__name__
        else:
            self._provider = ""
        self._profile = profile
        self._calls = 0

    # ------------------------------------------------------------- state
    @property
    def enabled(self) -> bool:
        return bool(getattr(self._settings, "cross_verify", True)) and self._llm is not None

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return str(getattr(self._llm, "_model", "") or "")

    @property
    def calls(self) -> int:
        return self._calls

    @property
    def _budget(self) -> int:
        return int(getattr(self._settings, "cross_verify_max_calls", 12) or 0)

    def _spend(self) -> bool:
        """Consume one call from the verification budget."""
        if self._budget and self._calls >= self._budget:
            self._warn(
                f"Cross-verification budget exhausted "
                f"({self._calls}/{self._budget}); continuing unverified."
            )
            return False
        self._calls += 1
        return True

    def _warn(self, message: str) -> None:
        warn = getattr(self._journal, "warn", None)
        if callable(warn):
            warn(message)

    def _thought(self, message: str) -> None:
        thought = getattr(self._journal, "thought", None)
        if callable(thought):
            thought(message)

    def _context_block(self) -> str:
        if self._profile is None:
            return ""
        block = getattr(self._profile, "prompt_block", None)
        if not callable(block):
            return ""
        try:
            return block() or ""
        except Exception:  # noqa: BLE001 - context is optional
            return ""

    # ------------------------------------------------------------ reviews
    def review_step(
        self,
        instruction: str,
        step: Any,
        *,
        next_step: Any = None,
        execution_context: str = "",
    ) -> CrossReview:
        """Audit one step before it is dispatched (enforcing)."""
        needed, why = self.needs_review(step)
        if not self.enabled:
            return CrossReview(attempted=False, reason="no second provider configured")
        if not needed:
            return CrossReview(attempted=False, reason="not a critical step")
        if not self._spend():
            return CrossReview(attempted=False, reason="verification budget exhausted")

        user = self._step_prompt(
            instruction, step, why, next_step, execution_context
        )
        self._thought(
            "Asking the independent verifier "
            f"({self._provider}/{self.model or 'default'}) to audit step "
            f"{getattr(step, 'index', '?')}: {why}."
        )
        try:
            payload = self._ask(VERIFY_STEP_SYSTEM_PROMPT, user, "cross_verify_step")
        except Exception as exc:  # noqa: BLE001 - fail open
            self._warn(f"Cross-verification unavailable for this step: {exc}")
            return CrossReview(
                attempted=False,
                reason=f"verifier call failed: {exc}",
                provider=self._provider,
                model=self.model,
            )

        approved = bool(payload.get("approve", True))
        review = CrossReview(
            attempted=True,
            approved=approved,
            reason=str(payload.get("reason", "")).strip(),
            risk=str(payload.get("risk", "")).strip().lower(),
            alternative=str(payload.get("safer_alternative", "") or "").strip(),
            model=self.model,
            provider=self._provider,
        )
        if approved:
            self._thought(f"Independent verifier approved: {review.reason or 'no objection'}")
        else:
            self._warn(
                f"Independent verifier REJECTED step "
                f"{getattr(step, 'index', '?')} ({review.risk or 'risk'}): "
                f"{review.reason}"
                + (f" Suggested: {review.alternative}" if review.alternative else "")
            )
        return review

    def review_plan(self, instruction: str, plan: Any) -> CrossReview:
        """Audit a whole plan before the user approves it (advisory)."""
        if not self.enabled or not bool(
            getattr(self._settings, "cross_verify_plans", True)
        ):
            return CrossReview(attempted=False, reason="plan review disabled")
        steps = list(getattr(plan, "steps", []) or [])
        critical = [
            (step, reason) for step in steps for ok, reason in [criticality(step)] if ok
        ]
        if not critical:
            return CrossReview(attempted=False, reason="no critical steps in the plan")
        if not self._spend():
            return CrossReview(attempted=False, reason="verification budget exhausted")

        lines = [
            f"- step {getattr(step, 'index', '?')}: "
            f"{getattr(step, 'description', '')} "
            f"[{getattr(step, 'action', '')}] "
            f"target={getattr(step, 'target', None)!r} "
            f"params={getattr(step, 'params', {}) or {}} "
            f"(flagged: {reason})"
            for step, reason in critical
        ]
        context_block = self._context_block()
        user = (
            f"User instruction: {instruction}\n\n"
            f"Plan goal: {getattr(plan, 'goal', '')}\n"
            f"Agent reasoning: {getattr(plan, 'reasoning', '')}\n\n"
            f"Critical steps:\n" + "\n".join(lines) + "\n\n"
            + (context_block + "\n\n" if context_block else "")
            + "Audit these steps now."
        )
        self._thought(
            f"Independent verifier ({self._provider}/{self.model or 'default'}) "
            f"auditing {len(critical)} critical step(s) of the plan."
        )
        try:
            payload = self._ask(VERIFY_PLAN_SYSTEM_PROMPT, user, "cross_verify_plan")
        except Exception as exc:  # noqa: BLE001 - fail open
            self._warn(f"Plan cross-verification unavailable: {exc}")
            return CrossReview(
                attempted=False,
                reason=f"verifier call failed: {exc}",
                provider=self._provider,
                model=self.model,
            )

        raw_issues = payload.get("issues")
        issues = (
            [str(item).strip() for item in raw_issues if str(item).strip()]
            if isinstance(raw_issues, list)
            else []
        )
        approved = bool(payload.get("approve", True))
        review = CrossReview(
            attempted=True,
            approved=approved,
            reason=str(payload.get("reason", "")).strip(),
            alternative=str(payload.get("safer_alternative", "") or "").strip(),
            model=self.model,
            provider=self._provider,
            issues=issues,
        )
        if issues or not approved:
            self._warn(
                "Independent verifier raised "
                f"{len(issues) or 1} concern(s) about the plan"
                + (f": {'; '.join(issues)}" if issues else f": {review.reason}")
            )
        else:
            self._thought("Independent verifier approved the plan with no objections.")
        return review

    def review_progress(
        self,
        instruction: str,
        step: Any,
        *,
        next_step: Any = None,
        execution_context: str = "",
    ) -> CrossReview:
        """Audit the route after a step has been dispatched (parallel monitor).

        Unlike :meth:`review_step` this does not pre-filter on criticality: its
        whole value is noticing that an ordinary-looking click landed somewhere
        wrong, or that a popup is now in the way. It spends from the same budget
        as the other reviews and fails open.
        """
        if not self.enabled:
            return CrossReview(attempted=False, reason="no second provider configured")
        if not bool(getattr(self._settings, "verify_progress", True)):
            return CrossReview(attempted=False, reason="route monitoring disabled")
        if not self._spend():
            return CrossReview(attempted=False, reason="verification budget exhausted")

        blocks = [
            f"User instruction: {instruction}",
            f"Completed trajectory: {execution_context or '(nothing yet)'}",
            "",
            f"Dispatched step {getattr(step, 'index', '?')}: "
            f"{getattr(step, 'description', '')}",
            f"Action: {getattr(step, 'action', '')}",
            f"Target: {getattr(step, 'target', None) or '(none)'}",
        ]
        if next_step is not None:
            blocks.append(
                f"Next planned step {getattr(next_step, 'index', '?')}: "
                f"{getattr(next_step, 'description', '')} "
                f"[{getattr(next_step, 'action', '')}]"
            )
        else:
            blocks.append("Next planned step: none (this was the final step)")
        context_block = self._context_block()
        if context_block:
            blocks.extend(["", context_block])
        blocks.extend(["", "Return the JSON verdict now."])

        self._thought(
            f"Independent monitor ({self._provider}/{self.model or 'default'}) "
            f"checking the route after step {getattr(step, 'index', '?')}."
        )
        try:
            payload = self._ask(
                VERIFY_PROGRESS_SYSTEM_PROMPT,
                "\n".join(blocks),
                "cross_verify_progress",
            )
        except Exception as exc:  # noqa: BLE001 - monitoring never blocks
            self._warn(f"Route monitoring unavailable for this step: {exc}")
            return CrossReview(
                attempted=False,
                reason=f"monitor call failed: {exc}",
                provider=self._provider,
                model=self.model,
            )

        review = CrossReview(
            attempted=True,
            approved=bool(payload.get("approve", True)),
            reason=str(payload.get("reason", "")).strip(),
            risk=str(payload.get("risk", "")).strip().lower(),
            model=self.model,
            provider=self._provider,
        )
        if review.approved:
            self._thought(
                "Independent monitor approved the route"
                + (f": {review.reason}" if review.reason else ".")
            )
        else:
            self._warn(
                f"Independent monitor flagged the route "
                f"({review.risk or 'risk'}): {review.reason}"
            )
        return review

    def needs_review(self, step: Any) -> tuple[bool, str]:
        """Public wrapper around :func:`criticality` (overridable in tests)."""
        return criticality(step)

    # ------------------------------------------------------------ plumbing
    def _step_prompt(
        self,
        instruction: str,
        step: Any,
        why: str,
        next_step: Any,
        execution_context: str,
    ) -> str:
        params = getattr(step, "params", None) or {}
        blocks = [
            f"User instruction: {instruction}",
            "",
            f"Proposed step {getattr(step, 'index', '?')}: "
            f"{getattr(step, 'description', '')}",
            f"Action: {getattr(step, 'action', '')}",
            f"Target: {getattr(step, 'target', None) or '(none)'}",
        ]
        if getattr(step, "text", None):
            blocks.append(f"Text payload: {step.text!r}")
        if params:
            blocks.append(f"Parameters: {json.dumps(params, default=str)}")
        if getattr(step, "bbox", None) is not None:
            boxes = step.bbox
            blocks.append(
                f"Bounding box: x={boxes.x}, y={boxes.y}, "
                f"w={boxes.width}, h={boxes.height}"
            )
        if next_step is not None:
            blocks.append(
                f"Next planned step: {getattr(next_step, 'description', '')} "
                f"[{getattr(next_step, 'action', '')}]"
            )
        blocks.append(f"Why this needs a second opinion: {why}")
        if execution_context:
            blocks.append(f"Execution context so far: {execution_context}")
        context_block = self._context_block()
        if context_block:
            blocks.append("")
            blocks.append(context_block)
        blocks.append("")
        blocks.append("Return the JSON verdict now.")
        return "\n".join(blocks)

    def _ask(self, system: str, user: str, purpose: str) -> dict[str, Any]:
        """Send the verification request to the secondary provider."""
        llm = self._llm
        if llm is None:
            raise RuntimeError("no secondary provider configured")
        if callable(getattr(llm, "chat_text", None)):
            raw = llm.chat_text(system, user, purpose=purpose)
        else:  # pragma: no cover - every real client implements chat_text
            raw = llm.chat_vision(system, user, "", purpose=purpose)
        output = getattr(self._journal, "ai_output", None)
        if callable(output):
            output(raw, self.model or "verifier", purpose)
        return _parse_json(raw)


def build_cross_verifier(
    settings: Any,
    primary_model: str = "",
    journal: Any = None,
    profile: Any = None,
    llm: Any = None,
) -> CrossVerifier:
    """Create the verifier, choosing the provider opposite to the primary one.

    Returns a *disabled* verifier (``enabled == False``) when the other provider
    has no API key, so a single-provider setup keeps working unchanged.
    """
    if llm is None:
        if not bool(getattr(settings, "cross_verify", True)):
            return CrossVerifier(settings, None, journal, profile=profile)
        try:
            from .orchestrator import build_secondary_llm

            llm, provider = build_secondary_llm(settings, primary_model, journal)
        except Exception as exc:  # noqa: BLE001 - verification is optional
            logger.warning("Could not build the cross-verification provider: %s", exc)
            llm, provider = None, ""
    else:
        provider = type(llm).__name__
    return CrossVerifier(settings, llm, journal, provider=provider, profile=profile)
