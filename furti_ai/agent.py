"""Top-level transparent task runner.

:class:`TaskAgent` ties the whole pipeline together for one user
instruction:

1. **Plan**  -- the :class:`TaskPlanner` decomposes the instruction into
   ordered steps (escalating to a smarter model when the fast tier
   struggles). Every thought is printed to the console and mirrored on the
   status window.
2. **Confirm** -- the user must approve the plan through the configured
   confirmation callback, or on the console (``y`` / ``n`` / ``edit``), before
   anything is executed.
3. **Execute** -- the :class:`PlanExecutor` grounds each step on the live
   screen (OCR + icon templates + gated screenshots) and performs it with a
   visible cursor move. The status window runs a Tk mainloop on the main
   thread while execution happens in a worker thread; the global kill
   hotkey and the window's STOP button both set a shared stop event.
4. **Report** -- a ``<task_name>.md`` report (plan, steps, full log, token
   usage and approximate API cost) is written to ``reports_dir``.

Running from a console-less process (e.g. a service) is safe: the status
window falls back to console-only output and the kill switch degrades to
Ctrl+C.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .config import Settings
from .context import TaskAborted
from .cost import UsageTracker
from .executor import ExecutionReport, PlanExecutor
from .models import ActionType
from .planner import BudgetExceeded, TaskPlan, TaskPlanner
from .status import KillSwitch, StatusWindow
from .tasklog import TaskJournal, plan_to_dict


@dataclass(frozen=True)
class UserChoiceRequest:
    """A model-authored question that needs a human answer before execution."""

    question: str
    options: tuple[str, ...] = ()


UserChoiceCallback = Callable[[UserChoiceRequest], str | None]


class TaskAgent:
    """Runs one instruction end-to-end with full transparency."""

    def __init__(
        self,
        settings: Settings,
        journal: TaskJournal,
        planner: TaskPlanner,
        executor: PlanExecutor,
        usage: UsageTracker,
        status_window: StatusWindow,
        kill_switch: Optional[KillSwitch],
        stop_event: threading.Event,
        confirmation_callback: Optional[
            Callable[[TaskPlan], bool | str | None]
        ] = None,
        user_choice_callback: Optional[UserChoiceCallback] = None,
        manage_status_window: bool = True,
        profile: Any = None,
        verifier: Any = None,
    ) -> None:
        self._settings = settings
        self._journal = journal
        self._planner = planner
        self._executor = executor
        self._usage = usage
        self._window = status_window
        self._kill_switch = kill_switch
        self._stop = stop_event
        self._confirmation_callback = confirmation_callback
        self._user_choice_callback = user_choice_callback
        self._manage_status_window = manage_status_window
        #: User/system context file (``profile.py``), for logging and history.
        self._profile = profile
        #: Independent second-opinion reviewer (``verifier.py``), plan level.
        self._verifier = verifier
        self._selected_plan: Optional[TaskPlan] = None
        self._plan_review: Any = None
        self._finished = False

    # ------------------------------------------------------------- main flow
    def run_task(self, instruction: str) -> bool:
        """Plan, confirm, execute and report one instruction. Returns success."""
        self._journal.instruction = instruction
        self._journal.task_name = instruction  # <task_name>.md report name
        self._journal.system(f"=== Task: {instruction!r} ===")
        self._journal.system(
            f"Guardrails: max_steps={self._settings.max_plan_steps} "
            f"max_retries={self._settings.max_step_retries} "
            f"max_plan_replans={self._settings.max_plan_replans} "
            f"max_llm_calls={self._settings.max_llm_calls_per_task} "
            f"kill_hotkey={self._settings.kill_hotkey} "
            f"cursor={'teleport' if self._settings.cursor_teleport else 'smooth-move'} "
            f"input_pause={self._settings.input_pause:.3f}s "
            f"typing_interval={self._settings.typing_interval:.3f}s"
        )

        self._journal.system("Phase 1/3: planning ...")
        self._publish_profile()
        for _ in range(3):
            try:
                plan = self._planner.plan(instruction)
            except TaskAborted as exc:
                self._journal.error(f"Planning aborted: {exc}")
                self._finish()
                return False
            except BudgetExceeded as exc:
                self._journal.error(f"LLM call budget exhausted while planning: {exc}")
                self._finish()
                return False
            except Exception as exc:
                self._journal.error(f"Planning failed: {exc}")
                self._finish()
                return False

            question = self._first_user_choice(plan)
            if question is None:
                break
            answer = self._ask_user(question)
            if not answer:
                self._journal.system("User did not answer the model's question.")
                self._finish()
                return False
            instruction = f"{instruction}\n\nUser clarification: {answer}"
            self._journal.system(f"User clarification received: {answer!r}")
        else:
            self._journal.error("The planner asked too many consecutive questions.")
            self._finish()
            return False

        self._journal.plan_snapshot = plan_to_dict(
            plan.task_name, plan.goal, plan.steps, plan.reasoning
        )
        self._selected_plan = plan
        # The independent verifier audits the critical steps before the user is
        # asked to approve, so the human sees the objection while deciding.
        self._review_plan(instruction, plan)
        if not self._confirm_plan(plan):
            self._journal.system("User declined the plan; nothing was executed.")
            self._finish()
            return False
        plan = self._selected_plan or plan

        self._journal.system("Phase 2/3: executing ...")
        if self._kill_switch is not None:
            self._kill_switch.start()
        self._journal.status = "executing"
        report = self._run_execution(instruction, plan)

        self._journal.system("Phase 3/3: reporting ...")
        self._journal.status = "finished"
        if report.aborted:
            self._journal.error(f"Task aborted: {report.reason}")
        elif report.success:
            self._journal.system("Task completed successfully.")
        else:
            self._journal.error("Task finished with failed step(s); see report.")
        self._remember_outcome(instruction, report.success)
        self._finish()
        return report.success

    @staticmethod
    def _first_user_choice(plan: TaskPlan) -> UserChoiceRequest | None:
        """Extract the first model-authored clarification from a plan."""
        for step in plan.steps:
            if step.action is not ActionType.ASK_USER:
                continue
            question = str(step.params.get("question") or step.description).strip()
            raw_options = step.params.get("options") or []
            options = tuple(
                str(option).strip() for option in raw_options if str(option).strip()
            )
            return UserChoiceRequest(question=question, options=options)
        return None

    def _ask_user(self, request: UserChoiceRequest) -> str | None:
        """Ask through the GUI callback, falling back to the console."""
        if self._user_choice_callback is not None:
            return self._user_choice_callback(request)
        print(f"\nFurti needs a choice: {request.question}")
        for index, option in enumerate(request.options, 1):
            print(f"  {index}. {option}")
        return input("Your choice: ").strip() or None

    def _remember_outcome(self, instruction: str, success: bool) -> None:
        """Record the run in the context file so later plans learn from it."""
        profile = self._profile
        if profile is None:
            return
        remember = getattr(profile, "record_task", None)
        if not callable(remember):
            return
        try:
            remember(instruction, success)
        except Exception as exc:  # noqa: BLE001 - history is a bonus
            self._journal.warn(f"Could not update the system context: {exc}")

    # ----------------------------------------------------------- confirmation
    def _publish_profile(self) -> None:
        """Log the user/system context the planner is working from."""
        profile = self._profile
        if profile is None:
            return
        summary = ""
        summarise = getattr(profile, "summary", None)
        if callable(summarise):
            try:
                summary = summarise()
            except Exception as exc:  # noqa: BLE001 - context is never critical
                self._journal.warn(f"Could not render the system context: {exc}")
                return
        if summary:
            self._journal.system(f"User/system context:\n{summary}")

    def _review_plan(self, instruction: str, plan: TaskPlan) -> None:
        """Ask the independent verifier to audit the plan's critical steps."""
        verifier = self._verifier
        if verifier is None or not getattr(verifier, "enabled", False):
            return
        review_plan = getattr(verifier, "review_plan", None)
        if not callable(review_plan):
            return
        try:
            review = review_plan(instruction, plan)
        except TaskAborted:
            raise
        except Exception as exc:  # noqa: BLE001 - advisory only
            self._journal.warn(f"Plan cross-verification errored: {exc}")
            return
        self._plan_review = review
        if not getattr(review, "attempted", False):
            self._journal.thought(
                f"Plan cross-verification skipped: {getattr(review, 'reason', '')}"
            )
            return
        if getattr(review, "approved", True) and not getattr(review, "issues", None):
            self._journal.confirm(
                f"Independent verifier ({getattr(review, 'provider', '?')}) "
                "reviewed the critical steps and raised no objection."
            )
            return
        self._journal.warn(
            "The independent verifier is not convinced by this plan: "
            + (
                "; ".join(getattr(review, "issues", []) or [])
                or getattr(review, "reason", "")
                or "unspecified concern"
            )
            + (
                f" (suggested change: {review.alternative})"
                if getattr(review, "alternative", "")
                else ""
            )
        )

    def _confirm_plan(self, plan: TaskPlan) -> bool:
        """Print the plan and require explicit user confirmation."""
        print("\n" + "=" * 68)
        print("PROPOSED PLAN (nothing will be executed yet)")
        print("=" * 68)
        print(plan.describe())
        print("-" * 68)
        if (
            self._plan_review is not None
            and getattr(self._plan_review, "attempted", False)
            and not getattr(self._plan_review, "approved", True)
        ):
            print(
                "INDEPENDENT VERIFIER CONCERN: "
                + (
                    "; ".join(getattr(self._plan_review, "issues", []) or [])
                    or getattr(self._plan_review, "reason", "")
                )
            )
            print("-" * 68)
        self._journal.waiting("Waiting for user confirmation before execution.")
        if self._confirmation_callback is not None:
            decision = self._confirmation_callback(plan)
            if isinstance(decision, str):
                new_instruction = decision.strip()
                if not new_instruction:
                    return False
                self._journal.system(
                    f"User edited the instruction to: {new_instruction!r}"
                )
                return self._replan_after_edit(new_instruction)
            if decision:
                self._journal.system("User approved the plan.")
            return bool(decision)
        while True:
            answer = input(
                "Execute this plan? [y]es / [n]o / [e]dit instruction: "
            ).strip().lower()
            if answer in {"y", "yes"}:
                self._journal.system("User approved the plan.")
                return True
            if answer in {"n", "no"}:
                return False
            if answer in {"e", "edit"}:
                new_instruction = input("New instruction: ").strip()
                if not new_instruction:
                    continue
                self._journal.system(
                    f"User edited the instruction to: {new_instruction!r}"
                )
                return self._replan_after_edit(new_instruction)
            print("Please answer y, n or e.")

    def _replan_after_edit(self, instruction: str) -> bool:
        """Re-plan after the user edited the instruction, then re-confirm."""
        self._journal.instruction = instruction
        self._journal.task_name = instruction
        try:
            plan = self._planner.plan(instruction)
        except Exception as exc:
            self._journal.error(f"Re-planning failed: {exc}")
            return False
        self._journal.plan_snapshot = plan_to_dict(
            plan.task_name, plan.goal, plan.steps, plan.reasoning
        )
        self._selected_plan = plan
        self._review_plan(instruction, plan)
        return self._confirm_plan(plan)

    # ------------------------------------------------------ threaded execution
    def _run_execution(self, instruction: str, plan: TaskPlan) -> ExecutionReport:
        """Execute, with the Tk mainloop on the current (main) thread."""
        done = threading.Event()
        holder: dict[str, Any] = {}

        def _work() -> None:
            try:
                holder["report"] = self._executor.execute(instruction, plan)
            except TaskAborted as exc:
                holder["aborted"] = str(exc)
            except Exception as exc:  # never let the worker die silently
                holder["error"] = repr(exc)
            finally:
                done.set()

        use_window = (
            self._manage_status_window
            and self._settings.enable_status_window
            and self._window is not None
            and self._window.available
        )
        if use_window:
            # start() must run on the thread that owns the Tk mainloop.
            self._window.start(self._stop)
            worker = threading.Thread(
                target=_work, name="furti-executor", daemon=True
            )
            worker.start()
            self._window.run_until(done)  # blocks on the main thread
            worker.join(timeout=5)
        else:
            _work()

        if "report" in holder:
            return holder["report"]
        if "aborted" in holder:
            self._journal.error(f"Execution stopped: {holder['aborted']}")
        else:
            error = holder.get("error")
            if not error:
                error = "execution worker ended without returning a report"
            self._journal.error(f"Execution crashed: {error}")
        reason = holder.get("aborted") or holder.get("error") or (
            "execution worker ended without returning a report"
        )
        return ExecutionReport(aborted=True, reason=reason)

    # ------------------------------------------------------------- finishing
    def _finish(self) -> None:
        """Print the cost summary and write the <task_name>.md report."""
        if self._finished:
            return
        self._finished = True
        if self._kill_switch is not None:
            self._kill_switch.stop()
        self._executor.close()
        summary = self._usage.summary()
        self._journal.status = "finished"
        for line in summary.lines():
            self._journal.cost(line)
        try:
            path = self._journal.write_report(summary)
            self._journal.system(f"Report written: {path}")
        except Exception as exc:
            self._journal.warn(f"Could not write the report file: {exc}")
