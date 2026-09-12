"""Top-level transparent task runner.

:class:`TaskAgent` ties the whole pipeline together for one user
instruction:

1. **Plan**  -- the :class:`TaskPlanner` decomposes the instruction into
   ordered steps (escalating to a smarter model when the fast tier
   struggles). Every thought is printed to the console and mirrored on the
   status window.
2. **Confirm** -- the user must approve the plan on the console (``y`` /
   ``n`` / ``edit``) before anything is executed.
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
from typing import Any, Optional

from .config import Settings
from .context import TaskAborted
from .cost import UsageTracker
from .executor import ExecutionReport, PlanExecutor
from .planner import BudgetExceeded, TaskPlan, TaskPlanner
from .status import KillSwitch, StatusWindow
from .tasklog import TaskJournal, plan_to_dict


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
    ) -> None:
        self._settings = settings
        self._journal = journal
        self._planner = planner
        self._executor = executor
        self._usage = usage
        self._window = status_window
        self._kill_switch = kill_switch
        self._stop = stop_event

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
        try:
            plan = self._planner.plan(instruction)
        except TaskAborted as exc:
            self._journal.error(f"Planning aborted: {exc}")
            return False
        except BudgetExceeded as exc:
            self._journal.error(f"LLM call budget exhausted while planning: {exc}")
            return False
        except Exception as exc:
            self._journal.error(f"Planning failed: {exc}")
            return False

        self._journal.plan_snapshot = plan_to_dict(
            plan.task_name, plan.goal, plan.steps, plan.reasoning
        )
        if not self._confirm_plan(plan):
            self._journal.system("User declined the plan; nothing was executed.")
            return False

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
        self._finish()
        return report.success

    # ----------------------------------------------------------- confirmation
    def _confirm_plan(self, plan: TaskPlan) -> bool:
        """Print the plan and require explicit console confirmation."""
        print("\n" + "=" * 68)
        print("PROPOSED PLAN (nothing will be executed yet)")
        print("=" * 68)
        print(plan.describe())
        print("-" * 68)
        self._journal.waiting("Waiting for user confirmation before execution.")
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
            self._settings.enable_status_window
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
        summary = self._usage.summary()
        for line in summary.lines():
            self._journal.cost(line)
        try:
            path = self._journal.write_report(summary)
            self._journal.system(f"Report written: {path}")
        except Exception as exc:
            self._journal.warn(f"Could not write the report file: {exc}")
