"""Always-on-top status window and the global kill hotkey.

* :class:`StatusWindow` -- a small Tk ``topmost`` window that mirrors what the
  agent is thinking/doing (phase, current step, latest thought, token usage
  and approximate cost). It is fed through a thread-safe queue so worker
  threads never touch Tk directly. Tk is started on the thread that calls
  :meth:`StatusWindow.run_until`.
* :class:`KillSwitch` -- a global hotkey listener. pyautogui has no
  keyboard-listener API (it can only *send* input), so the listener uses
  pynput; the default chord is ``Ctrl+Alt+K``. Pressing it sets a
  ``threading.Event`` that every executor loop polls, aborting the task
  between actions.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class StatusWindow:
    """Thread-safe always-on-top Tk overlay showing live task status."""

    MAX_LOG_LINES = 9

    def __init__(self, title: str = "Furti AI — live status") -> None:
        self._queue: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self._title = title
        self._root: Any = None
        self._widgets: dict[str, Any] = {}
        self._log_lines: list[str] = []
        self.available = True
        self._stop_callback: Optional[threading.Event] = None
        self._last_capture_epoch = 0.0
        self._last_capture_at = ""
        self._capture_signal_until = 0.0

    # ------------------------------------------------------------ wiring up
    def start(self, stop_event: Optional[threading.Event] = None) -> None:
        """Build the Tk window. Call from the thread that runs ``run_until``."""
        try:
            import tkinter as tk
        except ImportError as exc:  # headless / minimal Python installs
            logger.warning("tkinter unavailable; status window disabled: %s", exc)
            self.available = False
            return
        try:
            self._stop_callback = stop_event
            root = tk.Tk()
            root.title(self._title)
            root.attributes("-topmost", True)
            root.configure(bg="#101418")
            root.resizable(False, False)

            header = tk.Label(
                root,
                text="Furti AI",
                bg="#101418",
                fg="#7fd1ff",
                font=("Segoe UI", 11, "bold"),
            )
            header.pack(anchor="w", padx=10, pady=(8, 0))

            task_label = tk.Label(
                root, text="Task: —", bg="#101418", fg="#e8eef2",
                font=("Consolas", 9), wraplength=430, justify="left", anchor="w",
            )
            task_label.pack(fill="x", padx=10, pady=(2, 0))

            phase_label = tk.Label(
                root, text="Phase: idle", bg="#101418", fg="#ffd479",
                font=("Consolas", 10, "bold"), anchor="w",
            )
            phase_label.pack(fill="x", padx=10, pady=(6, 0))

            step_label = tk.Label(
                root, text="Step: —", bg="#101418", fg="#b6ff9e",
                font=("Consolas", 9), wraplength=430, justify="left", anchor="w",
            )
            step_label.pack(fill="x", padx=10, pady=(2, 0))

            activity_label = tk.Label(
                root,
                text="Signal: [IDLE]",
                bg="#252a30",
                fg="#c9d4dc",
                font=("Consolas", 9, "bold"),
                anchor="w",
            )
            activity_label.pack(fill="x", padx=10, pady=(6, 0))

            screenshot_label = tk.Label(
                root,
                text="Last screenshot: —",
                bg="#101418",
                fg="#9db4c0",
                font=("Consolas", 8),
                anchor="w",
            )
            screenshot_label.pack(fill="x", padx=10, pady=(2, 0))

            action_label = tk.Label(
                root,
                text="Current action: —",
                bg="#101418",
                fg="#ffd479",
                font=("Consolas", 9, "bold"),
                wraplength=430,
                justify="left",
                anchor="w",
            )
            action_label.pack(fill="x", padx=10, pady=(2, 0))

            ai_header = tk.Label(
                root,
                text="Latest AI output:",
                bg="#101418",
                fg="#7fd1ff",
                font=("Consolas", 8, "bold"),
                anchor="w",
            )
            ai_header.pack(fill="x", padx=10, pady=(5, 0))
            ai_output = tk.Text(
                root,
                height=5,
                width=62,
                bg="#0a0d10",
                fg="#d8e8ff",
                font=("Consolas", 8),
                state="disabled",
                relief="flat",
                wrap="word",
            )
            ai_output.pack(fill="x", padx=10, pady=(2, 0))

            log_frame = tk.Frame(root, bg="#101418")
            log_frame.pack(fill="both", expand=True, padx=10, pady=(6, 0))
            log_text = tk.Text(
                log_frame,
                height=self.MAX_LOG_LINES,
                width=62,
                bg="#0a0d10",
                fg="#c9d4dc",
                font=("Consolas", 8),
                state="disabled",
                relief="flat",
                wrap="word",
            )
            log_text.pack(side="left", fill="both", expand=True)

            stats_label = tk.Label(
                root, text="Tokens: —  |  Cost: —", bg="#101418", fg="#9db4c0",
                font=("Consolas", 8), anchor="w",
            )
            stats_label.pack(fill="x", padx=10, pady=(4, 0))

            stop_button = tk.Button(
                root,
                text="STOP TASK  (Ctrl+Alt+K)",
                command=self._on_stop_clicked,
                bg="#5a1e1e",
                fg="#ffb3b3",
                activebackground="#7a2626",
                activeforeground="#ffffff",
                font=("Segoe UI", 9, "bold"),
                relief="flat",
            )
            stop_button.pack(fill="x", padx=10, pady=(6, 8))

            self._root = root
            self._widgets = {
                "task": task_label,
                "phase": phase_label,
                "step": step_label,
                "activity": activity_label,
                "screenshot": screenshot_label,
                "action": action_label,
                "ai_output": ai_output,
                "log": log_text,
                "stats": stats_label,
            }
            self._refresh_capture_age()
        except Exception as exc:  # tkinter can fail to init on odd displays
            logger.warning("Could not start status window: %s", exc)
            self.available = False

    # ------------------------------------------------------------- API (any
    # thread; only enqueues)
    def post(self, snapshot: dict[str, Any]) -> None:
        if self.available:
            self._queue.put(snapshot)

    def close(self) -> None:
        self.post({"_close": True})

    # -------------------------------------------------------- Tk-thread only
    def run_until(self, done_event: threading.Event, poll_ms: int = 150) -> None:
        """Run the Tk mainloop until ``done_event`` is set, then destroy."""
        if not self.available or self._root is None:
            done_event.wait()
            return
        self._poll_queue(done_event, poll_ms)
        self._root.mainloop()

    def _poll_queue(self, done_event: threading.Event, poll_ms: int) -> None:
        root = self._root
        if root is None:
            return
        try:
            while True:
                snapshot = self._queue.get_nowait()
                if snapshot.get("_close"):
                    self._shutdown()
                    return
                self._apply(snapshot)
        except queue.Empty:
            pass
        if done_event.is_set():
            self._shutdown()
            return
        self._refresh_capture_age()
        root.after(poll_ms, self._poll_queue, done_event, poll_ms)

    def _apply(self, snapshot: dict[str, Any]) -> None:
        widgets = self._widgets
        if not widgets:
            return
        task = snapshot.get("task")
        if task:
            widgets["task"].config(text=f"Task: {task}")
        phase = snapshot.get("phase")
        if phase:
            widgets["phase"].config(text=f"Phase: {phase}")
        step = snapshot.get("step")
        if step:
            widgets["step"].config(text=f"Step: {step}")
        stats = snapshot.get("stats")
        if stats:
            widgets["stats"].config(text=f"Tokens: {stats}  |  {snapshot.get('cost', 'Cost: —')}")
        event_kind = snapshot.get("event_kind")
        message = snapshot.get("message")
        if snapshot.get("last_screenshot_epoch"):
            self._last_capture_epoch = float(snapshot["last_screenshot_epoch"])
            self._last_capture_at = str(
                snapshot.get("last_screenshot_at", "--")
            )
            self._capture_signal_until = time.monotonic() + 1.2
            self._refresh_capture_age()
            self._set_activity(
                "[SCREEN CAPTURED]",
                bg="#214b35",
                fg="#b6ff9e",
            )
        if snapshot.get("current_action"):
            widgets["action"].config(
                text=f"Current action: {snapshot['current_action']}"
            )
        if snapshot.get("ai_output") is not None:
            self._set_text(widgets["ai_output"], str(snapshot["ai_output"]))
        if event_kind == "ACTION":
            if snapshot.get("action_signal") == "searching":
                self._set_activity(
                    "[SEARCHING FOR TARGET]",
                    bg="#3b354f",
                    fg="#d8c8ff",
                )
            else:
                self._set_activity("[ACTING]", bg="#4b3b1d", fg="#ffd479")
        elif event_kind == "CONFIRM":
            self._set_activity("[ACTION CONFIRMED]", bg="#214b35", fg="#b6ff9e")
        elif event_kind == "WAIT":
            self._set_activity(
                "[WAITING FOR AI RESPONSE]",
                bg="#302c57",
                fg="#d8c8ff",
            )
        elif event_kind == "AI_OUTPUT":
            self._set_activity(
                "[AI RESPONSE RECEIVED]",
                bg="#1e3d59",
                fg="#9ed8ff",
            )
        elif event_kind == "THOUGHT":
            self._set_activity("[AI THINKING]", bg="#302c57", fg="#d8c8ff")
        elif event_kind == "WARN":
            self._set_activity("[RECOVERING]", bg="#4b3b1d", fg="#ffd479")
        elif event_kind == "ERROR":
            self._set_activity("[ERROR]", bg="#5a1e1e", fg="#ffb3b3")
        if message:
            line = f"[{snapshot.get('timestamp', '--:--:--')}] [{event_kind or 'INFO'}] {message}"
            self._log_lines.append(line)
            self._log_lines = self._log_lines[-self.MAX_LOG_LINES :]
            text_widget = widgets["log"]
            text_widget.config(state="normal")
            text_widget.delete("1.0", "end")
            text_widget.insert("1.0", "\n".join(self._log_lines))
            text_widget.see("end")
            text_widget.config(state="disabled")

    def _set_activity(self, text: str, bg: str, fg: str) -> None:
        widget = self._widgets.get("activity")
        if widget is not None:
            widget.config(text=f"Signal: {text}", bg=bg, fg=fg)

    @staticmethod
    def _set_text(widget: Any, text: str) -> None:
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.see("1.0")
        widget.config(state="disabled")

    def _refresh_capture_age(self) -> None:
        widget = self._widgets.get("screenshot")
        if widget is None:
            return
        if self._last_capture_epoch <= 0:
            widget.config(text="Last screenshot: —")
            return
        age = max(0.0, time.time() - self._last_capture_epoch)
        signal = (
            "[CAPTURED]"
            if time.monotonic() < self._capture_signal_until
            else "[CACHED]"
        )
        widget.config(
            text=(
                f"Last screenshot: {self._last_capture_at} "
                f"({age:.1f}s ago) {signal}"
            )
        )

    def _on_stop_clicked(self) -> None:
        if self._stop_callback is not None:
            self._stop_callback.set()

    def _shutdown(self) -> None:
        root = self._root
        self._root = None
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass


class KillSwitch:
    """Global hotkey that sets ``stop_event`` (pynput-backed)."""

    def __init__(self, hotkey: str, stop_event: threading.Event) -> None:
        self.hotkey = hotkey
        self.stop_event = stop_event
        self.enabled = False

    def start(self) -> None:
        """Begin listening on a daemon thread. Safe to call when pynput is absent."""
        try:
            from pynput import keyboard
        except ImportError:
            logger.warning(
                "pynput is not installed; the %s kill hotkey is unavailable. "
                "Use Ctrl+C or the status window's STOP button instead.",
                self.hotkey,
            )
            return
        try:
            listener = keyboard.GlobalHotKeys({self.hotkey: self._trigger})
            listener.daemon = True
            listener.start()
            self.enabled = True
            logger.info("Kill hotkey armed: %s", self.hotkey)
        except Exception as exc:
            logger.warning("Could not arm kill hotkey %s: %s", self.hotkey, exc)

    def _trigger(self) -> None:
        print(f"\n[KILL] Hotkey {self.hotkey} pressed — stopping task safely.")
        self.stop_event.set()
