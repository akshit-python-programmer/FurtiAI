"""Embedded (GUI-hosted) behaviour of :class:`StatusWindow`.

The GUI already owns a Tk mainloop, so the overlay must be a ``Toplevel`` of
that window rather than a second ``Tk`` root. These tests install a stub
``tkinter`` module so the window logic can be exercised without opening real
windows.
"""

from __future__ import annotations

import sys
import threading
import types

import pytest

from furti_ai.status import StatusWindow


class FakeWidget:
    """Records the Tk calls the status window makes."""

    def __init__(self, master=None, **kwargs):
        self.master = master
        self.kwargs = dict(kwargs)
        self.destroyed = False
        self.visible = True
        self.mainloop_calls = 0
        self.after_jobs: list[tuple[float, object, tuple]] = []
        self.after_cancelled: list[object] = []

    # -- geometry / lifecycle -------------------------------------------
    def pack(self, **kwargs):
        self.pack_info = kwargs

    def grid(self, **kwargs):
        self.grid_info = kwargs

    def config(self, **kwargs):
        self.kwargs.update(kwargs)

    configure = config

    def title(self, *args):
        self.title_text = args

    def attributes(self, *args):
        self.attributes_args = args

    def resizable(self, *args):
        self.resizable_args = args

    def protocol(self, *args):
        self.protocol_args = args

    def deiconify(self):
        self.visible = True

    def withdraw(self):
        self.visible = False

    def lift(self):
        self.lifted = True

    def destroy(self):
        self.destroyed = True

    def mainloop(self):
        self.mainloop_calls += 1

    # -- text widgets ----------------------------------------------------
    def delete(self, *args):
        pass

    def insert(self, *args):
        pass

    def see(self, *args):
        pass

    # -- scheduling ------------------------------------------------------
    def after(self, delay, func, *args):
        self.after_jobs.append((delay, func, args))
        return len(self.after_jobs)

    def after_cancel(self, ident):
        self.after_cancelled.append(ident)

    def tick(self):
        """Run the most recently scheduled callback, like the real mainloop."""
        delay, func, args = self.after_jobs[-1]
        func(*args)
        return delay


class TkRoot(FakeWidget):
    """Stub ``tk.Tk``."""


class Toplevel(FakeWidget):
    """Stub ``tk.Toplevel``."""


@pytest.fixture()
def fake_tk(monkeypatch):
    module = types.ModuleType("tkinter")
    module.Tk = TkRoot
    module.Toplevel = Toplevel
    module.Label = FakeWidget
    module.Frame = FakeWidget
    module.Text = FakeWidget
    module.Button = FakeWidget
    monkeypatch.setitem(sys.modules, "tkinter", module)
    return module


def test_embedded_window_is_a_toplevel_of_the_host(fake_tk):
    master = TkRoot()
    window = StatusWindow(master=master)
    window.start()

    assert isinstance(window._root, Toplevel)
    assert window._root.master is master
    assert window.available is True


def test_standalone_window_keeps_its_own_root(fake_tk):
    window = StatusWindow()
    window.start()

    assert isinstance(window._root, TkRoot)
    assert window._root.master is None


def test_embedded_window_is_always_on_top(fake_tk):
    window = StatusWindow(master=TkRoot())
    window.start()

    assert window._root.attributes_args == ("-topmost", True)


def test_closing_the_overlay_hides_instead_of_killing_the_app(fake_tk):
    window = StatusWindow(master=TkRoot())
    window.start()

    _, handler = window._root.protocol_args
    handler()

    assert window._root is not None
    assert window._root.visible is False


def test_start_is_idempotent_and_keeps_one_window(fake_tk):
    window = StatusWindow(master=TkRoot())
    window.start()
    first = window._root

    stop_event = threading.Event()
    window.start(stop_event)

    assert window._root is first
    assert window._root.visible is True
    assert window._stop_callback is stop_event


def test_hide_and_show_toggle_visibility(fake_tk):
    window = StatusWindow(master=TkRoot())
    window.start()

    window.hide()
    assert window._root.visible is False
    assert window._root.after_cancelled  # the queue pump was cancelled

    window.show()
    assert window._root.visible is True
    assert window._root.lifted is True


def test_embedded_poll_drains_the_queue_on_tk_ticks(fake_tk):
    window = StatusWindow(master=TkRoot())
    window.start()
    root = window._root

    window.post(
        {
            "phase": "Executing",
            "step": "Step 2/4",
            "current_action": "click Save",
        }
    )
    delay = root.tick()

    assert delay == 150
    assert window._widgets["phase"].kwargs["text"] == "Phase: Executing"
    assert window._widgets["step"].kwargs["text"] == "Step: Step 2/4"
    assert window._widgets["action"].kwargs["text"] == "Current action: click Save"
    assert len(root.after_jobs) == 2  # the pump rescheduled itself


def test_embedded_poll_stops_after_a_close_snapshot(fake_tk):
    window = StatusWindow(master=TkRoot())
    window.start()
    root = window._root

    window.close()
    root.tick()

    assert root.destroyed is True
    assert window._root is None
    assert len(root.after_jobs) == 1  # nothing rescheduled


def test_embedded_shutdown_leaves_the_host_alive(fake_tk):
    master = TkRoot()
    window = StatusWindow(master=master)
    window.start()

    window._shutdown()

    assert window._root is None
    assert window._widgets == {}
    assert master.destroyed is False


def test_embedded_run_until_does_not_start_a_mainloop(fake_tk):
    window = StatusWindow(master=TkRoot())
    window.start()
    root = window._root

    done = threading.Event()
    done.set()
    window.run_until(done)

    assert root.mainloop_calls == 0
    assert root.destroyed is False


def test_standalone_run_until_still_owns_the_mainloop(fake_tk):
    window = StatusWindow()
    window.start()
    root = window._root

    done = threading.Event()

    def mainloop():
        root.mainloop_calls += 1
        done.set()
        root.tick()  # the poll that run_until scheduled

    root.mainloop = mainloop
    window.run_until(done)

    assert root.mainloop_calls == 1
    assert root.destroyed is True
    assert window._root is None


def test_standalone_run_until_tolerates_an_already_finished_task(fake_tk):
    window = StatusWindow()
    window.start()
    root = window._root

    done = threading.Event()
    done.set()
    window.run_until(done)

    assert root.destroyed is True
    assert window._root is None


def test_unavailable_tk_disables_the_window(monkeypatch):
    monkeypatch.setitem(sys.modules, "tkinter", None)
    window = StatusWindow(master=object())

    window.start()

    assert window.available is False
    window.post({"phase": "ignored"})  # must not raise
