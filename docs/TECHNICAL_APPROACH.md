# Furti AI — Technical Approach & Feature Registry

This document is the complete technical reference for Furti AI's
**multi-step task pipeline** (`TaskAgent`). It records the architecture,
every feature/function, the guardrails against runaway loops, and the
approximate API cost model (tokens × Google Gemini list prices).

---

## 1. System overview

Furti AI now runs in two modes over one codebase:

1. **Reflex mode** (legacy, unchanged) — `AgentOrchestrator` compiles a
   single LLM plan into a cached template "reflex" and replays it with
   OpenCV. Zero-token replays on cache hits.
2. **Task mode** (new) — `TaskAgent` plans a whole instruction into
   **ordered steps**, shows the plan in the Tkinter application (or the
   console API), waits for explicit confirmation, executes step by step with
   live grounding, and writes a `<task_name>.md` report with token usage and
   approximate cost.

```
TaskAgent
├── TaskPlanner      plan in steps (fast model, escalates to smart)
├── VisualContextManager  rate-limited capture + OCR/icon grounding + visual gate
├── PlanExecutor     execute steps, re-anchor targets, retry, re-align, re-plan
├── DirectToolRunner OS-level tools: apps, shell, files, screenshots, windows
├── UserContext      context file about the user/system, injected into prompts
├── CrossVerifier    independent second opinion on the other provider
├── ReflexRealigner  re-anchor a stale reflex with the LLM before re-planning
├── MemoryManager    compile successful *and useful* steps into reusable reflexes
├── UsageTracker     token counting + approximate USD cost
├── TaskJournal      console + status-window + <task_name>.md transparency
├── FurtiApp         full Tkinter task editor, plan approval, status and logs
├── StatusWindow     always-on-top Tk overlay (capture / AI output / action signals)
└── KillSwitch       global hotkey (<ctrl>+<alt>+k) sets the shared stop event
```

### GUI threading model

`FurtiApp` owns the only Tk event loop on the main thread. It creates the
planner/executor in a daemon worker and passes three thread-safe hooks to
`build_task_agent()`:

1. `status_sink` places `TaskJournal` snapshots on a queue; the GUI consumes
   them during a short `after()` poll and updates the plan, status cards, AI
   output and event log.
2. `confirmation_callback` places a `ConfirmationRequest` on the same queue
   and waits on a `threading.Event`. Approve, decline, or edit/re-plan buttons
   resolve that event without touching Tk from the worker.
3. `user_choice_callback` places a model-authored `UserChoiceResponse` on the
  same queue. The modal choice dialog resolves it, and the answer is appended
  to a fresh planning request before plan approval.
4. `stop_event` is shared by the GUI Stop button, `KillSwitch`, planner and
   executor. This allows cancellation at the existing safe boundaries.

The embedded app passes `manage_status_window=False`, so the worker never
creates a second Tk root or calls `mainloop()`. The legacy console API keeps
the original `StatusWindow` behaviour.

### GUI layout

The task editor and the topic-tabbed settings form occupy the left column; the
right column is the proposed plan (with **Approve and run** / **Edit and
re-plan** / **Decline**) above a tabbed pane. That pane holds **Live Status**
(phase, step, screenshot age, current action, usage/cost, signal badge and the
always-on-top overlay toggle), **AI Output** and **Event Log**, so the plan keeps
the full height of the column instead of sharing it with a permanent status
block. The overlay checkbox lives in the Live Status tab.

The settings notebook groups every `Settings` field into **Models / Input /
Safety / Vision / Advanced / AI Context / Reflexes** tabs. Tab assignment is
declared in `_tabbed_setting_names` so each field gets exactly one control, and
every tab (including **Reflexes**) is a scrollable canvas so the pane stays
usable at any window size. The **Reflexes** tab is the reflex manager: a summary
line, the master `reflex_enabled` switch (sharing one `tk.BooleanVar` with the
Safety tab's control so the two stay in sync), `Refresh` / `Enable all` /
`Disable all`, and one row per cached reflex with its own checkbox, name, action
and hit/miss counters, and a delete button.

`FurtiApp` still shows the always-on-top readout itself: it builds
`StatusWindow(master=self)`, which makes the overlay a `Toplevel` of the main
window instead of a second `tk.Tk` root. The overlay is pumped by a
self-rescheduling `root.after()` tick on the GUI thread (no `mainloop()` of its
own), receives the same journal snapshots as the main panel, and is
`show()`n/`hide()`n rather than re-created per run. `StatusWindow` therefore has
two modes: standalone (`tk.Tk` + `run_until()`, used by the console path) and
embedded (`tk.Toplevel(master=...)`, used by the GUI), selected by the
`master` argument.

### Execution flow

1. **Plan** — `TaskPlanner.plan(instruction)`:
   - observes the screen once (throttled capture),
   - asks the fast model for a JSON plan (`goal`, `reasoning`, `steps[]`),
   - escalates to the smart model after two failed attempts,
   - keeps the planning-time screenshot on the plan (`plan.frame`) so
     bounding-box anchors can be cropped into templates later,
   - enforces `max_plan_steps` and the per-task LLM-call budget.
2. **Confirm** — the full plan is printed and the active UI waits for explicit
   approval: the GUI offers **Approve and run**, **Decline**, and **Edit and
   re-plan**; the console API accepts `y`, `n`, or `e`. **Nothing is executed
   before confirmation.**
3. **Execute** — `PlanExecutor.execute(plan)` runs each step:
   - fresh screen observation (1-second absolute capture floor),
   - target resolution in priority order: `icon:<name>` template match
     (against the saved-template library, which excludes one-off anchor crops)
     → OCR text match (whole-word label containment, or a label overlapping
     most of the description's meaningful words — a weak partial match yields
     no anchor instead of clicking an unrelated label) → planned bbox cropped
     from `plan.frame` and template-matched on the live frame → explicit
     model-supplied pixel (`x`/`y`) → focused-window fallback for
     `type`/`scroll`/`key_press`,
   - keyboard chords (`ctrl+shift+t`) and drags (anchor or explicit pixel →
     named or offset drop point) via `PyAutoGuiInput`; see *Keyboard and drag
     dispatch* below,
   - action via `PyAutoGuiInput` (`move` repositions the cursor without
     pressing; `click`/`double_click`/`right_click` settle `click_settle`
     seconds before pressing and honour `params.button` and `params.clicks`;
     smooth visible cursor movement by default; Windows typing uses a small
     per-character interval),
   - immediate `CONFIRM` logging after the input controller returns without
     an error,
   - one combined LLM review of the visible step result and next-step
     readiness (`FURTI_VERIFY_STEPS=true`); the review waits out the capture
     floor (at most one second) so it judges a frame taken **after** the
     action and selects `text`, `visual`, or `both` evidence. A `move` step is
     never judged a visual failure — repositioning the cursor changes nothing
     on screen, so a "failed" verdict would only trigger a pointless
     retry/re-plan loop,
   - failure → retry → per-step re-plan (smart-model escalation) with
     signature-based loop detection; after local retries are exhausted, the
     unfinished route is replaced from the current screen (bounded by
     `max_plan_replans`) and completed steps are not replayed.
4. **Compile** — every successfully anchored step is saved as a reusable
   reflex: crop → `templates/<task>_<step>.png` + `Skill` metadata in
   `memory.json` (description, action, screen size, anchor).
5. **Report** — `TaskJournal.write_report()` writes `<task_name>.md`
   (plan, per-step results, compiled reflexes, full timestamped log,
   tokens and approximate USD cost).

During planning, visual gating, verification, and adaptive replanning the
journal emits `THOUGHT`, `WAIT`, and response events. `WAIT` means the worker
is currently blocked on an LLM/network response; it is not an artificial
between-step delay. Successful steps proceed immediately after their input is
dispatched, subject to the render delay (below). The status window additionally
shows the last screenshot time and age, the latest model output, and a
color-highlighted current-action signal.

### The control JSON contract (`jsoncontract.py`)

Screen actions are model output, so a loosely read payload is a *movement* risk:
a stray string where a number belongs, a `NaN`, or a negative coordinate becomes
a cursor jump, a wrong click, or a drag that grabs the wrong thing. Two layers
keep that from reaching the input device.

**Ask for JSON.** `DeepSeekClient.chat_text`/`chat_vision` send
`response_format={"type": "json_object"}` and `GeminiClient` sends
`response_mime_type="application/json"` (`llm_json_mode`, default on). The hint
is an optimisation, not a requirement: a rejection that mentions the response
format/mime type is retried once without it (`_is_json_mode_rejection`), so an
endpoint that does not implement JSON mode still works. Every planner, review,
verifier and re-alignment prompt states the shape and asks for one bare object.

**Read strictly.** `extract_json_object` tolerates the wrappers a model adds
(fences, prose, nested braces inside strings) and refuses everything else:
empty answers, bare arrays, truncated objects, unbalanced braces and
`NaN`/`Infinity` (via a `parse_constant` hook). It is the only JSON reader in the
codebase -- the planner, verifier, re-aligner, post-step review and visual gate
all call it, so a payload cannot be parsed leniently in one place and strictly in
another.

**Validate before acting.** `PlanStep.from_dict` is where a control payload
enters the system, so the contract is enforced there:

| Field | Rule |
| --- | --- |
| `action` | must map onto a real `ActionType` (aliases and `ctrl+c`-style chords still resolve) -- an unknown name is **not** silently turned into a click |
| `x`/`y`, `point` | integers, `0`-`20000`, both coordinates present together; negative, non-numeric and absurd values raise |
| `bbox` | integer x/y/width/height; a negative origin or size raises, the documented all-zero placeholder stays "no bbox" |
| `text` | required for `type` (also read from `params.text`) |
| `params` | must be an object, never a list |

A violation raises `LLMJsonError`, which the planner treats as a failed attempt:
it re-asks and escalates to the smarter model (`_pick_model`) instead of
planning around a half-understood step.

**Refuse off-screen movement.** Even a well-formed pixel can be wrong (a
mis-scaled coordinate, a hallucinated value). Before dispatch,
`PlanExecutor._is_point_on_screen` checks the resolved grab and drop points
against `windows.virtual_screen_rect()`; a point on no monitor fails the step
with the reason recorded instead of throwing the cursor off-screen, where every
later click would land somewhere unintended. The *virtual* rectangle (all
monitors) rather than the primary monitor size keeps a second display at
negative coordinates working, and an unavailable rectangle fails open.

Step failures also keep their real reason now: giving up or loop detection
appends to the cause rather than replacing it, so a report says "resolved point
(4000, 3000) is outside the screen area; giving up after 1 attempts" instead of
only the last bookkeeping line.

### Interaction rules (focus, obstruction, timing)

Three rules shape every interaction, both in the planner prompt and in code that
enforces them regardless of what the model returns:

| Rule | Prompt contract | Enforcement |
| --- | --- | --- |
| Focus before typing | Never type without clicking the field first; name the field as `target` or explicit `x`/`y` | `TaskPlanner._enforce_focus_before_typing` inserts an explicit click step before every anchored `type` step; `PlanExecutor._type_focus_click` clicks a resolved field before typing; `_focus_click_already_happened` suppresses a second press when the previous step already clicked that point |
| Obstruction / modal dismissal | Inspect the UI state first; if a popup, cookie banner or modal covers the target, dismissing it IS the next step | `_obstruction_reason` asks `windows.window_at()` which top-level window owns the target pixel; `_clear_obstruction` dismisses (Close label, else ESC) or refocuses; `_clear_unresolved_obstruction` covers the "target not found + Close affordance present" case |
| Timing guards | Do not chain state-changing actions blindly; a UI needs 200-500 ms to render | `_await_render_delay` waits out `render_delay` after any action that can change the screen; a no-change failure is not repeated blindly -- the obstruction sweep runs before re-alignment |

**Focus before typing.** The sequence click -> verify -> type is made literal:
the focus click becomes its own plan step (visible in the preview, with its own
post-step review that checks the caret/focus for the next step), and the type
step keeps its anchor so the field is still re-resolved on the live screen. Two
clicks are never sent to the same field: `_focus_click_already_happened` treats a
click from the immediately preceding step within 12 px as "already focused", so
the executor's self-healing click does not fire on top of the explicit step. A
`type` step with no anchor at all is left alone -- the prompt reserves that for
"type into whatever already has focus" -- and `params.focus_click` overrides the
decision either way.

**Obstructions.** `window_at(x, y)` resolves the top-level window under a point
via `WindowFromPoint` + `GetAncestor(GA_ROOT)`, which is what tells an overlay
apart from a miss. A covering window is an obstruction when it is Furti's own
readout, or when the step names an intended window and something else owns the
pixel. Dismissal depends on size (`window_rect` against the captured frame): a
popup-sized window is a dialog and is dismissed by clicking an OCR label from
the close group (`Close`, `X`, `Dismiss`, ...) or, failing that, by pressing ESC;
a full-size other window is a different application, so the intended window is
refocused instead -- ESC there would do something unrelated. Close labels are
preferred over `OK`/`Accept`/`Allow`, which belong to the task rather than to the
recovery path, and a step whose own `target` is that `Close` is never dismissed
by it. When no window is named, the only obstruction the executor will act on is
its own always-on-top readout: guessing further would press ESC in unrelated
applications. Dismissals are budgeted by `max_obstruction_dismissals` and "do not
consume" a retry (`attempts` is rolled back), because the step never ran.

**Timing.** `render_delay` (0.35 s, `FURTI_RENDER_DELAY`) is enforced before any
action that follows a state-changing one, measured from the last dispatch
(including direct tools, which can also change the screen). `move` never waits:
it changes nothing on screen. The wait is `stop_event.wait(...)`, so STOP aborts
it immediately. The prompt additionally asks for an explicit `wait` step
(0.3-0.5 s) after navigation-triggering clicks; the post-step review's
`next_step_ready` is the "verify before the next click" half of the rule.

---

### Keyboard and drag dispatch

`pyautogui.press("ctrl+c")` is a **silent no-op**: `press` wraps the string into
`["ctrl+c"]`, `keyDown` finds no such entry in `pyautogui.KEYBOARD_KEYS` and
returns immediately, so the shortcut never reaches the application. The same
applies to every friendly spelling a model likes to emit (`escape`, `return`,
`cmd`, `pgup`, `del`, `page down`).

`keyboard.py` closes that gap. `KeyboardController` normalises aliases onto
pyautogui's names, splits a chord into `(modifiers, base key)` with the base key
last, and dispatches through `hotkey()`, which holds every key down together:

| Input | Dispatch |
| --- | --- |
| `enter`, `f5`, `+` | `press(key, presses=n, interval=gap)` |
| `ctrl+c` | `hotkey("ctrl", "c")` |
| `ctrl + shift + t` | `hotkey("ctrl", "shift", "t")` |
| `ctrl++`, `ctrl+plus` | `hotkey("ctrl", "+")` |
| `page down`, `caps lock` | `press("pagedown")`, `press("capslock")` |

Chord rules: case-insensitive; `+` or whitespace separates keys; a `+` with
spaces around it is a *separator* (`"ctrl + c"`), while a `+` glued to the
previous token is the plus key itself (`"ctrl++"`); `canonical_key` accepts
single characters, `f1`–`f24`, `num0`–`num9`, and the alias table. Two adjacent
tokens are rejoined only when the joined name is a known alias, so `"page down"`
becomes `pagedown` while `"ctrl+c"` stays two real keys. A chord with more than
one base key raises `ChordError` (plan one `key_press` step per keystroke), and
an unknown key name raises instead of doing nothing, so the executor's
retry/re-plan path can react.

### Cursor actions and explicit pixels

The plan vocabulary is `click`, `double_click`, `right_click`, `drag`, `move`,
`type`, `scroll` and `key_press`. `models.coerce_action` maps the spellings a
model actually emits onto that set, so an unknown name cannot silently degrade
into a click: exact enum names win, then the alias table (`mouse_move`,
`hover`, `click_at`, `send_keys`, …), then the same spelling with a decorative
token removed (`mouse_move` → `move`, `double_click_at` → `double_click`,
`mouse_right_click` → `right_click`), then the leading verb
(`move_the_mouse` → `move`).

`action=move` repositions the cursor without pressing anything; the click
actions press the button. A step carrying top-level `x`/`y` (or `point`) acts on
that exact pixel, which is the fallback for elements with no readable text or
saved icon template — without it, an unanchored step fails with "no target
anchor found". The point is converted from attached-image space to
`full_capture` pixels by `TaskPlanner._map_step_point_to_frame`, then scaled to
input space by `PyAutoGuiScreen.to_input_point`; unlike a resolved anchor it
keeps `capture_coordinates=True`. `params.button` (`left`/`middle`/`right`,
default `left`) and `params.clicks` (press count) are optional.

`PyAutoGuiInput.click` / `double_click` / `right_click` wait `click_settle`
(0.12 s by default, `FURTI_CLICK_SETTLE`) between the cursor arriving and the
button press. Windows delivers a press to whichever control currently owns the
pointer, so pressing immediately after `moveTo` can land on the previously
hovered control or be swallowed while the new one processes `WM_MOUSEMOVE` —
the "it moves there and doesn't click" symptom.

### Typing needs a caret, not just a window

`PlanExecutor._perform_action` dispatched `type_text` straight from the
resolved centre, so a multi-step `type` step only ever reached whichever
control already owned focus: the target was resolved (and logged) but never
clicked. The visible symptom is the user's "it starts typing but forgets to
click the input box first" — the text lands in the window that was focused
before, or nowhere at all. The legacy reflex replay
(`VisionReflex._perform_action`) always clicked first ("focus the field
first"), so the two paths disagreed; the multi-step executor was the wrong one.

`PlanExecutor._type_focus_click(step, target)` decides:

| Situation | Behaviour |
| --- | --- |
| `type` with a live anchor (`target` OCR/icon, planned bbox re-anchored, or explicit `x`/`y`); `TargetResolution.capture_coordinates` is true | Click the resolved point once (`PyAutoGuiInput.click`, same settle-then-press path as a click step) before typing |
| `type` with no anchor at all ("focused window (no anchor)") | Type where the focus already is — there is no field to click |
| `params.focus_click` | Forces either behaviour, so `false` means "type into the control that already has focus" |
| any other action | Never applies: `move` still only repositions, `scroll` resolves its own pane, `key_press` works on the focused window |

The flag is passed into `_perform_action(..., focus_click=...)` rather than
derived inside it, because only the caller knows whether the point came from
the screen or from the cursor fallback. The planner prompt asks for the field
as the `target` (or as explicit pixels) whenever the model can see it.

### Screen-corner abort (and why the mouse used to freeze)

pyautogui raises `FailSafeException` while the pointer sits exactly on a screen
corner: it is the documented "slam the mouse into a corner" human kill switch.
Combined with the planner schema advertising `"x": 0, "y": 0` as its "no
explicit pixel" placeholder, this produced a hard freeze — a model that echoed
the placeholder parked the cursor at `(0, 0)`, after which **every** later
`moveTo`/`click`/`mouseDown` raised before moving anything. The run looked like
"the agent planned but the mouse never moved", and the trace only showed up much
later as a step that "triggered the fail-safe".

Two guards prevent that:

* `PlanExecutor._resolve_target` discards a `(0, 0)` pixel unless the step sets
  `params.explicit_coordinate`, so a schema placeholder can never become a real
  click; the explicit-bbox-centre fallback is refused at the origin too.
* `PyAutoGuiInput` treats a corner park as an artifact: `_pointer_ready` nudges
the pointer 1 px inward (with the trip suspended for that single call) before
acting, and suspends the trip for gestures that *deliberately* target a corner
(the Start button lives in one). A drag whose grab or drop point is a corner
keeps the trip suspended for the whole hold→move→drop so `mouseUp` can always
fire. The first recovery is reported once through the journal.

`failsafe` (`FURTI_FAILSAFE`, default on) controls the trip itself. The abort a
user should rely on is the kill hotkey / STOP button, which the recovery
message points at.

`InputController.drag` performs hold → move → drop with the button pressed:

1. `hold_keys` (e.g. shift-drag to extend a selection) are pressed through
   `key_down`,
2. the cursor glides to the grab point,
3. `mouseDown(button=…)`, `moveTo(drop, duration=…)`, and `mouseUp(button=…)` —
   the release sits in a `finally` block so a failed move can never leave the
   mouse stuck down, and `key_up` releases `hold_keys` in an outer `finally`;
   the settle pause is applied before `mouseDown` and again after `mouseUp`.

The plan expresses a drag as a grab point plus a drop hint. The grab point is
`target`/`bbox` when the item is visible, `params.from_x`/`from_y` when it is
not, and otherwise the current cursor position (for "drag whatever is under the
pointer"). The drop hint is resolved in this order:

| Drop hint | Meaning |
| --- | --- |
| `to_target` | Named element/OCR label, re-resolved on the live frame |
| `to_bbox` | Planned bounding box, re-anchored like a normal step bbox |
| `to_x` + `to_y` | Explicit drop coordinates |
| `dx` / `dy` | Offset from the resolved grab point |

`button` (`left`/`middle`/`right`), `hold_keys` and `duration` are optional. An
unresolved named drop target is a **step failure**, not a guess: dropping a
payload at a stale coordinate is worse than not dragging. Compiled drag reflexes
store the delta in input space so replay reproduces the same gesture.

A drag releases the button, so it never clicks on its own. When a task needs a
press at the drop position the plan emits a separate `click` step at the same
coordinates; the executor does not synthesize one because "drag then click" and
"drag only" are different gestures. The post-`mouseUp` settle gives the target
window time to register the drop before that click is delivered.

A grab point and a drop point can live in different coordinate spaces (cursor in
input space, model-supplied pixel in capture space), which is why
`TargetResolution` carries `drop_capture_coordinates` separately from
`capture_coordinates`.

### Direct tools (the fast path)

Driving the mouse is the *fallback*, not the default. Many steps can be finished
by asking Windows to do the work, so `models.ActionType` carries a second group
of actions — the direct tools — that `executor.PlanExecutor` routes to
`tools.DirectToolRunner` instead of the input device.

| Action | Key params | What it replaces |
| --- | --- | --- |
| `launch_app` | `app`, `args`, `cwd`, `settle` | Win+R → type → Enter → wait for the window |
| `open_path` | `path` (file, folder or URL) | Explorer navigation / typing a URL into the address bar |
| `run_command` | `command`, `cwd`, `timeout`, `background`, `confirm` | Opening a terminal and typing the command |
| `write_file` | `path`, `content`, `append` | Open editor → type → Save → choose filename |
| `read_file` | `path`, `max_chars` | Open file → select all → copy |
| `screenshot` | `region`/`x,y,width,height`/`full`, `path`, `label`, `format`, `scale` | Print Screen → paste into an editor → Save As |
| `set_clipboard` | `content` | Type text then Ctrl+A / Ctrl+C |
| `get_clipboard` | `max_chars` | Focus a window → Ctrl+V → read |
| `focus_window` | `window` (or active window) | Alt+Tab hunting |
| `list_windows` | — | Reading the taskbar / Alt+Tab switcher |
| `close_window` / `minimize_window` / `maximize_window` | `window` (or active window) | Clicking the caption buttons (or title-bar right-click) |
| `create_folder` | `path` | Right-click → New → Folder → rename |
| `list_dir` | `path`, `pattern`, `include_hidden`, `limit` | Opening Explorer to see what is there |
| `copy_path` / `move_path` | `source`, `destination` | Drag-and-drop, or cut → paste → rename |
| `delete_path` | `path`, `confirm` | Right-click → Delete (Recycle Bin when `send2trash` is present) |
| `find_files` | `root`, `pattern`, `max_depth`, `limit` | Explorer search box |
| `path_info` | `path` | Hovering to check whether something exists |
| `wait` | `seconds` | Polling with screenshots |

**Region screenshots.** `_screenshot` captures through the injected
`ScreenCapture`, clamps the requested region to the frame, optionally downscales
it (`params.scale`), writes it to `<workspace>/screenshots/` (or an explicit
`params.path`, with `~`/`%VAR%` expansion) and returns the absolute path as the
tool's `output` — which lands in the step notes, the journal and the `.md`
report. The model can therefore refer to the exact image it reasoned about, and
so can the user. Region parsing accepts a nested `region`/`bbox`/`crop` object
(`x/y/width/height`, corner keys, or a four-item list) and the flat
`x`/`y`/`width`/`height` spellings.

**File work stays out of the GUI.** The planner prompt states that folders,
renames, moves, copies, deletes, listings, searches and existence checks are
tool work, never Explorer work. All path arguments expand `~` and environment
variables, and relative paths resolve against the workspace rather than the
process CWD so a plan means the same file wherever Furti was started.

**Delete safety** is layered: `params.confirm: true` (or
`FURTI_ALLOW_DESTRUCTIVE_COMMANDS=true`) is required, the Recycle Bin is
preferred via `send2trash`, and drive roots, the home directory, the workspace,
the templates directory and the memory file are refused outright.


Why they are faster, and why they are safer:

* **No grounding, no review.** `PlanExecutor._execute_tool_step` returns before
  the screen observation, the OCR pass, the template match and the
  `_review_step_and_next` model call. A tool already reports a definitive
  outcome, so paying for pixels to confirm it would be pure latency and tokens.
* **No reflex compile.** A tool call is already milliseconds; there is nothing
  worth caching, so no template crop is written.
* **Verified by the OS, not by the model.** `launch_app` returns a PID,
  `run_command` an exit code, `write_file` a byte count. These land in the step
  notes and the journal (`tool result`, `tool latency`, `tool output`).
* **Failures stay recoverable.** A tool failure becomes
  `ToolResult(ok=False, detail=...)` → a failed `StepResult` → the normal
  adaptive re-plan path (`_adaptive_replan`), so the model can pick another
  route.
* **Irreversible commands are refused.** `tools._destructive_reason` matches
  disk formatting, `diskpart`, recursive force deletes, `DROP TABLE`, bucket
  mass-deletes and force-pushes. Such a command must carry
  `params.confirm: true` (or `FURTI_ALLOW_DESTRUCTIVE_COMMANDS=true`) to run.
  A plan preview is not consent for destroying data.
* **Switches.** `FURTI_DIRECT_TOOLS=false` turns the whole group off,
  `FURTI_ALLOW_SHELL_COMMANDS=false` disables only `run_command`, and
  `FURTI_TOOL_TIMEOUT` bounds a hung command. Both flags are exposed in the GUI
  settings panel.

The planner prompt lists these actions and tells the model to prefer them
whenever one can reach the result, falling back to `click`/`type`/`key_press`
only for things that exist purely inside a GUI (a canvas, a custom dialog, a
web widget). The legacy single-action `BrainPlanner` deliberately keeps the
narrow `REFLEX_ACTIONS` enum, because its whole contract is to compile the
chosen action into a template reflex — and a tool has no template.

### Reflex quality gate (reflexes are earned)

`PlanExecutor._compile_reflex` used to cache *every* successful step that had a
template. That is the wrong bar: a blurry crop, a whole-screen grab or a one-off
typed paragraph costs a template file *and* a false cache hit on every later
run, which is worse than having no reflex at all. `_reflex_skip_reason` now
decides, and every refusal is journalled with its reason:

| Rule | Setting (default) | Why |
| --- | --- | --- |
| Action must be replayable | `REFLEX_ACTIONS` membership | Tools have no template to match |
| A template must exist and be readable | — | Nothing to match otherwise |
| The anchor must be *verified* | — | `explicit bbox centre` / `focused window` anchors are guesses |
| Match confidence must clear the bar | `reflex_min_anchor_confidence` (0.75) | A weak anchor becomes a wrong click later |
| Description must be specific | `reflex_min_description_chars` (8) | It is the cache key; `click` collides with everything |
| Template must be real geometry | `reflex_min_template_side` (6 px), `reflex_max_template_area_ratio` (0.6) | A 3 px crop cannot match; a whole-screen crop matches anything |
| Typed payload must be reusable | `reflex_max_typed_chars` (160) | Replaying a reflex re-types the same text |
| The gate must be enabled | `reflex_enabled` | One switch to stop caching entirely |

### Reflex re-alignment (repair before you re-plan)

A stored reflex that stops matching is usually *stale*, not wrong: the window
moved, the list scrolled, the theme changed. Two paths handle that:

* **Replay path** — `AgentOrchestrator._handle_reflex_failure` counts the miss,
  retires the reflex when it has failed `reflex_retire_failures` times, and
  otherwise calls `ReflexRealigner.realign`: the LLM receives the live
  screenshot plus the reflex's identity (name, action, original description,
  target label, last known bbox, anchor, previous re-alignments, failure reason)
  and must answer `{found, bbox, confidence, description, note}`. The answer is
  only accepted with `found: true`, a bbox that is on-frame, at least 4 px per
  side and no more than 60 % of the screen, **and** a confidence at or above
  `reflex_min_anchor_confidence` — overwriting a stored template on an
  unquantified guess is how a good reflex becomes a bad one. Accepted answers
  overwrite the template and `expected_bbox`, bump `realign_count`, reset
  `failure_count`, and are replayed once.
* **Step path** — inside a multi-step task, the *first* failure of a screen
  action triggers `TaskPlanner.realign_step` (same intent, same action where
  possible, corrected anchor from the current grounding) instead of
  `replan_step` (a different action). The realigned step is retried without
  registering its signature as "seen" — that is deliberate, because the whole
  point is to keep the route while the anchor changes; the attempt budget still
  bounds the loop. A second failure routes through `replan_step` as before. The
  stale cached reflex for that step is counted, and retired at the limit.

### Reflexes are opt-in (`Skill.enabled`)

The quality gate decides *whether* a step may become a reflex; `Skill.enabled`
decides whether that reflex may ever fire without the planner. It defaults to
`False`, so `PlanExecutor._compile_reflex` stores every new reflex disabled and
the journal says so. Consequences:

* `MemoryManager.get_skill(name, require_enabled=True)` is what the replay path
  (`AgentOrchestrator.run`) calls, so a stored-but-disabled reflex is a cache
  miss and the planner runs as usual. Stats-only callers (`record_success`,
  `_prune_reflex_for_step`) keep the unguarded lookup and still find it.
* A disabled reflex therefore accumulates no hits, no misses and no
  re-alignments — `reflex_retire_failures` can never delete a reflex that was
  never allowed to run.
* `MemoryManager.set_enabled(name, enabled)` / `set_all_enabled(enabled)` persist
  the flip through the normal `memory.json` save, and `Skill.from_dict` reads the
  flag with a `False` default so caches written before this feature stay opted
  out rather than silently becoming live.
* The GUI's **Reflexes** tab is the only place the flag is set: it lists each
  reflex (name, action, ok/miss counters), one checkbox per reflex plus the
  `Enable all` / `Disable all` / `Refresh` controls, and a `x` button that
  deletes the entry and its own crop (user icon templates are left alone).
  `reflex_enabled` remains the master switch shared with the Safety tab: off
  stops compiling, on compiles but stays per-reflex opt-in.

### The context file (`profile.py`)

`UserContext` detects the machine and its owner once, stores it in
`<workspace>/context/profile.json` and writes a readable mirror to
`profile.md`. Detected: identity (user/host/OS/Python/shell/workspace), display
size and Windows scaling, the real Desktop/Documents/Downloads paths (OneDrive
redirects included), drives with free space, installed applications (launchable
aliases first, then filtered Start-menu shortcuts) and available CLI tools.

It is refreshed when older than `profile_max_age_days`, and every section is
detected behind its own `_safe()` wrapper: one exotic probe costs one fact, not
the profile. `prompt_block()` renders a compact block that is injected into the
plan, re-align, re-plan and verification prompts, so the model stops guessing at
paths and application names.

Users can teach it: bullets under `## Notes` in `profile.md` are parsed back as
user facts on load, alongside `FURTI_USER_NOTES` and `UserContext.remember()`,
and `record_task()` keeps a bounded history of what the agent was asked to do.

### Cross-provider verification (`verifier.py`)

One model, one API key, one set of blind spots. `CrossVerifier` reviews critical
steps on the **other** provider (Gemini when the planner is DeepSeek and vice
versa, via `build_secondary_llm`), which is why it is a different *client and
key* rather than a different prompt.

* **Criticality** (`criticality`) is concrete: `run_command` (always, and
  specially flagged when `_destructive_reason` matches), `delete_path`,
  `move_path`, `write_file` that overwrites an existing file, `copy_path` onto an
  existing destination, `close_window`, committing key chords (`enter`, `ctrl+s`,
  `alt+f4`, `shift+delete`, …), and anything marked `params.critical`. Adding a
  folder, launching an app, taking a clipboard snapshot or clicking a button is
  never audited — that would cost latency and buy nothing.
* **Plan review** is advisory: `TaskAgent._review_plan` runs it before the user
  approves, prints any objection inside the confirmation prompt, and journals it.
* **Step review** is enforcing: `PlanExecutor._verify_or_raise` runs immediately
  before dispatch (for direct tools and for screen actions alike) and a
  rejection raises `StepBlockedByVerifier`, which the step loop converts into a
  failure note containing the verifier's reason **and** its suggested safer
  alternative — so the existing re-align/re-plan machinery works from the
  objection rather than discarding it.
* **Fail-open and bounded**: no second key, a call error, an unparsable verdict
  or an exhausted `cross_verify_max_calls` budget (default 12 per task) all mean
  "proceed", with the reason journalled. A verifier that halts work when it is
  merely unavailable would be worse than no verifier at all.


---

## 2. Feature / function registry

| # | Feature | Where | Status |
| --- | --- | --- | --- |
| 1 | Multi-step task planning (JSON plan with goal/reasoning/thoughts) | `planner.py` — `TaskPlanner`, `TaskPlan`, `PlanStep` | ✅ |
| 2 | Reuse of stored reflexes (planning hints + executor icon anchors) | `memory.py`, `ocr.py` — `IconMatcher` | ✅ |
| 3 | Compile successful steps into new reflexes | `executor.py` — `PlanExecutor._compile_reflex` | ✅ |
| 4 | Dynamic re-anchoring of planned bboxes on the live screen | `executor.py._resolve_target` + `vision.py.locate_on` | ✅ |
| 4a | Screenshot/model/DPI coordinate normalization | `planner.py`, `screen.py`, `vision.py` | ✅ |
| 5 | Console + GUI log of every thought/step/action (`[HH:MM:SS] [KIND]`) | `tasklog.py` — `TaskJournal`, `gui.py` — `FurtiApp` | ✅ |
| 6 | Cursor moves smoothly instead of teleporting (default) | `controller.py` — `PyAutoGuiInput._goto`; `cursor_teleport` flag | ✅ |
| 6a | Windows-safe typing pacing and responsive input timing | `controller.py` — `typing_interval`, `input_pause` |
| 6b | Real keyboard shortcuts (`ctrl+shift+t`, `alt+tab`, `win+r`, `page down`) | `keyboard.py` — `KeyboardController`, `parse_chord`, `ChordError`; `controller.py.press_key` |
| 6c | Cursor drag with button and modifier hold (`shift`-drag, right-drag) | `controller.py.drag` + `executor.py._resolve_drag_endpoint` | ✅ |
| 6d | Model-driven cursor move, explicit-pixel click/drag and repeat clicks | `models.py.coerce_action`, `planner.py._normalise_step_point`, `executor.py._step_point`, `_resolve_drag_target` | ✅ |
| 6e | Button-press settle so a click cannot race the pointer move | `config.py.click_settle` + `controller.py._settle` | ✅ |
| 6f | Direct tools: `launch_app`, `open_path`, `run_command`, `write_file`, `read_file`, clipboard, window state, `wait` | `tools.py` — `DirectToolRunner`; `executor.py._execute_tool_step` | ✅ |
| 6g | Tool steps skip screen capture, OCR and model review, and compile no reflex | `executor.py._execute_tool_step` | ✅ |
| 6h | Irreversible-command guard (`params.confirm` / `FURTI_ALLOW_DESTRUCTIVE_COMMANDS`) | `tools.py._destructive_reason` | ✅ |
| 6i | Window state control by title or active window | `windows.py` — `resolve_hwnd`, `minimize_window`, `maximize_window`, `restore_window`, `close_window` | ✅ |
| 6j | Region/full screenshots saved locally, path reported in the step result | `tools.py._screenshot`, `config.screenshots_dir` | ✅ |
| 6k | File-system tools so Explorer is never used for file work | `tools.py` — `create_folder`, `list_dir`, `copy_path`, `move_path`, `delete_path`, `find_files`, `path_info` | ✅ |
| 6l | Reflex quality gate: steps are cached only when successful *and* reusable | `executor.py._reflex_skip_reason` | ✅ |
| 6m | Reflex re-alignment on replay failure (LLM re-anchors, template rewritten) | `reflex.py` — `ReflexRealigner`; `orchestrator.AgentOrchestrator._handle_reflex_failure` | ✅ |
| 6n | Step re-alignment before route re-planning | `planner.TaskPlanner.realign_step`; `executor.py._try_realign` | ✅ |
| 6o | Useless reflexes retired after repeated misses | `executor.py._retire_stale_reflex`, `Skill.record_failure`, `reflex_retire_failures` | ✅ |
| 6p | New reflexes are stored disabled and replayed only once enabled | `models.Skill.enabled`, `memory.MemoryManager.get_skill(require_enabled=True)` / `set_enabled` / `set_all_enabled`, `orchestrator.AgentOrchestrator.run` | ✅ |
| 6q | A `type` step clicks its resolved field before typing (caret placement) | `executor.py._type_focus_click`, `_perform_action(..., focus_click=)`, `params.focus_click` | ✅ |
| 6r | Focus click is its own plan step, and never repeated on the same field | `planner.py._enforce_focus_before_typing`, `_focus_step_for`, `executor.py._focus_click_already_happened` | ✅ |
| 6s | Modals/popups/overlays covering the target are dismissed (or their window refocused) before the step runs | `executor.py._obstruction_reason`, `_clear_obstruction`, `_clear_unresolved_obstruction`, `windows.window_at`/`window_rect`/`is_furti_window` | ✅ |
| 6t | Render delay paces state-changing actions; a no-change failure is not repeated blindly | `executor.py._await_render_delay`, `_last_dispatch_at`, `render_delay` | ✅ |
| 6u | Control payloads must be one strict JSON object; the endpoint is asked for JSON-only output | `jsoncontract.py`, `planner.PlanStep.from_dict`, `brain.DeepSeekClient._create_json` / `GeminiClient._generate_json`, `llm_json_mode` | ✅ |
| 6v | An off-screen resolved point is refused instead of dispatched | `executor.py._is_point_on_screen`, `windows.virtual_screen_rect` | ✅ |
| 6p | User/system context file (paths, apps, drives, notes) injected into prompts | `profile.py` — `UserContext`; `<workspace>/context/profile.{json,md}` | ✅ |
| 6q | User-editable notes in `profile.md` reloaded as facts | `profile.py` — `_notes_from_markdown` | ✅ |
| 6r | Cross-provider verification of critical steps (different provider + API key) | `verifier.py` — `CrossVerifier`, `criticality`; `orchestrator.build_secondary_llm` | ✅ |
| 6s | Plan-level advisory review before user approval | `agent.py._review_plan` | ✅ |
| 6t | Gemini client at parity with DeepSeek (function calling, JSON mode, both SDKs) | `brain.py` — `GeminiClient`; `GEMINI_USE_FUNCTION_CALLING` | ✅ |
| 7 | `<task_name>.md` report after every task | `tasklog.py` — `TaskJournal.write_report` | ✅ |
| 8 | Anti-endless-loop guardrails | `config.py` guardrails + `PlanStep.signature` + `BudgetExceeded` | ✅ |
| 8a | Adaptive replacement route after a failed step | `executor.py._adaptive_replan` + `TaskPlanner.replan_remaining` | ✅ |
| 9 | Screenshot throttle (never every second) | `context.py` — `screenshot_min_interval` + 1 s absolute floor | ✅ |
| 10 | Fast OCR text grounding before the LLM | `ocr.py` — `TextDetector` (RapidOCR ONNX default with an automatic PaddleOCR fallback) | ✅ |
| 11 | Cached icon recognition from saved templates | `ocr.py` — `IconMatcher` (downscaled exact-scale matchTemplate; multiscale opt-in; one-off anchor crops excluded) | ✅ |
| 12 | LLM decides whether a raw screenshot is needed (visual gate) | `context.py` — `GATE_SYSTEM_PROMPT`, cached per instruction | ✅ |
| 13 | Always-on-top Tk status window (task/phase/step/log/tokens) | `status.py` — `StatusWindow` | ✅ |
| 13a | Screenshot age, latest AI output, and current-action visual signals | `status.py` + `TaskJournal` | ✅ |
| 13b | Same always-on-top readout inside the GUI (embedded `Toplevel`, toggle + auto-hide) | `status.py` + `gui.py` — `StatusWindow(master=…)`, *Always-on-top status window* checkbox | ✅ |
| 14 | Global kill hotkey (`<ctrl>+<alt>+k`) + window STOP button | `status.py` — `KillSwitch` (pynput) | ✅ |
| 15 | Approximate cost tracking (tokens × model price) | `cost.py` — `UsageTracker`, `MODEL_PRICES` | ✅ |
| 16 | Fast → smart model escalation | `planner.py` — `_pick_model`, `replan_step`; `fast_model`/`smart_model` | ✅ |
| 16a | Explicit DeepSeek image-capable provider selection | `config.py` / `orchestrator.py` — `FURTI_LLM_PROVIDER`, `deepseek-flash` | ✅ |
| 17 | Full transparency (console + Tk mirror of every log line) | `tasklog.py` + `status.py` | ✅ |
| 17a | Full Tkinter application (task editor, settings, plan approval, status and logs) | `gui.py` — `FurtiApp`; `app.py` — `launch_gui` | ✅ |
| 18 | Explicit confirmation before executing the plan | `agent.py` — `_confirm_plan`; `gui.py` — `ConfirmationRequest` | ✅ |
| 18a | Per-action dispatch confirmation plus combined current/next-step review | `executor.py` — `CONFIRM` journal events, `StepReview`, `verify_steps` | ✅ |
| 19 | Graceful degradation (no OCR, no tkinter, no pynput) | `ocr.py`, `status.py`, `context.py` fallbacks | ✅ |
| 20 | Legacy reflex demo unchanged | `__main__.py` (no args), `AgentOrchestrator` | ✅ |

---

## 3. Module map

| Module | Contents |
| --- | --- |
| `config.py` | `Settings`: guardrails, cursor, models/provider, OCR, window, hotkey, paths (env-overridable) |
| `planner.py` | `TaskPlanner`, `PlanStep`, `TaskPlan`, `BudgetExceeded`, `PLAN_SYSTEM_PROMPT` |
| `jsoncontract.py` | the strict control-JSON contract: `extract_json_object`, `as_pixel`/`as_optional_int`/`as_text`/`as_optional_bool`, `LLMJsonError` |
| `executor.py` | `PlanExecutor`, `StepResult`, `ExecutionReport`, step verification, direct-tool fast path, reflex gate, re-alignment, cross-verification hook |
| `tools.py` | `DirectToolRunner`, `ToolResult`, `is_direct_tool`, destructive-command guard, app aliases, clipboard (ctypes), screenshot + file-system tools |
| `profile.py` | `UserContext` — the persisted user/system context file and its prompt rendering |
| `verifier.py` | `CrossVerifier`, `CrossReview`, `criticality` — independent second opinion on the other provider |
| `reflex.py` | `ReflexRealigner`, `RealignResult` — LLM re-anchoring and retirement of stale reflexes |
| `context.py` | `VisualContextManager`, `SceneObservation`, `TaskAborted`, visual gate |
| `ocr.py` | `TextDetector` (RapidOCR/Paddle compatibility), `IconMatcher` (cached cv2 matchTemplate), `describe_scene` |
| `tasklog.py` | `TaskJournal` (console + status sink + report writer) |
| `gui.py` | `FurtiApp` (Tk main window, settings form, worker queue, plan approval, live log) |
| `status.py` | `StatusWindow` (Tk, always-on-top, capture/AI/action signals), `KillSwitch` (pynput) |
| `cost.py` | `UsageTracker`, `CostSummary`, `MODEL_PRICES`, `make_usage_callback` |
| `agent.py` | `TaskAgent` — the end-to-end runner (plan → confirm → execute → report) |
| `brain.py` | `DeepSeekClient`, `GeminiClient` (chat_text/chat_vision + usage callback), `BrainPlanner` |
| `controller.py` | `InputController` protocol, `PyAutoGuiInput` (smooth `_goto`, chords, drag) |
| `keyboard.py` | `KeyboardController` (chord parse/alias/dispatch), `parse_chord`, `ChordError` |
| `vision.py` | `VisionReflex` (+ `locate_on` for live re-anchoring) |
| `memory.py` | `MemoryManager` (reflex cache, `save_skill`, `normalize_name`) |
| `orchestrator.py` | `AgentOrchestrator`/`build_agent` (legacy), `build_task_agent` (wiring + UI hooks) |

---

## 4. Guardrails (why the agent cannot loop forever)

| Guardrail | Default | Effect |
| --- | --- | --- |
| `max_plan_steps` | 10 | Plan is truncated; a task can never grow unbounded |
| `max_step_retries` | 2 | Re-anchor + re-plan attempts per step |
| `max_plan_replans` | 3 | Full replacements of the unfinished route per task |
| `max_llm_calls_per_task` | 100 | Hard LLM budget; `BudgetExceeded` stops planning/re-planning |
| `max_consecutive_failures` | 3 | Aborts the whole run after N failed steps in a row |
| `cross_verify_max_calls` | 12 | Per-task ceiling for independent-verification calls; the budget running out means "proceed", never "block" |
| `reflex_retire_failures` | 3 | A reflex that misses this many replays is deleted instead of retried forever |
| `reflex_max_template_area_ratio` | 0.6 | Refuses to cache a template that covers most of the screen |
| `profile_max_age_days` | 7 | The context file is re-detected at most this often |
| `screenshot_min_interval` | 4.0 s | Normal captures throttled |
| absolute capture floor | 1.0 s | Even `force_fresh` captures cannot fire faster |
| `FURTI_OCR_BACKEND` | `rapidocr` | Preferred backend: `rapidocr` fast default, `paddle`, or `auto`. An unavailable/failed RapidOCR install falls back to PaddleOCR automatically, and the real backend state is logged per task |
| `FURTI_OCR_MAX_DIM` | 640 | Downscales OCR input for responsive desktop grounding, then restores boxes to capture pixels |
| `FURTI_OCR_MIN_CONFIDENCE` | 0.35 | Drops low-confidence OCR lines before the LLM sees them |
| `FURTI_OCR_MAX_LINES` | 80 | Bounds the number of OCR lines included in grounding |
| `FURTI_ICON_MAX_TEMPLATES` | 24 | Bounds saved templates scanned per frame |
| `FURTI_ICON_MAX_DIM` | 1280 | Downscales the frame for icon matching, then restores match boxes |
| `FURTI_ICON_MULTISCALE` | false | Enables the slower nine-scale icon fallback |
| `FURTI_OCR_MKLDNN` | false | Opts into PaddlePaddle oneDNN CPU inference for the legacy backend |
| `PlanStep.signature()` | — | sha1(description\|action\|target); a re-planned step that repeats a signature is aborted immediately |
| stop event | hotkey/STOP | Checked before every capture, LLM call and action |

No fixed sleep is inserted between successful steps. The executor proceeds
immediately; the only natural pauses are cursor movement, OS input delivery,
screen-capture throttling (which reuses the cached frame instead of sleeping),
and actual model/network response time.

### Coordinate spaces

There are three coordinate spaces that must not be mixed:

1. **Capture pixels** — the full BGR frame returned by `PyAutoGuiScreen`.
   OCR boxes, icon matches, and live template matches are reported here.
2. **Attached vision pixels** — the optional screenshot sent to the model.
   It may be downscaled. A plan can mark a bbox as `attached_image`; the
   planner restores it to capture pixels before execution. Bboxes outside the
   attached image bounds are treated as full-capture coordinates because the
   model also receives OCR coordinates in that space.
3. **PyAutoGUI input pixels** — the coordinates consumed by `moveTo`/`click`.
   `PyAutoGuiScreen.to_input_point` scales capture pixels to this space,
   covering Windows DPI scaling. Logs print both `frame=(x, y)` and
   `input=(x, y)` for every anchored action.

The executor never accepts a weak reverse substring match such as OCR `"A"`
for a descriptive target like `"Chrome icon on taskbar"`; this prevents
single-character OCR noise from sending the cursor to an unrelated point.

After execution, `PlanExecutor.close()` shuts down the context's two-worker
grounding pool. This prevents repeated GUI runs from accumulating idle
grounding threads.

---

## 5. Cost model (approximate)

Every LLM call reports `(model, prompt_tokens, completion_tokens)` via the
client's `usage_callback` into `UsageTracker`. At the end of a task,
`CostSummary` applies `MODEL_PRICES` (USD per **1M tokens**, approximate
Google Gemini list prices; env-overridable via `FURTI_MODEL_PRICES`):

| Model tier | Input / 1M | Output / 1M |
| --- | --- | --- |
| Gemini flash tier | $0.30 | $2.50 |
| Gemini pro tier | $1.25 | $10.00 |
| Gemini ultra tier | $2.00 | $12.00 |
| DeepSeek chat | $0.27 | $1.10 |
| DeepSeek reasoner | $0.55 | $2.19 |
| Unknown model | flash-tier fallback | flash-tier fallback |

Model names are matched by substring (`flash` / `pro` / …), so version
bumps keep the estimate roughly right.

### Worked example (Gemini flash, `gemini-3.6-flash`)

A typical task consumes roughly:

| Call | Prompt tokens | Completion tokens |
| --- | --- | --- |
| 1 visual gate (text-only) | 300 | 20 |
| 1 plan call (text grounding) | 900 | 400 |
| 1 combined step/next-step review × 6 steps | 6 × 400 = 2400 | 6 × 45 = 270 |

**Total ≈ 3,600 prompt + 690 completion tokens.**

- Prompt cost: 3,600 × $0.30 / 1,000,000 = **$0.0011**
- Completion cost: 690 × $2.50 / 1,000,000 = **$0.0017**
- **≈ $0.0028 per task** (~0.28 cents).

The dominant variable is **screenshots**: each downscaled 1280-px PNG adds
≈ 1,100–1,300 image tokens. The visual gate and combined-review evidence
selection keep these out of most calls; the 1-second capture floor keeps them
out of capture loops. If the plan call attaches one screenshot (~1,200 tokens)
the task cost rises to ≈ **$0.003**.

The console and the report both print the real, measured usage:

```
[COST] LLM calls: 8
[COST] Tokens: 3600 prompt + 690 completion = 4290
[COST] Approximate total API cost: $0.0028 USD
```

---

## 6. Usage

```powershell
# optional but recommended: fast OCR grounding
pip install -r requirements-ocr.txt

# Google Gemini (preferred)
$env:GOOGLE_API_KEY = "..."
# optional model tiers
$env:FURTI_FAST_MODEL  = "gemini-3.6-flash"
$env:FURTI_SMART_MODEL = "gemini-3.6-pro"

python app.py
```

The window provides task entry, provider/model settings, an advanced settings
panel, a proposed-plan preview, and **Approve**, **Edit and re-plan**, and
**Decline** controls. It prints every thought/action in the in-app event log
and the console, and waits for approval before touching the mouse. Press
`<ctrl>+<alt>+k` or the in-app Stop button to abort safely at the next step
boundary.

For a non-interactive wrapper, the same pipeline remains available:

```powershell
python app.py --task "Open Notepad, write 'hello', and save the file"
```

Reports land in `~/.furti_ai/reports/<task_name>.md`; reflexes and templates
in `~/.furti_ai/templates/` + `~/.furti_ai/memory.json`.

### Parallel verification (two DeepSeek keys)

Verification used to be strictly serial: dispatch, review, then the next step.
With two DeepSeek keys the reviewer is its own client, so the pair runs at once
and a step costs the slower call instead of the sum of both.

* `orchestrator.build_secondary_llm` returns a `DeepSeekClient` bound to
  `deepseek_api_key_2` (provider label `deepseek-secondary`, model
  `deepseek_reasoning_model`) whenever the primary plans with DeepSeek and a
  second key exists. The reviewer stays on the fast model (thinking off) because
  it runs after every dispatched step; the deep thinking pass is the escalation
  client, consulted only after a failure. With one key it falls back to the previous
  opposite-provider rule; with no key verification stays off (fail-open).
* `verifier.CrossVerifier.review_progress` audits the *route* after dispatch.
  It is deliberately not filtered by `criticality`, because its value is
  noticing that an ordinary click landed somewhere wrong or that a popup now
  covers the page. It spends from the same `cross_verify_max_calls` budget and
  fails open.
* `executor.PlanExecutor._parallel_review` submits both to a two-worker
  `ThreadPoolExecutor` and merges the verdicts: the primary review answers "did
  this step work / is the next step ready?", the monitor answers "is the route
  still grounded?". A monitor rejection becomes an ordinary step failure, so
  the existing re-align/re-plan path handles it; an erroring, disabled or
  budget-exhausted monitor leaves the primary verdict untouched.
* Losing the second key costs latency and the independent rate limit, never
  correctness: the pair degrades to the single-review path.

### Key environment variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `GOOGLE_API_KEY` | Gemini API key (falls back to DeepSeek if unset) | — |
| `DEEPSEEK_API_KEY` | Primary DeepSeek key used for planning and inference | — |
| `DEEPSEEK_API_KEY_2` | Second DeepSeek key for the independent reviewer, so verification has its own client, rate limit and model | — |
| `DEEPSEEK_REASONING_MODEL` | Escalation model for the deep pass, used only after a failure | `deepseek-flash` |
| `FURTI_DEEPSEEK_ESCALATION_THINKING` | Thinking mode for the escalation client only (routine clients keep function calling) | `true` |
| `FURTI_KEYS_FILE` | Path to the ignored `keys.json` when it is not at the repository root (authoritative when set) | — |
| `FURTI_VERIFY_PROGRESS` | Independent route monitor after each dispatched step | `true` |
| `FURTI_PARALLEL_VERIFY` | Run the step review and the route monitor concurrently | `true` |
| `FURTI_FAST_MODEL` / `FURTI_SMART_MODEL` | tiered models | provider default |
| `FURTI_LLM_PROVIDER` | `auto`, `deepseek`, or `gemini`; auto prefers DeepSeek when its key is present | `auto` |
| `FURTI_OCR_BACKEND` | Preferred backend: `rapidocr`, `paddle`, or `auto` (automatic PaddleOCR fallback) | `rapidocr` |
| `FURTI_OCR_MAX_DIM` | Maximum OCR input dimension; boxes are scaled back to capture pixels | `640` |
| `FURTI_OCR_MIN_CONFIDENCE` | Minimum retained OCR confidence | `0.35` |
| `FURTI_OCR_MAX_LINES` | Maximum retained OCR lines | `80` |
| `FURTI_ICON_MAX_TEMPLATES` | Maximum saved templates scanned per frame | `24` |
| `FURTI_ICON_MAX_DIM` | Maximum icon-matching screen dimension | `1280` |
| `FURTI_ICON_MULTISCALE` | Slower multi-scale icon fallback | `false` |
| `DEEPSEEK_MODEL` | OpenAI-compatible DeepSeek model; `chat_vision` sends `image_url` | `deepseek-flash` |
| `FURTI_CURSOR_TELEPORT` | `true` restores instant cursor jumps | `false` (smooth move) |
| `FURTI_CURSOR_MOVE_DURATION` | maximum smooth cursor travel duration (seconds) | `0.5` |
| `FURTI_INPUT_PAUSE` | pause after each low-level PyAutoGUI call (seconds) | `0.03` |
| `FURTI_TYPING_INTERVAL` | delay between typed characters (seconds) | `0.1` |
| `FURTI_DRAG_DURATION` | maximum cursor travel duration for a drag gesture (seconds) | `0.4` |
| `FURTI_CLICK_SETTLE` | pause between the cursor arriving and the button press (seconds) | `0.12` |
| `FURTI_RENDER_DELAY` | wait after a state-changing action before the next interaction, so it cannot race the UI render (seconds) | `0.35` |
| `FURTI_DISMISS_OBSTRUCTIONS` | inspect the window under the target and dismiss a popup/modal covering it before acting | `true` |
| `FURTI_MAX_OBSTRUCTION_DISMISSALS` | dismissals allowed per step (clearing is not a retry) | `2` |
| `FURTI_FAILSAFE` | keep pyautogui's corner abort; the controller still releases an accidental corner park and allows deliberate corner targets | `true` |
| `FURTI_KILL_HOTKEY` | abort hotkey (pynput syntax) | `<ctrl>+<alt>+k` |
| `FURTI_LLM_JSON_MODE` | Ask the endpoint for JSON-only output (DeepSeek `response_format`, Gemini `response_mime_type`); rejected hints are retried without | `true` |
| `FURTI_VERIFY_STEPS` | combined current-step and next-step review; dispatch confirmation is always logged | `true` |
| `FURTI_SCREENSHOT_INTERVAL` | capture throttle (seconds) | `4.0` |
| `FURTI_MAX_LLM_CALLS` | per-task LLM budget | `100` |
| `FURTI_MAX_PLAN_REPLANS` | adaptive unfinished-route replacements | `3` |
| `FURTI_MODEL_PRICES` | JSON price override `{"model":[in,out]}` | table above |
| `FURTI_STATUS_WINDOW` | `false` disables the Tk overlay | `true` |
| `FURTI_DIRECT_TOOLS` | `false` disables every direct tool, forcing mouse/keyboard routes | `true` |
| `FURTI_ALLOW_SHELL_COMMANDS` | `false` disables only `run_command` | `true` |
| `FURTI_ALLOW_DESTRUCTIVE_COMMANDS` | `true` lets irreversible commands run without `params.confirm` | `false` |
| `FURTI_TOOL_TIMEOUT` | `run_command` timeout (seconds) | `60` |
| `FURTI_TOOL_MAX_OUTPUT` | Characters of tool output surfaced per step | `4000` |
| `FURTI_SCREENSHOT_FORMAT` | Default image format for the screenshot tool | `png` |
| `FURTI_PROFILE` | Use the user/system context file | `true` |
| `FURTI_PROFILE_MAX_AGE_DAYS` | Re-detect the environment after this age | `7` |
| `FURTI_PROFILE_MAX_APPS` | Applications advertised in prompts | `40` |
| `FURTI_USER_NOTES` | Your own facts, `;`-separated | (none) |
| `FURTI_REFLEX` | Compile reflexes at all | `true` |
| `FURTI_REFLEX_MIN_CONFIDENCE` | Minimum anchor confidence worth caching | `0.75` |
| `FURTI_REFLEX_MIN_DESCRIPTION` | Minimum specificity of a reflex cache key | `8` |
| `FURTI_REFLEX_MAX_TYPED` | Longest typed payload that may become a reflex | `160` |
| `FURTI_REFLEX_RETIRE_FAILURES` | Misses before a reflex is retired | `3` |
| `FURTI_REFLEX_REALIGN` | LLM re-anchors stale reflexes | `true` |
| `FURTI_CROSS_VERIFY` | Audit critical steps on the other provider | `true` |
| `FURTI_CROSS_VERIFY_PROVIDER` | `auto` = the provider that is not the primary | `auto` |
| `FURTI_CROSS_VERIFY_MODEL` | Verifier model | secondary default |
| `FURTI_CROSS_VERIFY_MAX_CALLS` | Per-task verification ceiling | `12` |
| `FURTI_CROSS_VERIFY_PLANS` | Audit the plan before user approval | `true` |
| `FURTI_CROSS_VERIFY_APPLY_ALTERNATIVE` | Feed a rejection's safer alternative into the re-plan | `true` |
| `GEMINI_USE_FUNCTION_CALLING` | Request the `plan_action` tool from Gemini | `true` |

---

## 7. Failure modes & behaviour

| Situation | Behaviour |
| --- | --- |
| No API key | `build_task_agent()` raises immediately with a clear message |
| RapidOCR missing/failed | PaddleOCR fallback is initialised automatically; text grounding only turns off when no OCR backend is importable. The report logs the resolved backend and warns once when grounding is unavailable |
| tkinter missing (headless) | `python app.py --task ...` remains available; the GUI reports a clear startup error |
| pynput missing | hotkey unavailable; Ctrl+C / STOP button remain |
| Model returns malformed JSON | planner retries, then escalates to the smart model, then gives up with a logged error |
| Target not found on screen | step re-anchored/re-planned, loop-detected or retry-budgeted out |
| Named drag drop target not found | the drag is **not** dispatched; the step fails and re-plans rather than dropping at a stale point |
| Direct tool fails (`launch_app` target missing, `read_file` on a missing path, ...) | `ToolResult(ok=False, detail=...)` → failed `StepResult` with the tool's own reason → normal adaptive re-plan; no input is dispatched and no reflex is compiled |
| `run_command` names an irreversible operation | refused with a "potentially destructive command" reason unless `params.confirm: true` or `FURTI_ALLOW_DESTRUCTIVE_COMMANDS=true` |
| `run_command` exceeds `FURTI_TOOL_TIMEOUT` | the step fails with "command timed out after Ns"; the run continues/re-plans |
| `FURTI_DIRECT_TOOLS=false` | tool steps fail with a "disabled" reason instead of silently degrading to a mouse route |
| Reflex does not match the screen | counted as a miss, then re-aligned by the LLM (template + `expected_bbox` rewritten) and replayed once; retired after `reflex_retire_failures` misses |
| Reflex re-alignment answer is vague (no `found`, no confidence, off-frame bbox) | rejected; the stored template is left untouched and the planner is asked for a fresh route |
| A step has no usable anchor | first failure re-aligns the step (same action, corrected anchor); a second failure re-plans it; the stale reflex is counted/retired |
| Cross-verifier rejects a critical step | the step is **not dispatched**; the veto and its safer alternative become the failure note the planner re-plans from |
| No second provider key, verifier error, or verification budget exhausted | the step proceeds unverified and the journal records why (fail-open) |
| Context file missing or corrupt | detection runs again; any section that fails is simply absent, and `FURTI_PROFILE=false` turns the feature off |
| Gemini without function calling / rejecting tools | retried in JSON response mode, then without tools, then text-only for images |
| Unknown key name in a chord | `ChordError` propagates out of the input controller, so the step fails and re-plans instead of silently pressing nothing |
| Kill hotkey pressed | `TaskAborted` propagates; run stops cleanly and the report records the abort |
