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
   **ordered steps**, shows the plan to the user, waits for **console
   confirmation**, executes step by step with live grounding, and writes a
   `<task_name>.md` report with token usage and approximate cost.

```
TaskAgent
├── TaskPlanner      plan in steps (fast model, escalates to smart)
├── VisualContextManager  rate-limited capture + OCR/icon grounding + visual gate
├── PlanExecutor     execute steps, re-anchor targets, retry, re-plan
├── MemoryManager    compile successful steps into reusable reflexes
├── UsageTracker     token counting + approximate USD cost
├── TaskJournal      console + status-window + <task_name>.md transparency
├── StatusWindow     always-on-top Tk overlay (capture / AI output / action signals)
└── KillSwitch       global hotkey (<ctrl>+<alt>+k) sets the shared stop event
```

### Execution flow

1. **Plan** — `TaskPlanner.plan(instruction)`:
   - observes the screen once (throttled capture),
   - asks the fast model for a JSON plan (`goal`, `reasoning`, `steps[]`),
   - escalates to the smart model after two failed attempts,
   - keeps the planning-time screenshot on the plan (`plan.frame`) so
     bounding-box anchors can be cropped into templates later,
   - enforces `max_plan_steps` and the per-task LLM-call budget.
2. **Confirm** — the full plan is printed and the agent blocks on
   `input()`: `y` executes, `n` aborts, `e` lets the user edit the
   instruction and re-plan. **Nothing is executed before confirmation.**
3. **Execute** — `PlanExecutor.execute(plan)` runs each step:
   - fresh screen observation (1-second absolute capture floor),
   - target resolution in priority order: `icon:<name>` template match →
     OCR text match → planned bbox cropped from `plan.frame` and
     template-matched on the live frame → focused-window fallback for
     `type`/`scroll`/`key_press`,
   - action via `PyAutoGuiInput` (smooth visible cursor movement by
     default; Windows typing uses a small per-character interval),
   - immediate `CONFIRM` logging after the input controller returns without
     an error,
   - optional LLM verification of the visible step result
     (`FURTI_VERIFY_STEPS=true`) without adding an unconditional one-second
     sleep,
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
dispatched. The status window additionally shows the last screenshot time and
age, the latest model output, and a color-highlighted current-action signal.

---

## 2. Feature / function registry

| # | Feature | Where | Status |
| --- | --- | --- | --- |
| 1 | Multi-step task planning (JSON plan with goal/reasoning/thoughts) | `planner.py` — `TaskPlanner`, `TaskPlan`, `PlanStep` | ✅ |
| 2 | Reuse of stored reflexes (planning hints + executor icon anchors) | `memory.py`, `ocr.py` — `IconMatcher` | ✅ |
| 3 | Compile successful steps into new reflexes | `executor.py` — `PlanExecutor._compile_reflex` | ✅ |
| 4 | Dynamic re-anchoring of planned bboxes on the live screen | `executor.py._resolve_target` + `vision.py.locate_on` | ✅ |
| 4a | Screenshot/model/DPI coordinate normalization | `planner.py`, `screen.py`, `vision.py` | ✅ |
| 5 | Console log of every thought/step/action (`[HH:MM:SS] [KIND]`) | `tasklog.py` — `TaskJournal` | ✅ |
| 6 | Cursor moves smoothly instead of teleporting (default) | `controller.py` — `PyAutoGuiInput._goto`; `cursor_teleport` flag | ✅ |
| 6a | Windows-safe typing pacing and responsive input timing | `controller.py` — `typing_interval`, `input_pause` | ✅ |
| 7 | `<task_name>.md` report after every task | `tasklog.py` — `TaskJournal.write_report` | ✅ |
| 8 | Anti-endless-loop guardrails | `config.py` guardrails + `PlanStep.signature` + `BudgetExceeded` | ✅ |
| 8a | Adaptive replacement route after a failed step | `executor.py._adaptive_replan` + `TaskPlanner.replan_remaining` | ✅ |
| 9 | Screenshot throttle (never every second) | `context.py` — `screenshot_min_interval` + 1 s absolute floor | ✅ |
| 10 | PaddleOCR text grounding before the LLM | `ocr.py` — `TextDetector` (2.x and 3.x APIs; oneDNN opt-in) | ✅ |
| 11 | Icon recognition from saved templates | `ocr.py` — `IconMatcher` (multi-scale matchTemplate) | ✅ |
| 12 | LLM decides whether a raw screenshot is needed (visual gate) | `context.py` — `GATE_SYSTEM_PROMPT`, cached per instruction | ✅ |
| 13 | Always-on-top Tk status window (task/phase/step/log/tokens) | `status.py` — `StatusWindow` | ✅ |
| 13a | Screenshot age, latest AI output, and current-action visual signals | `status.py` + `TaskJournal` | ✅ |
| 14 | Global kill hotkey (`<ctrl>+<alt>+k`) + window STOP button | `status.py` — `KillSwitch` (pynput) | ✅ |
| 15 | Approximate cost tracking (tokens × model price) | `cost.py` — `UsageTracker`, `MODEL_PRICES` | ✅ |
| 16 | Fast → smart model escalation | `planner.py` — `_pick_model`, `replan_step`; `fast_model`/`smart_model` | ✅ |
| 16a | Explicit DeepSeek image-capable provider selection | `config.py` / `orchestrator.py` — `FURTI_LLM_PROVIDER`, `deepseek-flash` | ✅ |
| 17 | Full transparency (console + Tk mirror of every log line) | `tasklog.py` + `status.py` | ✅ |
| 18 | Console confirmation before executing the plan | `agent.py` — `_confirm_plan` (y / n / edit) | ✅ |
| 18a | Per-action dispatch confirmation and optional visible-effect verification | `executor.py` — `CONFIRM` journal events, `verify_steps` | ✅ |
| 19 | Graceful degradation (no OCR, no tkinter, no pynput) | `ocr.py`, `status.py`, `context.py` fallbacks | ✅ |
| 20 | Legacy reflex demo unchanged | `__main__.py` (no args), `AgentOrchestrator` | ✅ |

---

## 3. Module map

| Module | Contents |
| --- | --- |
| `config.py` | `Settings`: guardrails, cursor, models/provider, OCR, window, hotkey, paths (env-overridable) |
| `planner.py` | `TaskPlanner`, `PlanStep`, `TaskPlan`, `BudgetExceeded`, `PLAN_SYSTEM_PROMPT` |
| `executor.py` | `PlanExecutor`, `StepResult`, `ExecutionReport`, step verification |
| `context.py` | `VisualContextManager`, `SceneObservation`, `TaskAborted`, visual gate |
| `ocr.py` | `TextDetector` (PaddleOCR), `IconMatcher` (cv2 matchTemplate), `describe_scene` |
| `tasklog.py` | `TaskJournal` (console + status sink + report writer) |
| `status.py` | `StatusWindow` (Tk, always-on-top, capture/AI/action signals), `KillSwitch` (pynput) |
| `cost.py` | `UsageTracker`, `CostSummary`, `MODEL_PRICES`, `make_usage_callback` |
| `agent.py` | `TaskAgent` — the end-to-end runner (plan → confirm → execute → report) |
| `brain.py` | `DeepSeekClient`, `GeminiClient` (chat_text/chat_vision + usage callback), `BrainPlanner` |
| `controller.py` | `InputController` protocol, `PyAutoGuiInput` (smooth `_goto`) |
| `vision.py` | `VisionReflex` (+ `locate_on` for live re-anchoring) |
| `memory.py` | `MemoryManager` (reflex cache, `save_skill`, `normalize_name`) |
| `orchestrator.py` | `AgentOrchestrator`/`build_agent` (legacy), `build_task_agent` (wiring) |

---

## 4. Guardrails (why the agent cannot loop forever)

| Guardrail | Default | Effect |
| --- | --- | --- |
| `max_plan_steps` | 10 | Plan is truncated; a task can never grow unbounded |
| `max_step_retries` | 2 | Re-anchor + re-plan attempts per step |
| `max_plan_replans` | 3 | Full replacements of the unfinished route per task |
| `max_llm_calls_per_task` | 30 | Hard LLM budget; `BudgetExceeded` stops planning/re-planning |
| `max_consecutive_failures` | 3 | Aborts the whole run after N failed steps in a row |
| `screenshot_min_interval` | 4.0 s | Normal captures throttled |
| absolute capture floor | 1.0 s | Even `force_fresh` captures cannot fire faster |
| `FURTI_OCR_MAX_DIM` | 960 | Downscales OCR input for responsive desktop grounding, then restores boxes to capture pixels |
| `FURTI_OCR_MKLDNN` | false | Opts into PaddlePaddle oneDNN CPU inference; off by default for desktop compatibility |
| `PlanStep.signature()` | — | sha1(description\|action\|target); a re-planned step that repeats a signature is aborted immediately |
| stop event | hotkey/STOP | Checked before every capture, LLM call and action |

No fixed sleep is inserted between successful steps. The executor proceeds
immediately; the only natural pauses are cursor movement, OS input delivery,
screen-capture throttling (which reuses the cached frame instead of sleeping),
and actual model/network response time.

### Coordinate spaces

There are three coordinate spaces that must not be mixed:

1. **Capture pixels** — the full BGR frame returned by `PyAutoGuiScreen`.
   PaddleOCR boxes, icon matches, and live template matches are reported here.
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
| 1 verify call per step × 6 steps | 6 × 400 = 2400 | 6 × 30 = 180 |

**Total ≈ 3,600 prompt + 600 completion tokens.**

- Prompt cost: 3,600 × $0.30 / 1,000,000 = **$0.0011**
- Completion cost: 600 × $2.50 / 1,000,000 = **$0.0015**
- **≈ $0.0026 per task** (~0.26 cents).

The dominant variable is **screenshots**: each downscaled 1280-px PNG adds
≈ 1,100–1,300 image tokens. The visual gate keeps these out of most calls;
the 1-second capture floor keeps them out of capture loops. If the plan call
attaches one screenshot (~1,200 tokens) the task cost rises to ≈ **$0.003**.

The console and the report both print the real, measured usage:

```
[COST] LLM calls: 8
[COST] Tokens: 3600 prompt + 600 completion = 4200
[COST] Approximate total API cost: $0.0026 USD
```

---

## 6. Usage

```powershell
# optional but recommended: OCR grounding
pip install -r requirements-ocr.txt

# Google Gemini (preferred)
$env:GOOGLE_API_KEY = "..."
# optional model tiers
$env:FURTI_FAST_MODEL  = "gemini-3.6-flash"
$env:FURTI_SMART_MODEL = "gemini-3.6-pro"

python -m furti_ai --task "Open Notepad, write 'hello', and save the file"
```

The agent prints every thought/action, shows the proposed plan, and waits
for `y` before touching the mouse. Press `<ctrl>+<alt>+k` (or the STOP
button) to abort safely at the next step boundary.

Reports land in `~/.furti_ai/reports/<task_name>.md`; reflexes and templates
in `~/.furti_ai/templates/` + `~/.furti_ai/memory.json`.

### Key environment variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `GOOGLE_API_KEY` | Gemini API key (falls back to DeepSeek if unset) | — |
| `FURTI_FAST_MODEL` / `FURTI_SMART_MODEL` | tiered models | provider default |
| `FURTI_LLM_PROVIDER` | `auto`, `deepseek`, or `gemini`; auto prefers DeepSeek when its key is present | `auto` |
| `FURTI_OCR_MAX_DIM` | Maximum OCR input dimension; boxes are scaled back to capture pixels | `960` |
| `DEEPSEEK_MODEL` | OpenAI-compatible DeepSeek model; `chat_vision` sends `image_url` | `deepseek-flash` |
| `FURTI_CURSOR_TELEPORT` | `true` restores instant cursor jumps | `false` (smooth move) |
| `FURTI_CURSOR_MOVE_DURATION` | maximum smooth cursor travel duration (seconds) | `0.25` |
| `FURTI_INPUT_PAUSE` | pause after each low-level PyAutoGUI call (seconds) | `0.03` |
| `FURTI_TYPING_INTERVAL` | delay between typed characters (seconds) | `0.02` |
| `FURTI_KILL_HOTKEY` | abort hotkey (pynput syntax) | `<ctrl>+<alt>+k` |
| `FURTI_VERIFY_STEPS` | optional LLM visible-effect check; dispatch confirmation is always logged | `false` |
| `FURTI_SCREENSHOT_INTERVAL` | capture throttle (seconds) | `4.0` |
| `FURTI_MAX_LLM_CALLS` | per-task LLM budget | `30` |
| `FURTI_MAX_PLAN_REPLANS` | adaptive unfinished-route replacements | `3` |
| `FURTI_MODEL_PRICES` | JSON price override `{"model":[in,out]}` | table above |
| `FURTI_STATUS_WINDOW` | `false` disables the Tk overlay | `true` |

---

## 7. Failure modes & behaviour

| Situation | Behaviour |
| --- | --- |
| No API key | `build_task_agent()` raises immediately with a clear message |
| PaddleOCR missing/failed | text grounding off; icon matching + screenshot reasoning still work |
| tkinter missing (headless) | console-only transparency; execution runs inline |
| pynput missing | hotkey unavailable; Ctrl+C / STOP button remain |
| Model returns malformed JSON | planner retries, then escalates to the smart model, then gives up with a logged error |
| Target not found on screen | step re-anchored/re-planned, loop-detected or retry-budgeted out |
| Kill hotkey pressed | `TaskAborted` propagates; run stops cleanly and the report records the abort |
