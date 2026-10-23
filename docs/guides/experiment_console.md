---
title: Experiment console — track, inspect and debug any experiment from the terminal
kind: guide
status: current
owner: experiments
last_verified: 2026-10-23
related: [experiment_dashboard]
code_refs:
  - packages/merlin-experiments/src/merlin_experiments/tracking/console/cli.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/console/discovery.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/console/serve.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/console/diagnose.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/console/placement.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/console/launch.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/console/plain.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/console/sources/__init__.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/console/sources/timeline.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/console/sources/machine.py
  - packages/merlin-experiments/console/internal/ui/model.go
  - packages/merlin-experiments/console/internal/ui/views.go
  - packages/merlin-experiments/src/merlin_experiments/cli.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/telemetry/startup.py
---

# Experiment console

The experiment console is a terminal dashboard for every experiment on the machine. You can see which
runs are going and which are finished, follow a live run, open an old one, and find out what is
slowing a run down or why a capsule fails. Progress, stages, agents, the timeline, tokens, CPU cores,
performance, capsules, passes and events are all on one screen, and Enter on any row opens the record
behind it.

The console only reads records. It never writes into a run, never starts or stops a job, and never
touches hardware, so it is safe to leave open next to a running experiment.

## Quick start

```bash
merlin-experiment console --build        # once per checkout: compile the terminal UI (needs Go, see below)
merlin-experiment console                # open it: the running experiment, or the most recent one
```

Then press `b` to browse every experiment, `L` to jump to the one that is running, `Enter` to inspect
a row, `?` for help and `q` to quit. Use a terminal of at least 200×50 to see every panel at full
size. Smaller terminals work too: panels shrink, and `z` zooms any one of them to full screen.

Other ways in:

```bash
merlin-experiment console --experiment <run-id>     # open a specific experiment
merlin-experiment console --operator-private        # also show capsule names, paths, agent messages, golden references
merlin-experiment console --task <task-dir>         # only experiments in these task folders (repeatable)
merlin-experiment console --run-dir <dir>           # add a run or record directory from anywhere (repeatable)
merlin-experiment console --list                    # list experiments and their states, then exit
merlin-experiment console --once --experiment <id>  # print the panels as text once (CI, ssh logs, pipes)
merlin-experiment console --once --json --experiment <id>   # the raw snapshot as JSON
merlin-experiment watch --task <task-dir>           # opens the console too, once it is built
```

## One-time setup: build the terminal UI

The screen is a small Go program (Bubble Tea). Build it once per checkout:

```bash
merlin-experiment console --build
```

This writes `out/build/console/merlin-console`. The build needs Go 1.26 or newer. If `go` is not on
your `PATH`, install it in your home directory and point the console at it:

```bash
mkdir -p ~/.local
curl -fsSL https://go.dev/dl/go1.26.10.linux-amd64.tar.gz | tar -C ~/.local -xz
export MERLIN_CONSOLE_GO=~/.local/go/bin/go
merlin-experiment console --build
```

To share one build between checkouts, set `MERLIN_CONSOLE_BIN=/path/to/merlin-console`. Without a
build, `merlin-experiment console` still works: it shows the same information as refreshing plain
text and tells you how to build the full UI.

## Finding an experiment

The console scans every task folder on the machine for experiments: managed launches and run
directories under `<task>/merlin/out/runs/`. It finds a new run as soon as its runner starts, before
the run has written any rounds. Every experiment gets exactly one state, and the console shows the
record that decided it, so you never have to guess whether a run is still alive:

| State | Meaning | Decided by |
|---|---|---|
| ◉ LIVE | running | its runner process is alive |
| ◐ STALLED | running, but quiet | alive, but no record changed for 15 minutes |
| ◎ ACTIVE | probably running | no runner found, but records changed in the last 5 minutes |
| ✖ INTERRUPTED | stopped early | a stop record, an interrupted round, or a runner that died without a completion record |
| ✔ FINISHED | completed | a campaign completion record (or a finished construction study) |
| ◌ ENDED | ended without a verdict | none of the above, or the run directory is gone |
| ▣ STORED | reference records | a directory you named with `--run-dir` |

A relaunch is always a new experiment, never merged into the old one.

- **`b` (or `o`): browse.** A full-screen list of every experiment with its state, task, start time,
  duration, last record and deciding record. Type to search by run, task, state or reason. `Tab`
  cycles all / live / finished / interrupted / stored, and `Enter` opens the selected experiment.
- **`L`: running experiment.** Jumps to the running experiment. Press it again to cycle when several
  are running. While you look at an older run, the header shows `◉ 1 running: <run> [L]` as a
  reminder.
- **`e`: sidebar.** Shows or hides the experiment list on the left (it is shown automatically on wide
  terminals).

A stored or finished experiment is a replay: the header says `REPLAY · records until <time>` and
nothing animates. A live one pulses ◉.

## The screen

```
 ◉ LIVE  <run>  task <task>  elapsed 2h10m  last record 3s ago                 OPERATOR  12:00:00 UTC
 basis: <the record that decided the state>          ◉ 1 running: … [L]   console 0.9% cpu on 0-7
┌EXPERIMENTS┐┌PROGRESS────────────┐┌STAGES──────────────────────┐┌BOTTLENECKS─────────────┐
│ by state  ││ grade, cohorts,    ││ compiler stages pipeline   ││ ▶ AGENT REASONING BOUND│
│           ││ passing over time  ││ and per-stage table        ││ findings, worst first  │
│           │├TIMELINE────────────┴────────────────────────────┐┌AGENT TREE──────────────┐
│           ││ lead: setup/think/tool/await/idle shares          ││ lead + helpers, tokens, │
│           ││ agent lanes, job lanes, concurrency, ▼ markers    ││ state, current command  │
│           │├TOKENS────────────┐┌CORES──────────┐┌PERFORMANCE──┴─────────────────────────┐
│           ││ cumulative chart  ││ every CPU     ││ model vs gSIM cycles, floor, roofline │
│           │├CAPSULES──────────┴──────┐┌PASSES──┴────┐┌EVENTS────────────────────────────┐
│           ││ op, shapes, origin, tiers││ declared/used││ newest first, failures marked    │
└───────────┘└─────────────────────────┘└─────────────┘└──────────────────────────────────┘
 ⇥ focus  ↵ inspect  z zoom  b browse runs  L running  / filter  s sort  t window  ? help  q quit
```

| Panel | What it shows |
|---|---|
| **PROGRESS** | The latest grade (checked and passed), numerical executions, coverage per cohort (all tiers vs L2 only), the passing count over time, and the completion or plateau record. |
| **STAGES** | The compiler-construction stages (frontend → global optimization → interface → lowering → ISA dialect → OOT integration → transformations → tuning, plus host runtime) with actions, gated capsules passing/required, wall time and tokens. |
| **BOTTLENECKS** | What bounds the run (headline), then every finding, worst first. See [Why is it slow?](#why-is-it-slow). |
| **TIMELINE** | The wall clock: each agent's turns ▆, tool calls █ and awaits ░; job lanes for setup, build, verify, calibration, L2, L3 and queue (failures ▓); busy-agent and running-simulation concurrency; ▼ markers for rounds and the stop. The top line splits the lead agent's time. |
| **AGENT TREE** | The lead and its helpers: state, role path, tokens, active time, failed commands and the current command. |
| **TOKENS** | Cumulative fresh input, cache reads, output and reasoning tokens; cache-hit ratio; rates; the token budget and the provider rate limit. |
| **CORES** | Every CPU of the machine, shaded by load and coloured by holder: **this experiment**, **another run**, **reserved** (no run on it) or **general** pool. Below: load per holder over time, per-core history for this run's cores, and the console's own cost. |
| **PERFORMANCE** | Per capsule: Spike-model estimate, gSIM (RTL simulation) cycles, structural floor, efficiency, model-vs-gSIM gap and the busiest modeled resource; a roofline when the panel is tall. Operator mode adds golden-reference comparisons. |
| **CAPSULES** | Every capsule with cohort, operation, shapes, origin (model, program, ATen op), tiers L0–L3, cycles and failure category. |
| **PASSES** | Library and catalog counts, the compiler's pass pipeline, and each pass as declared/used, with how many capsules need or exercise it, cache-invalidation churn and profiled time. |
| **EVENTS** | One feed of agent messages, failed commands, self-checks, simulations, construction actions, round ends and stops. |

A missing value shows as `—`, never as `0`. Model estimates, simulation cycles and structural bounds
keep their labels and are never mixed. CPU load is telemetry, not evidence.

## Keys

| Key | Action |
|---|---|
| `Tab` / `Shift-Tab`, `←` `→`, `1`–`9`, mouse click | focus a panel |
| `j` `k`, `↑` `↓`, `PgUp` `PgDn`, `g` `G`, mouse wheel | move within a panel |
| `Enter` / `i` | inspect the selected row: the full record behind it, with its file paths |
| `z` | zoom the focused panel to full screen (`Esc` or `z` to return) |
| `b` / `o` | browse all experiments (type to search, `Tab` for state, `Enter` to open) |
| `L` | jump to the running experiment (again: the next one) |
| `e` | show or hide the experiment sidebar |
| `,` `.` / `<` `>` | in TIMELINE: move the time cursor by 1 or 10 cells (`Esc` clears it) |
| `t` | timeline window: 15 min, 1 h, 6 h, all |
| `/` | filter the rows of the focused panel |
| `s` | change the sort of the focused panel |
| `y` | copy the inspected record's path (or the run directory) to the clipboard |
| `p` / `Space` | freeze the display (collection continues) |
| `r` | refresh now |
| `?` | help |
| `q` / `Ctrl-C` | quit |

Clipboard copy uses the terminal's OSC 52 support. To select text with the mouse instead, hold
`Shift` (most terminals).

## Common tasks

### Is my run still alive?

Open the console and press `L`. The header shows the state and, underneath, the record that decided
it, for example `runner pid 3524984 runs this --run-id`. STALLED means the process is alive but
nothing has been written for 15 minutes. That is also the first finding in BOTTLENECKS.

### Why is it slow?

Look at the BOTTLENECKS headline and the first line of TIMELINE:

```
lead ███████ setup 2% think 59% tool 7% await 5% idle 28% · recorded think 91% · agents ∥ 0.9 avg/4 peak → agent reasoning bound
```

The lead agent drives the run, so its time is the critical path:

- **think**: a turn is open with nothing outstanding. The model is reasoning or generating.
- **tool**: the agent waits on its own commands (builds, self-checks).
- **await**: it waits on helpers or background jobs.
- **setup**: time before its first action.
- **idle**: no turn is open, for example between sessions.

When the run recorded its own measurement, that one is shown as `recorded` and decides the verdict.

Then move the time cursor (`,` `.` with TIMELINE focused) to a slow stretch. The bottom line says
which agents had a tool call or await open, how many jobs of each kind were running, and the nearest
event. `Enter` on a job lane shows its stage durations (p50/p95), and on an agent its recent
commands.

### Why does a capsule fail?

Focus CAPSULES and type `/` and part of the name, or press `s` to sort by status. `Enter` shows the
capsule's inputs and shapes, its origin, the result of each tier with timings, the failure plane and
category, the invocation stages with return codes, and the paths of every record. Press `y` to copy a
path. FAILURE_CLUSTER in BOTTLENECKS tells you when many capsules fail the same way.

### Is it using its CPUs well?

CORES shows the experiment's cores in blue next to everything else on the machine. A run whose cores
are idle while jobs queue gets CPU_IDLE_WITH_BACKLOG. A run whose cores are saturated with jobs
waiting gets CPU_SATURATED. A reservation that nothing runs on gets RESERVATION_IDLE, which means the
CPUs are held back from everyone else. Zoom (`z`) to see every core's history.

### Compare with an earlier run

Press `b`, type the run name, then `Enter`. Switching experiments resets the per-experiment charts, so
the two runs never mix. Press `L` to come back to the live one.

### Share what you see

`merlin-experiment console --once --experiment <id>` prints the panels as text, ready to paste into
a ticket or chat. `--json` gives the full snapshot for scripts. Without `--operator-private`, names,
paths and messages are left out, so the output is safe to share widely.

## Bottleneck findings

Every finding compares numbers the records already contain. If a number was not recorded, the rule
does not fire. No finding passes or fails anything.

| Finding | Fires when | Look at |
|---|---|---|
| AGENT_REASONING_BOUND | the lead spends ≥ 60% of its time thinking/generating | TOKENS (rates), AGENT TREE |
| TOOL_BOUND | ≥ 35% waiting on its own tool calls | TIMELINE tool spans; jobs stage p95 via Enter |
| WAIT_BOUND | ≥ 30% blocked on awaits | AGENT TREE (helpers), TIMELINE |
| LEAD_IDLE / SETUP_DOMINATED | no turn open ≥ 40% / setup ≥ 40% of 30+ min | TIMELINE markers, EVENTS |
| SIM_ON_CRITICAL_PATH | the lead waits while simulations run, ≥ 15% of the window | TIMELINE L2/L3 lanes |
| HELPERS_UNDERUSED / HELPERS_SATURATED | few helpers busy for the slots available / every slot busy | AGENT TREE, busy-agents lane |
| QUEUE_DOMINATED | simulation jobs wait longer than they run (medians) | TIMELINE queue lane |
| CPU_IDLE_WITH_BACKLOG / CPU_SATURATED | the run's CPUs idle with work queued / saturated with waiters | CORES |
| SETUP_STAGE_SLOW | one job stage owns at least half of the p95 time | Enter on a job lane |
| UNFINISHED_JOBS | jobs never recorded an end in a stopped run | TIMELINE, EVENTS |
| TOKENS_WITHOUT_PROGRESS | 200k+ output tokens over an hour with no change in the passing count | TOKENS, PROGRESS |
| TOKEN_BUDGET / RATE_LIMIT_PRESSURE | budget ≥ 90% used / provider window ≥ 80% used | TOKENS |
| FAILURE_CLUSTER | five or more capsules fail with the same category | CAPSULES (filter by the category) |
| INVALIDATION_CHURN | a pass invalidates cached results 100+ times | PASSES |
| STAGE_BLOCKED | a compiler stage is blocked | STAGES |
| LOW_ROOFLINE_EFFICIENCY / MODEL_GAP / UNKNOWN_PERF | a capsule far below its structural floor / Spike model and gSIM differ by 25%+ / performance records failed their binding check | PERFORMANCE |
| RESERVATION_IDLE | CPUs are reserved but no process or run uses them | CORES |
| RECORD_STALE / AGENT_IDLE | the run is alive but quiet / no agent activity for 15 min | header, EVENTS |
| READER_ERROR | a record could not be read | the panel's error line |

Identical findings from several snapshots collapse into one with a count (`×13`). Enter on a finding
lists each distinct case.

## Public and operator mode

By default the console is public: counts, states, timings and findings only. `--operator-private`
adds capsule names, file paths, agent commands and messages, and golden-reference comparisons. The
header always names the mode. Golden comparisons are refused, with the reason shown, when the program
or device differs.

## Running it next to an experiment

The console is built to stay out of the way of the experiment it watches:

- It finds which CPUs each running experiment uses (from the process affinity and the cgroup CPU
  reservations on the machine; nothing is configured) and moves itself to the other CPUs, at low
  priority. When every CPU is taken, it only runs when a CPU would otherwise be idle.
- It measures its own CPU use and slows its refresh if it goes over 15% of one core.
- It reads each record once and then only when it changes, within a fixed read budget per refresh.
  Finished experiments refresh less often.

On a live 20-CPU run the console used about 4.5% of one core in total, all outside the run's CPUs.
The header and CORES panel show its current cost.

## Troubleshooting

For a Codex run that stops before its first event or during continuation startup,
inspect `<run-dir>/rounds/author_startup.jsonl`. It records `start`, `finish` and
`error` events for config verification, resume command construction, runtime
binding, sandbox construction and provider process creation. Nested sandbox stages
identify runtime verification, base mounts, frozen-input and library overlays,
and final answer masks. Each event names its round, turn and stage ID, with UTC
and monotonic timestamps; completed events also include elapsed wall seconds.
Nested intervals overlap and must not be summed. A start without a matching finish
or error locates unfinished work; it does not prove a lock or other root cause.
The stream contains no argv, credentials or exception messages. It is private
operator diagnostics, does not change checks, and is not yet a dedicated UI panel.

| Problem | Fix |
|---|---|
| `merlin-console is not built; showing plain text` | `merlin-experiment console --build` (see [setup](#one-time-setup-build-the-terminal-ui)) |
| `building the console needs Go >= 1.26` | install Go as above and set `MERLIN_CONSOLE_GO` |
| No experiments found | point it at the folder holding your task folders: `--tasks-root <dir>` (or `MERLIN_CONSOLE_TASKS_ROOT`), or name the run with `--run-dir` |
| A running experiment shows as ACTIVE, not LIVE | its runner process was not found (another user, or a different launcher); the state is from record activity |
| A panel says `reader error` | the record exists but could not be read; the message names it, and the rest of the console keeps working |
| Panels look cramped | enlarge the terminal (200×50 shows everything) or zoom with `z` |
| Garbled colours over ssh | use a terminal with 256 colours, or `--once --no-color` for text |
| `y` does not copy | your terminal does not support OSC 52 clipboard; use Shift-drag to select |

## Limits

- Stages come from a stage ledger when the run writes one. Until then they are derived from existing
  records, and a stage with no evidence shows as unknown rather than guessed.
- Resource names in PERFORMANCE are what the records name (some only record an index).
- The core map shows the machine now. For a finished run it shows the run's recorded process-tree
  history instead of live cores.
- The console reads only this machine's records. Results on other machines (for example FPGA runs)
  appear only through records copied back here.
