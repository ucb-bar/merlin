---
title: Experiment console — track, inspect and debug any experiment from the terminal
kind: guide
status: current
owner: experiments
last_verified: 2026-10-26
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
  - packages/merlin-experiments/src/merlin_experiments/tracking/console/sources/lifecycle.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/console/sources/activity.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/console/sources/insight_common.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/console/sources/performance.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/console/sources/cases.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/console/sources/case_history.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/feedback/perf_history.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/console/sources/tools.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/console/sources/grader.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/console/sources/integrity.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/console/sources/attempts.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/console/sources/certification.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/console/sources/focus.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/console/sources/cost.py
  - packages/merlin-experiments/src/merlin_experiments/tracking/console/sources/library.py
  - packages/merlin-experiments/console/internal/ui/insights.go
  - packages/merlin-experiments/console/internal/ui/story.go
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
- **`e`: run list.** Shows or hides the experiment list on the left (shown automatically only on very
  wide terminals; `b` is the usual way to switch).

A stored or finished experiment is a replay: the header says `REPLAY · records until <time>` and
nothing animates. A live one pulses ◉.

## The screen

```
 ◉ LIVE  <run>  task <task>  elapsed 2h10m  last record 3s ago                 OPERATOR  12:00:00 UTC
 basis: <the record that decided the state>          ◉ 1 running: … [L]   console 0.9% cpu on 0-7
 NOW author starting a turn · library_overlay for 147s  │  cores (20) test/verify 12 │  grading 24/232 4✔ 19✖
┌PROGRESS──────────────────────────┐┌BOTTLENECKS─────────────────────────────────────────────┐
│ grades in flight, what the agent ││ critical path: setup 48% · findings worst first,       │
│ sees, certificates, cohorts,     ││ each with its evidence and → what to look at next      │
│ passes per grade, numerics       ││                                                        │
├TIMELINE──────────────────────────┴───────────────┐┌AGENT TREE──────────────────────────────┐
│ author turns  ▒startup █agent ✕stopped            ││ session state; per agent: state, last  │
│ agents working · build · test/verify · L2 · L3 ·  ││ record, time worked, tokens, failures  │
│ sim queue · then each agent's own lane            ││                                        │
├CORES─────────────────────────────────┐┌TOKENS────┴──┐┌PERFORMANCE────────────────────────┐
│ one row per CPU of the run, coloured  ││ cumulative  ││ model vs gSIM cycles, floor,      │
│ by activity, height = busy            ││ chart       ││ roofline                          │
├CAPSULES──────────────────┐┌PASSES──────┴──────┐┌EVENTS┴───────────────────────────────────┤
│ op, shapes, origin, tiers ││ applied, in order ││ AUTHOR / AGENT / CHECK / SIM / GRADE      │
└───────────────────────────┘└───────────────────┘└──────────────────────────────────────────┘
 ⇥ focus  ↵ inspect  z zoom  b browse runs  L running  e run list  / filter  s sort  t window  ? help  q quit
```

The **NOW** line answers "what is happening right now": what the author is doing (authoring,
starting a turn and in which stage, or stopped and why), what the run's cores are busy with, and the
latest grade.

| Panel | What it shows |
|---|---|
| **PROGRESS** | The background grades in flight: each grades an immutable snapshot of the agent's compiler, and several can run at once. Each line shows a bar of passed / failed / errored capsules against the grade's selection, the counts, how long the grade has run and when it last reported a result (amber after 20 minutes without one). Without a grade in flight, the last grade that graded something. *agent sees*: the copy the grader last published to the agent (`qa/progress.json`), with the tiers it covers, its age, and how far it trails the grades (amber when more than 10 minutes). *certified*: capsules with an L2 or L3 certificate in the tier ledger and the joint L3 obligation. Then each cohort's capsules as pass / fail / gated (held at a grading gate) / running / not yet, the passes per grade (grades that stopped before grading any capsule are counted, not plotted; round-end grades of a different capsule subset are used only when the run has no background grade), numerical executions across all grades, and the compiler stages once one has moved past pending. |
| **BOTTLENECKS** | The critical path of the lead agent's time, then every finding, worst first (severity, then wall time at stake). Each finding has one line of evidence and an `→` line saying what to look at or change. See [Why is it slow?](#why-is-it-slow). |
| **TIMELINE** | The wall clock (the last 30 minutes of a live run, so it moves), one labelled row per lane, each with a **now** column on the right saying what that row is doing at this moment (`▶ 3 oldest 2m10s`, `idle 4m`, `stopped 33m ago`): **author turns** (each turn's startup ▒, the agent working █, ✕ where the host stopped it), **agents working** (how many at once), the grader's job lanes (**build**, **test/verify**, **L2 sim (Spike)**, **L3 sim (gSIM)**, **sim queue wait**, **calibration**, **grader setup**; failures ▓), then each agent's own lane (▆ thinking, █ tool call, ░ waiting). ▼ marks rounds and stops. The top line splits the lead agent's time. |
| **AGENT TREE** | Agents grouped by round, newest first. Each round has one line saying how it went (turns, agent time, how and when it ended; a round whose turn was refused says `no agent ran`). The newest round with agents is expanded: lead and helpers with state, the age of their last record, time worked, tokens and failed commands. Older rounds collapse to one line (`z` expands them). An agent whose process was stopped reads `stopped`, never `working`. |
| **CORES** | A Gantt of the run's own CPUs: one row per core (two per row when the panel is short; `z` for one each), one cell per time slice, coloured by what the core worked on and as tall as it was busy. The legend gives each activity's share of the busy time; the line also shows the machine's load, the run's read rate and the console's own cost. See [What is each core doing?](#what-is-each-core-doing). |
| **TOKENS** | Cumulative fresh input, cache reads, output and reasoning tokens; cache-hit ratio; rates; the token budget and the provider rate limit. |
| **PERFORMANCE** | Per capsule: Spike-model estimate, gSIM (RTL simulation) cycles, structural floor, efficiency, model-vs-gSIM gap and the busiest modeled resource; a roofline when the panel is tall. Operator mode adds golden-reference comparisons. |
| **CAPSULES** | Every capsule with cohort, operation, shapes, origin (model, program, ATen op), tiers L0–L3, cycles and failure category. |
| **PASSES** | Only what the compiler applies: its pass pipeline in order (with the shared-pass boundary a pass implements), plus shared-pass families and controls recorded as applied or enabled; how many capsules' results used each pass, cache-invalidation churn, and profiled time when a profile exists. Declared-but-unused passes are left out (Enter shows the full matrix). |
| **EVENTS** | Milestones, newest first, tagged by part of the run: **AUTHOR** (turn admitted, startup finished and how long, turn ended and why), **AGENT** (helper started or finished, failed commands grouped per agent and minute; messages in operator mode), **CHECK** (self-checks), **SIM** (simulation results), **GRADE** (pass count changes), **ACTION** and **STOP**. |

A missing value shows as `—`, never as `0`. Model estimates, simulation cycles and structural bounds
keep their labels and are never mixed. CPU load is telemetry, not evidence.

### The insight page (`v`)

`v` switches to a second page of ten panels that answer "is the run getting better, and what holds it
back?" (`v` again returns; each page keeps its own focus; `--page insights` opens it first). Every
panel has summary lines (▲ to look at, ✖ bad, ✓ fine), then a table; `Enter` on a line inspects its
record, `z` zooms, `/` filters, and `s` cycles the table orders a panel offers (CASES does; the
title shows the current one). At 150 columns and more the page is four rows, CASES alone in the
second; narrower, five rows (CASES again alone in the second). Zoom CASES (`z`) to see every column
and more rows.

| Panel | What it shows | Read from |
|---|---|---|
| **PERFORMANCE** | gSIM cycles over time: geomean speedup of each re-measured test case (first ÷ latest) and against the previous attempt's latest, per family (dense, conv, epilogue, program, other) with median array utilization, the cases that moved in the last hour, the Spike-model ÷ gSIM gap (labelled *modeled*). Operator mode adds reference targets from `out/artifacts/perf-studies/*` (agent ÷ reference cycles; `study` marks the study's own agent measurement). | `.qa_channel/simresp_*.json` with `sim: gsim`; the previous attempt from the campaign manifest |
| **CASES** | One row per test case: cohort and family; status (L3 / L2 certified for the current bytes `✓`, re-proving after an invalidation `↻`, else the newest grade row's result and failure category); the newest gSIM cycles (*measured*) and timing-model estimate (*modeled*), and modeled ÷ measured for the newest pair of the same executable; from the case's current evidence the roofline bound and what bounds it (compute, read or write requests), gap = cycles ÷ bound (`*`: measured on an earlier executable than the current L2), headroom cycles, array utilization and the busiest modeled unit; per evidence kind the gain since the first passing observation (first ÷ best / first ÷ latest, above ×1 is faster) and when the best was seen. Summary: cases passing by cohort, which evidence the passing cases have, median and cycle-weighted utilization, median and worst gap, geomean speedup since first pass and how many cases improved (gSIM and model separately), and how much history is read. `s` sorts by biggest headroom, worst gap, lowest utilization, biggest gain, cohort, status or name; `/` filters on any cell (a cohort, a family, `fail`, `↻`). `Enter` shows the case's whole history: every change of its gSIM cycles and of its estimates with first / best / latest marked, the grades in which its result changed, and the same-executable calibration pair. Measured and modeled cycles never share a ratio. Names are operator-only (public mode shows `case NNN`). | `qa/perf_history.jsonl` when the grader keeps it; otherwise derived from `.qa_channel/simresp_*.json` and `resp_*.json` (author and idle-timing self-checks; variants excluded) and `qa_history/verdict_fast_*.json`; status from `qa/tier_state.json` and `qa/latest_completed_grade.json`; definitions from the `roofline-and-utilization` skill's `perf_summary.py` |
| **AGENT TOOLS** | Commands, edits and helper spawns over 30 min, 60 min and the whole run that mention each tool: `isa_tools.py debug`, `evaluate_variants`, optimization suggestions, the ranked list / performance brief, cycle and roofline fields, A/B variants, gSIM and scalability self-checks, the scheduling library, `iteration_notes.md`; helpers working against the ceiling and the lead's think/tool/wait split. A count says a command named the tool, not that it helped. | the agents' own streams |
| **GRADER** | Verdicts finished and delivered, the share superseded before delivery, the first verdict after the agent started, grade time p50/p95 (verdict time − owner tick), grades running and cases left, admission closures and deferrals by reason, gated test cases in the newest verdict, operator idle-timing requests, the most snapshot owners alive at once. | `qa_history/verdict_fast_*.json`, `_qa_work/background_snapshot_owners/`, `.qa_channel` |
| **INTEGRITY** | Campaign-log signals (snapshot marker mismatches, grade errors, skipped promotions, unavailable snapshots, broker and supervisor lines, dropped writes, a full disk, tracebacks), the self-check broker's health, grade health, whether the content store (`MERLIN_BUNDLE_CAS`) is private to the run, the runner's children (defunct ones are flagged), disk and other runs' busy cores. Example log lines are operator-only. | the runner's standard output (or the attempt's launch log), `feedback_health.json`, `grade_health.json`, `/proc` |
| **ATTEMPTS** | Every attempt of the campaign: when it started, how long it ran, how it stopped (service exit), its author startup (summed per turn), its grades and first verdict; total wall time, the time lost to attempts that delivered no verdict, and whether `PREVIOUS_SESSION.md` was staged. Archived attempts are read once; `reading…` until a record fits the refresh budget. | the campaign manifest (`out/artifacts/agentic-report/campaign-figures/*/*.campaign.yaml`), launch and accounting logs |
| **CERTIFICATION** | Headline: the cases passing numerics over every test case, by cohort — the CASES panel's own count (the console refreshes CASES first). Then functional tier certification on one line: L2 and L3 certified for the current compiled bytes, L3 re-proving after an invalidation and the joint L3 physical obligation for the newest reported source, divided by the cohorts the loop tier-certifies (those with tier records or physical obligations); the rest are listed as decided by the final gate. Per cohort: the newest verdict's pass / fail / gated / carried (sampled) / not run, with the failure categories that keep the rest from passing. Names are operator-only (Enter on a cohort). | CASES, `qa/tier_state.json`, `qa/joint_physical_correctness.json`, the newest fast verdict, the capsule catalog |
| **AGENT FOCUS** | Edits of the last 30 min by directory (by file in operator mode), the families their paths name (a hint), the lead's latest message and command (operator mode), helpers working, open self-checks and waiting suggestions. | the agents' streams, AGENT VIEW |
| **LIBRARY** | One row per library pass the workspace offers (the release's scheduling and composition capabilities and the catalogue mechanisms available for the target) plus other library modules the author imports or mentions: whether `submission/` imports it (`ast`, `compiler_library/` excluded), how many author commands and edits mention it, evidence files that name it, and the cycles ratio of variant self-checks that add it (variant ÷ `submission/` on the same case and evidence kind; gSIM and model never mixed). Sent least-touched first; the summary names the passes no command or edit ever mentioned. File names and case names are operator-only. | `shared-library.json`, `OPTIMIZATION_SUGGESTIONS.json`, `submission/`, `variants/`, `evidence/`, `.qa_channel/req_*`/`resp_*`, `qa/perf_history.jsonl`, `qa/idle_tuning.json`, the agents' streams; definitions from the `scheduling-api` skill's `library_inventory.py` |
| **COST** | Tokens this attempt (and the uncached part), the rate over the last hour, the budget, new L3 certificates against the previous attempt's ledger and tokens per new certificate. No prices are modeled. | TOKENS, CERTIFICATION, the previous attempt's ledger |

## Keys

| Key | Action |
|---|---|
| `Tab` / `Shift-Tab`, `←` `→`, `1`–`9`, mouse click | focus a panel |
| `j` `k`, `↑` `↓`, `PgUp` `PgDn`, `g` `G`, mouse wheel | move within a panel |
| `Enter` / `i` | inspect the selected row: the full record behind it, with its file paths |
| `z` | zoom the focused panel to full screen (`Esc` or `z` to return) |
| `v` | switch between the main page and the [insight page](#the-insight-page-v) |
| `b` / `o` | browse all experiments (type to search, `Tab` for state, `Enter` to open) |
| `L` | jump to the running experiment (again: the next one) |
| `e` | show or hide the experiment sidebar |
| `,` `.` / `<` `>` | in TIMELINE: move the time cursor by 1 or 10 cells (`Esc` clears it) |
| `t` | time window for TIMELINE and CORES: 10 min, 30 min, 2 h, all (a live run starts at 30 min, so the chart moves; a stored one at all) |
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

Read the NOW line, then the first finding in BOTTLENECKS and the first line of TIMELINE:

```
lead agent's time ███████ waiting for setup 48% think 26% tool 2% await 0% idle 25% · agents ∥ 1.2 avg/7 peak → setup bound
```

The lead agent drives the run, so its time is the critical path:

- **think**: a turn is open with nothing outstanding. The model is reasoning or generating.
- **tool**: the agent waits on its own commands (builds, self-checks).
- **await**: it waits on helpers or background jobs.
- **setup**: before its first action, and while the host prepares a turn (the author startup: sandbox,
  snapshot and library overlays). The **author turns** row shows each startup as ▒.
- **idle**: no turn is open, for example after the host stopped the author.

A startup that takes minutes gets its own finding, with the slowest stages compared to the fastest
turn (`library_overlay 7m16s (20x its fastest)`), the run's read rate and the machine's load while it
runs. Enter on the **author turns** row lists every turn's stages.

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

### What is each core doing?

CORES colours every slice of every core of the run by its activity:

| Colour | Activity | Recognised by |
|---|---|---|
| pink | authoring | the agent process and every command it runs |
| amber | author startup | the runner while the host prepares a turn |
| purple | checks | the self-check, model-check, ISA-tools and CCA brokers and what they run |
| orange | build | candidate-compiler calls, compiler executables, and grader stages that compile, emit or link |
| indigo | test/verify | grader verification stages, model grades and capsule execution |
| sky | L2 sim | the Spike executable the run selected |
| green | L3 sim | the gSIM executable the run selected |
| teal | calibration | the calibration service and owner |
| grey | harness | leases, scheduling and runner threads with no running stage |
| stone | not this run | busy time on the core from processes outside the experiment |

A process is named by its own command line, or else by its nearest recognised parent, so a compiler
the agent runs is authoring and a Spike run under a self-check is L2. A grader thread is named by the
stage record it is running. Thread time is charged to the core the thread was last seen on (a point
sample). The history starts when the console began watching the run. `Enter` on a core lists what
ran on it, sample by sample, and its time per activity.

A run whose CPUs are mostly idle gets CPUS_UNDERUSED. One whose CPUs run other processes gets
CPUS_SHARED. A run whose cores are idle while jobs queue gets CPU_IDLE_WITH_BACKLOG, and one whose
cores are saturated with jobs waiting gets CPU_SATURATED. A reservation that nothing runs on gets
RESERVATION_IDLE, which means the CPUs are held back from everyone else.

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
| AUTHOR_STARTUP_SLOW | a turn's startup took 2 min or more (an error when the agent then worked under a minute, or while a startup has run 5+ min) | TIMELINE author turns (Enter), CORES |
| AUTHOR_STOPPED | the host stopped the author while the run goes on | AGENT TREE, EVENTS |
| TOOL_BOUND | ≥ 35% waiting on its own tool calls | TIMELINE tool spans; jobs stage p95 via Enter |
| WAIT_BOUND | ≥ 30% blocked on awaits | AGENT TREE (helpers), TIMELINE |
| LEAD_IDLE / SETUP_DOMINATED | no turn open ≥ 40% / setup and turn startups ≥ 30% of 15+ min | TIMELINE author turns, EVENTS |
| SIM_ON_CRITICAL_PATH | the lead waits while simulations run, ≥ 15% of the window | TIMELINE L2/L3 lanes |
| HELPERS_UNDERUSED / HELPERS_SATURATED | few helpers busy for the slots available / every slot busy | AGENT TREE, busy-agents lane |
| QUEUE_DOMINATED | simulation jobs wait longer than they run (medians) | TIMELINE queue lane |
| CPU_IDLE_WITH_BACKLOG / CPU_SATURATED | the run's CPUs idle with work queued / saturated with waiters | CORES |
| CPUS_UNDERUSED / CPUS_SHARED | under 15% of the run's CPU time busy / 20%+ spent in other processes, over 5+ min | CORES |
| SETUP_STAGE_SLOW | one job stage owns at least half of the p95 time | Enter on a job lane |
| UNFINISHED_JOBS | jobs never recorded an end in a stopped run | TIMELINE, EVENTS |
| TOKENS_WITHOUT_PROGRESS | 200k+ output tokens over an hour with no change in the passing count | TOKENS, PROGRESS |
| TOKEN_BUDGET / RATE_LIMIT_PRESSURE | budget ≥ 90% used / provider window ≥ 80% used | TOKENS |
| FAILURE_CLUSTER | five or more capsules fail with the same category | CAPSULES (filter by the category) |
| INVALIDATION_CHURN | a pass invalidates cached results 100+ times | PASSES |
| STAGE_BLOCKED | a compiler stage is blocked | STAGES |
| LOW_ROOFLINE_EFFICIENCY / MODEL_GAP / UNKNOWN_PERF | a capsule far below its structural floor / Spike model and gSIM differ by 25%+ / performance records failed their binding check | PERFORMANCE |
| RESERVATION_IDLE | CPUs are reserved but no process or run uses them | CORES |
| RECORD_STALE / AGENT_IDLE | the run is alive but quiet / no agent activity for 15 min (unless the author is known to be stopped or starting) | header, EVENTS |
| PERF_TOOLS_UNUSED | after 50+ agent commands, a performance tool the guidance names was never used (advisory) | AGENT TOOLS (`v`) |
| GRADES_UNDELIVERED | three or more verdicts finished and none was delivered: every snapshot was superseded first | GRADER (`v`) |
| RUN_INTEGRITY | the campaign log records marker mismatches, grade errors, dropped writes or a full disk | INTEGRITY (`v`) |
| READER_ERROR | a record could not be read | the panel's error line |

The critical-path verdict itself (for example "agent reasoning") is the BOTTLENECKS headline, not a
finding. Identical findings from several snapshots collapse into one with a count (`×13`). Enter on a
finding lists each distinct case.

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
- It reads each record once and then only when it changes, within a fixed read budget per refresh
  (128 MiB). The small authoritative records (tier ledger, the agent's verdict, progress and latest
  completed grade, the joint physical obligation and the newest fast verdict) are read first, and a
  quarter of the budget is held back for the insight panels, so bulk scans of large responses and
  verdict histories cannot starve them; a record that could not be re-read shows as last read.
  A one-shot `--dump` / `--once` snapshot has no budget. Finished experiments refresh less often.
- CASES keeps only a small digest of each response and verdict, cached by the file's identity, so each
  record is read once. It reads new records newest first, at most 48 MiB per refresh and only from the
  ordinary budget (never the insight reserve), so on a long run it says `N still to read` for the first
  few refreshes (about 200 MiB on a six-hour run). When the grader keeps `qa/perf_history.jsonl` it
  only tails that file.

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
- Derived CASES history times simulation responses by their file modification time (they record no
  time of their own), which a copy of the run does not keep. The ledger `qa/perf_history.jsonl`
  records its own times, but only from the grade that first wrote it: a run whose grader gained the
  ledger mid-run shows history from that grade on.
- The core map shows the machine now. For a finished run it shows the run's recorded process-tree
  history instead of live cores.
- The console reads only this machine's records. Results on other machines (for example FPGA runs)
  appear only through records copied back here.
