# src/merlin/development/backend/optimization/whole_model_measured

This package owns the optional, installed whole-model measured experiment mode.
It prepares and resumes runs, builds exact candidate snapshots, schedules bounded
measurement jobs, and records evidence and agent feedback. Shared compiler,
runtime, and measurement primitives remain in `src/merlin/`.

Keep target machines, builders, input capsules, and instruction policies explicit
data selected by the experiment. Do not discover a source checkout or infer a
target from a simulator name. Candidate-facing feedback must not expose private
answers or host-only input snapshots.

Preserve the distinction between a functional-model screen, board measurements,
and certifying native evidence. A process exit, cached result, partial output,
or candidate-provided claim is not a whole-model verdict. Resume must verify
the frozen inputs and exact candidate/build/engine identities before reuse;
infrastructure failures must not be scored as candidate failures.

## Who owns what

Change a behaviour in the module that owns it; a second implementation elsewhere is how two owners
start disagreeing about one state.

- Identity and records
  - `identity`: package and program digests, the builder's static module closure, the store key
    (builder identity + machine + build options) and the file primitives every owner shares. Never
    hash `sys.modules`; never key a store on a spec string alone.
  - `jobs`: the job record vocabulary (states, roles, result shape) every owner reads and writes.
  - `attempts`: a job's measurements across all its attempts and which one stands; a retry moves the
    earlier attempt aside whole, never deletes it.
- Machines
  - `machines`: device identity by content from `src/merlin/resources/contracts/hardware_pins.yaml` before any number,
    and the queue invocation. A queue command drops `HOME`/`USER`/`LOGNAME`; a started board job is
    never cancelled.
  - `registry`: how to run a program on each machine a target declares; the file is the target's data,
    owned by its example.
  - `capabilities`: what each machine can do, from its own facts, and what the chosen one lacks.
  - `noise`: one machine's noise margin and drift tolerance, from its own repeated solo readings.
- Measurement
  - `gates`: the refusals before any machine time (the capsule screen, the coverage floor, the
    whole-ELF instruction rule, which fails closed, and the functional gate) and the infra-fault
    classifier and circuit breaker.
  - `service` / `worker` / `batch`: asynchronous jobs keyed by exact package bytes, one detached job run
    to a result document, and one board batch judged by the reference control measured inside it (a
    batch whose control drifted re-measures its candidates alone before the drift refuses them).
  - `merlin.perf.whole_model_builder` and `merlin.perf.whole_model_verdict` (core, shared with the
    whole-model build and the reference arm): the service builder and the protocol-log reader, which
    makes any failure `MEASURED_INVALID`.
  - `fast`: the fast tiers (structure on the functional model, the changed groups timed on the RTL
    emulator through `merlin.perf.whole_model_group_timing`, one group's check), exposed as broker
    actions and never the objective.
  - `retention` / `store_admin`: what a finished job keeps, and logged operator edits to a store that
    state why and delete nothing.
- Objective and feedback
  - `config` / `objective`: the launch config and the objective it builds (screen and certifier, the
    builder pin, each machine's reference result, the instruction policy), refused at launch when
    underdetermined.
  - `feedback` / `transfer` / `roofline` / `forms`: per-group comparisons against the reference on the
    same machine, held-out transfer by op kind, the derived per-group roofline, and perf-capsule
    coverage of the forms that hold a model's cycles. Data for the agent, never a remedy.
  - `ledger`: the run's `oot/` history. The harness commits each candidate (tree digest equals the
    store's package digest), tags `measured/<n>` when it lands, moves `best` only on a confirmed win,
    and assembles champion evidence for `merlin.targetgen.champions`.
- Cell mode
  - `cells` / `cell_prep` / `cell_runs` / `group_capsules` / `group_capsules_board`: a cell's groups (and
    held-out and collateral groups) built as one-group programs by the target's own driver, both arms
    timed on the emulator or in one board batch with the control; a cell run's config is composed from
    a measured run's own (`merlin-experiment cell prepare|launch`).
- Runs and sessions
  - `runs` / `snapshot` / `profiles` / `launch`: a run on disk (prepared, resumed, relaunched), its
    frozen source taken from a commit, its launch profile as data, and a detached launcher found again
    by its own record.
  - `rounds` / `round_audit`: one authoring session over the run's candidate (fresh workspace, broker,
    bwrap boundary, transcript audit joined to receipts, edit authority) and the replay of a recorded
    round's audit.
  - `sessions` / `progress` / `liveness` / `watchdog`: when a run stops (plateau, bar, budget, quota or
    infrastructure evidence, never a round count), its status read from its own records, its
    heartbeat, and a watchdog that only acts on a launcher already gone.
  - `cli` / `__main__`: the command line.

Nothing here names a target, a model, a board or a simulator binary: they arrive as data.

Tests use synthetic builders and process fixtures (`tests/wmm_fixtures.py`). Do not launch agents or
simulators while validating package structure or imports.
