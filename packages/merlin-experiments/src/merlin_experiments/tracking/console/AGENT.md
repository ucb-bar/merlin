# AGENT.md — packages/merlin-experiments/src/merlin_experiments/tracking/console

## Purpose

The experiment command console: discover, track, inspect and diagnose any experiment from its records.

## Modules

- `cli.py` — ``merlin-experiment console``: start the frontend on a terminal, or print text everywhere else.
- `diagnose.py` — Bottleneck findings derived from one console snapshot.
- `discovery.py` — Find every experiment on this machine and say, with its basis, whether it is running.
- `io.py` — Bounded, cached, read-only record access shared by every console source.
- `launch.py` — Find, build and start the Bubble Tea frontend; fall back to the plain renderer without it.
- `model.py` — Assemble one experiment's snapshot from the console sources, on a fast and a slow cadence.
- `placement.py` — Who holds which CPUs, and keeping the console itself off the experiments' CPUs.
- `plain.py` — The console snapshot as plain text: ``--once``, pipes, CI logs and hosts without the frontend.
- `serve.py` — The console's data process: JSON lines out, commands in. The Bubble Tea frontend spawns it.

## Subpackages

- `sources/`

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->

## Invariants

- Two processes, one owner of records: the Python data process (`serve.py`) reads every record; the
  Bubble Tea frontend (`packages/merlin-experiments/console`) only renders its JSON lines. Never add a
  record reader to the Go side.
- The section contract lives in `sources/__init__.py`; `console/internal/proto/proto.go` mirrors it.
  Change both together and extend `tests/test_console_*.py` and `internal/ui/ui_test.go`.
- Read-only and bounded: every source uses `io.Reader` (stat-cached documents, a per-refresh byte
  budget, offset tails) and bounded directory listings. Nothing writes into a run, a workspace, a
  launch directory or a store; nothing launches jobs or touches hardware.
- A missing value is `None` on the wire and `—` on screen, never 0. Modeled, measured and structural
  numbers keep their labels; CPU activity is telemetry; diagnostics grant no verdict.
- An experiment's state always carries the record that decided it (`discovery.classify`); an owner
  record alone never decides it, and a new launch is a new experiment.
- Private material (capsule names, paths, agent messages, golden references) only with
  `operator_private`; `golden` is not even collected otherwise (`model.OPERATOR_ONLY`).
- No `re`, no target-name literals: the stage taxonomy is data (`merlin/contract/compiler_stages.yaml`).
- Machine-agnostic: no CPU numbers, core counts or cgroup names in code. CPUs come from `/proc/stat`,
  run CPUs from runner affinity, reservations from cgroup containment relative to the console's own
  cgroup (`placement.py`). The console must not compete with the experiment: keep `place_console`,
  the self-cost throttle in `serve.py`, and memoized sections (return the same object when nothing
  changed so the server skips re-encoding).
