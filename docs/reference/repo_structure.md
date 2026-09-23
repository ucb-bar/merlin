---
title: Repository structure
kind: reference
status: current
owner: core
last_verified: 2026-09-21
related: [architecture, getting_started, storage]
code_refs: [pyproject.toml, src/merlin, packages, experiments/catalog.yaml]
---

# Repository structure

Start at `experiments/catalog.yaml` to select a phase workflow. Shared compiler implementation
lives in `src/merlin`; optional research implementation lives in independently installed
distributions under `packages/`. The old `merlin/python/merlin` directory is a compatibility
symlink to core, not the place to add new research modules.

```
src/merlin/    compiler IR, schedules, capture contracts, target/toolchain resolution, runtime, verification
packages/
  merlin-experiments/ phase orchestration, agent tooling, trusted grading and certification
  merlin-dse/         design-space exploration, design pressure and guidance
  merlin-mining/      mining/search campaigns and tuning studies
  merlin-analysis/    baselines, comparisons, reports and visualization
experiments/
  catalog.yaml       experiment discovery: merlin experiment list
  definitions/       versioned phase definitions and explicitly non-runnable templates
  reference-data/    retained, read-only analysis inputs; never a new output destination
build_tools/   build/dev tooling + measurement/analysis/sweep runners + repo linters (scripts/, cmake, docker)
docs/          this documentation (incl. generated cli.md = CLI surface)
examples/      runnable end-to-end examples; examples/triton/run.sh needs no toolchain
third_party/   hard build/test deps only (no analysis repos)
out/           the SINGLE generated-output root (gitignored except the skeletons + curated
               .gitignore negations named below)   ── see CLAUDE.md "Generated-output convention"
  runs/        aet experiment runs
  artifacts/   all other generated products; artifacts/targets/ = codegen packages
               (replaces retired generated_targets/; tracked baselines/champions via .gitignore negations)
  build/       generated build outputs + buildable OOT codegen repos (build/generated/)
merlin/
  python/merlin -> ../../src/merlin (legacy path compatibility only)
  contract/    the benchmark contract: capsule corpus, interface grammar, hardware pins
  prompts/     prompt templates loaded at runtime (product data, bundled into the wheel)
  runtime/     target-independent C runtime substrate (c/ + baremetal/ + abi/)
  targets/     reference target resources retained pending qualified OOT extraction
  schemas/     cross-workstream coordination contract
  benchmarks/  shared workload descriptions
  experiments/ retained native phase engines and experiment resources (not the catalog)
  tests/       integration/conformance/golden/data
```

## Find the owner before changing code

Use the paper's three phase names when describing workflows. A **capsule** is an
executable coverage contract, not a generated compiler. A **target backend** is the
reusable compiler output, not a per-workload tuned kernel. The **command buffer**
is shared across targets; a **target dialect** and its concrete schedule are
target-specific. Shared scheduling representations and checks remain in core.

| Paper concept | Implementation home | Execution and migration status |
| --- | --- | --- |
| Phase 0 — Hardware-Guided Test Generation | `packages/merlin-experiments/src/merlin_experiments/phase0/` | Installed phase-0 engine; no native controller required |
| Phase 1 — Functional Compiler Generation | `packages/merlin-experiments/src/merlin_experiments/phase1/` | Installed `python -m merlin_experiments.phase1`; native baseline launcher delegates to the same controller; catalog/treatment migration remains in progress |
| Phase 2 — Performance Optimization | `packages/merlin-experiments/src/merlin_experiments/phase2/` | Broker execution and evidence have canonical owners; large controllers remain in `merlin/experiments/gemmini_perf_bench/scripts/` pending extraction and target-policy separation |

Within Phase 1, `tools/` holds public standalone clients, `providers/` runs agent
transports, `brokers/` serves host-side tool requests, and `feedback/` owns trusted
grading, repair feedback and final freeze.
`controller.py` composes admission and authoring; `session.py` owns fresh/resumed
admission, `task_staging.py` owns task composition, and `workspace_transport.py`
owns workspace assembly and mask probes. Installed execution takes explicit operator
inputs; it does not infer a native target or toolchain layout. Diagnostic execution
tests do not qualify numerical compilation or OS sandbox isolation.
These are not interchangeable with candidate-visible tool clients or the generated
backend. Within Phase 2, `claims/` evaluates measured claims; `campaign.py` binds
functional inputs, while `measurement_evidence.py`, `statistics.py` and `reporting.py`
own measurement admission and interpretation. Whole-model authoring and post-freeze
measurement remain separate: an accepted edit is not a measured speedup.

Reviewed corpus handoff spans phases, so it lives beside them in
`merlin_experiments/corpus/`: `preparation.py` assembles inputs, `release.py` owns
review/sealing, and `admission.py` owns evaluated coverage and cohort selection.
It is distinct from Phase 0's generation engine and from core-only contracts.

Phase 1 post-hoc reporting lives in
`packages/merlin-analysis/src/merlin/agentreport/phase1/`: `runs.py` reads run
evidence, `by_treatment.py` compares treatments, and `by_model.py` compares agent
models. Readers take explicit target, run and report roots; they do not select a
native experiment on import. Historical native reporting commands are CLI adapters.
Reporting is separate from trusted grading and does not certify a compiler.

Use descriptive names inside those owners rather than repeating `phase1_`,
`phase2_` or a target name in every filename. Study-specific plotting, baselines
and paper evaluation are consumers of phase results, not phase-engine code.

| Task | Start here |
| --- | --- |
| Define or inspect phases 0, 1, 2 | `experiments/README.md`, then `packages/merlin-experiments/src/merlin_experiments/` |
| Change shared compiler or scheduling behavior | `src/merlin/compile/`, `src/merlin/xdsl_dialects/`, `src/merlin/sched/` |
| Change target discovery or OOT contracts | `src/merlin/targetgen/`, `build_tools/upstreams/target_support.json` |
| Change agent execution, tool brokers or final grading | `packages/merlin-experiments/src/merlin_experiments/phase1/` |
| Change shared evaluator or sandbox implementations | `packages/merlin-experiments/src/merlin/targetgen/` |
| Change research analysis or search | The owning `packages/merlin-{analysis,dse,mining}/src/` tree |
| Locate outputs or determine whether cleanup is safe | `merlin storage report`, `merlin/contract/storage.yaml`, [storage guide](../guides/storage.md) |
| Find a command's module and distribution | [Generated CLI reference](cli.md) |
| Find tests | `merlin/tests/<subsystem>/`, plus focused `packages/*/tests/` |

Use `merlin experiment runs --target TARGET --experiment EXPERIMENT` to discover
stored phase orchestrations under the configured run root. Omit either filter to
compare targets or experiments; use `--root` for another retained run root. Entries
come from hash-checked existing receipts, not a second database. Corrupt/incomplete
records are reported separately, and discovery neither follows directory aliases
nor changes files. Inspect an explicitly placed run with `merlin experiment status
/absolute/run/path`; native engine accounting remains available through `aet runs`.
Both discovery and status expose a `phases` summary with each phase's adapter,
latest execution state, attempt count, engine-output path and latest log. Resumed
optimization segments point to their recorded latest output, not the original
planned directory. These are navigation pointers, not new certification or checks
that historical output bytes are still present and valid.

The optional distributions contribute uniquely owned modules to the `merlin.*` namespace.
An import beginning with `merlin` does not imply core ownership. Core owns shared namespace
initializers; extensions must not overwrite them. See the
architecture and this ownership map for preserved identities and remaining
checkout-dependent engines. Target support lives OOT where qualified; retained in-tree resources
are not evidence that a staged OOT snapshot is an executable, certified compiler.
Target-specific code in core or optional distributions is remaining migration debt,
not an exception to target independence. Architecture-class abstractions must expose
their real capability assumptions; accelerator-specific tuning and errata belong OOT.

All generated output lives under the single `out/` root, with exactly three subdirs — `out/runs/`,
`out/artifacts/`, and `out/build/` (see CLAUDE.md "Generated-output convention"). The old top-level
`runs/`/`artifacts/`/`build/` and the retired `output/` (model recaptures now live at
`out/artifacts/recaptures/` via `recaptures_dir()`) are gone; the guard hook blocks writes outside
`out/`. Every directory contains an `AGENT.md`; under `out/` only `AGENT.md` / `README.md` /
`.gitkeep` (plus curated `.gitignore` negations) are tracked.
