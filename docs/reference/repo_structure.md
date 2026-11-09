---
title: Repository structure
kind: reference
status: current
owner: core
last_verified: 2026-11-09
related: [architecture, getting_started, storage]
code_refs: [pyproject.toml, src/merlin, packages, experiments/catalog.yaml]
---

# Repository structure

For ordinary compilation, start at `src/merlin/api.py`. For an agent experiment,
start at `experiments/catalog.yaml`. All Merlin Python implementation
ships in one `merlin` distribution from `src/`. Shared compiler, trusted evaluation and research
workflows retain distinct logical owners; dependency extras enable optional engines rather than
installing additional copies of the code. The old `merlin/python/merlin` directory is a compatibility
symlink, not the place to add new modules.

This map describes the current checkout. Compiler ownership is clearer than the
retained legacy areas, and some active modules still need responsibility-based
decomposition. A passing structure gate does not establish that every file is
short, cohesive or easy to read.

## Read the compiler in this order

```text
src/merlin/
  api.py                 public compile, run and inspection calls
  service_cli.py         command-line arguments for those same calls
  compiler/
    request.py           source and compile options
    artifact.py          products, manifests and deployment records
    system.py            discover implementations for a selected system
    pipeline.py          materialize a selected region plan
    invocation.py        call a provider and admit its returned products
    frontend/            capture/import and source-state binding
    quantization/        numerical contracts, preparation and legalization
    ir/                  shared graph, effect and boundary records
    backend/             provider contracts, offers, composition and call ABI
    planning/            choose compatible regions and implementations
    partition/           materialize selected boundaries
    transforms/          reusable source-IR transformations
    passes/              pass contracts, composition and execution adapters
    schedule/            shared tiling, retention, transfers and command ordering
    lower/
      scalar/            scalar computation lowering
      llvm/              shared terminal LLVM conversion and ABI admission
      c/                 portable control, dispatch and handoff emission
    build/               compile/link products and selected platform images
      zephyr/            configuration from selected SoC/board/tree capabilities
  runtime/               execute artifacts or prepare/collect external execution
  verification/          independent references, proofs and artifact checks
  target/spec/           reviewed numerical, instruction and effect declarations
  development/           experiment definition validation and command selection
```

To understand a single transform, open its implementation in `transforms/` or
`schedule/` first. To understand the complete compilation flow, open `api.py`,
then `system.py` or `pipeline.py` according to the selected route. These are
different reading tasks: the algorithm should be understandable in its owner;
the orchestrator explains how independently meaningful stages compose.

OOT lowering begins at the provider invocation in `compiler/invocation.py`:
`provider.compile` or the admitted `provider.compile_selected` hook. The actual
target dialect lowering, encoder and device implementation live in the selected
OOT package. Its products may be LLVM, objects, device binaries or commands.
The scalar provider starts in `compiler/backend/scalar.py`, reusing scalar and
LLVM lowering. The pass library spans source transforms and counted scheduling;
`passes/` supplies the contracts and invocation machinery, not every algorithm.

`llvmlower/`, `targetgen/`, `sched/` and `transforms/` outside `compiler/` still
contain active implementation and research interfaces. Some older modules under
`compile/`, `verify/` and other packages are exact compatibility identities.
Check the module's owner declaration rather than assuming an entire directory is
obsolete or another editable copy. The task table below gives current entry points.

The readability policy is in [CONTRIBUTING.md](../../CONTRIBUTING.md): one coherent
concept per owner, decomposition review near 400 handwritten lines, ordinarily
under 600 for new or substantially rewritten modules, and comments explaining
invariants and reasons. Splits need real independent responsibilities or contracts;
small forwarding modules alone do not improve readability.

## Repository roots

Read-only resource identity is independent of writable output roots.
`merlin.common.paths.data_path` selects this implementation's actual source
checkout, an explicit `MERLIN_REPO_ROOT`, or the installed core's bundled
`merlin/_data` files. An incidental caller CWD or `MERLIN_WORK_DIR` containing
another `merlin/runtime`, contract or schema tree cannot shadow installed data.
Resource-specific overrides such as `MERLIN_RUNTIME_DIR` remain explicit
selections. Compilation recipes should pin the actual C/header files they use
as well as the Python package; Python import isolation alone does not identify
the compiled runtime.

```
src/
  merlin/             compiler, runtime, verification and logical research/evaluation owners
  merlin_experiments/ phase orchestration, agent tooling, trusted grading and certification
  merlin_dse/         typed design-space exploration interfaces
packages/
  merlin-timing/     independent measurement package and dependency environment
  other directories retain historical metadata and inputs, not a second active test suite
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
  tests/       compact compiler/runtime, verification and contract cases with shared fixtures/data
```

## Find the owner before changing code

Use the paper's three phase names when describing workflows. A **capsule** is an
executable coverage contract, not a generated compiler. A **target backend** is the
reusable compiler output, not a per-workload tuned kernel. The **command buffer**
is shared across targets; a **target dialect** and its concrete schedule are
target-specific. Shared scheduling representations and checks remain in core.

| Paper concept | Implementation home | Execution and migration status |
| --- | --- | --- |
| Phase 0 — Hardware-Guided Test Generation | `src/merlin_experiments/phase0/` | Installed phase-0 engine; no native controller required |
| Phase 1 — Functional Compiler Generation | `src/merlin_experiments/phase1/` | Installed `python -m merlin_experiments.phase1`; native baseline launcher delegates to the same controller; catalog/treatment migration remains in progress |
| Phase 2 — Performance Optimization | `src/merlin_experiments/phase2/` | Reusable authoring, campaign and frozen-suite controllers live in `src/merlin_experiments/phase2/`; native compatibility launchers and target-specific study drivers retain their explicit checkout inputs |

Within Phase 1, `tools/` holds public standalone clients, `providers/` runs agent
transports, `brokers/` serves host-side tool requests, and `feedback/` owns trusted
grading, repair feedback and final freeze.
Frozen model-capsule bundle construction lives in `targetgen/capsule_bundle.py`,
with the existing `_capsule_bundle_worker.py` retaining its process boundary.
Private completed-build admission lives in
`merlin_experiments/phase1/feedback/private_model_completion.py`. These grader
owners preserve asset, input/output, loader and receipt checks; they do not
establish numerical execution, model accuracy or candidate-visible access.
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
`src/merlin/agentreport/phase1/`: `runs.py` reads run
evidence, `by_treatment.py` compares treatments, and `by_model.py` compares agent
models. Readers take explicit target, run and report roots; they do not select a
native experiment on import. Historical native reporting commands are CLI adapters.
Reporting is separate from trusted grading and does not certify a compiler.

Use descriptive names inside those owners rather than repeating `phase1_`,
`phase2_` or a target name in every filename. Study-specific plotting, baselines
and paper evaluation are consumers of phase results, not phase-engine code.

| Task | Start here |
| --- | --- |
| Define or inspect phases 0, 1, 2 | `experiments/README.md`, then `src/merlin_experiments/` |
| Compile or run explicit inputs | `merlin.api.compile` / `merlin.api.run`, with an explicit backend / runner; [architecture](architecture.md) |
| Change shared compiler or scheduling behavior | `src/merlin/compiler/{passes,transforms,schedule,lower}/`, `src/merlin/xdsl_dialects/`, `src/merlin/sched/`; historical `compile` imports retain compatibility aliases |
| Change convolution geometry or transfer emission | `compiler/schedule/window_geometry.py`, `convolution_input_rows.py`, `convolution_transfer_laws.py`, `convolution_tiles.py`; `window_transfer.py` owns column and epoch coordination |
| Change convolution alternatives or selected realization | `compiler/schedule/convolution_candidates.py`, `convolution_storage.py`, `convolution_rhs.py`, `convolution_admission.py`, `convolution_realization.py`; `convolution_exploration.py` retains the public orchestration |
| Change dense panel construction or checks | `compiler/schedule/dense_panel_build.py`, `dense_panel_projection.py`, `dense_panel_verify.py`; `dense_panel_pipeline.py` owns checked results and public scheduling |
| Change mechanism comparisons or route admission | `kernels/mechanism_metrics.py`, `mechanism_ports.py`, `mechanism_stream.py`, `mechanism_admission.py`; `mechanism_applicability.py` composes the public report |
| Find base scalar code and installed OOT lowering | `compiler/backend/scalar.py`, `compiler/lower/scalar/`, `compiler/lower/llvm/`; explicit OOT providers are selected through `compiler/backend/service.py` and `merlin.backends` |
| Change saved-model input binding or readiness | `runtime/modelinputs.py`, `compiler/frontend/materialized.py`, `compiler/frontend/preflight.py` |
| Change source routing, coverage or exported model interfaces | `compiler/planning/{model,complete,coverage}.py`, `compiler/partition/model.py` |
| Change original host/source model builds | `model.py`, `compiler/build/{saved,modeloptions,owned,siblings,spatial}.py`; execution lives in `runtime/model.py` |
| Change mesh execution or independent checks | `compiler/selection/mesh.py`, `runtime/mesh*.py`; independent algorithms live in `verification/{mesh,mesh_reference,model,completed_model,scalar_host}.py` |
| Change target discovery or OOT contracts | `src/merlin/targetgen/`, `build_tools/upstreams/target_support.json` |
| Change agent execution, tool brokers or final grading | `src/merlin_experiments/phase1/` |
| Change frozen performance-suite orchestration | `src/merlin_experiments/phase2/{suite,suite_runtime,source_layout}.py`; retained native scripts are compatibility identities |
| Change independent compiler verification | `src/merlin/verification/`; SMT/Lean algorithms and `lean/Merlin` mathematical sources live beside their owners |
| Change shared evaluator or sandbox implementations | `src/merlin/targetgen/` |
| Change research analysis or search | The logical owner under `src/merlin/{agentreport,compare,dse,dse_guidance,mining,perf,plotting}/` or `src/merlin_dse/` |
| Locate outputs or determine whether cleanup is safe | `merlin storage report`, `merlin/contract/storage.yaml`, [storage guide](../guides/storage.md) |
| Find a command's module and distribution | [Generated CLI reference](cli.md) |
| Find active tests and the complete collection budget | `merlin/tests/<subsystem>/`, `merlin/tests/AGENT.md`, `build_tools/scripts/check_test_budget.py` |

These are actual source owners, while the broader destination architecture remains
a plan. Compatibility aliases under `src/merlin/compile/` do not duplicate their
implementations. Source receipts bind the selected implementation and retained
compatibility files; independent references and qualification keep their original
authority. File relocation does not qualify a backend, simulator or platform.
The scheduling source roster includes the actual split owners in SDK26;
frozen SDK6–25 inventories and archives remain unchanged. Public imports keep
their established lookup identities. The [generated module index](module_index.md)
lists packages; the owner rows above locate their implementation files.

The permanent suite is capped at 150 collected pytest cases, counting every
parameterized instance: 70 compiler/runtime, 50 independent verification and 30
contracts. Run the repository-wide budget guard before execution:

```sh
python build_tools/scripts/check_test_budget.py --output out/build/python/ci/test-budget.json
python -m pytest merlin/tests -q -ra
```

The guard discovers owned collectors through Git rather than relying only on
`testpaths`; another project root cannot bypass the ceiling. Collection errors,
collection-time skips and deselection refuse the guard. Exact frozen companion
inputs retain their declared byte identities without becoming active regression
collectors. Retired matrices remain historical Git/frozen evidence, and source
verification algorithms remain under their production owners. A missing external
tool stays an explicit prerequisite, not successful qualification.

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

One distribution supplies the preserved `merlin.*`, `merlin_experiments.*` and `merlin_dse.*`
imports. Logical component ownership is recorded in `build_tools/logical_components.json`;
an import beginning with `merlin` does not imply shared compiler ownership or candidate access.
Shared code must not acquire operational dependencies on research controllers or graders.
Private references, grading, runtime masks and independent timing retain their existing authorities.
See the architecture and this ownership map for remaining checkout-dependent engines.
Target support lives OOT where qualified; retained in-tree resources are not evidence that a
staged OOT snapshot is an executable, certified compiler.
Target-specific code in shared or research components is remaining migration debt,
not an exception to target independence. Architecture-class abstractions must expose
their real capability assumptions; accelerator-specific tuning and errata belong OOT.

All generated output lives under the single `out/` root, with exactly three subdirs — `out/runs/`,
`out/artifacts/`, and `out/build/` (see CLAUDE.md "Generated-output convention"). The old top-level
`runs/`/`artifacts/`/`build/` and the retired `output/` (model recaptures now live at
`out/artifacts/recaptures/` via `recaptures_dir()`) are gone; the guard hook blocks writes outside
`out/`. Every directory contains an `AGENT.md`; under `out/` only `AGENT.md` / `README.md` /
`.gitkeep` (plus curated `.gitignore` negations) are tracked.
