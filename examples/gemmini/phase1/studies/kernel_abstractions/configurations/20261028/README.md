# Experiment configurations recorded on 2026-10-28

This directory preserves experiment setups and the exact input inventories
needed to identify what each author received. Start with
[`registry.json`](registry.json), then open an experiment's `configuration.json`,
`specs/`, and `receives.json`. `SHA256SUMS` verifies the archived files.
No authentication or `.env` contents are included.

The editable task remains `/work/merlin/tasks/el4-grader-throughput-20261024`,
branch `cc/el4-grader-throughput-20261024`. Frozen exports identify the executed
framework; a run's `environment.repo_sha` can instead describe the enclosing
editable checkout and must not substitute for the frozen source revision.

## The next three experiments

All three start with an empty target backend. They use the actual deployed
fresh10CCA framework `f9d7fb5c3d6f` plus common harness repairs and explicit
capability removals. The current frozen source is `3f3fab435f38`; preparation
and qualification records are in the original task's
`.tmp/fresh10-tooling-ablation-20261028/fresh10-tooling-20261028-r5/`.
The registry records preparation state; it does not itself authorize or prove a launch.
The three generation-5 services were launched after qualification. Their native
authoring clocks started at 18:31:57–18:32:12 UTC; authoring ends around
22:32 UTC, with final draining through approximately 22:47 UTC. The original
study directory retains `launched.json`, `health/latest.json`, and
`operator-checkpoints/index.json` for current status. `configuration.json`
records the dispatch/clock observation at its archive timestamp.

| Setup | Reference kernels | CCA / automatic kernel learning | Scheduling representations and legality | Supplied transformation/pass library | Full-source Python modules | SDK files |
|---|---|---|---|---|---:|---:|
| [A: infrastructure](fresh10-tooling-infra/configuration.json) | Same authorized bank as B | Off | Withheld | Withheld | 304 | 118 |
| [B: scheduling](fresh10-tooling-scheduling/configuration.json) | Same authorized bank as A | Off | Available | Withheld | 324 | 137 |
| [C: no reference kernels](fresh10-tooling-no-reference-kernels/configuration.json) | Withheld | Off | Available | Available | 1083 | 487 |

The counts above are Python modules in the full editable source, and all files
in the separate SDK. SDK Python module counts are 115, 134, and 484. Every full
source projection also carries the same fourteen generic data resources.
Each `receives.json` enumerates the exact source/SDK members, hashes, public
entry modules, unavailable API routes, tools, and allowed/denied input paths.

A and B receive the same 23 raw reference members from nine origins and the
complete 155-entry reference map. This includes authorized optimized kernel
sources and model ELFs/disassemblies. Neither receives reference-performance
annotations, prior reference task cards, live CCA, or automatic kernel learning.
C has zero author grants for kernel sources, ELFs, disassemblies, reference
maps, or CCA-derived artifacts. The corpus producer retains five inherited
reference files as host-only judge inputs; native sandbox checks must prove
that the author, descendants, and candidate cannot read these paths or aliases.
General compiler algorithms and passes remain available in C.

All three receive `merlin_infra`, `xdsl_kit`, `isa_tools`, `rtl_generators`, and
`rtl_facts`, plus common public contracts, ISA/runtime/toolchain inputs,
numerical requirements, hardware facts, ordinary self-checks, and feedback on
their own candidates. All retain general performance diagnostics. A/B have
eleven common procedural skills. C adds nine admitted compiler/pass skills and
its real catalog guidance. Unavailable pass/catalog instructions are removed
from A/B, including the provider's initial and resumed prompt suffixes.

| Shared setting | Value |
|---|---|
| Authoring clock | 14,400 seconds: four hours, beginning at the native campaign clock |
| Final drain/validation | 900 additional seconds; no extra authoring |
| Author model / reasoning | `gpt-6.1-sol`, high; helper reasoning medium |
| Resources | Fifteen CPUs and at most eight author helper agents per run |
| Workloads | The same `xs_capsules`: 109 core + 40 scalability + 3 connected public programs; 14 protected checks |
| Full-model obligations | The same three private models and six program obligations: ResNet-50 ×1, SmolVLA ×3, TinyLlama ×2 |
| Judge/model | Same fresh10 numerical, ISA, ordering, timing, calibration, and full-model policy; ordinary continuous promotion |
| Hardware-loop/sequencer roles | Prohibited in all three |
| Initial state | Fresh backend, client home and run state; no finished authored compiler supplied |
| Author CPU pools | A: 41–47,56–60,91–93; B: 61–75; C: 76–90 |
| Admission limit | At most three of these and five full-size campaigns machine-wide; external 4–37,48–55 untouched |

A versus B isolates the supplied scheduling abstractions. B versus C changes
both pass-library access and kernel-reference access, so that comparison alone
does not isolate a pass-library effect. C is the requested full general tooling
environment without reference kernels or CCA. A later full-tooling+references,
CCA-off control would separate those two differences; it is not one of this wave's runs.

## Earlier setups

| Setup | Executed source | What distinguishes it | Records |
|---|---|---|---|
| EL1 | `adff3ab95cae` | No Merlin compiler source or supplied Merlin tools; common contracts/runtime/tests and authorized shared inputs | [configuration](el1/configuration.json), [received inputs](el1/receives.json) |
| EL2 | `adff3ab95cae` | C++ OOT scaffold/component closure; author implements the dialect/compiler in C++ | [configuration](el2/configuration.json), [received inputs](el2/receives.json) |
| EL3 | `adff3ab95cae` | Larger component closure including xDSL support; no EL4 CCA/pass environment | [configuration](el3/configuration.json), [received inputs](el3/receives.json) |
| fresh10seed | `afcf34be293c` | Existing external seeded fresh10 baseline; its actual seed and tool selections are in the native plan | [configuration](fresh10seed/configuration.json), [received inputs](fresh10seed/receives.json) |
| fresh10cca | `f9d7fb5c3d6f` | Existing external fresh backend with full tooling, CCA, and authorized expert/reference material | [configuration](fresh10cca/configuration.json), [received inputs](fresh10cca/receives.json) |

EL1/2/3 completed their four-hour authoring periods. Their archived configuration
and final 14,400-second source snapshot pins are retained here; they are source
packages for subsequent FireSim testing, not evidence of completed RTL qualification.
All three earlier EL runs also received the same authorized raw reference-kernel
tree, as recorded in their actual `allowed` inventories; EL1 was not a
reference-free environment. EL2's three supplied entry modules are
`mlir_scaffold`, `llvm_plan`, and `target_repo`. EL3 additionally receives
`generate.xdsl` and the `iface`, `dialect`, `llvm_context`, `cmdbuf`, and `verify`
OOT starter APIs. Their exact qualified names are in `receives.json`.
The external fresh10 baselines have twelve-hour campaign budgets. Compare matching
author-time checkpoints rather than their final results against a four-hour study.
The earlier EL study also used deferred evaluation; it is not the normal
continuous-promotion policy selected for this new wave.

## Measurement and custody

Retain native transcripts/usage, request and response records, completed grades,
per-case outcomes/timings, evaluator and compiler hashes, model results,
promotion events, errors/retries, CPU masks, and immutable source checkpoints
at 0/1800/3600/7200/14400 author seconds. The operator observer starts before
authors; missed/moving snapshots are marked unavailable instead of replaced
with later source. Author-visible feedback remains the ordinary shared workflow.

For abstraction effects, derive observable invalid schedule/legality attempts,
rework and debugging effort, successful reuse, source churn, compile/validation
latency, requests per valid improvement, tokens and time to correctness or a
performance threshold, and transfer to identically selected held-out shapes.
Internal attempts that leave no artifact cannot be counted exactly: label
these measurements as observed counts or lower bounds. Do not manufacture
held-out transfer results or physical performance from provisional cycle models.

These are exact configuration snapshots with original absolute paths preserved.
Copying a YAML alone does not reproduce the source export, toolchains, sealed
inputs, services, or credentials. Follow the native plan and immutable input pins;
use a fresh run identity for a rerun. Prior failed, unlaunched preparation
generations are preserved in the task and must not be reported as experiments.

## Corrected reference-free EL1/EL2/EL3 replacements

The completed level-study runs received expert kernels and are ineligible as primary results for the intended reference-free ablation. Their evidence is preserved. See [the protocol correction](PROTOCOL_CORRECTION_NO_EXPERT.md).

The replacement configurations, common policy, guidance, frozen source receipt and static audit are archived in [el123-no-expert/](el123-no-expert/archive.json). Canonical preparation state lives in `.tmp/xs-el123-noexpert-20261028-r1/` in the original task. All three start fresh, exclude all expert kernel material and CCA, and keep the existing Spike model and feedback. Their source is pinned to `02d14a7d523f1ff72eaf4fb6ba6c56d2bbd47b33`. They retain four hours of authoring, a 900-second drain, fifteen CPUs and up to eight helpers per arm.

These replacements are queued. Native preflight on their actual fifteen-CPU groups must wait for the active tooling runs to finish and release those groups. No author launch or CPU reservation follows from this archive.
