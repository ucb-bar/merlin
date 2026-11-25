---
title: Compiler architecture
kind: reference
status: current
owner: compiler
last_verified: 2026-11-16
related: [repo_structure, ../guides/04-planning, ../guides/06-lowering]
code_refs: [src/merlin/api.py, src/merlin/compiler, src/merlin/common, src/merlin/target, src/merlin/runtime, src/merlin/verification, src/merlin/evaluation]
---

# Compiler architecture

Start with the [ordered guide](../README.md) for commands and examples. This
page describes the boundaries between the components that implement that flow.

```text
source program + precision intent + selected target system
    -> capture / preparation / import
    -> implementation discovery and region/resource planning
    -> selected partition and dispatch boundaries
    -> canonical Merlin Interface IR for OOT regions
    -> shared control + scalar or OOT region lowering
    -> artifact publication / optional image linking
    -> execution and independent verification
```

## Public entry points

`merlin.api.compile`, `inspect`, `inspect_backends`, `inspect_spec`, `run` and
`collect` are shared with the public CLI. An installed backend implements
`validate` and `compile`. An explicitly constructed `SystemCompiler` adds
source-level proposal discovery; `PlannedCompiler` consumes a selected plan.
The output can be IR, an object, a device binary, commands or a complete image.

The saved-model route also retains `merlin.model` and existing capture/build
entry points. Do not infer identical options or coverage from their common use
of shared lowering code.

## Ownership and dependencies

| Component | Owns | Must not decide |
| --- | --- | --- |
| Target/specification | Selected hardware, numerical meaning, topology and evidence | Which unverified backend claim becomes hardware truth |
| Frontend/preparation | Source capture, numerical realization and import | Device placement over source nodes that import may replace |
| Planning | Compatible region groupings, implementations, resources and boundaries | A new numerical meaning to rescue an unsupported implementation |
| Shared transforms | Reusable graph, loop, memory and scheduling algorithms | Target encodings or workload-name shortcuts |
| OOT backend | Implemented target capabilities, proposals, target lowering and encoders | Application acceptance thresholds or independent verification results |
| Build/runtime | Objects, images, deployment, invocation and observed completion | Whether a process exit alone proves numerical correctness |
| Verification/evaluation | Semantic checks, references and execution assessment | Candidate compiler optimization policy |
| Experiment orchestration | Authoring rounds, selected inputs and recorded studies | A second implementation of the shared compiler |

Shared compiler mechanisms remain below experiment orchestration. LLVM
finalization remains below scalar, vector and control compilation; it must not
import the driver or backend loader. Framework dependencies are imported only
by the workflows that need them. A package being installed does not grant a
candidate access to graders, hidden inputs or independent references.

## Where to change code

Follow the program through these owners. A folder groups one stage or one kind
of evidence; a backend-specific implementation stays in its OOT repository.

| Work | Start here |
| --- | --- |
| Compile request, output and orchestration | `compiler/compilation.py`, then `compiler/system.py` |
| Backend offers and callable region boundary | `compiler/backend/source_proposals.py`, `compiler/backend/c_abi.py` |
| Input capture and import | `compiler/frontend/program.py` and `export.py`; selected framework conversion runs through `capture/preparation/` |
| Numerical preparation | `compiler/quantization/` |
| Canonical OOT input | `compiler/lower/interface.py`: source coverage, conversion, parsing and structural admission |
| Live values, effects and partition boundaries | `compiler/ir/` and `compiler/partition/` |
| Shared transformations | `compiler/transforms/`; `source_rewrites/` groups source-IR rewrites |
| Hardware-aware scheduling library | `compiler/schedule/`: loops, panels, storage, shared operands and spatial schedules |
| Pass discovery and invocation | `compiler/schedule/api_catalog.py` describes available composition tools; `compiler/lower/native/invocation.py` invokes native pipelines |
| Scalar computation and LLVM finalization | `compiler/lower/native/` |
| Saved-model lowering | `compiler/lower/model/lower.py` coordinates preprocessing, native lowering and codegen; typed source families live in `compiler/lower/source/` |
| Objects and linked images | `compiler/build/`; `model_preparation.py` owns shared model preparation |
| Running an existing artifact | `runtime/execution/`, with dispatch in `runtime/dispatch/` and target execution adapters in `runtime/backends/` |
| Static observations and performance evidence | `analysis/observation/` and `analysis/perf/` |

The shared transformation library contains algorithms that a backend or authoring
agent can select, modify or extend. Backends provide the relevant resource
descriptions, command effects and encoders. Moving these algorithms into a shared
folder does not make every transformation legal for every target: the selected
facts, numerical contract and effect checks still determine applicability.

All backends use one selected Merlin base. An OOT package keeps its target
transforms in `src/<backend>/transforms/` and local alternatives to shared passes
in `transforms/overrides/`. Its pipeline selects those implementations explicitly.
An override names its base contract and preserves the required legality and
verification checks. A missing reusable hook belongs in the common Merlin API;
the backend does not carry a separately edited SDK or replace shared functions
through import order.

Merlin owns upstream conversion through canonical `interface` IR. The shared
owner is `compiler/lower/interface.py`: `to_interface` covers the source function,
`parse_interface` reads the registered grammar, and `validate_interface` checks
the shared boundary. The normal source kind is `merlin.interface.v1`. An OOT
backend selects target implementations from those operations, rather than
implementing another raw-linalg frontend. Interface source retains tensor
arguments/results and numerical meaning; the backend owns its physical ABI.

The shared source conversion admits a static signed i8-by-i8, i32-accumulating
matmul or batched matmul initialized from zero. Its complete output can remain
i32 or use a proved uniform ordered-binary32, nearest-even, saturating i8
readout with optional ReLU; exact integer saturation is also represented.
It preserves the complete function and refuses uncovered work. Static final-axis
channel scales and f32 bias use actual SSA constants and ordered readout steps;
target admission requires an exact proof for every channel before selecting a
store scale or integer preload. Dynamic qparams, other axes and broader mixed
graphs remain explicit compatibility scope. This does not establish full-model
import coverage.

Gemmini's `conversion/from_interface.py` selects its preserved primitive
strategies from canonical operations. `conversion/typed.py` accepts typed
Gemmini IR, and `conversion/to_llvm.py` owns terminal instruction conversion.
`conversion/from_source.py` is compatibility delegation through Merlin's shared
conversion. Historical direct adapters and kernel builders remain explicit
implementation paths; they do not define the normal upstream boundary.
These boundary changes have received static checks only; execution qualification
is deferred.
Shared whole-program transactions live in `compiler/build/host_regions.py`;
`compiler/build/llvm_bundle.py` materializes compatible LLVM inputs only.
Other device artifact kinds retain their own admitted build and invocation paths.

`compiler/backend/interface.py` defines source domains, named implementation
families and target pipeline declarations. Existing discovery consumes these
through the optional OOT `describe_compiler()` hook, checks agreement with source
patterns and lowering recipes, and records explicit legacy status when absent.
These are source-level interface additions; execution qualification is deferred.
Descriptions grant no target or numerical authority. Fresh backend authoring
uses `development/targetgen/authoring/compiler_contract.py` version 6 to require
actual Interface conversion, supported composition and program integration.
Historical contract versions retain their text. `backend/boundary.py` prepares
the Interface request before calling the provider, preserving the original
source identity alongside the Interface identity and emitted artifacts.
`planning/interface_candidates.py` owns the source-region rules and queries
the backend only for implementations of the resulting Interface program.

## Verification and evaluation

`verification/` checks a specific compiler claim. `translation/` contains SMT
encoding, dialect semantics, refinement and pass validation. `host/` checks
emitted host code, copies and packing. `lean/` runs artifact-bound obligations;
the mathematical sources remain in `lean/Merlin/`. `references/` contains
independent numerical references and completed-artifact checks. The public Lean
contract stays separate from private checker implementation.

`evaluation/` assembles those checks into an assessment. `capsule/` separates
operation grading, model grading and evidence. `model/` separates input bundles,
kernel replay, whole-model build/relink and machine binding. Its linalg evaluator
computes directly from source IR, independently of compiler lowering.
`route_admission.py` checks declared routing; `host_compute_accounting.py` checks
arithmetic in emitted host code. These are independent obligations: a process
returning successfully does not establish numerical correctness or device use.

Source moves change the implementation bytes bound by new evidence. Historical
results keep their original source snapshot. The current organization is not a
claim that every backend or hardware route has been qualified after the move.

## Compiler development

`development/testgen/` derives compiler tests from selected specifications and source
evidence. `development/backend/` owns the native authoring engine, which develops
correctness and performance together. `experiments/` inside the Python package selects
and tracks those workflows; the nine authored files at the repository root only provide
the catalog, generic definition and template. External-framework comparisons belong in
`evaluation/frameworks/`; the console and independent timing observer live in `tools/`.

Generated corpora, bundles, captures, compiler snapshots and caches live under ignored
`out/`. A run binds its explicitly selected reviewed inputs. Historical artifacts retain
their original bytes and source identities; they are not discovered as fresh defaults.

## Representations

The xDSL prototypes live in `src/merlin/compiler/dialects/`:

| Namespace | Role |
| --- | --- |
| `contract` | Facts, capabilities and obligations |
| `schedule` | Selected decisions and bindings |
| `interface` | Target-independent hardware/software abstractions |
| `runtime` | Generic launch, events and measurements |
| `dse` | Interface candidates and measured variant results |

These roles do not imply that every compile route materializes each dialect in
that order. Some routes use typed Python records and call a selected OOT service.
Target dialects are supplied by their backends.

`merlin.compiler.kernel_ir.ir` uses Python records and counted `mk` text. Shared block
scheduling uses explicit load/compute/store records in `compiler/schedule/`.
These are different representations from xDSL `schedule` operations and the
upstream MLIR Transform dialect; adapters must be explicit.

## Composition

A SoC integration selects component providers and contributes integration policy,
transfers and completion requirements. It reuses the component compilers rather
than copying them. Scalar and RVV implementations can share one CPU resource.
Availability, addressability and concurrency follow the selected topology.

A selected region's artifact must agree with its callable boundary. A raw binary
needs an admitted loader/wrapper; LLVM is only one possible output path.
Numerical fidelity, complete target lowering and observed device execution are
separate verification obligations, described in [chapter 9](../guides/09-verification.md).

## Scaled FP8 numerical admission

`compiler/quantization/contraction/scaled_fp8.py` owns the target-neutral `ScaledFP8Spec` and `ContractionNumerics` records. They are an admission boundary for a selected numerical route: comparing declarations does not prove that an implementation realizes them, and they confer no device qualification.

An i8 code tensor and a signed i32 power-of-two exponent tensor form one value, `decode_e4m3fn(code_bits) * 2**exponent`. Bytes `0x7f` and `0xff` are reserved NaNs and refuse. Exponent axes are explicit logical code axes in increasing order, and other axes broadcast; there is no E8M0 reinterpretation, hidden transpose, or implicit broadcast. Physical packing stays in the target repository.

`require_same_numerics` admits only exactly equal declarations of accumulation, readout, ordered stages, bias rounding, and activation form. Any difference refuses unless an explicit precision change is authorized.

## Source bias separation

`compiler/semantic/prequantized_bias.py` splits a final BF16 column bias from a prequantized contraction into a readout, an exact column gather, and an `ieee754-bf16-add-rne-v1` addition, without changing constants. An f32 bias stays unsplit. The split receipt records the original nodes and digests, and verification restores and replays the split exactly.
