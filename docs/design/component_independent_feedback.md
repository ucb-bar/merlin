---
title: Independent component feedback authority
kind: design
status: current
owner: experiments
last_verified: 2026-10-09
related: [component_phase2_workflow, component_final_evaluation]
code_refs:
  - packages/merlin-experiments/src/merlin_experiments/phase2/component_baseline.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/component_runtime.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/component_runtime_authority.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/component_runtime_qualification.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/component_execution.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/component_variants.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/component_measurement_qualification.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/component_applicability.py
  - src/merlin/perf/component_applicability.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/component_providers.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/component_launch_inputs.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/component_observer.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/rtl_engine_protocol.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/rtl_engine_probe.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/rtl_state_control.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/component_origin.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/component_tool_readiness.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/component_launch.py
  - build_tools/upstreams/native/circt-state-dependencies.patch
---

# Independent component feedback authority

Phase 1 authors a new compiler from the admitted public specification, independently
derived hardware facts and independent examples. Phase 2 optimizes that compiler.
The handwritten implementation supplies a sealed final performance baseline only.
Its lowering, harness, scheduling, decoder, compiler tables, artifacts and structural
answers are excluded from both phases.

## Separate authorities

`ComponentBaselineAdmission` binds the actual fresh Phase 1 origin, ordinary
functional qualification, frozen compiler bytes, selected target and generated
development cohort. CCA, analytical and RTL comparison services share this admission.
A hashed compiler directory or a JSON `qualified` label cannot replace its issuer.
Held functional and transfer members are not development feedback inputs.

`IndependentComponentRuntime` binds independently qualified runtime and semantic
support. Its fixed qualifier must execute actual controls before live admission.
The runtime source/tool membership, callback code, target descriptor, hardware intake
and qualifier are reopened on use. An independently authored compiler cannot certify
its own measurement callback merely by supplying it.

The runtime has separately qualified roles for ordinary grading, source-to-executable
stage/effect verification, analytical feature observation, CCA observation and RTL
execution. Correctness roles can become available before performance roles. Missing
roles refuse tool admission. No selected compiler-provider backend factory is used
by the independent provider constructors.

## What the runtime qualifier must establish

| Obligation | Required evidence |
| --- | --- |
| Instruction semantics | Decode predicates, operand fields, effects and forbidden instruction roles derived from the selected public RTL |
| Kernel execution protocol | Independently derived memory, ABI, entry and completion facts |
| Tool correspondence | Exact independently selected compiler, linker, runtime and simulator sources and their produced binaries |
| Isolation | Actual namespace controls proving the candidate cannot read private support, answers or final references |
| Functional authority | Normal candidate build/run records, complete original output comparisons and evaluated positive/negative semantic controls |
| Cost authority | Independently checked timer boundaries, cold/warm conditions, engine equivalence and held calibration qualification |

Structural FIRRTL intake establishes only the facts it actually derives. It cannot
grant instruction semantics, runtime correctness, bitstream identity or cycle timing.
Public SDK simulator code can supply independent execution semantics when its source
and build correspondence are established; a pre-existing shared object remains
unqualified until that correspondence is observed.

## How CCA guides optimization

CCA compares actual emissions from the fresh functional baseline and the current
candidate on the same generated source members. It exposes observed representation,
ownership, lifetime, resource, dispatch, synchronization and compiler-choice changes.
Missing observations remain `UNKNOWN`. The agent can use those facts to identify the
compiler pass or schedule owner responsible for a change, then test its effect with
independently calibrated costs and ordinary correctness checks. CCA does not supply
the handwritten implementation's choices or establish a measured speedup.

## Current implementation boundary

The historical normal-executor and selected-provider analytical factories have been
removed. The independent provider constructors accept only live runtime authority and the fresh compiler
baseline. No runtime qualifier is inferred from metadata or a support role label.
Ordinary execution accepts only the fresh seed or an exact snapshot of the current
controller-owned candidate, checked against the frozen edit authority whose seed
reopens the original fresh Phase 1 qualification.
The calibration adapter, its exact observed evidence, model dependencies and held
report must all belong to independently qualified support. Pinned external files
cannot introduce final-reference or historical optimization coefficients.

The fixed functional qualifier invokes fourteen positive and negative controls for
source correspondence, the complete original output roster and numeric gate,
instruction auditing, ownership, synchronization and hardware/runtime binding. It
reopens their actual source, process, executable and output evidence. These controls
grant only ordinary grading and semantic stage verification. Performance feedback
roles require independent measurement qualification before admission.

The separate measurement qualifier executes fourteen positive and negative controls
for source/executable, hardware/timer, complete stage accounting, cold/warm regimes,
feature observation, CCA correspondence and applicability scope. It consumes only independently prepared
methods belonging to the live functional runtime. Missing physical or timer
producers persist `UNAVAILABLE`; they cannot borrow authority from native functional
transport or static instruction counts.

Held measurement uses immutable, issued snapshots of allowed edits to the fresh
compiler. Every variant of a generated workload remains in the same preregistered
family group, and every admitted workload/variant pair is observed. Both cold and
warm screens must meet the existing ranking, error, coverage and evidence minima.
Calibration coefficients are fixed before held observations and the private
producer must reopen their independent provenance. The resulting authority binds
the exact fresh baseline, cohort, calibration and full timer scope.

The provider-independent artifact observer reopens actual controller-owned invocation
records, joins their exact products and measures static ELF section bytes. Its native
compile/run controls exercise that path. These observations are diagnostic: missing
instruction roles, complete semantic/effect checks and every unobserved cold/warm
stage remain `UNKNOWN`. They do not qualify a target or an experimental feedback tool.

## Launch assembly

`load_component_launch_inputs` consumes the live fresh Phase 1 origin and the
live runtime issued after independent complete measurement qualification. It
reuses the qualification, fresh baseline, development cohort, hardware owner,
unique candidate execution owner and frozen edit authority already bound by those
capabilities. The candidate must still match the original functional qualification
before a new authoring stage. Constructors and serialized reports cannot supply
these owners.

The closed v1 declaration supplies absolute, canonical paths. `candidate`, `view`,
`corpus`, `descriptor`, `source_root`, `contract_root`, `qualification_root` and
`edit_authority_root` must name the existing owners exactly; `edit_contract` names
the edit owner's `compiler_edit_authority.json`. Runtime and control rows contain
`source`, `destination` and `sha256` and must match the exact approved fresh Phase 1
closure. Codex and the private authentication source retain that closure's original
selection. Readiness probes use only admitted tool destinations and are actually
executed by launch qualification; a declared expected output is not a passed probe.

`inventory_runtime` accepts optional explicit dependency prefix relocations for
trusted tools whose loader paths belong to a selected SDK. The prefixes define
coordinates, not tree grants: only individual dependencies actually reported by
the selected executable are inventoried. The inventory preserves needed aliases
and layout, rejects escaping or overlapping mappings and destination collisions,
and pins every file before grants are constructed. A system destination does not
establish source independence or complete loader closure; actual tool execution
and separate admission remain required.

Fresh Phase 1 selects and pins its outer author sandbox explicitly. It cannot
substitute the nested native tool sandbox by destination name. Before authoring,
shared readiness uses a read-only inert candidate, and native readiness executes
the admitted commands through the selected client's actual tool profile. A
disposable exact scaffold copy accommodates temporary sandbox mountpoints;
original source files remain read-only, and new persistent files, directories or
symlinks refuse readiness. Complete original inputs and both scaffold copies are
rechecked before a model request. These credential-free controls establish tool
readiness only, not a fresh compiler origin or functional runtime qualification.
Phase 2 reopens that exact fresh origin and inherits its original outer sandbox
selection and complete control membership. A launch declaration or a nested
helper's destination name cannot supply a replacement launcher.

The opt-in [native dependency patch](../../build_tools/upstreams/native/circt-state-dependencies.md)
prepares clocked termination operands in one exact public Arc lowering revision.
It requires explicit source-hash checking and selection. It supplies no runtime
default or author grant. Assertion behavior, reset, program loading, complete
RTL equivalence and timer authority remain separate required controls.

The analytical declaration contains only `calibration_adapter`, `qualification`,
`objective`, `max_workers`, `memory_per_worker_bytes`, `engine_slots`, `output` and
`lease_path`. Calibration and the selected cold or warm held screen must match the
measurement owner. Output paths are fixed to `stage_root/analytical` and
`stage_root/worker-leases.json`. The stage must be fresh and disjoint from public
inputs and private evidence. The price table stays a private host input. Assembly
constructs analytical and CCA providers directly from issued callbacks and their
shared admission, then verifies `ComponentLaunchInputs`. It does not import a
factory, grade a compiler, execute a client or create a stage.

This prepares the route that qualified owners can consume. Current independently
prepared support still lacks complete physical semantics and timer controls, so
these measurement roles and an actual authoring launch remain unavailable. The
launch assembly fixture is explicitly diagnostic: its substituted owners are
unissued and the unchanged production verifier rejects them.

## Independent GSIM assessment

A clean build of public GSIM revision
`210689a170b559f1761efb739c71c9d71fb4bed2` succeeded using official Clang 19.1.1
packages extracted into an owned build directory. The public checkout remains
unchanged; local simulator adaptations and previous binaries were excluded.
The actual build and tool-help processes retain source, executable and output pins.
This establishes tool readiness, with target execution and timing still unknown.

Actual generation of the selected elaborated `FAMETop` FIRRTL found two front end
compatibility failures. Its FIRRTL 1.2 full connects (`<=`) are rejected by the
current public parser. A native CIRCT import/export roundtrip emits modern
`connect` syntax, but GSIM then rejects lowered `mem` declarations. Its
[public memory grammar](https://github.com/OpenXiangShan/gsim/blob/210689a170b559f1761efb739c71c9d71fb4bed2/parser/syntax.y)
accepts CHIRRTL `cmem`, `smem` and memory ports. The selected input has explicit
memory latency, mask and read-under-write semantics that must be preserved.

Two native CIRCT alternatives were observed. `firrtl-mem-to-reg-of-vec` leaves
lowered memories, so it does not close the compatibility gap.
`firrtl-lower-memory` produces `firrtl.memmodule` operations that the selected
FIRRTL exporter refuses. None of these transformations has independent structural
or numerical equivalence authority merely because its process returned successfully.
A newly normalized source needs a selected production receipt and matched hardware
closure; altered memory semantics additionally need native equivalence or complete
independent controls.

A separately generated two-clock counter control compiled and executed through the
clean public engine. Both counters advanced on each `step()` with both clock inputs
held low. This agrees with the
[public step emitter](https://github.com/OpenXiangShan/gsim/blob/210689a170b559f1761efb739c71c9d71fb4bed2/src/cppEmitter.cpp),
which advances the model and its global step counter. A simulator step is therefore
not evidence of an independently driven clock edge or a matched target timer.
A single active clock mapping could qualify only after its complete state and
external-module closure is derived and checked.

The remaining generic interfaces are concrete:

| Interface | Required behavior before authority |
| --- | --- |
| RTL input capability and normalization | Bind the input dialect level and operations, actual transformation tools and source products; preserve memory latency, masks, initialization and read-under-write behavior |
| Simulation clock and reset scope | Derive all active domains and reset sequences; qualify the mapping from engine steps to selected target clock edges |
| Program loading and observation | Audit the complete ELF before execution, load its declared segments through a source-bound memory or bus protocol, preserve zero initialization, and collect every original output |
| Platform effects | Independently check external modules, bus handshakes, completion, ownership and synchronization; target-specific implementations stay in independently qualified OOT support |
| Timer and cold/warm scope | Bind loading, setup, computation, output materialization and reuse boundaries to actual measurements; wall time and static counts cannot issue cycle roles |

The existing analytical worker leases already permit bounded CPU parallelism after
an engine and its feedback roles qualify. Parallel execution does not replace any
of these gates. No selected target ELF was executed in this GSIM assessment, no
performance role was issued, and the actual private numeric controls remain waiting
for a compatible independently qualified runtime route.

## Source-pinned engine readiness

`rtl_state_control` renders a bounded private transport driver from an actual
native-produced scalar I/O layout and a complete original stimulus roster. The
selected compiled engine implements state, memories and clocks. The driver only
writes declared input bytes, calls the explicitly named evaluation function and
reads every declared output. It does not infer clocks, reset sequences, cycle
counts, boot/loading protocols or target semantics. Unsupported native layouts
refuse, including lifecycle interfaces beyond the current scalar format.

Small original memory/clock controls exercise read latencies, write enables and
masks, independent clock edges, reset and complete readbacks through selected
stock tools. A selected-version register-vector conversion is rejected when it
changes the original no-edge readback. These controls qualify their observed
transport behavior only; whole-target preparation, physical correspondence,
program loading, ABI and timer authority remain separate requirements.

The experiment now has a generic private RTL control executor. An operator selects
explicit support sources, tools and producer records with
`prepare_rtl_engine_selection`; a `RtlProbeControl` declares original source bytes,
bounded argv steps, input/product members and every numeric output sample.
`probe_rtl_engine` executes those commands, retains ordinary invocation records,
checks unchanged sources and joins generated artifacts before native execution.
It never discovers an engine, loads a supplied PASS receipt or rewrites RTL.

The live issued `RtlEngineProbeAdmission` reopens all source/tool/product/output
pins and complete private evidence membership. `require_controls` refuses a
missing or failed required control before a costly candidate run. `readiness`
separately exposes input grammar, memory semantics, clock/reset and program
loading; a contract with no actual controls stays UNKNOWN. Partial contract
success does not suppress another control's refusal. This checks the selected
private controls only. Target runtime, hardware equivalence and target timer
authority remain independently qualified obligations.

Actual clean public-engine controls passed modern connect generation, global
step/reset samples including a reset after progress, and complete CHIRRTL memory
write/read/overwrite/boundary-address/reuse samples. Legacy full connects and
lowered memories with explicit latency/read-under-write were rejected before
native compilation. A held-low clock control failed the complete original
no-edge output oracle. Program loading has no control and remains UNKNOWN.
These outcomes establish an actionable compatibility boundary; they do not
establish selected target execution or performance.

## Scale and reuse applicability

The measured cost path requires a frozen joint semantic domain. Each cell binds
payload and unit meaning, live and working-set bytes against selected capacity,
tile shape/counts and tails, streaming, dependency depth, reuse state/count and
composition length, alongside exact hardware, numeric, input and timer identities.
The independent context must rederive those semantics from its pinned sources and
actual ELF derivation products. Data constructors and JSON files cannot issue roles.

Each required stratum declares held transfer groups independently of validation
frequency. Every joint cell must be observed in every declared held group, with
complete original output gates and matched complete cold/warm timers. Both global
screens and every stratum/regime must pass the fixed ranking, error and coverage
policy. Calibration groups remain disjoint from held transfer groups.

Applicability is exact joint-cell membership. An unseen size, tail, capacity,
streaming, dependency, reuse or composition combination remains UNKNOWN even if
its individual coordinates were observed separately. Inclusive cost arithmetic
retains region estimates while refusing complete totals outside that domain.
Range and repetition proofs are future independent authorities; this gate does
not infer them or admit target performance. The current prepared context lacks
the physical applicability and timer producers, so measurement roles remain unissued.

An optimizer may propose a new legal tile, streaming or reuse configuration that
is absent from this domain. UNKNOWN is a request for evidence, not a permanent
prohibition of that edit. A separate bounded evaluator route must admit the exact
fresh-compiler variant, original complete outputs and independent source/HW/timer
semantics, then requalify a declared domain extension against the fixed held groups
and acceptance policy. Neither a new coordinate nor a fast estimate can authorize
it. Final validation remains sealed; no final workload or golden result may select
these development cells, change their weights or supply calibration labels. This
measurement/requalification route is not implemented by the exact-cell gate.

The direct analytical thread coordinator forwards each remaining timeout and rejects a
result after the development deadline. It cannot forcibly interrupt a callback
that ignores that timeout, and executor shutdown waits for active threads.
The wrapper therefore supplies cooperative budgets and late-result refusal;
normal component workflow additionally forks callbacks through `bounded_feedback`.
That worker wrapper attempts process-tree cleanup with descendant snapshots.
It does not establish persistent descendant ownership, an independently retained
cleanup receipt or parent-owned lease release. A child that reparents before a
snapshot and parent-side result decoding require separate supervision; the direct
thread coordinator does not prove those bounds. A complete hard deadline needs
an admitted process/lease lifecycle with bounded result transfer and explicit
cleanup and release evidence. Late or incomplete results cannot become authority.
The current context has no qualified physical measurement roles. This limitation
does not relax their controls or turn a stale result into qualified feedback.
