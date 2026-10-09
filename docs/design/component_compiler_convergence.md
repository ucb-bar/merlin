---
title: Component-driven compiler convergence and experiment isolation
kind: design
status: draft
owner: experiments
last_verified: 2026-10-09
related: [agentic_experiment_integrity, capsule_phase_split, perf_phase2_wiring, phase0_specification]
code_refs:
  - packages/merlin-experiments/src/merlin_experiments/phase0/component_automatic.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/component_automatic_plan.py
  - src/merlin/targetgen/frontend_use_def.py
  - src/merlin/targetgen/frontend_operator_effects.py
  - src/merlin/targetgen/torch_schema_observer.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/operator_schema_intake.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/arithmetic_intake.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/component_arithmetic_obligations.py
  - src/merlin/targetgen/rtl/hw_arithmetic.py
  - src/merlin/targetgen/compiler_library.py
  - src/merlin/targetgen/package_runtime.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/component_experiment.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/numerical_readback.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/component_final_policy.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/protected_final_evaluation.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/resource_boundaries.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/component_semantic_basis.py
---

# Component-driven compiler convergence

The experiment asks whether a fresh target compiler can reach an independently
frozen handwritten reference's quality through generated component tests, with
substantially lower authoring and measurement cost. It uses reviewed shared
compiler infrastructure. That infrastructure's prior development history must
be disclosed; upstream publication alone never grants experimental access.

## Implemented boundary primitives

`CompilerLibraryContract` freezes exact reviewed Python modules, namespace
initializers, resources, direct shared-library imports and their bytes. Public
APIs are leaf modules; approving a module does not approve sibling or child
modules. Withheld evaluation identities remain forbidden. The host supplies the
contract and root explicitly to `integrity_scan`; a candidate manifest cannot
grant itself access. Legacy scans retain their existing input-grammar exception.

This is direct-import and byte admission, not proof of semantic generality,
dynamic dependency closure or complete interpreter isolation. A reviewer must
establish the public implementation's generality and dependencies. Runtime
isolation is required independently.

`materialize_component_view` copies explicit members into a fresh minimal view.
Its public manifest contains logical roles and content hashes, not original
private source paths. Only approved compiler members, contracts, generated
inputs and toolchain files are selected. Repository history, session state,
reports and test trees are excluded. Reverification rejects changed bytes,
symlinks, special files and added files or directories.

The Phase 0 generation digest binds an independently admitted receipt; a hash
alone does not prove that inputs were generated. Keep the existing Phase 0
derivation/evidence verifier authoritative.

`strict_tool_policy` constructs a networkless, clear-environment subprocess
boundary with explicit runtime-file mounts and a read-only minimal view. Authoring
keeps its candidate writable; readiness and runtime execution can explicitly
select `candidate_writable=False`. The selection requires a boolean and retains
the same inventory and source-symlink checks. It does not mount a whole checkout,
home or system tree. A native host syscall proxy can read any mounted file even
for a freestanding ELF: execution must select only its immutable ELF/data
workspace and public tool dependencies, with private grader/reference files
outside every mount. Process containment alone grants no target runtime role.
`run_isolation_probe` refuses failed/unavailable execution. A successful probe
does not qualify a model-client authoring transport or establish that every
private surface was checked. A formal campaign needs actual sandbox probes,
fresh-session admission and independent protected-evaluation execution.

## Publication and experiment admission

Publish reusable generic compiler/runtime changes in the shared compiler;
publish capture fidelity changes in the importer; publish target instructions,
physical schedules and ABI code in target support. Preserve rejected policies
and their evidence without enabling them as defaults.

Then review an explicit experimental dependency subset. Do not expose the
handwritten target compiler, reference programs, validation captures, weights,
goldens, layer census, tuning transcripts or performance journey to authors.
Do not expose Git history or enable network retrieval of excluded publications.
Start fresh agents with no inherited investigation context.

Keep graders, reference engines, holdouts and private receipts outside agent
read/edit authority. Diagnostic output from those owners must not expose
expected values, private IR or workload-specific selection guidance. Prove the
boundary from inside the actual sandbox, including denied reads/imports,
escaping links and network retrieval. No unavailable probe is a pass.

## Phase 0 generates the experiment

Derive training, calibration and hidden tests from pinned hardware/software
contracts and independent derivation workloads before backend authoring. Use
the existing capsule generators and commit/reveal protocol. Do not create
handwritten workload-layer fixtures or fit generation weights to validation
layer frequencies.

Required families include contractions, convolution, batching, layout changes,
tails, resource limits, quantization, mixed precision, ordered reductions,
multi-consumer observation, nonlinear operations, producer-consumer composition,
immutable reuse, effects, ownership, lifetime, publication and fallback. Include
zero-MAC work with explicit falsifiable cost objectives.

A trusted auditor may inspect validation graphs after generation. Its findings
cannot revise the corpus, supply exact validation shapes to authors, or choose
optimization actions. Frozen source coverage and numerical qualification are
different obligations; a small coverage witness basis proves neither arithmetic
nor a complete compiler.

### Reviewed component obligation plans

The ordinary component coverage option also accepts a closed
`merlin.component_automatic_policy.v1` containing only independently selected
hardware, software and source-basis identities and finite generation/execution
budgets. It cannot contain authored obligations, shapes or topologies. The
generator strictly replays original result ownership and all typed args/kwargs
edges before deriving interaction class presence. Supported classes produce
fresh bounded contraction and copy programs through the ordinary writer and
complete independent output oracle. Original shapes, graph sizes and frequencies
do not select generated dimensions.

Automatic derivation currently supports only a subset of source forms and
interactions. Every unsupported original operator or interaction, effect domain
and unresolved RTL resource/axis role stays a mandatory unavailable obligation.
The resulting plan remains incomplete while any required row is unavailable.
The protected report rederives the roster from exact selected original sources;
re-signed metadata cannot remove required unknowns or replace the fixed generic
semantic factories. Relocated automatic source replay is not qualified yet.

Automatic policy v2 additionally binds a live public operator-schema intake to
the same independent software origin. The fixed native observer joins captured,
registered and clean tracked source schemas. Exact original argument/result
bindings derive possible direct-tensor alias and write classes. A uniquely
supported may-alias movement selects bounded logical view/copy cases with all
input, view and copy outputs checked. Operator names do not supply effects.
Container, wildcard and changing aliases, actual physical alias/ownership,
non-schema effects, whole-effect completeness and hardware axis/resource roles
remain mandatory gaps. Schema equality does not prove the installed framework's
historical build correspondence. Policy v1 keeps its previous unknowns.

Automatic policy v3 binds a separate live arithmetic intake to the same original
hardware. A fixed native generic serialization and typed SSA reader follow exact
sign extension, multiplication, low input slices and modular addition to original
module outputs. Local relations retain conditional numerical compatibility gaps;
they do not establish a selected contraction, reduction count or order, instruction
semantics, physical resource role or tensor-axis map. Unrecognized outputs and
cross-state or cross-instance behavior remain mandatory unknowns. Source numerical
policies and complete independent output checks remain unchanged.
When schema effects are also selected, v3 binds their exact live intake in the
same ordinary derivation. Both complete source records are replayed; removing
either selected facet or its required unknowns cannot be repaired by re-signing
metadata. Policies v1 and v2 retain their original behavior.

Keep the selected software spec minimal: semantic and numerical behavior plus
operation support that the selected RTL cannot determine. Extract hardware
capabilities, geometry and resource counts from the pinned RTL. Independently
select and freeze any example training graph roster before authoring; inspect
those source-pinned graphs for relevant semantics, separately from protected
validation. A later validation audit cannot revise the frozen corpus. Layer
frequencies, validation shapes and past timing never choose production rules or
generation weights.

Campaign `component_performance` objectives may be declared in the explicitly
selected recipe; their exact source hash, reviewed operation links and selected
hardware binding are recorded by normal generation. This keeps campaign metrics
outside the minimal software spec. Legacy inline declarations remain readable;
supplying both recipe and SW declarations is refused. Review status is a
declaration, not independent review authority or measured performance evidence.

The explicit recipe may also select
`semantic_basis: {path: <reviewed roster>, sha256: <exact roster bytes>}`.
`ComponentSemanticBasis` validates the closed
`merlin.component_semantic_basis.v1` roster with `status: reviewed` and
`provenance: {role: independent_training_example, visibility: public,
selection: before_authoring}`. Each member declares `id`,
`kind: model2mlir_frontend_trace`, `path`, `sha256`,
`schema: m2m.frontend_trace.v1`, `operation_semantics` and `effect_semantics`.
The original graph content digest, typed graph edges and actual call target
roster must agree with the pinned bytes and reviewed operation list. Reviewed
effects contain `id`, `kind`, `basis` and a nonempty subset of those operations.
Source bytes marked with other provenance, unreviewed rosters, changed files and
authored weights are refused. Source path spelling grants no authority.

The existing private source freeze archives complete selected graph bytes before
authoring, rechecks roster membership and content, and routes generation only to
the archived files. Author-visible identity carries reviewed semantics and hashes;
graph paths, shapes, counts and frequencies stay in private audit inputs. The
coverage plan binds `semantic_basis_sha256`, and each obligation declares
`semantic_basis: [{member: <example id>, operations: [{source: <original call
target>, owner: <selected SW operation id>}], effects: [{source: <example effect
id>, owner: <plan effect id>}]}]`. Every operation owner needs a reviewed source
correspondence. Effect links must match kinds and still require concrete generated
witnesses. These correspondences never infer production lowering rules or certify
physical effects. Protected independent selection supplies review authority;
an author's review or provenance label cannot grant it.

The ordinary `generate_target` API accepts an external `component_coverage` path
only with `component_only=True`. The Phase 0 CLI spells it `--component-coverage`;
the experiment orchestrator spells it `--phase0-component-coverage` and freezes
it with the other explicit inputs. The closed `ComponentCoveragePlan` schema
binds exact selected hardware, software and numerical declarations. A declaration's
`status: reviewed` does not grant review authority: the protected experiment owner
must independently admit that exact source before authoring.

```yaml
schema: merlin.component_coverage_plan.v1
status: reviewed
hardware:
  contract_sha256: <selected canonical contract SHA256>
  raw_facts_sha256: <selected raw facts SHA256>
software_spec_sha256: <selected software source SHA256>
numerical_semantics_sha256: <canonical selected numerical semantics SHA256>
effects: []
budget: {max_members: 64, max_interaction_cells: 2000}
obligations:
  - id: rectangular_transfer
    mandatory: true
    cohort: functional_guard
    operations: [movement] # IDs in the selected reviewed software declaration
    effects: []
    expectation: admitted_program
    frontend: mlir
    base: {op: movement, kind: isa}
    axes:
      M: {kind: extent, values: [tile, tile+1]}
      N: {kind: extent, values: [tile+2, 2*tile+3]}
    interactions: []
```

Axes are explicit positive extents, JSON choices, or resource declarations.
Nested builder parameters use `{axis: NAME}` substitution. Every singleton and
pair projection is covered deterministically; explicitly declared interacting
axis groups receive exhaustive projection coverage. The algorithm extends
missing projections without enumerating the full Cartesian domain. Exceeding
either selected budget records missing coverage and fails mandatory obligations.
Malformed declarations fail before generation. Missing mandatory facts, builders,
effect witnesses or sealed capture runtimes leave an unavailable obligation.

Development members remain normal `dev` performance members with reviewed
objectives. Functional guard members and withheld transfer members are separate
cohorts. Every generated member uses the normal builder, writer, concrete
operation admission and independent full-output golden. An `unsupported_program`
obligation needs an explicit concrete refusal from every possible selected lane;
missing host support declarations are unknown, never an established refusal.
The Phase 1 compiler must independently produce its own appropriate refusal.

Normal MLIR builders and the closed builtin PyTorch programs are selectable;
arbitrary model loaders, captures, application selectors and authored program
imports are excluded. PyTorch generation requires the existing source-bound
sealed Model2MLIR runtime. The generic `component_program` builder adds signed
integer matmul, add, copy, transpose, logical alias and update DAGs. Its independent
mathematical evaluator supplies exact modular-width outputs, including escaped
intermediates and before/after epoch snapshots. Logical aliases are functionalized
into SSA; physical aliasing, allocation lifetime, invocation epochs and
synchronization still need separate execution witnesses. Existing integer golden
functions retain their arithmetic and ordering.

Private `_evidence/coverage/component-coverage.json` binds the selected declaration,
generator and selected evidence sources, actual emitted programs, capsule bytes,
concrete admission, independent golden source and complete output roster. Each
obligation is `generated`, `verified_refusal` or `unavailable`. `verify_report`
rechecks those bytes before admission; relocated freezes separately verify the
archived source closure. The public projection exposes only hashes and cohort
counts. Withheld shapes, graph selectors and source paths stay private. The v2
functional guard link enumerates exact functional members without asserting that
a compiler has passed them. Existing freeze and commit/reveal remain authoritative.

`declared_resource_boundary` extends the existing allocation declaration with
an explicit `resource` name for banks, accumulators, transfer segments, alignment
or caches. `capacity_fact` and each reservation select byte counts relative to
the selected refreshed facts body. It derives below/last-fitting/first-overflow
and declared tail points from simultaneous typed allocations; row layout is
applied only by the existing physical resident-store declaration below. Neither
frontier establishes resource placement, cache behavior or target execution.

### Numerical input palettes and connected programs

An external obligation may select `input_palette` independently of its numerical
engine. Patterns are deterministic, target agnostic, and derive format landmarks
from the shared dtype registry and codecs. They select an actual tensor axis or
row-major linear traversal, and reject values that would be rounded during input
encoding or patterns whose axis cannot contain every declared value.

```yaml
input_palette:
  schema: merlin.component_input_palette.v1
  inputs:
    - {name: Q, axis: -1, values: [cancellation_large, one, -cancellation_large], offset: 0}
    - {name: K, axis: linear, values: [one], offset: 0}
```

Names select normal builder inputs; `arg0`, `arg1`, etc. select an input position,
and `*` explicitly selects remaining inputs. Available floating landmarks include
`zero`, `one`, `min_subnormal`, `min_normal`, `max_finite`, `half_ulp_one` and
`cancellation_large`, with an optional negative sign. Signed integer storage adds
`min_signed` and `max_signed`. Finite numeric constants must be exactly representable.
Signed zero keeps its sign in narrow-format byte encoding. Infinity requires a
format and existing codec that represent it losslessly; existing NaN input
encoding remains refused. Block-scaled inputs need separately typed scale streams
and are unavailable through this scalar palette path.

The ordinary integer leaves, SIMT input synthesizer, SpecIR operand synthesizer,
and closed builtin PyTorch loaders consume the selection. Existing golden
arithmetic, reduction order and acceptance rules are unchanged. Explicit stress
patterns may intentionally repeat values; ordinary addressing/stride corpus members
retain their existing rigor checks. The private report checks full captured or
independently synthesized input values, including the sign of zero, before
crediting an input-effect obligation. `cancellation_input`, `rounding_tie_input`,
`signed_zero_input`, `subnormal_input`, `signed_input`, `maximum_finite_input` and
`nonfinite_input` describe source operand witnesses, not accepted transformations.
FENV, arithmetic ordering and compiled numerical semantics remain Phase 1 obligations.

The selected numerical declaration may explicitly add
`input_domain: {nonfinite: forbid}` and/or `output_domain: {nonfinite: forbid}`;
`allow` is the other supported declaration. The normal concrete-program screen
then observes every source input and independent reference output. A forbidden
nonfinite input or overflowed reference result establishes a source domain refusal;
missing full observations are unknown. This changes neither candidate numerical
comparison nor the meaning of a generation-time refusal. Absence preserves the
historical numerical behavior.

| Reviewed optimization family | Ordinary generated source | Concrete source mechanism and scenario axes |
| --- | --- | --- |
| Contraction/order/scale | MLIR matmul, attention QK, resident reuse; PyTorch reduce sum/embed scale | Rectangular M/K/N, reduction-axis cancellation/ties, explicit accumulator scales and immutable shared weights |
| Attention and masks | PyTorch attention full / attention residual norm | Rectangular Q/K, scaled QK, rectangular causal mask, softmax, PV, residual and layer normalization |
| Normalization/nonlinear/MLP | PyTorch layernorm/rmsnorm/geglu / MLP residual | Mean/variance or squared mean, epsilon, nonlinear hidden activation, two contractions, bias and residual publication |
| Convolution/residual/pooling | MLIR conv/residual builders; PyTorch conv residual pool | Convolution with explicit input/kernel geometry, residual, ReLU and pooling; pool divisibility is validated |
| Observation/ownership | MLIR component program DAG | Multiple consumers, escaped outputs, logical aliases and independently observable before/after update snapshots |
| Shared products | MLIR component program DAG and resident reuse | Independently published products sharing an immutable weight or input, rectangular extents and separate consumers |
| Quantizer observer | Closed PyTorch producer quantizer observer | Producer escape, nearest-even rounding, signed i8 clamp/cast, dequantization and consumer product; complete producer/code/decoded/consumer outputs |
| Numerical domain refusal | The same ordinary source builders | Explicit selected finite input/output domains, nonfinite inputs or overflowed full reference outputs |

Each reviewed matrix selects its own obligations, available normal frontends,
interacting axis groups and finite budget from the pinned declarations. These
families do not select production rules from a workload census or historical
results. Normal PyTorch generation still requires source-bound sealed Model2MLIR
conversion and concrete operation admission; Torch eager source execution alone
does not qualify conversion, hardware or a submitted compiler.

`producer_quantizer_observer` accepts explicit positive, exactly represented f32
`producer_scale` and power-of-two `quant_scale`. Bounds are the shared signed i8
format's range. Select input palettes with half-integers, values beyond those
bounds and negative zero; the independent private source witness counts actual
ties, clamps and negative-zero producer values. It rejects changed rounded codes
or decoded values using exact rational nearest-even projection independently of
Torch. Ordinary multiple-result publication binds every ABI result to the parsed
program signature and its full golden tensor. The source observer never enables
an arbitrary quantization recipe, loader, model import or unchecked scalar text.

Its code observations are published as f32 values to retain the ordinary single
homogeneous output policy. Exact source observations do not silently change
candidate acceptance. Exact code/sign-zero qualification needs an independently
reviewed zero-tolerance/sign policy or a reviewed per-output typed comparison;
the latter is not supplied by this source builder. A palette of ones for the
consumer weight can keep finite code sums exactly representable when a reviewed
exact f32 comparison is selected.

Convolution frontier tests generate actual rectangular source windows with
stride, asymmetric padding and dilation, cross an explicitly selected byte
capacity, and include aligned and tail extents under two synthetic geometries.
Their full outputs are checked by direct independent source-window sums.
Capacity inequalities establish declared demand, not physical allocation success.

Persistent immutable invocation context, repeated-invocation input mutation,
physical ownership/lifetime/alias epochs, and dynamic rounding-mode save/restore
need executor interfaces that bind state across actual calls and observe its
publication and FENV. These are not implemented by functional SSA alias/update
nodes or by a Torch eager call. Mandatory obligations naming unsupported effects
remain unavailable; compiler execution and repeated-call evidence belong to the
protected Phase 1 owner. MX scale-stream palettes and NaN payload encoding also
remain unavailable through the scalar palette interface.

### Declared resident size boundaries

The normal Phase 0 sweep resolver accepts `resident_allocation_boundary` on an
extent axis. An external shared template declares one named physical store,
`capacity_fact` and `reservation_facts` paths relative to the selected refreshed
facts body, a `quantum` using the existing tile grammar, and explicit integer
`tail_offsets`. All referenced counts are bytes. Capacity must name that store's
actual `memories[index].bytes`; each reservation must be an exact physical row
multiple. Missing or unsized selected facts produce an explicit unavailable
derivation, independently of malformed declarations, which stop generation.

Each named simultaneous allocation declares a `shape` of axis names or extent
tokens, a `dtype` (or `operand` for the selected operand type), positive integer
`copies`, and optional `round_up` dimension indices. Rounded dimensions use the
declared quantum; minor-axis physical row rounding uses the existing generic
address-space implementation. The resolver sums every allocation plus declared
reservations in that one store. Separate address spaces require separate
declarations and are never combined into a fictitious capacity.

Generation derives a positive aligned point below the boundary, the last
fitting aligned point, the first overflowing aligned point, and declared tails
around the latter two. A last fitting point need not exactly fill the capacity.
An unavailable below point is recorded rather than clamped to zero or copied
from another point. Distinct tails remain separate even when padding gives them
the same footprint. An offset outside the resolved quantum records an unavailable
tail while retaining aligned boundary points; an empty offset list explicitly
declares no tail coverage. Each emitted member retains the selected fact identities,
declaration, allocation footprints and aggregate inequalities. Normal builders,
interfaces, software screens and independent golden engines still execute;
the existing sweep member limit remains in force.

This implements a declared size obligation. It does not infer allocator
placement, live ranges, cache state, profitability or hardware admission.
Consumer/reuse/effect/publication/ablation scenario generation and comprehensive
supported-domain coverage remain separate work; this axis does not establish
those obligations or a generated compiler's correctness.

## Phase 1 establishes a functional compiler

Declare the supported domain and refuse unsupported cases. Join every source
compute/support operation to its lowering, placement, transfers, linked symbols
and executed artifacts. Include host nonlinear and support work in the census.

Compile and execute through normal installed APIs outside the source checkout.
Qualify numerical order, scales, effects, ownership, resources, complete writes,
host/device composition and final linking. Inspect all executable sections for
prohibited target instructions. Frozen public and hidden generated obligations
must pass without skipped members disappearing from the denominator.

Record effective ordered compiler stages and emission witnesses. Replacing a
transform must not discard a previously selected legalizer. A feature name or
constructor toggle is not evidence that the transformation reached LLVM/object
code. Generated tests establish a bounded supported domain, not correctness for
every future graph.

Keep numerical execution bounded by work and storage. Large independently
generated shapes need ordinary compilation through linking and independent
static legality checks, while small synthetic graphs test composition and reuse.
These are distinct evidence classes: compile-only success cannot count as a
numerical execution pass. The proposed protocol and missing implementation
extensions are in [scale generalization](component_scale_generalization.md).

## Phase 2 uses complete components

Give authors separately reviewed shared-compiler and target-support edit
surfaces. Protect hardware facts, numerical gates and evaluation owners. Use
typed schedule/resource records, complete CCA comparison and explicit unknowns.
Expose actual emitted stages, correctness/refusal diagnostics, complete cost,
uncertainty, resource pressure and concrete editable owners.

Progress from static legality to functional tests, calibrated analytical
comparison, and small matching-configuration RTL measurements when ranking is
unresolved. Cache by complete dependency identity, reuse compilation, batch
independent candidates and stop rejected candidates early. Keep whole-model
analysis and simulation outside the component optimization loop.

Price preparation, device work, commands, transfers, CPU arithmetic, tables,
cache states, packing, readout, dispatch, refinement and fallback. Separate
preparation from invocation and warm from cold reuse. Calibrate independently
of validation workloads and reference totals. Instruction retirement is a work
feature, not hardware timing; one global instruction-to-cycle multiplier is
insufficient. Different configurations cannot share calibration silently.

## Final comparison and accounting

Freeze source, installed packages, runtime, toolchains and selected options
before private full-workload compilation and final hardware evaluation. Match
hardware, inputs, accuracy criteria and timer boundaries to the frozen reference.
The new final policy requires every workload to independently match or beat the
frozen handwritten reference cycles. An aggregate improvement cannot hide one
regressing member. Historical v1 receipts retain their original five-percent rule.

Convergence time is reported separately using a matched historical method. It
does not gate new final performance acceptance. Report reasoning, compilation,
simulation, queue waits, token telemetry, candidate count and cache reuse.
Report reusable setup/calibration separately and also report total cost including
them. Missing historical records remain unknown. Fix seeds, budgets, stopping
and measurement policy before authoring.

`strict_final_component_campaign_gate` implements the new integer parity and
separate-time arithmetic. The historical `final_component_campaign_gate` keeps
its v1 five-percent/twenty-times semantics. Neither helper authenticates hardware
flags, arbitrary receipt hashes or historical telemetry. The host-private
`admit_protected_numerical_readback` reconstructs a `QualityObservation` from
two full readbacks through existing V4 private ownership, typed build services,
readback build receipts and the exact original frozen `QualityBudget`. Both
selected recipes, renderer sources, codec, command buffers, kernel objects,
harnesses, executables and complete output rosters are rechecked. Exact policy
compares stored words, including signed zeros and NaN payloads; elementwise
policy decodes declared float formats and uses the unchanged tolerance over
every value. Nonfinite values violate the elementwise policy. Integers too
large for that float64 comparison refuse; task metrics require their own owner.

This numerical observation is not a hardware or execution certificate. The
trusted evaluator must supply its own original provenance and capabilities;
a candidate-created snapshot cannot establish reference validity or actual
execution. The final evaluator still must join its protected source/package
freeze, compiler invocations and emitted IR/objects/ELF to staged queue bytes,
actual image/configuration, original inputs, timer boundaries, reference recipe
and every campaign member. Existing queue logs alone do not establish these
joins. Missing reference recipes and historical telemetry remain unknown.
`protected_final_evaluation.evaluate_protected_final_campaign` now joins the
original private campaign roster and binding files, the independently selected
execution verifier and full numerical reconstruction for both performance arms.
Both arms use the original numerical oracle, avoiding tolerance doubling between
two approximate outputs. It reopens all private selected bytes and never accepts
precomputed success flags as lifecycle inputs. See
[protected final evaluation](component_final_evaluation.md) for the ownership and
qualification contract. Actual target verifier qualification and final hardware
admission remain separate prerequisites; file pins do not establish them.

Final holdout results are not tuning feedback within that campaign. Disclosing
failures for repair ends the campaign; later work records the exposure and a new
campaign. Existing frozen receipts are never rewritten to acquire stronger claims.

## Remaining implementation and qualification

These primitives are not a completed Phase 0-to-2 launch. Mandatory remaining
work includes the protected fresh authoring transport, complete approved shared
dependency selection, generated mixed-precision/frontier coverage, normal
compiler stage witnesses, independently calibrated component providers, and
content-bound final hardware admission. A campaign must remain unavailable
until those obligations and actual runtime isolation are established.

Keep a structured optimization inventory containing semantic applicability,
owner, source/package/object identity, independent cases, complete costs,
numerical results, positive/negative evidence, promotion state, missing phase
tooling and measured token attribution. Workload-specific inventory remains
excluded from the agent environment.
