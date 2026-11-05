# Joint compiler development priorities

Build one reusable compiler for the entire public roster: 153 original correctness
capsules, 20 calibration candidates, and 77 scalability/performance capsules.
The 153 originals retain their required Spike and gSIM checks. Calibration
candidates require a specific unresolved mechanism and matching numerical evidence
before selective gSIM; large scalability cases use Spike during development.
Required physical correctness is not waived by selective model calibration.

Develop capability families concurrently. A small end-to-end canary can cheaply
expose shared ABI errors; choose it when useful, without making it a prerequisite
for independent construction or tests. Admit useful eligible cases continuously as resources free.
All work contributes to the same compiler. Integrate helpers' reusable changes
frequently; evaluate immutable snapshots while authoring continues. Preserve
per-result revision and input identities. Do not wait for all 153 originals before
working on calibration prerequisites or large connected workloads.

Fresh experiments start with an empty target backend; declared checkpoint
continuations preserve and improve their existing compiler. The generic ISA generator is available as
`merlin.targetgen.isa_spec.resolve_isa` and
`merlin.targetgen.isa_dialect_gen.generate_package`; it is not an already
generated target dialect. Inspect the selected software specification for an
`isa` declaration. If absent, construct the declaration from admitted public
ISA/RTL sources before calling the generator. Preserve source provenance and
explicit unknown semantics/effects; do not invent fields from instruction names
or import any existing target compiler/extractor. Generated files are outputs of
this authoring run. Independently check encoding before using the generated ops.

Useful divisions include compiler boundaries: source semantics and graph transformations,
numerics/layouts, schedule/storage, instruction lowering, and host execution.
Choose ownership and typed interfaces that let the work proceed concurrently.
Keep an executable path while helpers expand independent capabilities; do not assign a separate
compiler implementation per capsule. The first target executable is an early
construction milestone: emission, target consumption, linking, Spike, and every
output comparison establish more than a successful build. Choose when to broaden
tests from the evidence available. Known shared ABI failures are a reason to avoid
repeating affected work while independent paths proceed.

Use bounded fusion discovery and pass alternatives, retaining a correct unfused
route. Inspect the realization receipt: logical schedules, encoded stage
composition and physically measured fusion are different outcomes. The current
connected helper exposes ordered dense/pointwise products and buffer bindings;
target guest launch and complete-output execution still need an adapter. Develop reusable missing mechanisms in
the editable compiler instead of treating an unsupported helper as an instruction
to drop layers, omit outputs or replace device work silently with CPU work.

The 77 scalability cases are the primary performance targets. Push strongly for
complete-workload performance while establishing correctness; use small originals
primarily as regressions and model evidence. Select hypotheses from measured or
explicitly provisional whole-SoC feedback and structural roofline gaps. Unknown
cost is not zero. Compare alternatives under the same model epoch and execution
scope. Use tuned estimates when admitted; label provisional estimates honestly.

Use the editable generic compiler library, scheduling representations, graph
transformations, fusion, allocation and host codegen where useful. Discover pass
ordering and shape/resource heuristics from typed IR and selected hardware facts.
`merlin.transforms.pipeline_policy.evaluate_variants` supports bounded independent
alternatives and a shared preparation prefix. It produces compiler artifacts,
not numerical or timing qualifications. Preserve numerical order, all live-outs,
legal overlap and storage lifetimes; minimize redundant preparation and boundary
work. Never branch on capsule IDs or encode fixed answers.

Use compact capability and API queries rather than copying complete catalogs or
logs into each agent. All 20 CPUs form one dynamically shared reservation; there
are no permanent role-specific pools. Follow any explicitly selected runtime budget;
this guidance adds no token threshold or integration reserve.
Keep detailed logs on disk and concise evidence-linked handoffs in context.

Own the development loop. Use up to 16 helpers within the declared ceiling when
independent work exists, with no minimum occupancy requirement. Construct and
improve independent compiler subsystems concurrently, then integrate into one
compiler. The evaluator is an asynchronous feedback service, not an approval
step for every source edit or pass invocation. While one immutable snapshot is
evaluated, keep developing the next. Ask for targeted evaluation when a changed
capability is ready; reuse current results for unchanged relevant code and inputs.
Do not repeatedly submit the entire roster while an equivalent request is active.

When the operator selects full compiler access, the writable generic compiler
lives under submission/compiler_library. Inspect, modify, reorder and extend its
transformations, dialects, analyses, allocation and host generation directly.
Develop reusable missing abstractions instead of waiting for another wrapper API.
Use bounded alternatives and shape/resource heuristics; each helper contributes
to the same compiler and returns a compact change summary plus relevant evidence.

Reach executable coverage across all 250 original cases, including required
physical checks, while improving the large-shape implementation from the start.
Once a family is correct, concentrate its performance exploration on the larger
capsules and connected models. Keep exploring measured bottlenecks after initial
correctness: fusion, preparation reuse, storage lifetime, tiling, overlap, command
overhead and whole-chain scheduling. A successful build or a completed author turn
is not completion of the correctness and performance objectives. If evaluation is
pending, continue independent compiler work and consume results as they arrive.
