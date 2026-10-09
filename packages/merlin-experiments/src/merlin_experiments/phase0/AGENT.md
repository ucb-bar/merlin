# AGENT.md — packages/merlin-experiments/src/merlin_experiments/phase0

Profiles declare tests; sweeps expand them; writer constructs and validates capsules;
numerics owns the independent golden engines. Generation orchestrates those owners,
and provenance records emitted inputs without exposing hidden member names.

Keep numerical function bodies and ordering stable during structural work. Inputs
are external: never bundle profiles, holdouts, or generated goldens in this package.
All runs require an explicit output destination; never default writes to the
descriptor's source corpus. Both checkout and installed runs require explicit recipe
inputs or a legacy profiles directory; the former in-tree recipe default is retired.
Installed runs also require an explicit descriptor.

`load_profile` / `generate_target` support explicit `recipe`, `performance_template`,
`synth_profile`, `smt_profile`, and `hidden_profile` paths. Recipe mode requires the shared
template and never discovers siblings; it cannot be mixed with legacy `profiles_root`.
Missing optional declared sidecars mean omission. Public readers use `include_holdouts=False`,
which must not even stat the hidden path. Merge order remains public, shared performance,
synthesis, SMT, hidden. Frozen orchestration binds these inputs and optional-file absence;
the loader does not invent another seal or target-discovery registry.

The cache and experiment runner must bind the complete package source closure, not
only the CLI or numerical entry function. No private-helper compatibility facade:
tests and maintenance callers import and patch the authoritative owner.

Capture-derived stages share ONE grouping (`compute_groups` + `group_command`, stated by core
`group_capsule_entries`) and ONE binding (`group_capsule_entries.group_binding`, the corpus binding
with a derived, never literal, accumulator). `model_forms` derives functional MF forms and
`form_perf` the form-perf scope, both at `corpus derive` from the declared ITERATION captures only;
held-out models (claim and evaluation-only, `claim_boundary.held_out_models`) are refused by name.
`group_forms.write_group_capsule` is the only path a group capsule reaches disk. `instruction_roles`
derives the role taxonomy and resolves the experiment's `prohibited_instruction_roles`; it declares,
it does not enforce. Form-perf members, their coverage and the claim-model statistic never read a
claim model's capture before the Phase-1 freeze, and never write a capsule from one.

`component_only=True` is an explicit independent-input mode of the same generator.
It accepts fresh shared dev sweeps and reviewed source-bound HW/SW objective
declarations. Optional external `ComponentCoveragePlan` adds bounded independent
functional and private transfer cohorts; it never imports model/capture/history
selectors. Ordinary MLIR builders and sealed builtin PyTorch programs use the
same writer, independent goldens and concrete operation screen. Plan review status
does not grant author-controlled review authority. Missing mandatory facts,
builders or witnesses block coverage; unknown host admission cannot establish a
refusal. The private report binds actual member/source bytes and exposes only
hashes/counts publicly. Global domain completeness, candidate numerical acceptance,
physical alias/lifetime/epoch/synchronization and timing stay unestablished;
recording a zero-MAC objective does not invent cost or certification.
Campaign `component_performance` objectives belong in the explicit recipe, whose
selected bytes enter provenance; the minimal software spec needs only semantic,
numeric and reviewed operation support declarations. Legacy inline SW objective
declarations remain readable, but declaring both owners is refused. No objective
weights are inferred from examples, validation graphs or historical timing.
`rtl_intake` issues live structural authority from an actual fixed public
FIRRTL-to-HW replay and generic census. It never imports a target compiler,
support provider, ISA header or old fact table; protected campaign exclusions
are checked before source bytes are read. Public facts must match the complete
issued projection. Fresh generation binds the issued intake hash into the
coverage receipt and refuses legacy support/transcription source roles.
Protected exclusion prefixes remain selected when a tree is absent or pruned;
they are not readable input directories. Refuse indirect prefixes without
opening, creating or resolving the excluded tree.
The intake does not establish historical elaboration origin, numeric ISA
semantics, simulator/bitstream equivalence, runtime closure or performance;
those stay unknown until their independent owners qualify them. Saved JSON
cannot recreate live issuer authority. Full public-view admission additionally
checks software semantics, corpus, generic library, runtime and isolation.
`command_intake` binds a clean tracked hardware-source Git blob, actual CIRCT
generic serialization and lossless structural readers to live authority. Source
function declarations and parametric bundle layouts are not hardware legality or
numeric behavior. Local HW input observations follow only exact extracts and
contiguous concatenations; state and instances stop tracing. Numerical effects,
complete ISA legality, forbidden-command policy and physical ABI stay unknown.
An independently bound generation never invokes legacy support/readout hooks,
header taxonomy or declared execution capability providers.
`accessor_intake` runs fixed native public-header observers from protected simple
type/member specifications. No supplied values, masks, expressions, instruction
effects or schedules. Pin actual compiler, complete generated observer/artifacts,
and every observed non-system dependency matched to clean tracked public source.
Replay zero/single-bit/all-ones/mixed words before admitting field projections;
decode requested words by actual accessor calls and retain full invocation bytes.
This live authority certifies model/header observations only. Physical byte order,
CPU/RTL equivalence, legality, effects, ownership, completion and timers stay
unknown. Explicit software prohibitions resolve exact public declaration symbols;
never infer roles from substrings or copy numeric encoding tables.
`source_predicate_intake` binds exact public source predicates to the original
live command declaration authority and same clean tracked checkout. Recognize
only selected operand comparisons against admitted symbols and pure Boolean
composition; unsupported syntax, aliases, constants, operands or ambiguous
definitions refuse. Retain source expressions/spans and finite declared-domain
membership. Software policy chooses forbidden symbols; this reader assigns no
roles and does not certify historical elaboration or physical effects.
`software_intake` requires protected source/review selection, closed raw minimal
semantics and actual original independent-example replay with correspondences
for every operation owner. Status/source hashes alone cannot grant authority.
Refuse legacy extra fields before graph reads, including shapes/layouts,
model/capture/history/profiling, backend code, evidence descriptions and schedules.
Only generic independent numerical engines without source hooks are admitted by
the current minimal schema. Pin exact review/source/graph/reader bytes, bind the
live issued identity through generation and coverage, and verify the complete
`contract/software_spec.json` projection. Authoring requires this exact authority;
saved JSON or old reviewed specs do not suffice. Hardware numerical support,
historical origin, whole-domain correctness and physical effects stay separate.
`component_semantic_basis` owns explicit recipe-selected reviewed public example
rosters. Validate each pinned original frontend graph and exact call target roster;
only reviewed operation/effect semantics and hashes enter generation identity.
Shapes, counts, frequencies and source paths stay outside the author projection.
Freeze exact roster and complete graph bytes through the existing private source
snapshot. Protected selection establishes authority; status/provenance labels alone
do not. Obligations bind the selected roster hash and explicit source-to-owner
correspondences for every operation owner; matching effect kinds still need actual
generated witnesses. Refuse marked validation/private/history provenance or drift.
The trusted Boolean mode reaches the writer explicitly. Independent generation
must not look up, stat, read or stamp capture-derived shape census data; legacy
default generation retains its existing census annotation. Performance metadata
does not choose the input mode.

`resource_boundaries` derives aligned/tail extent points from explicit simultaneous
allocations in one selected physical store. Reuse generic address-space row sizing;
capacity/reservation paths refer only to refreshed selected facts. Keep missing
facts and missing below points explicit. Size inequalities do not prove placement,
lifetime, cache behavior, profitable scheduling or target execution.

`resource_frontiers` derives demand inequalities for explicitly selected byte-count
bank/accumulator/segment/alignment/cache declarations without assuming resident
row layouts. `component_program` source DAGs functionalize logical aliases and
updates into SSA; `component_numerics` independently evaluates exact modular
integer results. Original numerical engines and ordering remain unchanged.

`component_coverage_plan` owns immutable reviewed declarations;
`component_coverage_inputs` owns finite interactions and ordinary entry construction;
`component_coverage` owns private receipts/verifiers and guard links. Consumers
import their actual owners. `component_input_witnesses` checks full concrete
numeric operands against selected input palettes. `numeric_domains` screens
explicit allow/forbid nonfinite input/output declarations through normal admission;
absence retains historical behavior. Neither owner changes oracle acceptance.
`component_source_witnesses` independently checks actual full producer, RNE/clamp
code and decoded output observations for the closed quantizer observer. Ties,
clamps and preserved producer negative zeros require actual source values, not
metadata. Its scalar source arithmetic is f32, and code observations are published
as f32 values under the selected ordinary candidate comparison. Persistent
invocation contexts, physical alias/lifetime/epoch effects and FENV still need
explicit execution owners; declaring them without concrete witnesses blocks.

`component_execution_budget` owns source-derived small reference admission for
explicit v2 coverage plans. Freeze explicit per-member and total work, logical
tensor payload and scalar-width limits before authoring. Derive supported costs
without calling builders, capture, palettes or goldens; refuse unknown costs.
Apply the policy to every generated member before source binding or staging.
Keep denied mandatory obligations in the denominator; never convert them to
compile-only passes. Bind the complete private admission roster to generation
identity and rederive actual written source counts at receipt verification.
Historical v1 remains readable but does not establish bounded execution.
Logical reference counts are not process heap, compiler time or device timing.

`component_integer_bounds` owns pure source interval admission for integer DAGs.
Selected bounded_exact arithmetic requires every product, reduction prefix and
node result to fit the source and any explicitly declared internal signed widths.
Never approve an overflowing intermediate from a canceled/wrapped final output.
Only an explicitly admitted modular contract may use modular reference results;
unknown or unimplemented choices refuse before shaped data/reference allocation.
Keep original oracle arithmetic/order and numerical gates. Replay exact proof
and source/selected-semantic commitments before evaluating or admitting receipts.
Declared/source widths and their interval proof are not observed hardware effects.

`component_graph_variants` owns the closed independent bounded topology family:
fixed generic contraction/fork/ordered-join stages, selected extents/depth/fanout,
and complete logical-epoch/fresh-SSA pairs. Derive exact source costs from bounded
prototypes before topology unrolling, then rederive the ordinary full DAG cost.
Retain requested failures and half-pair gaps in coverage. `component_graph_relations`
reopens exact source membership and every original independent output before
issuing private full-output identity witnesses. Unknown arithmetic and overflowing
bounded-exact intermediates refuse. No source topology, shape, schedule or policy
comes from validation workloads or compiler references. Logical buffer epochs
and source aliases remain distinct from physical ownership/completion proofs.

`component_compile_plan` and `component_compile_sources` own separate independent
original source-only rosters. Use exact live HW/minimal SW origin and descriptor;
closed preauthor plans select integer rank-two copy/contraction sources and
explicit literal or fresh memory-volume/depth boundary extents. Emit through
the generic original source renderer and structurally reopen complete ordered
ABI without capsule bindings, shaped inputs or goldens. Explicit metadata and
aggregate source budgets apply to every requested guard and private member.
Retain failed members and static UNKNOWN obligations in the original denominator;
source readiness/refusal never grants numerical, candidate compile, address,
resource, streaming, whole-domain or physical authority. Saved reports cannot
recreate live issuance. Ordinary Phase1 owns actual candidate compile/link and
instruction admission from this exact protected original roster.
Source-only plan v2 adds complete logical-epoch/fresh-SSA graph pairs through
the existing independent topology factory. Select every matmul/copy/add owner
from the live minimal SW, check actual per-node source dtypes, and refuse missing
owners or signature mismatch before full topology construction. Derive exact
node/output counts from bounded factory prototypes; enforce explicit per-source
and complete-roster metadata budgets before unroll. Preserve every original
snapshot/escape/final ABI slot and requested pair member. Numerical v2 stays
separate and unchanged. Source aliases/epochs are functionalized SSA, never
physical reuse, timing, numerical equivalence or static target proof.

`frontend_use_def` strictly replays original result ownership and every serialized
args/kwargs reference against the complete typed edge roster. Historical basis
readers remain readable; missing use-def data cannot enter automatic derivation.
`component_automatic` accepts a closed protected policy containing only source
identities and finite budgets through the ordinary component coverage option.
Require the original live HW/minimal SW selection and exact protected example
roster before reading graph sources. Reviewed owner correspondences plus only
source interaction class presence select fixed generic one/two-contraction or
copy programs. Fresh bounded extents never copy original shapes, topology size,
frequencies, numerical policies or workload selectors. Ordinary budget admission,
writer, complete independent outputs and concrete screen remain authoritative.
Keep every unreviewed operator, unsupported original operator form/interaction,
effect interpretation and unqualified RTL resource/axis map mandatory UNKNOWN.
Source generation cannot certify the original graph or large-shape applicability.
Recompute the private derivation from its exact selected originals; re-signed
metadata cannot replace source factories or erase required missing classes.
Current automatic replay requires original source paths; relocated source
closures and automatic resource-role expansion are explicitly unqualified.

`operator_schema_intake` observes captured graph-level schemas against clean
tracked public function declarations and the actual registered native schemas.
The fixed reader retains typed alias sets, exact original argument/result joins
and process/environment evidence. Names never supply effect meanings. Direct
Tensor may-alias/may-write annotations remain distinct from concrete allocation
and mutation outcomes; wildcard, changing and container aliases stay UNKNOWN.
Automatic policy v2 binds the live schema/minimal-SW origin before generation.
An observed may-alias class selects a fresh bounded logical identity-view/copy
source with every input/view/copy output checked by the original normal oracle.
It does not cover arbitrary view layouts or grant physical pointer equality,
ownership, lifetime, completion, non-schema purity, hardware axis/resource roles
or installed-framework historical source correspondence. Retain those mandatory
gaps and unsupported effect classes. Policy/receipt v1 remains unchanged.

Operator-schema intake/selection v2 adds fixed native Tensor argument conversion
for exact finite original Python numeric literals in direct, unaliased schema
slots. The public API derives its numeric-as-Tensor guard from the original
dispatcher operator; callers cannot supply a guard or an operator-name table.
Bind the complete original node/path/schema/literal request and actual wrapped
number/dtype/value observation. Preserve original literal kind and floating
signed zero; ordinary scalar tensors can have different promotion. All compared
native scalar objects remain simultaneously live. Private storage observations
describe that framework invocation only. They grant no accelerator allocation,
numeric operation ownership, non-schema purity or complete effect-domain proof.
Pin the exact public API Git blobs, unmodified public header packaging replay,
native SDK/getter build, actual non-system dependencies and effective process
environments. Full installed framework/SDK and loader history stays UNKNOWN.
V1 preserves its Tensor-literal refusals; v2 cannot erase unreviewed operation,
numerical, unsupported container/alias/result or resource/effect obligations.

`hw_arithmetic` follows exact typed sign extension, multiplication, low input
slicing and modular addition to original HW output ordinals. Names and widths
alone assign no roles; state, instances, muxes and unimplemented arithmetic stop
the proof. `arithmetic_intake` binds a fresh native generic serialization of the
same live structural hardware source, complete reader/evidence bytes and effective
process environment. Its issued local bit-vector expressions never establish a
contraction, selected instruction path, reduction count/order, resource ownership
or tensor-axis map. Automatic policy v3 binds this exact live intake and retains
conditional compatible signed-integer numeric requirements plus the unrecognized
arithmetic domain as mandatory missing obligations in the ordinary generator.
Original numerical policies, bounded source counts, independent complete outputs
and unknown resource/axis obligations remain unchanged. No local result width is
silently promoted into an accumulator/K constraint or a whole-operation grant;
v1/v2 remain readable without new arithmetic authority.
Policy v3 may also select the exact optional independent operator-schema intake;
that selection requires the original same live minimal SW/HW and both identities
in one ordinary derivation. Replay both complete source records and retain all
logical outputs, numeric-path requirements and physical/unsupported effect gaps.
Missing, substituted or supplied-but-unselected observations refuse. A saved
record cannot drop either selected facet and obtain authority from re-signing.

Automatic policy/receipt v4 explicitly selects the next logical use-def source
family. Independently observed shared-producer and publication/further-use class
presence plus a unique reviewed movement owner selects fresh bounded three-copy
fork/chain graphs with every producer/user output published. The ordinary source
budget, writer and complete scalar oracle still apply before materialization;
denied members remain mandatory in both guard and private-transfer cohorts.
Original shapes, fanout, topology, frequencies and hardware geometry never drive
these sources. V1–v3 derivations retain their former required missing interactions.
V4 can select either or both exact independent schema/arithmetic observations;
supplied-but-unselected or missing/substituted facets refuse and replay never
grants effects from saved JSON. Logical source witnesses remain distinct from
mandatory physical reuse/layout/lifetime/completion and resource-axis UNKNOWNs.
Fresh odd/rectangular extents are teaching sources, not observed hardware tails.

`component_source_binding` prepares ordinary automatic tensor DAG originals from
the same live independent HW/minimal SW before a backend has been authored.
Source-only evidence selects no backend contract, provider, ISA classes, physical
oracle tier or tile geometry. Literal extents remain checked positive integers;
tile-relative extents and unimplemented reference/readout policies refuse.
Reopen the complete standard source against its original typed DAG and every
independent operation owner, retaining the original physical software screen.
Bind the explicit versioned source-semantic mode and every member screen to the
live selectors. `source_generated` members and `source_prepared` statuses mean
only original source preparation, never concrete hardware-admitted coverage or
an established Phase 2 guard. Legacy coverage verification refuses this scope,
including a re-signed complete status. Keep every automatic unsupported numeric,
effect, resource/axis and private-transfer obligation in the denominator. A
minimal descriptor with no selected corpus must not discover legacy siblings;
the ordinary registry destination/overlap protections remain in effect.
