# Graph-state chain: direct validation, semantic transport and strict receipts

This packet proves conditional graph-model preservation for actual consecutive
Merlin scheduling transformations. It contains five changed two-pass chains,
five separate fresh-process replays,55 fixed adversarial refusals and an audited
nonzero initialized joint model. It does **not** issue shared broker acceptance,
approve effects, prove actual source success/absence of poison, or qualify native,
hardware, numerical or full-model behavior.

Implementation is additive in `/home/jeremy/Projects/merlin` on
`feat/pass-verification`. The actual producer/projection source is the read-only
`merlin-graph-proof-intake` checkout at
`4499eebb959c0f92cefd92cc6d80646ba66474ed`. Production passes, shared definitions,
Main/CLI, frozen specifications, SPEC_LOCK, W and frontend work remain unchanged.
Stable GraphState v2 retains its1126-file packet and12 source bytes exactly.

## Existing anchor and exact relations

The existing `StateEffects.Action` represents deterministic total sequential
steps with a complete write frame and read-local writes. Existing
`commuteActions`/`exchangeAdjacent`/`preserveExchanges`, and the stable v2
`GraphState.check_sound`, remain the preservation roots.

The direct mode (`lean_graph_state_chain.check_chain`) checks each actual stage
join, reopens every complete module and independently derives a fresh complete
ORIGINAL→FINAL projection. That final `GraphState.check_sound` theorem preserves
every modeled state cell in arbitrary sequential surrounding context. This is
**direct composite validation** in one original source projection; it does not
compose independent stage semantics by equating their numeric SSA cells.

The separate supported transport mode (`GraphStateChain.lean` and
`lean_graph_state_chain_transport`) derives finite bijections from actual typed
operation/operand/result correspondence. A source-relative logical SSA cell
records its type and slot. Every physical memory, persistent field, shared alias
and ordered-observer identity stays exact and numerically fixed. Bijections
extend as identity outside the complete finite cell roster; both inverses and
injectivity are proved. Different numeric SSA IDs are never assumed equal.

For a common arbitrary **total pure primitive interpretation**, ordered input
slots and output slots construct actual state steps. Read coverage and EXACT
write membership are checked against the actual sorted footprints; complete
Action frames/locality are derived. Footprint sort order is not operand order.
Core descriptors bind opcode, operand/result types, metadata and physical slot
roles. The checker proves action and whole-execution conjugacy from those
descriptors, rather than assuming equality of arbitrary source/target actions.

For actual primitive programs P₀,P₁,P₂, maps ρ₀₁,ρ₁₂, common interpretation I and
arbitrary final-domain initial state σ, the composed relation is:

```text
execute_I(P₀, pull ρ₀₁ (pull ρ₁₂ σ))
  = pull ρ₀₁ (pull ρ₁₂ (execute_I(P₂, σ)))
```

Both stage `GraphState` schedule proofs are instantiated with the derived
primitive Action models, transported through their checked maps, and composed.
The middle primitive roster AND its actual raw cell-name roster are checked
equal. `independently_composed` is this separate theorem; `fixed_observations`
proves the physical-observer transport. `execute_transport` also covers any
supported matching primitive prefix/suffix by taking their complete list.
Arbitrary unrelated contexts across representations require a corresponding
semantic relation; no xDSL↔Kernel configuration-state bridge is assumed.

Supported transport is a flat defined single-block entry with admitted complete
effects and exact typed slots. Nested control/calls needing new semantics and
allocation/shared-resource identities that change with SSA renumbering are
refused by this mode. A successful direct check does not imply that a general
reindexing/operational bridge has been proved. No loop-storage/fan-out proof is
claimed or changed.

## Strict receipt mode and actual binding

`merlin.graph-state-chain.bound-receipt.v1` is a separate closed typed schema,
mode `direct-and-independent-common-primitive`. Issuance reruns all actual
direct, transport and byte-binding checks. Both stages must have a real checked
exchange; a printing-only or identity second stage is insufficient.

Every stage endpoint has an original file identity and a fixed retained raw
snapshot. All four endpoint buffers are compared byte for byte, including the
two identical middle endpoints. The original effect contract, current source
closure, selected pinned executable/profiles, actual imported proof objects,
Python/external parser dependencies, projections, query sources, compiled
objects, raw audits and exact theorem statements are bound. Parsing and hashing
use the same retained buffers; identities are reopened around verification and
replay. Receipt flags for public acceptance, approval, native fidelity,
numerical gates and full-model claims must be exact false booleans.

The kernel byte-binding checker in `GraphStateReceipt` is sound for its complete
binding data and direct all-state preservation relation. `GraphStateReceiptBytes`
packs bytes as `(size,Nat)` in little-endian order; size preserves zero bytes.
Lean proves encode/decode round trip and injectivity for every legal byte list.
The generated query proves decoded full stage-byte equality, not a hash
injectivity assumption. The host reader/encoder and IR parser are still trusted
implementations whose source identities are retained.

Replay requires explicit LIVE originals and the original contract. It first
refuses a substituted original against the unchanged checked snapshot, even
when all candidate metadata has fresh consistent hashes. It reproduces all
seven retained proof objects from their real query sources, audits their actual
imports and exact root rosters, then reruns all seven actual checks in a new
destination. Query bytes AND proof-object bytes must match. A saved status,
qualification string, digest or audit JSON grants no proof authority.

The fixed proof inventory is63 files for seven query/object/report/log groups.
Missing/empty/duplicate inventories, false statement/import rosters, malformed
bool/int/float fields, forged approval flags, stale source/tool/semantic IDs and
coherently rehashed proof substitutions are refused. The imported module/object
roster is compared to a fresh actual audit, including complete transitive
membership and object hashes. Hash-consistent fake metadata is insufficient.

## Actual producer coverage and fixed qualification

`../qualification-packed-v1/PLAN.json` was fixed before measurement. All source
and driver identities remained unchanged through the complete batch. The five
actual two-pass positives are:

| Case | Complete changed boundary |
| --- | --- |
| i8-cross | Moves independent multiply across configuration, then across the first scalar producer |
| i16-cross | Independent width with the same two actual changes |
| i32-tail-reverse | Extra live SSA context, different intermediate order and reverse producer arrangement |
| caller-memory | Two caller memrefs, constant in-bounds index and retained stores; caller alias cell remains shared |
| ordered-observers | Separately admitted resources with a shared ordered observation cell; both observers remain ordered |

Both stages invoke actual pinned `planning.apply_order`. All complete source,
middle and final modules, unchanged functions, signatures, operands/results,
physical effects and surrounding order are retained. Nonidentity numeric cell
maps occur in the actual middle/final projections. These are actual producer
API tests on explicit fixtures. Retained-model scheduling coverage and native
registered-MLIR/LLVM/machine execution are **not measured** by this packet.

The fixed qualification has five original strict checks, ONE separate batch of
five fresh-process strict replays,55 retained expected refusals, and one bound
joint witness. Refusals distinguish valid xDSL program changes, kernel-proved
false correspondence certificates, and malformed/binding/proof metadata.
They include dependent caller-store exchange, same-typed memory/operand
substitution, ordered-observer exchange, changed untouched function, wrong SSA
map/slots/types/order, missing physical identity, all endpoint substitutions,
exact-join counterexamples, and coherent object/audit/import forgeries.

Sixty owned tests passed in the candidate suite; two separately added,
nonoverlapping fresh-whole-chain cases passed in a focused run:62 distinct tests.
Ruff passes. Source-layout/access/library gates report125 passed,1 skipped and
the unchanged real-bwrap masking failure. The earlier exact diagnostic was
PermissionError reading a masked fixture while that test expects a successful
empty read. `check_structure.py` passes every check except the foreign
`zephyr_model.py` ceiling (3557>3537). These broad gates remain non-green; foreign
files were not changed to hide them.

## Inhabitation, audit and preserved failures

`GraphStateChainWitness` imports the ACTUAL checked i8 composite rosters/maps.
It binds actual result types and source/final return slots. Its total initialized
BitVec8 model has inputs2,3; source, middle and final observations are7, and final
persistent mode is2. Source/middle/final Action models and the whole joint-state
relation are proved. This is an explicit nonzero inhabitant of the conditional
model. It proves neither actual IR source success, runtime typing, absence of
operational poison, nor actual instruction/runtime delivery.

Every successful imported project proof closure is audited under the standard
three axioms only. No new axiom, sorry/admit, native_decide, bv_decide, native
reflection or proof implementation substitution is introduced. Per-query
overlapping closure counts are48 (direct),198 (transport),214 (independent
composition),257 (byte-binding receipt),228 (joint witness). All have zero
violations and use a subset of propext/Classical.choice/Quot.sound. Do not add
overlapping theorem counts. Exact types, real object/import identities and first
stdout/stderr buffers are retained.

The selected pure Lean4.35 executable has SHA256
`bf8d54e4714cc4b03d3f6bb34c83b7202b87e49c8bfcbff6895c085bb90ceb38`.
The current private profile is
`e0d5c978953bd208a57e9e4bb1e401a5602cf4968c042f725a7462c289008636`.
It retains exact stable v2 core objects and builds only the additive chain/byte/
receipt modules. No matching full Lean-MLIR manifest was selected or downloaded;
that route and shared live broker integration remain unavailable/unintegrated.

All first development failures remain preserved. In particular, a pre-packed
raw-list binding produced105/107MB objects; its first strict replay stopped on
the seventh proof reproduction without an object or diagnostic buffer. That
wrapper failed to record the child return code: the cause is UNKNOWN. It is an
unqualified development failure, not a completed replay or a repaired native
compiler bug. The compact representation is a separately proved representation
and new profile/qualification; its binding object is about227KB. Earlier syntax,
instance-derivation and generation errors retain their first raw buffers.
Nothing was retry-promoted into the frozen batch.

Stable v2's1126-file/12-source checkpoint remains exact. All old numerical
counterexamples, scalar-square source/policy separation, F12, failed native
contexts and frozen semantics/spec/locks remain preserved separately. This
packet establishes no semantics bridge to the older MerlinVerify numerical
domain and discharges no numerical gate.

## Remaining premises and paper row

Actual typed SSA/module matching and source-to-model projection are trusted
Python from the pinned producer plus the identified primary leaves. Complete
physical effects, aliases, observers, totality/purity, completion and the common
primitive interpreter's fidelity to actual instructions need independent
admission. Names, metadata flags or semantic hashes do not prove those facts.
The Python/Lean runtime, kernel implementation, host file I/O and encoding
implementation remain trusted. Actual operational typing/definedness/poison
properties, native/runtime/hardware conformance, unsupported control/allocation
bridges, source intent and full compiler/model coverage remain outside the claim.
No approved multiplication policy is manufactured and no refuted numerical law
is discharged by graph typing or source identity.

| Actual transformation / identities | Existing and new root / relation | Checked evidence | Outstanding gates |
| --- | --- | --- | --- |
| Five actual two-pass pinned4499 `planning.apply_order` chains; complete source/middle/final original/snapshot identities in each strict receipt | Existing StateEffects/GraphState preservation; separate direct original→final validation; GraphStateChain checked typed SSA bijections and independently_composed common-primitive state equality; GraphStateReceipt exact decoded byte-binding soundness | Five originals, five fresh strict replays with seven reproduced/fresh proofs each,55 fixed refusals,62 distinct tests, nonzero actual-roster joint model; raw std3 audits and exact types; source-only delivery/dependencies | Conditional model result. Effect/alias/observer/common-interpreter/host-projection fidelity require admission; full Lean-MLIR and shared broker integration unavailable; no xDSL↔Kernel, operational source success/poison, native/numerical, retained-model or full compiler claim |

Review should assess that exact scoped statement and its original effect
admissions, actual typed projection, common interpreter domain and supported
physical identity restrictions. Shared `verify_pipeline_change` /
`accepted_pipeline_receipt` live-custody integration needs an agreed interface;
no protected/shared code was edited to imply it already exists.
