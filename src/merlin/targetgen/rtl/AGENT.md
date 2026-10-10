# AGENT.md — merlin/python/merlin/targetgen/rtl

## Purpose

merlin-rtl-introspect: extract structure-only facts from elaborated RTL (CIRCT/FIRRTL).

## Modules

- `circt_introspect.py` — merlin-rtl circt-introspect (v2) — deterministic RTL fact extraction via the CIRCT HW dialect.
- `extraction_contract.py` — Target-owned declarations for source-specific RTL fact readers.
- `extract_module.py` — Extract a self-contained hw.module subtree (transitive closure) from a large CIRCT HW-dialect file.
- `firrtl_memory_lines.py` — Read explicit lowered FIRRTL memory blocks without changing hardware sources.
- `gen_iface_irdl.py` — Single-source IRDL bridge for the `merlin_iface` contract dialect.
- `gen_isa_module.py` — CIRCT facts -> generated, RTL-derived ISA encoder module (the 'moat').
- `gen_muon_digest.py` — Render muon_facts.json -> MUON_DIGEST.md (the Muon analog of gen_rtl_digest.py).
- `gen_numeric_facts.py` — CIRCT facts -> a numeric-SHAPE sanity checker (shrinks CIRCT's numeric blind spot).
- `gen_rtl_digest.py` — CIRCT facts -> one distilled RTL_DIGEST.md (so the CIRCT arm reads ONE spec sheet, not 55 RTL files).
- `introspect.py` — merlin-rtl-introspect — structure-only FIRRTL census and selected target-declared role probes.
- `hw_observations.py` — Observe exact HW input slices and equality constants without assigning ISA roles.
- `muon_introspect.py` — Deterministic, no-LLM extraction of Muon (RadianceMuonConfig) hardware facts from the real RTL.
- `replay_json_to_h.py` — Convert a RoCC replay spec JSON (gen_rocc_replay) into a C header the arc replay harness #includes.

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->

The installed `plain-word-relation` command consumes its closed source request
with explicit input/output byte bounds and retains conditional unknowns.

`plain_word_relation.observe_plain_word_relation` checks an explicitly pinned
plain source declaration and its exact reinterpret cast with complete ordered
literal/fixed-width fields. Primitive constructors and packing order are
explicit source-pinned conditional premises, never inferred target defaults.
Complete file/span, byte, token, nesting, field and word-width budgets are
retained; unsupported, partial, aliased or changed source selections refuse.
The observation does not issue primitive semantic review, source/binary or HW
correspondence, instruction length, ELF ABI, prohibition, effects or runtime
authority. A declaration-to-exact-HW-occurrence join remains separately required.

Answer-bearing replay generation lives in the experiments distribution under the
stable `merlin.targetgen.rtl.gen_rocc_replay` import name. Core owns this namespace
initializer and the structure-only tooling; extensions never overwrite it.

`hw_graph` projects uniquely owned legacy attributes into properties on a private
analysis clone for the selected discovery reader. Preserve every original field,
SSA edge and ordered typed module port; refuse ambiguous ownership, duplicate
symbols and incomplete graph identities. Original native source bytes and the
lossless parser representation remain unchanged. Representation compatibility
does not establish instruction roles, geometry or execution authority.

`hw_packing.equal_partitions` follows only typed extracts and contiguous
concatenations to a complete local input bitvector or opaque instance output.
Equal slices must cover that entire root without gaps or overlaps. Input/module
names, bit widths and slice counts never assign scalar signedness, tensor axes,
memory capacity, instruction routing, allocation or physical-tail semantics.
Registers, memories and other operations terminate the local observation.

`hw_combinational.prepare_combinational_observation` evaluates only a complete
selected scalar combinational module with explicit source/node/bit/case budgets.
All original inputs and outputs are retained; unreachable unsupported operations
also refuse. The scalar parser mode rejects dense literals before shaped-splat
materialization. Two-state local values and carry bits never assign address
spaces, allocation/capacity, memory ports, tensor axes, protocol or timing roles.
Source method probes require their own native/public source correspondence and
cannot replace the selected full RTL or its mandatory unknown obligations.

Integer comparison expressions retain the original i64 predicate and operand
widths. Signed predicates interpret the same bounded signless source bits as
two's-complement values. Local known-bit truth never grants four-state, reachable
range, command, memory history or admission semantics. Old pinned equality
records remain tied to their original readers; they are not upgraded in place.

Bounded scalar observations call the shared `contract.mlir_source_admission`
lexer guard before xDSL parsing: explicit source bytes, 64 syntax levels, scalar
width bounded by the requested limit or 64-bit metadata, and no aggregate
literals. This addresses preparse integer and nesting allocation failures; it
does not change the historical generic reader or replace a process resource
lease, exact operation typing, source correspondence or original obligations.

`hw_instance_inputs.prepare_instance_input_observation` follows only selected
original instance input ports through typed bounded combinational SSA. The
complete callee and actual instance input/output signatures are checked and
retained. Module arguments and direct instance results remain independent
symbolic roots; state and unknown expressions refuse. No opaque result is
followed across its instance or promoted to reachable state/effects. Original
undefined branches need a separate source validity premise; a local two-state
address or a declared memory depth does not establish allocation, tensor
capacity, temporal reuse, physical transfer correctness or timing authority.

Generic HW field readers accept a field in either its original attribute or
property dictionary. Duplicate ownership refuses even when both values agree;
the reader never chooses one spelling over another. Input signatures retain
each declared type and must match every original block argument. Supporting
both generic serializations does not assign new hardware or software roles.

`hw_memory_ports.memory_port_observations` retains every original local
`seq.firmem` declaration and typed read/write/read-write use. The public Seq
dialect supplies operand roles, optional enable/mask defaults and collision
enums. Explicit depth, width, latencies and exact scalar address/data/mask/clock
bindings remain declarations, with bounded expression metadata only. Original
state, opaque instance and unsupported producers stay symbolic roots. Retain
undefined read-under-write and out-of-range address domains; never evaluate
initial contents or infer command routing, software formats/axes, allocation,
physical capacity/tails or temporal completion from these local observations.

`hw_hierarchy_bindings.hierarchical_memory_bindings` preflights the complete
rooted occurrence, memory and port-binding roster before hierarchical expansion.
Follow only exact original named port/index/type bindings and supported scalar
combinational SSA through defined module outputs. Preserve distinct occurrence
identities for repeated callees. State, memory reads, external, parameterized
and unsupported producers remain explicit stops. Structural occurrence paths
and byte slices never establish command operands, decoded roles, tensor axes,
allocation/capacity use, collision validity, physical tails or temporal effects.

`hw_address_transitions.address_state_transitions` joins every original rooted
memory address endpoint to its exact typed local FirReg next/clock/reset/value
roster. Bounded local expressions stop at state, memory reads, instance outputs
and unsupported producers. Hold mux conditions require the exact same original
register result SSA; reset retains primitive priority. No clock event is evaluated
and no reset history, initialization, reachable range, decoded command, tensor
axis, allocation/capacity or temporal premise is granted. Original declared-depth
range statements remain unproved, including enabled out-of-range branches.

`hw_counter_intervals.observe_counter_intervals` reparses an explicitly selected
source module and exact FirReg/getter/clock endpoints before checking its complete
post-evaluation LOW/HIGH roster. Only one synchronous register with a typed
guarded unit increment/hold body and a direct `seq.to_clock` input is supported.
All original inputs and outputs, reset priority, held edges and modular wraps
are retained; interval increment totals across reset remain unknown. Endpoint
declarations, dataclasses and matching local values grant no source/runtime
authority. Initial reachability, producer custody, actual getter execution,
external events, physical units/loaded identity, omitted costs, cold/warm reuse
and independent held qualification remain required unknowns. Asynchronous,
split-state, opaque and unsupported clock/update forms refuse.

`hw_counter_state_timelines.observe_state_getter_timeline` checks every original
local FirReg state and input/output at every declared LOW/HIGH phase. Supported
scalar next/reset expressions use one pre-edge state for simultaneous updates;
the selected getter derives an exact MSB-first concatenation of complete register
results. Original state controls cannot be omitted or replaced with input labels.
Direct root clocks and synchronous constant resets retain their actual bindings;
opaque operations, gated clock expressions, asynchronous reset and preset refuse.
Random-initialization metadata remains recorded with initial reachability unknown.
Modular deltas and observed changes do not distinguish increments from writes;
unit-event meaning always remains unknown. Complete source-local rows grant no
sample custody, physical timing, source/SDK correspondence, startup/readback costs,
cold/warm composition or held-group qualification. The older one-register API
keeps its original supported domain and proof identity.

The installed `merlin-target-tools counter-source-observation` command consumes
an explicitly byte-bounded closed request and exact selected HW source. Both
readers check complete declared samples before a fresh data-only output is
written. The request/source hashes and all original unknowns remain in that
output; operator use grants no sample custody, physical timer, runtime or
performance qualification. It supplies no default samples or endpoint roles.

State timelines retain exact typed module output locations and symbol visibility.
Location membership must cover the complete original output roster; malformed or
unknown fields refuse. Nonempty `emit.fragments` references are unresolved semantic
emission dependencies, including unused bodies, and never discarded as locations.
Unsigned shifts preserve public Comb overshift-to-zero semantics; replication
requires an exact positive width multiple. Clock-to-bit casts only observe an
already derived original root clock. Nested SV, macros, fatal operations and
undefined effects remain required/refused; these expression additions do not
establish a complete public source cone, event units or physical correspondence.

The opt-in source macro premise resolves both original emission fragments from
the complete selected container. Every macro declaration has an explicit external
definedness/value row; declarations never default to definitions. Literal and
single-alias bodies and comment-only opaque text form the bounded supported domain.
Unused fragments and inactive regions are checked too. Scalar predicates observe
pre-edge states; every original conditional output or termination effect retains its
clock, branches and per-phase status. A timeline after triggered termination refuses.
The public SimToSV synthesis guard is conditional on the supplied premise, never
an implicit compile option. Supplied macro data, source-local effects and matching
values do not establish actual compiler/environment, scheduling or runtime custody.
Old requests without an explicit premise retain the unresolved-fragment refusal.

Counter-source request v2 carries the exact complete source macro premise for the
state-getter reader; its hash joins the request, source and returned effect rows.
The unit-counter reader admits only a null premise, preserving its narrower domain.
Request v1 retains its original fields, output field roster and unresolved
emission refusals; module metadata and source effects appear only in v2 output. Complete
conditional source samples and effect statuses grant no actual compiler options,
event scheduling, sample custody, physical timer or performance qualification.

`hw_transition_connectivity.transition_operand_connectivity` retains every
original state operand slot and crosses only exact named port/index/type
bindings in the already bounded rooted source hierarchy. Repeated callees keep
distinct occurrence identities. Complete declared root ports are identity data;
names and widths never classify a command interface or decoded resource role.
State, memory reads, opaque and unsupported producers stop traversal. Structural
root contact grants no event validity, state reachability, capacity, software
axis, physical effect, temporal closure or whole memory mapping requirement.

`hw_value_bindings.prepare_value_bindings` binds explicit original SSA slots to
exact supplied source bytes and complete rooted instance/port membership. Defined
combinational bindings may follow across instances; distinct occurrences retain
distinct state identities. Clock, state, memory, external, parameterized and
unsupported values remain cuts with original types/metadata. All operation and
nested effect membership in visited definitions is retained without evaluating
effects. Extract/concat connectivity assigns no opcode, getter, endpoint role,
sample custody, runtime, physical unit or complete-cost authority. The parsed-graph
API alone does not recheck source bytes; selections and exported JSON are data.
The installed `value-binding-observation` operator command requires a closed
original selection request, explicit input/output byte bounds and a fresh output.
It reopens request/source identities after export without granting source roles.

`hw_conditional_value_cones.prepare_conditional_value_cones` composes explicitly
selected typed source sinks under a complete exact roster of original integer
state, memory-read and opaque-result cuts. Only boundaries discovered by the
fresh structural reader can become conditional inputs; clock, noninteger and
unsupported reachable values refuse. Source hashes, occurrence paths, producer
ordinals, result slots and types must agree. Whole-source and rooted hierarchy
bounds precede parsing/expansion; complete case/output work precedes evaluation.
The metadata byte limit also bounds the complete returned canonical record,
including wrapper input/output/cut identities and original source membership.
Shared primitive semantics evaluate known bits only. Complete source operation,
state and effect membership and every unknown remain recorded. Supplied cut
values do not establish initialization, state transfers, history, collision
validity, opaque implementation, events, roles, ownership, physical costs or
admission. Existing value-binding and closed-module reader semantics stay fixed.

`hw_array_selection` preflights all rooted creation/get occurrences against
explicit operation, element and aggregate-bit budgets before scalar expansion.
Only exact single-level signless scalar arrays created in the same block are
supported. CIRCT lexical creation operands are MSB first; runtime index zero
selects the last operand. Opt-in C tracing exposes scalar dependencies only;
non-power-of-two index domains remain conditional stops and out-of-range local
values refuse. Nested and opaque aggregates remain stops. No default widening,
invented fill, state history, resource role or old source record upgrade occurs.

`hw_index_ranges` rejoins every original native address port to its exact local
memory declaration, SSA and signless type before symbolic unsigned type-domain
proofs. Explicit complete row/type/depth/proof budgets precede proof expansion;
no exponential endpoints are materialized. Domain containment is conditional
on defined known bits and unsigned indexing. Noncontained domains remain
unproved; address definedness, state/clock validity, memory history, physical
capacity, roles and effects remain required. Original T records stay unchanged.

`hw_partition_memory_bindings` joins complete original local bit partitions to
exact rooted memory data operands through typed named bindings, extracts and
concatenations. Intermediate occurrence identities survive upstream state,
read, opaque and other operation stops. Whole source/trace/join/materialization
budgets precede expansion; every original memory port is retained. Bit identity
is conditional on defined known bits, with no write event, software scalar,
command/axis, allocation/capacity, effect or whole packing admission. Existing
local, H and C records retain their original meanings.
