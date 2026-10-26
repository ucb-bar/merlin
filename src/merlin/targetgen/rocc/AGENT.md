# AGENT.md — merlin/python/merlin/targetgen/rocc

## Purpose

The RoCC coprocessor interface: decoding a trace of it, and emitting one.

## Modules

- `asm.py` — Derived RoCC assembler for ``inline_asm_insn`` targets.
- `decode.py` — Decode a target's RoCC instruction trace from a package's emitted ``lowered.llvm.mlir``.
- `semantics.py` — Generic RoCC operand semantics, driven entirely by a target's RTL facts and contract.

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->

## Invariants

**`target` is required and has no default.** The selected backend must expose `rocc_semantics`:
`isa_constants(target)`, `decode_instruction(funct, rs1, rs2, isa)`, and
`instruction_funct(name, rs1, isa)`. Operands are plain mappings with `raw`, `kind`,
`arg_index`, and `offset`; closed read-only module globals additionally carry their
original `symbol` with kind `global`, without inventing a runtime address. Constant
byte GEPs retain argument/global offsets. Dynamic indices, unproved aggregate layouts,
mutable/external globals and varying non-entry CFG arguments remain unknown.
The static operand observer may bind a proved singleton-loop induction only
inside its body; headers/exits retain unknown values. Compact loop bounds are
diagnostics, not a dynamic trace, resident-use proof or hazard qualification.
Accelerator ABI roles come from the selected contract;
field offsets and widths come from its original RTL facts. Missing capabilities refuse; there is no
fallback to another target's decoder. The provider must declare or derive its ISA facts.

**Fail closed, never drop.** An instruction form the decoder does not understand is recorded as class
`UNKNOWN` so `trace_check` can reject it. Silently skipping unparsed input is how a conformant backend
gets mis-measured: a too-narrow match once dropped every `.insn` in a trace and produced an empty one,
which read as "this backend emitted nothing" rather than "the decoder could not read it".

**Decode is runner-owned.** The trace is a measurement of what the package actually emitted, not an
artifact the package hands over — that is what makes it parity-clean between a baseline and an
assisted submission.

## Shared parsing and selected semantics

Core retains MLIR/SSA resolution, R-type transport parsing, fences, UNKNOWN records and trace
construction. CONFIG subtypes, packed addresses and transfer/compute meaning are not common
RoCC semantics. The generic data-bound decoder interprets explicit operand-to-bundle
bindings using extracted register layouts; it supplies no preferred lowering.
Legacy external semantics still require explicit provider selection.
Only parsed syntax is cached; every decode resolves its selected capability. Returned
traces never alias cache-owned dictionaries. Loaded-provider ownership checks still apply;
this is not an attestation of arbitrary in-process mutation or complete source bytes.
Assembly resolves an omitted kernel symbol through the selected contract's `harness_abi`.

Structural checking is a separate `rocc_semantics.rtl_checks` capability. It exposes
`load_default_facts(target)`, `project_facts(facts_rec)`,
`screen(trace, capsule, rtl_facts, *, target, command_buffer)`,
`compile_trace_checks(facts_rec, capsule, prefix)` and `render_trace(trace, facts_rec)`.
Reports use Merlin's shared `Check`/`CheckReport` types. Specialized geometry and
instruction ordering remain support-owned; missing coverage is not a passing check.

## Generic RoCC support (`semantics.py`)

`semantics.py` is a complete `rocc_semantics` capability written once for every RoCC target; its
`rtl_checks` attribute is `merlin.targetgen.rtl_checks_generic`. A support provider may select it
instead of shipping its own. It names no accelerator: funct codes and classes come from the
contract's `encoding.semantic_class`/`config_subtype` (codes range-checked against the facts'
decoder width, else the RoCC funct7 width); opcode, funct3 and DIM are RTL facts; every operand
field's offset/width is a `register_bundle_layouts` fact; the contract's `rocc_operand_roles`
says only which bundle field each decoded value reads and how it is interpreted. A field whose
bundle the facts lack decodes as UNKNOWN, never at an assumed offset. The rule engine reads the
contract's `rtl_checks` protocol (class roles, field meanings, local-address fields, labels, ordered
legality rules); a contract without one gets `RtlChecksUnavailable`, not another target's roles.
The engine checks legality only; a lowering choice (tile counts, which fields an emitter fills, how a
convolution is lowered) is never a finding.
The separately bound neutral contract route (`BoundSemantics` and `BoundChecks`)
currently implements `decode_clean` and `legal_funct` only. Its explicit
`instructions` roster does not reinterpret the legacy protocol contract above.
A bound contract may instead declare the role program above (`ROLE_PROGRAM_KEYS`
beside its roster) together with an `rtl_checks` protocol mapping. Its ISA
constants, decode and readout bits are then `derive_isa_constants` over the
bound snapshot, and `BoundProtocolChecks` runs the generic rule engine only
while the selected contract and facts equal that snapshot. Each program code
must be a roster code with the same label; a program cannot be combined with a
selected operand protocol, a rule list, or the absence of a protocol mapping.
An incomplete hardware decode table cannot establish universal instruction
legality. Unimplemented temporal/bounds rules and strategy rules refuse.

`BoundSemantics.operand_observer` is optional and driven only by the explicitly
selected `rocc_operand_protocol` version 1 data. It normalizes operand roles
without changing raw command/coverage identities or discovering authored inputs.
Actual field widths come from original source facts: reserved `slot_width` alone
does not qualify a field or address flag. Missing relations remain UNKNOWN.

`selected_operand_protocol.bind_original_operand_selection` additionally accepts
an explicitly pinned data sidecar over unchanged original facts. It rederives
selected FIRRTL packing and reopens ISA/controller sources before returning a
detached observer. Source subfields and closed conjunctions retain unknowns;
they grant no selector validity, queue acceptance or schedule qualification.
The caller must select the sidecar through the original-owned profile; this
helper supplies neither original-owner authority nor ambient activation.
Fresh original releases may explicitly select `rocc_operand_selection` with exact
release-local original-facts and sidecar byte pins. The normal bound runtime
factory consumes that selection; original raw decode/facts remain unchanged.
The release freezes all semantic and FIRRTL sources as checker-only inputs, and
the reviewed handoff rejects other contract changes or missing source closure.

Repeated bindings may reuse bounded detached structural observations for exact
facts, sidecar and raw-ISA bytes. Concurrent equal requests derive once; each
binding still checks current source paths/bytes, owner facts and final source
pins, and constructs a new observer. Cache entries confer no ownership or
acceptance, and are reset after fork. Changed declarations invalidate reuse.

`control_flow.constant_control_operand_witness` supplies a bounded fallback for
unresolved selected mode fields. It re-decodes the original module, visits every
watched transport/fence instruction through the shared width-aware evaluator, and
discards all observations on unknown control, poison or exhausted budgets. It
never unrolls emitted code, guesses pointer values, observes tensor data or supplies
numerical/device-execution qualification. Calls and multiple command-bearing
functions remain unsupported. Already-proved fields do not invoke this traversal.

The operand witness may summarize closed host-only loop regions using the original
constant-domain owner. No watched assembly, escaping SSA, opaque call or unproved
arithmetic assertion may be skipped. Rectangular control counts require only the
proved loop guards and no early exit; command-bearing loops remain exact. This
summary is control evidence, not host-memory or numerical qualification.
