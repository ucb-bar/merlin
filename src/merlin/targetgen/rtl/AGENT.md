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

Answer-bearing replay generation lives in the experiments distribution under the
stable `merlin.targetgen.rtl.gen_rocc_replay` import name. Core owns this namespace
initializer and the structure-only tooling; extensions never overwrite it.

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
