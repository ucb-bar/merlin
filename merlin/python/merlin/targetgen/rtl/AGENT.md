# AGENT.md — merlin/python/merlin/targetgen/rtl

## Purpose

merlin-rtl-introspect: extract structure-only facts from elaborated RTL (CIRCT/FIRRTL).

## Modules

- `circt_introspect.py` — merlin-rtl circt-introspect (v2) — deterministic RTL fact extraction via the CIRCT HW dialect.
- `extract_module.py` — Extract a self-contained hw.module subtree (transitive closure) from a large CIRCT HW-dialect file.
- `gen_arc_ports.py` — Generate a C header of port offsets/widths from an arcilator --state-file JSON.
- `gen_iface_irdl.py` — Single-source IRDL bridge for the `merlin_iface` contract dialect.
- `gen_isa_module.py` — CIRCT facts -> generated, RTL-derived ISA encoder module (the 'moat').
- `gen_muon_digest.py` — Render muon_facts.json -> MUON_DIGEST.md (the Muon analog of gen_rtl_digest.py).
- `gen_numeric_facts.py` — CIRCT facts -> a numeric-SHAPE sanity checker (shrinks CIRCT's numeric blind spot).
- `gen_rocc_replay.py` — Build a RoCC replay spec for the arcilator @Gemmini harness from a capsule's decoded trace + golden.
- `gen_rtl_digest.py` — CIRCT facts -> one distilled RTL_DIGEST.md (so the CIRCT arm reads ONE spec sheet, not 55 RTL files).
- `introspect.py` — merlin-rtl-introspect — structure-only fact extraction from elaborated Gemmini FIRRTL.
- `muon_introspect.py` — Deterministic, no-LLM extraction of Muon (RadianceMuonConfig) hardware facts from the real RTL.
- `replay_json_to_h.py` — Convert a RoCC replay spec JSON (gen_rocc_replay) into a C header the arc replay harness #includes.

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->
