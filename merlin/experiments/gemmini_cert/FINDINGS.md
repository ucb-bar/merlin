# Gemmini conformance — FINDINGS

RTL-grounded certification of the Gemmini target. Each cell lowers a Merlin command buffer,
generates the Gemmini kernel, runs it on an oracle, and gates **three-way bit-exact equality**
(`reference_outputs(cb) == simulate(cb) == oracle output`). Reproduce:
`python merlin/experiments/gemmini_cert/run.py --simulators spike|verilator`.

## Certifiable rungs (integer-exact)

| rung | shape | epilogue | spike (L1, bootstrap) | Verilator (L2, RTL-certified) | cycles (L2) |
|---|---|---|---|---|---|
| C0  | 16×16×16 | — | ✅ | ✅ | 211 |
| C1  | 16×16×16 | relu | ✅ | ✅ | 211 |
| C4  | 32×32×32 (multi-tile, K-accum) | — | ✅ | ✅ | 882 |
| C4e | 16×24×20 (zero-padded) | — | ✅ | ✅ | 529 |
| C5  | reuse-4 (one resident W) | — | ✅ | ✅ | 841 |

All five rungs are three-way bit-exact (reference == simulator == Verilator RTL).

(`derived_from_rtl`: spike = false (functional bootstrap), Verilator = true. cycles are
measured anchors, NOT a calibrated cost model.)

## Agentic generation (Milestone C)

A Claude Code agent (Opus, no API key), shown only the cb ABI + ISA reference + C0/C1 examples
(never the reference outputs), synthesized `generate_driver` that is **bit-exact on the held-out
shapes it never saw** (C4, C4e, C5) — first attempt. The agent kernel is **RTL-certified on
Verilator** (`derived_from_rtl=True`): C0 visible (259 cyc) and **C4 held-out multi-tile, three-
way bit-exact (1131 cyc)**. Certified kernel + transcript in `agent_generated/`; see
`docs/gemmini_target_prototype.md` and `test_gemmini_agentic.py`.

## Documented divergences (NOT bit-exact — by design, not oversight)

- **requant (C2/C3):** Gemmini `acc_scale` is ignored on the full-i32 readout; its i8-downscale
  path rounds differently from merlin's integer round-half-up shift. See
  `docs/gemmini_requant_reconciliation.md`.
- **transpose (C6):** needs an interface-level transpose op; the non-transposed WS path is
  already certified.

## RTL facts (Milestone D — CIRCT)

`merlin.targetgen.rtl.introspect` extracts structure-only facts from the elaborated FIRRTL
(mesh 16×16, scratchpad 262144 B, i8/i32, RoCC/TLB) and **reproduces the hand-curated contract**
capacities. FireSim (L3) runs the same RTL as L2 (correctness already earned); FPGA/queue-gated.
