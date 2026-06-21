# Gemmini target (prototype)

Gemmini (Berkeley systolic-array accelerator, `chipyard/generators/gemmini`) as a Merlin
**tensor-resident** target. Status: **prototype**, certification **uncertified**.

This target exists to prove the **RTL-grounded certification path**: lower the synthetic
`repeated_rhs_matmul` (C0: `i8 × i8 → i32`, matmul only, no epilogue) through the shared
`interface → gemmini → command-buffer` pipeline, then run the generated Gemmini kernel on a
real oracle and gate on three-way bit-exact equality
(`reference_outputs(cb) == simulate(cb) == oracle output`).

## Layers
- **Dialect** (`xdsl_dialects/targets/gemmini.py`): compiler-level `pack/matmul/commit/release`
  over `resident_tensor`/`accumulator`. NOT the ISA — no mvin/preload/compute/mvout ops here.
- **Command buffer**: reuses the target-independent ABI (`RES_PACK/MATMUL_RESIDENT/COMMIT/EVICT`).
- **Kernel codegen** (`runtime/backends/gemmini_codegen.py`): command buffer → bare-metal C using
  low-level `libgemmini` intrinsics. This is where the ISA mapping lives.
- **Backend** (`runtime/backends/gemmini.py`): compile + run on the oracle, parse, gate.

## Oracle ladder (see `contracts/target_contract.yaml`)
- L0 merlin reference + command-buffer simulator — `derived_from_rtl: false`
- L1 spike + gemmini extension (functional **bootstrap**) — `derived_from_rtl: false`
- L2 Gemmini Verilator RTL (`GemminiAndOPUShuttleConfig`) — `derived_from_rtl: true` (**certification**)
- L3 FireSim (same RTL, realistic memory/scale) — later

**Spike is bootstrap only; only Verilator/FireSim are RTL-certified.** Capacities in the
contract are hand-curated with provenance pending confirmation against `Configs.scala`; a future
`merlin-rtl-introspect` (CIRCT facts) pass will replace them and must reproduce them.

We do **not** compile RTL into this dialect. RTL provides facts + an executable oracle; the
target is generated from a constrained spec and accepted only after differential certification.
