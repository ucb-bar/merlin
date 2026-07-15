# agent_spec_v0_mlir_oot — comparison report (Experiment ABI v0.1)

> **Generated from recorded `results.yaml`** by `results/gemmini/gen_agent_spec_v0_report.py` — not from scrollback.

> Source runs root: `runs/agent_spec_v0_oot_rerun` (clean-rebuild rerun).


## Per-rung results (read from results.yaml)

| rung | spike | spike cyc | verilator | verilator cyc | cycle_accurate |
|---|---|---|---|---|---|
| g0_matmul | pass | 47 | pass | 308 | True |
| g1_relu | pass | 47 | pass | 308 | True |
| g2_acc_scale | pass | 51 | pass | 250 | True |
| g3_acc_scale_relu | pass | 51 | pass | 250 | True |
| g4_tiled_k32 | pass | 89 | pass | 1006 | True |
| g5_resident_reuse | pass | 56 | pass | 428 | True |
| h0_matmul (hidden) | pass | 47 | pass | 308 | True |
| h1_relu (hidden) | pass | 47 | pass | 308 | True |
| h2_acc_scale (hidden) | pass | 51 | pass | 250 | True |

**Spike pass: 9/9 · Verilator pass: 9/9 · highest contiguous public rung (verilator): G5**


## Cross-package comparison

| system | artifact path | artifact type | integrity_exempt | authoring.mode | real gemmini dialect? | separate passes? | G0 | G1 | G2 | G3 | G4 | G5 | h0 | h1 | h2 | spike pass | verilator pass | highest rung | clean rebuild | provenance complete? | public-facts audit? | caveats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **agent_spec_v0_mlir_oot** | `generated_targets/gemmini/agent_spec_v0_mlir_oot/` | mlir_oot_target_backend | **false** | agent_generated_from_recipe | yes | yes (3) | pass (308cyc) | pass (308cyc) | pass (250cyc) | pass (250cyc) | pass (1006cyc) | pass (428cyc) | pass (308cyc) | pass (308cyc) | pass (250cyc) | 9/9 | 9/9 | G5 | pass | yes | yes | OOT target-lowering pkg (runner owns compile+harness+oracle); RoCC ported from native (tooling adv); G4/G5 certified instances not arbitrary tiling; verilator-oracle-only |
| merlin_native_v0 (reference) | `generated_targets/gemmini/merlin_native_v0/` | mlir_oot_target_backend | **true** | hand_curated | n/a (wraps native) | n/a | C0 | C1 | Q0 | — | C4 | — | — | — | — | 9 cells | 9 cells | battery | n/a | n/a | n/a | integrity-EXEMPT; battery vehicle; NOT a fair generated competitor (see certification_ledger_oot.md) |
| hand_smoke_oot (reference) | `generated_targets/gemmini/hand_smoke_oot/` | mlir_oot_target_backend | false | hand_curated | partial (def-use walk, no target dialect) | no | pass | pass | — | — | — | — | — | — | — | g0,g1 | g0,g1 | g1 | n/a | partial | n/a | i32 g0/g1 only; hand exemplar |
| **baseline (raw Claude Code)** | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | **PLACEHOLDER — fill when the operator runs the baseline under a measured harness** |

## Honest claim

> agent_spec_v0_mlir_oot demonstrates that Merlin's structured targetgen flow can produce a non-exempt out-of-tree MLIR-level Gemmini target package satisfying Experiment ABI v0.1 and certifying the supported rungs through the shared command-buffer/reference/oracle ladder. It does not yet prove general RTL-derived target generation.


## Cross-checks & caveats

- Verilator cycles match the certified native package where rungs correspond (G0=308=C0, G1=308=C1, G2=250=Q0, G4=1006=C4; see `certification_ledger_oot.md`).

- This is an OOT target-LOWERING package, not a standalone backend: the runner owns final LLVM lowering, object compilation, linking, harness, and oracle invocation.

- RoCC encoding was ported from the certified native path + public Gemmini facts (tooling/authoring advantage; see `agent_spec_v0_mlir_oot_public_facts_audit.md`), not independently rediscovered.

- G4/G5 are certified INSTANCES (32³ tiled; 2-matmul reuse), not arbitrary tiling/residency generality. No conv/attention/networks. RTL-fact provenance = verilator-oracle-only. No performance optimization; no 'Merlin is easier' claim (baseline unrun, no process telemetry on both sides).

