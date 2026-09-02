# Provenance — `gemmini_xdsl_rtl_v0`

The gemmini out-of-tree MLIR backend that earned **33/33 at L3** in the Arm-4 capsule-bench run of
2026-09-02. Tracked here because it existed nowhere under version control: the run workspace it was
produced in lives under `out/runs/`, which `.gitignore` excludes, and `out/preserved_submissions/`
held the two preceding candidates but not this one.

## What earned the verdict

| | |
|---|---|
| run id | `merlincirct_arm4_func_20260902_codex3` |
| candidate | `_qa_work/cand_01/submission` (round 01) |
| verdict | `qa_history/verdict_round_01.json` — `n_passed 33 / n_capsules 33`, `all_pass true`, `highest_tier L3`, `integrity_status clean` |
| L3 engine | GSIM (32 capsules) + mesh dispatch ledger (1 capsule, `M3`) |
| authoring | agent-generated from RTL facts (`manifest.yaml: authoring.mode`) |

**The run recorded no package digest.** Its per-capsule submission stamp carries
`{"package": "<path>", "run_id": ..., "stated": ["run_id"]}` — a path, not bytes. `SHA256SUMS` in this
directory was therefore computed when the package was promoted, not by the run, and is the first
byte-level identity this artifact has had. Verify with `sha256sum -c SHA256SUMS`.

Relative to `cand_01/submission` this copy omits only non-source files: `__pycache__/`, `*.pyc`, and
three stray run outputs (`m2_target.mlir`, `parse.err`, `command_buffer.json`). Source content is
identical.

## What the verdict does and does not establish

The cohort is 34 capsules, not the corpus. From
`merlin/experiments/capsule_bench/targets/gemmini/target_experiment.yaml`:

```yaml
expected_cohort: {source_capsules: 48, admitted_capsules: 34}
capability_exclude_capsules: [GC1…GC6, GF1…GF5]      # 11 bf16 ops this RTL cannot execute
resource_bound:
  policy: representative_l3_capstones_v1
  exclude_capsules: [GX0_interop_rvv_lane, M0_small_llama_gemmini, M1_lstmnetvit_gemmini]
  required_admitted_models: [M2_microvit_gemmini, M3_host_island_seam_gemmini]
```

48 − 11 − 3 = 34. So:

- **Established.** 13 ISA capsules, 11 layers, 8 model slices and one model-kind capsule certified on
  elaborated RTL. Of the two admitted models, `M3_host_island_seam` passed — a GEMM → host LayerNorm →
  GEMM seam with an ordered dispatch ledger (`{on_mesh: 2, scalar_rvv_lane: 16}`), per-tile mesh
  verification, and `model_execution_check: pass`. `M2_microvit` ended `budget_exhausted`.
- **Not established.** Whole-model compilation. The three capstones that would show it — including
  `M0_small_llama` (embed / RMSNorm / RoPE / attention / SwiGLU / lm_head) — are excluded from the
  cohort by the `resource_bound` policy above. They are neither passed nor refuted here.

Do not cite "33/33" as evidence that a whole model compiles. It is evidence about operators, layers
and one host/device seam.

## Hardware

Verify the RTL revision from `merlin/contract/hardware_pins.yaml` (`gemmini_rtl`) before quoting any
cycle count. A verdict attributed to the wrong device is worse than no verdict.

## Contents

~1,100 lines of original code; the remainder is vendored third-party (`xdsl`, `typing_extensions`,
`immutabledict`, `ordered_set`). Kept vendored so this directory is the artifact that was graded. A
slimmed distribution with `xdsl` as a declared dependency is a follow-up, and would carry a different
digest.

| file | lines | role |
|---|---|---|
| `mlir_oot/lowering/isa.py` | 465 | interface commands → RoCC instruction trace |
| `mlir_oot/ir_ingest.py` | 182 | im2col rewrite |
| `mlir_oot/xdsl_dialects/merlin_iface.py` | 153 | the interface dialect it consumes |
| `mlir_oot/xdsl_dialects/gemmini.py` | 111 | the gemmini target dialect (8 IRDL ops) |
| `mlir_oot/targetgen/generate/llvm_artifact.py` | 76 | LLVM-dialect emission |
| `mlir_oot/transforms.py` | 47 | the interface → gemmini conversion pass |
| `mlir_oot/gemmini_opt.py` | 46 | driver behind `mlir_oot/gemmini-opt` |
| `mlir_oot/targetgen/synthesize/tiling.py` | 25 | tiling |

See [docs/guides/gemmini_oot_package.md](../../../../docs/guides/gemmini_oot_package.md) to run it.
