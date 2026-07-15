# capsule_bench_v0 — iteration history (path to green)

First-class evidence of *how* the artifact reached green, not just the final pass/fail. The
per-iteration evidence dirs (`iteration_XXX/` with `files_changed.patch`, status before/after,
first-failure, numeric/trace diffs, profile, notes) are written by
`merlin.targetgen.iteration_recorder`; this file is the human-readable summary.

## Backend lineage

`agent_spec_v1_mlir_oot` was **forked** from the RTL-certified `agent_spec_v0_mlir_oot` (which
certifies the G0–G5 matmul/relu/acc_scale/resident/tiled rungs three-way bit-exact on spike +
verilator). The capsule-bench corpus therefore inherited a proven weight-stationary matmul datapath;
the matmul-family capsules (A0, A2–A7, B0–B2, C0–C6) required **no backend repair** — they certified
on the first run through the shared command-buffer/reference/oracle ladder.

## Extension iterations (the genuine new work)

| step | change | first-failure addressed | outcome |
|---|---|---|---|
| 0 | fork v0→v1; run matmul-family corpus | — | 17/17 capsules pass L0/L1/trace first try |
| 1 | add `merlin_iface.conv2d` + im2col lowering (InterfaceToGemmini synthesizes the im2col activation arg; recipe carried on `gemmini.matmul`; runner materializes it additively in `materialize_inputs`) | conv2d had no lowering (target_to_llvm) | B3/B4 pass L0/L1/trace; GemminiToLLVM unchanged (conv reuses the matmul RoCC path) |
| 2 | add `merlin_iface.movement` + `gemmini.movement` + a no-compute MVIN→scratchpad→MVOUT kernel + a movement harness branch in `compile.link_elf` + VECTOR_MAP-identity in reference/simulator | movement had no lowering and broke the matmul-only harness | A1 passes L0/L1/trace **and** L2 spike (27cyc) + L3 verilator (143cyc, RTL) |

## Corpus-generation fixes (not backend repairs)

- `resident_reuse` / `conv2d` op names had to be added to the capsule schema `operation.op` enum.
- A self-inflicted `oracle_runner.py`-style risk was avoided: the new runner modules import merlin
  freely (they are runner code), while the **package** stays integrity-clean (no harness/reference
  imports) — verified by the integrity scan on every run.

## Freeze + hidden

After all public/dev capsules were green, the artifact was **frozen** (`freeze.json` pins the repo
commit + toolchain SHAs). The hidden capsules (renamed-tensor variants → different deterministic
data) were run **once** post-freeze to confirm data-independence; their results are recorded
separately. Hidden-repair mode was **not** enabled.

## Honest notes

- No failed→fixed *backend* repair iterations were needed for the matmul family because v1 reuses the
  certified v0 datapath; this is recorded truthfully rather than manufacturing artificial failures.
- The conv and movement extensions are the real new lowering work and were validated incrementally
  (L0/L1/trace, then oracle) before being declared green.
