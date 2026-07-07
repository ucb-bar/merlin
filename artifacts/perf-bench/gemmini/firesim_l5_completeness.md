# FireSim L5 completeness audit (run perf_full_0001)

Verdict: **the L5 cell set is complete for everything in scope.** 70 cycle-accurate cells across 19
kernels; all apparent gaps are capability gaps or deliberate scale exclusions, not missing/failed runs.
Reproduce: `firesim_arm_results.json` (cells) vs `perf_results.json` (pipeline) vs `kernels/kernel_corpus.yaml` (27 kernels).

## Cell accounting (70 = 60 + 4 + 6)
- **60** — 12 kernels with full 5-arm coverage (golden/baseline/merlin_targetgen/iree_dialect/merlin_native):
  G01, G06, G07, G08, K_attn_pv_64, K_attn_qk_64, K_attn_qk_128, M00, M01, M02, M03, M04.
- **4** — conv kernels, `merlin_targetgen` only: K_conv_1x1, K_conv_3x3, K_conv_3x3_stride2, K_conv_std_3x3.
- **6** — movement kernels, golden + merlin_targetgen: K_move_16x16, K_move_16x128, K_move_64x64.

## Gaps — all expected
1. **conv = merlin-gen only.** Only the merlin-gen (v1) backend compiles conv2d. golden's conv template is
   deferred; baseline/IREE/native don't lower conv. (This is the capability story, see fig_capability /
   fig_agentic_coverage.) NOT a coverage failure.
2. **movement = golden + merlin only.** Only the hand-C golden and merlin-gen implement movement ops;
   baseline/IREE/native don't. Expected.
3. **Small G-kernels (G00, G02, G03, G04, G05) are L3, not L5 — by design.** ≤32K MACs → verilator-feasible;
   they carry 4-arm L3 (verilator) cells (golden/baseline/merlin_targetgen/merlin_native). FireSim L5 is
   reserved for kernels too big for verilator. (IREE-L3 for these is the separate EmitC effort.)
4. **tiny_llama giants M05/M06/M07 — out of scope, never built.** They are absent from the perf pipeline
   entirely (24 perf rows of 27 corpus kernels). M05 = 1,048,576,000 MACs = ~500× the largest L5 kernel
   (G08/attn ≈ 2.1M MACs); at cycle-accurate RTL that is billions of cycles — infeasible on FireSim in
   reasonable wall time. Excluded from the start; no ELFs exist. Documented exclusion, not a failure.

## What "complete" means here
Every (kernel × approach) cell that (a) is in the perf pipeline AND (b) the approach can actually compile
AND (c) is in the L5 size tier — is present. The L3+L5 cert ladder is therefore complete for all 24
pipeline kernels. The only way to "fill more" L5 cells would be to add the giants (infeasible) or to add
arms to conv/move that don't implement those ops (impossible by capability).
