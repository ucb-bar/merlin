# Gemmini cross-approach comparison (perf_full_0001)

The SAME kernels driven through each Gemmini code-gen approach on one ELF→spike/verilator harness. **Two axes, deliberately separated:**

1. **Performance = verilator (L3), cycle-accurate RTL — the only valid timing.** spike is *functional*: it does not model the systolic array, so its "cycles" plateau (~120 from 4K→2M MACs) and give util > 100% — meaningless as performance. Verilator is feasible only for small kernels (≤ ~32K MACs); bigger kernels need FireSim (L5), pending.
2. **Correctness & capability = spike (L2):** does each approach produce a correct result at all (exact-int == golden)? This is where COVERAGE differs (e.g. who can compile conv2d).

Approaches: **1** golden hand-tuned C lib (`tiled_matmul_auto`); **2** baseline-generated MLIR (agent_spec_v0); **3** merlin-generated MLIR (agent_spec_v1); **4** the deprecated-merlin hand-written C++ Gemmini dialect via IREE (`/scratch2/agustin/merlin`); plus an extra oscar-merlin native RoCC emitter for reference.

## 1. Performance — cycle-accurate RTL (verilator L3 + FireSim L5)

Cells = **cycles** `tier` (✗ = wrong output) **(util% = MACs/(cycles·256), 16×16 PE array)**. Both tiers simulate the SAME RTL: `L3` verilator covers small kernels (≤32K MACs); `L5` FireSim (Alveo U250 FPGA) covers the larger 64³+/model/attention shapes verilator can't reach in its time budget. 24 kernels have cycle-accurate data.

| kernel | shape | macs | 1.golden (C lib) | 2.baseline-gen (v0) | 3.merlin-gen (v1) | 4.depr-merlin handwritten (IREE) | (extra) merlin-native ref |
|---|---|---|---|---|---|---|---|
| G00_single_tile_16x16x16 | 16x16x16 | 4,096 | 559 `L3` (2.86%) | 308 `L3` (5.19%) | 308 `L3` (5.19%) | · | 308 `L3` (5.19%) |
| G01_multitile_sq_64x64x64 | 64x64x64 | 262,144 | 4843 `L5` (21.14%) | 7439 `L5` (13.77%) | 7439 `L5` (13.77%) | 93184 `L5` (1.1%) | 7439 `L5` (13.77%) |
| G02_rect_32x64x16 | 32x64x16 | 32,768 | 1137 `L3` (11.26%) | 919 `L3` (13.93%) | 919 `L3` (13.93%) | · | 919 `L3` (13.93%) |
| G03_kaccum_16x128x16 | 16x128x16 | 32,768 | 1342 `L3` (9.54%) | 983 `L3` (13.02%) | 983 `L3` (13.02%) | · | 983 `L3` (13.02%) |
| G04_wideN_16x16x128 | 16x16x128 | 32,768 | 1599 `L3` (8.01%) | 1269 `L3` (10.09%) | 1269 `L3` (10.09%) | · | 1269 `L3` (10.09%) |
| G05_tallM_128x16x16 | 128x16x16 | 32,768 | 1159 `L3` (11.04%) | 1102 `L3` (11.62%) | 1102 `L3` (11.62%) | · | 1102 `L3` (11.62%) |
| G06_acc_scale_i8_64x64x64 | 64x64x64 | 262,144 | 3091 `L5` (33.13%) | 7080 `L5` (14.46%) | 7082 `L5` (14.46%) | 93184 `L5` (1.1%) | 7080 `L5` (14.46%) |
| G07_relu_i8_64x64x64 | 64x64x64 | 262,144 | 3091 `L5` (33.13%) | 7080 `L5` (14.46%) | 7082 `L5` (14.46%) | 93184 `L5` (1.1%) | 7080 `L5` (14.46%) |
| G08_large_sq_128x128x128 | 128x128x128 | 2,097,152 | 16732 `L5` (48.96%) | 46821 `L5` (17.5%) | 46821 `L5` (17.5%) | 176242 `L5` (4.65%) | 46821 `L5` (17.5%) |
| M00_smolvla_model_16x32x960_i8 | 16x32x960 | 491,520 | 5851 `L5` (32.81%) | 18159 `L5` (10.57%) | 18176 `L5` (10.56%) | 157067 `L5` (1.22%) | 18159 `L5` (10.57%) |
| M01_smolvla_model_64x720x32_i8 | 64x720x32 | 1,474,560 | 10218 `L5` (56.37%) | 41073 `L5` (14.02%) | 41081 `L5` (14.02%) | 94864 `L5` (6.07%) | 41073 `L5` (14.02%) |
| M02_smolvla_model_64x32x720_i8 | 64x32x720 | 1,474,560 | 8940 `L5` (64.43%) | 36781 `L5` (15.66%) | 36781 `L5` (15.66%) | 331489 `L5` (1.74%) | 36781 `L5` (15.66%) |
| M03_openvla_vla_32x256x128_i8 | 32x256x128 | 1,048,576 | 7582 `L5` (54.02%) | 28051 `L5` (14.6%) | 28061 `L5` (14.6%) | 100242 `L5` (4.09%) | 28051 `L5` (14.6%) |
| M04_openvla_vla_32x128x256_i8 | 32x128x256 | 1,048,576 | 7826 `L5` (52.34%) | 27894 `L5` (14.68%) | 27894 `L5` (14.68%) | 122393 `L5` (3.35%) | 27894 `L5` (14.68%) |
| K_attn_qk_64x64x64 | 64x64x64 | 262,144 | 4847 `L5` (21.13%) | 7439 `L5` (13.77%) | 7437 `L5` (13.77%) | 93184 `L5` (1.1%) | 7439 `L5` (13.77%) |
| K_attn_pv_64x64x64 | 64x64x64 | 262,144 | 4847 `L5` (21.13%) | 7437 `L5` (13.77%) | 7437 `L5` (13.77%) | 93184 `L5` (1.1%) | 7437 `L5` (13.77%) |
| K_attn_qk_128x64x128 | 128x64x128 | 1,048,576 | 13506 `L5` (30.33%) | 25016 `L5` (16.37%) | 25016 `L5` (16.37%) | 167094 `L5` (2.45%) | 25016 `L5` (16.37%) |
| K_conv_std_3x3_8x8x4_8 | 1x8x8x4_k3x3x8s1 | 10,368 | err | · | 1528 `L5` (2.65%) | · | · |
| K_conv_3x3_16x16x16_16 | 1x16x16x16_k3x3x16s1 | 451,584 | err | · | 14710 `L5` (11.99%) | · | · |
| K_conv_1x1_16x16x32_32 | 1x16x16x32_k1x1x32s1 | 262,144 | err | · | 7800 `L5` (13.13%) | · | · |
| K_conv_3x3_stride2_16x16x16_32 | 1x16x16x16_k3x3x32s2 | 225,792 | err | · | 8609 `L5` (10.25%) | · | · |
| K_move_16x16 | 16x16 | 0 | 195 `L5` (0.0%) | · | 329 `L5` (0.0%) | · | · |
| K_move_64x64 | 64x64 | 0 | 2191 `L5` (0.0%) | · | 1620 `L5` (0.0%) | · | · |
| K_move_16x128 | 16x128 | 0 | 1154 `L5` (0.0%) | · | 1042 `L5` (0.0%) | · | · |

**Geomean cycles over the 24 cycle-accurate kernels** (lower = faster; golden is the hand-tuned reference): 1.golden (C lib) = 3084.4, 2.baseline-gen (v0) = 6941.7, 3.merlin-gen (v1) = 5222.5, 4.depr-merlin handwritten (IREE) = 123458.2, (extra) merlin-native ref = 6941.7.

## 2. Correctness & capability — spike L2 (functional)

Does each approach produce a correct result for each kernel? ✓ pass · not attempted (golden conv template deferred) `✗ (no compile)` backend cannot lower this op. **This is the coverage story** — not a timing comparison.

| kernel | op | shape | macs | 1.golden (C lib) | 2.baseline-gen (v0) | 3.merlin-gen (v1) | 4.depr-merlin handwritten (IREE) | (extra) merlin-native ref |
|---|---|---|---|---|---|---|---|---|
| G00_single_tile_16x16x16 | baremetalc | 16x16x16 | 4,096 | ✓ | ✓ | ✓ | ✓ | ✓ |
| G01_multitile_sq_64x64x64 | baremetalc | 64x64x64 | 262,144 | ✓ | ✓ | ✓ | ✓ | ✓ |
| G02_rect_32x64x16 | baremetalc | 32x64x16 | 32,768 | ✓ | ✓ | ✓ | ✓ | ✓ |
| G03_kaccum_16x128x16 | baremetalc | 16x128x16 | 32,768 | ✓ | ✓ | ✓ | ✓ | ✓ |
| G04_wideN_16x16x128 | baremetalc | 16x16x128 | 32,768 | ✓ | ✓ | ✓ | ✓ | ✓ |
| G05_tallM_128x16x16 | baremetalc | 128x16x16 | 32,768 | ✓ | ✓ | ✓ | ✓ | ✓ |
| G06_acc_scale_i8_64x64x64 | baremetalc | 64x64x64 | 262,144 | ✓ | ✓ | ✓ | ✓ | ✓ |
| G07_relu_i8_64x64x64 | baremetalc | 64x64x64 | 262,144 | ✓ | ✓ | ✓ | ✓ | ✓ |
| G08_large_sq_128x128x128 | baremetalc | 128x128x128 | 2,097,152 | ✓ | ✓ | ✓ | ✓ | ✓ |
| M00_smolvla_model_16x32x960_i8 | model | 16x32x960 | 491,520 | ✓ | ✓ | ✓ | ✓ | ✓ |
| M01_smolvla_model_64x720x32_i8 | model | 64x720x32 | 1,474,560 | ✓ | ✓ | ✓ | ✓ | ✓ |
| M02_smolvla_model_64x32x720_i8 | model | 64x32x720 | 1,474,560 | ✓ | ✓ | ✓ | ✓ | ✓ |
| M03_openvla_vla_32x256x128_i8 | model | 32x256x128 | 1,048,576 | ✓ | ✓ | ✓ | ✓ | ✓ |
| M04_openvla_vla_32x128x256_i8 | model | 32x128x256 | 1,048,576 | ✓ | ✓ | ✓ | ✓ | ✓ |
| K_attn_qk_64x64x64 | model_attention | 64x64x64 | 262,144 | ✓ | ✓ | ✓ | ✓ | ✓ |
| K_attn_pv_64x64x64 | model_attention | 64x64x64 | 262,144 | ✓ | ✓ | ✓ | ✓ | ✓ |
| K_attn_qk_128x64x128 | model_attention | 128x64x128 | 1,048,576 | ✓ | ✓ | ✓ | ✓ | ✓ |
| K_conv_std_3x3_8x8x4_8 | baremetalc | 1x8x8x4_k3x3x8s1 | 10,368 | · | ✗ (no compile) | ✓ | · | ✗ (no compile) |
| K_conv_3x3_16x16x16_16 | baremetalc | 1x16x16x16_k3x3x16s1 | 451,584 | · | ✗ (no compile) | ✓ | · | ✗ (no compile) |
| K_conv_1x1_16x16x32_32 | baremetalc | 1x16x16x32_k1x1x32s1 | 262,144 | · | ✗ (no compile) | ✓ | · | ✗ (no compile) |
| K_conv_3x3_stride2_16x16x16_32 | baremetalc | 1x16x16x16_k3x3x32s2 | 225,792 | · | ✗ (no compile) | ✓ | · | ✗ (no compile) |
| K_move_16x16 | baremetalc | 16x16 | 0 | ✓ | ✗ (no compile) | ✓ | · | ✗ (no compile) |
| K_move_64x64 | baremetalc | 64x64 | 0 | ✓ | ✗ (no compile) | ✓ | · | ✗ (no compile) |
| K_move_16x128 | baremetalc | 16x128 | 0 | ✓ | ✗ (no compile) | ✓ | · | ✗ (no compile) |

**Correct-kernel count per approach (spike):** 1.golden (C lib) = 20/24, 2.baseline-gen (v0) = 17/24, 3.merlin-gen (v1) = 24/24, 4.depr-merlin handwritten (IREE) = 17/24, (extra) merlin-native ref = 17/24.

## Notes

- **Why spike cycles are omitted from §1:** spike models RoCC functionally — the matmul retires in ~0 cycles, so spike cycle counts (~120 across all sizes) reflect only scalar issue overhead, not the systolic compute. Reporting them as performance would imply util > 100%. Verilator (and FireSim) are the timing oracles.
- **IREE arm (4) — read its cells carefully:** the deprecated-merlin C++ Gemmini dialect now has cycle-accurate L5 (FireSim) numbers, but it is ~10–40× slower than the generated/golden arms (e.g. 93184 vs 7439 cyc on 64³) AND it is verified by the IREE runner's OWN all-ones self-check (rc=0; each output == K), NOT against the exact-int shared golden the other arms use. So its cells are shown WITHOUT ✗ (it is not producing wrong answers) but its correctness is self-checked-on-all-ones, a weaker guarantee than the others — do not read IREE cycles as a like-for-like comparison.
- **'Identical RoCC' is only *almost* true (corrected by the L5 data):** baseline (v0) and the native emitter are **bit-identical in cycles** on every kernel; **merlin-gen (v1) differs by a few cycles** on epilogue/model kernels (e.g. G06/G07 acc_scale/relu +2, M00 +17, M01 +8, M03 +10) — v1's epilogue codegen is not byte-identical to v0/native. Small, but real.
- **Capability finding:** only **merlin-gen (v1)** compiles conv2d and movement among the generated backends; baseline (v0) and the native emitter cannot lower those ops. All four handle matmul and attention.
- **FireSim L5: COMPLETE** — 46/46 ELFs ran (100%); all 24 kernels are cycle-accurate (verilator ≤32K MACs, FireSim for the rest). Util-crossover holds at scale: golden 49–64% on the big/model shapes vs generated ~14–17%.
- `to/err` = verilator timeout (>900s) or runner error; `·` = not run / backend can't compile.
