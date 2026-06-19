# Cross-framework fp32 GEMM ceiling matrix (spike, one substrate)

All columns measured on **spike** (functional, ISA `rv64gcv_zfh_zvfh`), `mode=inner_compute`, bit-exact verified vs a scalar reference, cycles read from the `mcycle` CSR (a **cycle proxy**, not cycle-accurate).

## Cycles

| shape (M=N=K) | OpenBLAS | XNNPACK | ours-baseline | ours-vfmacc | ours-tiled |
|---|---|---|---|---|---|
| 32^3 | 11,039 | 13,289 | 2,807,285 | 14,962 | 45,021 |
| 64^3 | 84,483 | 101,705 | 22,443,221 | 135,389 | not_run |
| 128^3 | 664,811 | 798,857 | 179,490,612 | not_run | not_run |

## Attainment

`expert/ours` columns = (kernel cycles) / (ours-baseline cycles): how many **ours-baseline** runs fit in one expert run (>1 => the expert is slower than our baseline). `best-expert / ours-best` = min(OpenBLAS, XNNPACK) divided by our fastest fork (>1 => ours beats the best expert; <1 => still a gap, the factor we trail by is its reciprocal).

| shape | OpenBLAS/ours-base | XNNPACK/ours-base | best-expert | ours-best | best-expert / ours-best |
|---|---|---|---|---|---|
| 32^3 | 3.93e-03x | 4.73e-03x | 11,039 | 14,962 | 0.74x (ours 1.4x slower) |
| 64^3 | 3.76e-03x | 4.53e-03x | 84,483 | 135,389 | 0.62x (ours 1.6x slower) |
| 128^3 | 3.70e-03x | 4.45e-03x | 664,811 | 179,490,612 | 3.70e-03x (ours 270.0x slower) |

## not_run (honest blockers)

- **ours_vfmacc_tiled @ 64^3**: spike run failed: spike faulted rc=57; stderr: *** FAILED *** (tohost = 1337); stdout tail:
- **ours_vfmacc_contraction @ 128^3**: /scratch/agustin/projects/oscar-merlin/tmp/kernels/saturn-vectors/benchmarks/common/crt.S:136:(.text.init+0x124): relocation truncated to fit: R_RISCV_JAL against symbol `_init' defined in .text section in /tmp/cc0AjFQ3.o
- **ours_vfmacc_tiled @ 128^3**: spike run failed: spike faulted rc=57; stderr: *** FAILED *** (tohost = 1337); stdout tail:

Reading the blockers: **`ours_vfmacc_tiled` faults on spike (tohost=1337) at M≥64** — a genuine codegen bug in that experimental fork feature at larger shapes (it passes and verifies at 32^3). **`ours_vfmacc_contraction @ 128^3` hits an `R_RISCV_JAL relocation truncated`** — the heavily-unrolled 128^3 `model.o` `.text` exceeds the ±1 MB JAL reach of the shared Saturn `crt.S`/`test.ld` bare-metal layout (a harness link limit, NOT a numerical/codegen-quality result; the fork builds, runs and verifies at 32^3 and 64^3). Neither is faked into a cycle number.

## Comparability caveats (read before trusting the numbers)

- **Same substrate / timer / harness for ALL five columns.** Every column is a standalone bare-metal Saturn ELF (crt.S + syscalls.c + test.ld, `-nostdlib`, `-march=rv64gcv_zfh_zvfh -mabi=lp64d`, `-O3 -ffast-math`) run on the SAME functional spike, timing the compute with `read_csr(mcycle)`. The cycle count is a **functional-spike proxy** (`cycle_accurate=false`), identical in kind for ours and the experts — NOT a Saturn-RTL / FireSim cycle-accurate number. On the functional model IPC=1, so `cycles ≈ instret` (retired instructions); the proxy therefore ranks codegen by **instruction count**, not by RTL timing — a real Saturn would re-rank vector-heavy kernels, but the cross-framework ORDERING here is robust because all columns share it.
- **Inner-compute scope, with one honest asymmetry.** For all columns the one-time setup is hoisted OUT of the timed region (experts: operand pack; ours: memref-descriptor build). The experts time ONLY the GEMM microkernel call. **Ours times `_mlir_ciface_forward`**, which for this single-op workload is the compiler-emitted `linalg.fill` (zeroing C) **plus** `linalg.matmul` — i.e. the GEMM plus a thin compiler wrapper, no multi-op model and no Zephyr/threading. So the columns are directly comparable up to that extra `fill` of the M×N output.
- **We deliberately do NOT use the runner's whole-model spike `cycles`.** That number (e.g. ~27.1 M cycles for hand_v0 at 64^3) is the entire Zephyr SMP image — boot, thread-create, cpu-pin, `merlin_run`, reboot — and is NOT comparable to an inner-compute kernel measurement. Using it would invalidate the comparison; this matrix uses the bare-metal inner-compute path for ours instead, on identical footing.
- **Kernel notes.** OpenBLAS `sgemm_kernel_8x8_zvl128b` (MR=NR=8, A ncopy / B tcopy pre-packed). XNNPACK `xnn_f32_gemm_ukernel_1x4v__rvv` (mr=1, called M times; weights goi-pre-packed; NR=`vsetvlmax_e32m4`=16 @ vlen128). Shapes 32/64/128 are divisible by both 8 and 16, so neither kernel takes a tail path. Ours = the frozen `hand_v0` RVV transform schedule (tile/vector [4,8,1]) with the named default-off impr feature.
