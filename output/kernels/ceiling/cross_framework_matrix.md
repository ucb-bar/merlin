# Cross-framework fp32 GEMM ceiling matrix (spike, one substrate)

All columns measured on **spike** (functional, ISA `rv64gcv_zfh_zvfh`), `mode=inner_compute`, bit-exact verified vs a scalar reference, cycles read from the `mcycle` CSR (a **cycle proxy**, not cycle-accurate).

## Cycles

| shape (M=N=K) | OpenBLAS | XNNPACK | ours-baseline | ours-vfmacc | ours-tiled |
|---|---|---|---|---|---|
| 32^3 | 11,039 | 13,289 | 2,804,206 | 11,883 | 166,251 |
| 64^3 | 84,483 | 101,705 | 22,430,926 | 123,094 | 1,328,219 |
| 128^3 | 664,811 | 798,857 | 179,441,453 | not_run | 10,665,305 |

## Attainment

`expert/ours` columns = (kernel cycles) / (ours-baseline cycles): how many **ours-baseline** runs fit in one expert run (>1 => the expert is slower than our baseline). `best-expert / ours-best` = min(OpenBLAS, XNNPACK) divided by our fastest fork (>1 => ours beats the best expert; <1 => still a gap, the factor we trail by is its reciprocal).

| shape | OpenBLAS/ours-base | XNNPACK/ours-base | best-expert | ours-best | best-expert / ours-best |
|---|---|---|---|---|---|
| 32^3 | 3.94e-03x | 4.74e-03x | 11,039 | 11,883 | 0.93x (ours 1.1x slower) |
| 64^3 | 3.77e-03x | 4.53e-03x | 84,483 | 123,094 | 0.69x (ours 1.5x slower) |
| 128^3 | 3.70e-03x | 4.45e-03x | 664,811 | 10,665,305 | 0.06x (ours 16.0x slower) |

## not_run (honest blockers)

- **ours_vfmacc_contraction @ 128^3**: /scratch/agustin/projects/oscar-merlin/tmp/kernels/saturn-vectors/benchmarks/common/crt.S:136:(.text.init+0x124): relocation truncated to fit: R_RISCV_JAL against symbol `_init' defined in .text section in /tmp/cc0AjFQ3.o

Reading the blockers: the only remaining not_run is **`ours_vfmacc_contraction @ 128^3`**, which hits an `R_RISCV_JAL relocation truncated` — the FULLY-UNROLLED 128^3 `model.o` `.text` (16,384 fma, code that grows with M·N·K) exceeds the ±1 MB JAL reach of the shared Saturn `crt.S`/`test.ld` bare-metal layout (a harness link limit, NOT a numerical/codegen-quality result; the full-unroll fork builds, runs and verifies at 32^3 and 64^3). It is exactly the unbounded-code failure that the new **`ours_vfmacc_tiled`** (scalable) column fixes: ours-tiled's inner body is a CONSTANT 64 fma at every shape (K is a loop), so its .text is bounded and it builds, runs and verifies bit-exact at 32^3, 64^3 AND 128^3 (no JAL wall, and no more `tohost=1337` spike fault — that fault was an oversized vector<64x16>/<4x64> regalloc spill overrunning the stack into BSS, removed by bounding the K tile). Nothing is faked into a cycle number.

## Comparability caveats (read before trusting the numbers)

- **Same substrate / timer / harness for ALL five columns.** Every column is a standalone bare-metal Saturn ELF (crt.S + syscalls.c + test.ld, `-nostdlib`, `-march=rv64gcv_zfh_zvfh -mabi=lp64d`, `-O3 -ffast-math`) run on the SAME functional spike, timing the compute with `read_csr(mcycle)`. The cycle count is a **functional-spike proxy** (`cycle_accurate=false`), identical in kind for ours and the experts — NOT a Saturn-RTL / FireSim cycle-accurate number. On the functional model IPC=1, so `cycles ≈ instret` (retired instructions); the proxy therefore ranks codegen by **instruction count**, not by RTL timing — a real Saturn would re-rank vector-heavy kernels, but the cross-framework ORDERING here is robust because all columns share it.
- **Inner-compute scope; the fill asymmetry is now SUBTRACTED (caveat #1 fixed).** For all columns the one-time setup is hoisted OUT of the timed region (experts: operand pack; ours: memref-descriptor build). The experts time ONLY the GEMM microkernel call. Ours' compiled `_mlir_ciface_forward` is `linalg.fill` (zeroing C) + `linalg.matmul`; to make ours head-to-head with the experts' kernel-only timing, the ours driver now also times a **fill-only baseline** (a tight store loop zeroing the same M×N output, the exact traffic `linalg.fill` does, on the same `mcycle` CSR) and the `ours-*` cycles above are MATMUL-ONLY = (fill+matmul) − (fill-only). The fill is a small fraction (~3K/12K/49K cycles at 32/64/128); the driver also records `CYCLES_FULL` (fill+matmul). So the columns now compare GEMM-compute to GEMM-compute, no fill bias.
- **We deliberately do NOT use the runner's whole-model spike `cycles`.** That number (e.g. ~27.1 M cycles for hand_v0 at 64^3) is the entire Zephyr SMP image — boot, thread-create, cpu-pin, `merlin_run`, reboot — and is NOT comparable to an inner-compute kernel measurement. Using it would invalidate the comparison; this matrix uses the bare-metal inner-compute path for ours instead, on identical footing.
- **Kernel notes.** OpenBLAS `sgemm_kernel_8x8_zvl128b` (MR=NR=8, A ncopy / B tcopy pre-packed). XNNPACK `xnn_f32_gemm_ukernel_1x4v__rvv` (mr=1, called M times; weights goi-pre-packed; NR=`vsetvlmax_e32m4`=16 @ vlen128). Shapes 32/64/128 are divisible by both 8 and 16, so neither kernel takes a tail path. Ours = the frozen `hand_v0` RVV transform schedule (tile/vector [4,8,1]) with the named default-off impr feature.
