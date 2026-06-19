# RVV f32 GEMM ceiling — ours vs expert (XNNPACK, OpenBLAS) on spike

Op: `matmul`, dtype `f32`, shape **M=N=K=64**. Target: **spike** (`--isa=rv64gcv_zfh_zvfh`,
default VLEN=128). Cycle counts are the `mcycle` CSR delta around the kernel compute call
(`instret` matches within ~2, so cycle CSR did not trap). `cycle_accurate=false` — this is an
instruction/cycle proxy, the same basis as our compiler's spike cycles.

## Fair-comparison mode: `inner_compute`
Both experts consume PRE-PACKED operands. To match how our spike number measures the *emitted
compute* (not one-time setup), all packing/setup is done OUTSIDE the timed region; only the kernel
compute call(s) are wrapped in `read_csr(mcycle)`:

- **OpenBLAS** `sgemm_kernel_8x8_zvl128b`: A pre-packed (ncopy_8, col-major-in-panel), B pre-packed
  (tcopy_8), C col-major. Computes `C += alpha*A·B`. Packing done outside timing.
- **XNNPACK** `xnn_f32_gemm_ukernel_1x4v__rvv` (NR = vsetvlmax_e32m4 = 16 lanes @ VLEN=128): weights
  pre-packed goi (bias[NR] then K·[NR] panels), streamed by pointer-advance; A row-major streamed
  scalar-by-scalar; bias is the accumulator init. Packing + bias setup done outside timing; the
  kernel is called once per M row (mr=1) inside the timed loop.

Both kernels VERIFY PASS vs a scalar triple-loop reference (maxabs_err = 0), so the cycle numbers
are for a CORRECT 64×64×64 GEMM, not a no-op.

## Measured cycles (lower = better)

| # | kernel                                   | cycles      | kind   |
|---|------------------------------------------|-------------|--------|
| 1 | OpenBLAS `sgemm_kernel_8x8_zvl128b`      |      84,483 | expert |
| 2 | XNNPACK `f32_gemm_ukernel_1x4v__rvv`     |     101,705 | expert |
| 3 | ours `impr_rvv_v5` (vfmacc fork)         |     135,574 | ours   |
| 4 | ours `impr_rvv_v3` (LMUL fork)           |  24,156,367 | ours   |
| 5 | ours `hand_v0` (baseline)                |  27,118,799 | ours   |

## Attainment = expert_cycles / our_cycles  (>1 = we beat the expert; <1 = expert is faster)

| our variant            | our cyc     | vs OpenBLAS | vs XNNPACK |
|------------------------|-------------|-------------|------------|
| hand_v0 (baseline)     |  27,118,799 |      0.0031 |     0.0038 |
| impr_rvv_v5 (vfmacc)   |     135,574 |      0.6232 |     0.7502 |
| impr_rvv_v3 (LMUL)     |  24,156,367 |      0.0035 |     0.0042 |

## Reading
- Our **vfmacc fork (v5)** is the only competitive variant: **0.62× OpenBLAS / 0.75× XNNPACK** —
  i.e. ~1.6× slower than the best expert, same order of magnitude. It restructured the inner loop
  into `vfmacc.vf` outer-products (8065 `vfmacc.vf` in its histogram), which is exactly the shape
  both experts use.
- The **baseline (hand_v0)** and **LMUL fork (v3)** sit ~200–320× off the expert (attainment
  ~0.003–0.004): they emit `vfmul.vv/vfadd.vv` reductions with heavy `vslideup`/`vmv` shuffling and
  almost no vector reuse, so the spike instruction count explodes.
- Expert ordering: OpenBLAS (8×8 register-blocked, LMUL=2) edges out XNNPACK (1×NR, LMUL=4) at this
  small square shape, mostly from the wider M-blocking amortizing the B-broadcast loads.

## Reproduce
```
.venv/bin/python -m merlin.kernels.ceiling_drivers.run_expert_gemm   # appends ceiling.jsonl rows
.venv/bin/python -c "from merlin.kernels import attainment; \
  [print(r) for r in attainment.compute('output/kernels/ceiling/ceiling.jsonl','runs/rvv_experiment')]"
```
Drivers: `merlin/python/merlin/kernels/ceiling_drivers/{openblas_sgemm,xnnpack_gemm}_driver.c`
(+ `common.h` / `src/xnnpack/gemm.h` shims, `run_expert_gemm.py`).
