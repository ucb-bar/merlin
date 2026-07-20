# AGENT.md — merlin/python/tests/data/cca_asm

## Purpose

REAL RVV disassembly fixtures (`llvm-objdump -d --no-aliases` text, genuine encoding bytes) used to
pin the **CCA asm-level abstraction's faithfulness**: that `cca.lift_asm` reads the expert-win
properties — accumulator-residency, register-block (MR, NR), and whether NR tracks vsetvlmax —
correctly off an expert GEMM and off our own compiler output.

## Provenance (how each was produced — all SPIKE/HOST toolchain, no K1)

- `openblas_sgemm_rvv.objdump` — `<openblas_sgemm_kernel>` from the OpenBLAS expert ceiling driver
  (`ceiling_drivers/openblas_sgemm_driver.c` + `sgemm_kernel_8x8_zvl128b.c`), built by
  `run_expert_gemm._build` with the Saturn bare-metal flags. Register-blocked, accumulator-resident.
- `xnnpack_f32_gemm_rvv.objdump` — `<xnn_f32_gemm_ukernel_1x4v__rvv>` from the XNNPACK expert driver
  (`ceiling_drivers/xnnpack_gemm_driver.c` + `f32-gemm-1x4v-rvv.c`). MR=1, NR = vsetvlmax (VL-loop).
  Extracted from a LINKED bare-metal ELF, so branch displacements are RESOLVED: `spans_reliable()`
  is True and `envelope.calls_in_loop` is a trustworthy 0.
- `xnnpack_qd8_gemm_rvv.objdump` — `<xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4v__rvv>` from the
  vendored XNNPACK int8 GEMM ukernel `qd8-f32-qc8w-gemm-1x4v-minmax-rvv.c`. Cross-compiled to a
  `.o` with the SpacemiT K1 clang (`--target=riscv64-unknown-linux-gnu -march=rv64gcv_zfh_zvfh
  -mabi=lp64d -O3 -DNDEBUG`, `-I ceiling_drivers` for the header shim), disassembled with the repo's
  `llvm-objdump` (`decode/objdump.py::disassemble_text`). W8A8 datapath: a WIDENING `vwmacc` MAC
  accumulating int8×int8 products in i32 — `cca.lift_asm` reads `compute.widening=True`,
  `accumulator_dtype=i32`.
- `xnnpack_f16_gemm_rvv.objdump` — `<xnn_f16_gemm_minmax_ukernel_7x4v__rvvfp16arith>` from the
  vendored XNNPACK f16 GEMM ukernel `f16-gemm-7x4v-minmax-rvvfp16arith.c`, same SpacemiT-clang
  cross-compile (plus a tiny prelude typedef'ing `xnn_float16 = _Float16` and
  `struct xnn_f16_minmax_params`, which the ceiling header shim does not carry).
  **fp16 NUMERICS CAVEAT**: this ukernel accumulates NATIVELY in f16 (`vfmacc.vv` at e16, NOT a
  widening `vfwmacc` to f32), so its result is numerically NON-comparable to our f32-accumulate
  datapath — a K-length f16 reduction drifts where f32-accumulate does not. Any CCA/expert-wall
  comparison against this fixture inherits that caveat: `compute.accumulator_dtype=f16`,
  `widening=False` (unlike int8's f32-accumulate-via-i32 widening).

- **Per-family teacher fixtures** (`xnnpack_gelu_rvv.objdump`, `xnnpack_sigmoid_rvv.objdump`,
  `xnnpack_reduce_rvv.objdump`, `xnnpack_clamp_rvv.objdump`, `xnnpack_vbinary_add_rvv.objdump`,
  `xnnpack_vbinary_mul_rvv.objdump`) — the non-GEMM XNNPACK RVV ukernels that teach the beam's PER-OP
  teacher (`merlin.rvvgen.wholemodel_proposer.FAMILY_TEACHERS`). Harvested by
  `build_tools/scripts/harvest_xnnpack_fixtures.py`: each family's `*rvv*.c` ukernel cross-compiled to
  a `.o` with the SpacemiT K1 clang (`--target=riscv64-unknown-linux-gnu -march=rv64gcv_zfh_zvfh
  -mabi=lp64d -O3 -DNDEBUG`, `-I ceiling_drivers` for the `src/xnnpack/*.h` shim + `-I XNNPACK/src`)
  and disassembled with the repo `llvm-objdump` (`decode/objdump.disassemble_text`). What each teaches
  `cca.lift_asm`: vgelu/vsigmoid are inline minimax polynomials -> `activation_vectorization=
  vectorized_polynomial` (vs our scalar libm -> routes to `vectorized_transcendental_activation`);
  f32-rsum -> `reduction_form=vredsum_tree` (vs our scalar accumulate -> `vectorize_reduction`); this
  rsum fixture is ALSO the `softmax` teacher (its row-sum), lifted with op tag `reduce`. `clamp`/binary
  are harvested for completeness (thin CCA diff; no fork is forced if `compare` emits nothing). Regenerate:
  `.venv/bin/python build_tools/scripts/harvest_xnnpack_fixtures.py`. NO XNNPACK primitive exists for
  attention (`sdpa`/`batch_matmul`), `layer_norm`, or gather (`embedding`) -> NO fixture, an honest
  no-teacher record (never a faked divergence).

Both new `qd8`/`f16` fixtures are UNLINKED single-ukernel objects, so their intra-function branch
displacements are still unrelocated (each branch resolves to its own address). The decoder detects
this (`InsnStream.spans_reliable()` is False) and `cca.lift_asm` reports `envelope.calls_in_loop`
as `None` (honestly UNKNOWN) rather than a misleading 0 — the P5a honesty fix. Their dtype-datapath
facets (widening / accumulator_dtype / sew / lmul) are read from the compute stream and remain
trustworthy regardless of loop-structure relocation.
- `ours_baseline_matmul.objdump` — `<forward>` of our FROZEN baseline RVV lowering for a 64^3 f32
  matmul (`apply_rvv_package(hand_v0)` -> `model.o`). vfmul+vfadd (no fma), no register block.
- `ours_accum_resident_matmul.objdump` — `<forward>` of the `accumulator_resident_microkernel`
  impr feature for the same 64^3 matmul. Forms vfmacc but STILL spills the accumulator through the
  stack inside the K loop (whole-register vsNr/vlNre), so the abstraction reads it as
  accumulator_resident=False — the honest, measured gap to the experts.

## Invariants

- These are FAITHFUL real disassembly (the kernel function only, trimmed at symbol boundaries to
  keep them small); the encoding bytes and mnemonics are exactly what the toolchain emitted.
- The test reads structure (loops, vfmacc chains, spills, vtype) from them via the same
  `decode.rvv` decoder used in production — never a regex over the text.
- Regenerate by rebuilding the drivers (see Provenance) and re-extracting the named symbol.
