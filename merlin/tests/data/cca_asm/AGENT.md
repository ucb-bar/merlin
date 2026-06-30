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
