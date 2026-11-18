---
name: quantization
description: "Read a test case's quantization (scales, zero points, bias, rounding, clamp, per-tensor vs per-column, the target's readout order) from public inputs, keep the integer pipeline it defines exact (i8 x i8 accumulated in i32, bias in the accumulator, f32 scale with half-even rounding, rectifier, saturation, single-rounding residual adds), lower it with the library's proven readout APIs and fuse it into the accelerator's readout, and localize numeric mismatches. Use when a case declares an epilogue or scales, when lowering a quantized contraction or its requantization, or when a self-check reports numeric failures."
---

`$SKILL` = the directory holding this file. Run commands from the workspace root.

Settled policy: every contraction, attention included, runs int8 x int8 -> i32; f32 (never bf16) only
where the source keeps it (float-island ops, f32 outputs). Never invent a scale, change a rounding
or widen a tolerance.

1. **Read what the case declares.**
   - `python $SKILL/scripts/quant_params.py case NAME` prints operand dtypes, epilogue stages in
     order, scales (per tensor; per column as equal-scale runs; residual `lhs_scale`/`rhs_scale` and
     `bound_lsb`; `requant_shift`), output dtype, numeric policy, accumulator bound and the integer
     pipeline they imply. `--prove` adds the library's full-domain readout proofs. Also read
     `integer_partial_sum_bound`: its `signed_positive_limit` can be narrower than i32.
   - `quant_params.py ir FILE.mlir [--path]` (model cases, `public_sources/`): every integer
     contraction's accumulator chain as the library's fusion matcher reads it (scalar = per tensor,
     vector = per channel; a function argument is a stored tensor not in the file), quantize zero
     points/ranges, and float contractions with or without captured quantization evidence.
     `--path` adds the readout cut the target absorbs exactly, with each refusal.
   - `quant_params.py target`: the output path in `target_contract.yaml` (readout order, rounding,
     clamp, columns per scale, exits) and `quantization.methodology` of the public software spec;
     `from_output_path` means "the value that path states".
   - Only public data exists here: the self-check withholds expected values (`first_mismatch` has
     `your_value` only); `qa/verdict.json` keeps status, policy, mismatch count. Do not look for
     goldens or hidden cases.
2. **Preserve the semantics** (each rule with its library line: `reference/semantics.md`):
   accumulate exactly in i32 from 0; add the i32 bias in the accumulator; `acc_scale` is `f32(acc)`
   times the f32 scale, rounded half to even once; ReLU, then saturation to the output dtype;
   `requant` (shift) rounds half up, a different operation; a residual add rounds once, after the
   f32 sum; quantize is `roundeven(x * f32(1/scale)) + zp`. Keep stage order. Do not fold two f32
   scales into one unless `prove_scale` accepts it. An i32 output has no readout stage.
3. **Implement with the library** (exact calls, tested: `reference/library_api.md`). Fuse the
   readout into the contraction (skill `fusion`): the output fusion pass moves bias, scale, ReLU,
   clamp and pool into the accelerator's narrowing readout where its proof holds. A full-width
   readout may write the raw accumulator, dropping a declared ReLU or scale. Grades flag readout
   left on the host (`epilogue_store_path`, model rows) and host loops doing the unit's arithmetic
   (`host_compute`; fails a passing case). For host readout use `integer_readout.emit_readout`.
4. **Debug a mismatch** (`reference/debugging.md`): read the failing row's `numeric` and `tiers`;
   rule out `output_never_written` / `output_shape_mismatch` / `missing_outputs` first; classify
   `mismatch_indices` (whole columns: scale or bias column mapping; rows or tiles: layout, stride,
   tail; scattered +-1: rounding or scale order; large errors at the extremes: saturation or
   overflow). L0 failing means the command buffer computes the wrong thing; L0 passing with a
   device tier failing means the emitted artifact differs. Recompute suspects with the library's
   reference arithmetic on your own values. For models, bisect with the public model-derived layer
   cases and `agent_model_check.py replay-kernel --model M --program P --group ID`.
5. **Performance.** int8 operands are a quarter of the bytes of i32/f32: more K and N fit in local
   memory and the array stays fed. Narrowing on the device stores 1 byte per output, not 4, and
   saves a host pass. Each scale change costs a readout setting; `partition_column_scales` gives the
   equal-scale column runs. Measure with skill `roofline-and-utilization`; compare variants with
   skill `ab-library-pass` (fewer cycles, unchanged numerics).
