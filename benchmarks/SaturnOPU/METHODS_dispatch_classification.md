# Dispatch-classification methodology

This document specifies how each horizontal bar in
`per_model_decomposition_over110.png` (and its variants) is computed.
The figure's categories are derived mechanically from the **linked
RISC-V assembly dumps** of the compiled IREE binaries (one `.s` file
per model, obtained with
`--iree-hal-dump-executable-intermediates-to=<dir>`). The only source
of truth is the dispatch function names and the OPU opcodes inside
each dispatch body. The classifier is a single Python script:

> `benchmarks/SaturnOPU/classify_dispatches.py`

Reviewers can re-run it and verify every number end-to-end.

## 1. What counts as a "dispatch"

A **dispatch** is a function body that IREE emits to execute one fused
region of the model at runtime. Dispatch function names match one of:

| name pattern | meaning |
| --- | --- |
| `<modelname>$async_dispatch_<N>_<op>_<shape>_<dtypes>` | a fused-region body emitted by IREE for an operator call |
| `_encoding_<N>_encode_<shape>_<dtypes>` | a compile-time pack / tile / layout-materialization body |
| `_initializer_<N>_dispatch_<N>_...` | a one-time constant-folding / initializer body |

Everything else in the ELF — math-library helpers (`fma`,
`__truncsfhf2`, `__extendhfsf2`, `__gnu_h2f_ieee`, `__math_invalidf`,
…), IREE HAL runtime helpers (`iree_hal_executable_library_query`),
and local labels (`.L...$local`) — is **NOT a dispatch**. These
helpers get linked into the dispatch ELF by the compiler for math/FP16
support and do not correspond to user-visible computation. The
classifier excludes them from both the numerator and denominator of
every percentage.

Empirical check: across all 6 benchmark models the filter drops
**exactly 33 helper symbols** per binary. That constant is the
complete list of math/runtime helpers the IREE RISC-V toolchain links
into each dispatch ELF, and the classifier reports the before/after
count for every model so the filter is auditable.

## 2. OPU classification rule

Inside each dispatch body, count occurrences of the Saturn OPU
`VOPACC` outer-product instruction encoding:

```
.insn r 87, 2, 81, rd, rs1, rs2
```

(opcode 87 = `0x57`, funct3 = 0x2, funct7 = 0x51). The
`VOPACC` count determines whether the dispatch is OPU-accelerated and
what tile shape the compiler chose for it:

| VOPACC count | dispatch-name pattern | segment | what it is |
| --- | --- | --- | --- |
| ≥ 4 VOPACCs in body | `matmul_like_*` | `encoding_32x32_tile` | encoding-resolver emit of a 32×32 sub-tile (4 unrolled VOPACCs × 16×16 each) |
| 1 VOPACC in body | `matmul_like_*` | `encoding_16x16_tile` | encoding-resolver emit of a 16×16 tile |
| ≥ 1 VOPACC in body | `matmul_*` (no `_like_`) | `inline_vopacc` | LLVM's `vector.contract` pattern replaced a generic vector-matmul's inner MAC with a single 16×16 VOPACC |

The two 16×16-tile paths (`encoding_16x16_tile` and `inline_vopacc`)
share a color and legend entry in the figure because the hardware work
they produce is identical (single 16×16 outer-product VOPACC). They
differ only in the compile path. Both are treated as `AOT 16×16 OPU`
visually.

A matmul/matvec dispatch with **zero VOPACCs** in its body is
classified as `rvv_matmul`: the compiler did not inline any OPU
instruction, and the matmul runs as plain RVV `vmacc`/`vwmacc`
multiply-accumulate loops.

## 3. Non-OPU elementwise sub-classification

For non-matmul dispatches, the segment is chosen by the **dtype
signature** that IREE embeds in the dispatch name. This is what IREE
itself decided about the fused region — not a guess about the
high-level PyTorch op. Ordered from most specific to most general:

| segment | matches | represents |
| --- | --- | --- |
| `data_movement` | `_encoding_*_encode_*`, `_initializer_*` | compile-time pack/unpack bodies; only execute once on first invocation |
| `direct_conv` | `conv_*` dispatch name, zero VOPACCs | convolution lowered to RVV spatial loops (no im2col→mmt4d→OPU) |
| `rvv_reduction_softmax_norm` | `reduction_*`, `softmax_*`, `_norm*` | LayerNorm + Softmax + RVV integer reductions |
| `elementwise_multi_dtype` | ≥3 dtype suffixes in the name (e.g. `_i8xf32xf32xi8xf32`) | a fused region that takes multiple-dtype inputs — typically BN + scale + bias |
| `transpose_reshape` | `elementwise_transpose*`, `generic_*` | pure memory-layout movement (transpose, reshape, memcpy) |
| `quantize_f32_to_i8` | elementwise name ending `_f32xi8` | `f32→i8` quantization at op boundaries |
| `dequantize_i8_to_f32` | elementwise name ending `_i8xf32` or `_i32xf32` | `i8→f32` or `i32→f32` dequantization before a float op |
| `requantize_i32_to_i8` | elementwise name ending `_i32xi8` or `_i8xi32` | `i32` accumulator → `i8` output requantization after each matmul |
| `activation` | elementwise body with just one dtype suffix (`_i8`, `_f32`) | ReLU / GELU / SiLU / Sigmoid |
| `elementwise_other` | residue — unmatched elementwise bodies | rare; flag if this grows |

The classifier reads the name, not the MLIR or graph annotations, so
the mapping is objective: two different high-level ops that fuse into
the same dtype signature end up in the same bucket. This is an
acceptable loss of precision for a paper figure; if the breakdown
needs to be sharper the fix is to also read the `sources/*.mlir`
side-channel IREE emits.

## 4. Granularity

A **dispatch is an IREE fused region**, not a PyTorch `nn.Module`. The
mapping is many-to-many:

- One PyTorch op can produce *several* dispatches (a quantized Conv
  often lowers to `quantize` + `mmt4d` + `dequantize` as three
  dispatches, plus pack/unpack if data-tiling is on).
- Several PyTorch ops can fuse into *one* dispatch (`BN + ReLU +
  residual add` frequently collapses into a single elementwise
  dispatch that the classifier will tag as
  `elementwise_multi_dtype` or `activation` depending on the dtype
  signature).

Consequently, **dispatch share is a structural metric, not a
cycle-time metric**. A model with 3 OPU matmul dispatches out of 10
total dispatches (MLP-Wide) might still spend the majority of its
runtime cycles in those 3 OPU dispatches; the right-side annotation in
the figure (`X% OPU, N/M dispatches, P params`) reports the structural
ratio + the parameter count so readers can reason about that.

For cycle-weighted "% OPU time" we report the measured end-to-end
OPU-vs-RVV speedup separately (Table: `sweep_iters.csv`), which is the
honest runtime metric.

## 5. How the percentages are computed

```
bar_width(model M, segment S) = count(dispatches in M classified as S)
                                / total_dispatches(M)

OPU dispatch share(M) = Σ bar_width(M, S) for S ∈ OPU_SEGMENTS
                      = count(dispatches in OPU_SEGMENTS) / total_dispatches(M)
```

`OPU_SEGMENTS` is the set listed under "OPU classification rule" above
(encoding_*_tile, runtime_*_tile, fused_qdq, inline_vopacc). All other
segments — including `direct_conv`, `rvv_matmul`, the reduction
family, and every granular elementwise sub-category — are non-OPU.

## 6. Known limitations

- The classifier does not currently distinguish 1×16 / 4×16 / 8×16
  narrow-matmul fallback tiles. They all fold into
  `encoding_narrow_tile` (labeled *AOT Mx16 OPU* in the figure). No
  model in the present figure exercises this path enough to warrant a
  finer split.
- The `fused_qdq` OPU path (pre-defined in the palette) is not
  currently active on any shipped binary; the infrastructure is kept
  for future runs where it might be.
- The `rvv_reduction_softmax_norm` bucket groups LayerNorm, Softmax,
  and reduce-sum together because they share the same codegen
  pattern (`vfredusum`-free reductions). If per-op breakdown becomes
  necessary, extend the classifier with name-based splits.

## 7. Reproducing the figure

```bash
# 1. Dump linked assembly for each model (one-time, ~2 min per model):
iree-compile <model>.q.int8.mlir <all the profile flags> \
  --iree-hal-dump-executable-intermediates-to=/tmp/verify_all/<model>_DIR \
  -o /dev/null

# 2. Regenerate both CSVs from the dumps:
uv run benchmarks/SaturnOPU/classify_dispatches.py \
  --dumps-root /tmp/verify_all

# 3. Render the figure:
uv run benchmarks/SaturnOPU/plot_model_decomposition.py \
  --include-models=opu_bench_convnet,tinyllama,yolov8_nano,opu_bench_vit,dronet,mlp_fast \
  --rename=opu_bench_vit:ViT,mlp_fast:MLP \
  --figsize-w=4.5 \
  --out-name=per_model_decomposition_over110
```

The intermediate CSVs
(`model_dispatch_decomposition.csv`, `per_model_summary.csv`) live
under `benchmarks/SaturnOPU/` and can be inspected directly.
