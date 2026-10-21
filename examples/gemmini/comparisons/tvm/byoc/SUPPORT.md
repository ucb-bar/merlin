# Operation coverage and hardware limits

This backend accepts semantic **Relax**, with dense signed INT8 operands,
INT32 contraction results, positive static shapes and K≤131071. It implements
the C1 variant requested in the task. It does not silently quantize FP32/BF16,
change PyTorch integer promotion, or claim a complete ATen frontend.

“CPU” below means the operation remains ordinary Relax and is available to
TVM's legalization, fusion and LLVM codegen. It does not mean that every ATen
overload imports in the pinned frontend, or that every shape fits baremetal.
Unsupported frontend names must be decomposed by a frontend that preserves
their semantics, or rejected. The static exporter still rejects unsupported
control flow, noncompact external buffers, hidden allocators and excessive
memory. Host support is deliberately distinguished from native offload.

## Arithmetic: all 23 requested operations

| Requested operation | Implementation or justified limit |
| --- | --- |
| `mm` | Native INT8×INT8→INT32 matmul; other dtypes use TVM CPU. Positive static M/N, K≤131071, representable DMA strides. Matmul, tails and worst-case signed inputs are tested. |
| `bmm` | Native per-batch contractions after ordinary Relax slicing/reshape/concat. Equal static rank-three batches of 1–64 are admitted. Shared rank-two weights use the existing linear flattening pass. Other ranks/broadcast batches and batches >64 remain CPU, avoiding unbounded graph expansion. |
| `addmm` | The equivalent Relax matmul+add (alpha=beta=1) forms a BYOC composite. TVM fuses it into one TensorIR function, with native matmul and a CPU INT32 epilogue inside the tiled function. Generic alpha/beta stay explicit CPU arithmetic. This version does not move bias into the reduction initializer or claim bias preload fusion. |
| `convolution` | Native static INT8 NCHW/OIHW **2D**, groups=1, INT32 output, through the existing semantic im2col/matmul/restoration decomposition; strides, dilation and zero padding retain their existing contract. Packing/restoration are TVM CPU work. 1D/3D/grouped/transposed and floating convolutions remain CPU. Full im2col memory is explicit and subject to export admission. |
| `convolution_backward` | Not added to the ATen frontend or device backend. Training gradients normally require floating operands, multiple output gradients, grouped/transposed convolution and output-padding rules. The pinned array is INT8/INT32. A mathematically possible convolution decomposition does not justify quantizing those gradients or borrowing the forbidden conv FSM. Explicitly expressed supported integer sub-contractions can still offload. |
| `add.Tensor`, `sub.Tensor` | CPU Relax add/subtract, including broadcasting. General operands may be FP32, INT64 or arbitrary INT32. The pinned configuration has **no accumulator-load scaler**; its INT8 input scaler rounds and saturates. Scaling by −1 is not exact INT8 negation at −128. Therefore the proposed generic accumulate-on-load replacement is not a semantics-preserving implementation. |
| `add.Scalar`, `sub.Scalar` | CPU broadcast arithmetic. The DMA unit does not generate arbitrary typed constants. Loading a synthesized matrix just to add a constant costs a host fill and round trip; native load scaling also has the dtype/rounding restrictions above. The exact ATen scalar overloads are absent from the pinned exported-program map; a semantic frontend decomposition is required. |
| `mul.Scalar`, `div.Scalar`, `neg` | CPU multiply/divide/negative. Arbitrary scaling is not exact through the saturating INT8 input scaler, and full-width INT32 readout bypasses the scaled narrow-output path. FP division, integer promotion, rounding modes and division by zero cannot be replaced with an unrestricted reciprocal scale. `neg.default` imports; the scalar multiply/divide overload names need frontend decomposition. |
| `relu` | TVM CPU ReLU, fused into the selected matmul composite when there is no leaked intermediate. On this RTL, **full-width accumulator output bypasses activation/scaling**. Enabling readout ReLU would change the INT32 output contract to narrowed/clipped data. Standalone and floating ReLU also remain TVM operations. |
| `mean.dim` | CPU reduction with its declared dtype and divisor; tested alongside scalar arithmetic. Arbitrary floating means and INT64 reductions do not match an INT8 ones-vector matmul. No new reduction order or quantization is imposed. |
| `avg_pool1d`, `avg_pool2d`, `avg_pool3d` | CPU Relax `nn.avg_pool1d/2d/3d`. Exact padding counts, ceil mode and divisor overrides matter; Gemmini's pooling datapath is **max pooling**, not a general average reducer. Window packing plus GEMM plus scaling would add work and alter floating/rounding semantics unless a separate narrow integer contract were introduced. The pinned ATen map directly includes only 2D. |
| `adaptive_avg_pool1d`, `_adaptive_avg_pool2d`, `_adaptive_avg_pool3d` | CPU Relax `nn.adaptive_avg_pool1d/2d/3d` when supplied semantically. Adaptive windows have position-dependent bounds/divisors. The exact requested ATen names are absent from the pinned exported-program map; it has `adaptive_avg_pool2d.default`, which is a different spelling. No import coverage is inferred from Relax availability. |
| `avg_pool2d_backward`, `_adaptive_avg_pool2d_backward` | Not added to frontend/backend. These require overlapping gradient scatter-add, divisor and boundary rules, usually with floating gradients. DMA is a row mover, not a general indexed floating atomic reducer. A proper training decomposition is needed; silently using inference pooling is incorrect. |
| `col2im` | Not added. Overlapping patches require indexed accumulation and carefully defined output shape/stride/padding. DMA does not perform general scatter-add; a host index program and an accumulation dtype/range contract are necessary. This backend's im2col forward packing is not col2im support. |

## Movement: all 15 requested operations

| Requested operation | Implementation or justified limit |
| --- | --- |
| `clone` | Pinned frontend treats clone as identity in functional Relax. Returned aliases are copied into distinct caller output slots by static export. Observable mutation/storage-identity behavior is outside this functional ABI. |
| `repeat`, `cat` | CPU Relax tile/concat. DMA-only implementations still need host indexing and repeated commands; no benefit is established for small strided copies. Concat is exercised by the BMM decomposition. |
| `slice_scatter`, `select_scatter` | Exact ATen overloads are unsupported in the pinned importer. A pure Relax scatter/slice-update decomposition can be supplied by a frontend, with bounds and update semantics proved there. No arbitrary indexed DMA scatter primitive is exposed by this backend. |
| `fill.Scalar` | CPU Relax full with the declared dtype/shape; the pinned frontend has this overload. A constant fill has no matrix multiplication to accelerate. |
| `full`, `full_like` | CPU Relax full/full_like. `full` is present in the FX frontend, but these exact overloads are not both present in the exported-program map. Frontend support and device support remain separate. |
| `constant_pad_nd` | CPU Relax `nn.pad(..., pad_mode="constant")` for shapes supported by TVM. Exact ATen overload not registered here. DMA cannot synthesize arbitrary typed padding values without explicit host materialization. Native conv decomposition independently handles zero padding. |
| `reflection_pad1d`, `reflection_pad2d`, `reflection_pad3d` | CPU Relax reflect padding when represented with valid widths. Exact ATen names are absent from the pinned map. Reflection is a noncontiguous index transform; it is not a single rectangular DMA copy. |
| `replication_pad2d`, `replication_pad3d` | CPU Relax edge padding when represented semantically. Exact ATen names are absent. Edge duplication similarly needs index generation/materialization. |
| `_to_copy` | CPU Relax cast/copy; the pinned frontend contains `_to_copy.default`. INT8→INT32 is an exact cast. Accumulator readout is already INT32; moving INT8 data into accumulator SRAM is not automatically a general sign-extending cast instruction, and staging/fencing just to widen is unnecessary overhead here. |

## Views and metadata: all 11 requested operations

| Requested operation | Implementation or justified limit |
| --- | --- |
| `as_strided`, `diagonal` | General overlapping/strided aliases are outside the static compact-buffer ABI, and the exact ATen names are absent. A frontend may materialize their values as an explicit index computation. They cannot be declared universally free: diagonal data is generally not compact and as_strided may overlap. |
| `permute` | CPU Relax permute_dims. The pinned frontend imports it. Static export may materialize a copy because device operands must be compact; arbitrary transposes are not free pointer metadata. |
| `scalar_tensor` | A typed Relax constant/host scalar once a frontend supplies it. This exact ATen overload is not in the pinned exported-program map. Existing model-specific ONNX translations are separate from this backend. |
| `select.int`, `slice.Tensor` | CPU Relax take/strided_slice. Both frontend entries exist. Materialization may be needed for the compact ABI; slicing is exercised by BMM. |
| `split_with_sizes` | Exact ATen overload absent. Semantic Relax split/tuple projections can represent static splits, with the existing export rules for tensor tuples. The importer entry `split.Tensor` alone does not prove this overload. |
| `squeeze.dim`, `unsqueeze`, `view` | Ordinary Relax reshape/expand_dims when statically legal; pinned frontend entries exist. BMM exercises reshape. No Gemmini instruction is needed, though static export may compile copies instead of preserving arbitrary runtime aliases. |
| `squeeze.dims` | Semantic Relax squeeze supports an axis list; this exact ATen overload is absent from the pinned map. A frontend must normalize it rather than assuming the `squeeze.dim` registration handles every overload. |

## Linear maps: both requested operations

| Requested operation | Implementation or justified limit |
| --- | --- |
| `mean.default` | CPU Relax mean once represented semantically. This exact exported-program overload is absent; FX `mean` is present. General floating means cannot preserve their semantics through an INT8 ones-vector multiplication and narrow readout scale. |
| `sum.dim_IntList` | CPU Relax sum; this frontend overload exists and CPU reduction is tested. PyTorch integer sum commonly promotes to INT64, while the array accumulator is INT32. A restricted INT8→INT32 sum with a proved bound could use a ones-vector GEMM, but this backend does not reinterpret arbitrary reductions as that narrower operation. |

## Source facts behind these limits

The selected physical lineage is Gemmini `63f0b68a68f185a016a2993f2fb32e1661ea9e6c`.
Its [Configs.scala](https://github.com/ucb-bar/gemmini/blob/63f0b68a68f185a016a2993f2fb32e1661ea9e6c/src/main/scala/gemmini/Configs.scala)
declares INT8 operands, INT32 accumulators, 20-bit PE output, 16×16 geometry,
max pooling, saturating/rounded input scaling and `mvin_scale_acc_args=None`.
[AccumulatorScale.scala](https://github.com/ucb-bar/gemmini/blob/63f0b68a68f185a016a2993f2fb32e1661ea9e6c/src/main/scala/gemmini/AccumulatorScale.scala)
keeps `full_data` separate from activated/scaled/clipped `data`;
[Scratchpad.scala](https://github.com/ucb-bar/gemmini/blob/63f0b68a68f185a016a2993f2fb32e1661ea9e6c/src/main/scala/gemmini/Scratchpad.scala)
selects the full-width path for full accumulator readout. The pinned vendor
`gemmini.h` routes WS matmul, conv and residual helpers through loop commands;
this implementation uses only explicit OS load/preload/compute/store commands.

The frontend observations above come from the local TVM pin's
`python/tvm/relax/frontend/torch/exported_program_translator.py` and
`fx_translator.py`. In particular, the pinned exported-program `bmm.default`
and `matmul.default` converters request FP32 output. This backend's native
INT8→INT32 tests enter through explicit semantic Relax, as do the existing
integer recipes. These results do not claim that stock `torch.export` will
automatically choose this integer contract for a floating model.
