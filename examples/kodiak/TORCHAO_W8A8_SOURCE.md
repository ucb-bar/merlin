# TorchAO W8A8 source contract for SpectFormer

This note binds the trained SpectFormer capture to the integer matrix interface. It is a
source and compiler contract; it does not certify an OPU or VME hardware run.

## Pinned source

- Model2MLIR capture: `spectformer_torchao_w8a8_20261106` in the managed task
  `spectformer-torchao-w8a8`. `model.mlir` SHA256
  `9ae3f772ed2d192899cea502e5dec0098d5abf95f56f1707c7ba3cdfcf5481ab`.
- Capture receipt SHA256
  `bd554a8cba6caf1b9c9c133073b7a839d3e596b70b2304d99c66aa03cd31038d`.
  Merlin verifies the materialized artifact hashes, but the receipt explicitly says
  `source_closure_verified: false`.
- Trained checkpoint SHA256
  `77cacfdbe53f3615666291d2a9c45aa6dd9132e48e0448bee90c3a3f1a116e3e`.
  The source weights and deterministic input are byte-identical to the earlier
  weight-only SpectFormer capture. The earlier capture remains a diagnostic baseline.
- Independent torchAO framework `golden.npy` SHA256
  `a131a6790842bad1e31007382f741536376df1b8d6bfc3252ee82ed09bb572f3`.
  This is the W8A8 reference for this source; it is not an fp32 reference.

The module declares `prov.quantization = "int8_dyn_act_int8_weight"`. Its 41 actual
`aten._int_mm.default` roots have signed i8 operands, an i32 zero init and accumulator,
and `extsi`, `muli`, `addi` bodies. Counting tagged constants and splats would report
123 occurrences and is incorrect. The valid integer matrix work is 1,644,423,168 MACs.

## Arithmetic and backend boundary

The exported activation path computes a dynamic scale per input row: maximum absolute
value divided by 127.5, floored at `1.1920929e-07`; it uses round-to-nearest-even,
zero-point zero, and clamps to `[-128, 127]` before converting to signed i8. The
weights are signed i8 with one scale per output channel. Each integer product
accumulates in i32; the source converts i32 to f32, multiplies the row and channel
scales, then adds the f32 bias. Those scale and bias operations remain outside the
matrix kernel. The next quantized Linear layer dynamically quantizes its own input.

The existing Kodiak OPU and Iris VME shim ABIs consume signed i8 A/B and write
i32 C. Kodiak's compact VME route is under construction and must satisfy this
same interface before it is a model target. Their packers may change layout,
and their instruction words are different, but neither may change the source's
values or apply another quantization pass. The `--dtype int8_w8a8` token names
the torchAO **source recipe**. Matrix routing sees
`signed_int8` as the operand storage format and `int32` as the accumulator format;
the `int8` entry in `quant_formats.registry.yaml` describes the older weight-only
capture and must not stand in for this source recipe.

The imported graph also has 57,802,752 float matmul MACs (including the patch
convolution's 38,535,168) and 157,351,936 float attention BMM MACs. Thus explicit
source integer matrix work is 88.4299% of the 1,859,577,856 dense MACs. The compiler
must preserve the remaining float contractions for framework parity.

## Utilization accounting

For each emitted matrix call, count valid MACs as `batch * M * N * K`. Count issued
matrix slots from the *actual emitted* tile and K-tail instruction trace; **issued-slot
efficiency** is valid MACs divided by those slots. A route decision alone is not an
execution measurement. Record the actual routed valid MACs from the routing sidecar,
the dynamic call/fallback counts from the shim, and hardware matrix-active cycles or
instruction counters where available. Physical array duty needs enabled-cell-cycle
counters: `enabled_cell_cycles / (physical_cells * matrix_active_cycles)` within active
matrix intervals, and `enabled_cell_cycles / (physical_cells * inference_cycles)` for
the complete inference. OPU DLEN256 has 1,024 cells and compact VME DLEN128 has 256
cells per substep. When those counters are absent, report array duty as unmeasured.
Valid-MAC whole-inference effective peak is a separate rate:
`valid_routed_MACs / (physical_cells * inference_cycles)`. Do not substitute
end-to-end speedup for any of these quantities.

As a **projection only**, the current geometry cost model selects all 41 integer
roots for OPU VLEN512/DLEN256: 1,644,423,168 valid MACs and 2,164,260,864 padded
logical tile slots, or 75.9808% projected tile fill. For compact VME VLEN512/DLEN128 it
selects 40 roots, declining only the 1×1000×256 head: 1,644,167,168 valid MACs and
2,147,483,648 padded slots, or 76.5625%. Every SpectFormer integer root has K
divisible by 64, so this workload does not test K-tail handling. A separate VME
128×128×33 biased case with four full logical tiles and signed `-128`/`127` endpoints
passed pinned Zvt Spike. Its direct compact RTL attempt reached `KT BEGIN` but stopped
at the RAM guard before a case verdict, so RTL remains unverified. A new OPU 64×64×33
endpoint case passed the package's 42-shape quick kernel sweep on a fresh extension
built from independently attested exact Saturn `ea373800` source bytes plus the
published Kodiak config patch. The extension SHA is
`fe9d77b85a0a9d9be6a391e9d154226208a75f115ac569505f980960204f8338`;
the 13-file source attestation SHA is
`fb8c30d5e716418850badc0c8d1adec6ca226502850f5fce14599d82a93b35bf`.
The built-in pin checker cannot verify a Git archive without `.git`, so this is an
exact-byte attested simulator test, not a built-in pin-check pass. Neither
microkernel result is a whole-model torchAO verdict.

The acceptance sequence is framework golden → unchanged-source Merlin host output
→ matrix twin on Spike → shipped image on the declared OPU simulator → FireSim on the
matched bitstream. A passing argmax or an old Merlin-derived W8A8 golden cannot replace
the framework-to-source numerical gate.

## Current numerical gate

The unchanged-source Merlin host run completed with all 1,000 outputs finite. Its
`host_source.npy` SHA256 is
`908c7f605b080542e53d0fb4e66eea1d3cc753759b2052a58254c9b7e2e02368`.
Against the independent torchAO framework golden it has cosine `0.99997751`,
relative L2 error `0.00718247`, maximum absolute error `0.08121049`, and the same
argmax (`885`). The strict W8A8 gate **fails**: maximum-normalized error `0.0167634`
exceeds `0.01`, and significant-element relative error `0.663304` exceeds `0.05`.
Comparison covers the full 1,000-element output. This remains a **failed framework
fidelity gate**. A separately labelled target-arithmetic diagnostic may compare a
source-preserving OPU image and scalar twin on exact Spike/FireSim; such a result
cannot be called torchAO framework parity while this gate fails.
