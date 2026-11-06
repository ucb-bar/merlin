# TinyLlama W8A8: fill the 64-row matrix tile with independent requests

This is a source-preserving **next experiment**, not a measured speedup or a
passing TinyLlama model gate. The current true-source capture is one sequence
of eight tokens (receipt SHA256
`f8e4d1652d606492956504551cc4b10c848a051ac59f662b457ec731a50c92ee`,
`model.mlir` SHA256
`0cb6ffc2bc89c5beedfdedd4afa321dd51eed81dc926b1376d2c1f82bee08428`).
Its 155 explicit signed i8 × i8 → i32 Linear contractions
have five shapes: 44 each of `(M,N,K)=(8,2048,2048)`, `(8,256,2048)`, and
`(8,5632,2048)`; 22 of `(8,2048,5632)`; and one `(8,32000,2048)`.
Counting `M×N×K` yields **8,275,361,792 valid integer MACs**. Every N and K
is divisible by 64. These are the integer roots only: float attention,
normalization, dynamic activation quantization, and output work still run.

The pinned OPU (`VLEN=512`, `DLEN=256`) and compact VME (`VLEN=512`,
`DLEN=128`) both use a **64-row logical int8 tile**. The current M=8 shape
uses only 8/64 rows. The cost model treats a leading batch axis as separate
matrix calls, so adding `batch=8` to an `(8,N,K)` contraction by itself does
not improve tile fill. TorchAO 0.18's `Int8Tensor` Linear implementation,
however, explicitly reshapes quantized activations to
`(-1, x_vals_int8.shape[-1])` before `int_scaled_matmul` and reshapes the
per-token scales to `(-1,1)`. A source capture with eight independent
eight-token sequences should therefore expose `(64,N,K)` integer roots if
Model2MLIR preserves that flattening. That shape must be inspected in the new
receipt; it is not inferred from the batch axis alone.

| Integer-root case | Valid MACs | Padded logical slots | Static tile fill |
| --- | ---: | ---: | ---: |
| Current B=1, S=8, M=8 | 8,275,361,792 | 66,202,894,336 | 12.5% |
| B=8, S=8, eight separate M=8 calls | 66,202,894,336 | 529,623,154,688 | 12.5% |
| B=8, S=8, flattened Linear M=64 | 66,202,894,336 | 66,202,894,336 | 100% |

The table assumes all 155 shapes retain their N and K values and all are
routed. The last row is a **projection** until the B=8 source MLIR and emitted
routing sidecar prove it. It does not imply 100% physical PE duty. For the
declared units, whole-inference valid-MAC effective peak is
`routed_valid_MACs / (physical_MAC_cells × target_inference_cycles)`, using
1,024 cells for OPU and 256 for compact VME. Call-window effective peak uses
the same numerator and measured cycles inside those exact calls. Enabled-cell
duty needs actual enabled-cell-cycle counters; none exist in the current OPU
FPGA bitstream.

For one representative `q_proj` shape `(M,N,K)=(8,2048,2048)`, Merlin's
current geometry-derived `matrix_route_cost.estimate` produces the following
unit-cycle estimates. Its OPU geometry is `(tile_edge=64, datapath_bytes=32)`;
compact VME is `(64,16)`. Both route in all three cases. These are model
estimates, not Spike or FPGA timings.

| Unit | One M=8 call | Eight separate M=8 calls | One flattened M=64 call | Estimated eight-request call reduction |
| --- | ---: | ---: | ---: | ---: |
| OPU | 280,576 | 2,244,608 | 409,600 | 5.48× |
| Compact VME | 1,069,056 | 8,552,448 | 1,212,416 | 7.05× |

The installed TorchAO `Int8DynamicActivationInt8WeightConfig` was probed on
the pinned Torch `2.16.0.dev20261104+cpu` with one CPU thread. A quantized
`Linear(64,128,bias=True)` and an actual-K `Linear(2048,256,bias=False)`
each gave **bitwise equal** results for one B=8,S=8 call versus stacking eight
B=1,S=8 calls, with maximum absolute difference zero. The actual-K case
repeated bitwise on independent seeds `20261106` and `991`. The installed
TorchAO Linear implementation file has SHA256
`23e228095a52b99043630bcd799d217e38be62b6839037484b2831e41034f80c`.
That checks the
TorchAO Linear boundary for two sizes. It does not certify all 22 Llama
layers or CPU kernel choices at every size. The installed Llama attention
keeps a batch dimension in Q/K/V and its causal mask, so its source structure
does not mix sequences; its implementation file has SHA256
`31bf660a663259134324bc65da4e155951dc89c5ca46471d2325a9938e859e26`.
Full-model equality must still be measured.

The captured quantized weight file is 1,299,134,608 bytes and should be
shared across requests; re-capture must verify identical weight hashes. The
f32 logits alone grow from `8×32000×4 = 1,024,000` bytes to
`8×8×32000×4 = 8,192,000` bytes. Other activations, compiler scratch, and
the target binary also grow; FireSim memory fit must be measured. The full
model already exceeds physical Kodiak's 512 MiB memory with weights alone,
so batching does not make it a physical-chip workload.

## Required experiment

1. Set `M2M_BATCH=8 M2M_SEQ=8` on
   `examples/workloads/causal_lm_section/loader.py` to capture a
   deterministic `(8,8)` token-ID input with eight independent sequences.
   Keep the same checkpoint and TorchAO W8A8 configuration. Before capture,
   compare the complete eager batched logits with eight separate eager calls;
   record exact tensor hashes and numerical differences. A batch-size change
   may select different float kernels, so bitwise full-model equality is not
   assumed from the Linear probe.
2. Verify the new capture receipt and framework golden. Inspect all 155
   integer roots: require signed i8 operands, i32 accumulators, M=64, N/K
   unchanged, and no opaque or second quantization pass. Check that the
   output shape and per-token scales still distinguish all 64 tokens.
3. Compile OPU and compact VME from that exact source. Verify emitted
   routing sidecars and valid/padded slots; do not substitute this static
   projection. Compare shipped images with scalar twins on pinned Spike,
   including complete output extraction and a full TorchAO-golden gate.
4. On the matched OPU FireSim bitstream, compare B=8 throughput with eight
   serial B=1 requests using target cycles. Report valid MACs per inference
   cycle and per OPU call-window cycle. Compute physical PE duty only from
   enabled-cell-cycle counters if a counter-equipped bitstream is available.
   Report batching wait time and per-request latency so the throughput gain
   does not hide a latency cost.

A smaller logical tile is a different hardware experiment: these providers
derive tile edge from their VLEN512 configuration. A VLEN or array change
needs a new fitted board image, source pin and counter calibration. It cannot
be credited as utilization of the existing OPU or compact VME bitstream.
