# Torch 2.16 AVX2 f32 LayerNorm lowering, 2026-11-06

The optional `torchao-cpu-avx2-layer-norm` Merlin pass replaces a complete
captured `aten.native_layer_norm.default` decomposition with the pinned Torch
`2.16.0.dev20261104+cpu` AVX2 f32 width-256 Welford order. It is off by
default. It leaves TorchAO `int8_dyn_act_int8_weight` quantization, signed-i8
activation codes, and the OPU/VME matrix ABI unchanged. It does not change a
golden or tolerance.

## Source contract and trust boundary

Selection requires `MERLIN_TORCHAO_CPU_AVX2_LAYER_NORM=1` (or the named
optional pass), `MERLIN_TORCHAO_CPU_SOURCE_COMMIT` equal to
`7f24a19c0967db04f2d5ba44c1e56eca489932ec`, and
`MERLIN_TORCHAO_CPU_CAPABILITY=AVX2`. The module must declare
`prov.quantization = "int8_dyn_act_int8_weight"`; any present conflicting
`prov.torch_source_commit` or `prov.torch_cpu_capability` fails. The pass
requires a two-reduction native LayerNorm region with f32 contiguous
last-axis width 256, f32 affine vectors, correct mean/variance division,
positive epsilon, and the complete centered-variance/rsqrt/affine dataflow.
Wrong or incomplete native LayerNorm regions fail instead of receiving a
partial rewrite. Region IDs delimit the source operation; they do not select
the numeric policy.

**The environment source commit and capability are assertions by the
runner, not independent capture proof.** The historical exact SpectFormer
`model.mlir` carries only the TorchAO quantization scheme. Its
`capture_receipt.json` records Torch version and the MLIR SHA-256, but lacks
the Torch git commit and CPU dispatch capability. The pass can reject wrong
assertions and conflicting IR metadata; it cannot prove that the old capture
ran under the asserted source CPU from those files alone. A future capture
should record `torch.version.git_version` and
`torch.backends.cpu.get_cpu_capability()` in SHA-bound source metadata.

## Narrow verification

The [bitwise source reproducer](torch216_avx2_ln256_probe_2026-11-06.md)
established the arithmetic order. The new pass was run on the exact 1.7 MiB
SpectFormer TorchAO captured MLIR without compiling the model: it rewrote
all **25** native LayerNorm regions and returned no unsupported region.
Peak RSS was 66,940 KiB; elapsed time 4.21 seconds. The result was also
idempotent on a standalone source-region fixture.

The focused compiler test lowers a representative captured decomposition to
LLVM and compiles the tiny function with LLVM 23 Clang. Its generated host
output matches the unchanged Torch native LayerNorm **bitwise** for each
input, using the trained TorchAO model's actual block-0 norm2 gamma, beta,
and epsilon:

| Same-input filter tensor | Matching output words |
| --- | ---: |
| Trained | 50,176 / 50,176 |
| Independent holdout | 50,176 / 50,176 |
| Independent random | 50,176 / 50,176 |

The generated LLVM IR also compiles to an RV64GCV object with LLVM 23
Clang; that is a compile check, not a device execution or source-parity
claim. The fixture test covers default-off behavior, a renamed region ID,
idempotence, wrong source commit, wrong AVX capability, wrong quantization,
conflicting capture metadata, unsupported width and dtype, and malformed
dataflow. The exact trained/holdout test requires the existing saved tap
archives and checkpoint through explicit environment paths; it skips when
they are absent. No full SpectFormer host, Spike, FireSim, or physical Kodiak
run occurred in this work.

The strict 1,000-logit TorchAO source gate remains open. Patch convolution,
FFT, and other repeated float operations can still cross dynamic
quantization thresholds. A fresh full-model compile and trained+holdout
comparison must run after the concurrent VME route releases memory.
