# TorchAO W8A8 arithmetic on Kodiak OPU and compact VME

The current SpectFormer and TinyLlama source captures carry explicit integer
matrix products and explicit quantization arithmetic. Their `prov.quantization`
label is **not** enough to reconstruct that arithmetic. The capture receipt
binds the exact `model.mlir` and independent TorchAO `golden.npy` bytes; use
both hashes when handing a build to Kevin.

| Source capture | Receipt SHA256 | Explicit signed i8 × i8 → i32 roots | K values |
| --- | --- | ---: | --- |
| `spectformer_torchao_w8a8_20261106` | `bd554a8cba6caf1b9c9c133073b7a839d3e596b70b2304d99c66aa03cd31038d` | 41 | 256, 1024 |
| `tinyllama_full22_s8_true_w8a8_int8_w8a8_full` | `f8e4d1652d606492956504551cc4b10c848a051ac59f662b457ec731a50c92ee` | 155 | 2048, 5632 |

In **these exact captures**, TorchAO takes a per-token absolute maximum,
divides by `127.5`, clamps the scale to at least `1.1920929e-07`, then uses
the reciprocal, round-to-nearest-even, zero point zero, and clamp to
`[-128, 127]`. We inspected the emitted operations in both receipted
`model.mlir` files; for example, SpectFormer block 0 `fc1` and TinyLlama layer
0 `q_proj` both contain the `127.5` divisor, epsilon, and clamp endpoints.
Each trained weight channel has its own scale. Matrix products accumulate in
i32, then the surrounding graph applies row and channel scales in f32 and
adds f32 bias. The matrix shim passes no integer bias for these source roots.

Merlin also has a **different**, older TorchAO 0.16 opaque-call fallback:
`torchao_affine.py` transcribes `max|x|/127`, epsilon `1e-5`, and
`[-127, 127]` from that version's reduced-range API. The newer explicit
captures do not call it. The source W8A8 preflight now rejects opaque affine
calls whose numerical parameters were lost at import. A different TorchAO
version can still be used when its actual graph records its arithmetic;
the scheme name does not select a fixed range.

Both OPU BME and compact VME Zvt kernels accept signed i8 A and B, perform
signed products, and return i32 C. They must preserve the source bytes and
leave scale, zero-point, rounding, and f32 bias in the surrounding graph.
The source W8A8 preflight now rejects any tagged integer contraction without
a proven zero initializer; a matrix shim writes its output and would otherwise
drop a live addend. Both exact captures pass this check. OPU scalar tile and
fallback paths use unsigned 32-bit accumulation for defined modulo arithmetic,
matching the VME scalar twin at a future overflowing endpoint.
The worst possible signed product magnitude is `128 × 128 = 16,384`. The
largest captured K is 1,024 for SpectFormer and 5,632 for TinyLlama, so a
worst-case dot has magnitude at most 16,777,216 and 92,274,688 respectively,
both below the positive i32 limit 2,147,483,647. This proves the current
matrix roots do not need saturation or wider accumulation, even for endpoint
operands. It does not certify the subsequent f32 model output.

The two units have different instruction encodings and physical arrays but
both use a 64-row logical tile for their VLEN512 int8 configurations. All
TinyLlama roots have M=8 and N and K divisible by 64, so their **static
logical tile fill is 8/64 = 12.5%** on either unit. It is a poor match for
single-sequence inference; token batching or a smaller logical tile would
need a separately proven source/accuracy and hardware configuration. A
large speedup from this workload cannot be called high array duty. The
[eight-request batching experiment](../../experiments/reference-data/tinyllama_torchao_batch_utilization_plan_2026-11-06.md)
specifies the source shape and complete-model gates needed to test whether
flattening independent requests fills the existing 64-row tile. Its full-model
eager TorchAO probe found 155 M=64 integer calls and bitwise equality between
one B=8 call and eight B=1 calls on one seeded input. The B=8 MLIR capture,
routed image, and hardware utilization remain unverified.

The captured activation maximum, reciprocal, round/clamp, dequantization, and
float attention/normalization remain outside the integer matrix unit; their
time must remain in the whole-inference denominator. SpectFormer has a
projected OPU fill of 75.9808% for 41 roots and compact VME fill of 76.5625%
for 40 selected roots. These are static projections, not measured enabled
cell cycles. See [utilization](UTILIZATION.md) for the measured-rate formulas.

Numerical status is separate: the complete SpectFormer source-to-TorchAO
framework gate fails, and TinyLlama whole-model parity is not certified.
Signed endpoint microkernels pass OPU Spike and VME Spike, and a smaller VME
K-tail case passes direct RTL/metasim. A whole-model OPU/VME certificate
requires the framework gate, exact twin/device Spike bytes, and a matching
FireSim board run in that order.
