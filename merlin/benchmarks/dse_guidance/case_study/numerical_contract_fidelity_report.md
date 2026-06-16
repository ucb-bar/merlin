# Numerical-contract fidelity

> Does the capture preserve the numerical contract — what is stored, computed, accumulated, dequantized, and requantized, in which precision? These are structural observations from real captures; lowering precision is a candidate to **measure**, never a claimed speedup or accuracy.

| capture | declared quant | storage | compute | dequant ops | packed layout | lost | severity |
|---------|----------------|---------|---------|-------------|---------------|------|----------|
| groot_n1d7_fp8_consistent | float8_weight_only_e4m3 | fp8 | f32 | 0 | LOST | native_low_bit_compute, packed_low_bit_layout | high |
| groot_n1d7_int8_consistent | int8_weight_only | int8 | f32 | 116 | LOST | native_low_bit_compute, packed_low_bit_layout | high |
| molmoact_fp8_consistent | float8_weight_only_e4m3 | fp8 | f32 | 0 | LOST | native_low_bit_compute, packed_low_bit_layout | high |
| molmoact_int8_consistent | int8_weight_only | int8 | f32 | 17 | LOST | native_low_bit_compute, packed_low_bit_layout | high |
| openvla_fp8_consistent | float8_weight_only_e4m3 | fp8 | f32 | 0 | LOST | native_low_bit_compute, packed_low_bit_layout | high |
| openvla_int8_consistent | int8_weight_only | int8 | f32 | 26 | LOST | native_low_bit_compute, packed_low_bit_layout | high |
| pi05_fp8_consistent | float8_weight_only_e4m3 | fp8 | f32 | 2346 | LOST | native_low_bit_compute, packed_low_bit_layout | high |
| pi05_int8_consistent | int8_weight_only | int8 | f32 | 782 | LOST | native_low_bit_compute, packed_low_bit_layout | high |
| rdt2_fp8_consistent | float8_weight_only_e4m3 | fp8 | f32 | 0 | LOST | native_low_bit_compute, packed_low_bit_layout | high |
| rdt2_int8_consistent | int8_weight_only | int8 | f32 | 23 | LOST | native_low_bit_compute, packed_low_bit_layout | high |
| rdt_fp8_consistent | float8_weight_only_e4m3 | fp8 | f32 | 0 | LOST | native_low_bit_compute, packed_low_bit_layout | high |
| rdt_int8_consistent | int8_weight_only | int8 | f32 | 20 | LOST | native_low_bit_compute, packed_low_bit_layout | high |
| small_llama_fp8_consistent | float8_weight_only_e4m3 | fp8 | f32 | 0 | LOST | native_low_bit_compute, packed_low_bit_layout | high |
| small_llama_int8_consistent | int8_weight_only | int8 | f32 | 15 | LOST | native_low_bit_compute, packed_low_bit_layout | high |
| smolvla_fp8_consistent | float8_weight_only_e4m3 | fp8 | bf16 | 0 | LOST | native_low_bit_compute, packed_low_bit_layout | high |
| smolvla_int8_consistent | int8_weight_only | int8 | bf16 | 302 | LOST | native_low_bit_compute, packed_low_bit_layout | high |
| tiny_llama_fp8_consistent | float8_weight_only_e4m3 | fp8 | f32 | 0 | LOST | native_low_bit_compute, packed_low_bit_layout | high |
| tiny_llama_int8_consistent | int8_weight_only | int8 | f32 | 15 | LOST | native_low_bit_compute, packed_low_bit_layout | high |
| xr0_fp8_consistent | float8_weight_only_e4m3 | fp8 | f32 | 0 | LOST | native_low_bit_compute, packed_low_bit_layout | high |
| xr0_int8_consistent | int8_weight_only | int8 | f32 | 19 | LOST | native_low_bit_compute, packed_low_bit_layout | high |

**Finding:** 20/20 captures store weights low-bit but run **wide (f32) matmuls** — native low-bit compute and the packed layout are absent from the capture. The hidden DSE axes (`native_lowbit_compute`, `resident_packed_lowbit_weights`, `fused_dequant_matmul`) are structural candidates; ranking is blocked on accuracy sweeps + low-bit kernel calibration.

## Evidence labels

`recovered_from_ir` (dtypes, op counts, quantization) · `assumed_reference` (expected contract, if supplied) · candidates `uncalibrated` until measured.
