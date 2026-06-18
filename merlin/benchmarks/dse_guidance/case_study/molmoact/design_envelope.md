# Design envelope — molmoact / repeated_head

> Requirements derived from the workload contract — NOT calibrated to any hardware. No speedup is claimed.

- K = 8  ·  replan deadline = 1600.0 ms (assumed_reference)  ·  captured dtype = f32

## Requirements

| requirement | value | unit | evidence |
|-------------|-------|------|----------|
| macs_per_replan | 6.060e+10 | MAC | recovered_from_ir |
| resident_capacity_required | 3.5 GB | B | recovered_from_ir |
| weight_reload_bytes_per_replan | 28.2 GB | B | recovered_from_ir |
| avoidable_weight_reload_bytes | 24.7 GB | B | recovered_from_ir |
| dispatches_per_replan | 1.360e+02 | dispatch | derived_requirement |
| required_compute_rate | 3.787e+10 | MAC/s | derived_requirement |
| required_weight_bandwidth | 1.894e+10 | B/s | derived_requirement |
| required_activation_bandwidth | 5.202e+07 | B/s | derived_requirement |
| required_command_rate | 8.500e+01 | dispatch/s | derived_requirement |

## Resident capacity by storage format

| format | resident set |
|--------|--------------|
| bf16 | 1.8 GB |
| fp8 | 903.0 MB |
| int8 | 903.0 MB |
| int4 | 451.5 MB |
| fp6 | 677.2 MB |

## Candidate design axes (structural; quantification gated)

- **resident_bf16_head_weights** — needs: target memory hierarchy + bandwidth model, packed-layout support, quantization accuracy
- **resident_int8_head_weights** — needs: target memory hierarchy + bandwidth model, packed-layout support, quantization accuracy
- **resident_int4_head_weights** — needs: target memory hierarchy + bandwidth model, packed-layout support, quantization accuracy
- **autonomous_K_loop** — needs: actual host/device submit latency, runtime model
- **command_buffer_per_replan** — needs: measured per-dispatch host submit cost
