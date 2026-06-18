# Design envelope — xr0 / repeated_head

> Requirements derived from the workload contract — NOT calibrated to any hardware. No speedup is claimed.

- K = 10  ·  replan deadline = 333.3 ms (assumed_reference)  ·  captured dtype = f32

## Requirements

| requirement | value | unit | evidence |
|-------------|-------|------|----------|
| macs_per_replan | 9.083e+09 | MAC | recovered_from_ir |
| resident_capacity_required | 117.4 MB | B | recovered_from_ir |
| weight_reload_bytes_per_replan | 1.1 GB | B | recovered_from_ir |
| avoidable_weight_reload_bytes | 1.0 GB | B | recovered_from_ir |
| dispatches_per_replan | 1.600e+02 | dispatch | derived_requirement |
| required_compute_rate | 2.725e+10 | MAC/s | derived_requirement |
| required_weight_bandwidth | 3.692e+09 | B/s | derived_requirement |
| required_activation_bandwidth | 1.497e+08 | B/s | derived_requirement |
| required_command_rate | 4.800e+02 | dispatch/s | derived_requirement |

## Resident capacity by storage format

| format | resident set |
|--------|--------------|
| bf16 | 58.7 MB |
| fp8 | 29.3 MB |
| int8 | 29.3 MB |
| int4 | 14.7 MB |
| fp6 | 22.0 MB |

## Candidate design axes (structural; quantification gated)

- **resident_bf16_head_weights** — needs: target memory hierarchy + bandwidth model, packed-layout support, quantization accuracy
- **resident_int8_head_weights** — needs: target memory hierarchy + bandwidth model, packed-layout support, quantization accuracy
- **resident_int4_head_weights** — needs: target memory hierarchy + bandwidth model, packed-layout support, quantization accuracy
- **autonomous_K_loop** — needs: actual host/device submit latency, runtime model
- **command_buffer_per_replan** — needs: measured per-dispatch host submit cost
