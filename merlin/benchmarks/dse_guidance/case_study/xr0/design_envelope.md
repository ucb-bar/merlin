# Design envelope — xr0 / repeated_head

> Requirements derived from the workload contract — NOT calibrated to any hardware. No speedup is claimed.

- K = 10  ·  replan deadline = 333.3 ms (assumed_reference)  ·  captured dtype = f32

## Requirements

| requirement | value | unit | evidence |
|-------------|-------|------|----------|
| macs_per_replan | 1.311e+07 | MAC | recovered_from_ir |
| resident_capacity_required | 5.0 MB | B | recovered_from_ir |
| weight_reload_bytes_per_replan | 50.0 MB | B | recovered_from_ir |
| avoidable_weight_reload_bytes | 45.0 MB | B | recovered_from_ir |
| dispatches_per_replan | 2.000e+01 | dispatch | derived_requirement |
| required_compute_rate | 3.932e+07 | MAC/s | derived_requirement |
| required_weight_bandwidth | 1.573e+08 | B/s | derived_requirement |
| required_activation_bandwidth | 3.994e+05 | B/s | derived_requirement |
| required_command_rate | 6.000e+01 | dispatch/s | derived_requirement |

## Resident capacity by storage format

| format | resident set |
|--------|--------------|
| bf16 | 2.5 MB |
| fp8 | 1.2 MB |
| int8 | 1.2 MB |
| int4 | 640.0 KB |
| fp6 | 960.0 KB |

## Candidate design axes (structural; quantification gated)

- **resident_bf16_head_weights** — needs: target memory hierarchy + bandwidth model, packed-layout support, quantization accuracy
- **resident_int8_head_weights** — needs: target memory hierarchy + bandwidth model, packed-layout support, quantization accuracy
- **resident_int4_head_weights** — needs: target memory hierarchy + bandwidth model, packed-layout support, quantization accuracy
- **autonomous_K_loop** — needs: actual host/device submit latency, runtime model
- **command_buffer_per_replan** — needs: measured per-dispatch host submit cost
