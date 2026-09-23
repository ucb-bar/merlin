# Design envelope — groot_n1d7 / repeated_head

> Requirements derived from the workload contract — NOT calibrated to any hardware. No speedup is claimed.

- K = 4  ·  replan deadline = 533.3 ms (assumed_reference)  ·  captured dtype = f32

## Requirements

| requirement | value | unit | evidence |
|-------------|-------|------|----------|
| macs_per_replan | 8.157e+10 | MAC | recovered_from_ir |
| resident_capacity_required | 2.0 GB | B | recovered_from_ir |
| weight_reload_bytes_per_replan | 8.2 GB | B | recovered_from_ir |
| avoidable_weight_reload_bytes | 6.1 GB | B | recovered_from_ir |
| dispatches_per_replan | 4.640e+02 | dispatch | derived_requirement |
| required_compute_rate | 1.530e+11 | MAC/s | derived_requirement |
| required_weight_bandwidth | 1.650e+10 | B/s | derived_requirement |
| required_activation_bandwidth | 5.994e+08 | B/s | derived_requirement |
| required_command_rate | 8.700e+02 | dispatch/s | derived_requirement |

## Resident capacity by storage format

| format | resident set |
|--------|--------------|
| bf16 | 1.0 GB |
| fp8 | 524.6 MB |
| int8 | 524.6 MB |
| int4 | 262.3 MB |
| fp6 | 393.5 MB |

## Candidate design axes (structural; quantification gated)

- **resident_bf16_head_weights** — needs: target memory hierarchy + bandwidth model, packed-layout support, quantization accuracy
- **resident_int8_head_weights** — needs: target memory hierarchy + bandwidth model, packed-layout support, quantization accuracy
- **resident_int4_head_weights** — needs: target memory hierarchy + bandwidth model, packed-layout support, quantization accuracy
- **autonomous_K_loop** — needs: actual host/device submit latency, runtime model
- **command_buffer_per_replan** — needs: measured per-dispatch host submit cost
