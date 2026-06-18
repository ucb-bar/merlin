# Design envelope — pi05 / repeated_head

> Requirements derived from the workload contract — NOT calibrated to any hardware. No speedup is claimed.

- K = 10  ·  replan deadline = 1000.0 ms (assumed_reference)  ·  captured dtype = f32

## Requirements

| requirement | value | unit | evidence |
|-------------|-------|------|----------|
| macs_per_replan | 1.829e+13 | MAC | recovered_from_ir |
| resident_capacity_required | 8.6 GB | B | recovered_from_ir |
| weight_reload_bytes_per_replan | 85.8 GB | B | recovered_from_ir |
| avoidable_weight_reload_bytes | 77.2 GB | B | recovered_from_ir |
| dispatches_per_replan | 2.880e+03 | dispatch | derived_requirement |
| required_compute_rate | 1.829e+13 | MAC/s | derived_requirement |
| required_weight_bandwidth | 9.211e+10 | B/s | derived_requirement |
| required_activation_bandwidth | 4.588e+10 | B/s | derived_requirement |
| required_command_rate | 2.880e+03 | dispatch/s | derived_requirement |

## Resident capacity by storage format

| format | resident set |
|--------|--------------|
| bf16 | 4.3 GB |
| fp8 | 2.1 GB |
| int8 | 2.1 GB |
| int4 | 1.1 GB |
| fp6 | 1.6 GB |

## Candidate design axes (structural; quantification gated)

- **resident_bf16_head_weights** — needs: target memory hierarchy + bandwidth model, packed-layout support, quantization accuracy
- **resident_int8_head_weights** — needs: target memory hierarchy + bandwidth model, packed-layout support, quantization accuracy
- **resident_int4_head_weights** — needs: target memory hierarchy + bandwidth model, packed-layout support, quantization accuracy
- **autonomous_K_loop** — needs: actual host/device submit latency, runtime model
- **command_buffer_per_replan** — needs: measured per-dispatch host submit cost
