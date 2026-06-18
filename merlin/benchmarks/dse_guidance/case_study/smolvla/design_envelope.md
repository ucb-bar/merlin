# Design envelope — smolvla / repeated_head

> Requirements derived from the workload contract — NOT calibrated to any hardware. No speedup is claimed.

- K = 10  ·  replan deadline = 1666.7 ms (assumed_reference)  ·  captured dtype = f32

## Requirements

| requirement | value | unit | evidence |
|-------------|-------|------|----------|
| macs_per_replan | 7.069e+09 | MAC | recovered_from_ir |
| resident_capacity_required | 30.0 MB | B | recovered_from_ir |
| weight_reload_bytes_per_replan | 300.3 MB | B | recovered_from_ir |
| avoidable_weight_reload_bytes | 270.3 MB | B | recovered_from_ir |
| dispatches_per_replan | 1.900e+02 | dispatch | derived_requirement |
| required_compute_rate | 4.241e+09 | MAC/s | derived_requirement |
| required_weight_bandwidth | 1.890e+08 | B/s | derived_requirement |
| required_activation_bandwidth | 2.486e+07 | B/s | derived_requirement |
| required_command_rate | 1.140e+02 | dispatch/s | derived_requirement |

## Resident capacity by storage format

| format | resident set |
|--------|--------------|
| bf16 | 15.0 MB |
| fp8 | 7.5 MB |
| int8 | 7.5 MB |
| int4 | 3.8 MB |
| fp6 | 5.6 MB |

## Candidate design axes (structural; quantification gated)

- **resident_bf16_head_weights** — needs: target memory hierarchy + bandwidth model, packed-layout support, quantization accuracy
- **resident_int8_head_weights** — needs: target memory hierarchy + bandwidth model, packed-layout support, quantization accuracy
- **resident_int4_head_weights** — needs: target memory hierarchy + bandwidth model, packed-layout support, quantization accuracy
- **autonomous_K_loop** — needs: actual host/device submit latency, runtime model
- **command_buffer_per_replan** — needs: measured per-dispatch host submit cost
