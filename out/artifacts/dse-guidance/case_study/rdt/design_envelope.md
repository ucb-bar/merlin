# Design envelope — rdt / repeated_head

> Requirements derived from the workload contract — NOT calibrated to any hardware. No speedup is claimed.

- K = 5  ·  replan deadline = 2133.3 ms (assumed_reference)  ·  captured dtype = f32

## Requirements

| requirement | value | unit | evidence |
|-------------|-------|------|----------|
| macs_per_replan | 1.973e+11 | MAC | recovered_from_ir |
| resident_capacity_required | 375.0 MB | B | recovered_from_ir |
| weight_reload_bytes_per_replan | 1.8 GB | B | recovered_from_ir |
| avoidable_weight_reload_bytes | 1.5 GB | B | recovered_from_ir |
| dispatches_per_replan | 1.050e+02 | dispatch | derived_requirement |
| required_compute_rate | 9.250e+10 | MAC/s | derived_requirement |
| required_weight_bandwidth | 9.216e+08 | B/s | derived_requirement |
| required_activation_bandwidth | 2.792e+08 | B/s | derived_requirement |
| required_command_rate | 4.922e+01 | dispatch/s | derived_requirement |

## Resident capacity by storage format

| format | resident set |
|--------|--------------|
| bf16 | 187.5 MB |
| fp8 | 93.8 MB |
| int8 | 93.8 MB |
| int4 | 46.9 MB |
| fp6 | 70.3 MB |

## Candidate design axes (structural; quantification gated)

- **resident_bf16_head_weights** — needs: target memory hierarchy + bandwidth model, packed-layout support, quantization accuracy
- **resident_int8_head_weights** — needs: target memory hierarchy + bandwidth model, packed-layout support, quantization accuracy
- **resident_int4_head_weights** — needs: target memory hierarchy + bandwidth model, packed-layout support, quantization accuracy
- **autonomous_K_loop** — needs: actual host/device submit latency, runtime model
- **command_buffer_per_replan** — needs: measured per-dispatch host submit cost
