# Design envelope — rdt2 / repeated_head

> Requirements derived from the workload contract — NOT calibrated to any hardware. No speedup is claimed.

- K = 5  ·  replan deadline = 2133.3 ms (assumed_reference)  ·  captured dtype = f32

## Requirements

| requirement | value | unit | evidence |
|-------------|-------|------|----------|
| macs_per_replan | 4.959e+09 | MAC | recovered_from_ir |
| resident_capacity_required | 295.4 MB | B | recovered_from_ir |
| weight_reload_bytes_per_replan | 1.4 GB | B | recovered_from_ir |
| avoidable_weight_reload_bytes | 1.2 GB | B | recovered_from_ir |
| dispatches_per_replan | 1.300e+02 | dispatch | derived_requirement |
| required_compute_rate | 2.325e+09 | MAC/s | derived_requirement |
| required_weight_bandwidth | 7.260e+08 | B/s | derived_requirement |
| required_activation_bandwidth | 1.529e+07 | B/s | derived_requirement |
| required_command_rate | 6.094e+01 | dispatch/s | derived_requirement |

## Resident capacity by storage format

| format | resident set |
|--------|--------------|
| bf16 | 147.7 MB |
| fp8 | 73.8 MB |
| int8 | 73.8 MB |
| int4 | 36.9 MB |
| fp6 | 55.4 MB |

## Candidate design axes (structural; quantification gated)

- **resident_bf16_head_weights** — needs: target memory hierarchy + bandwidth model, packed-layout support, quantization accuracy
- **resident_int8_head_weights** — needs: target memory hierarchy + bandwidth model, packed-layout support, quantization accuracy
- **resident_int4_head_weights** — needs: target memory hierarchy + bandwidth model, packed-layout support, quantization accuracy
- **autonomous_K_loop** — needs: actual host/device submit latency, runtime model
- **command_buffer_per_replan** — needs: measured per-dispatch host submit cost
