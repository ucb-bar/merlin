# Design envelope — openvla / repeated_head

> Requirements derived from the workload contract — NOT calibrated to any hardware. No speedup is claimed.

- K = 7  ·  replan deadline = 1400.0 ms (assumed_reference)  ·  captured dtype = f32

## Requirements

| requirement | value | unit | evidence |
|-------------|-------|------|----------|
| macs_per_replan | 1.101e+08 | MAC | recovered_from_ir |
| resident_capacity_required | 3.0 MB | B | recovered_from_ir |
| weight_reload_bytes_per_replan | 21.0 MB | B | recovered_from_ir |
| avoidable_weight_reload_bytes | 18.0 MB | B | recovered_from_ir |
| dispatches_per_replan | 1.050e+02 | dispatch | derived_requirement |
| required_compute_rate | 7.864e+07 | MAC/s | derived_requirement |
| required_weight_bandwidth | 1.573e+07 | B/s | derived_requirement |
| required_activation_bandwidth | 3.226e+06 | B/s | derived_requirement |
| required_command_rate | 7.500e+01 | dispatch/s | derived_requirement |

## Resident capacity by storage format

| format | resident set |
|--------|--------------|
| bf16 | 1.5 MB |
| fp8 | 768.0 KB |
| int8 | 768.0 KB |
| int4 | 384.0 KB |
| fp6 | 576.0 KB |

## Candidate design axes (structural; quantification gated)

- **resident_bf16_head_weights** — needs: target memory hierarchy + bandwidth model, packed-layout support, quantization accuracy
- **resident_int8_head_weights** — needs: target memory hierarchy + bandwidth model, packed-layout support, quantization accuracy
- **resident_int4_head_weights** — needs: target memory hierarchy + bandwidth model, packed-layout support, quantization accuracy
- **autonomous_K_loop** — needs: actual host/device submit latency, runtime model
- **command_buffer_per_replan** — needs: measured per-dispatch host submit cost
