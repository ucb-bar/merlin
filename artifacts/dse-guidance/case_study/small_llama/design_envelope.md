# Design envelope — small_llama / repeated_head

> Requirements derived from the workload contract — NOT calibrated to any hardware. No speedup is claimed.

- K = 32  ·  replan deadline = 1066.7 ms (assumed_reference)  ·  captured dtype = f32

## Requirements

| requirement | value | unit | evidence |
|-------------|-------|------|----------|
| macs_per_replan | 1.096e+08 | MAC | recovered_from_ir |
| resident_capacity_required | 1.6 MB | B | recovered_from_ir |
| weight_reload_bytes_per_replan | 52.2 MB | B | recovered_from_ir |
| avoidable_weight_reload_bytes | 50.6 MB | B | recovered_from_ir |
| dispatches_per_replan | 4.800e+02 | dispatch | derived_requirement |
| required_compute_rate | 1.027e+08 | MAC/s | derived_requirement |
| required_weight_bandwidth | 5.136e+07 | B/s | derived_requirement |
| required_activation_bandwidth | 5.053e+06 | B/s | derived_requirement |
| required_command_rate | 4.500e+02 | dispatch/s | derived_requirement |

## Resident capacity by storage format

| format | resident set |
|--------|--------------|
| bf16 | 836.0 KB |
| fp8 | 418.0 KB |
| int8 | 418.0 KB |
| int4 | 209.0 KB |
| fp6 | 313.5 KB |

## Candidate design axes (structural; quantification gated)

- **resident_bf16_head_weights** — needs: target memory hierarchy + bandwidth model, packed-layout support, quantization accuracy
- **resident_int8_head_weights** — needs: target memory hierarchy + bandwidth model, packed-layout support, quantization accuracy
- **resident_int4_head_weights** — needs: target memory hierarchy + bandwidth model, packed-layout support, quantization accuracy
- **autonomous_K_loop** — needs: actual host/device submit latency, runtime model
- **command_buffer_per_replan** — needs: measured per-dispatch host submit cost
