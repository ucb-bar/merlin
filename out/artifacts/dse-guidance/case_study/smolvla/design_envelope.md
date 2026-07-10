# Design envelope — smolvla / repeated_head

> Requirements derived from the workload contract — NOT calibrated to any hardware. No speedup is claimed.

- K = 10  ·  replan deadline = 1666.7 ms (assumed_reference)  ·  captured dtype = bf16

## Requirements

| requirement | value | unit | evidence |
|-------------|-------|------|----------|
| macs_per_replan | 5.094e+10 | MAC | recovered_from_ir |
| resident_capacity_required | 196.6 MB | B | recovered_from_ir |
| weight_reload_bytes_per_replan | 1.9 GB | B | recovered_from_ir |
| avoidable_weight_reload_bytes | 1.7 GB | B | recovered_from_ir |
| dispatches_per_replan | 1.160e+03 | dispatch | derived_requirement |
| required_compute_rate | 3.057e+10 | MAC/s | derived_requirement |
| required_weight_bandwidth | 1.237e+09 | B/s | derived_requirement |
| required_activation_bandwidth | 1.559e+08 | B/s | derived_requirement |
| required_command_rate | 6.960e+02 | dispatch/s | derived_requirement |

## Resident capacity by storage format

| format | resident set |
|--------|--------------|
| bf16 | 196.6 MB |
| fp8 | 98.3 MB |
| int8 | 98.3 MB |
| int4 | 49.1 MB |
| fp6 | 73.7 MB |

## Candidate design axes (structural; quantification gated)

- **resident_bf16_head_weights** — needs: target memory hierarchy + bandwidth model, packed-layout support, quantization accuracy
- **resident_int8_head_weights** — needs: target memory hierarchy + bandwidth model, packed-layout support, quantization accuracy
- **resident_int4_head_weights** — needs: target memory hierarchy + bandwidth model, packed-layout support, quantization accuracy
- **autonomous_K_loop** — needs: actual host/device submit latency, runtime model
- **command_buffer_per_replan** — needs: measured per-dispatch host submit cost
