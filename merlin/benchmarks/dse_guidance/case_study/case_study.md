# Cross-workload provenance case study

> Flat model captures are insufficient DSE units. With `prov.fqn` provenance, Merlin recovers region roles from real IR, attaches real compute/memory facts to the repeated action head, and emits structural DSE candidates — while refusing quantitative benefit until calibration exists.

Real recaptured workloads: **4** (rdt, openvla, small_llama, tiny_llama). Captures are real architectures via the `prov.fqn`-enabled model2MLIR at small/random configs — structure & provenance real, magnitudes are small instances.

## Per-workload recovery

| workload | class | matmuls | roles recovered (from prov.fqn) | repeated_head facts | quant |
|----------|-------|---------|---------------------------------|---------------------|-------|
| rdt | diffusion/denoise_steps | 20 | repeated_head:20 | 20 mm, 391 MB, 39.4 GMAC/step xK=5 | blocked: missing_calibration |
| openvla | autoregressive_vla/action_token_decode | 26 | backbone_once:8, repeated_head:15, unknown:3 | 15 mm, 3 MB, 0.0 GMAC/step xK=7 | blocked: missing_calibration |
| small_llama | llm/token_decode | 15 | repeated_head:14, unknown:1 | 14 mm, 2 MB, 0.0 GMAC/step xK=32 | blocked: missing_calibration |
| tiny_llama | llm/token_decode | 15 | repeated_head:15 | 15 mm, 614 MB, 0.6 GMAC/step xK=32 | blocked: missing_calibration |

## What flattening hides vs what provenance recovers

- **Flat view:** weights used once, no K-loop, no backbone/head split, no deadline — so residency / autonomous-loop / partition axes are invisible or illegal.
- **Recovered view:** roles from `prov.fqn`, real per-region MACs/bytes, the repeated head and (for OpenVLA) the vision-backbone/LM split made explicit.
- **Honest gate:** every candidate carries real facts but stays `blocked_by: missing_calibration`; no speedup is claimed.

## Evidence provenance legend

`recovered_from_ir` · `recovered_from_prov_fqn` · `assumed_reference` · `calibrated` · `uncalibrated` · `unavailable`

See `cross_workload_provenance.csv` for the per-item flat-vs-recovered table.
