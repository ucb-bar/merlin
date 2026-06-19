# Cross-workload provenance case study

> Flat model captures are insufficient DSE units. With `prov.fqn` provenance, Merlin recovers region roles from real IR, attaches real compute/memory facts to the repeated action head, and emits structural DSE candidates — while refusing quantitative benefit until calibration exists.

Real recaptured workloads: **11** (rdt, openvla, small_llama, tiny_llama, rdt2, groot_n1d7, molmoact, smolvla, pi05, xr0, bitvla). Captures are real architectures via the `prov.fqn`-enabled model2MLIR at small/random configs — structure & provenance real, magnitudes are small instances.

## Headline: flat capture vs recovered contract

| workload | flat view | recovered view | real facts | DSE implication | quantification |
|----------|-----------|----------------|------------|-----------------|----------------|
| rdt | weights once, no K-loop | repeated_head | 20 mm, 391 MB, 39.4 GMAC/step xK=5 | resident_action_head_weights | blocked: missing_calibration |
| openvla | weights once, no K-loop | repeated_head + backbone_once split | 15 mm, 3 MB, 0.0 GMAC/step xK=7 | resident_action_head_weights, backbone_head_partition | blocked: missing_calibration |
| small_llama | weights once, no K-loop | repeated_head | 15 mm, 2 MB, 0.0 GMAC/step xK=32 | resident_action_head_weights | blocked: missing_calibration |
| tiny_llama | weights once, no K-loop | repeated_head | 15 mm, 614 MB, 0.6 GMAC/step xK=32 | resident_action_head_weights | blocked: missing_calibration |
| rdt2 | weights once, no K-loop | repeated_head | 23 mm, 301 MB, 0.9 GMAC/step xK=5 | resident_action_head_weights | blocked: missing_calibration |
| groot_n1d7 | weights once, no K-loop | repeated_head + backbone_once split | 16 mm, 296 MB, 2.6 GMAC/step xK=4 | resident_action_head_weights, backbone_head_partition | blocked: missing_calibration |
| molmoact | weights once, no K-loop | repeated_head | 17 mm, 3787 MB, 7.6 GMAC/step xK=8 | resident_action_head_weights | blocked: missing_calibration |
| smolvla | weights once, no K-loop | repeated_head + backbone_once split | 19 mm, 31 MB, 0.7 GMAC/step xK=10 | resident_action_head_weights, backbone_head_partition | blocked: missing_calibration |
| pi05 | weights once, no K-loop | repeated_head + backbone_once split | 288 mm, 9211 MB, 1828.5 GMAC/step xK=10 | resident_action_head_weights, backbone_head_partition | blocked: missing_calibration |
| xr0 | weights once, no K-loop | repeated_head + backbone_once split | 16 mm, 123 MB, 0.9 GMAC/step xK=5 | resident_action_head_weights, backbone_head_partition | blocked: missing_calibration |
| bitvla | weights once, no K-loop | repeated_head | 15 mm, 6 MB, 0.0 GMAC/step xK=7 | resident_action_head_weights | blocked: missing_calibration |

## Per-workload recovery

| workload | class | matmuls | roles recovered (from prov.fqn) | repeated_head facts | quant |
|----------|-------|---------|---------------------------------|---------------------|-------|
| rdt | diffusion/denoise_steps | 20 | repeated_head:20 | 20 mm, 391 MB, 39.4 GMAC/step xK=5 | blocked: missing_calibration |
| openvla | autoregressive_vla/action_token_decode | 26 | backbone_once:11, repeated_head:15 | 15 mm, 3 MB, 0.0 GMAC/step xK=7 | blocked: missing_calibration |
| small_llama | llm/token_decode | 15 | repeated_head:15 | 15 mm, 2 MB, 0.0 GMAC/step xK=32 | blocked: missing_calibration |
| tiny_llama | llm/token_decode | 15 | repeated_head:15 | 15 mm, 614 MB, 0.6 GMAC/step xK=32 | blocked: missing_calibration |
| rdt2 | diffusion/denoise_steps | 23 | repeated_head:23 | 23 mm, 301 MB, 0.9 GMAC/step xK=5 | blocked: missing_calibration |
| groot_n1d7 | diffusion/denoise_steps | 18 | backbone_once:2, repeated_head:16 | 16 mm, 296 MB, 2.6 GMAC/step xK=4 | blocked: missing_calibration |
| molmoact | autoregressive_vla/action_token_decode | 17 | repeated_head:17 | 17 mm, 3787 MB, 7.6 GMAC/step xK=8 | blocked: missing_calibration |
| smolvla | flow_matching/denoise_steps | 106 | backbone_once:87, repeated_head:19 | 19 mm, 31 MB, 0.7 GMAC/step xK=10 | blocked: missing_calibration |
| pi05 | flow_matching/denoise_steps | 777 | backbone_once:489, repeated_head:288 | 288 mm, 9211 MB, 1828.5 GMAC/step xK=10 | blocked: missing_calibration |
| xr0 | diffusion/denoise_steps | 19 | repeated_head:16, backbone_once:1, prefix_builder:2 | 16 mm, 123 MB, 0.9 GMAC/step xK=5 | blocked: missing_calibration |
| bitvla | autoregressive_vla/action_token_decode | 15 | repeated_head:15 | 15 mm, 6 MB, 0.0 GMAC/step xK=7 | blocked: missing_calibration |

## What flattening hides vs what provenance recovers

- **Flat view:** weights used once, no K-loop, no backbone/head split, no deadline — so residency / autonomous-loop / partition axes are invisible or illegal.
- **Recovered view:** roles from `prov.fqn`, real per-region MACs/bytes, the repeated head and (for OpenVLA) the vision-backbone/LM split made explicit.
- **Honest gate:** every candidate carries real facts but stays `blocked_by: missing_calibration`; no speedup is claimed.

## Evidence provenance legend

`recovered_from_ir` · `recovered_from_prov_fqn` · `assumed_reference` · `calibrated` · `uncalibrated` · `unavailable`

See `cross_workload_provenance.csv` for the per-item flat-vs-recovered table, and `numerical_contract_fidelity_report.md` for the precision/quantization contract (the orthogonal axis: every int8/fp8 zoo capture stores weights low-bit but runs f32 matmuls — native low-bit compute and the packed layout are absent).
