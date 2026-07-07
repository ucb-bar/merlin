# Corpus-expansion plan (all)

> A recommendation only — no new data is ingested here. Which registry model families lack a committed recapture, and the capture-fidelity improvements that would most raise cross-workload confidence before any quantitative DSE.

Captured models: bitvla, groot_n1d7, molmoact, openvla, pi05, rdt, rdt2, smolvla, tiny_llama, xr0 (families: autoregressive_vla, diffusion, flow_matching, llm).

**3 registry models lack a recapture.**

## Missing captures by family

| family | model | loop kind | reference K | note |
|---|---|---|---|---|
| autoregressive_vla | openvla_oft | action_token_decode | 7 |  |
| llm | small_llama | token_decode | 7 | LLaMA-style decoder; K=7 captured decode length (IR-recovered). |
| llm | small | token_decode | 32 |  |

## Capture-fidelity asks (raise confidence on existing + new captures)

- preserve the host K-loop (do not unroll the denoise/decode loop into a single pass)
- preserve packed low-bit weight layout + per-channel scales (do not dequantize to bf16)
- preserve the KV-cache / attention region (do not lower attention to dense matmuls)
- tag region roles (backbone-once vs repeated-head) in prov.fqn
- record the real loop count / control cadence (replace the assumed K reference)
