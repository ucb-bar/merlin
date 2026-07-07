# Workload-family summary (all)

> The recaptured workloads grouped by architecture family, with each family's recovered signal profile and which findings are family-specific vs cross-family. Structural only.

## autoregressive_vla

- workloads: bitvla, molmoact, openvla
- total MACs (single-pass capture): 8,473,739,264
- dominant shape class: wide_skinny
- inter-op parallelism (work/span) range: [1.13, 1.93]
- reference loop count K range: [7, 8]

## diffusion

- workloads: groot_n1d7, rdt, rdt2, xr0
- total MACs (single-pass capture): 61,967,136,768
- dominant shape class: wide_skinny
- inter-op parallelism (work/span) range: [1.11, 1.36]
- reference loop count K range: [4, 5]

## flow_matching

- workloads: pi05, smolvla
- total MACs (single-pass capture): 2,256,631,539,200
- dominant shape class: wide_skinny
- inter-op parallelism (work/span) range: [1.31, 1.61]
- reference loop count K range: [10, 10]

## llm

- workloads: tiny_llama
- total MACs (single-pass capture): 923,795,456
- dominant shape class: wide_skinny
- inter-op parallelism (work/span) range: [1.62, 1.62]
- reference loop count K range: [7, 7]

## Family-specific findings (one family only)

- — _(families: —)_

## Cross-family findings (≥2 families)

- head weight bytes _(families: autoregressive_vla, diffusion, flow_matching, llm)_
- total macs _(families: autoregressive_vla, diffusion, flow_matching, llm)_
- n matmuls _(families: autoregressive_vla, diffusion, flow_matching, llm)_
- matmul bias epilogues _(families: autoregressive_vla, diffusion, flow_matching, llm)_
- head cadence _(families: autoregressive_vla, diffusion, flow_matching, llm)_
- accuracy int8 w8a8 _(families: autoregressive_vla, diffusion, llm)_
- measured dispatch ratio _(families: —)_
- coverage under 10pct _(families: autoregressive_vla, diffusion, flow_matching, llm)_
- available parallelism _(families: autoregressive_vla, diffusion, flow_matching, llm)_
- avoidable weight reload _(families: autoregressive_vla, diffusion, flow_matching, llm)_
- resident int8 B _(families: autoregressive_vla, diffusion, flow_matching, llm)_
- boundary pressure score _(families: —)_
- max regret _(families: —)_
