# AGENT.md — src/merlin/capture

Owns shared capture bundle access, deterministic rewrites, and workload identities.
No dependency on experiment orchestration, baseline frameworks, or DSE analysis.
Preserve on-disk formats and numerical tolerances when moving code.

`safetensors.py` owns bounded header framing and JSON decoding for capture rewrites
and weight packers. It preserves metadata and entry order, never reads tensor data,
and rejects malformed, duplicate-key or non-finite JSON. Its default 16 MiB header
bound is a Merlin policy, not a format limit. Consumers still own tensor layout,
dtype, payload identity and execution-authority validation.
