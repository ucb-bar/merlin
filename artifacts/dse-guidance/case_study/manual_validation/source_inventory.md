# Phase 0 — Source + artifact inventory (P19 forensic audit)

All 11 workloads have a concrete source repo (no "source not found"). See `source_inventory.csv` for the
machine-readable table; this file records the cross-cutting facts every downstream audit must respect.

## Corpus-wide caveats (apply to every per-workload audit)

1. **Only `tiny_llama` is a real checkpoint.** The other 10 are random-init and/or deliberately tiny
   configs (small-config env knobs in `model2MLIR/workloads/<wl>/capture.toml [env]`: `M2M_*_LAYERS`,
   `M2M_*_VOCAB`, `M2M_RDT_DEPTH=2`, etc.). **Therefore every MAC/byte magnitude is structural-only** —
   shapes, ratios, op mix, and rankings are the signal; absolute numbers are NOT deployment-real and must
   never be presented as such.

2. **K / H / control-rate are sidecar/config, not IR-recovered.** They come from
   `merlin/python/merlin/dse_guidance/models.py::MODEL_ARCH` (every entry tagged `loop_count_source =
   "assumed"`). The captured MLIR contains a single forward pass — the K-loop is unrolled away by
   torch.export. Any residency / cadence / command-rate claim depends on this assumed K.

3. **The capture is a single denoise/decode step (or one forward), not the full loop.** Diffusion/flow
   models (rdt, rdt2, groot_n1d7, pi05, smolvla, xr0) capture ONE denoise step; autoregressive models
   (openvla, molmoact, bitvla, *_llama) capture one forward (no KV-cache growth across the decode loop).

4. **Attention is a single fused op in the EXPORTED graph.** The FX dump shows
   `aten.scaled_dot_product_attention.default` (e.g. rdt ×4) — attention is fully structured at export
   level and is only decomposed into matmul+softmax+reduce during torch-mlir *lowering*. So the fidelity
   ladder is: **source (nn.Module attention) → export (1 SDPA op, all dims) → flat MLIR (decomposed
   generics) → high-level MLIR (linalg_ext.softmax + bmm generics) → Merlin artifact (attention MACs
   recovered from the generics, P18)**. Each lowering step erases structure; the exported graph is the
   richest representation.

## Per-workload locations (summary; full table in source_inventory.csv)

| workload | family | repo | model class | inference entry | checkpoint |
|---|---|---|---|---|---|
| bitvla | autoregressive_vla | BitVLA | bitvla_for_action_prediction.py:21 | predict_action:312 | random (BitLinear fake-quant) |
| groot_n1d7 | diffusion | Isaac-GR00T | gr00t_n1d7.py:38 | get_action_with_features:312 | random |
| molmoact | autoregressive_vla | molmoact | modeling_molmoact.py:1493 | forward:1493 | random |
| openvla | autoregressive_vla | HF openvla-7b | OpenVLAForActionPrediction | forward | random (tiny) |
| pi05 | flow_matching | openpi | pi0_pytorch.py | denoise_step:422 | random |
| rdt | diffusion | RoboticsDiffusionTransformer | models/rdt/model.py:22 | forward:126 | not real (lang embed only) |
| rdt2 | diffusion | RDT2 | models/rdt/model.py:11 | forward:134 | random |
| small_llama | llm | model2MLIR (inline) | loader.py:71 | forward:77 | random (toy) |
| smolvla | flow_matching | lerobot v0.5.1 | modeling_smolvla.py | denoise_step:871 | random |
| tiny_llama | llm | HF TinyLlama-1.1B | AutoModelForCausalLM | forward | **REAL** |
| xr0 | flow_matching | Xiaomi-Robotics-0 | XR0.py | dit_forward:568 | random |

## Capture availability
- **Flat MLIR (all 11):** `merlin/benchmarks/dse_guidance/recaptures/<wl>/model.mlir`.
- **Level MLIR (rdt, openvla, pi05, bitvla only):** `recaptures_levels/<wl>/model_{highlevel,qdq}.mlir`.
- **Exported FX dump (re-run here):** `manual_validation/exported_fx/<wl>.txt` (gitignored; regenerable via
  `dump_exported_fx.py`).
