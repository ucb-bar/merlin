# openvla — source → FX → MLIR → Merlin forensic audit

**Workload:** `openvla` (autoregressive vision-language-action; OpenVLA-7b Prismatic VLM).
**Scope:** source-grounded structural audit. No performance claims. All magnitudes are
structural-only and reflect the **tiny random config** the m2m loader builds
(`M2M_OPENVLA_LLM_LAYERS=2`, `VIT_LAYERS=2`, `VOCAB=512`, `HIDDEN=128`, `IMG=64`,
`SEQ=4`), not the real 7B model (real: 32 LLM layers, hidden 4096, vocab 32064, ViT 24/27 blocks).

## Sources cited
- **HF dynamic source (upstream model):** `~/.cache/huggingface/modules/transformers_modules/openvla/openvla_hyphen_7b/47a0ec7.../modeling_prismatic.py` (26 KB, fetched .py only — `OpenVLAForActionPrediction(PrismaticForConditionalGeneration)`). This is real, cached, and readable; not dep-blocked.
- **m2m loader / capture cfg:** `/scratch/agustin/projects/model2MLIR/workloads/openvla/{loader.py,capture.toml}`.
- **Exported FX:** `merlin/benchmarks/dse_guidance/case_study/manual_validation/exported_fx/openvla.txt` (686 lines).
- **Flat MLIR:** `merlin/benchmarks/dse_guidance/recaptures/openvla/model.mlir` (1925 lines, `prov.level = "linalg-on-tensors"`).
- **Level MLIR:** `recaptures_levels/openvla/model_{highlevel,qdq}.mlir`.
- **Merlin artifacts:** `case_study/{operator_full_inventory.csv, work_coverage_table.csv, operator_shape_table.csv}` (grep `openvla`); `merlin/python/merlin/dse_guidance/models.py` `MODEL_ARCH["openvla"]`.

## openvla-specific questions answered

**(a) Inference shape / where the action is decoded.**
Autoregressive token generation. `predict_action` (modeling_prismatic.py:506) is a thin wrapper
around `.generate(input_ids, max_new_tokens=self.get_action_dim(unnorm_key))` (line 518). The
7-DoF action is emitted as **7 action *tokens***; they are converted to a discrete bin index by
`discretized_actions = self.vocab_size - predicted_action_token_ids` (line 522), mapped to
continuous values via `self.bin_centers[...]` (line 524), then **un-normalized** with per-dataset
`q01/q99` stats (lines 527–533). So: action tokens via `lm_head` → bin de-tokenize → un-normalize.
The capture stops at raw `lm_head` logits (`[1,20,512]`); the de-tokenize/un-normalize math is
**not** in the graph (it is host/numpy, data-dependent).

**(b) KV cache in source?** Yes. `PrismaticForConditionalGeneration.forward` takes
`past_key_values` + `use_cache` (lines 298–299), has an explicit "Cached Generation" branch
asserting `past_key_values is not None` (lines 320–337), `prepare_inputs_for_generation`
(line 450), and `_skip_keys_device_placement="past_key_values"` (line 182).

**(c) What was captured?** A **single forward over the full prompt, `use_cache=False`, no cache
growth** — the `_LogitsOnly` wrapper hard-codes `use_cache=False` (loader.py) and returns `.logits`.
The loader comment is explicit: `predict_action` wraps a `.generate()` loop that is "data-dependent,
not export-friendly", so it captures the single multimodal forward instead. The K=7 decode loop is
**absent** from FX/MLIR. Sequence length in the graph is 20 = 1 (BOS/text slice) + 16 patch tokens
projected + 3 remaining text tokens (FX line 328–332: `multimodal_embeddings = cat([emb[:, :1],
projected_patch_embeddings, emb[:, 1:]])`), from the 4 input text tokens (`input_ids: i64[1,4]`).

**(d) FX attention/conv form; flat-MLIR lowering.**
FX: **ViT attention = `aten.scaled_dot_product_attention` ×4** (timm eager path, 2 featurizers ×
its 1 surviving block — see (e)); **LM attention = explicit `aten.matmul` (qkᵀ) + `aten.softmax` +
`aten.matmul` (attn·v)** ×2 layers; **`aten.conv2d` ×2** (the two patch-embed projections);
`aten.embedding` ×1; `aten.silu` (LM MLP) + `aten.gelu` (ViT/projector MLP); RoPE as a
`wrap_with_set_grad_enabled` submod (sin/cos/matmul). Flat MLIR keeps **NO** fused attention/conv
op: SDPA, conv2d and softmax are all decomposed — flat histogram is `linalg.matmul` 26,
`linalg.generic` 190, `linalg.reduce` 21, `tensor.*`, `arith.*`; **zero** tosa/stablehlo, **zero**
quant ops. Attention contractions survive as `linalg.generic` tagged `attention_contraction /
batch_matmul`; conv survives as `linalg.generic` tagged `conv / conv2d`; softmax as
`linalg.generic` tagged `softmax`. (Merlin re-recovers these from `prov.*` attrs.)

**(e) Both vision backbone AND LM/action head captured?** Yes — region attribution is present and
correct: `operator_full_inventory.csv` tags **94 ops `backbone_once`** (fused DINO+SigLIP ViTs +
3-layer projector) and **122 ops `repeated_head`** (Llama LM layers + lm_head). **CAVEAT (real
export loss, upstream, not a Merlin bug):** only **ViT `blocks.0`** of each featurizer appears in
*both* the flat MLIR (`prov.fqn` has 244 `blocks.0` refs, **0** `blocks.1`) *and* the inventory,
whereas **both LM `layers.0` and `layers.1`** are present. Cause: loader.py re-wraps each
featurizer's forward to `get_intermediate_layers(x, n={n_blocks-2})`; for the 2-block tiny ViT
that returns block index 0, so block 1 is dead-code-eliminated at export. The whole-model trace
therefore under-counts ViT depth by construction (config says VIT_LAYERS=2; graph has 1 effective
block/featurizer). The LM is fully unrolled (2/2 layers).

**(f) Are H and decode cadence anywhere but MODEL_ARCH?** No. `MODEL_ARCH["openvla"]` (models.py:64)
carries `loop_count K=7` (`action_token_decode`), `action_horizon H=7`, `control_rate_hz=5.0`,
all `loop_count_source="assumed"` reference values. None of K/H/cadence appears in FX or any MLIR
level — they are **sidecar/config-only** architecture facts. The flat single-pass capture hides
the host autoregressive decode loop the weights are actually reused over.

**(g) decode_loop_controller / KV-blocked classification correct?** Yes. models.py:182–183 assigns
`loop_carried_state = ["kv_cache"]` for non-`denoise_steps` families (openvla → kv_cache), matching
the source's `past_key_values`. `runtime_object_candidates.yaml` lists openvla under
`decode_loop_controller`, `bounded_loop_command`, and `loop_carried_state_handle (lifetime=across_K)`.
`numerical_contract.py:240` gates `quantized_KV_cache` to autoregressive/decode workloads.
Classification is consistent with source.

## Feature audit table

| Feature | Source (modeling_prismatic / loader) | Exported FX | Flat / level MLIR | Merlin artifacts | Status |
|---|---|---|---|---|---|
| model class | `OpenVLAForActionPrediction(PrismaticForConditionalGeneration)` (HF dynamic) | wrapper `_LogitsOnly.forward` only | `prov.module="vla"` | `MODEL_ARCH["openvla"]` family `autoregressive_vla` | present_and_preserved |
| submodule boundary (vision/projector/LM) | `vision_backbone`/`projector`/`language_model` (lines 236–248) | fqn prefixes in arg names | `prov.fqn` prefixes | inventory `prov_fqn` + role split | present_and_preserved |
| forward entry | `forward(input_ids, pixel_values, use_cache, ...)` (line 291) | single `forward` graph | `func.func @forward(...)→1x20x512xf32` | one capture (`capture_id=openvla`) | present_and_preserved |
| action / lm head | `lm_head` logits → de-tokenize + un-normalize (predict_action 518–533) | `linear_33: f32[1,20,512]` = lm_head; return logits | `matmul_34` `vla.language_model.lm_head` | inventory row 25 `lm_head repeated_head` | present_and_preserved (logits only; de-tokenize host-side) |
| de-tokenize / un-normalize action | `vocab-id`, `bin_centers`, `q01/q99` (522–533) | absent (host/numpy, after generate) | absent | absent | present_in_source_erased_by_export |
| K decode loop (K=7) | `.generate(max_new_tokens=action_dim)` (518) | absent (`use_cache=False`, single fwd) | absent | sidecar `loop_count=7 assumed` | present_in_source_erased_by_export |
| H (action horizon =7) | DoF = `len(q01)` (get_action_dim 554) | absent | absent | `MODEL_ARCH.action_horizon=7` | sidecar_or_config_only |
| control cadence (5 Hz) | not in source code (deployment fact) | absent | absent | `MODEL_ARCH.control_rate_hz=5.0 assumed` | sidecar_or_config_only |
| q/k/v projections | LM `q/k/v_proj`; ViT fused `qkv` | `linear` q/k/v (LM), `qkv` (ViT) | `matmul` q/k/v_proj + ViT `attn.qkv` | inventory `attention_qkv_projection` | present_and_preserved |
| qkᵀ & attn·v bmm | implicit (SDPA in ViT; explicit matmul in LM) | LM: `matmul`+`matmul`; ViT: inside SDPA | `linalg.generic` `attention_contraction/batch_matmul` ×9 | inventory `attention_contraction`; work_coverage `n_attention_ops=9` | present_and_preserved (decomposed, re-recovered) |
| softmax | LM `aten.softmax`; ViT inside SDPA | `softmax.int` ×2 (LM); SDPA ×4 (ViT) | `linalg.generic` tagged `softmax` ×12 | inventory `softmax/softmax`; `n_softmax=12` | present_and_preserved (SDPA decomposed by lowering) |
| KV cache | `past_key_values`/`use_cache` cached branch (298–337) | absent (`use_cache=False`) | absent | classified `loop_carried_state=["kv_cache"]`; runtime_object kv candidates | present_in_source_erased_by_export |
| conv / vision patch-embed | timm `PatchEmbed` Conv2d ×2 | `aten.conv2d` ×2 (`192x3x16x16`, `384x3x16x16`) | `linalg.generic` `conv/conv2d` ×4 | inventory `conv/conv2d`; `n_conv=4` | present_and_preserved (decomposed to generic) |
| linear / GEMM | nn.Linear throughout | `aten.linear` ×34 + `aten.addmm`/`mm` | `linalg.matmul` ×26 | inventory 26 matmul rows; `n_linear_matmul=26` | present_and_preserved |
| dtype | fp32 build (`model.to(torch.float32)`) | all `f32` tensors | flat: f32 only (2898 f32, no i8) | inventory/shape `dtype=f32`, `evidence_dtype=recovered_from_ir` | present_and_preserved |
| accumulator | not explicit in fp32 source | n/a (fp32) | flat fp32; qdq dequant→`i32` zero-points, f32 accumulate | `accumulator_contract_table.csv` | present_and_preserved (fp32); i32 in qdq sidecar |
| quant / scales / packed layout | none in fp32 source | none | flat: 0 quant ops; **qdq level**: `quant_ext.dequantize_per_channel`, i8 weights, f32 scales, i32 zp, `axis=1`, `int8_weight_only`, `openvla_int8.safetensors` | qdq MLIR + dtype tables | not_present_in_source / present_in_qdq_sidecar (added by Merlin quant pass) |
| scales (per-channel) | n/a | n/a | qdq: per-channel f32 scale operands | dtype_capacity / accuracy_gate | sidecar_or_config_only (qdq level only) |
| region attribution backbone_once vs repeated_head | structural (vision once, LM per-token) | implicit in fqn | `prov.region_id`/fqn | inventory roles: 94 backbone_once / 122 repeated_head | present_and_preserved |
| tied weights | Llama does NOT tie; loader sets `tie_word_embeddings=False` | separate `embed_tokens` & `lm_head` params | separate weight operands | lm_head distinct row | present_and_preserved (untied) |
| repeated-weight lifetime (reuse across K) | weights reused every decode step (generate loop) | single use (flat trace, weight-once) | each weight read once | `reuse_lifetime_table.csv`; arch note "0 contract facts when flat" | present_in_source_erased_by_export (re-exposed only by multi-rate/arch view) |
| loop-carried / KV state | KV cache carried across decode steps | absent | absent | `loop_carried_state=["kv_cache"]`; contract_graph EDGE_LOOP_CARRIED | present_in_source_erased_by_export (recovered by arch/contract layer, not IR) |
| ViT depth (blocks.1) | config VIT_LAYERS=2 (2 blocks/featurizer) | SDPA×4 = block0 only (loader `n={n_blocks-2}` prunes block1) | flat: only `blocks.0` (0 `blocks.1` refs) | inventory: only `blocks.0` (78 rows) | present_in_source_erased_by_export |

## FINDINGS
1. **openvla is autoregressive action-token generation**, not direct regression: `predict_action`
   → `.generate(max_new_tokens=7)` → tokens de-binned (`vocab-id`→`bin_centers`) and un-normalized
   (q01/q99). Source has full KV-cache support (`past_key_values`/`use_cache`).
2. **The capture is one full-prompt forward with `use_cache=False`, no cache growth.** The K=7
   decode loop, KV cache, and action de-tokenize/un-normalize are all in source but **erased by
   export** (host-side, data-dependent). They survive only as Merlin **sidecar/arch facts**
   (`MODEL_ARCH` K=7/H=7/5 Hz, `loop_carried_state=["kv_cache"]`, runtime-object candidates).
3. **FX faithfully shows ViT SDPA ×4, LM explicit qkᵀ-matmul+softmax+attn·v ×2, conv2d ×2,
   embedding, RoPE.** Flat MLIR decomposes all of it: 26 `linalg.matmul` + `linalg.generic`, with
   attention/conv/softmax recoverable only via `prov.*` tags — which Merlin uses to re-attribute
   them (`n_attention_ops=9, n_softmax=12, n_conv=4, visible_linear_fraction=0.986`).
4. **Both regions are captured and correctly attributed** (94 `backbone_once` vision+projector vs
   122 `repeated_head` LM/head). Weights are read once (flat, single-pass) — the reuse-across-K
   lifetime is re-exposed only by the arch/multi-rate view, matching the documented "whole-model
   captures emit 0 contract facts" result.
5. **Real export under-count (upstream, not Merlin):** only ViT `blocks.0` survives per featurizer
   (loader's `get_intermediate_layers(n={n_blocks-2})` makes block 1 dead code for the 2-block tiny
   ViT). flat MLIR + inventory faithfully mirror this — both have `blocks.0` only, while both LM
   layers are present. Merlin's recovery is accurate; the loss is in the traced model.
6. **Quant is not in source and not in the flat capture** (fp32); it appears only in the
   `model_qdq.mlir` Merlin-generated sidecar (`int8_weight_only`, per-channel i8 weights, f32 scales,
   i32 zero-points, `openvla_int8.safetensors`).

```SUMMARY_ROWS
openvla,model_class,present_and_preserved,modeling_prismatic.py OpenVLAForActionPrediction; prov.module=vla; MODEL_ARCH autoregressive_vla
openvla,inference_mode,present_in_source_erased_by_export,predict_action->generate(max_new_tokens=7) autoregressive action tokens; capture is single use_cache=False forward
openvla,action_head_lm_head,present_and_preserved,FX linear_33 f32[1,20,512]; flat matmul_34 vla.language_model.lm_head; inventory row25 repeated_head
openvla,action_detokenize_unnormalize,present_in_source_erased_by_export,modeling_prismatic.py:522-533 vocab-id/bin_centers/q01-q99 host numpy; not in FX/MLIR
openvla,K_decode_loop,present_in_source_erased_by_export,generate loop absent from FX/MLIR; MODEL_ARCH loop_count=7 assumed sidecar
openvla,action_horizon_H,sidecar_or_config_only,MODEL_ARCH.action_horizon=7; not in any IR
openvla,control_cadence_5hz,sidecar_or_config_only,MODEL_ARCH.control_rate_hz=5.0 assumed; deployment fact not in source/IR
openvla,qkv_projections,present_and_preserved,LM q/k/v_proj + ViT attn.qkv in FX/flat; inventory attention_qkv_projection
openvla,attention_bmm,present_and_preserved,LM explicit matmul x2 + ViT SDPA; flat linalg.generic attention_contraction x9; work_coverage n_attention_ops=9
openvla,softmax,present_and_preserved,FX softmax.int x2 (LM) + SDPA x4 (ViT); flat generic softmax x12 tag
openvla,kv_cache,present_in_source_erased_by_export,past_key_values/use_cache cached branch lines 298-337; capture use_cache=False; reclassified loop_carried_state=[kv_cache]
openvla,conv_patch_embed,present_and_preserved,FX aten.conv2d x2 (192x3x16x16,384x3x16x16); flat generic conv/conv2d x4
openvla,linear_gemm,present_and_preserved,FX aten.linear x34; flat linalg.matmul x26; work_coverage n_linear_matmul=26
openvla,dtype,present_and_preserved,fp32 build; flat all f32 no i8; inventory dtype=f32 evidence recovered_from_ir
openvla,accumulator,present_and_preserved,fp32 accumulate flat; qdq dequant uses i32 zero-points; accumulator_contract_table
openvla,quant_scales_packed,not_present_in_source,fp32 source/flat 0 quant ops; only model_qdq.mlir int8_weight_only per-channel i8 weights f32 scales i32 zp openvla_int8.safetensors
openvla,region_attribution,present_and_preserved,inventory 94 backbone_once (vision+projector) vs 122 repeated_head (LM+head)
openvla,tied_weights,present_and_preserved,loader tie_word_embeddings=False; Llama untied; lm_head distinct from embed_tokens
openvla,repeated_weight_lifetime,present_in_source_erased_by_export,weights read once in flat single-pass; reuse-across-K re-exposed only by arch/multirate view (0 contract facts flat)
openvla,loop_carried_kv_state,present_in_source_erased_by_export,KV carried across decode in source; absent from IR; contract_graph EDGE_LOOP_CARRIED + loop_carried_state=[kv_cache]
openvla,vit_block_depth,present_in_source_erased_by_export,config VIT_LAYERS=2 but loader get_intermediate_layers(n=n_blocks-2) prunes block1; flat+inventory have blocks.0 only (LM has both layers)
openvla,capture_unit,present_and_preserved,single multimodal forward vision_backbone->projector->cat->language_model->logits; func.func @forward ->1x20x512xf32
```
