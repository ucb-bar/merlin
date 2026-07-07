# smolvla — source → exported-FX → flat-MLIR → Merlin forensic audit

**Workload:** smolvla (LeRobot SmolVLA flow-matching VLA). The **only** capture in the
corpus that retains `scf.for` in its flat MLIR.

**Scope / disclaimers.** Source-grounded; every claim cites a file:line or artifact row.
Magnitudes are **structural-only** — the capture is a small/random-config instance
(`M2M_SMOLVLA_VLM_LAYERS` defaults to 16, weights random init; loader.py docstring), so MAC
counts reflect *shape*, not a trained checkpoint. **No performance/latency claims** are made
(no FireSim/measured cycles exist for smolvla; only xr0 carries `measured_cycles`,
models.py:62). "not visible" is marked as a missing/erased status, never inferred.

**Sources audited**
- Source: `lerobot` v0.5.1 @ `/scratch/agustin/projects/vla-arena/external/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py`
  (`denoise_step`:871, `sample_actions`:800, `embed_prefix`:625, `embed_suffix`:719,
  `make_att_2d_masks`:102); config `configuration_smolvla.py` (chunk_size=50:32, n_action_steps=50:33,
  num_steps=10:66, num_vlm_layers=16:99, vlm_model_name SmolVLM2-500M:87, expert_width_multiplier=0.75:101).
- m2m wrapper: `/scratch/agustin/projects/model2MLIR/workloads/smolvla/{loader.py,capture.toml}`.
- Exported FX: `manual_validation/exported_fx/smolvla.txt` (op histogram at head).
- Flat MLIR: `recaptures/smolvla/model.mlir` (`prov.level = "linalg-on-tensors"`, line 1).
- Merlin: `case_study/{operator_full_inventory,work_coverage_table,operator_shape_table,data_movement_table,resident_state_table,reuse_lifetime_table}.csv`;
  `merlin/python/merlin/dse_guidance/models.py` MODEL_ARCH['smolvla'] (lines 51-52).

---

## Provenance table

Status vocab: `present_and_preserved`, `present_in_source_erased_by_export`,
`present_in_source_erased_by_lowering`, `present_but_not_recovered_by_merlin`,
`not_present_in_source`, `unknown_source_not_found`, `sidecar_or_config_only`.

| Feature | Source | Exported-FX | MLIR (flat) | Merlin | Status |
|---|---|---|---|---|---|
| **model class** | `VLAFlowMatching` (modeling_smolvla.py); wrapped by `SmolVLADenoiseStep` (loader.py) | not a node; only FQN strings (`p_model_vlm_with_expert_...`) | `prov.fqn = model.vlm_with_expert...`, `prov.module="model"` | `models.py` MODEL_ARCH key `smolvla`, family `flow_matching` | present_and_preserved |
| **submodule boundary** (VLM vs expert) | `vlm_with_expert.vlm.model.vision_model` vs `vlm_with_expert.lm_expert.layers.*` | FQN substrings only (no module nodes) | preserved verbatim in `prov.fqn` | role split `backbone_once`(643 ops) vs `repeated_head`(72 ops); 243 unattributed (inv. role col) | present_and_preserved |
| **forward entry** | `denoise_step` is the capture unit (loader.py forward) | single flat `forward` graph | `func.func @forward(...) -> tensor<1x50x32xf32>` (line 2) | n/a (op-level) | present_and_preserved |
| **denoise / action expert head** | `lm_expert` + `action_in_proj`/`action_out_proj`/`state_proj` (denoise_step:879-903) | linear/matmul nodes w/ expert FQNs | `prov.fqn = model.action_out_proj` etc., bf16 expert weights %198+ | inventory rows role `repeated_head`, fqn `lm_expert.layers.*` / `action_*_proj` | present_and_preserved |
| **K denoise loop (K=10)** | `for step in range(num_steps)`, `num_steps=10` (sample_actions:836, config:66) | **erased** — loader captures ONE `denoise_step`; no loop in graph | **absent** — single pass; the only `scf.for`s are gather artifacts (see below) | `loop_count=10`, `loop_count_source="assumed"` (models.py:51); `repeated_head` invocations=10 in data_movement/resident_state | present_in_source_erased_by_lowering → recovered sidecar_or_config_only by Merlin |
| **H (action horizon = 50)** | `chunk_size=50` (config:32); `suffix_out[:, -chunk_size:]` (denoise_step:901) | tensor shapes `[1,50,32]` on action path | output `tensor<1x50x32xf32>`; expert bmm rows M=50 (op 757/817) | `action_horizon=50` (models.py:51); op_shape rows M=50 batch=15 | present_and_preserved (+ config echo) |
| **control cadence (30 Hz)** | **not in source** — no fps/Hz field in config | absent | absent | `control_rate_hz=30.0` (models.py:51, note "30Hz") | sidecar_or_config_only |
| **q/k/v proj** | `self_attn.{q,k,v}_proj` (SmolVLM + Gemma expert) | `aten.linear.default` ×302 | `linalg.matmul`/`addmm`, `semantic_class=attention_qkv_projection` | inventory rows op_class `linear_gemm`, role-split | present_and_preserved |
| **qkᵀ & attn·v bmm** | scaled-dot-product attention inside both encoders | `aten.scaled_dot_product_attention.default` ×12 (vision) + `aten.matmul`×64 | `linalg.generic ... batch_matmul`, `prov.family="contraction"` ×283 | 32 `attention_contraction` rows (two per layer: M=1024/64 vision, M=50 expert) | present_and_preserved |
| **softmax** | attention softmax | `aten.softmax.int` ×32 | `prov._pattern_hint="softmax"`, `prov.family="normalization"`, masked w/ `-inf` (0xff800000) | `n_softmax=48`; op_class `softmax` ×48 | present_and_preserved |
| **KV / prefix state** | `past_key_values` from prefix pass, reused across K (sample_actions:824-831; denoise_step `fill_kv_cache=False`:898) | **erased** — single denoise step gets `past_key_values` as opaque input | KV cache not a named region; prefix folded into per-call attention | `kv_bytes=unavailable` in data_movement_table; modeled as `repeated_head` weights reused K=10 | present_in_source_erased_by_lowering |
| **conv / vision** | SigLIP patch-embed conv (`embed_image`, embed_prefix:653) | `aten.conv2d.padding` ×1 → `f32[1,768,32,32]` (FX:85) | `linalg.generic` conv2d, `prov._pattern_hint="conv2d"` `prov.family="contraction"` (model.mlir:13) | `n_conv=2`; op_class `conv` ×2 | present_and_preserved |
| **linear / GEMM** | all proj / mlp linears | `aten.linear.default` ×302, `aten.matmul` ×64 | `linalg.matmul`/`linalg.generic` | `n_linear_matmul=106`; 106 `linear_gemm` rows | present_and_preserved |
| **dtype** | mixed: VLM f32, expert bf16 (config/checkpoint) | per-tensor f32/bf16 annotations | **visible in `func.func` signature**: %0–196 f32 (VLM), %198+ bf16 (expert) | op_shape `dtype` col = f32 (backbone rows); bf16 on expert | present_and_preserved |
| **accumulator** | implicit fp32 matmul accumulate | not an explicit node | linalg reduction in f32; no explicit acc-dtype attr | not surfaced as a separate field | present_in_source_erased_by_lowering |
| **quant** | **none** — `int8`/`fp8` variants are sidecar files (smolvla_int8.mlir / smolvla_fp8.mlir under m2m), NOT this capture | no quant ops | **no `quant`/`scale`/`dequant` prov attrs** (grep=0); "f8" matches are substrings of `0xff800000` (-inf), not f8 dtype | no quant columns populated | not_present_in_source (this capture) / sidecar_or_config_only (int8/fp8 variants) |
| **scales** | n/a (not quantized) | absent | absent (no `prov.scale`) | `scale_bytes=unavailable` (data_movement_table) | not_present_in_source |
| **packed layout** | n/a | absent | dense `tensor<...>` only; no packed/blocked layout | not present | not_present_in_source |
| **region attribution — backbone (VLM)** | `vlm.model.vision_model` + VLM text layers, run once per replan | FQN strings | `prov.fqn` carries it | role `backbone_once`, invocations=1, weight_bytes 426 MB (data_movement_table) | present_and_preserved |
| **region attribution — repeated_head (expert)** | `lm_expert` + action heads, run K times | FQN strings | `prov.fqn` carries it | role `repeated_head`, invocations=10, `load_once_reuse_K` (reuse_lifetime_table) | present_and_preserved |
| **tied weights** | expert reuses VLM KV/prefix; no explicit weight tying in source read | not represented | distinct weight operands per region | not flagged as tied | not_present_in_source |
| **repeated-weight lifetime** | expert weights live across the K loop (source loop hoists prefix/kv) | erased w/ loop | single pass — lifetime not in-graph | `resident_state_table`: weights `loop_invariant`, `reused_times=10`; `reuse_lifetime_table`: `across_K`, `load_once_reuse_K` | present_in_source_erased_by_lowering → recovered_by_merlin (analytical) |
| **loop-carried latent (x_t)** | `x_t = x_t + dt*v_t` carried across K (sample_actions:864) | **erased** — one step; x_t is `noise` input, v_t is output | not a loop carry; `noise` is func arg %260, output is v_t | modeled implicitly via K replication, not as carried SSA | present_in_source_erased_by_lowering |
| **the scf.for** | **NOT a model loop** — see (b); source has no Python loop here, it is SmolVLM vision-embed mask scatter (`modeling_smolvlm.py:137`) | `aten.index.Tensor` + `aten.index_put_.default` (FX:160,172) — bool-mask gather/scatter | **two `scf.for`** (model.mlir:224, 243) w/ `prov.family="gather_scatter"`, `prov.aten=aten.index.Tensor`/`aten.index_put.default`, `prov.fqn=...vision_model.embeddings` | `prov.family="gather_scatter"` ×18; not a temporal region | present_in_source_erased_by_lowering (data-dependent control, NOT the K loop) |

---

## smolvla-specific findings

**(a) Flow-matching denoise loop; is K=10 captured or sidecar?**
Yes — smolvla is flow-matching (`VLAFlowMatching`; `sample_actions` integrates an ODE,
`dt = -1.0/num_steps`, `x_t = x_t + dt*v_t`, sample_actions:833-864). The K=10 loop is
`num_steps: int = 10` (config:66). The capture is **one** `denoise_step` (loader.py forward;
docstring: "exposes one flow-matching denoise step ... as the capture unit"). So **K is
erased from the graph and re-supplied as a sidecar constant** (`loop_count=10`,
`loop_count_source="assumed"`, models.py:51). Merlin reflects K only analytically:
`repeated_head` carries invocations=10 (data_movement_table) and
`reused_times=10 / load_once_reuse_K` (resident_state / reuse_lifetime tables).

**(b) What the scf.for EXACTLY is — confirmed gather artifact, NOT the denoise loop.**
The two `scf.for` (model.mlir:224 and 243) are a **bool-mask gather/scatter pair**, not the
flow-matching loop:
- They carry `prov.family = "gather_scatter"`, `prov._pattern_hint = "mask_gather"` /
  `"index_put"`, `prov.aten = "aten.index.Tensor"` / `"aten.index_put.default"`, and
  `prov.fqn = model.vlm_with_expert.vlm.model.vision_model.embeddings` (model.mlir:219-243).
- Loop body is a mask-driven compaction: `%470 = tensor.extract %447[%467] : tensor<1024xi1>`
  (read mask bit), `scf.if %470 { tensor.insert ... ; addi count }` (conditional
  append) — the canonical compaction lowering of a boolean fancy-index.
- Source: SmolVLM vision embeddings,
  `position_ids[patch_attention_mask.view(batch_size,-1)] = pos_ids[patch_attention_mask.view(batch_size,-1)]`
  at `transformers/models/smolvlm/modeling_smolvlm.py:137` (cited verbatim in FX comments at
  smolvla.txt:158, 166). Exported as `aten.index.Tensor` (FX:160) feeding
  `aten.index_put_.default` (FX:172), then `aten.embedding` (FX:175).
- The dynamic count `u0` (constrained `0 <= u0 <= 1024`, FX:163-170) is what forces a
  *data-dependent* loop instead of a static `linalg.generic` — that is the entire reason this
  one capture retains `scf.for`. It is the SmolVLM patch-mask compaction, period.

The flow-matching K loop, by contrast, never appears in the IR (single denoise-step capture).

**(c) Action horizon + control rate beyond MODEL_ARCH?**
- **H=50 is source-real and graph-visible**: output `tensor<1x50x32xf32>` (func sig, model.mlir:2),
  `suffix_out[:, -chunk_size:]` with chunk_size=50 (denoise_step:901, config:32); echoed in
  MODEL_ARCH `action_horizon=50`. So H is *not* MODEL_ARCH-only.
- **Control rate 30 Hz is MODEL_ARCH-only** (`control_rate_hz=30.0`, models.py:51). There is
  **no fps/Hz field in `configuration_smolvla.py`** — it is a sidecar/config annotation, not
  source- or graph-derived.

**(d) VLM-backbone vs action-expert split — captured & correctly attributed?**
Yes. The split is preserved through `prov.fqn` and correctly mapped to cadence roles:
- `backbone_once` (643 ops, invocations=1, 426 MB weights, lifetime `across_replan`) =
  `vlm.model.vision_model` + VLM text layers, f32 weights (%0–196).
- `repeated_head` (72 ops, invocations=10, 31 MB weights, `load_once_reuse_K`) =
  `lm_expert.layers.*` + `action_in_proj`/`action_out_proj`/`state_proj`, bf16 weights (%198+).
The mixed-precision boundary (f32 VLM / bf16 expert) is directly visible in the `func.func`
signature and matches the module boundary. (243 inventory rows have an empty role column —
these are unattributed elementwise/layout glue, not a mis-attribution of compute regions.)

**(e) Why is smolvla's attention MAC fraction the highest in the corpus?**
`work_coverage_table.csv`: smolvla `visible_linear_fraction = 0.8237` — the **lowest** linear
fraction = **highest attention fraction** of all 11 workloads (next is groot_n1d7 0.8804; LLMs
are ~0.99). Attention MACs = 19.4 G of 110 G recovered ≈ **17.6%**. This is **source-real and
structural**: smolvla runs a full **SmolVLM2-500M vision+text transformer backbone over a
1024-token image-patch prefix** (per-layer attention bmm M=1024,N=1024 = 0.81 G MACs ×2 ×16
layers, op rows 46/55 etc.), whereas the LLM/AR-VLA workloads attend over tiny action-token
sequences (M≤32). The large 1024-token prefix sequence — not quantization or a capture
artifact — is what drives the attention share up. (pi05, the other big VLM-backbone
flow-matcher, has the largest *absolute* attention MACs at 81 G but a higher linear fraction
0.9635 because its MLP/proj GEMMs are even larger.) Magnitudes are structural (random init,
16 VLM layers), so this is a *shape-driven* ratio, not a trained-checkpoint measurement.

---

## FINDINGS

1. **The `scf.for` is a gather artifact, not a model loop.** Both `scf.for` (model.mlir:224,243)
   are the SmolVLM vision-embedding boolean-mask compaction
   (`position_ids[mask] = pos_ids[mask]`, modeling_smolvlm.py:137) — `prov.family="gather_scatter"`,
   `aten.index.Tensor`/`index_put`, dynamic count `u0∈[0,1024]`. The flow-matching K loop is
   **absent** from the IR.
2. **K=10 is erased and re-supplied as a config sidecar** (`loop_count_source="assumed"`).
   Merlin recovers it only analytically as `repeated_head` invocations=10 / reuse_times=10.
3. **The VLM-backbone vs action-expert split is faithfully preserved** through `prov.fqn` and
   correctly mapped to `backbone_once` (f32, 1×) vs `repeated_head` (bf16, K×, load_once_reuse_K).
4. **H=50 is graph-visible and source-real**; **30 Hz control rate is MODEL_ARCH-only** (no
   source field).
5. **No quantization in this capture** — int8/fp8 are separate sidecar MLIRs; the "f8" grep
   hits are `0xff800000` (-inf softmax mask) substrings, not an f8 dtype.
6. **Highest attention MAC fraction (~18%, visible_linear_fraction 0.8237) is structural and
   source-real**, driven by the 1024-token SmolVLM image-patch prefix, not by lowering.

```SUMMARY_ROWS
smolvla,scf_for_is_gather_not_loop,present_in_source_erased_by_lowering,"two scf.for at model.mlir:224/243 carry prov.family=gather_scatter aten.index.Tensor/index_put fqn=...vision_model.embeddings; source modeling_smolvlm.py:137 position_ids[mask]=pos_ids[mask]; FX index_put_ at smolvla.txt:172; NOT the K denoise loop"
smolvla,flow_matching_K_denoise_loop,present_in_source_erased_by_lowering,"sample_actions:836 for step in range(num_steps=10) config:66; loader captures ONE denoise_step; Merlin loop_count=10 source=assumed models.py:51, repeated_head invocations=10"
smolvla,action_horizon_H50,present_and_preserved,"chunk_size=50 config:32, suffix_out[:,-50:] denoise_step:901, MLIR output tensor<1x50x32xf32> model.mlir:2, MODEL_ARCH action_horizon=50"
smolvla,control_rate_30hz,sidecar_or_config_only,"no fps/Hz field in configuration_smolvla.py; only models.py:51 control_rate_hz=30.0"
smolvla,vlm_backbone_vs_expert_split,present_and_preserved,"prov.fqn vision_model vs lm_expert; backbone_once 643 ops inv=1 426MB f32 %0-196, repeated_head 72 ops inv=10 31MB bf16 %198+ load_once_reuse_K"
smolvla,attention_mac_fraction_highest,present_and_preserved,"work_coverage visible_linear_fraction=0.8237 (lowest=>attn highest); attn 19.4G/110G ~17.6%; 1024-token SmolVLM patch prefix bmm M=1024 ops 46/55; structural random-init"
smolvla,quant_scales,not_present_in_source,"no quant/scale/dequant prov attrs in model.mlir (grep=0); f8 hits are 0xff800000 -inf softmax mask; int8/fp8 are separate sidecar MLIRs"
smolvla,kv_prefix_state,present_in_source_erased_by_lowering,"past_key_values prefix reuse sample_actions:824-831 denoise_step fill_kv_cache=False:898; kv_bytes=unavailable in data_movement_table"
smolvla,dtype_mixed_f32_bf16,present_and_preserved,"func.func sig %0-196 f32 (VLM) %198+ bf16 (expert) model.mlir:2"
smolvla,qkv_proj_and_attn_bmm,present_and_preserved,"aten.linear x302 + sdpa x12 + matmul x64 in FX; 32 attention_contraction batch_matmul rows; 48 softmax"
smolvla,conv_vision_patch_embed,present_and_preserved,"SigLIP conv embed_prefix:653; aten.conv2d.padding FX:85; linalg conv2d model.mlir:13; Merlin n_conv=2"
```
