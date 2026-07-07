# bitvla — Source → FX → MLIR → Merlin artifact forensic audit

**Workload:** bitvla (BitNet / W1.58-A8 ternary VLA, OpenVLA-OFT variant)
**Question under test:** is the low-bit structure *real* in the artifacts, or *erased*?
**Verdict (one line):** the source HAS genuine ternary/packed low-bit machinery, but the captured path **runs the fake-quant (simulated) branch** and **export dequantizes everything to f32**; the true `quantize_to_int2` packed path is never entered and never reaches FX/MLIR. Merlin correctly flags the low-bit abstractions as structurally-legal-but-quantification-blocked.

All claims are random-init (small smoke config: `BITVLA_LLM_LAYERS=2`, `hidden=256`); **no perf claims**, magnitudes are structural only.

---

## Evidence anchors

- **Source model class / forward entry:** `bitvla_for_action_prediction.py:21` (`class BitVLAForActionPrediction(LlavaForConditionalGeneration)`), `predict_action` at `:312`.
- **BitLinear + quant machinery (the crux):** `transformers/src/transformers/models/llava/modeling_bitnet.py`
  - `WeightQuant` (fake-quant, ternary STE) `:85-99` — `s = 1/x.abs().mean()`, `x=(x*s).round().clamp(-1,1)/s` → returns **f32**.
  - `ActQuant` (A8, per-token absmax) `:101-115` — `s=127/x.abs().max(dim=-1)`, `round().clamp(-128,127)/s` → returns **f32**.
  - `absmean` `:118-125`; `quantize_to_int2` (TRUE packing, 4×uint2 → 1 uint8) `:128-141`; `dequantize_from_int2` `:145-154`.
  - `class BitLinear(nn.Linear)` `:157-193`. `forward` `:182-193`:
    ```python
    def forward(self, input):
        input = ActQuant.apply(input)                       # A8 fake-quant (f32 out)
        if self.enable_qlora:                               # packed path (NOT taken in capture)
            weight = dequantize_from_int2(self.q_weight, self.w_step.item(), ...).type(input.dtype)
        else:                                               # capture default: enable_qlora=False
            weight = WeightQuant.apply(self.weight)         # ternary fake-quant, f32 weight
        return F.linear(input, weight, self.bias)           # f32 GEMM
    ```
  - `enable_qlora` is only set True inside `quantize_weights()` (`:166-180`), which registers `q_weight`/`w_step` buffers. **The m2m loader never calls `quantize_weights()`** → `enable_qlora=False` → the **`WeightQuant.apply` fake-quant branch** is what gets traced.
- **m2m loader:** `model2MLIR/workloads/bitvla/loader.py:140-230` builds a random config, `vla = BitVLAForActionPrediction(cfg).to(torch.float32)`, captures the inner VLM forward on `inputs_embeds` (f32). No `quantize_weights()` call; header comment `loader.py:18` explicitly says it traces "W1.58-A8 BitLinear math (round/clamp/absmean quant, traced through each autograd.Function's forward)" — i.e. the *simulated* path.
- **capture.toml:** `model2MLIR/workloads/bitvla/capture.toml:8-15` — `# BitNet (W1.58) layers can't be torchao-quantized (abs().mean() unsupported); quantize only the plain lm_head Linear`. `[quant.int8] per_module = { lm_head = "int8_weight_only" }`, `[quant.fp8] lm_head=float8_weight_only_e4m3`.
- **Exported FX:** `case_study/manual_validation/exported_fx/bitvla.txt`. Op histogram: `28 aten.round.default`, `28 aten.clamp_.default`, `28 aten.clamp.default`, `15 aten.linear.default`, `4 aten.matmul.default`, `2 aten.softmax.int`. Every BitLinear param is declared `f32` (line 50 signature: `..._q_proj_weight: "f32[256,256]"`, etc.). The quant is open-coded: `:115-123` (ActQuant), `:133-144` (WeightQuant) — all `f32`, ending in `F.linear` (`:151` `linear: "f32[1,32,256]" = aten.linear.default(...)`, source-mapped to `modeling_bitnet.py:193`). **No `int2`/`uint8`/pack/`q_weight`/`w_step` nodes anywhere.**
- **Flat MLIR:** `recaptures/bitvla/model.mlir`. Dtype histogram: **4572 f32, 1010 i64, 38 i32, 0 low-bit** (the "f8" greps are all `0xff800000` = -inf softmax/max constants, not a dtype). Fake-quant survives as provenance-tagged `linalg.generic` regions: `prov.family="minmax"` (clamp), `prov.op="round"` (`math.roundeven`), feeding `linalg.matmul {prov.family="contraction", prov.transposed_b="true"}` — e.g. `model.mlir:189/217/314` for `vla.language_model.model.layers.0.self_attn.q_proj`. All ins/outs `tensor<...xf32>`.
- **Level MLIR:** `recaptures_levels/bitvla/model_{highlevel,qdq}.mlir`. qdq level keeps the same `round`/`minmax`/`matmul` f32 regions (`model_qdq.mlir:189/217/314`); high-level: `140 round`, `15 linalg.matmul`, `371 linalg.generic`, **no real f8/i8/i2 dtype** (its 14 "f8" are again `0xff800000`). No `quant.` dialect, no QDQ-pair op, no packed-tensor type at any level.
- **Safetensors (the "all F32?" check):** `model2MLIR/workloads/bitvla/bitvla.safetensors.manifest.json` → **77/77 tensors `float32`**. `bitvla_int8.safetensors.manifest.json` → 76 f32 + 1 `int8` + 1 i64; the lone int8 tensor (key `"76"`) is `vla.language_model.lm_head.parametrizations.weight.original0` shape `[1024,256]`. `bitvla_fp8` → 76 f32 + 1 `float8_e4m3fn`, same lm_head. **Only lm_head is ever stored low-bit; every BitLinear weight is f32.**
- **Merlin artifacts:** `case_study/abstraction_pressure_table.csv:95,97` (`packed_layout_preservation`, `resident_packed_lowbit_weights` → `missing accuracy sweep + low-bit kernel calibration + resident-capacity model`); `case_study/bitvla/abstraction_candidates.yaml:161,211` (status `legality: structural`, `quantification_blocked_by: missing_calibration`); `case_study/bitvla/numerical_candidate_certificates.yaml:8` (`dse_status: accuracy_legal_structural_candidate`, dtype `int8_w8a8`, requires `packed_lowbit_tensor + resident_weight_object + scale_object`); fp8/int4 variants `:35,62,89` `blocked_by_missing_accuracy`. `operator_full_inventory.csv` has 392 bitvla rows; `work_coverage_table.csv` 1 bitvla row. `capture_fidelity_matrix.csv` **not present** in `case_study/`.

---

## Source / Exported FX / MLIR / Merlin artifact table

| Feature (row) | Source | Exported FX | MLIR (flat + levels) | Merlin artifact |
|---|---|---|---|---|
| model class | `bitvla_for_action_prediction.py:21` BitVLAForActionPrediction → **present_and_preserved** | captured unit is inner VLM forward (loader `_VLMLogits`); class name not in graph → **present_in_source_erased_by_export** | single `func` module, no class boundary → **present_in_source_erased_by_lowering** | `prov.module="vla"` on every op → **present_and_preserved** |
| submodule boundary (attn/mlp/BitLinear) | `BitNetAttention:319`, `BitNetMLP:289`, `BitLinear:157` → **present_and_preserved** | flattened to op list; recoverable only via source-file comments in node meta → **present_in_source_erased_by_export** | recoverable via `prov.fqn` (e.g. `...layers.0.self_attn.q_proj`) → **present_and_preserved** (in provenance) | `region_attribution.yaml`, `operator_full_inventory.csv` keyed by fqn → **present_and_preserved** |
| forward entry | `forward` `:41` / `predict_action:312` (host prep excluded) → **present_and_preserved** | graph root = inner forward on `inputs_embeds` → **present_in_source_erased_by_export** (host-side prep dropped by design) | `func.func` root → **present_but_not_recovered_by_merlin** (no top-entry semantics) | workload_contract entry implied → **present_and_preserved** |
| action / lm head | `lm_head` (Llava), action binning `bitvla_for_action_prediction.py:29` (`bin_centers`), `_regression_or_discrete_prediction` → **present_and_preserved** in source | capture stops at logits; action head + binning **not in graph** (host-side) → **present_in_source_erased_by_export** | absent → **present_in_source_erased_by_lowering** | sidecar only (capture unit note) → **sidecar_or_config_only** |
| K decode loop | none — bi-directional single forward, `use_bi_attn=True` (`loader.py:14`, `predict_action` no autoregressive loop) → **not_present_in_source** | n/a → **not_present_in_source** | n/a → **not_present_in_source** | n/a → **not_present_in_source** |
| H (chunk / heads) | heads `num_attention_heads=8`, kv heads `4` (`loader.py` config); action chunk host-side → **present_and_preserved** | head reshape/transpose in graph (`view`/`transpose`) → **present_and_preserved** | shape `tensor<...x8x...>` reshapes in MLIR → **present_and_preserved** | `operator_shape_table.csv` / `shape_summary_by_region.csv` → **present_and_preserved** |
| cadence (multi-rate) | single-rate (one forward, no denoise loop) → **not_present_in_source** | single graph → **not_present_in_source** | single region set → **not_present_in_source** | `multi_rate_contract.yaml` (single-rate) → **sidecar_or_config_only** |
| q/k/v proj | `BitLinear` q/k/v/o `modeling_bitnet.py:349-352` → **present_and_preserved** | `aten.linear` ×15 with fqn comments → **present_and_preserved** | `linalg.matmul prov.fqn=...q_proj/k_proj/v_proj` (`model.mlir:314/466/...`) → **present_and_preserved** | `operator_full_inventory.csv` (per-proj rows) → **present_and_preserved** |
| qkᵀ & attn·v bmm | `torch.matmul(q,kᵀ)/√d` `:406`, `matmul(attn,v)` → **present_and_preserved** | `aten.matmul.default ×4` → **present_and_preserved** | `linalg.matmul`/`linalg.batch_matmul` f32 → **present_and_preserved** | `operator_cluster_table.csv` contraction rows → **present_and_preserved** |
| softmax | attn softmax (HF eager path) → **present_and_preserved** | `aten.softmax.int ×2` → **present_and_preserved** | `prov.op="softmax"` region (`model.mlir:786`) → **present_and_preserved** | `epilogue_pattern_table.csv` / normalization family → **present_and_preserved** |
| KV cache | `past_key_value.update` `:399-401` (Cache) — but capture is single bi-dir forward, no past → **present_in_source_erased_by_export** | no cache nodes (use_cache off in capture) → **present_in_source_erased_by_export** | absent → **present_in_source_erased_by_lowering** | `quantized_KV_cache` candidate flagged speculative `abstraction_pressure_table.csv:99` → **sidecar_or_config_only** |
| conv | SigLIP `patch_embedding` Conv2d (vision tower) — vision tower runs **host-side**, out of capture (`loader.py:18-22`) → **present_in_source_erased_by_export** | f32[128,3,16,16] patch_embedding weight is an *unused arg* in signature; no conv op in body → **present_in_source_erased_by_export** | no `linalg.conv` in captured body → **present_in_source_erased_by_lowering** | not in operator inventory body → **present_but_not_recovered_by_merlin** |
| linear / GEMM | `BitLinear`/`F.linear` `:193` → **present_and_preserved** | `aten.linear.default ×15` → **present_and_preserved** | `linalg.matmul transposed_b=true` f32 → **present_and_preserved** | contraction family in inventory → **present_and_preserved** |
| dtype (weights/acts) | **W ternary {-1,0,+1}, A int8** semantically; but tensors held as **f32** and quantized via STE (`WeightQuant`/`ActQuant` return f32) → **present_and_preserved** (as fake-quant f32) | all params `f32[...]`, quant open-coded round/clamp in f32 → **present_in_source_erased_by_export** (true low-bit dtype simulated away) | 4572×f32, 0 low-bit dtypes → **present_in_source_erased_by_lowering** | `accuracy_gated_dtype_candidates.csv` lists int8/fp8/int4 as candidates → **present_but_not_recovered_by_merlin** (candidate-only, not in graph) |
| accumulator dtype | implicit f32 (F.linear accum); never written as i32 → **not_present_in_source** (no explicit i32 accumulator) | f32 matmul output, no accum type → **not_present_in_source** | matmul out `tensor<...xf32>`; no i32 accumulator → **not_present_in_source** | `accumulator_contract_table.csv` proposes accumulator_object → **sidecar_or_config_only** |
| quant metadata (method/bits) | `WeightQuant`=absmean ternary, `ActQuant`=absmax-per-token A8; documented in FX header → **present_and_preserved** | FX header line 1-2: "weight bits: 32 ... method: absmean, act bits: 32 ... absmax_per_token" (note bits **collapsed to 32**) → **present_in_source_erased_by_export** | survives only as `prov.family="minmax"/"round"` tags, no bit-width metadata → **present_in_source_erased_by_lowering** | numerical_contract.yaml records absmean/absmax method → **present_and_preserved** (as contract) |
| packed layout (uint2×4→uint8) | `quantize_to_int2` `:128-141` produces real packed uint8 — **present_and_preserved** in source, but **dead in capture** (path not entered) | no pack/bitshift/uint8 nodes → **present_in_source_erased_by_export** | no packed type, no bit-shift `linalg` → **present_in_source_erased_by_lowering** | `packed_layout_preservation` candidate `abstraction_pressure_table.csv:95` blocked → **present_but_not_recovered_by_merlin** |
| scales / zero-points | per-**tensor** weight scale `step=absmean` (scalar, `:130`); per-**token** act scale (`:108`, dim=-1); zero-point = symmetric (none) → **present_and_preserved** | scales appear as f32 `mean`/`max`/`reciprocal`/`mul`/`div` chains (`:115-123,133-144`), not a scale object → **present_in_source_erased_by_export** | f32 scalar/`1x32x1` minmax+div regions, no scale-as-metadata → **present_in_source_erased_by_lowering** | `scale_object` listed as required hw abstraction (`numerical_candidate_certificates.yaml:17`) → **present_but_not_recovered_by_merlin** |
| region attribution | fqn-bearing modules → **present_and_preserved** | source-file/line comments per node → **present_and_preserved** | `prov.region_id`/`prov.fqn` on every op → **present_and_preserved** | `bitvla/region_attribution.yaml` → **present_and_preserved** |
| tied weights | config `tie_word_embeddings=False` (`loader.py`) → embed/lm_head untied → **not_present_in_source** | separate `embed_tokens_weight` + `lm_head_weight` f32 in signature → **not_present_in_source** | two distinct constants → **not_present_in_source** | n/a → **not_present_in_source** |
| repeated-weight lifetime | BitLinear weights invariant across tokens / re-used each forward (repeated_head region) → **present_and_preserved** | static params, constant across graph → **present_and_preserved** | constant operands to matmul → **present_and_preserved** | `reuse_lifetime_table.csv`, `state_lifetime.yaml`, candidate "weights invariant across K" → **present_and_preserved** |
| loop-carried state | none in capture (no decode loop, no cache update in graph) → **not_present_in_source** (within capture unit) | no carried state → **not_present_in_source** | no `scf`/iter_args → **not_present_in_source** | `resident_state_table.csv` minimal → **sidecar_or_config_only** |

---

## bitvla-specific answers

**(a) TRUE packed low-bit weights or fake-quant f32?**
Source contains BOTH. `quantize_to_int2` (`modeling_bitnet.py:128-141`) is genuine packing (`round().clamp(-1,1).to(uint8)+1`, then `q0|q1<<2|q2<<4|q3<<6` → 4 ternary values per uint8) with a per-tensor `step=absmean`. BUT the captured `BitLinear.forward` (`:182-193`) only enters that path when `enable_qlora=True`, which requires `quantize_weights()` to have been called. The loader never calls it → `enable_qlora=False` → the traced path is `WeightQuant.apply(self.weight)` (`:192`), a **straight-through fake-quant that returns f32** (`:93-94`). So the *captured* weights are fake-quant f32, even though the real packed kernel exists in source.

**(b) Where are scales / granularity?**
Weight scale: per-**tensor** scalar `step = weight.abs().mean().clamp(1e-5)` (`:130`, also `WeightQuant:92`). Activation scale: per-**token** (per last-dim row) `s = 127/x.abs().max(dim=-1,keepdim=True)` (`ActQuant:108`). Symmetric → no zero-point. No per-channel, no group. In FX/MLIR these are open-coded `mean`/`max`+`reciprocal`+`mul`/`div` arithmetic, not a named scale object.

**(c) Is low-bit compute actually used, or dequantized to f32?**
Dequantized. Even the packed branch ends `dequantize_from_int2(...).type(input.dtype)` then `F.linear` (`:185-193`); the captured branch does the same with `WeightQuant`. Activations are quant→dequant'd back to f32 by `ActQuant` before the GEMM. Compute is **always f32 `F.linear`** — there is no integer/ternary matmul anywhere.

**(d) Do FX / MLIR preserve packed layout + scales, or only f32 matmul?**
Only f32. FX: all params `f32[...]`, quant is `round`/`clamp`/`div` in f32, terminating in `aten.linear`/`aten.matmul` f32 (no `int2`/`uint8`/pack nodes; header even reports "weight bits: 32"). Flat MLIR: 4572 f32 / 0 low-bit dtypes; fake-quant survives only as `prov.family="minmax"/"round"` tags around `linalg.matmul ...xf32`. **Safetensors: base manifest = 77/77 `float32`.** The int8/fp8 variants quantize exactly **one** tensor — `vla.language_model.lm_head...weight` `[1024,256]` (int8 / float8_e4m3fn); every BitLinear weight stays f32. Matches `capture.toml` ("quantize only the plain lm_head").

**(e) Is "low-bit blocked" because source lacks it, export erased it, or Merlin can't recover it?**
Precisely: **source HAS it; the capture takes the fake-quant branch and export DEQUANTIZES it to f32.** It is NOT a source gap and NOT primarily a Merlin-recovery gap. The packed `quantize_to_int2`/`enable_qlora` path is real but dead in this capture (never invoked by the loader), and `torch.export` traces the f32 STE branch — so the packed layout, the uint8 type, and the scale-as-metadata are gone before Merlin ever sees the graph. Merlin then can only see f32 round/clamp/matmul, which it correctly tags but cannot promote to a packed-low-bit contract without a re-capture of the packed path + an accuracy sweep.

**(f) Are packed_lowbit / scale_object / native_lowbit_matmul abstractions correctly blocked?**
Yes, correctly. `resident_packed_lowbit_weights` requiring `packed_lowbit_tensor + resident_weight_object + scale_object` is marked `legality: structural` / `quantification_blocked_by: missing_calibration` (`abstraction_candidates.yaml:211`) and `accuracy_legal_structural_candidate` for int8_w8a8 (measured_pass) vs `blocked_by_missing_accuracy` for fp8/int4 (`numerical_candidate_certificates.yaml:8,35,62,89`). `packed_layout_preservation` likewise blocked (`abstraction_pressure_table.csv:95`). This is the right call: the abstractions are structurally legal (the model IS ternary) but cannot be quantified because the captured graph is f32 fake-quant, so there is no packed tensor / native low-bit matmul in-graph to attach them to.

---

## FINDINGS

**Right**
- Base-capture weights are f32 fake-quant, not packed low-bit: corroborated three ways (FX params f32 + open-coded round/clamp; flat MLIR 4572 f32 / 0 low-bit dtype; safetensors 77/77 float32). Solid.
- Only `lm_head` is genuinely quantized in the int8/fp8 variants (key 76 = `lm_head...weight`), exactly matching `capture.toml`'s "BitNet layers can't be torchao-quantized." Solid.
- Compute is always f32 `F.linear`; both BitLinear branches dequantize before the GEMM. Solid.
- Merlin's low-bit abstractions are correctly `structural / blocked_by_missing_calibration` (not falsely "present" and not wrongly "absent in source").

**Wrong / corrected**
- "f8 in MLIR" is a **false positive**: every `f8` grep hit is the `0xff800000` (-inf) softmax/max constant, not a float8 dtype. The captured flat MLIR has **no** float8.
- Do NOT claim "low-bit blocked because source lacks low-bit" — source HAS the full packed machinery; the loss is capture-branch + export dequantization.

**Weak / not-visible (marked missing)**
- `capture_fidelity_matrix.csv` is **not present** in `case_study/` (could not cross-check fidelity row).
- The packed `enable_qlora` path is never exercised, so its FX/MLIR shape is **unknown** (would need a re-capture with `quantize_weights()` called); claims about how packing *would* lower are not source-grounded here.
- Vision-tower Conv2d and the action head/binning are host-side and out of the capture unit — present in source, absent from graph by design (not a defect).

```SUMMARY_ROWS
bitvla,model_class,present_and_preserved,bitvla_for_action_prediction.py:21; prov.module=vla on every MLIR op
bitvla,submodule_boundary,present_and_preserved,erased in FX flatten but recoverable via prov.fqn (e.g. layers.0.self_attn.q_proj)
bitvla,forward_entry,present_in_source_erased_by_export,capture unit = inner VLM forward on inputs_embeds (loader.py:14); predict_action host-prep dropped
bitvla,action_lm_head,present_in_source_erased_by_export,binning bin_centers bitvla_for_action_prediction.py:29 host-side; capture stops at logits
bitvla,K_decode_loop,not_present_in_source,bi-directional single forward use_bi_attn=True; no autoregressive loop
bitvla,H_heads,present_and_preserved,num_attention_heads=8 kv=4; view/transpose in FX+MLIR; operator_shape_table.csv
bitvla,cadence,not_present_in_source,single-rate one forward; multi_rate_contract.yaml single-rate sidecar
bitvla,qkv_proj,present_and_preserved,BitLinear q/k/v/o modeling_bitnet.py:349-352; aten.linear x15; linalg.matmul prov.fqn=*_proj
bitvla,qk_attn_v_bmm,present_and_preserved,matmul(q,kT)/sqrt(d) :406; aten.matmul x4; linalg.matmul f32
bitvla,softmax,present_and_preserved,aten.softmax.int x2; prov.op=softmax model.mlir:786
bitvla,kv_cache,present_in_source_erased_by_export,past_key_value.update :399-401 but use_cache off in capture; quantized_KV_cache candidate speculative
bitvla,conv,present_in_source_erased_by_export,SigLIP patch_embedding host-side; unused f32[128,3,16,16] arg, no conv op in body
bitvla,linear_gemm,present_and_preserved,F.linear :193; linalg.matmul transposed_b=true f32
bitvla,dtype,present_in_source_erased_by_lowering,W-ternary/A8 semantics but tensors f32; FX all f32; MLIR 4572 f32 / 0 low-bit; safetensors 77/77 float32
bitvla,accumulator_dtype,not_present_in_source,implicit f32 F.linear accum; no explicit i32; accumulator_contract_table.csv is sidecar proposal
bitvla,quant_metadata,present_in_source_erased_by_export,WeightQuant=absmean ternary ActQuant=absmax-per-token; FX header collapses bits to 32; MLIR only prov.family=minmax/round
bitvla,packed_layout,present_but_not_recovered_by_merlin,quantize_to_int2 modeling_bitnet.py:128-141 real but dead (enable_qlora=False); no pack/uint8 in FX/MLIR; packed_layout_preservation blocked
bitvla,scales_zero_points,present_in_source_erased_by_export,per-tensor weight absmean :130 + per-token act absmax :108 symmetric; open-coded mean/max/div f32, no scale_object
bitvla,region_attribution,present_and_preserved,prov.region_id/prov.fqn on every op; bitvla/region_attribution.yaml
bitvla,tied_weights,not_present_in_source,tie_word_embeddings=False; embed_tokens + lm_head distinct f32 constants
bitvla,repeated_weight_lifetime,present_and_preserved,BitLinear weights invariant per forward; reuse_lifetime_table.csv; "weights invariant across K"
bitvla,loop_carried_state,not_present_in_source,no decode loop / no cache update in captured graph; no scf iter_args
bitvla,low_bit_compute,present_in_source_erased_by_export,both BitLinear branches dequant before F.linear; compute always f32; no integer/ternary matmul anywhere
bitvla,packed_lowbit_abstraction,present_but_not_recovered_by_merlin,resident_packed_lowbit_weights structural/blocked_by_missing_calibration; correctly NOT promoted (graph is f32 fake-quant)
```
