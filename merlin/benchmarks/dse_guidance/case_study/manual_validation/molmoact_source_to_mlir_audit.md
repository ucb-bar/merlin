# molmoact — source → FX → MLIR → Merlin forensic audit

**Workload:** `molmoact` (AllenAI MolmoAct — Molmo-lineage vision-language-action model).
**Source of record:** `/scratch/agustin/projects/molmoact/olmo/hf_model/molmoact/modeling_molmoact.py`.
**Capture unit (what was actually exported):** the LLM decoder only, as a clean
`input_ids → logits` causal-LM forward (`MolmoActForCausalLM` wrapped by `_LogitsOnly`);
ViT, adapter, image-token splicing, and the generation loop are deliberately excluded
(`/scratch/agustin/projects/model2MLIR/workloads/molmoact/loader.py` docstring + `get_model_and_inputs`).
**Capture config:** `M2M_MOLMOACT_LAYERS=4` (full=48), `M2M_MOLMOACT_VOCAB=4096` (full=152064),
`M2M_SEQ=8`, `use_cache=False`, `_attn_implementation="eager"`
(`.../workloads/molmoact/capture.toml`, `loader.py:get_model_and_inputs`).
**Random init:** all weights random — magnitudes here are *structural only* (shapes / op counts / MAC
counts), never performance.

## Provenance / artifact sources
- Exported FX op histogram + graph: `case_study/manual_validation/exported_fx/molmoact.txt`.
- Flat MLIR (the capture Merlin reads): `merlin/benchmarks/dse_guidance/recaptures/molmoact/model.mlir`
  (1850 lines; no per-level MLIR for this workload).
- Merlin artifacts: `case_study/{operator_full_inventory.csv, work_coverage_table.csv, operator_shape_table.csv}` (grep `^molmoact,`).
- Reference arch table: `merlin/python/merlin/dse_guidance/models.py` `MODEL_ARCH["molmoact"]`
  (`autoregressive_vla`, `action_token_decode`, K=8, control_rate=5.0 Hz, H=8).

---

## Four-column status table

Columns: **Source** = present in PyTorch source; **Exported-FX** = survives torch.export into
`exported_fx/molmoact.txt`; **MLIR** = present in flat `recaptures/molmoact/model.mlir`;
**Merlin** = recovered into the case_study CSVs.

| Feature | Source | Exported-FX | MLIR | Merlin | Status |
|---|---|---|---|---|---|
| model class | `MolmoActForCausalLM` (modeling_molmoact.py:1493); full VLA = `MolmoActForActionReasoning` (:1790) | wrapper `_LogitsOnly` over `MolmoActForCausalLM` (FX root `GraphModule`) | flat func; class name erased, `prov.module="lm"` only | name not a Merlin field; only `prov.module`/`fqn` | **present_in_source_erased_by_export** (full VLA class never entered; only the causal-LM submodel is captured) |
| submodule boundary | `MolmoActLlm.blocks[i]` (:1239,1331) | `p_lm_model_blocks_slice_none__4__none____modules__N___...` param names | `prov.fqn="lm.model.blocks.slice(None, 4, None).N.self_attn"` etc. | `module`/`region_fqn` cols carry the same fqn | **present_and_preserved** |
| forward entry | `MolmoActForCausalLM.forward` (:1528) | FX `forward(...)` root | flat top-level func | implicit (single graph) | **present_and_preserved** |
| action / lm head | `lm_head = nn.Linear(H, vocab, bias=False)` (:1503); action de-tokenize in `parse_action` (:1936) | `aten.linear` → `p_lm_lm_head_weight: f32[4096,3584]` | `matmul` region `matmul_25`, fqn `lm.lm_head` | `operator_shape_table` row16 `lm.lm_head` matmul 8×4096×3584, `lm_head_projection` | **present_and_preserved** (lm_head); action de-tokenize is **present_in_source_erased_by_export** (host-side numpy in `parse_action`, never traced) |
| K decode loop | `generate()` via `GenerationMixin`; K=8 action tokens (MODEL_ARCH; openvla analogue :64) | NOT in graph — single forward, no loop | no loop in MLIR | not an op; K only in `MODEL_ARCH["molmoact"].loop_count=8` | **present_in_source_erased_by_export** (capture is one prefill pass; K lives only in the sidecar arch table) |
| H (action horizon) | not in modeling source; only `MODEL_ARCH` H=8 | absent | absent | `MODEL_ARCH["molmoact"].action_horizon=8` | **sidecar_or_config_only** |
| cadence (control rate) | not in modeling source | absent | absent | `MODEL_ARCH["molmoact"].control_rate_hz=5.0` | **sidecar_or_config_only** |
| q/k/v proj | fused `att_proj = nn.Linear(H, 3584+512+512)` then `.split` (:866,912-913) | `aten.linear` f32[4608,3584] + `aten.split_with_sizes [3584,512,512]` | `addmm` region `matmul_0..` per-layer fqn `...self_attn.att_proj`; `split` op | `operator_shape_table` att_proj rows (8×4608×3584, addmm/bias_addmm) | **present_and_preserved** (fused QKV preserved as one GEMM + split) |
| qkᵀ & attn·v bmm | `torch.matmul(q, kᵀ)` and `torch.matmul(attn, v)` in `eager_attention_forward` (:821,829) | `aten.matmul` ×2 per layer (8 matmuls total in histogram) | 2 `prov.op="batch_matmul"` (`aten.bmm.default`) per layer, fqn `...N.self_attn` (regions matmul_2/3, _8/9, _14/15, _20/21) | `operator_full_inventory` 8 `attention_contraction batch_matmul` self_attn rows (+1 rotary) → **n_attention_ops=9** | **present_and_preserved** (caught as batch_matmul — see attention-recovery check) |
| softmax | `nn.functional.softmax(...,dtype=fp32)` (:826) | `aten.softmax.int` ×4 (one/layer) | 44 `prov.op="softmax"`, 11 per self_attn fqn | `work_coverage n_softmax=12` | **present_and_preserved** |
| KV cache | `past_key_value.update(...)` Cache/DynamicCache (:934-936; :12) | NOT traced — `use_cache=False`, `past_key_values=None` | no cache tensors; mask is static `[8,9]` (target_len=seq+1) | not recovered; `decode_kv_cache`/`decode_loop_controller` boundary marks KV `unavailable` | **present_in_source_erased_by_export** (capture forces single forward; KV structure absent) |
| conv / vision | NO conv anywhere; ViT patch embed is `nn.Linear` (:491); whole ViT excluded from capture | absent | absent | `work_coverage n_conv=0` | **not_present_in_source** (no conv exists; vision backbone also out of capture scope) |
| linear / GEMM | `att_proj`, `attn_out`, `mlp.ff_proj`, `mlp.ff_out`, `lm_head` (nn.Linear) | `aten.linear` ×17 | `addmm`/`matmul` contraction regions | `work_coverage n_linear_matmul=17`, `linear_gemm_macs=7.57e9` | **present_and_preserved** |
| dtype | fp32 capture (`.to(torch.float32)` in loader); softmax forced fp32 (:826) | `f32[...]` throughout; `aten.to.dtype` | `f32` tensors; `prov.orig_dtype="float32"` | `operator_shape_table` dtype col `f32` | **present_and_preserved** (single fp32 datapath; real model is fp32/bf16 — low-bit not in this capture) |
| accumulator | implicit fp32 (eager torch) | implicit (f32 matmul outs) | linalg.generic `outs(... f32)` accumulation | not a distinct Merlin field | **present_in_source_erased_by_lowering** (acc dtype implied by f32 result tensor, no explicit acc annotation) |
| quant | none in source/capture (fp32) | absent | absent | none in CSVs; numerical_contract treats low-bit as `blocked_by_missing_accuracy` | **not_present_in_source** (for this capture; W8A8 etc. are DSE candidates, not in the graph) |
| packed layout | none (dense fp32) | absent | absent | absent (boundary `packed_low_bit` = erased/unavailable) | **not_present_in_source** |
| scales | none (no quant) | absent | absent | absent | **not_present_in_source** |
| region attribution | module paths via `nn.Module` hierarchy | param-name prefixes | `prov.fqn`/`prov.region_id`/`prov.module` per op | `module`, `region_fqn`, `region_role=repeated_head` cols populated | **present_and_preserved** |
| tied weights | `_tied_weights_keys = []` — explicitly NOT tied (:1494,1792); lm_head ≠ wte | wte (`embedding`+`new_embedding` cat) and `lm_head` are separate params | separate `embedding` op and `lm_head` matmul | separate inventory rows; no tie metadata | **not_present_in_source** (model declares weights untied; correctly nothing to recover) |
| repeated-weight lifetime | same `blocks[i]` weights reused every decode token across K=8 (host loop) | not represented (single pass) | not represented | `region_role="repeated_head"` flags the per-layer regions as repeated heads | **present_in_source_erased_by_export** (single-pass capture; reuse only inferable from `repeated_head` role + sidecar K) |
| loop-carried KV state | growing KV across decode steps (`past_key_value`) | absent (use_cache off) | absent | `decode_loop_controller` knob `kv_update_in_loop`, metadata "loop-carried token state" — declared but `unavailable`/`blocked` | **present_in_source_erased_by_export** (boundary catalog names it but cannot quantify from the flat capture) |

---

## molmoact-specific questions

**(a) Action: autoregressive token decode, not regression.**
`MolmoActForActionReasoning` has no regression action head — it inherits `GenerationMixin` and emits
text via `generate()`; the action vector is produced *host-side* by `parse_action` (modeling_molmoact.py:1936-1985):
it extracts bracketed action-token lists from the generated text, maps token ids → discretized bin
indices (`discretized = vocab_size - id`, then `bin_centers`), then un-normalizes with per-dataset
q01/q99 (`n_action_bins=256`, :1809-1812). So actions are **discretized tokens decoded autoregressively**,
then de-tokenized in numpy — none of the de-tokenize math is in the captured graph (it is downstream of
`logits`).

**(b) KV cache.** Present in source (`Cache`/`DynamicCache`, `past_key_value.update`, :934-936). The
capture is a **single prefill forward with `use_cache=False`** (loader forces `cfg.use_cache=False` and
calls `self.lm(input_ids=..., use_cache=False)`; the FX graph has no cache tensors and the causal mask is
the static prefill mask `f32[8,9]`). It is **not** one decode step and **not** a full `generate()` — it is
the prefill pass only.

**(c) H and cadence.** Neither `action_horizon` (H) nor `control_rate_hz` appears in the modeling source.
H=8 and 5.0 Hz exist **only** in `MODEL_ARCH["molmoact"]` (models.py:66), tagged as reference/`assumed`
values, mirroring the openvla VLA family. → sidecar/config-only.

**(d) ATTENTION RECOVERY CHECK (the known-gap probe).**
- `work_coverage_table.csv` → **molmoact `n_attention_ops = 9`** (and `attention_macs = 1,835,520`).
- In the flat MLIR, attention is **NOT** a fused `sdpa` op and carries **no** `prov.op="sdpa"` /
  `family="attention"` tag (grep for `sdpa` / `"attention"` returns nothing). Because the loader sets
  `_attn_implementation="eager"`, `eager_attention_forward` uses two explicit `torch.matmul` calls, which
  lower to **`linalg.generic` with `prov.op="batch_matmul"`, `prov.aten="aten.bmm.default"`,
  `family="contraction"`** — the pattern Merlin **does** catch.
- Count: 8 self-attn batch_matmuls (2 per layer × 4 layers — qkᵀ regions matmul_2/8/14/20 and attn·v
  regions matmul_3/9/15/21) tagged `...N.self_attn`, **plus 1 rotary/cos-sin bmm** (matmul_0,
  `inventory` idx17, no self_attn fqn) → the 9th attention_op. softmax (`n_softmax=12`) and `repeat_kv`
  expands are also present per layer.
- **Verdict: Merlin RECOVERS molmoact's attention.** This workload sits on the *good* side of the known
  sdpa-erasure gap precisely because the eager attention path keeps the two contractions as visible
  bmms. (Had the capture used `_attn_implementation="sdpa"`, attention would fuse to a single op Merlin's
  batch_matmul detector misses — but that is not this capture.) Caveat: the 9th "attention" op is a
  rotary-table bmm mis-bucketed into `attention_contraction`, so 8 of the 9 are true attention
  contractions.

**(e) decode_loop_controller / KV-blocked classification.** Sensible and honest.
`MODEL_ARCH` types molmoact as `autoregressive_vla` / `action_token_decode`, and the boundary catalog
(`boundary_placement.py:275`) defines `decode_loop_controller` with `region_roles=["repeated_head"]`,
`cp_axis="decode_kv_cache_path"`, knobs `decode_bound` + `kv_update_in_loop`, and risk
"needs bounded decode semantics + (unavailable) KV structure". The companion `decode_kv_cache` boundary
is marked `kv=True` with metadata "growing length (**unavailable — attention lowered**)". This matches
the capture: the decode loop and KV growth are *real in source* but *absent from the single-pass flat
capture*, so they are correctly declared-but-blocked rather than fabricated. The one nuance: KV is listed
as "unavailable — attention lowered", whereas for molmoact attention is actually *recovered* (bmm); KV is
unavailable because `use_cache=False` erased it at export, not because attention was lowered. The
classification outcome (blocked) is right; the stated *reason* is slightly off for this workload.

---

## FINDINGS

1. **Capture scope is the dominant erasure.** Only the 4-layer fp32 causal-LM **prefill** is exported;
   the full `MolmoActForActionReasoning`, the ViT/adapter, the `generate()` K-loop, and the numpy
   action de-tokenizer are all out of scope → many VLA-specific features are
   `present_in_source_erased_by_export`, not Merlin failures.
2. **Attention is recovered** (`n_attention_ops=9`, 8 true self-attn bmms + 1 rotary bmm) because the
   loader pins eager attention → `aten.bmm` → `prov.op="batch_matmul"`. molmoact is on the safe side of
   the sdpa-fusion gap.
3. **No conv, no quant, no tied weights, no packed/scale structure** exist in the source for this path —
   correctly absent everywhere (vision patch-embed is `nn.Linear`; `_tied_weights_keys=[]`).
4. **Dense GEMM coverage is essentially complete**: 17 linear/matmul ops, `visible_linear_fraction=0.9998`,
   fused QKV preserved as one GEMM + split, lm_head attributed.
5. **KV / decode-loop / horizon / cadence are sidecar-or-blocked**: K=8, H=8, 5 Hz live only in
   `MODEL_ARCH`; KV cache + loop-carried state are erased by `use_cache=False` and honestly flagged
   `blocked`/`unavailable` by the boundary catalog (with a minor mis-stated reason — "attention lowered"
   vs. the real cause "use_cache off").

```SUMMARY_ROWS
molmoact,model_class,present_in_source_erased_by_export,"full VLA class MolmoActForActionReasoning (modeling_molmoact.py:1790) never entered; capture wraps causal-LM MolmoActForCausalLM (:1493) as _LogitsOnly (loader.py)"
molmoact,submodule_boundary,present_and_preserved,"prov.fqn lm.model.blocks.slice(None,4,None).N.self_attn in recaptures/molmoact/model.mlir matches param names in exported_fx/molmoact.txt"
molmoact,forward_entry,present_and_preserved,"MolmoActForCausalLM.forward (:1528) -> single FX GraphModule.forward -> flat func"
molmoact,action_head,present_and_preserved,"lm_head nn.Linear(:1503) -> aten.linear -> matmul_25 fqn lm.lm_head (operator_shape_table row16); action de-tokenize in parse_action(:1936) is host numpy, erased_by_export"
molmoact,K_decode_loop,present_in_source_erased_by_export,"generate() K=8 not traced; single prefill forward; K only in MODEL_ARCH[molmoact].loop_count=8 (models.py:66)"
molmoact,action_horizon_H,sidecar_or_config_only,"H=8 only in MODEL_ARCH[molmoact].action_horizon; absent from source/FX/MLIR"
molmoact,cadence,sidecar_or_config_only,"control_rate_hz=5.0 only in MODEL_ARCH; absent from source/FX/MLIR"
molmoact,qkv_proj,present_and_preserved,"fused att_proj nn.Linear(:866) + split(:913) -> aten.linear f32[4608,3584] + split_with_sizes[3584,512,512] -> addmm + split in MLIR"
molmoact,qk_av_bmm,present_and_preserved,"eager torch.matmul (:821,829) -> aten.bmm -> 8 prov.op=batch_matmul self_attn regions (matmul_2/3,8/9,14/15,20/21)"
molmoact,softmax,present_and_preserved,"nn.functional.softmax fp32 (:826) -> aten.softmax.int x4 -> 44 prov.op=softmax in MLIR; work_coverage n_softmax=12"
molmoact,kv_cache,present_in_source_erased_by_export,"past_key_value.update Cache/DynamicCache (:934-936) but loader forces use_cache=False; no cache tensors in FX/MLIR; static prefill mask f32[8,9]"
molmoact,conv_vision,not_present_in_source,"no nn.Conv anywhere; ViT patch_embedding is nn.Linear(:491) and whole ViT excluded from capture; work_coverage n_conv=0"
molmoact,linear_gemm,present_and_preserved,"17 nn.Linear -> aten.linear x17 -> addmm/matmul; work_coverage n_linear_matmul=17 linear_gemm_macs=7.57e9 visible_linear_fraction=0.9998"
molmoact,dtype,present_and_preserved,"fp32 capture (loader .to(torch.float32)); f32 throughout FX/MLIR; operator_shape_table dtype=f32; low-bit not in this capture"
molmoact,accumulator,present_in_source_erased_by_lowering,"implicit fp32 acc; linalg.generic outs f32 with no explicit acc-dtype annotation"
molmoact,quant,not_present_in_source,"fp32 datapath; no quant in source/FX/MLIR; W8A8 only a DSE candidate (numerical_contract blocked_by_missing_accuracy)"
molmoact,packed_layout,not_present_in_source,"dense fp32; no packed structure in source/FX/MLIR"
molmoact,scales,not_present_in_source,"no quant -> no scales anywhere"
molmoact,region_attribution,present_and_preserved,"prov.fqn/prov.region_id/prov.module per op -> module/region_fqn/region_role=repeated_head cols in CSVs"
molmoact,tied_weights,not_present_in_source,"_tied_weights_keys=[] (:1494,1792); wte and lm_head separate params; nothing to recover"
molmoact,repeated_weight_lifetime,present_in_source_erased_by_export,"blocks[i] reused across K=8 host loop; single-pass capture; only inferable from region_role=repeated_head + sidecar K"
molmoact,loop_carried_kv_state,present_in_source_erased_by_export,"growing KV across decode; absent (use_cache off); decode_loop_controller boundary declares it but unavailable/blocked"
molmoact,attention_recovery,present_and_preserved,"RECOVERED: n_attention_ops=9 (work_coverage); MLIR attention is prov.op=batch_matmul/aten.bmm (eager), NOT sdpa/family=attention; 8 true self-attn bmms + 1 rotary bmm"
```
