# tiny_llama — source→FX→MLIR→Merlin forensic audit

**Workload:** `tiny_llama` (TinyLlama-1.1B-Chat-v1.0). The only real published checkpoint in
the DSE corpus — **but see the critical M=4 caveat below: the capture used to generate the
Merlin artifacts is a tiny PREFILL, not a decode step, and (under `M2M_LLAMA_LAYERS=2`) the
weights may be random-init, not the real checkpoint.** Structure and provenance are real;
magnitudes are capture-shape-dependent.

## Sources consulted (all paths absolute)
- Loader: `/path/to/model2MLIR/workloads/tiny_llama/loader.py`
- Capture config: `/path/to/model2MLIR/workloads/tiny_llama/capture.toml`
- Exported FX: `/path/to/merlin/merlin/benchmarks/dse_guidance/case_study/manual_validation/exported_fx/tiny_llama.txt`
- Flat MLIR: `/path/to/merlin/merlin/benchmarks/dse_guidance/recaptures/tiny_llama/model.mlir`
- Merlin: `.../case_study/{operator_full_inventory.csv, operator_shape_table.csv, work_coverage_table.csv, tile_waste_table.csv, accumulator_contract_table.csv, resident_state_table.csv, reuse_lifetime_table.csv, shape_summary_by_workload.csv, primitive_regret_table.csv}`
- Arch table: `/path/to/merlin/merlin/python/merlin/dse_guidance/models.py:73`

---

## The M=4 question (CRITICAL — resolved)

**True source of M=4: it is the capture SEQUENCE LENGTH (a small PREFILL), set by the
`M2M_SEQ` env var, NOT batch and NOT token-decode M=1.**

Evidence chain:
- `loader.py` builds the input as `input_ids = torch.randint(0, vocab, (1, seq))` with
  `seq = int(os.environ.get("M2M_SEQ", "8"))`. So batch is fixed at **1**; M is the sequence
  length. (loader.py lines: `seq = int(... "M2M_SEQ", "8")`, `input_ids = torch.randint(..., (1, seq), ...)`).
- The **exported FX dump** was produced with the default `seq=8`: every activation is
  `f32[1, 8, …]` and `input_ids: i64[1, 8]` (exported_fx/tiny_llama.txt graph signature + e.g.
  `embedding: "f32[1, 8, 2048]"`).
- The **flat MLIR recapture** used `M2M_SEQ=4`: the dominant activation tensor is
  `tensor<1x4x2048xf32>` (111×), with `tensor<4x2048xf32>`, `tensor<4x32000xf32>`,
  `tensor<1x4x5632xf32>` (model.mlir shape histogram). So M=4 here.
- **Every Merlin matmul row carries `M=4`** (operator_full_inventory.csv: all 15 matmuls
  `…,4,2048,2048,1,…`; operator_shape_table.csv: M=4 for all). The Merlin artifacts are
  therefore consistent with the **4-token** MLIR recapture, not the 8-token FX dump.

So **M=4 is a real, but artificially small, prefill sequence length** — neither a batch, nor
an "artificial capture batch", nor a true single-token decode (which would be M=1). The two
input dumps disagree (8 vs 4), evidencing the recaptures were taken at a different `M2M_SEQ`
than the FX export.

**This is true PREFILL, not decode.** `loader.py` forces `use_cache=False` (both `from_config`
and `from_pretrained` paths, plus the `_LogitsOnly.forward` passes `use_cache=False`). One
forward, no KV growth, no `past_key_values` input — confirmed by the FX signature having no
`past_key_*` argument and no KV-cache `cat` of prior steps. Yet `models.py` declares
`tiny_llama` as `("llm","token_decode", 32, …)` — a **decode** family with K=32. The K=32 is a
**MODEL_ARCH-declared decode loop count, not observed in the capture**; the capture itself is a
single 4-token prefill.

### Is the GEMV-like headline driven by tiny_llama's M=4?
Partly the cause, but not unique to it. With M=4, every projection is `4×K×N` with M≪N,K, so
the shape-classifier labels all 15 matmuls `gemv_like` and `gemv_lane_64` gives
`tile_utilization=1.0, padding_waste=0.0` while `tile_8x8`…`tile_32x32` waste 50–87%
(tile_waste_table.csv). shape_summary_by_workload.csv: tiny_llama is **100% gemv_like
(mac_fraction 1.0)**. **However**, the cross-workload `gemv_lane_64` headline
(primitive_regret_table.csv) is driven by **7 workloads** (`bitvla; groot_n1d7; molmoact;
openvla; rdt2; tiny_llama; xr0`) over 1051 ops and 2.29e12 MACs — tiny_llama is one
contributor, not the sole driver. The small-M effect is real and shared by all the
small-sequence captures, so the GEMV conclusion is **capture-shape-induced**, which is a
weakness for tiny_llama specifically (a real decode would be M=1 → even more GEMV-like, so the
*direction* is right; the *magnitude/waste numbers* are artifacts of M=4 + tiny instance).

---

## 4-column status table

| Feature | Source (HF/loader) | Exported FX | Flat MLIR | Merlin artifacts | Status |
|---|---|---|---|---|---|
| Model class | `LlamaForCausalLM` wrapped in `_LogitsOnly` | `GraphModule` (flattened) | flat func | `llm` family (models.py:73) | present_in_source_erased_by_export (class name gone; family re-declared) |
| Submodule boundary (decoder layer) | `LlamaDecoderLayer` ×N | gone, but `prov_fqn` retains `layers.0/1` | retained in attrs | region `repeated_head`, fqn `lm.model.layers.{0,1}` | present_and_preserved (via prov_fqn) |
| Forward entry | `_LogitsOnly.forward(input_ids)` | graph root | func args | n/a | present_in_source_erased_by_export |
| LM head | separate `nn.Linear` (untied) | `p_lm_lm_head_weight: f32[32000,2048]` | `tensor<2048x32000xf32>` | inventory op 14 `lm.lm_head` M=4,K=2048,N=32000 | present_and_preserved |
| K decode loop | NOT in capture (use_cache=False, 1 fwd) | absent | absent | declared K=32 (models.py); resident_state `reused_times=32` | present_in_source_erased_by_lowering (declared, not captured) |
| H (hidden) = 2048 | config | `f32[1,8,2048]` | `1x4x2048` | M/K/N=2048 in matmuls | present_and_preserved |
| Cadence (per-token) | decode cadence | single prefill | single prefill | "xK=32" annotation only (case_study.md) | sidecar_or_config_only |
| q/k/v proj | `q/k/v_proj` (GQA: 32 q, 4 kv heads) | linear→`[2048]`,`[256]`,`[256]` | `2048x2048`, `2048x256` | ops 0–2 / 7–9 fqn `self_attn.{q,k,v}_proj`, N=2048/256/256 | present_and_preserved |
| qkᵀ & attn·v bmm | inside fused SDPA | `aten.scaled_dot_product_attention` (fused) | decomposed | inventory ops 46 (M4,N64,K4,b32) & 55 (M4,N4,K64,b32) = qkᵀ/attn·v | present_and_preserved (recovered as 2 batch_matmul) |
| Softmax | inside SDPA | inside SDPA (no explicit op) | decomposed | ops 50–52 / 89–91 `softmax` but M/N/K=0 (dims not recovered) | present_but_not_recovered_by_merlin (dims=0) |
| KV cache | none (use_cache=False) | none | none | family axis `decode_kv_cache_path`/`quantized_KV_cache` declared only | not_present_in_source (capture); sidecar declared |
| Conv | none in Llama | none | none | work_coverage `n_conv=0` | not_present_in_source |
| Linear/GEMM | F.linear ×15 | 15 `aten.linear` | 15 matmuls | 15 `linear_gemm` rows | present_and_preserved |
| Dtype | f32 (loader forces `dtype=torch.float32`) | `f32` | `f32` | `dtype=f32` everywhere | present_and_preserved |
| Accumulator | implicit f32 | implicit | implicit | accumulator_contract: `accumulator_dtype=f32, committed_directly` | present_and_preserved (inferred) |
| Quant | none in this recapture (int8 only in separate `output/tiny_llama_int8_*`) | none | none | dtype_capacity has only `f32` row; scales `unavailable` | not_present_in_source |
| Packed layout | none (f32) | none | none | `packed_layout_preservation` is a declared family axis only | sidecar_or_config_only |
| Scales | none | none | none | accumulator_contract: `scale_dtype/granularity=unavailable` | not_present_in_source |
| Region attribution | module tree | `prov_fqn` strings | retained | every op tagged `lm.model.…` fqn + region `repeated_head` | present_and_preserved |
| Tied weights (embed/lm_head) | **explicitly UNtied** (`tie_word_embeddings=False`, head weight cloned) | two separate weights `embed_tokens` & `lm_head` | two separate tensors | embed (op gen 0) and lm_head (matmul 14) counted separately | present_and_preserved (untied by design) |
| Repeated-weight lifetime across decode | weights reused across K decode tokens (declared) | n/a (1 fwd) | n/a | reuse_lifetime: `repeated_head.weights, across_K, load_once_reuse_K`; resident_state `reused_times=32` | present_in_source_erased_by_lowering (declared via arch, not observed) |
| Loop-carried KV state | would exist in real decode | absent (use_cache=False) | absent | not modeled in capture | not_present_in_source |

---

## tiny_llama-specific answers
- **(a) M=4 source:** capture **sequence length** (prefill), set by `M2M_SEQ`. Batch=1 fixed.
  Not decode (decode would be M=1), not batch. FX dump used seq=8; MLIR/Merlin used seq=4 —
  the two dumps are from different `M2M_SEQ` runs.
- **(b) GEMV headline:** the M=4 small-M shape makes all 15 matmuls `gemv_like` and gives
  `gemv_lane_64` 100% utilization for tiny_llama, but the cross-workload `gemv_lane_64` result
  is shared across 7 workloads; tiny_llama is a clean example, not the sole driver.
- **(c) Decode vs prefill:** PREFILL. `use_cache=False` forced in three places; no
  `past_key_values`, no KV growth, single forward.
- **(d) KV/QKV blocking:** q/k/v/o + gate/up/down all recovered with correct shapes (GQA: q/o
  N=2048, k/v N=256, gate/up N=5632, down K=5632). qkᵀ and attn·v recovered as two
  batch_matmul (batch=32 heads, head_dim=64). **KV-cache axes are NOT blocked** because they
  do not exist in this capture (use_cache=False) — only declared as family axes.
- **(e) embed↔lm_head tying:** **NOT tied** — loader sets `tie_word_embeddings=False` and clones
  the head weight; FX/MLIR carry two distinct `f32[32000,2048]` weights. Merlin counts embed
  (gen op, 0 MACs) and lm_head (matmul, 262M MACs) **separately, correctly — no double-count**.
  (Note: the *real* TinyLlama-1.1B-Chat checkpoint ties them; this capture deliberately unties.)

---

## FINDINGS

**Right**
- Full Llama decoder structure recovered with correct GQA shapes, per-op `prov_fqn`,
  RMSNorm/SiLU/RoPE elementwise ops, and both attention contractions as batch_matmul.
- dtype f32, accumulator f32, region attribution, and embed/lm_head separation are all faithful.
- gemv_like classification and gemv_lane_64 zero-waste utilization are arithmetically correct
  for the captured shapes.

**Wrong / misleading**
- The capture is a **4-token prefill** while the arch table labels it `token_decode` with K=32;
  the "0.6 GMAC/step ×K=32", `reused_times=32`, and KV-cache family axes are **declared, not
  observed**. Anyone reading the Merlin tables as a decode profile is reading a prefill.
- **Two input shapes disagree:** FX export = seq 8, MLIR/Merlin = seq 4. The FX dump is not the
  capture the artifacts were built from — a provenance hazard for this audit trail.
- Under `M2M_LLAMA_LAYERS=2` (capture.toml), the model is built via `from_config` with
  **random init** (not the real checkpoint) and only 2 of 22 layers — so "the only real
  checkpoint" caveat: real *architecture*, but weights here are likely random and truncated.

**Weak**
- softmax ops present but with M/N/K=0 (dims not recovered) — attention softmax cost invisible.
- All magnitudes (attention_macs=131200, etc.) are tiny-instance artifacts of M=4 + 2 layers.
- No quant/scales/packed-layout in this f32 recapture; those rows are config-only declarations.

```text
SUMMARY_ROWS
tiny_llama,m_dim_source,present_and_preserved,M=4 is capture seq length (prefill) via M2M_SEQ; batch=1; loader.py input_ids=(1,seq)
tiny_llama,decode_vs_prefill,present_in_source_erased_by_lowering,use_cache=False single 4-token prefill but models.py declares token_decode K=32
tiny_llama,fx_vs_mlir_seq_mismatch,present_but_not_recovered_by_merlin,FX dump seq=8 vs MLIR/Merlin seq=4 (different M2M_SEQ runs)
tiny_llama,model_class,present_in_source_erased_by_export,LlamaForCausalLM->flat graph; family re-declared as llm/token_decode
tiny_llama,decoder_layer_boundary,present_and_preserved,prov_fqn lm.model.layers.0/1 + region repeated_head
tiny_llama,lm_head,present_and_preserved,inventory op14 lm.lm_head M4 K2048 N32000
tiny_llama,k_decode_loop,present_in_source_erased_by_lowering,K=32 declared models.py:73; reused_times=32 resident_state; not in capture
tiny_llama,hidden_2048,present_and_preserved,tensor<1x4x2048xf32>; matmul K/N=2048
tiny_llama,qkv_proj_gqa,present_and_preserved,q/o N=2048,k/v N=256 (32 q / 4 kv heads), head_dim=64
tiny_llama,qk_av_bmm,present_and_preserved,inventory ops46/55 batch_matmul batch=32 (qk^T M4N64K4, av M4N4K64)
tiny_llama,softmax,present_but_not_recovered_by_merlin,softmax ops50-52/89-91 with M/N/K=0
tiny_llama,kv_cache,not_present_in_source,use_cache=False; no past_key_values; only declared family axis
tiny_llama,conv,not_present_in_source,work_coverage n_conv=0
tiny_llama,linear_gemm,present_and_preserved,15 linear matmuls recovered
tiny_llama,dtype_f32,present_and_preserved,f32 throughout FX/MLIR; accumulator_contract f32
tiny_llama,accumulator,present_and_preserved,accumulator_dtype=f32 committed_directly (inferred)
tiny_llama,quant,not_present_in_source,f32 recapture; int8 only in separate output/ dirs
tiny_llama,packed_layout,sidecar_or_config_only,packed_layout_preservation only a declared family axis
tiny_llama,scales,not_present_in_source,accumulator_contract scale_dtype/granularity=unavailable
tiny_llama,region_attribution,present_and_preserved,every op tagged lm.model.* prov_fqn
tiny_llama,tied_weights,present_and_preserved,tie_word_embeddings=False; embed & lm_head separate f32[32000,2048]; not double-counted
tiny_llama,repeated_weight_lifetime,present_in_source_erased_by_lowering,reuse_lifetime across_K load_once_reuse_K declared; single-fwd capture
tiny_llama,loop_carried_kv_state,not_present_in_source,no KV state in use_cache=False capture
tiny_llama,gemv_like_classification,present_and_preserved,M=4 small-M -> 100% gemv_like; gemv_lane_64 util=1.0 waste=0 (shared across 7 workloads)
```
