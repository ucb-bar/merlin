# small_llama — source → exported-FX → flat-MLIR → Merlin forensic audit

**Workload:** small_llama — a tiny, *hand-rolled* LLaMA-style decoder defined **inline**
in the m2m loader (not an upstream repo / not HF `transformers.LlamaModel`). Its docstring
states it is "a small but COMPLETE LLaMA-style transformer for fast end-to-end RVV/spike
verification … Same op surface as a real LLaMA (RMSNorm, RoPE, causal attention + softmax,
SwiGLU MLP, tied-free lm_head) at tiny dims … The Merlin pipeline that runs this is identical
to the one that runs tiny_llama-1.1B / smolVLA; only the dims differ."
(`/path/to/model2MLIR/workloads/small_llama/loader.py:1-5`)

**Scope / disclaimers.** Source-grounded; every claim cites a file:line or artifact row.
Magnitudes are **structural-only** — the model is random-init with a toy config
(`vocab=256, d=128, h=4, layers=2, hidden=344`, loader.py:71) and a single fixed input
`ids = randint(0,256,(1,8))` (loader.py:84). MAC counts reflect *shape*, not a trained
checkpoint or a realistic sequence length. **No performance / latency claims** — no FireSim /
`measured_cycles` exists for small_llama (`models.py` ln entry carries none, lines 71-72);
`dse_readiness_summary.csv` shows `ready_quantitative_DSE=False`. "not visible" is marked as a
missing/erased status, never inferred.

**Sources audited**
- Source (INLINE): `/path/to/model2MLIR/workloads/small_llama/loader.py`
  — `class SmallLlama`:71, `forward`:78; submodules `RMSNorm`:13, `rope`:21, `class Attn`:38
  (`forward`:40), `class MLP`:51 (`forward`:57), `class Block`:60 (`forward`:65);
  entry `get_model_and_inputs`:82.
- m2m wrapper: **no `capture.toml` present** — the workload dir contains only `loader.py`
  (+ `__pycache__`); `rg` for `M2M_*` env returns nothing. So there is **no sidecar config**;
  all dims are literal defaults in `SmallLlama.__init__` (loader.py:71).
- Exported FX: `manual_validation/exported_fx/small_llama.txt` (OP HISTOGRAM at head; full ATen graph).
- Flat MLIR: `recaptures/small_llama/model.mlir` — flat `func.func @forward` (line 2),
  **no level/region MLIR file**, `linalg-on-tensors`.
- Merlin artifacts: `case_study/{operator_full_inventory.csv, work_coverage_table.csv,
  operator_shape_table.csv, shape_summary_by_workload.csv, dse_readiness_summary.csv}`;
  `case_study/small_llama/{region_attribution.yaml, state_lifetime.yaml}`;
  `models.py` ln['small_llama'] (lines 71-72).

---

## Architecture (from source, loader.py)

| Fact | Value | Cite |
|---|---|---|
| Implementation | hand-rolled (`nn.Module` subclasses); **NOT** HF `LlamaModel` | loader.py:1-5, 38-79 |
| n_layers | 2 (`layers=2`) | loader.py:71 |
| d_model | 128 (`d=128`) | loader.py:71 |
| n_heads | 4 (`h=4`) → head_dim 32 (`d//h`) | loader.py:39, 71 |
| MLP hidden | 344 (SwiGLU: g,u → silu(g)*u → dn) | loader.py:51-58, 71 |
| vocab | 256 | loader.py:71 |
| seq length T | **8** (input `(1,8)`) | loader.py:84 |
| batch B | **1** | loader.py:84 |
| input shape | `ids: i64[1,8]` | loader.py:84; FX:32 |
| RoPE | yes, hand-rolled `rope()` (cat/cos/sin/neg) | loader.py:21-35, 44 |
| RMSNorm | yes (pow→mean→rsqrt→mul) | loader.py:13-18 |
| causal mask | `full((T,T),-inf).triu(1)` then softmax | loader.py:46-47 |
| KV cache / use_cache | **absent** — no `past_key`, no cache arg, single forward over all T | loader.py:40-49, 78-81 |
| lm_head | `nn.Linear(d, vocab, bias=False)`, separate param | loader.py:74 |
| tied weights | **NO** — `emb` and `lm` are independent params (no `lm.weight = emb.weight`) | loader.py:73-74 |

---

## Source / Exported-FX / MLIR / Merlin — feature table

Status vocab: `present_and_preserved`, `present_in_source_erased_by_export`,
`present_in_source_erased_by_lowering`, `present_but_not_recovered_by_merlin`,
`not_present_in_source`, `unknown_source_not_found`, `sidecar_or_config_only`.

| Feature | Source (loader.py) | Exported-FX | Flat MLIR | Merlin artifact | Status |
|---|---|---|---|---|---|
| Model class | `SmallLlama` (l.71) | not a node; only param FQNs `p_blocks_*`,`p_emb_weight`,`p_lm_weight` (FX:32) | `func.func @forward` (l.2); `prov.module`/`prov.fqn` strings | `models.py` ln['small_llama'] family `llm`, loop_kind `token_decode` (l.71) | present_and_preserved |
| Submodule boundary | `RMSNorm/Attn/MLP/Block` (l.13-65) | erased to flat op list; recoverable only via `# File:` comments | `prov.fqn="blocks.0.attn"`, `"blocks.0.mlp"`, `"blocks.0.n1"` etc. | region_attribution.yaml role `repeated_head`, 15 matmuls by prov_fqn | present_and_preserved |
| Forward entry | `SmallLlama.forward(ids)` (l.78) | `forward(self, …params…, ids)` (FX:32) | `func.func @forward(...%21: tensor<1x8xi64>...)->tensor<1x8x256xf32>` (l.2) | single capture; invocations=32 (=K) annotated, not a 2nd func | present_and_preserved |
| lm head | `self.lm = Linear(128,256)` (l.74) | `linear_14 = aten.linear(…, p_lm_weight)` (FX:328) | `matmul_18` `ins(8x128, 128x256)->8x256`, `prov.fqn="lm"` (MLIR:1283) | operator_shape_table semantic_class `lm_head_projection`, M8 N256 K128 (op14) | present_and_preserved |
| K decode loop | not in source (single forward, no autoregress) | not in FX | not in MLIR (single pass) | **K=32 reference/assumed**: ln['small_llama'] loop_count=32 token_decode (models.py:71); dse_readiness K_source=`assumed_reference` | sidecar_or_config_only |
| H (action horizon) | n/a (LLM, not VLA) | n/a | n/a | ln action_horizon=`None` (models.py:71) | not_present_in_source |
| Cadence (control_rate_hz) | n/a | n/a | n/a | ln control_rate_hz=`None` (models.py:71) | not_present_in_source |
| q/k/v proj | `q/k/v=Linear(d,d,bias=False)` (l.41-43) | `linear`,`linear_1`,`linear_2` f32[1,8,128] | `matmul_0/1/2` `ins(8x128,128x128)` fqn `blocks.0.attn.{q,k,v}`, transposed_b (MLIR:79,93,107) | operator_shape_table semantic_class `attention_qkv_projection` M8 K128 N128 (ops 0-2,7-9) | present_and_preserved |
| o proj | `o=Linear(d,d)` (l.48 `self.o(out)`) | `linear_3` | `matmul_5` fqn `blocks.0.attn.o` | semantic_class `attention_output_projection` (op3,10) | present_and_preserved |
| qkᵀ bmm | `q @ k.transpose(-2,-1)` (l.45) | `matmul` f32[1,4,8,8] (FX) | `matmul_3` linalg.generic `prov.op="batch_matmul"`, `aten.bmm.default`, `ins(4x8x32,4x32x8)->4x8x8` (MLIR:366) | operator_full_inventory `attention_contraction`/`batch_matmul` M8 K32 N8 batch4 (gen op40,103) | present_and_preserved |
| attn·v bmm | `att @ v` (l.48) | `matmul_1` f32[1,4,8,32] | `matmul_4` `prov.op="batch_matmul"` `ins(4x8x8,4x8x32)->4x8x32` (MLIR:488) | `attention_contraction`/`batch_matmul` M8 K8 N32 batch4 (gen op53,116) | present_and_preserved |
| softmax | `.softmax(-1)` (l.47) | `aten.softmax.int` ×2 (histogram) | 3 `prov.op="softmax"` regions/layer (max/exp/normalize) fqn `blocks.0.attn` (MLIR:436-467) | work_coverage n_softmax=6 (3/layer ×2 layers) | present_and_preserved |
| KV cache | absent (l.40-49) | absent (no past_key/cache nodes) | absent | not modeled; state_lifetime lists only `weights` resident object | not_present_in_source |
| conv | absent | absent (no aten.conv) | absent | work_coverage n_conv=0 | not_present_in_source |
| linear / GEMM | 4 attn + 3 MLP + lm per source | 15× `aten.linear.default` (histogram) | 15× `linalg.matmul` (`aten.mm`, transposed_b) | work_coverage n_linear_matmul=15; visible_linear_fraction=0.9905 | present_and_preserved |
| dtype | f32 (random init, no quant in source) | `f32[…]` everywhere | `tensor<…xf32>`; `prov.orig_dtype="float32"` | operator_shape_table dtype `f32`; accuracy_status `pass` | present_and_preserved |
| accumulator | not specified in source (implicit f32) | not in FX | linalg matmul init `arith.constant 0.0:f32` (e.g. MLIR:355) | accumulator inferred f32 from IR (`recovered_from_ir`) | present_in_source_erased_by_lowering |
| quant | none in source | none | none (all f32) | accuracy_gate covers int8/fp8 *candidates* separately; base capture f32 | not_present_in_source |
| packed layout | none in source | none | row-major `tensor<>` only | no packed-layout attr; tile_waste uses logical shapes | not_present_in_source |
| scales | none (no quant) | none | none | n/a | not_present_in_source |
| region attribution | module tree (l.13-71) | FQN strings only | `prov.fqn`/`prov.region_id` per op | region_attribution.yaml: status `attributed`, level 1, role `repeated_head`, 15 matmuls by prov_fqn, conf 0.7 | present_and_preserved |
| tied weights (emb↔lm) | **NOT tied** — separate params (l.73-74) | two distinct params `p_emb_weight`,`p_lm_weight`, both f32[256,128] (FX:32) | distinct func args %0 (emb) and %20 (lm), both `tensor<256x128xf32>` (MLIR:2) | not flagged as tied (each its own state) | not_present_in_source |
| repeated-weight lifetime | weights reused across decode tokens (loader docstring) | n/a (single pass) | n/a | state_lifetime.yaml: `weights` resident_weight_object, loop_invariant, bytes 1712128 `recovered_from_ir`, reused_times=32 (=K, prov_fqn) | present_and_preserved |
| loop-carried KV state | none (no cache) | none | none | state_lifetime lists only weights; no KV state object | not_present_in_source |

---

## small_llama-specific questions

**(a) Prefill / decode / generic? M source? Representative?**
The capture is a **single generic forward / prefill over all 8 tokens at once** — source runs the
whole T=8 sequence through `forward(ids)` (loader.py:78-81) with no autoregressive loop and no
cache. **M = T = 8, B = 1** (input `(1,8)`, loader.py:84), which is exactly the leading dim of
every recovered GEMM: all 15 `linalg.matmul` have **M=8** (operator_shape_table, MLIR ins `8x128`).
The toy shape (`vocab=256, d=128, h=4, hidden=344`) is **artificial** — random init, chosen "so a
spike functional-sim run finishes in seconds" (loader.py:2-4); it is shape-faithful to a LLaMA op
surface but not magnitude-representative of tiny_llama-1.1B.

**(b) KV cache + decode loop erased?**
They are **never in the source** — `use_cache` is off because there is no cache path at all; `forward`
takes only `ids` and recomputes attention over the full T each call (loader.py:40-49, 78-81). So
nothing was "erased": FX and MLIR faithfully contain a single prefill pass. The *decode loop* exists
only as a **reference/assumed K=32** in `models.py` ln['small_llama'] (loop_kind `token_decode`,
loop_count 32; dse_readiness K_source=`assumed_reference`) — a sidecar annotation, not a captured loop.

**(c) q/k/v/o + MLP projections recovered with correct shapes?**
Yes, all 7 per-block projections per layer are recovered with exact shapes and FQNs:
q/k/v/o = M8 K128 N128; MLP g/u = M8 K128 N344; MLP dn = M8 K344 N128 (operator_shape_table ops 0-13;
MLIR matmul_0..matmul_17). lm = M8 K128 N256. semantic_class is correctly assigned per role
(attention_qkv_projection / attention_output_projection / mlp_projection / lm_head_projection).

**(d) dominant_shape_class meaning (GEMV vs skinny projection)?**
Every matmul is classified **`wide_skinny`** (shape_summary_by_workload: small_llama, wide_skinny,
op_count 15, mac_fraction 1.0). Because **M=8** (not 1), these are **skinny GEMMs**, not true GEMV:
aspect_ratio_MN ≈ 0.0625 and `is_tail_heavy=True` for all (operator_shape_table). The wide_skinny
class is driven entirely by the tiny prefill M=8; under the assumed K=32 token-decode regime a real
deployment decode step would have M=1 (GEMV), so this classification is an **artifact of the M=8
prefill capture**, not an intrinsic GEMV.

**(e) ATTENTION RECOVERY CHECK (known gap) — RECOVERED.**
work_coverage_table.csv: **n_attention_ops=4**, attention_macs=32768. In operator_full_inventory the
two attention contractions per layer are `op_class=attention_contraction`, **`prov_op=batch_matmul`**
(gen ops 40,53 layer0; 103,116 layer1) — qkᵀ (M8 K32 N8 batch4) and att·v (M8 K8 N32 batch4). The
flat MLIR confirms `prov.op="batch_matmul"`, `prov.aten="aten.bmm.default"` (MLIR:366, 488, 922…).
**Attention is recovered as batch_matmul, NOT missed as an opaque sdpa/attention op.** This is the
*good* case: the hand-rolled attention decomposes to explicit bmm+softmax in FX (no `scaled_dot_product_
attention` fused op exists in the graph — `rg sdpa|scaled_dot` finds nothing), so the pipeline sees
real contractions instead of a black-box SDPA.

**(f) Is embed tied to lm_head?**
**No.** Source declares `self.emb=nn.Embedding(vocab,d)` and `self.lm=nn.Linear(d,vocab,bias=False)`
with no weight assignment between them (loader.py:73-74). FX exposes two independent params
`p_emb_weight` and `p_lm_weight`, both f32[256,128] (FX:32); MLIR carries them as separate func args
%0 and %20. They are the same *shape* (256×128) but distinct tensors — "tied-free lm_head" as the
loader docstring states (loader.py:4).

---

## FINDINGS

1. **Faithful, fully-recovered capture.** Hand-rolled (non-HF) 2-layer LLaMA; all 15 linear GEMMs +
   4 attention bmm + 6 softmax recovered with correct shapes/FQNs. visible_linear_fraction=0.9905,
   role attribution `attributed` (level-1, prov_fqn), accuracy_status `pass`. No information was lost
   at lowering beyond the usual flattening of module boundaries (preserved via prov.fqn).
2. **Attention is the success case, not the known gap.** Because the source spells out qkᵀ/att·v as
   explicit `@`/bmm (no torch SDPA), FX emits `aten.bmm` and Merlin recovers
   `attention_contraction=batch_matmul` (n_attention_ops=4). Contrast with models whose attention
   collapses to an opaque sdpa op.
3. **M=8 is the prefill source.** All M=8 = seq length T (B=1). The `wide_skinny` shape class and
   tail-heavy flags are artifacts of the tiny prefill, not a real decode GEMV (which would be M=1).
4. **K=32 decode loop and family=token_decode are sidecar-only** (models.py reference, `assumed`),
   not in the captured IR; `ready_quantitative_DSE=False`, no measured cycles.
5. **No KV cache, no conv, no quant, no packed layout, no tied weights** — all genuinely absent from
   source (not erased). emb and lm are distinct same-shaped params.

```SUMMARY_ROWS
small_llama,model_class,present_and_preserved,SmallLlama loader.py:71; func.func @forward MLIR:2; ln['small_llama'] models.py:71
small_llama,submodule_boundary,present_and_preserved,RMSNorm/Attn/MLP/Block loader.py:13-65; prov.fqn blocks.0.attn/mlp/n1 in MLIR
small_llama,forward_entry,present_and_preserved,forward(ids) loader.py:78; func.func @forward(...1x8xi64)->1x8x256 MLIR:2
small_llama,lm_head,present_and_preserved,Linear(128,256) loader.py:74; matmul_18 ins(8x128,128x256) MLIR:1283; lm_head_projection op14
small_llama,K_decode_loop,sidecar_or_config_only,no loop in source/FX/MLIR; ln loop_count=32 token_decode models.py:71 K_source=assumed_reference
small_llama,H_action_horizon,not_present_in_source,LLM not VLA; ln action_horizon=None models.py:71
small_llama,cadence,not_present_in_source,ln control_rate_hz=None models.py:71
small_llama,qkv_proj,present_and_preserved,q/k/v Linear(d,d) loader.py:41-43; matmul_0/1/2 ins(8x128,128x128) MLIR:79-107; attention_qkv_projection
small_llama,qkT_and_attnv_bmm,present_and_preserved,q@kT/att@v loader.py:45,48; matmul_3/4 prov.op=batch_matmul aten.bmm MLIR:366,488; attention_contraction
small_llama,softmax,present_and_preserved,.softmax(-1) loader.py:47; 3 softmax regions/layer MLIR:436-467; n_softmax=6
small_llama,kv_cache,not_present_in_source,no past_key/use_cache loader.py:40-49,78; absent in FX/MLIR; state_lifetime only weights
small_llama,conv,not_present_in_source,no aten.conv in FX; work_coverage n_conv=0
small_llama,linear_gemm,present_and_preserved,15 aten.linear -> 15 linalg.matmul; work_coverage n_linear_matmul=15 visible_linear_fraction=0.9905
small_llama,dtype,present_and_preserved,f32 random init; tensor<...xf32> prov.orig_dtype=float32; operator_shape_table dtype f32
small_llama,accumulator,present_in_source_erased_by_lowering,implicit f32 in source; matmul init arith.constant 0.0:f32 MLIR:355 recovered_from_ir
small_llama,quant,not_present_in_source,no quant in source/FX/MLIR (all f32); int8/fp8 only as separate accuracy_gate candidates
small_llama,packed_layout,not_present_in_source,row-major tensor only; no packed-layout attr
small_llama,scales,not_present_in_source,no quant so no scales
small_llama,region_attribution,present_and_preserved,prov.fqn per op; region_attribution.yaml status=attributed level1 role=repeated_head 15 matmuls conf0.7
small_llama,tied_weights,not_present_in_source,emb and lm separate params loader.py:73-74; distinct f32[256,128] args %0,%20 MLIR:2 (NOT tied)
small_llama,repeated_weight_lifetime,present_and_preserved,weights reused per token loader docstring; state_lifetime resident_weight_object loop_invariant bytes1712128 reused_times32
small_llama,loop_carried_kv_state,not_present_in_source,no cache; state_lifetime lists only weights, no KV state object
```
