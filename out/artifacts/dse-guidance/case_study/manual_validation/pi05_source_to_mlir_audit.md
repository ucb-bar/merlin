# pi05 — Source → FX → MLIR → Merlin Artifact Forensic Audit

Workload: **pi05** (OpenPI π0.5, flow-matching VLA: PaliGemma `gemma_2b` backbone + `gemma_300m` action expert, `pi05=True`).
Method: source-grounded. Every claim cites `file:line`, an FX-histogram line, an MLIR op count, or an artifact row. Not-visible ⇒ marked missing. **No perf claims.** All magnitudes are structural only (random init, no checkpoint; tiny single-batch config).

## Provenance of the inputs
- **Source model.** `/path/to/openpi/src/openpi/models_pytorch/pi0_pytorch.py`: `sample_actions` Euler loop `while time >= -dt/2` (lines 407–420) **calls** `denoise_step` (def at line 422). `denoise_step` runs `embed_suffix` → `paligemma_with_expert.forward(...)` (eager attn) → `action_out_proj` (lines 450–372/463).
- **Configs.** `pi0_config.py:23` `action_horizon=50` (**H=50**); `pi0_config.py:26` `action_dim=32`; `pi0_config.py:30` `pi05=True`; `max_token_len=200` for pi05 (`pi0_config.py:38`). Gemma variants `models/gemma.py:69-77` `gemma_300m` (width=1024, **depth=18**, mlp=4096, heads=8, kv=1, head_dim=256) and `:78-86` `gemma_2b` (width=2048, **depth=18**, mlp=16384). Both experts depth=18 (`gemma.py:352` asserts equal depth).
- **m2m capture unit.** `/path/to/model2MLIR/workloads/pi05/loader.py`: `Pi05DenoiseStep.forward` runs `embed_prefix` + a **`use_cache=True` prefix pass** (paligemma 2b backbone, builds `past_key_values`) **then exactly ONE `denoise_step`** (action expert pass). Docstring: "a single flow-matching `denoise_step` … without the diffusion while-loop." **No layer cap** — full depth=18 is exported for both experts. `capture.toml` only pins venv/python/upstream.
- **Exported FX.** `…/manual_validation/exported_fx/pi05.txt` OP HISTOGRAM (lines 1–45).
- **Flat MLIR** `recaptures/pi05/model.mlir` (48 338 lines); **levels** `recaptures_levels/pi05/model_{highlevel,qdq}.mlir`.
- **Merlin artifacts** under `case_study/`: `operator_full_inventory.csv` (5379 pi05 rows), `operator_shape_table.csv` (777 pi05 rows), `work_coverage_table.csv` (1 pi05 row), `models.py:53` `MODEL_ARCH['pi05']`.

## 4-column matrix (Source / Exported-FX / MLIR / Merlin artifact)

| Feature | Source | Exported-FX | MLIR (flat / levels) | Merlin artifact | Status |
|---|---|---|---|---|---|
| model class | `PI0Pytorch` + `Pi05DenoiseStep` wrapper (loader.py:23) | single `GraphModule` (pi05.txt:47) | single `func.func` (highlevel:1 func.func) | `MODEL_ARCH['pi05']` family `flow_matching` (models.py:53) | present_and_preserved |
| submodule boundary | vision_tower / paligemma LM / gemma_expert / proj heads (pi0_pytorch.py:100-131) | flattened, only `prov_fqn`-like names survive in node meta | erased to one flat region | recovered via `prov_fqn` strings, e.g. `…vision_tower…`, `…language_model.layers.slice(None,18,None)…` (shape_table rows) | present_in_source_erased_by_lowering |
| forward entry | `denoise_step` (pi0_pytorch.py:422) | top-level graph | single `func.func` | one `capture_id=pi05` (inventory/shape_table) | present_and_preserved |
| denoise / action head | `action_out_proj` linear (pi0_pytorch.py:101,372) | in 782 `aten.linear` (pi05.txt:2) | a `(50,32,1024)` + `(50,1024,32)` matmul pair (shape_table) | shape `M=50,N=1024,K=32` count 1 + `50,32,1024` count 1 (shape table tail) | present_and_preserved |
| K denoise loop | host `while` Euler loop (pi0_pytorch.py:407-420) | **NOT exported** (loader runs one step) | absent (flat single pass) | `loop_kind=denoise_steps, loop_count=10` tagged **assumed** (models.py:53, header L19-21) | sidecar_or_config_only |
| H (action_horizon) | `=50` (pi0_config.py:23) | seq dims 50 in action-expert nodes | matmul M/N=50 rows (shape_table) | `action_horizon=50` (models.py:53); H=50 appears as M in proj shapes | present_and_preserved |
| cadence (control_rate) | not in model code (deployment fact) | n/a | n/a | `control_rate_hz=50.0` tagged reference (models.py:53) | sidecar_or_config_only |
| q/k/v proj | `q_proj/k_proj/v_proj` (HF gemma) / `qkv_einsum` (gemma.py:177) | `aten.linear` (pi05.txt:2) | `linalg.matmul` (`968,2048,2048` q; `968,256,2048` k,v) | `attention_qkv_projection` semantic_class rows (shape_table) | present_and_preserved |
| qkᵀ & attn·v bmm | inside SDPA eager path (denoise_step:447) | `81 aten.scaled_dot_product_attention` (pi05.txt:16) **+** `72 aten.matmul` (pi05.txt:22) | **702 `linalg.batch_matmul`** (flat grep) — SDPA decomposed | `op_class=attention_contraction` 234 ops, 8 distinct shapes, 8.12e10 MACs (inventory) | present_in_source_erased_by_lowering |
| softmax | inside SDPA / eager attn | `81 SDPA` (fused) + `36 aten.softmax.int` (pi05.txt:24) | decomposed to `linalg.generic`/`linalg.reduce` (no softmax op; 2015 softmax/attn text hits are name fragments only) | `op_class=softmax` 348 ops (inventory) | present_in_source_erased_by_lowering |
| KV / prefix state | `past_key_values` from `use_cache=True` prefix pass (loader.py:45-50; denoise_step arg) | prefix pass + cache in graph (concat/slice on k,v) | concat/slice generics; no first-class KV-cache op | not a distinct artifact field; folded into backbone matmuls | present_in_source_erased_by_lowering |
| conv / vision | SigLIP patch-embed conv (vision_tower) | `3 aten.conv2d.padding` (pi05.txt:30) | conv lowered to generics; `op_class=conv` survives in inventory | `conv` 6 ops (inventory) | present_and_preserved |
| linear / GEMM | nn.Linear throughout | **782 `aten.linear`** (pi05.txt:2) | **777 `linalg.matmul`** (flat grep) | **777 `linear_gemm`** rows (shape_table & inventory) | present_and_preserved |
| dtype | `Pi0Config(dtype="float32")` in loader (loader.py:55); config default bf16 overridden | `337 aten.to.dtype` casts | f32 tensors | `dtype=f32` on every shape_table row | present_and_preserved |
| accumulator | implicit f32 (no separate decl) | not represented | implicit in `linalg.matmul` f32 | no separate accumulator field | present_in_source_erased_by_export |
| quant | none (fp32 capture; pi05 has no quant in source) | none | `model_qdq.mlir` has only **3 `arith.sitofp`**, no quant/dequant ops — degenerate qdq | no quant columns populated; `evidence_dtype=recovered_from_ir` f32 | not_present_in_source |
| packed layout | none in source | none | dense f32 only | none | not_present_in_source |
| scales | none (no quant) | none | none | none | not_present_in_source |
| region attribution | module tree (pi0_pytorch.py) | lost to flat graph | flat | `region_role` from `prov_fqn`: `backbone_once` 489 / `repeated_head` 288 linear rows (inventory) | present_but_not_recovered_by_merlin (partial — see (e)) |
| tied / repeated weights | depth=18 layers, weight reused per host denoise step | unrolled once (depth) but loop NOT unrolled | each weight emitted once (flat) | header note: "single-pass capture uses each weight once → 0 contract facts" (models.py:8-12) | present_in_source_erased_by_export |
| repeated-weight lifetime | reused across K=10 denoise steps | not visible (1 step captured) | not visible | re-exposed only by multi-rate view (models.py:11-12) | present_in_source_erased_by_export |
| loop-carried latent | `x_t = x_t + dt*v_t` (pi0_pytorch.py:418) | NOT in graph (one step) | absent | not modeled as artifact | present_in_source_erased_by_export |

## pi05-specific questions

**(a) Why 777 matmuls? Is the capture one denoise step?**
The capture is **definitively ONE denoise step**: `loader.py` runs `embed_prefix` + one prefix `forward(use_cache=True)` (backbone) + one `denoise_step` (action expert), and explicitly skips the `while` Euler loop (loader docstring; pi0_pytorch.py:407-420 not exercised). The 777 figure is **repeated transformer layers, not loop unrolling and not 777 distinct source ops**: FX shows 782 `aten.linear` (pi05.txt:2) → 777 `linalg.matmul` (flat MLIR) → 777 `linear_gemm` rows. With depth=18 in **both** experts plus 27 SigLIP vision layers, the linears are 18×(backbone per-layer linears) + 18×(action-expert per-layer linears) + vision-encoder linears + a handful of projection heads — i.e. **many instances of a few per-layer shapes** (see (b)/(c)). The K=10 denoise loop is NOT present.

**(b) Distinct (M,N,K) count — resolved: 17 (Merlin is correct, not a bug).**
Using a proper CSV parser on `operator_shape_table.csv` pi05 rows: **17** distinct `(M,N,K)`, and **17** under `(M,N,K,batch_product)`. Independently, `operator_full_inventory.csv` `op_class=linear_gemm` distinct `(M,K,N)` = **17** — the two artifacts agree exactly. So Merlin's `operator_pareto n_distinct_shapes=17` is **correct**.
The "prior source check counted 20" is a **CSV-parsing artifact**: 283 of the 777 pi05 shape-table rows have a `prov_fqn` like `"model.…language_model.layers.slice(None, 18, None).0.self_attn.q_proj"` — the `slice(None, 18, None)` contains **two embedded commas**, so the field is double-quoted. A naive `split(",")` mis-shifts the M/N/K columns for exactly those rows, injecting spurious tuples such as `('False','none','50')` and `('True','bias_addmm','1')`. Such a parse yields a corrupted count, not the true 17. The closest "20-ish" numbers come from *grouping that does not collapse transpose-pairs/semantics*: `(M,N,K,semantic_class)` = **19**, and **linear_gemm(17) + attention_contraction(8) = 25** distinct GEMM-family shapes if attention bmms are folded in. None of these is 20 under correct parsing. **True distinct linear-GEMM shape count = 17; Merlin's 17 is right and the prior 20 is a misparse.**

**(c) MAC dominance — pi05 is NOT genuinely diffuse; it is "many instances of few shapes".**
Top shapes (shape-table MAC sum 2.146e12):
- `(968,16384,2048)` ×34 → 51.5% (gemma_2b backbone MLP up-proj)
- `(968,2048,16384)` ×17 → +25.7% (cum 77.2%, backbone MLP down-proj)
- `(968,2048,2048)` ×34 → +6.4% (cum 83.6%, backbone attn proj)
- `(256,1152,1152)` ×324 + `(256,4304,1152)` ×81 + `(256,1152,4304)` ×81 → SigLIP vision (cum 98.3%)
**~3 distinct shapes cover ~84% of MACs; ~6 cover ~98%.** The high op *count* (777) is repetition of identical-shape layers (e.g. 324 instances of one vision shape), so apparent diffuseness is an artifact of repeated identical-shape layers, **not** shape diversity. A tile/codegen kernel for ~6 shapes covers essentially all the work.

**(d) Attention: lowered, not omitted — and recovered.**
FX shows attention present: `81 aten.scaled_dot_product_attention` + `36 aten.softmax.int` + `72 aten.matmul` (pi05.txt:16,24,22). Flat MLIR **decomposes SDPA**: 702 `linalg.batch_matmul` (qkᵀ and attn·v) with softmax as `linalg.generic`/`reduce` — no first-class attention/softmax op survives. Merlin **recovers** it: `attention_contraction` = 234 ops over **8 distinct shapes**, 8.12e10 MACs; `softmax` = 348 ops (inventory). `work_coverage_table` pi05: `n_attention_ops` non-zero, `attention_macs`=8.123e10, `total_recovered_macs`=2.2273e12, `visible_linear_fraction`=0.9635. So attention is lowered (erased as a named op) but its MACs are recovered.

**(e) Region roles — semantically weak / partly degenerate.**
Roles are derived from `prov_fqn`: `backbone_once` (489 linear, 162 attn) vs `repeated_head` (288 linear, 70 attn); 2 attn rows have an **empty role** (inventory). The labeling is **inverted relative to compute reality**: the heavy gemma_2b prefix MLPs (the 51%+26% MAC shapes, `968,16384,2048`) are tagged `backbone_once`, while `repeated_head` is the lighter gemma_300m action expert — yet under the *real* deployment the action expert (head) is what actually repeats K=10× per chunk and per control tick, and the prefix backbone runs once. So "repeated_head" names the right module but the `_once`/`repeated` semantics don't encode the K-loop multiplicity (that lives only in `MODEL_ARCH.loop_count`, tagged assumed). Roles are **structurally attributed but semantically weak**: correct module split, but the "repeated"/"once" suffix does not reflect the host-loop reuse, and 2 rows are unroled (blank).

## FINDINGS

**Right**
- Capture is one denoise step with the K-loop and loop-carried latent correctly excluded (loader + pi0_pytorch.py:407-422).
- 782 linear → 777 matmul → 777 `linear_gemm` chain is internally consistent across FX/MLIR/artifact.
- **Distinct linear-GEMM shape count = 17**, agreeing in both shape_table and inventory; Merlin's `n_distinct_shapes=17` is correct.
- Attention MACs lowered then recovered (234 attn ops / 8 shapes / 8.12e10 MACs; visible_linear_fraction 0.9635).
- MAC concentration: ~6 shapes ≈ 98% — workload is repetition-heavy, not shape-diverse.

**Wrong / misleading**
- The external "20 distinct shapes" is a **CSV misparse** of quoted `prov_fqn` fields (`slice(None, 18, None)` commas); not reproducible under proper parsing.
- "pi05 is diffuse" framing is an artifact of repeated identical-shape layers, not genuine diversity.

**Semantically weak**
- `region_role` `backbone_once`/`repeated_head` names modules correctly but its once/repeated semantics don't encode the K=10 host-loop reuse; 2 attn rows have blank role.
- K denoise loop, cadence, repeated-weight lifetime are **config/sidecar only** (assumed `loop_count=10`), absent from the captured flat graph.
- qdq level is degenerate for this fp32 capture (3 `arith.sitofp`, no real quant) — quant/scales/packed-layout `not_present_in_source`.

```SUMMARY_ROWS
pi05,capture_is_one_denoise_step,present_and_preserved,loader.py runs embed_prefix+1 prefix pass+1 denoise_step; while-loop pi0_pytorch.py:407-420 excluded
pi05,777_matmuls_are_repeated_layers,present_and_preserved,782 aten.linear(FX:2)->777 linalg.matmul(flat)->777 linear_gemm rows; depth=18 both experts gemma.py:73,82
pi05,distinct_shape_count_is_17,present_and_preserved,csv-parsed shape_table 17 (M,N,K); inventory linear_gemm 17 (M,K,N); Merlin n_distinct_shapes=17 correct
pi05,prior_20_is_csv_misparse,present_and_preserved,283/777 rows quote prov_fqn slice(None, 18, None); naive split shifts M/N/K cols -> spurious tuples
pi05,mac_concentration_not_diffuse,present_and_preserved,top3 shapes 83.6% top6 98.3% of 2.146e12 MACs; 324 instances of one vision shape (shape_table)
pi05,attention_lowered_then_recovered,present_in_source_erased_by_lowering,FX 81 SDPA(:16); flat 702 linalg.batch_matmul; inventory attention_contraction 234 ops/8 shapes/8.12e10 MACs
pi05,softmax_decomposed,present_in_source_erased_by_lowering,FX 36 softmax(:24); MLIR generic/reduce no softmax op; inventory softmax 348 ops
pi05,conv_vision_preserved,present_and_preserved,FX 3 conv2d(:30); inventory conv 6 ops (SigLIP patch embed)
pi05,K_denoise_loop,sidecar_or_config_only,MODEL_ARCH loop_count=10 tagged assumed (models.py:53); absent from flat single-pass graph
pi05,region_roles_semantically_weak,present_but_not_recovered_by_merlin,backbone_once 489 / repeated_head 288 linear + 2 blank attn rows; once/repeated ignores K-loop reuse
pi05,quant_not_in_fp32_capture,not_present_in_source,loader dtype=float32 loader.py:55; model_qdq.mlir only 3 arith.sitofp no quant ops
pi05,loop_carried_latent,present_in_source_erased_by_export,x_t=x_t+dt*v_t pi0_pytorch.py:418 not in one-step graph
```
