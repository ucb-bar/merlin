# rdt — source → FX → MLIR → Merlin forensic audit

## 1. Header

- **Model family:** Robotics Diffusion Transformer (RDT), a DiT-style **diffusion denoise step**.
  The captured `forward` (`models/rdt/model.py:126`) is exactly the per-step denoise network the
  DDPM/DPM sampler (`models/rdt_runner.py:119 conditional_sample`) calls in its loop. It is a
  bidirectional (non-causal) transformer stack: self-attention + **cross-attention** to lang/img
  condition tokens + GELU-MLP, repeated `depth` times.
- **What the capture is:** **ONE denoise step.** The diffusion while-loop, the frozen vision/text
  encoders (SigLIP/DINOv2/T5), and the lang/img/state adaptor MLPs all live OUTSIDE the graph
  (`workloads/rdt/loader.py:1-17,35-48`). The wrapper feeds already-adapted hidden-size condition
  tokens. **K (the denoise-step count, "K=5") is a sidecar/reference constant, NOT in the graph.**
- **Config (tiny smoke, NOT real 1B):** `M2M_RDT_DEPTH=2` (loader.py:54; real 1B = 28),
  hidden=2048, heads=32, head_dim=64, horizon=64, output_dim=128, max_lang=1024, img_cond=4096,
  dtype=float32 (loader.py:58-65). The runtime token count is `N = horizon+3 = 67`
  (model.py:141-146 prepends timestep+freq tokens to the `horizon+1` state+action sequence).
- **Checkpoint status:** **not a real checkpoint.** Weights are random/`xavier_uniform_`
  (`initialize_weights`, model.py:68-124); the only structured tensor is an empty lang embed.
  **All MAC/byte magnitudes are structural-only.** Shapes, ratios, op-mix and rankings are the signal.

## 2. Feature table

| Feature | Source PyTorch | Exported FX | MLIR (flat / level) | Merlin artifact | STATUS |
|---|---|---|---|---|---|
| model class | `class RDT(nn.Module)` model.py:22 | module name erased; flattened `GraphModule.forward` (rdt.txt:27) | `func.func @forward` flat:2; `prov.module="model"` on every op | `workload==rdt`, `prov.module=model` (operator_full_inventory all rows) | present_in_source_erased_by_export |
| exported submodule boundary | `RDTBlock`/`CrossAttention`/`Attention`/`Mlp` nn.Modules (blocks.py:72,144,186) | submodule structure gone; only flat op list + `# File:` comments (rdt.txt:107,143) | reconstructed via `prov.fqn` e.g. `model.blocks.0.cross_attn.kv` (flat:598) | `prov_fqn` column groups ops into blocks/heads (operator_full_inventory) | present_in_source_erased_by_export |
| forward entry | `def forward(self,x,freq,t,lang_c,img_c,...)` model.py:126 | graph signature lists all params+inputs (rdt.txt:28) | `func.func @forward(%0..%63)` flat:2 | implicit (single capture per workload) | present_and_preserved |
| action / denoise head | `FinalLayer` (norm+GELU-MLP→out_channels) blocks.py:186-202 | `linear_18/gelu_2/linear_19`→slice (rdt.txt:303-318) | `model.final_layer.ffn_final.fc1/fc2` matmul_26/27 (operator inv idx18-19) | rows op_index 18,19 `prov_fqn=model.final_layer.ffn_final.*` | present_and_preserved |
| K denoise loop | sampler loop in `conditional_sample` (rdt_runner.py:119-141), NOT in `RDT.forward` | absent — only one step traced (rdt.txt has no scf/loop) | no `scf.for` (capture_level_ablation `scf_for=0` all levels) | `invocations=5`, `lifetime=across_K` (data_movement); "K=5 assumed_reference" (cross_workload_provenance:3) | sidecar_or_config_only |
| action horizon H | `horizon=64` → `x[:, -horizon:]` model.py:164; loader horizon=64 | output `slice_2: f32[1,64,128]` (rdt.txt:318) | final `tensor<1x64x128xf32>` return (flat:2 ret type) | output_bytes / shape via final matmul N=128, M=67→sliced 64 | present_and_preserved |
| control / replan cadence | `freq` ctrl-frequency scalar → `freq_embedder` (model.py:49,142); no replan logic in graph | `freq` input → timestep_embedding → mlp (rdt.txt:58-83) | `model.freq_embedder` matmul_2/3 (operator inv idx2-3) | freq_embedder GEMV rows; replan cadence itself only in sidecar (abstraction_pressure_ranking async_chunk_overlap) | sidecar_or_config_only |
| attn q/k/v projections | self-attn fused `qkv` (timm Attention); cross-attn split `q`+`kv` (blocks.py:94-95) | `attn_qkv_weight[6144,2048]`; `cross_attn_q[2048,2048]`,`cross_attn_kv[4096,2048]` (rdt.txt:28) | matmul_4 (qkv 6144), matmul_8/9 (cross q/kv) flat:279,598 | inv idx4 (qkv N=6144), idx6 (cross q), idx7/14 (cross kv) | present_and_preserved |
| attn qkᵀ & attn·v (bmm) | `q@k.T`, `attn@v` (blocks.py:126,132) / inside SDPA (blocks.py:117) | folded into `aten.scaled_dot_product_attention` (rdt.txt:122,170,224,266) ×4 | decomposed: `batch_matmul` generics `aten.bmm` matmul_5/6 (flat:419,487) | `n_attention_ops=8`, `attention_macs` (work_coverage rdt) | present_but_decomposed (present_and_preserved at artifact) |
| softmax / reduction | `attn.softmax(dim=-1)` blocks.py:129 / in SDPA | inside SDPA op (no standalone softmax in FX) | flat: decomposed reduce-max/exp/sum generics `prov.op=softmax` (flat:435-466); **high-level: `linalg_ext.softmax` ×4** (hl:435,738,1103,1384) | `n_softmax=12` (work_coverage); `linalg_ext_softmax=4` at high_level (capture_level_ablation) | present_in_source_erased_by_export (recovered by lowering) |
| KV cache state | none — bidirectional diffusion, no autoregressive cache (blocks.py recomputes k,v each step) | none | none; `kv_bytes=unavailable` | `kv_bytes=unavailable` (data_movement rdt) | not_present_in_source |
| conv / vision ops | none in captured graph (vision encoder is host-side, loader.py:6-8) | no conv in histogram (rdt.txt:1-25) | no conv ops | `n_conv=0` (work_coverage rdt) | not_present_in_source |
| linear / GEMM ops | `nn.Linear` ×many (blocks.py:94-99,162; model.py:48-49) | `aten.linear.default ×20` (rdt.txt:2) | `linalg.matmul ×20` (capture_level_ablation linalg_matmul=20) | 20 matmul rows (operator_full_inventory) | present_and_preserved |
| dtype | loader forces float32 (loader.py:65); source default bf16 (model.py:37) overridden | `f32[...]` everywhere (rdt.txt:28) | `tensor<...xf32>` flat; `prov.orig_dtype="float32"` | `dtype=f32` (operator_shape_table) | present_and_preserved |
| accumulator dtype | not expressed in source (PyTorch implicit) | not expressed (FX has no accum type) | not expressed (matmul outs f32, no explicit i32 accum) | not represented | unknown_source_not_found |
| quantization metadata | none in source (fp model) | none in FX | **qdq level only**: `prov.quantization="int8_weight_only"` (qdq:1) | `quant_qdq` level row exists (capture_level_ablation) | not_present_in_source (synthesized at qdq level) |
| packed layout | none in source | none | weights `tensor<...xi8>` + `linalg.transpose` to packed (qdq:154-159) | only via qdq level (capture_level_ablation quant_ext_dequantize=20) | not_present_in_source (synthesized at qdq level) |
| scales / zero-points | none in source | none | qdq: `dequantize_per_channel(w, scale:f32, zp:i32)` axis=1 (qdq:63) | qdq-level params (`tensor<2048xf32>` scale, `tensor<2048xi64>` zp args) | not_present_in_source (synthesized at qdq level) |
| region attribution (backbone/repeated_head/prefix) | block list = repeated head; no separate prefix/backbone in graph | not expressed | inferred from `prov.fqn` | **all rows `role=repeated_head`** (operator_full_inventory); `region=repeated_head` (data_movement) | present_but_not_recovered_by_merlin (no backbone/prefix split — everything labeled repeated_head) |
| tied / shared weights | per-block distinct params; no weight tying in source (separate `RDTBlock` instances, model.py:62-64) | distinct params `blocks_0_*`,`blocks_1_*` (rdt.txt:28) | distinct func args `%12..` vs `%33..` | distinct fqns per block; no tie flag | not_present_in_source |
| repeated-weight lifetime across K | weights are **invariant across denoise steps** (same `RDT` reused each sampler iter, rdt_runner.py:130-138) | NOT in graph (K loop absent) | NOT in graph | `lifetime=across_K`, `avoidable_weight_reload=1564475392` (data_movement) | sidecar_or_config_only (lifetime is a reference annotation, not captured) |
| loop-carried latent state | `x_t` updated by scheduler.step across K (rdt_runner.py:130-138), in sampler not in `forward` | absent (one step) | absent | not represented (no loop-carried tensor) | sidecar_or_config_only |

## 3. rdt-specific questions

**(a) Is `model.blocks.1.cross_attn.kv` (87% of MACs) real?**
Yes, structurally real — and it is the dominant op. Source: `self.kv = nn.Linear(dim, dim*2)`
(blocks.py:95) → with dim=2048 that is **Linear(2048, 4096)**, applied to the cross-attn context
`c`. Confirmed in FX: `cross_attn_kv_weight: f32[4096, 2048]` (rdt.txt:28). The size driver is which
context feeds it: `forward` alternates `conds[i%2]` (model.py:158), so **block 0 → lang_c (L=32)**
and **block 1 → img_c (L=4096)**. Flat MLIR confirms the asymmetry: matmul_9 (block0) is
`32x2048 · 2048x4096` but matmul_20 (block1) is `4096x2048 · 2048x4096` (flat:598 vs flat:1336),
i.e. M=4096 because the 4096-token image context is projected. MACs = 4096·2048·4096 = 34.36 G,
which is **84.6% of recovered MACs** (40.60 G total; operator_full_inventory idx14, work_coverage).
(The "87%" headline is the right order; exact recovered-MAC share is 84.6%.) Note this is an
artifact of depth=2 making block1's img-context kv the single heavy op; at real depth=28 it would be
1 of 14 such img-blocks, not 87% alone.

**(b) One denoise step or the full K loop?** **One denoise step.** `RDT.forward` (model.py:126) is
a single network evaluation; the loop is in `conditional_sample` (rdt_runner.py:119-141) outside the
capture. No `scf.for` at any MLIR level (capture_level_ablation `scf_for=0`).

**(c) Is K captured or config-only?** **Config/sidecar only.** No loop in the graph. `invocations=5`
in data_movement_table and `lifetime=across_K` are **reference annotations**; cross_workload_provenance
(rdt, "K-step loop") states it explicitly: *"absent (loop unrolled by torch.export)… K=5 assumed_reference"*.

**(d) Are t_embedder / freq_embedder true GEMV-like tile-hostile ops?** **Yes.** Each is
`Linear(256→2048)·SiLU·Linear(2048→2048)` on a single token (TimestepEmbedder, blocks.py:32-66).
FX shows `linear: f32[1,2048]` (rdt.txt:46,52,74,80) — M=1. Merlin classifies all four as
`shape_class=gemv_like` with `M=1`, `aspect_ratio_MN≈0.0005`, `is_tail_heavy=True`
(operator_shape_table idx0-3). Genuinely tile-hostile (M=1 GEMV).

**(e) Are repeated-head weights invariant across K in source?** **Yes.** The sampler reuses the same
`RDT` instance every denoise iteration (rdt_runner.py:130-138 calls `self.model(...)` inside the
loop); weights are not re-derived per step. But this invariance is a property of the **sampler loop,
not of the captured `forward`**, so Merlin's `across_K` lifetime is an assumed_reference, not a
graph-recovered fact.

**(f) Does attention appear as `aten.scaled_dot_product_attention` in FX but decomposed in flat MLIR?**
**Yes.** FX histogram: `4 aten.scaled_dot_product_attention.default` (rdt.txt:14, ops at
122/170/224/266). Flat MLIR has **no** SDPA op — it is decomposed into two `aten.bmm` `batch_matmul`
generics (qkᵀ flat:419, attn·v flat:487) plus a fully-expanded softmax (reduce-max/exp/reduce-sum/div,
`prov.op=softmax` flat:435-466). High-level MLIR re-fuses the softmax into `linalg_ext.softmax` ×4
(hl:435,738,1103,1384) but keeps the bmm as generics. Fidelity ladder confirmed end-to-end.

## 4. FINDINGS

**What Merlin gets RIGHT:**
- All 20 Linear/GEMM ops recovered with correct M/K/N and fqn attribution
  (operator_full_inventory; flat `linalg.matmul ×20`). Shapes match FX param shapes exactly.
- The dominant op is correctly identified: `blocks.1.cross_attn.kv` = 34.36 GMAC (idx14), the only
  `squareish_gemm`, driven by the 4096-token image context — matching source semantics (b).
- Embedder GEMVs flagged `gemv_like / is_tail_heavy` (d). t/freq embedders correctly separated.
- Attention work is recovered post-lowering: `n_attention_ops=8`, `n_softmax`, `n_normalization`
  tracked; SDPA→bmm+softmax decomposition is faithful and the level ladder
  (flat generics → high_level `linalg_ext.softmax`) is real and consistent (f).
- Honesty flags are correct: K-loop/replan/lifetime are explicitly marked `assumed_reference` /
  `missing_calibration` rather than presented as captured (cross_workload_provenance).
- `visible_linear_fraction=0.9712` correctly signals the model is GEMM-dominated.

**What Merlin gets WRONG or is semantically-weak:**
- **Region attribution is degenerate:** every op is `role=repeated_head` (operator_full_inventory,
  data_movement region). RDT has no backbone/prefix/repeated_head split in the captured step (it is a
  single uniform stack), so the column carries no discriminating information here — it is a label, not
  a recovered structure. (Marked present_but_not_recovered_by_merlin.)
- **K, weight-reuse lifetime, and replan cadence are reference constants, not graph facts.** The
  `invocations=5` / `across_K` / `avoidable_weight_reload` numbers depend entirely on an assumed K=5;
  they are structural extrapolations, correctly flagged but easy to misread as measured.
- **Depth=2 distorts the MAC profile.** The 84.6%-single-op headline is an artifact of the tiny
  smoke config (only block1 sees img_c). At real depth=28 the img-context kv would be ~14 ops, not a
  single 85% outlier. The artifact does not annotate this depth-sensitivity.
- **No accumulator dtype anywhere** (unknown_source_not_found) — relevant for an int8 DSE story but
  absent at every level including qdq.

**Correctness bugs:** None found in op recovery. One numeric nuance to flag (not a bug): the prompt's
"87% of MACs" is **84.6%** of *recovered* MACs by the committed artifact; the gap is rounding/headline
vs the exact operator_full_inventory ratio, not a Merlin error.

## 5. SUMMARY_ROWS

```
rdt,model_class,present_in_source_erased_by_export,"class RDT model.py:22; flattened to func.func @forward, recovered only via prov.module=model"
rdt,exported_submodule_boundary,present_in_source_erased_by_export,"RDTBlock/CrossAttention nn.Modules (blocks.py:72,144) gone in FX; rebuilt from prov.fqn e.g. model.blocks.0.cross_attn.kv (flat:598)"
rdt,forward_entry,present_and_preserved,"forward model.py:126 -> FX signature rdt.txt:28 -> func.func @forward flat:2"
rdt,action_denoise_head,present_and_preserved,"FinalLayer blocks.py:186 -> matmul_26/27 model.final_layer.ffn_final.* (operator_full_inventory idx18-19)"
rdt,K_denoise_loop,sidecar_or_config_only,"loop in conditional_sample rdt_runner.py:119; absent from graph (scf_for=0); K=5 assumed_reference (cross_workload_provenance:3)"
rdt,action_horizon_H,present_and_preserved,"horizon=64 x[:,-64:] model.py:164 -> output tensor<1x64x128xf32> (rdt.txt:318)"
rdt,control_replan_cadence,sidecar_or_config_only,"freq ctrl scalar -> freq_embedder model.py:142 (matmul_2/3); replan cadence only in sidecar (abstraction_pressure_ranking)"
rdt,attn_qkv_projections,present_and_preserved,"qkv[6144,2048] + cross q[2048,2048]/kv[4096,2048] (rdt.txt:28) -> matmul_4/8/9 (operator_full_inventory idx4,6,7)"
rdt,attn_qk_av_bmm,present_and_preserved,"q@k.T,attn@v in SDPA (blocks.py:117) -> bmm batch_matmul generics flat:419,487; n_attention_ops=8 (work_coverage)"
rdt,softmax_reduction,present_in_source_erased_by_export,"softmax in SDPA in FX; decomposed reduce/exp flat:435; re-fused linalg_ext.softmax x4 at high_level (capture_level_ablation)"
rdt,kv_cache_state,not_present_in_source,"bidirectional diffusion, k/v recomputed each step (blocks.py:107-108); kv_bytes=unavailable (data_movement)"
rdt,conv_vision_ops,not_present_in_source,"vision encoder host-side (loader.py:6); no conv in FX histogram; n_conv=0 (work_coverage)"
rdt,linear_gemm_ops,present_and_preserved,"nn.Linear x many -> aten.linear x20 (rdt.txt:2) -> linalg.matmul x20 (capture_level_ablation)"
rdt,dtype,present_and_preserved,"loader forces f32 (loader.py:65); f32 everywhere in FX/MLIR; dtype=f32 (operator_shape_table)"
rdt,accumulator_dtype,unknown_source_not_found,"not expressed in source/FX/MLIR; matmul outs f32 with no explicit accum type"
rdt,quantization_metadata,not_present_in_source,"fp source; synthesized only at qdq level prov.quantization=int8_weight_only (qdq:1)"
rdt,packed_layout,not_present_in_source,"no packing in source; qdq level emits tensor<...xi8>+transpose (qdq:154-159)"
rdt,scales_zero_points,not_present_in_source,"absent in source/FX; qdq dequantize_per_channel(scale f32, zp i32) axis=1 (qdq:63)"
rdt,region_attribution,present_but_not_recovered_by_merlin,"single uniform stack; every op labeled role=repeated_head (operator_full_inventory) - no backbone/prefix split"
rdt,tied_shared_weights,not_present_in_source,"distinct per-block params blocks_0_* vs blocks_1_* (rdt.txt:28); no weight tying"
rdt,repeated_weight_lifetime_across_K,sidecar_or_config_only,"same RDT reused per sampler iter (rdt_runner.py:130) but not in forward; lifetime=across_K assumed_reference (data_movement)"
rdt,loop_carried_latent_state,sidecar_or_config_only,"x_t updated by scheduler.step across K (rdt_runner.py:130-138), in sampler not forward; not in graph"
```
