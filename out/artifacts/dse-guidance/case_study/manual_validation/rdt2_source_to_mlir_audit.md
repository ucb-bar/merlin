# rdt2 — source → FX → MLIR → Merlin forensic audit

**Workload:** `rdt2` (RDT2 / thu-ml RDT-style diffusion-policy action expert).
**Scope:** source-grounded structural audit. No performance claims. All magnitudes are
**structural-only** and reflect the **tiny random config** the m2m loader builds
(`M2M_RDT2_DEPTH=2`, `hidden_size=1024`, `num_heads=8`, `num_kv_heads=4`, `horizon=24`,
`action_dim=20`, `lang_len=64`, random init, no checkpoint), **not** the real default config
(real `depth=14`, per loader.py:18). Every claim is cited; not-visible items are marked missing.

## Sources cited
- **Source model:** `/scratch/agustin/projects/RDT2/models/rdt/model.py` — `class RDT(nn.Module)`
  (model.py:11), `forward` (model.py:134). Blocks: `models/rdt/blocks.py` (`RDTBlock`, `FeedForward`
  SwiGLU at blocks.py:93, `FinalLayer` at blocks.py:~205, `TimestepEmbedder` at blocks.py:56).
  Attention: `models/rdt/attention.py` (`Attention` self-attn at :76–111, `CrossAttention` at
  :172–218, GQA `repeat_kv` at :11). `RMSNorm`: `models/rdt/norm.py:24`. Runner / denoise loop:
  `models/rdt_runner.py` (`conditional_sample` :164, Euler ODE; `num_inference_timesteps`
  :81 from `noise_scheduler` config).
- **Config:** `/scratch/agustin/projects/RDT2/configs/bimanual_video_data.yaml` — `action_horizon: 24`
  (yaml:10), `action.shape: [20]` (yaml:99–100), bimanual UMI (two cameras + two arms).
- **m2m loader / capture cfg:** `/scratch/agustin/projects/model2MLIR/workloads/rdt2/{loader.py,capture.toml}`
  (`M2M_RDT2_DEPTH=2`).
- **Exported FX:** `merlin/benchmarks/dse_guidance/case_study/manual_validation/exported_fx/rdt2.txt`
  (657 lines, fully unrolled 2-block graph + final layer).
- **Flat MLIR:** `merlin/benchmarks/dse_guidance/recaptures/rdt2/model.mlir` (1907 lines,
  linalg-on-tensors; no level/high-level MLIR present for rdt2).
- **Merlin artifacts:** `case_study/{operator_full_inventory.csv, work_coverage_table.csv,
  operator_shape_table.csv}` (grep `rdt2`); `merlin/python/merlin/dse_guidance/models.py`
  `ln["rdt2"] = ModelArch("rdt2","diffusion","denoise_steps", K=5, 30.0Hz, H=64,
  note="RDT-style diffusion policy.")` (models.py:57).

## rdt2-specific questions answered

**(a) Is rdt2 structurally similar to rdt (same RDT block / cross-attn KV pattern)?**
Same *family*, **different cross-attention plumbing**. Both are DiT-style adaLN-Zero stacks:
TimestepEmbedder → N×(self-attn + cross-attn + SwiGLU FFN, each adaLN-modulated) → FinalLayer,
with GQA (`num_kv_heads<num_heads`, `repeat_kv`). BUT the cross-attn conditioning source differs:
- **rdt** projects the language/condition tokens *inside the graph* via a `cross_attn.kv` Linear
  (`operator_shape_table` rows `model.blocks.*.cross_attn.kv`) — it captures the `lang_c` branch.
- **rdt2** captures the **`lang_c_kv` (KV-cache) branch** (loader.py:63–75, model.py:184,194–198):
  the per-block (k,v) tensors are precomputed by the frozen Qwen2.5-VL VLM **host-side** and enter
  as **graph inputs** (MLIR `%55..%58 : tensor<1x4x64x128xf32>`, FX inputs `kv_0..kv_3`). So rdt2
  has **no `cross_attn.kv` op at all** — cross-attn computes only `wq` on the action stream
  (`model.blocks.*.cross_attn.wq`, 28×1024×1024) plus the two attention bmm.

**(b) Does rdt2 ALSO contain a giant cross-attn KV op dominating MACs?** **No.** This is the headline
structural divergence. Top ops by MAC (structural, depth=2 random config):

| rank | rdt (top op) | MACs | rdt2 (top op) | MACs |
|---|---|---|---|---|
| 1 | `blocks.1.cross_attn.kv` **squareish_gemm** M=4096,N=4096,K=2048 | **34.36 G** | `final_layer.ffn.fc1` 28×1024→4096 | **117.4 M** |
| 2 | `blocks.{0,1}.attn.qkv` 67×2048→6144 | 0.843 G ea | `blocks.*.ffn.{w1,w3,w2}` 28×1024↔2816 | 80.7 M ea (6×) |
| 3 | `blocks.0.cross_attn.kv` 32×4096→2048 | 0.268 G | `blocks.*.{attn,cross_attn}` proj/wq/wkv/wo 28×1024→1024 | 29.4 M ea |
| — | total recovered | **40.60 G** | total recovered | **0.952 G** |

rdt's MAC profile is **dominated by a single ~34 G `cross_attn.kv` op** (84% of all recovered MACs;
the M=4096 = full lang context × all layers materialized), classed `squareish_gemm`. rdt2 has **no op
above 0.12 G**; every GEMM is `wide_skinny`/`gemv_like`/`projection_like` and the workload is
**FFN/projection-dominated and flat** — the giant-KV-op conclusion from rdt does **NOT** generalize to
rdt2. (Caveat: rdt's 34 G figure partly reflects a *different captured branch*, not a deeper model;
rdt2 here is shallower — depth=2 vs rdt's depth=2 too, but rdt captured the in-graph KV projection of
the full context.)

**(c) Does rdt2 support or weaken the primitive-set / residency conclusions?** **Supports the
primitive set, weakens "one dominant GEMM".** Primitive set is identical to the rdt family and to the
other diffusion heads: `linalg.matmul` (23) + `linalg.generic` (193, incl. batch_matmul / softmax /
RMSNorm / SwiGLU / sin-cos / sigmoid) + `linalg.reduce` (21) + `linalg.transpose` (47); **zero conv,
zero tosa/stablehlo, zero quant** in the flat MLIR. Residency: rdt2 weights are tiny and **uniformly
sized** (no >0.12 G op, `is_tail_heavy=True` on nearly all GEMMs → all wide-skinny tails), so a
"keep-the-one-big-matmul-resident" policy that pays off for rdt has **nothing to latch onto** in rdt2;
the relevant residency question becomes the **loop-carried latent `x` (1×28×1024) + the 4 input KV
tensors**, which persist across all blocks and (in deployment) across the K=5 denoise steps.

**(d) Are role attributions correct or degenerate?** **Mostly correct, one degenerate axis.**
- Region split: **every** rdt2 op is tagged `region_role = repeated_head` (or blank for the two
  pre-block `expand`/`add` glue ops). This is **correct, not degenerate**: rdt2's captured graph is
  *entirely* the action-expert denoiser; the VLM backbone runs host-side and is not in the graph
  (loader.py:11–15), so there is legitimately no `backbone_once` region. Contrast openvla, where the
  split is real.
- `prov.fqn` attribution is **accurate**: every GEMM carries its true module path
  (`model.blocks.0.attn.wq`, `…cross_attn.wo`, `…ffn.w1/w2/w3`, `…adaLN_modulation.1`,
  `final_layer.ffn.fc1/fc2`), recovered from IR.
- `semantic_class` is **partly degenerate**: `wq`/`wkv`/`adaLN_modulation`/`cross_attn.wq` are tagged
  `unknown` rather than `attention_qkv_projection`/`modulation`; only `wo`/proj rows get
  `attention_output_projection` and `ffn.*` get `mlp_projection`. So Q/KV-projection roles are
  **under-recovered** (named `unknown`) even though the fqn makes the role unambiguous — a recoverable
  miss, not a misattribution.

**(e) ATTENTION RECOVERY CHECK (known gap) — VERDICT: RECOVERED (caught).**
rdt2 sets `use_flash_attn=False` (loader.py:46) and the manual path uses **explicit
`torch.matmul` + `F.softmax` + `torch.matmul`** (attention.py:107–109, 212–216), so the FX dump has
**no `aten.scaled_dot_product_attention`** — it has `aten.matmul ×8` (the 4 attn instances × {qkᵀ,
attn·v}) and `aten.softmax.int ×4`. In the flat MLIR every attention contraction is tagged
`prov.op = "batch_matmul"`, `prov.family = "contraction"`, `prov.aten = "aten.bmm.default"`,
`prov._pattern_hint = "batch_matmul"` — region_ids `matmul_5, 6, 9, 10, 18, 19, 22, 23` (8 distinct
batch_matmul regions). `work_coverage_table.csv` records **`n_attention_ops = 8`,
`attention_macs = 10,551,296`, `n_softmax = 12`** for rdt2. So attention is recovered via
`prov.op = batch_matmul` (caught), **NOT** missed as `sdpa`/`family=attention`. (Note `n_softmax=12`
> the 4 attention softmaxes because the counter also folds RMSNorm-family normalizations; the 4
true attention softmaxes are region_ids `softmax_0..3`.) This **contradicts the suspected SDPA-erasure
gap** for this workload — it only bites models whose source keeps `F.scaled_dot_product_attention`
(e.g. openvla's ViT branch), not the manual-attention RDT family.

## Per-feature audit (Source / Exported-FX / MLIR / Merlin)

| Feature | Source | Exported-FX | MLIR | Merlin | Status |
|---|---|---|---|---|---|
| model class | `RDT(nn.Module)` model.py:11 | wrapped as `RDT2DenoiseStep` forward graph | `func.func @forward` | `ln["rdt2"]` models.py:57 | present_and_preserved |
| submodule boundary | `blocks[i]`, `final_layer`, `t_embedder` | fqn comments per op | `prov.fqn = model.blocks.0.attn…` | `prov_fqn` column populated | present_and_preserved |
| forward entry | `RDT.forward` model.py:134 | graph root | single `@forward` | one capture `rdt2` | present_and_preserved |
| denoise/action head | this IS the action-expert head | full body | full body | role `repeated_head` (all) | present_and_preserved |
| K denoise loop | Euler loop in `conditional_sample` rdt_runner.py:164; K=`num_inference_timesteps` | **absent** (one step captured, loader.py:15) | absent | `ModelArch.K=5` (config) | sidecar_or_config_only |
| H (horizon) | `action_horizon:24` yaml:10 → seq 28 (=24+4 reg) | `x: f32[1,24,1024]`, tokens `[1,28,1024]` | `%52: tensor<1x24x1024xf32>`, 28-len | `ModelArch.H=64` (real-cfg) | present_in_source_erased_by_export *(H=64 in arch vs 24 captured: config divergence)* |
| cadence (Hz) | not in captured graph | absent | absent | `ModelArch` 30.0 Hz | sidecar_or_config_only |
| q/k/v proj | `wq`, fused `wkv` (split via unbind) attention.py:50–80 | `linear`(wq)+`linear`(wkv)+`view`+`unbind` | `linalg.matmul` wq/wkv | rows `attn.wq`,`attn.wkv` | present_and_preserved |
| qkᵀ & attn·v bmm | `torch.matmul` attention.py:107,109,212,216 | `aten.matmul ×8` | `linalg.generic` `prov.op=batch_matmul` ×8 | `n_attention_ops=8` | present_and_preserved |
| softmax | `F.softmax(.float())` attention.py:108,215 | `aten.softmax.int ×4` | `linalg.generic` `prov.op=softmax` (`softmax_0..3`) | `n_softmax=12` (incl norms) | present_and_preserved |
| KV/prefix state | `lang_c_kv` cache from VLM, model.py:184,194–198 | **graph inputs** `kv_0..kv_3` | `%55..%58: tensor<1x4x64x128xf32>` | (inputs; not a GEMM row) | present_and_preserved |
| conv | none in action expert | none | **0** conv ops | `n_conv=0` | not_present_in_source |
| linear/GEMM | `nn.Linear` everywhere | `aten.linear`/`addmm` ×23 | `linalg.matmul ×23` | 23 inventory rows | present_and_preserved |
| dtype | `torch.float32` (loader override; real bf16) | `f32` | `f32` throughout | `dtype=f32` | present_and_preserved |
| accumulator | implicit fp32 | not explicit | `linalg.generic` accumulate in f32 | not separately recorded | present_in_source_erased_by_lowering |
| quant | none (random fp32) | none | **0** quant ops | no quant cols | not_present_in_source |
| packed layout | fused `wkv` (k,v packed, `…*2` then unbind) attention.py:52,79–80 | `view[…,2]`+`unbind(-1)` | `tensor.expand/collapse` + slice | wkv recorded as single GEMM | present_in_source_erased_by_lowering |
| scales | adaLN `scale/shift/gate` + `attn_scale` 0.0884 attention.py:107 | `mul` by scalar + chunk(9) gates | `arith.mulf` const + chunked adds | not a distinct artifact field | present_in_source_erased_by_lowering |
| region attribution | head-only (VLM host-side) | n/a | `prov.module=model` | all `repeated_head` | present_and_preserved |
| tied weights | none | none | none | none | not_present_in_source |
| repeated-weight lifetime | per-block distinct weights; **reused across K steps** | per-block params (static) | distinct func args per block | `repeated_head` role | present_but_not_recovered_by_merlin *(K-step reuse not modeled; loop absent)* |
| loop-carried latent | `x`(1×28×1024) carried block→block & step→step | threaded `add_*` residuals | SSA value threaded | not surfaced as a residency field | present_in_source_erased_by_export *(cross-step carry; only intra-graph carry visible)* |

## FINDINGS

1. **rdt2 is the same DiT/adaLN-Zero RDT family as rdt but captures a different cross-attn branch:**
   the VLM KV cache enters as **graph inputs** (`%55..%58`), so rdt2 has **no in-graph `cross_attn.kv`
   projection**. Consequence: **no giant op.** rdt's top op is a 34.36 G `squareish_gemm`
   `blocks.1.cross_attn.kv` (84% of MACs); rdt2's top op is a 0.117 G FFN `fc1`. The dominant-GEMM /
   one-resident-matmul conclusion from rdt **does not generalize** to rdt2.
2. **Primitive set generalizes; residency story flips.** Same matmul+generic+reduce+transpose set,
   zero conv/quant/tosa. But rdt2 is flat and FFN/projection-dominated (all GEMMs wide-skinny,
   `is_tail_heavy`), so the interesting residency target is the **loop-carried latent + the 4 KV
   inputs across K=5 steps**, not a single big weight.
3. **Attention recovery: RECOVERED.** Manual attention (`use_flash_attn=False`) → FX has explicit
   `matmul×8 + softmax×4`, no SDPA; MLIR tags all 8 as `prov.op=batch_matmul`; `n_attention_ops=8`,
   `attention_macs=10.55 M`. The SDPA-erasure gap does **not** affect rdt2.
4. **Role attribution correct, one recoverable miss.** All-`repeated_head` is correct (VLM is
   host-side, genuinely not in graph). `semantic_class` leaves `wq`/`wkv`/`adaLN_modulation` as
   `unknown` despite unambiguous fqns — under-recovery, not misattribution.
5. **Config vs capture divergence (not a bug):** `ModelArch` carries H=64/K=5/30Hz (real deploy)
   while the captured tiny graph is H=24 (seq 28), depth=2, single step — the K loop and cadence are
   sidecar/config-only and cross-step latent carry is invisible to the single-step capture.

```SUMMARY_ROWS
rdt2,model_class,present_and_preserved,RDT nn.Module model.py:11 -> func.func @forward
rdt2,submodule_boundary,present_and_preserved,prov.fqn=model.blocks.0.attn etc populated
rdt2,forward_entry,present_and_preserved,single @forward capture rdt2
rdt2,denoise_action_head,present_and_preserved,whole graph is the action expert; role repeated_head
rdt2,K_denoise_loop,sidecar_or_config_only,Euler loop rdt_runner.py:164 absent from graph; ModelArch.K=5
rdt2,H_horizon,present_in_source_erased_by_export,yaml:10 H24 captured (seq28) vs ModelArch.H64 real-cfg
rdt2,cadence_hz,sidecar_or_config_only,30.0Hz only in ModelArch models.py:57
rdt2,qkv_proj,present_and_preserved,wq+fused wkv attention.py:50-80 -> linalg.matmul rows
rdt2,qkT_attnv_bmm,present_and_preserved,torch.matmul x8 -> prov.op=batch_matmul x8 (matmul_5,6,9,10,18,19,22,23)
rdt2,softmax,present_and_preserved,F.softmax x4 -> prov.op=softmax softmax_0..3; n_softmax=12 (incl norms)
rdt2,kv_prefix_state,present_and_preserved,lang_c_kv cache enters as graph inputs %55..%58 (1x4x64x128)
rdt2,conv,not_present_in_source,n_conv=0 no conv in action expert
rdt2,linear_gemm,present_and_preserved,23 nn.Linear -> 23 linalg.matmul rows
rdt2,dtype,present_and_preserved,f32 loader override preserved end-to-end
rdt2,accumulator,present_in_source_erased_by_lowering,implicit f32 accumulate in linalg.generic
rdt2,quant,not_present_in_source,zero quant ops random fp32
rdt2,packed_layout,present_in_source_erased_by_lowering,fused wkv unbind -> expand/collapse; single GEMM row
rdt2,scales,present_in_source_erased_by_lowering,adaLN gates + attn_scale 0.0884 -> arith.mulf consts
rdt2,region_attribution,present_and_preserved,all repeated_head correct (VLM host-side)
rdt2,tied_weights,not_present_in_source,no weight tying
rdt2,repeated_weight_lifetime,present_but_not_recovered_by_merlin,per-block distinct; K-step reuse not modeled (loop absent)
rdt2,loop_carried_latent,present_in_source_erased_by_export,x carried block+step; only intra-graph carry visible
rdt2,giant_cross_attn_kv_op,not_present_in_source,NO cross_attn.kv (KV cache is graph input); top op 0.117G fc1 vs rdt 34.36G
rdt2,attention_recovery,present_and_preserved,RECOVERED as batch_matmul; n_attention_ops=8 attention_macs=10551296
```
