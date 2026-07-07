# xr0 — source-to-MLIR forensic audit

**Workload:** xr0 (Xiaomi-Robotics-0), a VLA model — Qwen3-VL-4B backbone (KV-cache) + a
**DiT (Diffusion Transformer) action head** decoding an action chunk by **rectified-flow Euler
integration** over `num_steps`. xr0 is the workload that motivated the 3D/4D-activation × 2D-weight
`M = prod(leading dims)` fold in the matmul extractor.

**Capture unit:** ONE DiT denoise step = `XR0.dit_forward` (a single rectified-flow velocity
prediction), built from a small random-init config; the Qwen3-VL backbone is never constructed (its
KV-cache is supplied as input tensors). Magnitudes are **structural only** (random init); **no perf
claims**. Default capture uses `XR0_DIT_LAYERS=2` (real model = 16).

**Sources cited**
- Source: `/scratch/agustin/projects/Xiaomi-Robotics-0/xr0/mibot/models/VLA/XR0.py`
- m2m wrapper: `/scratch/agustin/projects/model2MLIR/workloads/xr0/loader.py`, `capture.toml`
- Exported FX: `…/case_study/manual_validation/exported_fx/xr0.txt`
- Flat MLIR: `…/dse_guidance/recaptures/xr0/model.mlir`
- Merlin: `case_study/{operator_full_inventory.csv, work_coverage_table.csv, operator_shape_table.csv}`;
  `merlin/python/merlin/dse_guidance/{models.py, attribution.py}`

---

## Source ground truth (XR0.py)

- `XR0` model class (`@MIMODEL.register_module()`, XR0.py:402-403). `dit_forward` at XR0.py:568.
- Submodules: `DiT` (XR0.py:348) → `DecoderLayer` ×`layer_num` (XR0.py:278) → `DiTAttention`
  (XR0.py:174, GQA + QK-RMSNorm + **`F.scaled_dot_product_attention`**, XR0.py:246) + `DiTMLP`
  (XR0.py:257, **SwiGLU** `down(silu(gate(x))*up(x))`, XR0.py:275). 4 RMSNorms/layer +
  `adaln_table` (AdaLN-Zero, XR0.py:303).
- Real config (XR0.__init__, XR0.py:409-417): `action_shape=(30,32)`, `dit_num_layers=16`,
  `dit_hidden_size=1024`, **`num_steps=5`**, `flow_sampling="beta"`. Head_dim 128, 8 KV heads
  (loader `_KV_HEADS_DIT=8`, `_HEAD_DIM=128`).
- Denoise loop is a Python `for step in range(self.num_steps)` Euler integrator
  (`_flow_generate`, XR0.py:546-561) — **host-side control flow, never in the graph**
  (loader.py:7-18 declares one step as the capture unit).
- DiT input token order: `[sink, state, noisy_action]` (XR0.py:610 / loader.py:205), i.e.
  q_len = 1 + state_len(1) + action_len(30) = **32**; the VLM KV-cache (length 16 here) is
  prepended inside attention (XR0.py:243), giving K/V length 48.

---

## Four-column feature table

Status vocab: `present_and_preserved` · `present_in_source_erased_by_export` ·
`present_in_source_erased_by_lowering` · `present_but_not_recovered_by_merlin` ·
`not_present_in_source` · `unknown_source_not_found` · `sidecar_or_config_only`

| Feature | Source (XR0.py) | Exported FX (xr0.txt) | Flat MLIR (model.mlir) | Merlin artifacts | Status |
|---|---|---|---|---|---|
| Model class | `XR0` (402) | n/a (graph is `dit_forward`) | `func.func @forward` (2) | `MODEL_ARCH['xr0']` diffusion/denoise_steps (models.py:61) | present_and_preserved |
| Submodule boundary | DiT/DecoderLayer/DiTAttention/DiTMLP | lost; flat ATen graph w/ `# File:` comments | recovered via `prov.fqn` (`model.dit.layers.0.attn.qkv_proj`, …) | per-op `prov_fqn` in inventory | present_in_source_erased_by_export (re-attributed via prov.fqn) |
| Forward entry | `dit_forward` (568) | graph root | `@forward` | capture_id `xr0` | present_and_preserved |
| DiT denoise head | DiT (348) | entire graph | all 19 matmul + attention regions | 1 backbone + 2 prefix + 16 repeated-head matmuls | present_and_preserved |
| K denoise loop (num_steps=5) | `for` Euler (546-561) | **absent** (1 step captured) | absent | `denoise_steps=10` (models.py:61) — config, **≠ source 5** | sidecar_or_config_only |
| H (layers) | `dit_num_layers=16` (410) | 2 (XR0_DIT_LAYERS default) | 2 (`dit.layers.0/1`) | 2 layers' worth of ops | sidecar_or_config_only |
| Cadence (per-step vs per-token) | rectified flow, per-step | n/a | n/a | family "diffusion", "denoise_steps" (models.py:61) | sidecar_or_config_only |
| q/k/v proj | fused `qkv_proj` Linear (195) | `linear` → `f32[1,32,3072]` (FX:132) | `matmul_7/13` `1x32x1024 @ 1024x3072` + bias generic | inventory idx 7,12 `attn.qkv_proj` M=32 K=1024 N=3072 | present_and_preserved |
| qkᵀ & attn·v bmm | inside `F.sdpa` (246) | `aten.scaled_dot_product_attention` ×2 (FX:219,432) — **fused, not decomposed** | **present** as `linalg.generic`, `prov.family="attention"` `prov.op="sdpa"`, regions `attention_0/1` (qkᵀ: 5-D reduction generic line 442; attn·v: line 500) | **counted as 0** — `n_attention_ops=0`, `attention_macs=0` (work_coverage) | **present_but_not_recovered_by_merlin** |
| Softmax | inside `F.sdpa` (246) | inside fused SDPA op | present as the elementwise chain inside `attention_0/1` (`prov.op="sdpa"`) | `n_softmax=0` (no `prov.op="softmax"` exists; SDPA tag is "sdpa") | present_but_not_recovered_by_merlin |
| KV/prefix state | `past_key_values`, `repeat_kv`, `cat` (239-244) | `repeat_interleave` ×4, `cat` (FX:207-216) | `repeat_interleave`→`cat_4/5` `1x8x48x128`; kv inputs `%45-48` | inputs only; folded into attention region | present_and_preserved (as input tensors) |
| Conv | **none** (DiT is pure transformer) | none | none | `n_conv=0` | not_present_in_source |
| Linear / GEMM | `nn.Linear` ×many (qkv,o,gate,up,down,projectors,t_embed,t_proj) | 19 `aten.linear.default` | 19 `linalg.matmul` (`prov.aten="aten.linear.default"`, `transposed_b="true"`) | **19** linear GEMMs, 1,115,879,424 MACs | present_and_preserved |
| dtype | fp32 (loader forces `.to(torch.float32)`, loader:217) | `f32` throughout | `f32` operands; `prov.orig_dtype="float32"` | dtype=f32 in shape table | present_and_preserved |
| Accumulator | implicit fp32 | implicit | implicit (no explicit acc type on `linalg.matmul`) | not modeled as a separate field here | present_in_source_erased_by_lowering |
| Quant | none (fp32 capture) | none | none | int8/fp8 are separate sidecar captures (`output/xr0_int8_*`), not this one | not_present_in_source (this capture) |
| Packed layout | none (row-major) | none | `tensor.collapse/expand_shape` reshapes only | `prov.family="layout"` generics (n_layout=0 counted; reshapes are tensor ops) | present_and_preserved |
| Scales | none (fp32) | none | none | none | not_present_in_source |
| Region attribution | implicit module tree | `# File:` comments | `prov.region_id` per op | role split: 1 backbone_once / 2 prefix_builder / 16 repeated_head | present_and_preserved (heuristic — see Finding b) |
| Tied weights | none observed | distinct params per layer | distinct operands | distinct fqns | not_present_in_source |
| Repeated-weight lifetime | per-layer weights reused across 5 denoise steps | not in graph (1 step) | not in graph | inferable from denoise_steps config only | sidecar_or_config_only |
| Loop-carried latent | `z` updated each Euler step (XR0.py:558-560) | **absent** (1 step) | absent | not represented | present_in_source_erased_by_export |

---

## xr0-specific critical questions

### (a) Are the ~19 linear GEMMs real 3D/4D-activation × 2D-weight, and is the M-fold correct?
**Yes, and the fold is exact.** 19 `linalg.matmul` in MLIR, all `prov.aten="aten.linear.default"`,
`transposed_b="true"`. The activation LHS keeps its leading dims literally in the IR, e.g. the
largest, `model.dit.layers.0.mlp.down_proj` (MLIR line 682):
`ins(%688, %694 : tensor<1x32x4096xf32>, tensor<4096x1024xf32>)`.
The extractor folds `M = math.prod(ls[:-1])` (attribution.py:147) → M = 1·32 = **32**, K=4096, N=1024;
weight = RHS `[K,N]=[4096,1024]`. MAC = 32·4096·1024 = **134,217,728**, matching the inventory exactly.
Sum over all 19 = **1,115,879,424**, matching `work_coverage_table` `linear_gemm_macs` exactly →
**no double count, no under-count.** The 2-D case (`t_embedder`, `t_projector` collapsed to 2-D) is
unchanged because `prod(ls[:-1]) == ls[0]`. The fold is correct.

### (b) Are region roles correct?
The inventory matmul role split is **1 backbone_once / 2 prefix_builder / 16 repeated_head**
(matches the P16 claim). This is a **heuristic, not a real DiT structural split**: xr0 has no
prefix/backbone vs denoise-head architecture — it is a *uniform* stack of identical DecoderLayers.
The split comes from `role_from_fqn`:
- `backbone_once` = `model.t_projector.layers.0` (the timestep→6·hidden AdaLN projector), a
  per-step-once op — defensible "once per step".
- `prefix_builder` = the two `dit.layers.*.attn.qkv_proj` (idx 7, 12) — labelled prefix_builder
  presumably by an fqn pattern, but these are **ordinary per-layer QKV projections**, not a
  prefix/KV-cache builder. The genuine prefix (VLM KV-cache) is supplied as *input tensors*, never
  computed in this graph. **This role label is degenerate/misleading for xr0.**
- `repeated_head` = everything else (16 ops). Reasonable as a bucket but conflates per-step-once
  projectors (state/action/output, t_embedder) with the truly per-layer-repeated DiT GEMMs.

Verdict: roles are **internally consistent but semantically degenerate** for xr0 — there is no real
backbone/prefix/head partition in the source; the labels are fqn-pattern artifacts.

### (c) WHY did Merlin recover 0 attention contractions? — **RECOVERY GAP (confirmed)**
The attention is **source-real** and **present in the IR**, but **lost by Merlin's classifier**:
1. Source: `F.scaled_dot_product_attention` (XR0.py:246), one per DiT layer.
2. Export: stays **fused** as `aten.scaled_dot_product_attention.default` (2 in FX histogram;
   FX:219, 432) — it does **not** decompose to bmm/`aten.bmm`/`aten.matmul` generics.
3. MLIR: m2m **does** lower SDPA into explicit `linalg.generic` contractions, correctly tagged
   `prov.family="attention"`, `prov.op="sdpa"`, `prov.region_id="attention_0"/"attention_1"`
   (38 such ops). The qkᵀ is a 5-D reduction generic `1x8x32x128 @ 1x8x48x128 → 1x8x32x48`
   (MLIR line 442) and attn·v is `…x32x48 @ …x48x128 → …x32x128` (line 500) — both are genuine
   batch-matmul-shaped contractions with full operand shapes.
4. Merlin: `extract_non_gemm_ops` only assigns `OPC_ATTENTION` when
   `fam == "contraction" and prov_op == "batch_matmul"` (attribution.py:250). xr0's generics carry
   `fam == "attention"`, `prov_op == "sdpa"` — they match **no** branch (`contraction`,
   `normalization`, `reduce`, `layout`, `elementwise`) and fall to `OPC_OTHER`. Hence
   `n_attention_ops=0`, `attention_macs=0`, `n_softmax=0` in `work_coverage_table`, and the 18
   "n_other" generics in that table are exactly these dropped SDPA ops.

**This is a Merlin recovery gap, not a source absence.** Fix is one classifier line: treat
`prov.family=="attention"`/`prov.op=="sdpa"` reduction generics as `OPC_ATTENTION` and read MACs from
operand shapes (the shapes are right there). MACs are recoverable with no model-card config. With H=2,
prefix=16: qkᵀ ≈ 8·32·48·128 and attn·v ≈ 8·32·48·128 per layer (structural only).

### (d) Is the DiT denoise loop captured or config?
**Config only.** The Euler loop is host-side Python (XR0.py:546-561; loader.py:7-18). The graph is
one step. `num_steps=5` in source (XR0.py:412) but `MODEL_ARCH['xr0']` carries `denoise_steps=10`
(models.py:61) — a **mismatch** (config drift vs source); both are sidecar metadata, neither is in
the IR. (Flagged for the correctness pass.)

### (e) Does xr0 change the primitive-frontier / residency conclusions?
Yes, materially — **because of the (c) gap.** As recovered, xr0 looks like a 100%-linear-GEMM
workload (`visible_linear_fraction=1.0`, attention_macs=0), which would wrongly place the entire
attention datapath (qkᵀ/softmax/attn·v) outside the primitive frontier and outside residency
accounting. In reality xr0 has a full per-layer attention block whose contraction MACs and softmax
live-set are present in the IR. Any frontier/residency conclusion that uses xr0's recovered work
mix is **biased toward GEMM-only** until the SDPA classifier gap is fixed. The linear-GEMM side of
the conclusions is sound (a verified).

---

## FINDINGS

1. **The 0-attention-recovered result is a Merlin recovery gap, NOT a source/lowering absence.**
   xr0's attention is real in source (`F.sdpa`, XR0.py:246), survives export as a fused
   `aten.scaled_dot_product_attention`, and is correctly lowered by m2m into shape-complete
   `linalg.generic` contractions tagged `prov.family="attention"`/`prov.op="sdpa"` (regions
   `attention_0/1`, MLIR lines 442 & 500). Merlin drops them only because
   `extract_non_gemm_ops` gates `OPC_ATTENTION` on `family=="contraction" && op=="batch_matmul"`
   (attribution.py:250), which the SDPA-fused tags never match. One-line classifier fix.
2. **The M=prod(leading-dims) fold is exact.** down_proj `1x32x4096 @ 4096x1024` → M=32,
   MAC=134,217,728 ✓; all-19 sum = 1,115,879,424 ✓ = work_coverage total. No double count
   (matmul vs generic extractors are disjoint by op name).
3. **Region roles are degenerate for xr0.** 1 backbone / 2 prefix / 16 head is fqn-pattern noise;
   xr0 is a uniform DecoderLayer stack with no real prefix/backbone split (the VLM prefix is input
   tensors, not computed work). The 2 "prefix_builder" ops are just per-layer qkv_proj GEMMs.
4. **The denoise loop and layer count are config, not captured** (1 step, H=2 of real 16), and
   `denoise_steps=10` in models.py disagrees with source `num_steps=5` — a config-drift bug to fix.
5. **No conv, no quant, no scales, no tied weights** in this fp32 single-step capture (source-true;
   int8/fp8 variants are separate sidecar captures).

```SUMMARY_ROWS
xr0,model_class,present_and_preserved,XR0.py:402 -> func.func @forward (model.mlir:2)
xr0,submodule_boundary,present_in_source_erased_by_export,recovered via prov.fqn (e.g. model.dit.layers.0.attn.qkv_proj)
xr0,forward_entry,present_and_preserved,dit_forward XR0.py:568 -> @forward
xr0,dit_denoise_head,present_and_preserved,DiT XR0.py:348; 19 matmul + 2 attention regions in MLIR
xr0,K_denoise_loop,sidecar_or_config_only,host Euler loop XR0.py:546-561 not in graph; models.py:61 denoise_steps=10 (!= source num_steps=5)
xr0,H_layers,sidecar_or_config_only,dit_num_layers=16 (src) vs XR0_DIT_LAYERS=2 captured
xr0,cadence,sidecar_or_config_only,family diffusion/denoise_steps models.py:61
xr0,qkv_proj,present_and_preserved,fused qkv Linear XR0.py:195 -> matmul_7/13 M=32 K=1024 N=3072
xr0,qkT_attn_v_bmm,present_but_not_recovered_by_merlin,F.sdpa XR0.py:246 -> linalg.generic prov.family=attention op=sdpa (MLIR 442,500) but n_attention_ops=0
xr0,softmax,present_but_not_recovered_by_merlin,inside sdpa region; n_softmax=0 (tag is op=sdpa not softmax)
xr0,kv_prefix_state,present_and_preserved,repeat_kv+cat XR0.py:239-244 -> repeat_interleave/cat_4/5; kv as inputs
xr0,conv,not_present_in_source,pure transformer DiT; n_conv=0
xr0,linear_gemm,present_and_preserved,19 linalg.matmul; 1,115,879,424 MACs (work_coverage match)
xr0,dtype,present_and_preserved,fp32 forced loader.py:217; prov.orig_dtype=float32
xr0,accumulator,present_in_source_erased_by_lowering,no explicit acc type on linalg.matmul
xr0,quant,not_present_in_source,fp32 capture; int8/fp8 are separate sidecar captures
xr0,packed_layout,present_and_preserved,row-major; only tensor.collapse/expand_shape reshapes
xr0,scales,not_present_in_source,fp32 capture
xr0,region_attribution,present_and_preserved,1 backbone_once/2 prefix_builder/16 repeated_head (heuristic, degenerate for xr0)
xr0,tied_weights,not_present_in_source,distinct per-layer weights
xr0,repeated_weight_lifetime,sidecar_or_config_only,weights reused across denoise steps not in 1-step graph
xr0,loop_carried_latent,present_in_source_erased_by_export,Euler z update XR0.py:558-560 absent from 1-step graph
```
