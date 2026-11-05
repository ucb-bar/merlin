# qwen kernel bench at model shapes

Integration note (2026-11-05): This is historical evidence replayed from `5884524e3cf85027e08734526c1e3dbf0a093686`; its original measurements and unsupported-variant observations have not been rerun by this port. Current D128 attention and width-1536 RMSNorm support, explicit Qwen epsilon, local source selection, and validation status are described in the [Qwen integration README](../README.md).

Regenerate: `python3 examples/atlas/model_bringup/qwen/full_model/run.py -o out/qwen-full-model-new`

This report uses the pretrained-model export inventory. Kernel-library schedules are hand-written; the Atlas compiler checks and encodes them. Real captured operand qualification and graph-extraction checks are summarized in [the model bringup report](../docs/qwen-model-bringup.md).

Input: `coverage/coverage.json` (SHA-256 `9a1925c685953050e571c649d1024bbd89f90026858048efd744ea5b9f6e1275`), coverage status INCOMPLETE. The coverage derives 9 distinct launches; 4 executed on npu_model `47b8599d8ffbe5ddd96059ffe2b5be2427a410f5` with numerics="rtl" on seeded synthetic inputs at the model's shape (2 worker processes; 182 job-seconds in all, longest job 62 s).

Model cycles are npu_model Core cycle-model counts, not RTL cycles. Cycles x count multiplies one launch by the number of launches in the model; the sums below are device-only estimates that ignore host time, launch overhead, and data movement between launches. Bit-exact compares output bytes with the independent lib/golden reference where the family has one; cosine and rel-err compare the decoded output with an FP32 PyTorch reference on the unquantized inputs, so they include FP8 input quantization. Inputs are synthetic, so fidelity and data-dependent cycles are representative, not the model's.

A distinct launch groups the same shape and variant; a distinct program is the address-specific program text, as in the separate 1,589-program kit check.

Shares below are by launch kind: `epilogue`, `rne_add`, `rope`, and `smol_fusion` include their GEMM projection and fused stage, while `projection_gemm` covers unfused panels. They are not shares of non-GEMM time; the generator metadata does not provide separately measured cycles for those stages.

Compiler backend: machine IR, atlas-opt stream and timing verifiers, and atlas-emit, with minimum tier `stream-verified`. Successful jobs require emitted words to equal npu_model's assembler; per-job JSON records binary SHA-256 digests, diagnostics, and word counts.

| device launches | derived here | qualified kit |
| --- | ---: | ---: |
| projection_panels | 18,681 | not declared |
| attention | 84 | not declared |
| rmsnorm | 171 | not declared |

## Summary

Device cycle estimate (sum of model cycles x count over 18,681 of 18,936 launches): 8,401,492,020 npu_model model cycles. It ignores host time and launch overhead.

Compiler tier counts by distinct launch: 4 `verified`.

| launch kind | launches | launches with cycles | model cycles x count | share |
| --- | ---: | ---: | ---: | ---: |
| `projection_gemm` | 18,681 | 18,681 | 8,401,492,020 | 100.00% |
| `attention` | 84 | 0 | 0 | 0.00% |
| `rmsnorm` | 171 | 0 | 0 | 0.00% |

Top shapes by total cycles:

| family | shape | count | model cycles | cycles x count |
| --- | --- | ---: | ---: | ---: |
| `projection_gemm` | 32x1536x128 (logical 1x1536x128) | 11,782 | 358,100 | 4,219,134,200 |
| `projection_gemm` | 32x1536x128 | 5,891 | 358,100 | 2,109,567,100 |
| `projection_gemm` | 32x8960x128 (logical 1x8960x128) | 672 | 2,056,340 | 1,381,860,480 |
| `projection_gemm` | 32x8960x128 | 336 | 2,056,340 | 690,930,240 |

Unsupported or failed launches: 5.
- `attention` 32x32 12/2 d128 causal x28: no-variant: attention v2 requires head dimension 64; this launch has 128
- `attention` 1x33 12/2 d128 causal x28: no-variant: attention v2 requires head dimension 64; this launch has 128
- `attention` 1x34 12/2 d128 causal x28: no-variant: attention v2 requires head dimension 64; this launch has 128
- `rmsnorm` 32x1536 (1 rows) x114: no-variant: kernels/rmsnorm/gen.py --rows 32 --width 1536 --norm-scale-exponent -6: gen.py: error: argument --width: invalid choice: '1536' (choose from 2048)
- `rmsnorm` 32x1536 x57: no-variant: kernels/rmsnorm/gen.py --rows 32 --width 1536 --norm-scale-exponent -6: gen.py: error: argument --width: invalid choice: '1536' (choose from 2048)

## Launches

| family | shape | export dtype | variant | tier | words | count | model cycles | cycles x count | golden | bit-exact | timed equals functional | cosine | rel-err | modules |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | :-: | :-: | ---: | ---: | --- |
| `attention` | 32x32 12/2 d128 causal | float32 | no-variant | n/a | | 28 | | | | | | | | `decoder.model.layers.*.self_attn` |
| `attention` | 1x33 12/2 d128 causal | float32 | no-variant | n/a | | 28 | | | | | | | | `decoder.model.layers.*.self_attn` |
| `attention` | 1x34 12/2 d128 causal | float32 | no-variant | n/a | | 28 | | | | | | | | `decoder.model.layers.*.self_attn` |
| `projection_gemm` | 32x1536x128 (logical 1x1536x128) | float32 | default schedule (no traffic-reuse recipe for this shape) | verified | 1014 | 11,782 | 358,100 | 4,219,134,200 | ordered_matmul | exact | exact | 0.99779 | 6.670% | `decoder.model.layers.*.self_attn.q_proj`, `decoder.model.layers.*.self_attn.k_proj`, `decoder.model.layers.*.self_attn.v_proj` ... |
| `projection_gemm` | 32x1536x128 | float32 | default schedule (no traffic-reuse recipe for this shape) | verified | 1014 | 5,891 | 358,100 | 2,109,567,100 | ordered_matmul | exact | exact | 0.99828 | 5.857% | `decoder.model.layers.*.self_attn.q_proj`, `decoder.model.layers.*.self_attn.k_proj`, `decoder.model.layers.*.self_attn.v_proj` ... |
| `projection_gemm` | 32x8960x128 (logical 1x8960x128) | float32 | default schedule (no traffic-reuse recipe for this shape) | verified | 5654 | 672 | 2,056,340 | 1,381,860,480 | ordered_matmul | exact | exact | 0.99137 | 13.116% | `decoder.model.layers.*.mlp.down_proj` |
| `projection_gemm` | 32x8960x128 | float32 | default schedule (no traffic-reuse recipe for this shape) | verified | 5654 | 336 | 2,056,340 | 690,930,240 | ordered_matmul | exact | exact | 0.99390 | 11.025% | `decoder.model.layers.*.mlp.down_proj` |
| `rmsnorm` | 32x1536 (1 rows) | float32 | no-variant | n/a | | 114 | | | | | | | | `decoder.model.layers.*.input_layernorm`, `decoder.model.layers.*.post_attention_layernorm`, `decoder.model.norm` |
| `rmsnorm` | 32x1536 | float32 | no-variant | n/a | | 57 | | | | | | | | `decoder.model.layers.*.input_layernorm`, `decoder.model.layers.*.post_attention_layernorm`, `decoder.model.norm` |

`words` is the executed program size in 32-bit instruction words (after the arena relocation the bench applies). Bit-exact covers the logical rows and columns the model consumes; a padding note counts output words in the zero-padded rows or columns that differ from the golden (`+0/-0`: only the sign of zero differs).

## Unmapped ops (138 rows, 4,133 instances, 28,974,426 FLOPs)

| phase | module | op | shape | count | FLOPs |
| --- | --- | --- | --- | ---: | ---: |
| prefill | `decoder.model.embed_tokens` | `aten.embedding.default` | 151936x1536, 1x32 -> 1x32x1536 float32 | 1 | 0 |
| prefill | `decoder.model` | `aten.arange.start` | - -> 32 int64 | 1 | 0 |
| prefill | `decoder.model` | `aten.unsqueeze.default` | 32 -> 1x32 int64 | 1 | 0 |
| prefill | `decoder.model` | `aten.full.default` | - -> 32x33 float32 | 1 | 0 |
| prefill | `decoder.model` | `aten.triu.default` | 32x33 -> 32x33 float32 | 1 | 1,056 |
| prefill | `decoder.model` | `aten.arange.default` | - -> 33 int64 | 1 | 0 |
| prefill | `decoder.model` | `aten.reshape.default` | 32 -> 32x1 int64 | 1 | 0 |
| prefill | `decoder.model` | `aten.gt.Tensor` | 33, 32x1 -> 32x33 bool | 1 | 1,056 |
| prefill | `decoder.model` | `aten.mul_.Tensor` | 32x33, 32x33 -> 32x33 float32 | 1 | 1,056 |
| prefill | `decoder.model` | `aten.unsqueeze.default` | 32x33 -> 1x32x33 float32 | 1 | 0 |
| prefill | `decoder.model` | `aten.unsqueeze.default` | 1x32x33 -> 1x1x32x33 float32 | 1 | 0 |
| prefill | `decoder.model` | `aten.expand.default` | 1x1x32x33 -> 1x1x32x33 float32 | 1 | 0 |
| prefill | `decoder.model.layers.*.self_attn.q_proj` | `aten.add.Tensor` bias | 1x32x1536, 1536x1536 -> 1x32x1536 float32 | 28 | 1,376,256 |
| prefill | `decoder.model.layers.*.self_attn.k_proj` | `aten.add.Tensor` bias | 1x32x1536, 256x1536 -> 1x32x256 float32 | 28 | 229,376 |
| prefill | `decoder.model.layers.*.self_attn.v_proj` | `aten.add.Tensor` bias | 1x32x1536, 256x1536 -> 1x32x256 float32 | 28 | 229,376 |
| prefill | `decoder.model.layers.*.self_attn` | `aten.view.default` | 1x32x1536 -> 1x32x12x128 float32 | 28 | 0 |
| prefill | `decoder.model.layers.*.self_attn` | `aten.transpose.int` | 1x32x12x128 -> 1x12x32x128 float32 | 28 | 0 |
| prefill | `decoder.model.layers.*.self_attn` | `aten.view.default` | 1x32x256 -> 1x32x2x128 float32 | 28 | 0 |
| prefill | `decoder.model.layers.*.self_attn` | `aten.transpose.int` | 1x32x2x128 -> 1x2x32x128 float32 | 28 | 0 |
| prefill | `decoder.model.layers.*.self_attn` | `aten.view.default` | 1x32x256 -> 1x32x2x128 float32 | 28 | 0 |
| prefill | `decoder.model.layers.*.self_attn` | `aten.transpose.int` | 1x32x2x128 -> 1x2x32x128 float32 | 28 | 0 |
| prefill | `decoder.model.layers.*.self_attn.rotary_emb` | `aten.slice.Tensor` | 131072x128 -> 32x128 float32 | 28 | 0 |
| prefill | `decoder.model.layers.*.self_attn.rotary_emb` | `aten.to.dtype` | 32x128 -> 32x128 float32 | 28 | 0 |
| prefill | `decoder.model.layers.*.self_attn.rotary_emb` | `aten.slice.Tensor` | 131072x128 -> 32x128 float32 | 28 | 0 |
| prefill | `decoder.model.layers.*.self_attn.rotary_emb` | `aten.to.dtype` | 32x128 -> 32x128 float32 | 28 | 0 |
| prefill | `decoder.model.layers.*.self_attn` | `aten.index.Tensor` | 32x128, 1x32 -> 1x32x128 float32 | 56 | 229,376 |
| prefill | `decoder.model.layers.*.self_attn` | `aten.unsqueeze.default` | 1x32x128 -> 1x1x32x128 float32 | 56 | 0 |
| prefill | `decoder.model.layers.*.self_attn` | `aten.mul.Tensor` | 1x12x32x128, 1x1x32x128 -> 1x12x32x128 float32 | 56 | 2,752,512 |
| prefill | `decoder.model.layers.*.self_attn` | `aten.slice.Tensor` | 1x12x32x128 -> 1x12x32x64 float32 | 56 | 0 |
| prefill | `decoder.model.layers.*.self_attn` | `aten.neg.default` | 1x12x32x64 -> 1x12x32x64 float32 | 28 | 688,128 |
| prefill | `decoder.model.layers.*.self_attn` | `aten.cat.default` | 1x12x32x64, 1x12x32x64 -> 1x12x32x128 float32 | 28 | 0 |
| prefill | `decoder.model.layers.*.self_attn` | `aten.add.Tensor` | 1x12x32x128, 1x12x32x128 -> 1x12x32x128 float32 | 28 | 1,376,256 |
| prefill | `decoder.model.layers.*.self_attn` | `aten.mul.Tensor` | 1x2x32x128, 1x1x32x128 -> 1x2x32x128 float32 | 56 | 458,752 |
| prefill | `decoder.model.layers.*.self_attn` | `aten.slice.Tensor` | 1x2x32x128 -> 1x2x32x64 float32 | 56 | 0 |
| prefill | `decoder.model.layers.*.self_attn` | `aten.neg.default` | 1x2x32x64 -> 1x2x32x64 float32 | 28 | 114,688 |
| prefill | `decoder.model.layers.*.self_attn` | `aten.cat.default` | 1x2x32x64, 1x2x32x64 -> 1x2x32x128 float32 | 28 | 0 |
| prefill | `decoder.model.layers.*.self_attn` | `aten.add.Tensor` | 1x2x32x128, 1x2x32x128 -> 1x2x32x128 float32 | 28 | 229,376 |
| prefill | `decoder.model.layers.*.self_attn` | `aten.unsqueeze.default` | 1x2x32x128 -> 1x2x1x32x128 float32 | 28 | 0 |
| prefill | `decoder.model.layers.*.self_attn` | `aten.expand.default` | 1x2x1x32x128 -> 1x2x6x32x128 float32 | 28 | 0 |
| prefill | `decoder.model.layers.*.self_attn` | `aten.reshape.default` | 1x2x6x32x128 -> 1x12x32x128 float32 | 28 | 0 |
| prefill | `decoder.model.layers.*.self_attn` | `aten.unsqueeze.default` | 1x2x32x128 -> 1x2x1x32x128 float32 | 28 | 0 |
| prefill | `decoder.model.layers.*.self_attn` | `aten.expand.default` | 1x2x1x32x128 -> 1x2x6x32x128 float32 | 28 | 0 |
| prefill | `decoder.model.layers.*.self_attn` | `aten.reshape.default` | 1x2x6x32x128 -> 1x12x32x128 float32 | 28 | 0 |
| prefill | `decoder.model.layers.*.self_attn` | `aten.transpose.int` | 1x12x32x128 -> 1x12x128x32 float32 | 28 | 0 |
| prefill | `decoder.model.layers.*.self_attn` | `aten.slice.Tensor` | 1x1x32x33 -> 1x1x32x32 float32 | 28 | 0 |
| prefill | `decoder.model.layers.*.self_attn` | `aten.add.Tensor` | 1x12x32x32, 1x1x32x32 -> 1x12x32x32 float32 | 28 | 344,064 |
| prefill | `decoder.model.layers.*.self_attn` | `aten.to.dtype` | 1x12x32x32 -> 1x12x32x32 float32 | 28 | 0 |
| prefill | `decoder.model.layers.*.self_attn` | `aten.transpose.int` | 1x12x32x128 -> 1x32x12x128 float32 | 28 | 0 |
| prefill | `decoder.model.layers.*.self_attn` | `aten.contiguous.default` | 1x32x12x128 -> 1x32x12x128 float32 | 28 | 0 |
| prefill | `decoder.model.layers.*.self_attn` | `aten.reshape.default` | 1x32x12x128 -> 1x32x1536 float32 | 28 | 0 |
| prefill | `decoder.model.layers.*` | `aten.add.Tensor` | 1x32x1536, 1x32x1536 -> 1x32x1536 float32 | 28 | 1,376,256 |
| prefill | `decoder.model.layers.*.mlp.act_fn` | `aten.silu.default` | 1x32x8960 -> 1x32x8960 float32 | 28 | 8,028,160 |
| prefill | `decoder.model.layers.*.mlp` | `aten.mul.Tensor` | 1x32x8960, 1x32x8960 -> 1x32x8960 float32 | 28 | 8,028,160 |
| prefill | `decoder.model.layers.*` | `aten.add.Tensor` | 1x32x1536, 1x32x1536 -> 1x32x1536 float32 | 28 | 1,376,256 |
| prefill | `decoder` | `aten.to.dtype` | 1x32x151936 -> 1x32x151936 float32 | 1 | 0 |
| prefill | `(top)` | `aten.slice.Tensor` | 1x32x151936 -> 1x1x151936 float32 | 1 | 0 |
| prefill | `(top)` | `aten.argmax.default` | 1x1x151936 -> 1x1 int64 | 1 | 151,936 |
| decode | `decoder.model.embed_tokens` | `aten.embedding.default` | 151936x1536, 1x1 -> 1x1x1536 float32 | 2 | 0 |
| decode | `decoder.model` | `aten.arange.start` | - -> 1 int64 | 2 | 0 |
| decode | `decoder.model` | `aten.unsqueeze.default` | 1 -> 1x1 int64 | 2 | 0 |
| decode | `decoder.model` | `aten.full.default` | - -> 1x34 float32 | 1 | 0 |
| decode | `decoder.model` | `aten.arange.default` | - -> 34 int64 | 1 | 0 |
| decode | `decoder.model` | `aten.reshape.default` | 1 -> 1x1 int64 | 2 | 0 |
| decode | `decoder.model` | `aten.gt.Tensor` | 34, 1x1 -> 1x34 bool | 1 | 34 |
| decode | `decoder.model` | `aten.mul_.Tensor` | 1x34, 1x34 -> 1x34 float32 | 1 | 34 |
| decode | `decoder.model` | `aten.unsqueeze.default` | 1x34 -> 1x1x34 float32 | 1 | 0 |
| decode | `decoder.model` | `aten.unsqueeze.default` | 1x1x34 -> 1x1x1x34 float32 | 1 | 0 |
| decode | `decoder.model` | `aten.expand.default` | 1x1x1x34 -> 1x1x1x34 float32 | 1 | 0 |
| decode | `decoder.model.layers.*.self_attn.q_proj` | `aten.add.Tensor` bias | 1x1x1536, 1536x1536 -> 1x1x1536 float32 | 56 | 86,016 |
| decode | `decoder.model.layers.*.self_attn.k_proj` | `aten.add.Tensor` bias | 1x1x1536, 256x1536 -> 1x1x256 float32 | 56 | 14,336 |
| decode | `decoder.model.layers.*.self_attn.v_proj` | `aten.add.Tensor` bias | 1x1x1536, 256x1536 -> 1x1x256 float32 | 56 | 14,336 |
| decode | `decoder.model.layers.*.self_attn` | `aten.view.default` | 1x1x1536 -> 1x1x12x128 float32 | 56 | 0 |
| decode | `decoder.model.layers.*.self_attn` | `aten.transpose.int` | 1x1x12x128 -> 1x12x1x128 float32 | 56 | 0 |
| decode | `decoder.model.layers.*.self_attn` | `aten.view.default` | 1x1x256 -> 1x1x2x128 float32 | 56 | 0 |
| decode | `decoder.model.layers.*.self_attn` | `aten.transpose.int` | 1x1x2x128 -> 1x2x1x128 float32 | 56 | 0 |
| decode | `decoder.model.layers.*.self_attn` | `aten.view.default` | 1x1x256 -> 1x1x2x128 float32 | 56 | 0 |
| decode | `decoder.model.layers.*.self_attn` | `aten.transpose.int` | 1x1x2x128 -> 1x2x1x128 float32 | 56 | 0 |
| decode | `decoder.model.layers.*.self_attn.rotary_emb` | `aten.slice.Tensor` | 131072x128 -> 33x128 float32 | 28 | 0 |
| decode | `decoder.model.layers.*.self_attn.rotary_emb` | `aten.to.dtype` | 33x128 -> 33x128 float32 | 28 | 0 |
| decode | `decoder.model.layers.*.self_attn.rotary_emb` | `aten.slice.Tensor` | 131072x128 -> 33x128 float32 | 28 | 0 |
| decode | `decoder.model.layers.*.self_attn.rotary_emb` | `aten.to.dtype` | 33x128 -> 33x128 float32 | 28 | 0 |
| decode | `decoder.model.layers.*.self_attn` | `aten.index.Tensor` | 33x128, 1x1 -> 1x1x128 float32 | 56 | 7,168 |
| decode | `decoder.model.layers.*.self_attn` | `aten.unsqueeze.default` | 1x1x128 -> 1x1x1x128 float32 | 112 | 0 |
| decode | `decoder.model.layers.*.self_attn` | `aten.mul.Tensor` | 1x12x1x128, 1x1x1x128 -> 1x12x1x128 float32 | 112 | 172,032 |
| decode | `decoder.model.layers.*.self_attn` | `aten.slice.Tensor` | 1x12x1x128 -> 1x12x1x64 float32 | 112 | 0 |
| decode | `decoder.model.layers.*.self_attn` | `aten.neg.default` | 1x12x1x64 -> 1x12x1x64 float32 | 56 | 43,008 |
| decode | `decoder.model.layers.*.self_attn` | `aten.cat.default` | 1x12x1x64, 1x12x1x64 -> 1x12x1x128 float32 | 56 | 0 |
| decode | `decoder.model.layers.*.self_attn` | `aten.add.Tensor` | 1x12x1x128, 1x12x1x128 -> 1x12x1x128 float32 | 56 | 86,016 |
| decode | `decoder.model.layers.*.self_attn` | `aten.mul.Tensor` | 1x2x1x128, 1x1x1x128 -> 1x2x1x128 float32 | 112 | 28,672 |
| decode | `decoder.model.layers.*.self_attn` | `aten.slice.Tensor` | 1x2x1x128 -> 1x2x1x64 float32 | 112 | 0 |
| decode | `decoder.model.layers.*.self_attn` | `aten.neg.default` | 1x2x1x64 -> 1x2x1x64 float32 | 56 | 7,168 |
| decode | `decoder.model.layers.*.self_attn` | `aten.cat.default` | 1x2x1x64, 1x2x1x64 -> 1x2x1x128 float32 | 56 | 0 |
| decode | `decoder.model.layers.*.self_attn` | `aten.add.Tensor` | 1x2x1x128, 1x2x1x128 -> 1x2x1x128 float32 | 56 | 14,336 |
| decode | `decoder.model.layers.*.self_attn` | `aten.cat.default` | 1x2x32x128, 1x2x1x128 -> 1x2x33x128 float32 | 56 | 0 |
| decode | `decoder.model.layers.*.self_attn` | `aten.unsqueeze.default` | 1x2x33x128 -> 1x2x1x33x128 float32 | 28 | 0 |
| decode | `decoder.model.layers.*.self_attn` | `aten.expand.default` | 1x2x1x33x128 -> 1x2x6x33x128 float32 | 28 | 0 |
| decode | `decoder.model.layers.*.self_attn` | `aten.reshape.default` | 1x2x6x33x128 -> 1x12x33x128 float32 | 28 | 0 |
| decode | `decoder.model.layers.*.self_attn` | `aten.unsqueeze.default` | 1x2x33x128 -> 1x2x1x33x128 float32 | 28 | 0 |
| decode | `decoder.model.layers.*.self_attn` | `aten.expand.default` | 1x2x1x33x128 -> 1x2x6x33x128 float32 | 28 | 0 |
| decode | `decoder.model.layers.*.self_attn` | `aten.reshape.default` | 1x2x6x33x128 -> 1x12x33x128 float32 | 28 | 0 |
| decode | `decoder.model.layers.*.self_attn` | `aten.transpose.int` | 1x12x33x128 -> 1x12x128x33 float32 | 28 | 0 |
| decode | `decoder.model.layers.*.self_attn` | `aten.slice.Tensor` | 1x1x1x34 -> 1x1x1x33 float32 | 28 | 0 |
| decode | `decoder.model.layers.*.self_attn` | `aten.add.Tensor` | 1x12x1x33, 1x1x1x33 -> 1x12x1x33 float32 | 28 | 11,088 |
| decode | `decoder.model.layers.*.self_attn` | `aten.to.dtype` | 1x12x1x33 -> 1x12x1x33 float32 | 28 | 0 |
| decode | `decoder.model.layers.*.self_attn` | `aten.transpose.int` | 1x12x1x128 -> 1x1x12x128 float32 | 56 | 0 |
| decode | `decoder.model.layers.*.self_attn` | `aten.reshape.default` | 1x1x12x128 -> 1x1x1536 float32 | 56 | 0 |
| decode | `decoder.model.layers.*` | `aten.add.Tensor` | 1x1x1536, 1x1x1536 -> 1x1x1536 float32 | 56 | 86,016 |
| decode | `decoder.model.layers.*.mlp.act_fn` | `aten.silu.default` | 1x1x8960 -> 1x1x8960 float32 | 56 | 501,760 |
| decode | `decoder.model.layers.*.mlp` | `aten.mul.Tensor` | 1x1x8960, 1x1x8960 -> 1x1x8960 float32 | 56 | 501,760 |
| decode | `decoder.model.layers.*` | `aten.add.Tensor` | 1x1x1536, 1x1x1536 -> 1x1x1536 float32 | 56 | 86,016 |
| decode | `decoder` | `aten.to.dtype` | 1x1x151936 -> 1x1x151936 float32 | 2 | 0 |
| decode | `(top)` | `aten.slice.Tensor` | 1x1x151936 -> 1x1x151936 float32 | 2 | 0 |
| decode | `(top)` | `aten.argmax.default` | 1x1x151936 -> 1x1 int64 | 2 | 303,872 |
| decode | `decoder.model` | `aten.full.default` | - -> 1x35 float32 | 1 | 0 |
| decode | `decoder.model` | `aten.arange.default` | - -> 35 int64 | 1 | 0 |
| decode | `decoder.model` | `aten.gt.Tensor` | 35, 1x1 -> 1x35 bool | 1 | 35 |
| decode | `decoder.model` | `aten.mul_.Tensor` | 1x35, 1x35 -> 1x35 float32 | 1 | 35 |
| decode | `decoder.model` | `aten.unsqueeze.default` | 1x35 -> 1x1x35 float32 | 1 | 0 |
| decode | `decoder.model` | `aten.unsqueeze.default` | 1x1x35 -> 1x1x1x35 float32 | 1 | 0 |
| decode | `decoder.model` | `aten.expand.default` | 1x1x1x35 -> 1x1x1x35 float32 | 1 | 0 |
| decode | `decoder.model.layers.*.self_attn.rotary_emb` | `aten.slice.Tensor` | 131072x128 -> 34x128 float32 | 28 | 0 |
| decode | `decoder.model.layers.*.self_attn.rotary_emb` | `aten.to.dtype` | 34x128 -> 34x128 float32 | 28 | 0 |
| decode | `decoder.model.layers.*.self_attn.rotary_emb` | `aten.slice.Tensor` | 131072x128 -> 34x128 float32 | 28 | 0 |
| decode | `decoder.model.layers.*.self_attn.rotary_emb` | `aten.to.dtype` | 34x128 -> 34x128 float32 | 28 | 0 |
| decode | `decoder.model.layers.*.self_attn` | `aten.index.Tensor` | 34x128, 1x1 -> 1x1x128 float32 | 56 | 7,168 |
| decode | `decoder.model.layers.*.self_attn` | `aten.cat.default` | 1x2x33x128, 1x2x1x128 -> 1x2x34x128 float32 | 56 | 0 |
| decode | `decoder.model.layers.*.self_attn` | `aten.unsqueeze.default` | 1x2x34x128 -> 1x2x1x34x128 float32 | 28 | 0 |
| decode | `decoder.model.layers.*.self_attn` | `aten.expand.default` | 1x2x1x34x128 -> 1x2x6x34x128 float32 | 28 | 0 |
| decode | `decoder.model.layers.*.self_attn` | `aten.reshape.default` | 1x2x6x34x128 -> 1x12x34x128 float32 | 28 | 0 |
| decode | `decoder.model.layers.*.self_attn` | `aten.unsqueeze.default` | 1x2x34x128 -> 1x2x1x34x128 float32 | 28 | 0 |
| decode | `decoder.model.layers.*.self_attn` | `aten.expand.default` | 1x2x1x34x128 -> 1x2x6x34x128 float32 | 28 | 0 |
| decode | `decoder.model.layers.*.self_attn` | `aten.reshape.default` | 1x2x6x34x128 -> 1x12x34x128 float32 | 28 | 0 |
| decode | `decoder.model.layers.*.self_attn` | `aten.transpose.int` | 1x12x34x128 -> 1x12x128x34 float32 | 28 | 0 |
| decode | `decoder.model.layers.*.self_attn` | `aten.slice.Tensor` | 1x1x1x35 -> 1x1x1x34 float32 | 28 | 0 |
| decode | `decoder.model.layers.*.self_attn` | `aten.add.Tensor` | 1x12x1x34, 1x1x1x34 -> 1x12x1x34 float32 | 28 | 11,424 |
| decode | `decoder.model.layers.*.self_attn` | `aten.to.dtype` | 1x12x1x34 -> 1x12x1x34 float32 | 28 | 0 |
| decode | `(top)` | `aten.cat.default` | 1x1, 1x1 -> 1x3 int64 | 1 | 0 |
| decode | `(top)` | `aten.cat.default` | 1x1x151936, 1x1x151936 -> 1x3x151936 float32 | 1 | 0 |
