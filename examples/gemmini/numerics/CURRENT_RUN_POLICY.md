# Numerical policy for new runs, 2026-10-26

This user-directed update applies to **all future ResNet50, TinyLlama and
SmolVLA runs**, including TVM comparisons and tuning chunks. The machine-readable
settings are in [CURRENT_RUN_POLICY.json](CURRENT_RUN_POLICY.json). This policy
takes precedence over earlier recipe/quality descriptions for new runs. Preserve
frozen protocols, prepared bytes, failed checks and historical results unchanged;
changing calibration or arithmetic creates a new numerical release.

## Arithmetic and placement

- W8A8 operands are signed symmetric INT8 with zero point 0 and INT32
  accumulation. Weights use **−127 through 127**, with static scales shared by
  contiguous groups of **16 output channels**. Record the output-channel axis
  and handling of the last partial group.
- Activations use **−128 through 127** with one static scale per tensor.
  Calibrate once, then freeze scales for evaluation and timing. Per-row/per-token
  max-abs scales recomputed during inference do not implement this policy.
- Round to nearest, ties to even, and saturate on INT8 narrowing. Add bias in
  the INT32 accumulator domain, using the appropriate activation × weight scale;
  prove accumulator/bias bounds. FP32 bias after dequantization is a different
  numerical computation.
- INT8 requantized outputs and raw INT32 outputs are both allowed. Record each
  layer's choice, scales, bias representation and any subsequent FP32 rescaling.
- FP32 RMSNorm, RoPE, softmax, SiLU and other non-matmul host work are allowed.
  Eligible convolutions, linear layers and attention matmuls stay on Gemmini.
  Inventory every eligible operation's actual placement; floating-point QK/PV
  attention is not covered by permission for FP32 softmax or RoPE.
- Record any higher-precision decomposition, including operand planes, extra
  device matmuls/MACs, host reconstruction, bytes transferred and measured cost.
  Do not count only the equivalent single INT8 multiplication.

Static scales remove inference-time scale estimation. They do not eliminate
FP32-to-INT8 conversions after host floating-point work. The pinned Gemmini can
scale integer DMA/readout data; that is not a dynamic FP32 row quantizer.

## Model quality gates

| Model | Required comparison |
| --- | --- |
| TinyLlama | Perplexity **≤ 1.10 × FP32 perplexity** on the same **WikiText-2 test** evaluation. |
| ResNet50 | Top-1 accuracy loss **≤ 1 percentage point** versus FP32 on the same fixed evaluation set. |
| SmolVLA | Cosine similarity **≥ 0.99** using the same output space, channels, samples and aggregation as the existing results. This does not establish robot task success. |

For TinyLlama publish the actual FP32 baseline PPL, model/tokenizer revisions,
dataset/configuration/revision, exact test text/token hash, context length,
stride, BOS/EOS/separator policy, masks and scored-token count. Aggregate NLL
over scored tokens before exponentiating; use exactly the same evaluation for
FP32 and quantized models. A short continuation or validation-split probe does
not establish test-set PPL. **No matching FP32 test PPL or complete evaluation
settings have been established by the F2 chunk archive; this gate is NOT_RUN.**

For ResNet publish the fixed image/label roster and hash, preprocessing,
sample count and both accuracies; prediction agreement on one image is not
accuracy loss. For SmolVLA publish the reference identity, normalized/decoded
output space, channel treatment, zero-vector handling and aggregation. Retain
all-output and state checks even when the quality metric uses a subset of channels.
Keep calibration and quality evaluation data separate.

Quality and compiler correctness are separate gates. Passing a quality threshold
does not waive the selected prepared-IR reference checks, exact integer results,
finite-float requirements or positive/wrong-oracle controls. Conversely, matching
a quantized reference on one input does not establish model quality. Any changed
host-math policy must be explicit and independently checked; a RoPE mismatch
alone does not establish that TVM core needs a patch.

## New release and run records

Bind the policy file/hash, checkpoint, calibration roster, frozen scale/weight
bytes, prepared IR, fixtures, independent references, compiler/native-library
and hardware identities. Keep a per-layer ledger with shapes, quantization
axis/group, scale hashes, INT32 bias, output format and placement. Attach the
matched FP32 quality baseline and evaluation settings, or report the quality
gate as **NOT_RUN**; do not inherit admission from a previous release.

Tuning may begin with a correctness-checked chunk while full-model quality is
pending, provided its numerical scope and pending gates are explicit. Use
boundaries from that same prepared release, keep real reduction dimensions and
include quantization/layout/epilogue costs in whole-chunk confirmation. Never
promote a reduced output panel directly to a different full shape.

The October 26 TinyLlama checkpoint probes use dynamic activation row scales
and per-output-channel weight scales. They are **legacy diagnostics** and must
be requested explicitly with `prepare_firesim_chunks.py checkpoint-probes
--legacy-diagnostic`. Their timing and output checks remain valid for those
programs; they do not satisfy this new policy. ResNet's retained capture has
static activations and all 54 stored epilogue scale arrays repeat within groups
of 16. Source inspection also finds FP32 bias added after INT32-to-FP32
conversion. The new round-two preparation in
`out/artifacts/handoff/gemmini-firesim-dse-round2/v1/latest/` contains an actual
INT32-bias source and bounds for all 54 additions. All bounds are below 2^24,
proving equivalence to the former FP32 ordering over admitted INT8 inputs.
Weights and calibration scales remain frozen. This is a new source identity;
check compiled placement, layer egress and fixed-set accuracy separately.
Preserve the historical capture and its receipts.

The existing tuning driver does not prove numerical-policy conformance from
arbitrary IR. Review the release and layer ledger before submitting paid work.
Do not fabricate calibrated scales from a tuning input to bypass preparation.
