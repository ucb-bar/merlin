# TinyLlama numerical diagnosis

New runs use [CURRENT_RUN_POLICY.md](../../numerics/CURRENT_RUN_POLICY.md).
The new test-set PPL gate is separate from the prepared-IR correctness checks.
The October 26 F2 chunks execute projections with precomputed FP32 boundaries;
they do not execute full-model RoPE on the target. Their successful checks
neither reproduce nor resolve a layer-19 mismatch. A claim that newlib trig/FMA
drift necessarily needs a TVM change requires first-mismatch IR, operand bits,
host-math/FP-contraction settings and target assembly evidence. Diagnose the
export/reference, compiler and libm boundaries before choosing a correction;
do not patch TVM core or relax tolerances based only on that report.

Keep the selected tolerance (`rtol=1e-5`, `atol=2e-6`), prepared model and
reference identity fixed while diagnosing a mismatch. Matching top-1 predictions
or high cosine similarity does not establish elementwise agreement. A different
host-math policy requires references derived under that policy.

## Locate the first differing cache layer

Export full arrays with `read_package_session.py`, then compare each call's
`out0` logits, `out1` KV cache and `out2` positions:

```bash
python examples/gemmini/comparisons/tvm/compare_tinyllama_outputs.py \
  --reference reference-step.npz --actual candidate-step.npz
```

The command prints JSON and returns 1 on a numerical failure. It checks exact
integer positions, finite floating outputs, all cache slots, and active keys and
values separately for every layer. Active lengths come from the reference
positions; actual positions are reported and compared independently. Padding is
reported separately so identical unused zeros cannot hide active-cache errors.
This checks numerical agreement, not execution provenance or EL4 admission.

For a decode failure, run two diagnostics: one using the candidate's carried
cache, and one using the reference cache as the decode input. The second isolates
decode arithmetic from prefill error propagation. Preserve the first as the
actual recurrent result; feeding reference state cannot qualify the session.

## Reproduce host arithmetic sensitivity locally

`diagnose_tinyllama_numerics.py` streams the pinned full checkpoint through its
22 layers using the existing **legacy dynamic W8A8 recipe**. It compiles small
floating kernels through the selected TVM Torch frontend/LLVM VM and changes
one family at a time. Integer projections and their quantizers use the same
PyTorch reference in every arm. It records the first changed INT8 input, its
scale and rounding boundary, layer checkpoints, final logits and source hashes.

Use the Torch/TVM environment described in README, with this checkout's `src`
on `PYTHONPATH` and `MERLIN_REPO_ROOT` pointing to this checkout:

```bash
python examples/gemmini/comparisons/tvm/diagnose_tinyllama_numerics.py \
  --checkpoint /path/to/pinned/TinyLlama/model.safetensors \
  --tokens-json examples/gemmini/numerics/inputs/tiny_llama/visible.json \
  --tokens 16 --activation-scale power_of_two
```

Repeat with `--activation-scale maxabs` to diagnose that distinct legacy recipe.
Use `--layers 1 --modes rms` for a bounded smoke check. Products are created with
the artifact helper under `audits/gemmini-tvm-numerics`.

The `rms` and `silu` arms replace only that operation family. The `exp_trig` and
`portable` arms also express softmax as exp/sum/divide and SiLU as division;
they do **not** isolate the libm implementation from fused operator arithmetic.
`all` covers the tested kernels, not every instruction in a compiled model.
This diagnostic runs prefill on the host, not decode or the remote RISC-V ELF.
It does not use the selected EL4 static calibrated/SmoothQuant/outlier recipe
and cannot supply its oracle or qualification.

## Follow an integer threshold crossing

At the first failing layer, compare inputs and results in order: RMSNorm,
quantization codes/scales, INT32 projection accumulator, FP32 dequantization,
RoPE, QK, softmax, PV, SiLU and the next quantizer. Compare the integer kernel
against an independent exact contraction **on its actual operand codes**.
Different codes caused by an earlier floating operation are a different problem
from a wrong device accumulation on identical codes.

`portable_exp_trig` specifies exp/sin/cos only. It does not by itself define
RMSNorm reduction order, softmax reduction order, reciprocal/multiply versus
division, or contraction of multiply/add. Audit emitted arithmetic and folded
constants as well as the linked scalar math. Bind the same declared policy to
the independent reference and candidate; do not relabel a native PyTorch output
as a portable reference or widen the tolerance to hide a mismatch.
