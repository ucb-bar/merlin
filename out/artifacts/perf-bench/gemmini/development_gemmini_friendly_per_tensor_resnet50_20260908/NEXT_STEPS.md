# Next implementation seam: native epilogue and residency

The compiler now completes, so the limiting seam is no longer parsing, integer contraction
formation, placement, or LLVM construction. It is the boundary immediately after each i32
contraction.

## Current source spelling

`compiler/mlir_oot/frontend/gemmini_friendly_quant.py` emits the contraction followed by a pointwise
f32 scale generic. The original graph then carries layout views, the accumulator-domain bias,
activation, and output quantization as separate host operations. The existing recognizer in
`compiler/mlir_oot/frontend/quantized_epilogue.py:309` only accepts the older ordered
per-channel-f32 spelling, so it forms zero epilogues for this capture. The target capability in
`compiler/mlir_oot/lowering/model_lane.py:54` also currently declares `bias_domains=("none",)`.

## Executable implementation order

1. In the integer preparation pass, recognize the immediate symmetric i32 bias QDQ and add it to
   the i32 contraction initializer (or emit a canonical i32 pointwise bias stage) before converting
   the accumulator to f32. Refuse nonzero bias zero-points, mismatched `s_bias != s_a*s_w`, shared
   bias users, and non-channel broadcast maps.
2. Emit a canonical scalar epilogue description containing `bias_i32`, `acc_scale`, round-even,
   saturation, and optional ReLU. Extend the target-neutral recognizer with this spelling; do not
   weaken its old per-channel proof.
3. Extend Gemmini's native capability and schedule only after a fixture proves the exact bias-D
   preload, scalar CONFIG_ST scale, round-even/saturation order, and ReLU order. The selected layer
   fixture produced by `capture_native_aligned_resnet50.py` is the first real checkpoint test.
4. Re-run `run_structural_census.py`. Success is not “54 contractions”; it is 54 selected narrow
   epilogues, fewer host tasks/regions, and i8 producer-consumer boundaries. Then propagate NHWC
   across those boundaries and recover source convolution tasks from the im2col/view chain.
5. Only after the frozen deployment gate passes on a fresh held-out run and a complete ImageNet
   validation may the exact same portable PT2E QDQ graph be benchmarked in Merlin and ExecuTorch.

## Required regression gates

- Exact integer reference for accumulator bias + scale + round-even + saturation over boundary and
  random accumulator values.
- Fail-closed structural tests for scale mismatch, nonzero zero-points, non-channel bias maps,
  shared users, and layout chains with nonzero padding.
- Command-buffer census: all 54 sites remain accelerator tasks; selected native epilogues rise from
  zero; host tasks, regions, host operations, and spill bytes do not regress.
- Full compiler cap remains below 120 seconds and every emitted operation retains the correct
  `merlin.global_task` owner.
- Paired backend fairness and numerical qualification as declared in
  `validation/deployment_gate.json`.
