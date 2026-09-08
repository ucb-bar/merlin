# Next implementation seam: convolution orientation and residency

The compiler now completes and the representable terminal FC epilogue is native. The remaining
limiting seam is the physical orientation of convolution output channels.

## Current source spelling

`compiler/mlir_oot/frontend/native_aligned_epilogue.py` now canonicalizes a proven i32 bias,
precomputed scalar multiplier, round-even/saturating i8 readout, and optional ReLU. It is opt-in
under `native_aligned_i32_bias_scalar_requant_v1`; all other integer contracts retain their prior
behavior. The target recognizer independently re-proves the generated structure.

The terminal FC is `[1,2048] x [2048,1000]`, so its length-1000 bias is Gemmini's N-axis repeating
D row and is recovered. Every convolution is currently `[Co,K] x [K,P]`; its length-Co bias is the
M axis and cannot use that preload. The census therefore admits 1 and explicitly refuses 53.

## Executable implementation order

1. Recover source convolution geometry from the proven im2col/gather/view chain, retaining the
   original activation and OIHW weight leaves plus stride, dilation, padding, and groups.
2. Prefer direct LOOP_CONV where its accumulator preload naturally accepts one i32 value per Co;
   otherwise choose the transposed `[P,K] x [K,Co] -> [P,Co]` physical matmul and propagate NHWC.
3. Extend the canonical epilogue across only proven layout views, then fuse the following ReLU and
   maxpool where scale/zero-point ordering is exact.
4. Re-run `run_structural_census.py`. Success is 54 selected native epilogues, fewer host tasks and
   accelerator regions, and i8 producer-consumer boundaries—not merely 54 mesh contractions.
5. Only after the frozen deployment gate passes on a fresh held-out run and a complete ImageNet
   validation may the exact same portable PT2E QDQ graph be benchmarked in Merlin and ExecuTorch.

## Required regression gates

- Extend the existing exact boundary-value and structural refusal coverage to every new
  source-convolution/layout formation.
- Command-buffer census: all 54 sites remain accelerator tasks; selected native epilogues rise from
  zero; host tasks, regions, host operations, and spill bytes do not regress.
- Full compiler cap remains below 120 seconds and every emitted operation retains the correct
  `merlin.global_task` owner.
- Paired backend fairness and numerical qualification as declared in
  `validation/deployment_gate.json`.
