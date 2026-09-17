# q536 native-LOOP_CONV hybrid: hardware headroom, not a compiler result

Date: 2026-09-08. Branch: `feat/target-generalization`.

## Verdict

FireSim queue job 536 is a valid and useful **hybrid diagnostic**, but it is not Merlin's
end-to-end compiled ResNet-50 result and must not be presented as one.

The measured warm-run compute window is **555,991,472 cycles** and all 1,000 int8 logits match
Jack's frozen TVM integer reference exactly. The package, staged, and executed ELF SHA-256 are
`5703184cf70ea0e11e5a9730502c48bb86485d111255a036b5481b82c80915f3`. The queue completed
`firesim kill -> firesim infrasetup -> firesim runworkload -> firesim kill`, exit code 0.

## Ownership boundary

The hybrid replaces all 53 TVM convolution bodies with uniquely named Merlin-generated native
`LOOP_CONV` kernels:

- 3,550 Merlin `LOOP_CONV` descriptors;
- 2,316 Merlin i32-bias `LOAD3` descriptors;
- target-neutral scalar-scale/narrow-int8 epilogues inside those kernels.

But the TVM-generated program still owns:

- the activation arena and complete call graph;
- all 20 residual rescale/add/ReLU blocks;
- first-layer input padding and max-pool;
- global average pool/flatten;
- dense output.

The stitcher states this boundary explicitly in
`development_native_loop_conv_int8_epilogue_v1_20260908/validation/stitch_hybrid_resnet50.py`.
It rewrites TVM C functions and links 53 renamed Merlin objects. That is a cross-compiler hybrid
harness, not output from one Merlin full-model compilation.

## What may and may not be inferred

The arithmetic ratios are large—2.368x versus q535's 1,316,619,699 cycles and 2.765x versus q534's
1,537,416,019 cycles—but they are **not fair compiler speedups**. q536 has different whole-program
ownership and an int8 TVM reference/output boundary, whereas q535 is Merlin's canonical source
graph with its own ABI and exact f32/logit contract.

q536 does establish three important facts:

1. The public Gemmini native-convolution mechanism and narrow scalar epilogue can execute every
   ResNet convolution on the pinned hardware.
2. The performance headroom from deleting im2col/full-width boundaries is large enough to cross
   the one-billion-cycle target.
3. The missing work is compiler integration: exact source-semantic epilogue formation, encoding
   propagation, residual ownership, reentrant state initialization, and whole-graph emission.

It does not establish that Merlin's canonical compiler has completed that integration.

## Merge policy

Do not merge the TVM stitcher, TVM activation arena, layer-index rewrite, or mixed-runner harness
into the canonical compiler. That would be the mixed-lane shortcut the experiment is designed to
exclude.

The reusable inputs are limited to the general compiler mechanisms and tests:

- scalar-scale narrow int8 `LOOP_CONV` epilogue;
- i32 `LOAD3` bias and output-channel-aware bias offsets;
- exact one-layer and warm/reentrant tests;
- descriptor/capability guards.

Those mechanisms must be re-expressed through Merlin's target-neutral source graph and then pass the
canonical four-model compile gate plus exact warm/measured full-model qualification. Until then,
q535 remains the accepted end-to-end Merlin compiler checkpoint; q536 is a measured opportunity
bound.

Portable machine receipt:
`q536_native_loopconv_hybrid_receipt.json`, SHA-256
`d27cb6996e5856068779bcd733bda32774bdc55dd9e79dbbf7c418112fc5ffa7`.

