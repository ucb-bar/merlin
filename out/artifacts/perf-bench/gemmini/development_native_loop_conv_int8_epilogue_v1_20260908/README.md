# Native LOOP_CONV INT8 epilogue recovery

This isolated compiler copy implements scalar-scale narrow INT8 convolution
epilogues with dedicated i32 `LOAD3` bias and output-channel-aware bias offsets.
It does not alter the working 92/96 compiler or earlier artifacts.

The committed snapshot is intentionally minimal: the compiler, artifact-local
target, tests, deterministic validation scripts/receipts, and one exact warm
FireSim ELF.  Redundant build trees, the unchanged Merlin support snapshot,
caches, full logit vectors, and the reusable 44 MB bitstream archive remain
untracked.

Evidence:

- 23 artifact-local compiler tests pass.
- The one-layer local-Spike probe is exact for 192/192 outputs.
- All 53 Jack-native-aligned ResNet-50 convolutions build and select native
  LOOP_CONV (3550 descriptors, 2316 bias LOAD3 descriptors).
- The full hybrid ELF replaces all 53 convolutions with Merlin kernels and is
  exact for 1000/1000 logits against the independent Jack TVM INT8 reference;
  top-1 is 258.
- FireSim qualification is tracked separately in `firesim_candidate_warm_v1`.

Reproduce the artifact-local regression suite from the repository root with:

```bash
artifact=out/artifacts/perf-bench/gemmini/development_native_loop_conv_int8_epilogue_v1_20260908
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$artifact/compiler:$artifact/runtime_target" \
  .venv/bin/python -m pytest -q "$artifact/tests"
```

Plain `pytest` without those artifact-local import roots is not a supported
invocation.

This is a hybrid graph, not a claim that Merlin compiles the residual, pooling,
dense, arena, or orchestration portions.
