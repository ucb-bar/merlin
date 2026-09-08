# Universal Gemmini live-D admission recovery

This is the compiler snapshot recovered after FireSim queue job 533 disproved
the radix-128 accumulator-readout strategy on Jack's exact Universal U250 RTL.
It is a **refusal-safety repair**, not a new ResNet-50 candidate.

The target contract already declared
`hardcode_d_to_garbage_address: true`, but the compiler's target-profile loader
only imported the two accumulator-read-width flags. The four-pass recovery then
assumed it could preload `-recovered_so_far` through D. In weight-stationary
mode this RTL replaces that D path with the garbage address, so Spike produced
an exact result for behavior the hardware does not implement.

The repair:

1. imports the hardcoded-D fact as `d_preload_from_dram`;
2. records it in every emitted target-profile receipt; and
3. refuses radix recovery before allocating buffers or emitting commands when
   the target has no live D preload.

Reproduce the guard from this directory:

```sh
PYTHONPATH="$PWD/compiler:$REPO/merlin/python" \
  "$REPO/.venv/bin/python" -m pytest -q tests/test_target_profile_d_preload.py
```

The next correct implementation path is narrow INT8 graph lowering: fuse each
convolution's bias, per-channel requantization, and activation before exposing
the i32 intermediate, and select native `LOOP_CONV` where its output contract is
representable. Until that exists, this compiler must decline the Universal
whole model rather than claim a Spike-only success.

Job 533's pinned hardware failure is retained in
`../resnet50_qdq_v2_quantfix_exact_smallread_firesim_bundle_20260908/validation/firesim_cold_job_533_failed/`.
