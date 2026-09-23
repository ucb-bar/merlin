# q535 affine-im2col Phase-2 hardware result

Date: 2026-09-08. Branch: `feat/target-generalization`.

## Result

The second substantial full-model Phase-2 optimization is hardware-qualified:

| comparison | predecessor cycles | candidate cycles | cycles saved | reduction | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| q534 native-scalar -> q535 affine im2col | 1,537,416,019 | **1,316,619,699** | **220,796,320** | **14.3615%** | **1.16770x** |
| q530 schedule -> q535 cumulative | 1,704,064,223 | **1,316,619,699** | **387,444,524** | **22.7365%** | **1.29427x** |

All 1,000 ResNet-50 logits are exact: `bad=0`, `nonfinite=0`, top-1 258, checksum
`c6e777c3fe0aae90`. The candidate remains 316,619,699 cycles above the one-billion target.

The local Spike proxy predicted 245,561,917 cycles/instructions saved. Hardware measured
220,796,320 cycles saved, so Spike overpredicted the saved cycles by only 1.11216x for this lever.
It ranked the candidate correctly and was substantially better calibrated here than for q534.

## What changed

The target-neutral host lowering classifies affine interior spans from static convolution geometry.
It hoists row bases and y-range decisions, uses guardless copy loops for common interior columns,
and retains exact guarded gathers at borders. There are no ResNet names, layer IDs, fixed shape
allowlists, arithmetic approximations, or Gemmini-specific schedule rules.

Of 14,613,760 logical packed bytes:

- 13,628,944 bytes (93.26%) avoid repeated x/y guard and clamp work.
- 12,757,195 bytes (87.30%) use fully guardless interior loops.

The compiler's accelerator command buffer is byte-identical to q534. It still contains 3,787
`LOOP_WS` launches and 1,050 fences. FireSim reports the same 1,142,808,576 read-DMA bytes and
54,136,736 write-DMA bytes. This isolates the hardware gain to host-side dynamic-instruction
deletion rather than a mesh-schedule, transfer-volume, or model-specific change.

The same compiler was compile-qualified sequentially on the complete TinyLLaMA, LSTMNetVIT, and
SmolVLA graphs. Their targets and command buffers remain byte-identical to q534. This is a
cross-model non-regression gate, not a claim that their remaining host-heavy paths are optimized.

## Measurement contract

FireSim queue job 535 used only `/scratch/firesim_queue/bin/firesim-queue`. The daemon owned one
uninterrupted, verified lifecycle:

```text
firesim kill -> firesim infrasetup -> firesim runworkload -> firesim kill
```

The ELF ran one sparse-marker warm inference, performed an untimed 90,083,520-byte mutable-arena
reset, then ran the uninstrumented measured entry. The accepted metric is only
`MERLIN_METRIC cycles=1316619699` between measured begin/end markers. The
3,945,377,522 simulator-total cycles include warmup and harness work and are not the benchmark
result.

The package, staged ELF, and executed ELF all have SHA-256
`86aea9b8bc6aad86487e652a51629f76dcd472201c85e19077ecb7fac3dbde78`.
The HWDB hash remained
`c8ab5a86d7feab25160423f689db5fac3b94731be486ae8f81b008d713e3f706`
before every lifecycle phase and after teardown.

## Artifacts

Sealed hardware package:

```text
out/artifacts/perf-bench/gemmini/resnet50_merlin_phase2_affine_im2col_spans_w8a8_warm_measured_firesim_candidate_20260908
```

Inside it, `verify_bundle.sh` checks hashes, ABI, stack, complete accelerator descriptors, exact
Spike output, four-model gate, FireSim lifecycle, exact hardware output, and ELF identity. The
package is blocked against unchanged resubmission.

Portable files:

- `q535_affine_im2col_hardware_receipt.json`: complete machine-readable hardware receipt.
- `phase2_affine_im2col_q535.patch`: exact two-file compiler delta from the q534 scalar artifact.
  SHA-256: `8f2dc19086823b8322c9cce6d54cf169fdd1fa47367625f0d5929352dfeccc6b`.
- Hardware receipt SHA-256:
  `399fb17188aa7da4bc1b0e61d6876ef0e2657165ac87e3da6627bb11681571c1`.

## Interpretation and next lever

q535 reaches 3.107 MAC/end-to-end cycle, while loop-matmul is active for only 4.065% and the
reservation station for 4.096% of measured cycles. Those ratios are diagnostics, not a calibrated
physical roofline. They show that mesh micro-scheduling is still not the first lever.

This pass makes im2col cheaper but still performs all 14.6 MB of logical host packing and retains
546 packing-related synchronization points. The next macro path is to delete boundaries and
packing, not merely optimize their scalar implementation:

1. exact target-neutral non-residual and two-tensor residual epilogue formation;
2. narrow-layout propagation through those chains;
3. capability-selected native convolution once its warm/reentrancy and encoding contracts pass;
4. only then tile/queue/latency-hiding refinement.

The result validates the Phase-2 hierarchy: cheap exact full-model proxies selected a genuinely
material, general compiler transformation, and a single bounded hardware promotion quantified it.

