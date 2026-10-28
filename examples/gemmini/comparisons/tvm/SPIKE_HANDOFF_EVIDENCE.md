# Current Spike checks and reproducible FireSim handoff

[Published verification delivery](../../../../out/artifacts/handoff/gemmini-resnet-kernel-spike/v1/handoff_gemmini-resnet-kernel-spike_v1_20261028T051934Z_b632e4c/README.md) ·
[per-case hashes and results](../../../../out/artifacts/handoff/gemmini-resnet-kernel-spike/v1/handoff_gemmini-resnet-kernel-spike_v1_20261028T051934Z_b632e4c/summary.json) ·
[remote build/run instructions](FIRESIM_HANDOFF.md).

| Check | Result |
| --- | --- |
| Operator fixtures: original 51 + 20 device + 3 reuse/tail controls | 74 pass |
| Complete kernels G01–G06, C01–C03, T01–T07 | 16 pass; all outputs and Gemmini placement |
| ResNet stem, 16 bottleneck blocks, classifier | 18 pass |
| Connected published ResNet parent, 1000 logits | Pass |
| Deliberately wrong-reference ELFs | Rejected as expected for every suite/case |
| Focused host/kernel/runner/watchdog tests | 36 pass |

TVM pin: `bb4ad9f8dbd21dd96a4dc63121dfebc24742f260`. The evidence archive
contains original receipts, placement audits, reference identities and logs; it
does not contain prebuilt ELFs. Committed builders, deterministic kernel
generators and the already committed ResNet split archive provide all workload
inputs for remote rebuilding. Historical absolute paths in receipts are not
download locations. Regenerate remote studies rather than executing archived
local-path plans. Fresh extraction finds 21 ResNet contractions/10 owners and
17 kernel contractions/16 owners.

The first C01 definition passed numerically on CPU and is excluded. The admitted
flat-output TensorIR definition passes on Gemmini with unchanged inputs/reference
outputs. The kernel builder now refuses CPU fallback. Explicit convolution bias
initialization is part of its mathematical IR, not an automatic compiler pass.

These are **diagnostic Spike functional results**, not FireSim cycles, RTL
qualification or proof that every software-spec family is supported. The repaired
RNE oracle is distinct from the missing spec-pinned provider. The published
ResNet parent passes its independent reference; binding its frozen numerical
ledger to the selected 8ec protocol remains a separate admission requirement.
The complete-call profile is warm-repeat, not historical 2 MiB cache eviction.
Full TinyLlama and SmolVLA are excluded.

Repository-wide checks retain nine module-size failures and eight no-regex call
sites in unchanged files. The target-name check and staged no-regex check pass.
These are source-organization/lint failures, not failed executable checks.
