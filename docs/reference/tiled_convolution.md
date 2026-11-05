---
title: Bounded convolution operand reuse
kind: reference
status: current
owner: compile
last_verified: 2026-10-20
related: [shared_compiler_passes, ../guides/shared_pass_phase2]
code_refs:
  - src/merlin/compile/scheduling/block_schedule.py
  - merlin/tests/infra/test_tiled_convolution_schedule.py
  - merlin/tests/infra/test_shared_scheduling_export.py
---

# Bounded convolution operand reuse

`schedule_tiled_convolution(conv, geometry, knobs)` schedules the raw integer
convolution core when complete operands or all live outputs do not fit on chip.
The existing `schedule_convolution` keeps its previous behavior. Both share the
same operation builder and independent residency/reduction checker.

The new family searches bounded M/N output panels and chooses the largest
storage-legal K panel for each. Input panels are reused across output channels,
weight panels across output pixels, and weights can stay in the mesh across M.
K panels follow the original kernel-tap/channel order. Their private operand
rows can be reused after completion while accumulators remain live. Output
rows can be reused after the final stores. Padding, strides, dilation, batches
and partial channel/pixel tiles keep their source coordinates and edge depths.

The return value contains the checked `BlockSchedule`, selected M/N/K block
sizes, number of considered candidates, requested operand **elements**, and
`ConvolutionPanel` operation ranges with their global M/N/K starts and extents.
The requested-element count includes logical padding zeros. It does not measure
DRAM bytes, stalls or cycles. Candidate ordering first minimizes that volume,
then loads and K-panel count. It is a policy to measure, not a timing oracle.

Use explicit derived usable `Geometry`, accounting for other live allocations
before calling. Separate operand and accumulator spaces are required. Unknown
bank depth refuses bank-aligned placement. Search and emitted-operation limits
are checked before expansion; oversized schedules refuse. The default limits
are 4,096 M/N choices and 1,000,000 logical operations. These are bounded host
verification policies, not target capacity or executable-size allowances.

## Backend consumption through Phase 2

The public leaf is `merlin.compile.scheduling.block_schedule`. An updated
`export_shared_scheduling` release includes these APIs byte for byte without
adding imports to the public-module roster. Exporting code does not select it
for a frozen experiment. Keep the original Phase 1 package and library grants
unchanged; separately review and select the new release for both Phase 2 arms.
The [ordinary Phase 2 guide](../guides/shared_pass_phase2.md) describes that route.

All target compiler changes must be produced and evaluated through Phase 2 from
the original compiler. A manually edited target package cannot substitute for
that route or become its seed. For the EL4 V26 origin the exact package digest
is `29213759089b71de26f8a5ab26ca6ef1b8071dc0bb690041eb7de5d0efb53f24`.
Reference programs can suggest scheduling hypotheses; they supply no code,
lookup rule, capability, numerical contract or performance qualification.

A generated consumer must establish actual target completion at panel reuse,
exact accumulator continuation, physical row mapping, packing and legal
primitive instructions. Logical panel boundaries supply no fences or proof of
asynchronous safety. Bias initialization, scaling, rounding, activation,
pooling and output readout retain their original independently checked recipes;
this raw-core pass does not implement them. The caller must validate the actual
emitted program and complete outputs. Instructions and target encodings belong
to the generated backend. No FSM/hardware-loop operation is emitted by this pass.
Ordinary host counted loops may implement a repeated primitive sequence.

This API materializes finite logical operations. A backend must still meet its
own program-size and compilation limits; blindly unrolling a large schedule can
violate those limits. Compact counted-loop emission, cross-operation residency,
asynchronous overlap and automatic epilogue composition are outside this family.
The existing compact dense scheduler separately supports counted dense loops;
convolution gathers cannot simply be treated as a contiguous dense input.

## Why functional coverage missed this gap

Correctness and capacity refusals did not require resource-bounded reuse.
Small contractions fit the existing whole-operand schedule, while a backend's
single-output-tile fallback could pass all arithmetic checks on larger cases.
Neither establishes good command counts, input reuse, mesh reuse or avoidance
of repeated completion barriers. The missing test was a larger contraction that
cannot use the whole-operand schedule but must preserve exact outputs while
reusing bounded storage.

The new tests exercise that case across several independently derived resource
geometries, every loop order, K continuation, tails, and physical row reuse.
They compare execution at the actual scheduled addresses with a separate direct
NumPy convolution and reject corrupted operands, continuation, accumulator
addresses and output coverage. A consistently corrupted gather demonstrates
why residency alone is insufficient. Isolated export execution checks that the
new API works without the host repository or installed Merlin.

Phase 0 performance-scale cases can expose these resource cliffs to Phase 2.
They are a performance consumer requirement; adding them does not imply that
Phase 1 arithmetic authoring must optimize them, nor that a test pass establishes
speed. Phase 2 must connect the pass, decode actual emission, retain independent
numerics and compare performance at representative scale. No accelerator
speedup or complete-model validation follows from the shared tests.

Reproduce the core checks with:

```sh
python -m pytest -q merlin/tests/infra/test_block_schedule.py \
  merlin/tests/infra/test_tiled_convolution_schedule.py \
  merlin/tests/infra/test_shared_scheduling_export.py
```
