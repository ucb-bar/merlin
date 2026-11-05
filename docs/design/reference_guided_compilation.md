---
title: Bottom-up lifting and top-down reconstruction
kind: design
status: current
owner: mining
last_verified: 2026-10-27
related: [kernel_reconstruction, kernel_mining, rvv_kernel_mining_methodology, core_dialects]
code_refs: [src/merlin/kernels/reference_capture.py, src/merlin/kernels/kernel_evidence.py, src/merlin/kernels/reconstruction.py, src/merlin/compile/scheduling/schedule_ir.py, packages/merlin-mining/src/merlin/mining/reconstruction_reference.py, packages/merlin-mining/src/merlin/mining/reconstruction_experiments.py]
---

# Bottom-up lifting and top-down reconstruction

We’ve connected expert-kernel analysis to the automatic compiler loop. We work
from both ends: lift expert code using ISA semantics, shapes and addresses to
recover tiling, reuse and dependencies; then lower our frontend through the
existing transformation library and scheduling dialect.

Both sides must meet at the same semantic region and CCA representation. Match
fusion, layouts and numerical behavior explicitly; shape equality alone does
not establish this. Static command identities can be recovered without an ELF,
but missing register values and execution counts remain unknown.

```text
Expert:   load weights once, reuse across input tiles
Compiler: explore blocking and residency through schedule.plan
Verify:   inspect emitted code → check correctness → measure cycles
```

Differences become tasks for the existing automatic loop: try pass compositions,
adjust scheduling, or implement a missing adapter. Several compilation paths may
explain the expert kernel. Successful paths then need testing on unseen shapes.

The objective is to approach trusted expert performance through general compiler
rules. Prioritize observed reuse, transfer, fusion and scheduling gaps; exact
instruction matching is unnecessary. When the library cannot express a supported
mechanism, add a checked pass or schedule consumer on permitted compiler surfaces.
Keep positive and negative results by operation family and shape/resource guards
in the existing optimization ledger and pinned reconstruction history. Validate
rules on tails, capacity boundaries and independent shapes before shape-policy
promotion. Preserve the incumbent outside the validated guard. Expert measurements
remain trusted targets with their supplied conditions; modeled cycles retain
their own evidence kind.

The checked dense reconstruction path and loop wiring are implemented. The author
analysis tool selects mapped regions, compares fresh machine views and reports
evidence gaps. Constructing a scheduling plan establishes intent; its actual
lowering consumer and emitted code must demonstrate the requested mechanism.
Automatic fused/quantized bindings and broad generalization still need work.
