---
title: Check complete public composition and parameter storage before building
kind: guide
status: current
owner: experiments
last_verified: 2026-10-19
related: [phase2_source_roles, phase2_development_sources, phase2_test_justification]
code_refs:
  - packages/merlin-experiments/src/merlin_experiments/phase0/development_composition_preflight.py
  - src/merlin/targetgen/oot_starterkit/plan.py
  - src/merlin/targetgen/bundle_pack.py
---

# Check complete public composition and parameter storage before building

Use the original mixed-program source/return checker and pointer-layout planner
before admitting a full build. Parameter payload alone omits external inputs,
temporaries, alignment and all live-outs. A source with parameters below a build
ceiling can exceed that ceiling once the complete emitted program is laid out.

The optional `development_composition_preflight.check` API accepts exact existing
source, command-buffer, LLVM, weight-manifest and safetensors paths. It opens the
complete files before and after checking, checks every source argument and
return through the original plan owner, preserves captured immutable sharing,
and verifies the complete captured parameter namespace, shape, dtype and payload
extent. It calls `bundle_pack.plan` with an explicit selected row granule and
counts both constant and mutable storage. Missing or extra payloads, omitted
source tasks/live-outs, changed artifacts and an exceeded original ceiling refuse.
The check never generates reference values, allocates dense parameters, builds
an ELF or executes a target.

The CLI acquires an original shared CPU leaf. Select all ceilings and the row
granule from the applicable original build and compiler/resource contracts:

```bash
python -m merlin_experiments.phase0.development_composition_preflight \
  --source "$SOURCE" --command-buffer "$COMMAND_BUFFER" --lowered "$LLVM" \
  --weight-manifest "$WEIGHT_MANIFEST" --weights "$WEIGHTS" \
  --row-pitch-elements "$ROW_GRANULE" \
  --max-storage-bytes "$ORIGINAL_STORAGE_CEILING" \
  --max-artifact-bytes "$ORIGINAL_ARTIFACT_CEILING"
```

Retain the JSON result with original source and producer pins. A pass is a public
structural and storage preflight, with numerical equivalence, device execution,
carried-state semantics and timing explicitly unverified. Original sealed source,
independent references, arithmetic/readout contracts, actual execution/product
checks, writer and private-exclusion owners remain mandatory. A historical build
rechecked here stays historical; neither its producer nor its outputs become a
fresh execution receipt.

Complete reduced-depth public models retain their nonlinear operations and state.
This preflight cannot supply missing selected RMSNorm, RoPE, softmax, floating PV,
activation or whole-graph numerical contracts. Integer QK/PV chains are useful
connected controls, but do not establish complete softmax or FlashAttention.
