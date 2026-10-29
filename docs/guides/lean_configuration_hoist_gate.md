---
title: Check scoped configuration loop hoisting with Lean
kind: guide
status: current
owner: verification
last_verified: 2026-10-28
related: [verification, lean_configuration_elision_gate, lean_loop_transform_gate]
code_refs:
  - src/merlin/verify/configuration_hoist_intake.py
  - src/merlin/verify/configuration_hoist_capture.py
  - src/merlin/verify/configuration_hoist_bridge.py
  - src/merlin/verify/configuration_hoist_gate.py
  - src/merlin/verify/lean/Merlin/GraphStateConfigurationHoistChecker.lean
---

# Check scoped configuration loop hoisting with Lean

This direct developer API checks complete Kernel IR before/after moving one
invariant configuration setter immediately before a positive counted loop.
It does not install compiler-wide enforcement or establish native ISA fidelity.

```python
from pathlib import Path
from merlin.verify.configuration_hoist_intake import prepare
from merlin.verify.configuration_hoist_gate import check, deliver, replay

candidate = prepare(
    Path("source.mk").read_text(), Path("final.mk").read_text(),
    reference="/path/to/detached-e048-producer",
    facts="/path/to/original-facts.json",
    software="/path/to/original-software.yaml",
    target="selected-target",
    source_manifest="/path/to/PRODUCER-SOURCE-MANIFEST-e0484f488.json",
)
lean = Path("/path/to/selected/lean")
result = check(candidate, lean=lean, destination="out/build/hoist/check-1")
if result.status != "checked":
    raise RuntimeError(result.reason)
deliver(result.completion, candidate, consume_checked_kernel)
again = replay(result.completion.report, lean=lean,
               destination="out/build/hoist/replay-1")
assert again.status == "checked", again.reason
```

Destinations must be fresh. The read-only producer must remain detached at
e0484f4886ae0780d0325cdfa998842b61bb647c. The source manifest has pinned SHA256
bb35450b5374e4fea125d0f766588830eea7a771131de7d1e961d5a4a4ea2626.
Its 2,009 Python source blobs were independently compared with that Git revision.
Used file origins and alias helpers, original facts/software, fresh Python-cache
absence, and complete typed Kernel text/parse roundtrips are checked. These are
source-attribution controls, not proofs of Python, parser or ISA correctness.

The initial domain has one positive static top-level loop, one leading pure
constant setter, a nonempty flat ordered body, and no body writes/clobbers of
setup fields. Even identical overlapping writes are outside this subset.
Recognized instruction operands are integer constants. Completion lists and
call asynchronous/unit/staging annotations must be empty. Zero-trip, nested,
multi-prefix, IV-dependent and unsupported operand/control forms are refused.
Kernel headers, arguments, descriptors, call fields and metadata stay bound to
the complete actual endpoints.

`ActualConfigurationHoist.complete_kernel` instantiates the proved partial-loop
checker. Under one common indexed call interpretation, a total exact pure setter
and the declared successful body frames, it preserves complete successful and
error/terminal states from arbitrary initial state. The checker derives setup
fixedness and the body frame condition; it assumes neither source success nor
whole-run equality. Surrounding partial calls and arbitrary modeled observers
inherit equality. The interpretation must represent all relevant caller, memory,
return and observation state with a nonaliasing configuration mapping.

Native checking uses the installed package's nine canonical proof modules,
independent of caller `MERLIN_REPO_ROOT`, plus a generated query in a fresh
private tree. All fifteen roots/types/imports and standard-three-axiom evidence
are audited. Live delivery revalidates source/tool/process/query/object bindings;
saved or copied tokens do not authorize consumption. Replay rebuilds fresh.

Human effect-policy approval, primitive/native/hardware fidelity, actual-model
coverage and composition with adjacent passes remain separate obligations.
This is a scoped conditional result, not verification of the complete hoisting
algorithm, all scheduling transforms, completion-token loops or numerical policy.
