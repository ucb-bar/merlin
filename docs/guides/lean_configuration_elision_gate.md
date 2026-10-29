---
title: Check redundant configuration writes with Lean
kind: guide
status: current
owner: verification
last_verified: 2026-10-28
related: [verification, lean_loop_transform_gate, lean_buffer_reuse_gate]
code_refs:
  - src/merlin/verify/configuration_elision_intake.py
  - src/merlin/verify/configuration_elision_bridge.py
  - src/merlin/verify/configuration_elision_gate.py
  - src/merlin/verify/lean/Merlin/GraphStateElision.lean
  - src/merlin/verify/lean/Merlin/GraphStateConfigurationEntries.lean
---

# Check redundant configuration writes with Lean

This developer API checks complete before/after modules for bounded straight-line
constant configuration-write elimination. It starts each invocation with unknown
configuration and permits deleting a whole instruction only when every written
field is already known equal and the setter has the exact total pure-overwrite
interpretation. Unknown or unordered effects clear knowledge; declared clobbers
invalidate fields. Other instructions, SSA operands, order, entries, headers and
declarations must remain bound to the actual input/output.

```python
from pathlib import Path
from merlin.verify.configuration_elision_intake import prepare
from merlin.verify.configuration_elision_gate import check, deliver, replay

candidate = prepare(
    Path("before.mlir").read_text(), Path("after.mlir").read_text(),
    reference="/path/to/pinned-producer-checkout",
    facts="/path/to/original-facts.json",
    software="/path/to/original-software.yaml",
    target="selected-target",
)
lean = Path("/path/to/selected/lean")
result = check(candidate, lean=lean, destination="out/build/configuration/check-1")
if result.status != "checked":
    raise RuntimeError(result.reason)
deliver(result.completion, candidate, lambda text: consume_checked_module(text))
rechecked = replay(result.completion.report, lean=lean,
                   destination="out/build/configuration/replay-1")
assert rechecked.status == "checked", rechecked.reason
```

Use fresh destinations. The reference resolver must have exact revision
4499eebb959c0f92cefd92cc6d80646ba66474ed and matching source bytes. It supplies
the source-owned ISA/effect selection; qualification labels do not prove that
an instruction implementation actually has those effects. Actual producer
reexecution and artifact attribution are separate qualification evidence.

The query root `ActualConfigurationElision.complete_entries` instantiates the
existing `GraphState.Elision.check_sound` known-state proof and the new complete
entry correspondence. Under one common admitted operational interpretation, it
preserves every modeled defined-entry lookup, successful state and complete
error/terminal state, from arbitrary initial state. Removed setters must be
total exact overwrites with no pulse, token, memory, failure or other observation;
retained instructions may fail and modify other modeled state. The complete
state must include relevant SSA/caller/return/observer data, with a nonaliasing
configuration-field mapping. These interpretation premises are explicit.

The scope is function-only, single-block independent entries, integer constant
ISA operands/immediates/fields and result-free pure setters. Nested regions,
CFG, internal calls and dynamic selected-ISA operands are refused. Raw reserved
`op_name__` attributes are refused before parsing, including escaped keys,
because xDSL otherwise overwrites that metadata. Unchanged declarations and
other retained operations do not gain independent execution contracts.

Checking builds seven project modules and the query in a new private object
tree, audits all seventeen required roots/types and imports, and validates
source/tool/query/process/object bindings before live delivery. Forged/copied
tokens or saved JSON grant no authority. Replay reconstructs and builds fresh;
it does not recover saved delivery authority.

Python/xDSL projection, source/ISA decoding and host transport remain trust
boundaries. Human effect review, native/hardware fidelity, whole-model coverage,
composition with adjacent passes and compiler-wide enforcement are separate
obligations. This API installs no new compiler pass selection or numerical
policy approval. Historical failed and unqualified source cuts retain their
original status.
