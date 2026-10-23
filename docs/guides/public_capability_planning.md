---
title: Plan compiler construction from public capabilities
kind: guide
status: current
owner: experiments
last_verified: 2026-10-23
related: [conditional_compiler_pipelines, ../reference/compiler_optimization_workflow, artifact_evaluation]
code_refs:
  - packages/merlin-experiments/src/merlin_experiments/phase1/capability_inventory.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/capability_source.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/session.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/run_inputs.py
---

# Plan compiler construction from public capabilities

Fresh joint campaigns stage `PUBLIC_CAPABILITIES.md` and
`PUBLIC_CAPABILITIES.json` before the ordinary treatment seal. The initial task
points to the short summary; detailed records remain on demand. Resume preserves
those exact bytes rather than rebuilding guidance from newer source.

The inventory uses the same admitted public capsule selection and frozen
scalability selection as native grading. It preserves the original correctness
roster, adds every selected scalability case and rejects duplicate IDs or hidden
labels. Original required gSIM checks remain distinct from calibration-witness
permissions. An optional operator probe outside the public roster contributes
only an unresolved-whitelist count, never a new public member or dispatch grant.

## Read requirements without flooding context

Start with the Markdown summary, then select a family or member from JSON:

```python
import json
from pathlib import Path

inventory = json.loads(Path("PUBLIC_CAPABILITIES.json").read_text())
print(inventory["cohorts"])
# Choose a name from the summary's family list.
family = next(iter(inventory["families"]))
for name in inventory["families"][family][:3]:
    print(name, inventory["members"][name])
```

Records include public input shapes/dtypes/layout requirements, exact declared
numerical policy, source operation counts, graph edges and each case's engine
obligations. The existing interface and typed-MLIR parsers inspect only the
declared source program. No tensor arrays, golden outputs or reference compiler
implementations are opened. Unsupported or missing source analysis is explicit
`unknown`; it does not make the source an empty or supported graph.

`capability_groups` separates source operations, input dtypes, numerical policy
and declared layout requirements while sharing across shape sizes. Each group
suggests a small end-to-end canary and a large generalization case, ordered by
declared input volume. These are advisory starting points, not execution-time
estimates, sufficient test coverage, a dispatch queue or automatic fusion proofs.
Graph indices refer to the accompanying ordered operation list. Public
contraction and movement declarations are source contracts, not selected target
instructions or prescribed device schedules.

## Construct one compiler and evaluate continuously

Give the coordinator the entire compact capability inventory. Assign independent
helpers compiler boundaries such as graph transformations, target scheduling and
host numerical lowering. Each helper receives its relevant requirements and API
references rather than the whole catalog or conversation.

Test a small end-to-end case when a capability first becomes executable. Continue
building other capabilities while immutable snapshots run through the ordinary
rolling queue. Shared ABI or emission failures should be repaired once before
resubmitting affected cases. The inventory does not replace readiness execution,
change the queue, issue a pass or bypass any native prerequisite.

Original correctness and calibration prerequisites can advance concurrently with
large-shape checks. Prioritize scalability performance once numerical execution
works; small original cases remain mandatory correctness regressions and useful
model evidence. Query detailed per-case feedback only when needed. When selected,
use the shared conditional pipeline APIs to enumerate legal alternatives and
consume their transformed artifacts in actual emission. Keep checks and timing
measurements attached to the exact compiler and model versions.

The inventory carries no qualification authority. Final hidden and complete-model
obligations remain outside this public denominator and retain their existing
independent owners.
