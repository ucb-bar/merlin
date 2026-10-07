---
title: Compiler feature selection
kind: reference
status: current
owner: ir
last_verified: 2026-10-06
related: [lowering_pipeline]
code_refs: [src/merlin/llvmlower/impr_features.py, packages/merlin-mining/src/merlin/mining/wholemodel_proposer.py]
---

# Compiler feature selection

Compiler improvements are named, default-off `ImprFeature` values. The registry's
`normalize` function checks names and expands the transitive `implies` closure
before validating selection constraints.

| Metadata | Meaning |
| --- | --- |
| `alternative_group` | At most one selected member of the named group is allowed. |
| `requires_exactly_one_of` | Exactly one of the named prerequisites must be selected, including implied features. |

Neither constraint chooses a prerequisite or removes another feature. For example,
a reduction schedule can require exactly one output schedule, while the output
schedules themselves share an alternative group. Selecting only the reduction
schedule fails; selecting two output schedules also fails. Runtime checks within
individual lowering features remain in place.

Both metadata fields default to empty values. Existing features without the new
metadata retain their selection behavior, and an empty feature set remains empty
with no edits to the baseline pipeline, schedule, or compiler flags.

The optional mining package's `wholemodel_proposer._feature_fork` consumes
`alternative_group`: proposing a new choice removes explicitly selected members
of the same group from the parent list, preserves other features, and validates
the resulting combination. An implied conflict still fails validation. This lets
automatic search replace one tuning choice with another instead of stacking
incompatible choices or dropping an unrelated prerequisite.
