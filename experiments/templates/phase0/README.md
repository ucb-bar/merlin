# Shared Phase 0 performance template

[`performance.yaml`](performance.yaml) is the authored, target-independent definition
of performance families and their comparison policies. Each target's experiment
definition selects it explicitly as `config.performance_template`; the target recipe
supplies hardware-specific inputs without copying the shared policy.

This file was relocated byte-for-byte from the legacy capsule profile tree.
Family identities, membership rules and claim thresholds are unchanged. New runs
freeze the declared path and contents; existing frozen runs retain their original
template and source inventories. Do not rewrite historical receipts after a move.

These declarations are not generated capsules or performance evidence. See the
[experiment workflow](../../README.md) for derivation, review and phase handoff.
