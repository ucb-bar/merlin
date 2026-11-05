# Platform-owned input selections

This directory owns authored source-production policy, board identities and
package-bound host declarations. Select each file explicitly from the experiment
or descriptor. Hardware audit assertions are checked against selected evidence;
host declaration review does not imply compiler or deployment execution.

Accelerator semantics belong in `../target/software-spec.yaml`. Operator probes
belong in `../verification/`. Generated facts, adjacent validation reports, ISA
bindings, host packages and receipts belong under the configured output root.
Do not add device kernels, schedules, copied vendor routines or private evidence.
Update current selectors when moving an input, while preserving every historical
frozen path, digest and receipt.

Mixed-precision host rules are separately authored routing-only declarations with
unreviewed numerical contracts. Preserve source dtype and require exact source
bodies, selected index span, linked images and ordinary full-output gates. Never
modify a sealed corpus profile to add a new rule; export fresh private selections.
