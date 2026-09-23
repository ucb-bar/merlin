# Phase 1: functional compiler

Use the [registered definition](../experiment.yaml) and the
[configuration-specific descriptor](../target/descriptor.yaml). The authored
contract and ABI are example-owned; the old task, harness and generated public
bundles remain at the descriptor's explicit compatibility resource root. The
bundle named in the definition is historical preparation input, **not** an
approved frozen input closure.

Prepare and seal a fresh [Phase 0 release](../phase0/README.md), copy the
definition, and select the release descriptor, corpus seal and matching generated
RTL-checks bundle together. Provision a genuine oracle timing record, external
toolchain and ordinary (non-symlink-grant) input trees. Inspect and preflight
the copied definition before `merlin experiment run`; resume the same run only
while every frozen input remains unchanged. A changed descriptor or grant needs
a new run. The installed controller is `python -m merlin_experiments.phase1`.

The old Universal RTL link names another user's workspace and is not copied or
endorsed here. Its target contract also records a missing registered runtime
backend and a mismatch between the L2 functional model and narrow hardware
readout. Until these are resolved and native L3 grading is actually reached,
neither a successful process exit nor an L2 pass is a functional certificate.
