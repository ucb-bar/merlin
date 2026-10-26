# EL4 input conformance audit

Read the [audit and chunk definitions](../../../../../../examples/gemmini/comparisons/tvm/EL4_INPUT_AUDIT.md)
for the findings, blockers and mapping of TVM traces to full-model operators.

`report.json` records the actual ResNet array comparison, source/checkpoint
identities and SmolVLA parameter shapes. `smolvla-chunks.pending.json` contains
five blocked extraction definitions and zero proposal allocation.

`authority/` retains byte-exact snapshots of the three requested files from
EL4 commit `f6f1e499565f7cb766da1b680f490cbc3dc0b54b`. They are audit data,
not a standalone protocol tree. Resolve their relative links through the
canonical files under `examples/gemmini/`.

ResNet input/provenance match, with selected-oracle/quality gates pending.
The synthetic TinyLlama source fails selected-profile conformance and remains
a diagnostic. SmolVLA weights are cached, but observation admission and prepared
exports/scales/references remain blocked. No new paid jobs were launched.
