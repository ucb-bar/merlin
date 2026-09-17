# AGENT.md — scheduled_backend_v0

Status: active

## Purpose

Generate Gemmini backend packages whose block schedule comes from merlin's scheduling pass
(`merlin/python/merlin/compile/scheduling/block_schedule.py`) instead of a hand edit. The measured
scheduling variants under `out/artifacts/targets/gemmini/gemmini_xdsl_rtl_v1_*` each differ from their
parent only in a hand-edited `_matmul_trace`; `scripts/mint.py` produces the same instruction streams
from the pass and a knob setting, so a measured scheduling result is the compiler's, not a hand edit's.

## What is here

- `scripts/mint.py` — derive the target geometry from its RTL facts, refuse unless it equals the parent's
  baked constants, vendor the pass byte for byte, render `package_template/scheduled_trace.py`, wrap the
  parent's two traces, record lineage, write `SHA256SUMS`, freeze under a content address.
- `package_template/scheduled_trace.py` — spells the pass's abstract block moves with the parent's own
  instruction helpers. It computes no address the parent does not already define.

## Invariants

- **Generated, never edited.** A package is re-minted, not patched; same inputs, same package id.
- **The pass is vendored verbatim.** `block_schedule.py` must import nothing from merlin (the package
  integrity scan rejects a merlin import); `merlin/tests/infra/test_block_schedule.py` pins that.
- **Unmodelled commands stay with the parent.** Transposed contractions and pooling epilogues route to the
  parent's `*_parent` lowering, unchanged — never approximated by the pass.
- **Equivalence is the acceptance.** `merlin/tests/gemmini/test_scheduled_package.py` mints from each
  measured variant and requires the RAW instruction stream (configs included) to equal the hand-written
  package's, op for op; the cells the pass refuses must make the generated package refuse too.
- Source equivalence is not a certificate: it proves the same instructions, not RTL behaviour. A package
  minted here is graded like any other before its cycles are cited.

## Outputs

`out/artifacts/targets/gemmini/gemmini_xdsl_sched_<label>_<digest12>/` (untracked, tool-generated).
