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
- **Coalescing is opt-in (`--coalesce-gather`).** It moves a gathered conv block in one multi-row MVIN per
  run of equally spaced DRAM rows (and one zero-page MVIN per run of halo rows), using the pass's
  `gather_runs`. The bytes reaching each row are unchanged -- `test_scheduled_package.py` checks that row
  by row against the one-MVIN-per-row stream -- but the stream is not op-for-op the parent's, so the
  equivalence acceptance above uses it off.
- **Unmodelled commands stay with the parent.** Transposed contractions and pooling epilogues route to the
  parent's `*_parent` lowering, unchanged — never approximated by the pass.
- **Equivalence is the acceptance.** `merlin/tests/gemmini/test_scheduled_package.py` mints from each
  measured variant and requires the RAW instruction stream (configs included) to equal the hand-written
  package's, op for op; the cells the pass refuses must make the generated package refuse too.
- Source equivalence is not a certificate: it proves the same instructions, not RTL behaviour. A package
  minted here is graded like any other before its cycles are cited.

## Outputs

`out/artifacts/targets/gemmini/gemmini_xdsl_sched_<label>_<digest12>/` (untracked, tool-generated).
