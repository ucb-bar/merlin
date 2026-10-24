# Phase 0 (functional, all native ops)

The deterministic `merlin.targetgen.functional_truth.expectation` is vendored from
`phase0-release/truth/truth_rules.py` in the core-aten Phase 0 release snapshot, including the
2026-10-20 forced-zero-division ruling, scalar threshold promotion and quantized
float widening. Reporting rosters, payload checking and run13 placement I/O are
excluded. The rules inspect schema, scalar arguments, dtype and shapes only. The
2026-10-20 user ruling "exact identities are bit-exact" (`exact_identity`: real x*1 and x/1,
complex addmm with alpha=0 and beta=1) is vendored as well.
The specification remains primitive families and constraints; operation dispatch
in the semantic classifier implements the user-reviewed truth derivations.

The vendored truth belongs to the strict Phase 0 checker only. It lives in the
experiments distribution (`packages/merlin-experiments/src/merlin/targetgen/`) and
is imported by `build_tools/scripts/make_bounded_phase0.py` (lane commitments)
and `core_aten_capsules.py` (lane check). The compiler (`src/merlin`) never reads
it: placement comes only from native families, their constraints, the provider
proof and the excluded-construction detectors, and the numeric contract comes
from the IR (`floating_contract.py`).

Strict Phase 0 commits device/must-accelerate, no-work, or host-guard lanes before
compiler admission and fails mismatches in either direction. A correct result
on the wrong lane does not satisfy the commitment.

Future formal grading can represent the operator's deferred private full-model
gate with `--defer-private-full-models REASON`. This records deferred, not passed;
public/hidden obligations remain unchanged. Installing this option and the Codex
agent/VCS timing patches does not execute experiment Phase 1.
