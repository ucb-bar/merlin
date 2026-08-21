"""Unclassified is a THIRD state, not a synonym for ineligible.

`undetermined` = the target's evidence could not decide whether it supports the family.
`unclassified` = OUR taxonomy has no name for the op, so eligibility failed closed.
Folding the second into the ineligible bucket makes "we do not have a word for this" read as "the
hardware cannot do it" — the exact denominator defect the undetermined bucket already exists to prevent.
"""

from __future__ import annotations

from merlin.targetgen import semantic_families as sf


def test_pooling_resolves_to_reduction():
    """Pooling is a reduction over a sliding window; `max`/`sum` already map there. Missing the pooling
    spellings sent 3 capsules carrying must_accelerate: true to family=None."""
    for op in ("maxpool2d", "avgpool2d", "maxpool", "avgpool",
               "global_average", "global_avg_pool", "adaptive_avg_pool2d"):
        assert sf.from_op(op) == "reduction", op


def test_the_reduction_primitives_it_is_built_from_still_map():
    assert sf.from_op("max") == "reduction" and sf.from_op("sum") == "reduction"


def test_an_unknown_op_still_fails_closed():
    """The point is not to name everything — an op we genuinely cannot classify must stay None so it is
    counted as unclassified rather than guessed into a family."""
    assert sf.from_op("wholly_unknown_op_xyz") is None
    assert sf.from_op("model") is None          # a whole model is not a semantic family


def test_coverage_reports_unclassified_separately():
    from merlin.targetgen.coverage_report import _acceleratable_coverage
    results = [{"capsule": "known", "tiers": {}}, {"capsule": "nameless", "tiers": {}}]
    caps = {"known": {"name": "known", "semantic": {"semantic_family": "contraction"},
                      "operation": {"op": "matmul"},
                      "inputs": [{"name": "A", "role": "input", "dtype": "int8", "shape": [16, 16]}]},
            "nameless": {"name": "nameless", "semantic": {},
                         "operation": {"op": "wholly_unknown_op_xyz"},
                         "inputs": [{"name": "A", "role": "input", "dtype": "int8", "shape": [16, 16]}]}}
    cov = _acceleratable_coverage(results, caps, "gemmini")
    assert cov["n_unclassified"] == 1
    assert cov["unclassified_capsules"] == ["nameless"]
    assert "n_undetermined" in cov            # the two are reported side by side, never merged
