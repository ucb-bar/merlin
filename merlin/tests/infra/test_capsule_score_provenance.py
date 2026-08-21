"""A capsule score asserts a hardware result, so the provenance gate must see it.

The score says N of M capsules passed on named oracle tiers — a hardware claim that never uses the words
`certified`/`correct`/`passed`. Keying only on those booleans made score_capsule.json, the primary result
artifact of the whole bench, structurally invisible to the gate on every target: the mechanism existed and
pointed away from the thing it should have been checking.
"""

from __future__ import annotations

import importlib.util

from merlin.common.paths import repo_root

_spec = importlib.util.spec_from_file_location(
    "cp", str(repo_root() / "build_tools/scripts/check_provenance.py"))
cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cp)


def test_a_capsule_score_is_a_verdict_claim():
    assert cp._claims_a_verdict({"task": "atlas-mlir-oot-capsule", "functional_pass": 0,
                                 "n_passed": 14, "n_capsules": 26}) is True


def test_an_empty_suite_asserts_nothing():
    """Nothing was graded, so no hardware is being described — requiring attribution would be noise."""
    assert cp._claims_a_verdict({"n_passed": 0, "n_capsules": 0}) is False


def test_a_zero_score_over_a_real_suite_still_claims():
    """Failing on real hardware is still a statement about that hardware."""
    assert cp._claims_a_verdict({"task": "t", "functional_pass": 0,
                                 "n_passed": 0, "n_capsules": 26}) is True


def test_the_explicit_boolean_shapes_still_work():
    for k in ("certified", "correct", "passed"):
        assert cp._claims_a_verdict({k: True}) is True
    assert cp._claims_a_verdict({"certified": False}) is False


def test_malformed_counts_do_not_raise():
    assert cp._claims_a_verdict({"task": "t", "n_passed": 1, "n_capsules": "many"}) is False
    assert cp._claims_a_verdict("not a dict") is False


def test_the_grader_emits_a_provenance_block():
    """Shape check on the emitter, so a score cannot go out unattributed."""
    from merlin.common import provenance as P
    block = P.record(pins=None, extra={"target": "t", "n_capsules": 1, "n_passed": 1})
    assert "merlin" in block and "commit" in block["merlin"]


def test_a_per_round_qa_verdict_is_not_a_published_verdict():
    """Deliberately narrower than "anything with pass counts". A per-round QA verdict carries the same
    counts but is an intermediate written every round inside a run dir; the run's final score is what gets
    published and cited. Demanding a block on each round would flag hundreds of historical files and train
    people to bypass the gate — which costs more attribution than it buys."""
    assert cp._claims_a_verdict({"qa_gate": "capsule_bench_v0_pilot", "all_pass": False,
                                 "n_passed": 14, "n_capsules": 29}) is False
