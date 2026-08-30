"""A plateau that happens INSIDE a round has to be detectable inside that round.

`--plateau-rounds N` counts consecutive ROUNDS with no progress, so it needs at least N+1 rounds to
fire. A wall-capped continuous schedule produces ONE round -- the wall is only checked between rounds --
so the terminator could not fire however stuck the run was. Measured on a four-arm ladder that passed
`--plateau-rounds 3`: every arm reached its final score in the first ~50 minutes of a 90-136 minute
round; one then spent 48 further minutes and 13 further self-checks without moving, ~28% of its wall
asleep polling for verdicts that never changed. Nothing was watching.

These pin the shared detector's semantics -- especially the cases where it must NOT fire, since a
plateau stop that triggers early destroys a productive run.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]
                       / "experiments/capsule_bench/harness"))

import plateau as PL  # noqa: E402


def v(n_passed, mismatches=(), *, n_capsules=38, **kw):
    """A verdict with `n_passed` passes and one non-passing capsule per entry in `mismatches`."""
    rows = [{"capsule": f"P{i}", "status": "pass"} for i in range(n_passed)]
    rows += [{"capsule": f"F{i}", "status": "fail", "mismatch_count": m}
             for i, m in enumerate(mismatches)]
    return {"n_passed": n_passed, "n_capsules": n_capsules, "per_capsule": rows, **kw}


def test_a_flat_pass_count_that_is_still_reducing_mismatch_is_progress():
    """The mismatch-aware key is what keeps a productive run alive; without it this stops at 33."""
    d = PL.Detector(3)
    assert not d.observe(v(33, [900]))
    assert not d.observe(v(33, [400]))
    assert not d.observe(v(33, [80]))
    assert not d.observe(v(33, [12]))
    assert d.stall == 0, "every step reduced the residual — that is progress, not a plateau"


def test_it_fires_after_n_consecutive_dead_checks():
    d = PL.Detector(3)
    assert not d.observe(v(33, [12]))
    assert not d.observe(v(33, [12]))
    assert not d.observe(v(33, [12]))
    assert d.observe(v(33, [12])) is True
    assert "no progress" in d.why()


def test_a_regression_does_not_reset_the_counter():
    """Going BACKWARDS is not progress. An arm was observed dropping 32 -> 30 -> 32 while stuck."""
    d = PL.Detector(2)
    d.observe(v(32, [10]))
    assert not d.observe(v(30, [90]))
    assert d.observe(v(32, [10])) is True


def test_subset_checks_are_never_counted():
    """A subset check has a different denominator; mixing them invents or hides a plateau."""
    d = PL.Detector(2)
    for _ in range(9):
        assert not d.observe(v(3, [1], n_capsules=4), capsules_arg="R0_gemm_fp32")
    assert d.stall == 0


def test_a_broken_build_cannot_end_a_round():
    """A transient degenerate verdict is not evidence of a plateau in either direction."""
    d = PL.Detector(2)
    for _ in range(9):
        assert not d.observe(v(0, n_capsules=38, build_failed=True))
        assert not d.observe(v(0, n_capsules=38, no_results=True))
    assert d.stall == 0


def test_convergence_is_the_other_terminator_and_never_reads_as_a_stall():
    d = PL.Detector(1)
    for _ in range(5):
        assert not d.observe(v(38, all_pass=True))


def test_disabled_by_default_keeps_prior_behaviour_byte_for_byte():
    d = PL.Detector(0)
    for _ in range(50):
        assert not d.observe(v(1, [5]))
    assert not d.stalled()


def test_the_round_loop_and_the_broker_share_one_definition_of_progress():
    """Two detectors that disagree is how one ends up watching the wrong signal."""
    import run_baseline_qa_loop  # noqa: F401  -- import must not fail after the dedupe
    assert PL.progress_key(v(33, [12])) == (33, -12)
    # per_capsule rows arrive spelled both ways depending on which side produced the verdict
    assert PL.progress_key({"n_passed": 2, "per_capsule": [{"pass": True}, {"pass": True}]}) == (2, 0)
