"""Shape COVERAGE is answerable without a golden, and that is why this probe exists.

The numerical generalization difftest needs a CPU reference for the operand format, so it cannot run on
an MX/fp8 datapath -- and those are exactly the targets whose shape coverage is most in doubt. Measured
while building this: all four multi-tile probes on an fp8 target were skipped for want of a golden and
the suite reported ``0 graded``, which reads as "nothing to report" rather than "could not look".

The instrument here is a structural invariant instead: **a program that computes a larger problem cannot
be smaller than the one that computes the smaller problem.** It needs no ISA knowledge, no oracle and no
reference values -- only the emitted artifact. A backend that silently declines a shape emits its
terminator and nothing else, so its artifact COLLAPSES, and that is visible for the cost of one call to
the emit path.

Measured on the frozen submission this was built for: 418 instruction words at one tile, 5 at two
M-tiles (a bare ECALL), 1187 at two K-tiles, 1205 at two N-tiles -- the same M-versus-K/N boundary a
post-freeze holdout took a paid run to find. The control target lowers all four and its work grows
monotonically.
"""
from __future__ import annotations

import pytest

from merlin.targetgen import lowering_coverage as LC


# --------------------------------------------------------------- the interface it probes with

def test_the_probe_interface_is_the_capsule_shape_with_different_extents():
    """Only the extents differ from a real corpus capsule -- otherwise this is a new op, not a probe."""
    mlir = LC.contraction_interface(64, 32, 32, target="t", operand_mlir="f8E4M3FN", accum_mlir="bf16")
    assert "tensor<64x32xf8E4M3FN>" in mlir, "A0 carries M x K"
    assert "tensor<32x32xf8E4M3FN>" in mlir, "W carries K x N"
    assert "merlin_iface.acc<bf16>" in mlir
    assert "tensor<64x32xbf16>" in mlir, "the commit carries M x N"
    assert 'merlin_iface.target = "t"' in mlir


def test_every_corner_is_a_multiple_of_the_tile_edge():
    """The corners are ratios, resolved against the target's DERIVED tile -- never absolute literals."""
    assert LC.CORNERS["tile"] == (1, 1, 1)
    assert LC.CORNERS["m_2tiles"] == (2, 1, 1)
    assert LC.CORNERS["k_2tiles"] == (1, 2, 1)
    assert LC.CORNERS["n_2tiles"] == (1, 1, 2)


# --------------------------------------------------------------- the invariant

def _sweep(monkeypatch, work_by_corner, declined=()):
    """Drive sweep() with a fake emit path so the invariant is tested, not a real backend."""
    monkeypatch.setattr(LC, "_binding", lambda t: type(
        "B", (), {"operand_dtype": "int8", "accum_dtype": "int32",
                  "mlir_dtype": staticmethod(lambda tok: {"int8": "i8"}.get(tok, "i32"))})())
    monkeypatch.setattr(LC, "tile_edge", lambda t: 32)

    def fake(package, *, target, m, k, n, operand_mlir, accum_mlir, contract=None, timeout=300):
        corner = next(c for c, (fm, fk, fn) in LC.CORNERS.items()
                      if (32 * fm, 32 * fk, 32 * fn) == (m, k, n))
        if corner in declined:
            return "declined", "no loop over this axis", 0
        return "lowered", None, work_by_corner[corner]

    monkeypatch.setattr(LC, "probe_shape", fake)
    return LC.sweep("pkg", target="t")


def test_a_program_that_shrinks_on_a_bigger_problem_is_a_silent_refusal(monkeypatch):
    """The measured atlas shape: 418 words at one tile, 5 at two M-tiles."""
    r = _sweep(monkeypatch, {"tile": 418, "m_2tiles": 5, "k_2tiles": 1187, "n_2tiles": 1205})
    by = {c["corner"]: c["outcome"] for c in r["corners"]}
    assert by["m_2tiles"] == "collapsed"
    assert by["k_2tiles"] == by["n_2tiles"] == "lowered"
    assert r["multi_tile_axes_uncovered"] == ["m"], "names the AXIS, which is the actionable part"
    assert r["all_covered"] is False
    assert "cannot compute more by doing less" in next(
        c["detail"] for c in r["corners"] if c["corner"] == "m_2tiles")


def test_work_that_grows_on_every_axis_is_covered(monkeypatch):
    """The measured gemmini shape: 29 words at one tile, 37/37/40 at two."""
    r = _sweep(monkeypatch, {"tile": 29, "m_2tiles": 37, "k_2tiles": 37, "n_2tiles": 40})
    assert r["all_covered"] is True
    assert r["multi_tile_axes_uncovered"] == []
    assert r["n_collapsed"] == 0


def test_a_stated_decline_is_uncovered_but_not_a_collapse(monkeypatch):
    """Declining is the HONEST way to not cover a shape. Still uncovered; no longer silent."""
    r = _sweep(monkeypatch, {"tile": 418, "m_2tiles": 0, "k_2tiles": 900, "n_2tiles": 900},
               declined=("m_2tiles",))
    by = {c["corner"]: c["outcome"] for c in r["corners"]}
    assert by["m_2tiles"] == "declined"
    assert r["n_declined"] == 1 and r["n_collapsed"] == 0
    assert r["multi_tile_axes_uncovered"] == ["m"], "honest, but still not covered"


def test_equal_work_is_not_flagged(monkeypatch):
    """The invariant is 'must not SHRINK', deliberately not 'must grow'.

    A backend may legitimately emit a loop whose text does not grow with the trip count. Flagging that
    would make the probe produce false accusations, which is worse than missing a case.
    """
    r = _sweep(monkeypatch, {"tile": 100, "m_2tiles": 100, "k_2tiles": 100, "n_2tiles": 100})
    assert r["all_covered"] is True
    assert r["n_collapsed"] == 0


def test_a_failing_baseline_refuses_to_attribute_anything_to_shape(monkeypatch):
    """With the one-tile baseline down, every multi-tile corner fails for reasons unrelated to shape.

    Reporting "M, K and N all uncovered" there would be a confident, specific, wrong answer -- which is
    worse than no answer. Measured: substituting a gradeable operand dtype moved the probe onto a
    different lowering path with a different tile edge and produced exactly that false reading.
    """
    r = _sweep(monkeypatch, {"tile": 0, "m_2tiles": 0, "k_2tiles": 0, "n_2tiles": 0},
               declined=("tile", "m_2tiles", "k_2tiles", "n_2tiles"))
    assert r["baseline_tile_lowered"] is False
    assert r["multi_tile_axes_uncovered"] == []
    assert r["all_covered"] is False
    assert "fix the baseline" in r["unmeasured"]


# --------------------------------------------------------------- end to end, on the real submissions

@pytest.mark.parametrize("target,pkg,expected_axes", [
    ("atlas", "out/runs/atlas/capsule-bench/merlin_assisted/merlincirct_atlas_arm4_v1/submission", ["m"]),
    ("gemmini",
     "out/runs/gemmini/capsule-bench/merlin_assisted/merlincirct_gemarm4_codex/submission", []),
])
def test_the_frozen_submissions_reproduce_their_measured_holdout_boundary(target, pkg, expected_axes):
    """The whole point, end to end: this finds -- with no holdout, no golden and no oracle -- the same
    boundary that previously took a post-freeze holdout on a paid run."""
    from merlin.common.paths import repo_root
    p = repo_root() / pkg
    if not (p / "manifest.yaml").is_file():
        pytest.skip(f"frozen submission not present: {pkg}")
    import os
    os.environ["MERLIN_TARGET_EXPERIMENT"] = str(
        repo_root() / f"merlin/experiments/capsule_bench/targets/{target}/target_experiment.yaml")
    cov = LC.sweep(p, target=target, contract=str(repo_root() / "merlin/contract"))
    assert cov["baseline_tile_lowered"] is True, "the one-tile baseline must lower for this to mean anything"
    assert cov["multi_tile_axes_uncovered"] == expected_axes
