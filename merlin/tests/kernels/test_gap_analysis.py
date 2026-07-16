"""WS-C C4: the completeness critic — 'did the CCA capture EVERYTHING that makes the expert faster?'"""
from __future__ import annotations

from merlin.kernels import cca
from merlin.kernels.gap_analysis import gap_analysis


def _cca(contraction="fused_fma", resident=True):
    return cca.CCA(op="matmul", backend=["rvv"],
                   compute=cca.ComputeFacet(op="matmul", contraction_form=contraction,
                                            accumulator_resident=resident))


def test_unexplained_gap_flags_cca_incomplete():
    # the '72% slower' signal: our CCA == expert (all divergences closed) but still much slower ->
    # the CCA missed a performance-determining decision -> EXPAND it, don't keep tuning.
    r = gap_analysis(_cca(), _cca(), ours_perf=172.0, expert_perf=100.0)
    assert r.unexplained_gap is True and r.explained is False
    assert r.open_divergences == []
    assert round(r.pct_slower) == 72
    assert "CCA INCOMPLETE" in r.verdict and "EXPAND" in r.verdict


def test_explained_gap_points_to_open_divergences():
    # ours still mul_add (an open divergence) + slower -> the gap is (partly) explained; close it next
    r = gap_analysis(_cca(), _cca(contraction="mul_add"), ours_perf=172.0, expert_perf=100.0)
    assert r.explained is True and r.unexplained_gap is False
    assert "compute.contraction_form" in r.open_divergences


def test_parity_means_the_cca_explained_the_expert():
    r = gap_analysis(_cca(), _cca(), ours_perf=101.0, expert_perf=100.0)   # within tol
    assert r.unexplained_gap is False and r.explained is False
    assert "parity" in r.verdict


def test_no_perf_measured_is_honest():
    r = gap_analysis(_cca(), _cca(contraction="mul_add"))   # no perf numbers
    assert r.attainment is None and r.unexplained_gap is False
    assert "compute.contraction_form" in r.open_divergences
