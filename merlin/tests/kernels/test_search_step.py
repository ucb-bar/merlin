"""WS-C C4: the per-step search record (audit wire + real-vs-fake speedup, the beam's step instrument)."""
from __future__ import annotations

from merlin.kernels import cca, search_step
from merlin.kernels.action_catalog import route
from merlin.kernels.cca_compare import Divergence


def _action():
    return route(Divergence("compute.contraction_form", "fused_fma", "mul_add", "rvv"))


def _cca(contraction):
    return cca.CCA(op="matmul", backend=["rvv"],
                   compute=cca.ComputeFacet(op="matmul", contraction_form=contraction))


def test_achieved_fork_closes_the_divergence_and_credits_real_speedup():
    s = search_step.make_step(_action(), _cca("fused_fma"), correctness_ok=True, speedup=7.9)
    assert s.achieved is True and s.residual == []           # the intended facet was delivered
    assert s.category == "instruction-selection"
    assert s.correctness_ok and s.speedup == 7.9             # real speedup credited


def test_unachieved_fork_leaves_a_residual_to_escalate():
    # emitted asm still mul_add -> the promise (fused_fma) was NOT kept -> residual (audit caught it)
    s = search_step.make_step(_action(), _cca("mul_add"), correctness_ok=True, speedup=1.1)
    assert s.achieved is False
    assert s.residual == ["compute.contraction_form"]


def test_real_vs_fake_no_speedup_credit_when_numerics_fail():
    # a fork that broke numerics gets NO speedup credit even if a number was measured (fail-closed)
    s = search_step.make_step(_action(), _cca("fused_fma"), correctness_ok=False, speedup=9.0)
    assert s.correctness_ok is False
    assert s.speedup is None                                 # fake speedup rejected
    assert "FAILED-numerics" in s.to_line()


def test_audit_fork_lifts_cca_from_objdump_text_and_records():
    # the beam-side entry: audit a certified fork from its emitted objdump text (no toolchain re-run)
    from pathlib import Path

    from merlin.common.paths import merlin_dir
    text = (merlin_dir() / "tests" / "data" / "cca_asm" / "openblas_sgemm_rvv.objdump").read_text()
    s = search_step.audit_fork(_action(), text, op="matmul", correctness_ok=True, speedup=7.9)
    assert s.achieved is True and s.residual == []      # openblas emits fused_fma -> facet achieved
    assert s.category == "instruction-selection" and s.speedup == 7.9


def test_to_dict_is_instrumentable():
    s = search_step.make_step(_action(), _cca("fused_fma"), correctness_ok=True, speedup=2.0)
    d = s.to_dict()
    assert set(d) >= {"axis", "category", "action_class", "achieved", "residual",
                      "correctness_ok", "speedup", "rationale"}
