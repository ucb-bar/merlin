"""R6: merlin-rvv-mine driver — abstraction(policies) -> expert CCA -> divergences -> actions,
minted as a versioned run folder."""
from __future__ import annotations

import yaml

from merlin.mining import mine

_POLICIES = [
    {"policy": "fma_broadcast_policy", "evidence": ["openblas_rvv_gemm", "xnnpack_rvv_gemm"]},
    {"policy": "lmul_grouping_policy", "evidence": ["openblas_rvv_axpy"]},
    {"policy": "int8_widening_policy", "evidence": ["xnnpack_rvv_gemm"]},
    {"policy": "vl_tail_policy", "evidence": ["openblas_rvv_copy"]},
]


def test_expert_cca_maps_policies_with_axis_evidence():
    # `backend` is REQUIRED now. It used to default to one target, which was invisible coupling
    # hiding behind a module path that claimed that ISA: the register-block lookup read a features
    # block keyed by that name and returned empty -- so no divergence -- for every other target.
    cca, ev = mine.expert_cca_from_policies(_POLICIES, op="matmul", backend="rvv")
    assert cca.compute.contraction_form == "fused_fma"
    assert cca.compute.widening is True
    assert cca.vector.lmul == 4.0 and cca.vector.vl_strategy == "vsetvl_loop"
    # per-axis evidence cites the policy that asserts that axis (the GEMM kernels), not an aggregate
    assert ev["compute.contraction_form"] == ["openblas_rvv_gemm", "xnnpack_rvv_gemm"]


def test_mine_run_mints_versioned_folder(tmp_path, monkeypatch):
    mined = tmp_path / "mining_src"
    mined.mkdir()
    (mined / "policy_rules.yaml").write_text(yaml.safe_dump(_POLICIES))
    # fake an "ours" CCA so the run is self-contained (no toolchain/object needed)
    from merlin.kernels import cca as ccamod
    ours = ccamod.CCA(op="matmul", backend=["rvv"],
                      compute=ccamod.ComputeFacet(op="matmul", contraction_form="mul_add",
                                                  widening=False),
                      vector=ccamod.VectorFacet(sew=32, lmul=2.0, vl_strategy="vsetivli_fixed"),
                      provenance={"level": "asm"})
    monkeypatch.setattr(mine, "_our_cca_from_run", lambda *a, **k: (ours, "hand_v0_matmul"))
    out = tmp_path / "out"
    run = mine.mine_run("rvv", "matmul", tmp_path, mined, out)
    man = yaml.safe_load((run / "manifest.yaml").read_text())
    assert run.name.startswith("mining_rvv_v1_")
    assert man["n_divergences"] >= 3 and man["n_actions"] >= 3
    acts = yaml.safe_load((run / "actions.yaml").read_text())
    cf = next(a for a in acts if a["axis"] == "compute.contraction_form")
    assert cf["class"] == "PASS" and "openblas_rvv_gemm" in cf["evidence"]
