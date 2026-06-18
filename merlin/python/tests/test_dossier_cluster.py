"""Kernel dossier composition + clustering (the agent's review-set reduction)."""
from merlin.kernels.types import NormalizedKernel
from merlin.kernels.dossier import build_dossier
from merlin.kernels.cluster import cluster_dossiers, representatives
from merlin.kernels.framework_contracts import load_contract, available_frameworks

_GEMM_1X4 = ("size_t vl = __riscv_vsetvl_e32m4(n);\n"
             "vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4(w, vl); w = w + nr;\n"
             "vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, a0, vb, vl);\n"
             "__riscv_vse32_v_f32m4(c0, vacc0, vl);\n")
_GEMM_1X4_B = _GEMM_1X4.replace("c0", "c1")   # same decisions/struct, different name -> same cluster


def _nk(text, src="xnnpack", op="gemm", dt="f32"):
    return NormalizedKernel(source=src, target="rvv", path=f"{op}_{dt}.c", op=op, dtype=dt,
                            raw_text=text)


def test_framework_contracts_present():
    fw = available_frameworks()
    assert {"xnnpack", "openblas", "saturn"} <= set(fw)
    xc = load_contract("xnnpack")
    assert xc["operand_prepack"]["operand"] == "rhs"     # the prepack/transpose contract
    assert "transpose_assumption" in xc


def test_dossier_composes_all_layers():
    d = build_dossier(_nk(_GEMM_1X4))
    assert d.decisions.get("fma_form") == "vf"
    assert d.struct.get("pointer_advance_prepack") is True
    assert "scalar_broadcast_fma" in d.motifs
    assert d.framework_contract.get("framework") == "xnnpack"   # contract attached by source
    assert d.to_dict()["source"] == "xnnpack"


def test_clustering_groups_equivalent_kernels():
    doss = [build_dossier(_nk(_GEMM_1X4)), build_dossier(_nk(_GEMM_1X4_B)),
            build_dossier(_nk(_GEMM_1X4, op="conv"))]   # different op -> different cluster
    clusters = cluster_dossiers(doss)
    # the two gemm variants collapse to one cluster; conv is its own
    sizes = sorted(len(c.members) for c in clusters)
    assert sizes == [1, 2]
    assert len(representatives(doss)) == len(clusters)
