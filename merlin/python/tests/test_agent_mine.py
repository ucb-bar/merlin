"""Dual-mode agent mining harness (mock LLM, deterministic): representative vs per_kernel + the
mode comparison. Real LLM runs (merlin.common.llm.complete) need ANTHROPIC_API_KEY; the harness
degrades to finding={} without it, so this suite uses an injected mock."""
from merlin.kernels.types import NormalizedKernel
from merlin.kernels.dossier import build_dossier
from merlin.kernels import agent_mine

_G_VF = ("size_t vl=__riscv_vsetvl_e32m4(n);\n"
         "v=__riscv_vfmacc_vf_f32m4(v,a,b,vl);\n__riscv_vse32_v_f32m4(c,v,vl);\n")
_G_NOFMA = ("size_t vl=__riscv_vsetvl_e32m1(n);\n"
            "x=__riscv_vfmul_vv_f32m1(a,b,vl);\nx=__riscv_vfadd_vv_f32m1(x,c,vl);\n")


def _nk(t, p):
    return NormalizedKernel(source="xnnpack", target="rvv", path=p, op="gemm", dtype="f32",
                            raw_text=t)


def _mock_llm(prompt):
    ex = "true" if '"fma_form": "vf"' in prompt else "false"
    return ('{"algorithm":"gemm","is_exemplary":' + ex +
            ',"compiler_levers":["use vfmacc.vf","e32m4 LMUL"],'
            '"contract_refinements":[],"caveats":[]}')


def _doss():
    return [build_dossier(_nk(_G_VF, "a.c")), build_dossier(_nk(_G_VF, "b.c")),
            build_dossier(_nk(_G_NOFMA, "c.c"))]


def test_build_prompt_includes_dossier_facts():
    p = agent_mine.build_prompt(build_dossier(_nk(_G_VF, "a.c")))
    assert '"fma_form": "vf"' in p and "JSON" in p


def test_representative_mode_clusters_calls():
    out = agent_mine.mine(_doss(), mode="representative", llm_fn=_mock_llm)
    # two identical vf gemms collapse to one cluster -> 2 calls, all 3 kernels covered
    assert out["n_calls"] == 2 and out["n_kernels_covered"] == 3
    assert out["findings"][0]["finding"]["compiler_levers"]


def test_per_kernel_mode_calls_each():
    out = agent_mine.mine(_doss(), mode="per_kernel", llm_fn=_mock_llm)
    assert out["n_calls"] == 3 and out["n_kernels_covered"] == 3


def test_compare_modes_measures_cost_and_agreement():
    cmp = agent_mine.compare_modes(_doss(), llm_fn=_mock_llm)
    assert cmp["representative"]["n_calls"] == 2
    assert cmp["per_kernel"]["n_calls"] == 3
    assert cmp["call_ratio"] == 1.5
    assert cmp["exemplary_agreement"] == 1.0   # rep finding matched per-kernel on all overlaps


def test_parse_findings_tolerant():
    assert agent_mine.parse_findings('```json\n{"is_exemplary": true}\n```')["is_exemplary"] is True
    assert agent_mine.parse_findings(None) == {}
    assert agent_mine.parse_findings("no json here") == {}
