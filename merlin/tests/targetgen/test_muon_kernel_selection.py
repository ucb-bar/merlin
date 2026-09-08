"""Compiler-side selection of schedule families mined from the pinned kernel library."""
from __future__ import annotations

from copy import deepcopy

import pytest

from merlin.common.paths import repo_root
from merlin.runtime.backends.base import get_backend


KS = get_backend("muon").muon_kernel_selection
CONTRACT = (
    repo_root()
    / "merlin/experiments/capsule_bench/targets/radiance/contracts/kernel_library_pr1_v1.yaml"
)


def _contract():
    return KS.load_selection_contract(CONTRACT)


def _hardware(*features: str):
    return {"features": list(features)}


def _decision(report, family: str):
    return next(item for item in report.decisions if item.family == family)


def test_every_declared_family_has_one_compiler_rule():
    census = KS.validate_selection_contract(_contract())
    assert census["qualified_families"] == 21
    assert census["experimental_families"] == 2
    assert census["compiler_rules"] == 23
    assert census["missing_rules"] == []
    assert census["selection_is_numeric_qualification"] is False


def test_weight_stationary_wins_but_all_alternatives_remain_auditable():
    report = KS.select_kernel_family(
        {"op": "matmul", "dtype": "mxfp8", "shape": {"M": 256, "K": 2048, "N": 64}},
        _hardware("mx_mesh", "block_e8m0", "shared_memory"),
        _contract(),
    )
    assert report.selected_family == "kernels/gemm_mxgemmini_ws"
    assert _decision(report, "kernels/gemm_mxgemmini").status == "eligible_not_selected"
    assert _decision(report, "kernels/gemm_mxgemmini_ws_restream").status == "eligible_not_selected"
    assert len(report.decisions) == 23
    assert report.to_dict()["selection_is_numeric_qualification"] is False


def test_missing_shared_memory_falls_back_to_the_generic_mesh_strategy():
    report = KS.select_kernel_family(
        {"op": "matmul", "dtype": "mxfp8", "shape": {"M": 256, "K": 2048, "N": 64}},
        _hardware("mx_mesh", "block_e8m0"),
        _contract(),
    )
    assert report.selected_family == "kernels/gemm_mxgemmini"
    ws = _decision(report, "kernels/gemm_mxgemmini_ws")
    assert ws.status == "refused"
    assert "shared_memory" in " ".join(ws.reasons)


def test_shape_specific_batched_strategy_fails_closed_outside_qualified_shape():
    report = KS.select_kernel_family(
        {"op": "batched_matmul", "dtype": "mxfp8", "shape": {"M": 48, "K": 2048, "N": 128}},
        _hardware("mx_mesh", "block_e8m0", "shared_memory"),
        _contract(),
    )
    assert report.selected_family is None
    candidates = [d for d in report.decisions if "gemv_batched_fp8" in d.family]
    assert candidates and all(d.status == "refused" for d in candidates)
    assert any("shape M=48" in reason for d in candidates for reason in d.reasons)


@pytest.mark.parametrize(
    ("semantic_request", "family", "blocked_reason"),
    [
        (
            {"op": "attention_gqa_softcap_window", "dtype": "mxfp8",
             "shape": {"Sq": 64, "Sk": 256, "D": 256, "QH": 8, "KVH": 4}},
            "kernels/flash_attention_mx_gemma",
            "mesh-written output lacks reliable numeric readback",
        ),
        (
            {"op": "quantized_matmul", "dtype": "mxfp6",
             "shape": {"M": 64, "K": 256, "N": 64}},
            "kernels/flash_attention_mx_fp6",
            "final-tile store stalls and does not complete RTL simulation",
        ),
    ],
)
def test_experimental_families_are_unconditionally_fail_closed(
    semantic_request, family, blocked_reason,
):
    report = KS.select_kernel_family(
        semantic_request,
        _hardware("mx_mesh", "block_e8m0", "simt", "shared_memory", "exp", "tanh"),
        _contract(),
    )
    decision = _decision(report, family)
    assert decision.status == "disabled"
    assert blocked_reason in decision.reasons[0]
    assert report.selected_family is None


def test_selection_does_not_observe_capsule_identity_or_expected_output():
    base = {
        "op": "bias_add",
        "dtype": "fp32",
        "shape": {"rows": 32, "cols": 768},
    }
    disguised = dict(base, capsule_name="special_case", expected_output=[123.0])
    a = KS.select_kernel_family(base, _hardware("simt"), _contract()).to_dict()
    b = KS.select_kernel_family(disguised, _hardware("simt"), _contract()).to_dict()
    assert a == b
    assert a["selected_family"] == "kernels/bias_add"


def test_registry_hole_is_a_contract_error_not_silent_missing_coverage():
    contract = deepcopy(_contract())
    contract["compiler_selection"]["rules"].pop()
    with pytest.raises(KS.KernelSelectionContractError, match="without compiler rules"):
        KS.validate_selection_contract(contract)


def test_unknown_shape_dimension_is_a_recorded_refusal():
    report = KS.select_kernel_family(
        {"op": "patch_embed", "dtype": "fp32",
         "shape": {"channels": 3, "patch_h": 16, "patch_w": 16}},
        _hardware("simt", "shared_memory"),
        _contract(),
    )
    assert report.selected_family is None
    decision = _decision(report, "kernels/patch_embed")
    assert decision.status == "refused"
    assert "shape dimension out_channels is unknown" in decision.reasons
