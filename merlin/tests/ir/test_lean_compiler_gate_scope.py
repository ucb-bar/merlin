"""The compile selector admits integrated proofs and excludes draft numerical policy."""

import sys

import pytest

from merlin.verify import bf16_policy_gate as draft_policy
from merlin.verify import gate_checks, gate_worker, pass_validation


def forbidden(*args, **kwargs):
    raise AssertionError("draft policy or proof tooling entered the compiler gate")


def test_all_selects_integrated_proofs_without_policy_admission(tmp_path, monkeypatch):
    # SDK-free selection test: no request is submitted, and fake tools grant no proof.
    monkeypatch.setattr(pass_validation.CheckerTools, "__post_init__", lambda self: None)
    monkeypatch.setattr(draft_policy, "require_admission", forbidden)
    monkeypatch.setattr(draft_policy, "check_policy", forbidden)
    monkeypatch.setenv(pass_validation.ENV_PASSES, "all")
    monkeypatch.setenv(pass_validation.ENV_SDK_BIN, str(tmp_path / "unused-sdk"))
    monkeypatch.setenv(pass_validation.ENV_COMPILER_PYTHON, sys.executable)
    validation = pass_validation.from_environment(tmp_path)
    try:
        assert validation.passes == frozenset(
            {"prune_dead_pure_tensor_ops", "cse", "ordered_loop_unrolling", "ordered_loop_segmentation", "buffer_reuse"}
        )
        assert pass_validation.CHECKS == {
            "prune_dead_pure_tensor_ops": "tensor_refinement",
            "cse": "tensor_refinement",
            "ordered_loop_unrolling": "compiler_loop_refinement",
            "ordered_loop_segmentation": "compiler_loop_refinement",
            "buffer_reuse": "compiler_buffer_refinement",
            "bf16_f32acc_draft_contract": "compiler_macc_contract",
            "lower_quant_ext": "compiler_qdq_refinement",
            "configuration_write_elision": "compiler_configuration_refinement",
            "configuration_loop_hoisting": "compiler_configuration_hoist_refinement",
            "host_graph_fusion": "compiler_host_graph_fusion",
        }
        assert gate_worker.BUILTIN_CHECKS == (
            "tensor_refinement",
            "compiler_loop_refinement",
            "compiler_loop_generic_refinement",
            "compiler_buffer_refinement",
            "compiler_memory_dce_refinement",
            "compiler_macc_contract",
            "compiler_qdq_refinement",
            "compiler_configuration_refinement",
            "compiler_configuration_hoist_refinement",
            "compiler_host_graph_fusion",
        )
        assert not hasattr(gate_checks, "bf16_f32acc_policy")
        with pass_validation.validating(validation):
            calls = []
            for name in ("lower_quant_ext", "preprocess_text_textual", "lower_bf16_matmul_f32acc"):
                assert pass_validation.observe(name, object(), lambda module: calls.append(module)) is None
                assert not pass_validation.submit(name, "before", "after")
            assert len(calls) == 3
        assert validation.gate.verdicts == []
    finally:
        validation.gate.close()


@pytest.mark.parametrize(
    "selection",
    [
        "bf16_f32acc_draft_contract",
        "lower_quant_ext",
        "configuration_write_elision",
        "configuration_loop_hoisting",
        "host_graph_fusion",
    ],
)
def test_explicit_integrated_selectors_do_not_approve_numerical_policy(tmp_path, monkeypatch, selection):
    # Selection alone opens no source-profile recognizer, kernel proof or draft
    # admission. These are real registered gates, deliberately outside `all`.
    monkeypatch.setattr(pass_validation.CheckerTools, "__post_init__", lambda self: None)
    monkeypatch.setattr(pass_validation, "source_admission", forbidden)
    monkeypatch.setattr(draft_policy, "require_admission", forbidden)
    monkeypatch.setattr(draft_policy, "check_policy", forbidden)
    monkeypatch.setenv(pass_validation.ENV_PASSES, selection)
    monkeypatch.setenv(pass_validation.ENV_SDK_BIN, str(tmp_path / "unused-sdk"))
    monkeypatch.setenv(pass_validation.ENV_COMPILER_PYTHON, sys.executable)
    validation = pass_validation.from_environment(tmp_path)
    try:
        assert selection in pass_validation.CHECKS
        assert selection not in pass_validation.DEFAULT_CHECKS
        assert validation.passes == frozenset({selection})
        assert validation.gate.verdicts == []
    finally:
        validation.gate.close()


@pytest.mark.parametrize(
    "selection", ["preprocess_text_textual", "lower_bf16_matmul_f32acc", "w8a8_policy"]
)
def test_unregistered_gate_is_rejected_before_tool_or_policy_access(tmp_path, monkeypatch, selection):
    monkeypatch.setenv(pass_validation.ENV_PASSES, selection)
    monkeypatch.delenv(pass_validation.ENV_SDK_BIN, raising=False)
    monkeypatch.setattr(pass_validation.CheckerTools, "__post_init__", forbidden)
    monkeypatch.setattr(draft_policy, "require_admission", forbidden)
    with pytest.raises(ValueError, match="subset.*cse.*prune_dead_pure_tensor_ops"):
        pass_validation.from_environment(tmp_path)
