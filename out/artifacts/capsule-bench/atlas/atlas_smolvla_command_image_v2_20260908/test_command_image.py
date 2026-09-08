"""Focused integration regressions for the Atlas command-image recovery."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
sys.path.insert(0, str(REPO / "merlin/python"))
sys.path.insert(0, str(ROOT / "submission"))

from mlir_oot.cmdbuf import build_command_buffer  # noqa: E402
from mlir_oot.codegen import emit_program  # noqa: E402
from mlir_oot.frontend import parse_verified  # noqa: E402


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def test_commit_result_is_a_real_consumer_operand() -> None:
    interface = ROOT / "cases/chained/two_matmuls.mlir"
    workload = parse_verified(interface.read_text())
    matmuls = [op for op in workload.ops if op["op"] == "matmul"]
    assert len(matmuls) == 2
    assert matmuls[1]["lhs"] == "Y0"
    cb = build_command_buffer(workload)
    assert cb["commands"][4]["operands"]["lhs"] == "Y0"
    assert cb["kernel_abi"]["outputs"] == ["Y0", "Y1"]
    bases = [spec["base"] for spec in cb["tensors"].values()]
    assert len(bases) == len(set(bases))
    assert sum(line.lstrip().startswith(".word") for line in emit_program(workload).splitlines()) == 1882


def test_saved_elaborated_rtl_results_are_exact() -> None:
    expected = {
        "bf16_movement_single": (391, 1075),
        "bf16_movements": (671, 1877),
        "independent": (2359, 6010),
        "chained": (1882, 7314),
        "smolvla_tail_50_720_32": (12605, 300068),
        "smolvla_state_proj_1_32_960": (1509, 154458),
    }
    for case, (words, cycles) in expected.items():
        result = _load(f"cases/{case}/gsim_result.json")
        assert result["oracle"]["derived_from_rtl"] is True
        assert result["oracle"]["fidelity"] == "elaborated_rtl"
        assert result["instruction_words"] == words
        assert result["cycles"] == cycles
        assert result["all_outputs_bit_exact"] is True
        assert all(value["mismatches"] == 0 for value in result["comparisons"].values())


def test_raw_smolvla_readback_has_no_expected_payload() -> None:
    spec_path = ROOT / "cases/smolvla_tail_50_720_32/raw_gsim_spec.json"
    spec = json.loads(spec_path.read_text())
    assert set(spec) == {"words", "preload", "reads", "max_cycles"}
    assert len(spec["words"]) == 12605
    raw = _load("cases/smolvla_tail_50_720_32/raw_readback.json")
    output = (ROOT / "cases/smolvla_tail_50_720_32/raw_Y0.bf16.bin").read_bytes()
    assert len(output) == 3200
    assert hashlib.sha256(output).hexdigest() == raw["comparisons"]["Y0"]["raw_sha256"]
    assert raw["all_outputs_bit_exact"] is True
    assert raw["cycles"] == 300068


def test_oracle_controls_detect_instrument_mismatch_and_no_echo() -> None:
    functional = _load("cases/bf16_movement_single/functional_result.json")
    rtl = _load("cases/bf16_movement_single/gsim_result.json")
    assert functional["kernel_sha256"] == rtl["kernel_sha256"]
    assert functional["all_outputs_bit_exact"] is False
    assert rtl["all_outputs_bit_exact"] is True
    func_raw = _load("cases/bf16_movement_single/functional_run/func_out.json")
    unsupported = [item["word"] for item in func_raw["unsupported"]]
    assert unsupported == [163967, 168063, 172159, 176255, 180351, 184447, 188543, 192639]
    assert all("opcode 0x7f" in item["reason"] for item in func_raw["unsupported"])
    negative = _load("cases/bf16_movements/halt_control_result.json")
    assert negative["control_failed_as_expected"] is True
    assert all(value["nonzero_readback"] == 0 for value in negative["comparisons"].values())


def test_declined_whole_model_cannot_emit_a_trivial_image() -> None:
    interface = """builtin.module attributes {prov.weights_file = "weights.safetensors"} {
  func.func @forward(%x: tensor<1x1xf32>) -> tensor<1x1xf32> {
    %z = tensor.empty() : tensor<1x1xf32>
    %y = linalg.matmul {prov.op = "matmul"} ins(%x, %x : tensor<1x1xf32>, tensor<1x1xf32>) outs(%z : tensor<1x1xf32>) -> tensor<1x1xf32>
    return %y : tensor<1x1xf32>
  }
}
"""
    workload = parse_verified(interface)
    assert workload.ops[0]["op"] == "model"
    with pytest.raises(ValueError, match="declined model workload"):
        emit_program(workload)


if __name__ == "__main__":
    test_commit_result_is_a_real_consumer_operand()
    test_saved_elaborated_rtl_results_are_exact()
    test_raw_smolvla_readback_has_no_expected_payload()
    test_oracle_controls_detect_instrument_mismatch_and_no_echo()
    print("ok: command chain, real-shape RTL readback, and oracle controls")
