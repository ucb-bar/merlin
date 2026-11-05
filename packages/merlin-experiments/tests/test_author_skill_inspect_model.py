"""The inspect-model skill reads only staged public files and ranks a model's operator cases by work."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from inspect_skill_support import check_skill_layout, load_script, skill_dir, source

from merlin_experiments.phase1 import workspace_transport as WT

SKILL = "inspect-model"
LAYER = """module attributes {merlin_iface.version = "0.1", merlin_iface.target = "t", merlin_iface.abi_version = "0.1"} {
  %A = merlin_iface.tensor {name = "A", role = "input"} : tensor<12x48xi8>
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<48x64xi8>
  %R = merlin_iface.resident_pack %W {layout = "row_major"} : (tensor<48x64xi8>) -> !merlin_iface.resident
  %C = merlin_iface.matmul %A, %R : (tensor<12x48xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %Y0 = merlin_iface.commit %C {name = "Y0", epilogue = [], output_dtype = "i32"} : (!merlin_iface.acc<i32>) -> tensor<12x64xi32>
}
"""


def _layer(ws, name, *, k, interface=LAYER):
    root = ws / "layers" / name
    root.mkdir(parents=True)
    (root / "capsule.interface.mlir").write_text(interface)
    (root / "capsule.yaml").write_text(yaml.safe_dump({
        "name": name, "kind": "layer", "label": "public", "interface_mlir": "capsule.interface.mlir",
        "inputs": [{"name": "A", "role": "input", "shape": [12, 48], "dtype": "i8"}],
        "operation": {"op": "matmul", "attributes": {"lhs": "A", "weight": "W", "out": "Y0", "epilogue": [],
                                                     "output_dtype": "i32"}},
        "integer_partial_sum_bound": {"reduction_extent": k}, "semantic": {"semantic_family": "contraction"}}))


def _model(ws):
    root = ws / "model" / "M"
    root.mkdir(parents=True)
    (root / "capsule.interface.mlir").write_text("module {}\n")
    (root / "frontend-trace.json").write_text(json.dumps({"graphs": {"quantized": {"by_target": {"aten.x": 3}}}}))
    (root / "capsule.yaml").write_text(yaml.safe_dump({
        "name": "M", "kind": "model", "label": "public", "interface_mlir": "capsule.interface.mlir",
        "inputs": [{"name": "I0", "role": "input", "shape": [1, 3, 8, 8], "dtype": "f32"}],
        "operation": {"op": "model", "attributes": {
            "model": "m", "quant_scheme": "s", "arg_order": ["I0", "Y0"], "weights": "w.safetensors",
            "integerization_receipt": {"schema": "r", "qparams": [
                {"activation": {"granularity": "tensor"}, "weight": {"granularity": "channel"},
                 "accumulator_dtype": "int32"}], "precision_decisions": [], "remaining_dequant_count": 0}}},
        "numeric_policy": {"compare": "tolerance_float"}, "required_oracle_tiers": ["L0"],
        "model_qualification": {"status": "qualified", "device_groups": 3, "covered_groups": 3,
                                "group_capsules": [
                                    {"groups": [0], "count": 1, "capsule": "layers/small", "golden_sha256": "f" * 64},
                                    {"groups": [1, 2], "count": 2, "capsule": "layers/big", "golden_sha256": "e" * 64},
                                    {"groups": [3], "count": 1, "capsule": "layers/absent"}]},
        "software_screen": {"status": "admitted", "placements": {"accelerator": 3, "host": 2}, "decisions": [
            {"operation": "mm", "count": 1, "classification": "accelerator_candidate", "placement": "accelerator"},
            {"operation": "mm", "count": 2, "classification": "accelerator_candidate", "placement": "accelerator"},
            {"operation": "relu", "count": 2, "classification": "host_required", "placement": "host"}]}}))
    return root


def test_layout_rules(tmp_path):
    check_skill_layout(SKILL, tmp_path)


def test_summary_ranks_operator_cases_and_withholds_answers(tmp_path):
    script = load_script(SKILL, "model_summary")
    ws = tmp_path / "ws"
    _layer(ws, "small", k=8)
    _layer(ws, "big", k=48)
    report = script.summarize(_model(ws), ws)
    cases = report["operator_cases"]
    assert [c["capsule"] for c in cases] == ["layers/big", "layers/small", "layers/absent"]
    assert cases[0]["output"] == ["Y0", [12, 64], "i32"] and cases[0]["macs"] == 12 * 64 * 48
    assert cases[0]["total_macs"] == 2 * 12 * 64 * 48 and cases[0]["share"] == round(2 * 48 / (2 * 48 + 8), 4)
    assert cases[2]["note"].startswith("operator case not staged")
    assert "golden_sha256" not in json.dumps(report) and report["withheld"] == ["w.safetensors"]
    assert report["placement"]["by_operation"][0] == {"placement": "accelerator", "operation": "mm",
                                                      "classification": "accelerator_candidate", "count": 3}
    assert report["quantization"]["weight_granularity"] == {"channel": 1}
    assert report["frontend_ops"] == {"quantized": {"aten.x": 3}}
    assert script.main([str(ws / "model" / "M"), "--ws", str(ws)]) == 0
    single = script.summarize(ws / "layers" / "big", ws)
    assert single["work"]["macs"] == 12 * 64 * 48 and single["work"]["reduction_extent"] == 48


def test_output_shape_reader_handles_dynamic_types():
    script = load_script(SKILL, "model_summary")
    assert script.tensor_dims("tensor<9x24xi32>") == [9, 24]
    assert script.tensor_dims("tensor<?x24xi32>") is None
    assert script.output_dims(LAYER, "Y0") == [12, 64] and script.output_dims(LAYER, "Z") is None


def test_visibility_statements_match_the_transport_filter():
    for name in ("golden.npz", "x.golden.yaml", "expected_command_buffer_0.json", "expected_instruction_coverage.yaml",
                 "capsule.weights.safetensors"):
        assert WT._is_answer_file(Path(name)), name
    text = (skill_dir(SKILL) / "SKILL.md").read_text()
    assert "Goldens, expected command buffers, expected coverage and weights are" in text
    selfcheck = source("packages/merlin-experiments/src/merlin_experiments/phase1/tools/selfcheck.py")
    model_check = source("packages/merlin-experiments/src/merlin_experiments/phase1/tools/model_check.py")
    for flag in ('"--offload-census"', '"--attach"', '"--wait-budget"'):
        assert flag in selfcheck, flag
    for token in ('"check", "replay-kernel", "wait"', '"--group"', '"--job-id"', '"--full-json"', '"--engine"'):
        assert token in model_check, token
    census = source("src/merlin/targetgen/offload_census.py")
    for field in ("routed_macs", "contraction_offload_fraction", "host_compute_vetoed", "dominant_block", "outcome"):
        assert f'"{field}"' in census, field
    gate = source("packages/merlin-experiments/src/merlin_experiments/phase1/feedback/visible_model_gate.py")
    for field in ("accelerator_coverage", "host_reason_counts", "eligible_but_host", "host_time_cycles"):
        assert field in gate, field
