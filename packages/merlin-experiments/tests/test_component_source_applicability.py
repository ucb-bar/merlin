"""Actual structural source analysis; no target or runtime qualification claims."""

import json
from dataclasses import replace

import pytest
from merlin_experiments.phase1 import component_source_applicability as S
from merlin_experiments.phase1.component_witness import REQUIRED_EXECUTION_EFFECTS
from merlin_experiments.phase2.contracts import StageGateError, sha256_file

SOURCE = """
builtin.module {
  func.func @evaluate(%a: tensor<2x3xi32>) -> tensor<2x3xi32> {
    %initial = tensor.empty() : tensor<2x3xi32>
    %result = linalg.copy ins(%a : tensor<2x3xi32>) outs(%initial : tensor<2x3xi32>) -> tensor<2x3xi32>
    func.return %result : tensor<2x3xi32>
  }
}
"""


def _observe(tmp_path, text=SOURCE, frontend="mlir"):
    source = tmp_path / "source.mlir"
    source.write_text(text)
    return S.evaluate_component_source_applicability(
        source=source,
        source_program_sha256=sha256_file(source),
        frontend=frontend,
    )


def test_registered_tensor_value_semantics_derive_source_only_absence(tmp_path):
    observation = _observe(tmp_path)
    observation.verify()
    record = observation.record()
    assert not record["unresolved"]
    assert record["facts"]["static_input_domain"]["status"] == "PASS"
    assert record["facts"]["observable_input_mutation"]["status"] == "N_A"
    assert record["facts"]["observable_input_address_alias"]["status"] == "N_A"
    assert record["operations"][3]["source_effect"] == "structured_tensor_value_update"
    assert record["runtime_effects"] == dict.fromkeys(REQUIRED_EXECUTION_EFFECTS, "UNKNOWN")
    assert record["numerical_finiteness"] == "UNKNOWN"


@pytest.mark.parametrize("dtype", ["f32", "i64"])
def test_rank_zero_tensor_retains_its_tensor_abi_and_unknown_runtime_effects(tmp_path, dtype):
    observation = _observe(tmp_path, SOURCE.replace("2x3xi32", dtype))
    observation.verify()
    record = observation.record()
    assert not record["unresolved"]
    assert record["inputs"] == record["outputs"] == [f"tensor<{dtype}>"]
    assert record["facts"]["static_input_domain"]["status"] == "PASS"
    assert record["runtime_effects"] == dict.fromkeys(REQUIRED_EXECUTION_EFFECTS, "UNKNOWN")
    assert record["numerical_finiteness"] == "UNKNOWN"


@pytest.mark.parametrize(
    "source",
    [
        "module {}",
        SOURCE.replace("tensor<2x3xi32>", "tensor<?x3xi32>"),
        SOURCE.replace("tensor<2x3xi32>", "memref<2x3xi32>"),
        SOURCE.replace("tensor<2x3xi32>", "memref<f32>"),
        SOURCE.replace("linalg.copy", "unknown.unregistered"),
        SOURCE.replace(
            "%initial = tensor.empty() : tensor<2x3xi32>",
            "%initial = func.call @evaluate(%a) : (tensor<2x3xi32>) -> tensor<2x3xi32>",
        ),
    ],
)
def test_unavailable_or_effectful_source_never_receives_not_applicable_credit(tmp_path, source):
    record = _observe(tmp_path, source).record()
    assert record["unresolved"]
    assert {row["status"] for row in record["facts"].values()} == {"UNKNOWN"}
    assert set(record["runtime_effects"].values()) == {"UNKNOWN"}


def test_python_source_cannot_skip_actual_fx_and_import_authority(tmp_path):
    record = _observe(tmp_path, "def forward(value): return value\n", frontend="pytorch").record()
    assert "FX capture/import" in record["unresolved"][0]
    assert {row["status"] for row in record["facts"].values()} == {"UNKNOWN"}


def test_changed_source_or_forged_applicability_requires_actual_rederivation(tmp_path):
    observation = _observe(tmp_path)
    record = json.loads(observation.analysis_json)
    record["runtime_effects"]["alias"] = "PASS"
    with pytest.raises(StageGateError, match="applicability or parser authority changed"):
        replace(observation, analysis_json=json.dumps(record)).verify()
    observation.source.write_text(SOURCE + "\n")
    with pytest.raises(StageGateError, match="admitted source/frontend"):
        observation.verify()


def test_parser_membership_drift_invalidates_even_matching_source_bytes(tmp_path, monkeypatch):
    observation = _observe(tmp_path)
    owner = tmp_path / "new_parser_owner.py"
    owner.write_text("# a newly selected parser dependency\n")
    sources = S._analysis_sources
    monkeypatch.setattr(S, "_analysis_sources", lambda: (*sources(), (owner, sha256_file(owner))))
    with pytest.raises(StageGateError, match="parser authority changed"):
        observation.verify()
