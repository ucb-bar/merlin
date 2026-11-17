"""Finite operator resource selection stays separate from source shape facts."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from test_execution_model_correspondence import selection as selection
from test_execution_model_staging import capture as capture

from merlin.frontends.compile_inputs import stage_compile_inputs
from merlin.frontends.execution_result_signature import checked_allocation_policy, select_signature, validate_signature
from merlin.frontends.linalg_mlir import parse_mlir_text

POLICY = {
    "schema": "merlin.owned-result-limits.v2",
    "max_bytes": 64 << 20,
    "max_allocations": 16384,
    "max_output_bytes": 64 << 20,
    "index_bits": 64,
}


def test_explicit_work_budget_stages_from_source_without_opening_evaluators(capture, tmp_path, monkeypatch):
    root, receipt = capture
    receipt["execution_allocation_policy"] = deepcopy(POLICY)
    (root / "capture_receipt.json").write_text(json.dumps(receipt))
    original = Path.read_bytes

    def guarded(path):
        if path.name in ("inputs.npz", "golden.npy", "goldens.npz", "tensor_words.npz", "execution.json", "meta.json"):
            raise AssertionError("resource selection opened evaluator data")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    target = tmp_path / "compiler"
    stage_compile_inputs(root, target, result_mode="execution-owned")
    signature = json.loads((target / "signature.json").read_text())
    assert signature["allocation_policy"] == POLICY
    assert signature["outputs"] == [{"kind": "tensor", "dtype": "i32", "shape": [4]}]
    assert validate_signature(parse_mlir_text((target / "program.mlir").read_text()), signature) == signature


@pytest.mark.parametrize(
    "damage", ("zero", "negative", "bool", "float", "huge", "index", "missing", "extra", "unknown", "legacy", "null")
)
def test_malformed_or_unselected_resources_cannot_publish_a_compiler_stage(capture, tmp_path, damage):
    root, receipt = capture
    value = deepcopy(POLICY)
    if damage in ("zero", "negative", "bool", "float", "huge"):
        value["max_allocations"] = {"zero": 0, "negative": -1, "bool": True, "float": 16384.0, "huge": 1 << 63}[damage]
    elif damage == "index":
        value["index_bits"] = 32
    elif damage == "missing":
        value.pop("max_output_bytes")
    elif damage == "extra":
        value["skip_checks"] = True
    elif damage == "unknown":
        value["schema"] = "unchecked"
    elif damage == "legacy":
        value = checked_allocation_policy()
    else:
        value = None
    receipt["execution_allocation_policy"] = value
    (root / "capture_receipt.json").write_text(json.dumps(receipt))
    target = tmp_path / "refused"
    with pytest.raises(ValueError):
        stage_compile_inputs(root, target, result_mode="execution-owned")
    assert not target.exists()


def test_original_resource_profile_and_result_mode_keep_their_meaning(selection):
    parent, _bound, _trace, _interfaces = selection
    original = {"entry": "forward", "inputs": [{"shape": [4], "dtype": "i32"}]}
    signature = select_signature(parent, original, "execution-owned")
    assert signature["allocation_policy"]["max_allocations"] == 4096
    assert signature["allocation_policy"]["schema"] == "merlin.owned-result-limits.v1"
    assert validate_signature(parent, signature) == signature
    forged = deepcopy(signature)
    forged["allocation_policy"]["max_allocations"] = 16384
    with pytest.raises(ValueError, match="fixed limits"):
        validate_signature(parent, forged)
    with pytest.raises(ValueError, match="explicit execution-owned"):
        select_signature(parent, original, None, allocation_policy=POLICY)


@pytest.mark.parametrize("field", ("max_bytes", "max_allocations", "max_output_bytes"))
def test_each_resource_bound_is_independent_positive_and_exact(field):
    selected = {**POLICY, field: (1 << 63) - 1}
    assert checked_allocation_policy(selected) == selected
    for invalid in (0, -1, True, 1.0, 1 << 63):
        with pytest.raises(ValueError):
            checked_allocation_policy({**POLICY, field: invalid})
