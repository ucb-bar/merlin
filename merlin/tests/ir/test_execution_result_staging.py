"""Executed result signatures preserve source types and evaluator exclusion."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from test_compile_inputs import _write_capture
from test_owned_result_execution import COMPACT
from xdsl.dialects.builtin import StringAttr

from merlin.frontends.compile_inputs import stage_compile_inputs
from merlin.frontends.execution_result_signature import (
    MODE,
    PROFILE,
    program_text,
    select_signature,
    validate_signature,
)
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.execution_result_inputs import resolve


def capture(path):
    _write_capture(path)
    module = parse_mlir_text(COMPACT)
    module.attributes["prov.weights_file"] = StringAttr("origin/weights.safetensors")
    source = program_text(module, MODE)
    (path / "model.mlir").write_text(source)
    trace = {"status": "complete", "mlir": {"sha256": hashlib.sha256(source.encode()).hexdigest()}}
    (path / "frontend-trace.json").write_text(json.dumps(trace))
    meta = json.loads((path / "meta.json").read_text())
    meta.update(
        input_abi=[{"dtype": "i32", "shape": [None]}],
        output_abi=[{"dtype": "i32", "shape": [999]}] * 3,
        frontend_trace={"sha256": hashlib.sha256((path / "frontend-trace.json").read_bytes()).hexdigest()},
    )
    (path / "meta.json").write_text(json.dumps(meta))
    receipt = json.loads((path / "capture_receipt.json").read_text())
    for name in receipt["artifacts"]:
        raw = (path / name).read_bytes()
        receipt["artifacts"][name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    (path / "capture_receipt.json").write_text(json.dumps(receipt))


def test_staged_sizes_come_from_source_types_without_reading_evaluator_data(tmp_path, monkeypatch):
    root = tmp_path / "capture"
    capture(root)
    original = Path.read_bytes

    def guarded(path):
        if path.name in {"inputs.npz", "golden.npy", "tensor_words.npz"}:
            raise AssertionError("compiler selected runtime/reference data")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    stage = tmp_path / "stage"
    manifest = stage_compile_inputs(root, stage, result_mode=MODE)
    signature = json.loads((stage / "signature.json").read_text())
    assert signature["schema"] == "merlin.compile-signature.v6"
    assert signature["result_abi"] == manifest["result_mode"] == PROFILE
    assert [r["shape"] for r in signature["outputs"]] == [[None]] * 3
    assert signature["input_constraints"]["inputs"] == [{"dtype": "i32", "shape": [None]}]
    assert not {"shape_program", "word_bindings"} & signature.keys()
    assert not any("PRIVATE" in p.read_text(errors="ignore") for p in stage.iterdir())
    validate_signature(parse_mlir_text((stage / "program.mlir").read_text()), signature)


@pytest.mark.parametrize("mutation", ("output", "dtype", "policy", "extra", "coordinate"))
def test_rehashed_false_signatures_do_not_replace_source_relationships(tmp_path, mutation):
    root = tmp_path / "capture"
    capture(root)
    stage = tmp_path / "stage"
    stage_compile_inputs(root, stage, result_mode=MODE)
    signature = json.loads((stage / "signature.json").read_text())
    if mutation == "output":
        signature["outputs"][0]["shape"] = [999]
    elif mutation == "dtype":
        signature["inputs"][0]["dtype"] = "i64"
    elif mutation == "policy":
        signature["allocation_policy"]["max_bytes"] *= 2
    elif mutation == "extra":
        signature["shape_program"] = {}
    else:
        signature["input_constraints"]["dimension_bindings"] = {"other": {"argument": 3, "axis": 0}}
    with pytest.raises(ValueError):
        validate_signature(parse_mlir_text((stage / "program.mlir").read_text()), signature)


def test_default_staging_and_older_runtime_keep_their_original_contracts(tmp_path):
    root = tmp_path / "capture"
    _write_capture(root)
    stage = tmp_path / "default"
    manifest = stage_compile_inputs(root, stage)
    assert "result_mode" not in manifest
    signature = json.loads((stage / "signature.json").read_text())
    assert signature["schema"] == "merlin.compile-signature.v1"
    selected = select_signature(parse_mlir_text((stage / "program.mlir").read_text()), signature, MODE)
    (stage / "selected.json").write_text(json.dumps(selected))
    arguments = tmp_path / "arguments.npz"
    np.savez(arguments, in0=np.array([1, 2], dtype=np.float32))
    from merlin.llvmlower.runtime_invocation import resolve as resolve_preallocated

    with pytest.raises(ValueError, match="explicit staged signature"):
        resolve_preallocated(stage / "program.mlir", stage / "selected.json", arguments)


def test_original_scalar_bounds_and_input_dimension_relations_are_still_checked():
    source = """module {func.func @forward(%x:tensor<?xi32>,%n:i64)->tensor<?xi32> {
      return %x:tensor<?xi32>
    }}"""
    original = {
        "entry": "forward",
        "inputs": [
            {"dtype": "i32", "shape": ["N"]},
            {"kind": "scalar", "dtype": "i64", "shape": [], "scalar": "N", "bounds": [0, 31]},
        ],
    }
    signature = select_signature(parse_mlir_text(source), original, MODE)
    x = np.arange(3, dtype=np.int32)
    assert len(resolve(source, signature, [x, np.array(3, dtype=np.int64)])) == 2
    for length, n in ((3, 2), (32, 32)):
        with pytest.raises(ValueError, match="input bounds|input dimensions"):
            resolve(source, signature, [np.arange(length, dtype=np.int32), np.array(n, dtype=np.int64)])
    changed = deepcopy(signature)
    changed["inputs"][0]["shape"] = [4]
    with pytest.raises(ValueError, match="actual source"):
        resolve(source, changed, [x, np.array(3, dtype=np.int64)])


def test_unknown_mode_refuses_before_any_capture_is_read(tmp_path):
    with pytest.raises(ValueError, match="result mode"):
        stage_compile_inputs(tmp_path / "absent", tmp_path / "stage", result_mode="guess")
    assert not (tmp_path / "stage").exists()


def test_staged_content_derived_results_execute_without_a_shape_prediction(tmp_path):
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.owned_result_program import load, save
    from merlin.llvmlower.owned_results import compile_owned_results

    if not toolchain.available():
        pytest.skip("native MLIR unavailable")
    root = tmp_path / "capture"
    capture(root)
    stage = tmp_path / "compiler"
    stage_compile_inputs(root, stage, result_mode=MODE)
    source = (stage / "program.mlir").read_text()
    signature = json.loads((stage / "signature.json").read_text())
    validate_signature(parse_mlir_text(source), signature)
    policy = signature["allocation_policy"]
    program = compile_owned_results(
        source,
        workdir=tmp_path / "host",
        timeout=30,
        **{name: policy[name] for name in ("max_bytes", "max_allocations", "max_output_bytes")},
    )
    reopened = load(save(program))
    for values in ([], [0], [-1, 7, 0, 3], [-9, -2], [17, 2, 5]):
        x = np.array(values, dtype=np.int32)
        outputs, report = reopened.execute(resolve(source, signature, [x]))
        assert report["source_calls"] == 1
        for out, expected in zip(outputs, (x[x > 0], x[x > 0], x), strict=True):
            assert (out.dtype, out.shape, out.tobytes()) == (expected.dtype, expected.shape, expected.tobytes())
