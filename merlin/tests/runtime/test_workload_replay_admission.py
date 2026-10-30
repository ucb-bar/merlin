"""Untrusted child receipts cannot supply unchecked workload replacement data."""

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from merlin.common.digest import sha256_bytes
from merlin.common.paths import repo_root

pytestmark = pytest.mark.target("atlas")


@pytest.fixture
def admission(monkeypatch):
    path = repo_root() / "examples/atlas/comparisons"
    monkeypatch.syspath_prepend(str(path))
    spec = importlib.util.spec_from_file_location("workload_replay_admission", path / "workload_replay_admission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return sha256_bytes(path.read_bytes())


@pytest.fixture
def child(tmp_path, admission):
    """Small synthetic evidence contract; this fixture claims no device run."""
    root = tmp_path / "model"
    root.mkdir()
    source = root / "producer-sources"
    source.mkdir()
    code = {}
    for name in admission.PRODUCER_FILES:
        (source / name).write_text("# synthetic admission fixture\n")
        code[name] = admission.identity(source / name)
    shape = (32, 32)
    values = {name: np.zeros(shape, np.float32) for name in ("A", "B", "native_C")}
    np.savez(root / "source.npz", **values)
    np.savez(root / "encoded.npz", A=np.zeros(shape, np.uint8), B=np.zeros(shape, np.uint8))
    (root / "source-manifest.json").write_text("{}\n")
    (root / "program.S").write_text("ecall\n")
    (root / "program.bin").write_bytes(bytes.fromhex("73000000"))
    (root / "initial-dram.bin").write_bytes(b"synthetic")
    for name in ("expected-bf16-tiles.bin", "actual-bf16-halves-0.bin"):
        (root / name).write_bytes(bytes(2048))
    np.savez(root / "result.npz", target_C=np.zeros(shape, np.float32),
             actual_bits=np.zeros(shape, np.uint16), expected_bits=np.zeros(shape, np.uint16))
    pins = {name: item["sha256"] for name, item in code.items()}
    receipt = {"producer": admission.PRODUCER_ID,
               "status": "passed_model_only", "shape": [32, 32, 32],
               "runs": [{"cycles": 123, "all_output_bytes_exact": True,
                         "input_and_guards_preserved": True, "drained_before_halt_observation": True}],
               "code_before": pins.copy(), "code_after": pins.copy(),
               "source_inputs": {"kind": "synthetic"}, "model_binding_before": {"kind": "synthetic"},
               "source_bookends": {"unchanged": True, "input_before": {"kind": "synthetic"},
                                   "input_after": {"kind": "synthetic"}, "model_after": {"kind": "synthetic"}},
               "artifacts": {path.name: digest(path) for path in root.iterdir() if path.is_file()}}
    (root / "receipt.json").write_text(json.dumps(receipt))
    runtime = {name: admission.array_identity(value) for name, value in values.items()}
    return root, code, receipt, runtime


def admit(admission, child):
    root, code, _, runtime = child
    return admission.admit_child_result(root, code, runtime, (32, 32, 32), 32)


def save_receipt(child):
    root, _, receipt, _ = child
    (root / "receipt.json").write_text(json.dumps(receipt))


def test_complete_synthetic_contract_returns_only_authenticated_words(admission, child):
    words, receipt, binding = admit(admission, child)
    assert words.shape == (32, 32) and words.dtype == np.uint16 and not words.any()
    assert receipt["runs"][0]["cycles"] == 123
    assert binding["artifacts"]["result.npz"]["sha256"] == digest(child[0] / "result.npz")


def test_other_producer_id_refused(admission, child):
    child[2]["producer"] = "unknown"
    save_receipt(child)
    with pytest.raises(ValueError, match="producer/status/source bookends"):
        admit(admission, child)


def test_current_atlas_receipt_updates_admitted_without_model(admission, child):
    """Evaluate the selected producer's admission fields, without importing it.

    Synthetic arrays, artifacts and run flags exercise the complete admission
    boundary; this gate makes no claim of model or numerical execution.
    """
    selected = os.environ.get("ATLAS_SOURCE_ROOT")
    if not selected:
        pytest.skip("Set ATLAS_SOURCE_ROOT for the current companion contract gate")
    tools = admission.safe_path(selected) / "tools"
    source = admission.safe_path(tools / admission.PRODUCER_FILE)
    tree = ast.parse(source.read_text(), filename=str(source))
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    code_function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "code_sources")
    names = next(node.value for node in code_function.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "names" for target in node.targets))
    assert set(ast.literal_eval(names)) == set(admission.PRODUCER_FILES)
    root, code, receipt, _ = child
    for name in code:
        snapshot = root / "producer-sources" / name
        snapshot.write_bytes(admission.safe_path(tools / name).read_bytes())
        code[name] = admission.identity(snapshot)
    pins = {name: record["sha256"] for name, record in code.items()}
    context = dict(before=pins, after=pins.copy(), m=32, k=32, n=32,
                   inputs_before=receipt["source_inputs"], inputs_after=receipt["source_inputs"].copy(),
                   model_before=receipt["model_binding_before"], model_after=receipt["model_binding_before"].copy())
    required = {"producer", "code_before", "code_after", "shape", "source_inputs",
                "model_binding_before", "source_bookends"}
    observed = {}
    for node in run.body:
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)):
            continue
        call = node.value
        if not (isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "receipt" and call.func.attr == "update"):
            continue
        for keyword in call.keywords:
            if keyword.arg in required:
                expression = compile(ast.Expression(keyword.value), str(source), "eval")
                observed[keyword.arg] = eval(expression, {"__builtins__": {}}, context)
    assert set(observed) == required, "Atlas receipt.update admission fields changed"
    status = next(node.value for node in reversed(run.body) if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                          and target.value.id == "receipt" and isinstance(target.slice, ast.Constant)
                          and target.slice.value == "status" for target in node.targets))
    receipt.update(observed, status=ast.literal_eval(status))
    save_receipt(child)
    words, admitted, _ = admit(admission, child)
    assert admitted["producer"] == observed["producer"]
    assert words.shape == (32, 32) and not words.any()


@pytest.mark.parametrize("field", ("helper_tuple", "input_bookends", "model_bookends", "truthy_unchanged"))
def test_incomplete_direct_caller_contract_refused(admission, child, field):
    _, code, receipt, _ = child
    if field == "helper_tuple":
        del code["atlas_vector_arithmetic.py"]
    elif field == "input_bookends":
        receipt["source_inputs"] = {}
        receipt["source_bookends"].update(input_before={}, input_after={})
    elif field == "model_bookends":
        receipt["model_binding_before"] = {}
        receipt["source_bookends"]["model_after"] = {}
    else:
        receipt["source_bookends"]["unchanged"] = "yes"
    save_receipt(child)
    with pytest.raises(ValueError):
        admit(admission, child)


@pytest.mark.parametrize("path,value", [
    (("runs",), []),
    (("runs", 0, "all_output_bytes_exact"), False),
    (("runs", 0, "cycles"), 0),
    (("code_after", "atlas_vector_arithmetic.py"), "0" * 64),
    (("source_bookends", "input_after"), {"kind": "different"}),
])
def test_incomplete_execution_or_bookends_refused(admission, child, path, value):
    target = child[2]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    save_receipt(child)
    with pytest.raises(ValueError):
        admit(admission, child)


def test_result_digest_checked_before_np_load(admission, child, monkeypatch):
    path = child[0] / "result.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["actual_bits"][0, 0] = 1
    np.savez(path, **arrays)
    calls = []
    original = admission.np.load
    def observed(path, **kwargs):
        calls.append(Path(path).name)
        return original(path, **kwargs)
    monkeypatch.setattr(admission.np, "load", observed)
    with pytest.raises(ValueError, match="hash"):
        admit(admission, child)
    assert "result.npz" not in calls


def test_missing_result_digest_refused_before_load(admission, child):
    del child[2]["artifacts"]["result.npz"]
    save_receipt(child)
    with pytest.raises(ValueError, match="omits"):
        admit(admission, child)


def test_changed_source_with_rebound_artifact_digest_refused(admission, child):
    path = child[0] / "source.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["A"][0, 0] = 1
    np.savez(path, **arrays)
    child[2]["artifacts"]["source.npz"] = digest(path)
    save_receipt(child)
    with pytest.raises(ValueError, match="actual runtime"):
        admit(admission, child)


def test_agreeing_mutated_result_cannot_replace_physical_output(admission, child):
    path = child[0] / "result.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["actual_bits"][0, 0] = 0x3F80
    arrays["expected_bits"][0, 0] = 0x3F80
    arrays["target_C"][0, 0] = 1.0
    np.savez(path, **arrays)
    child[2]["artifacts"]["result.npz"] = digest(path)
    save_receipt(child)
    with pytest.raises(ValueError, match="physical output"):
        admit(admission, child)


@pytest.mark.parametrize("artifact", ("source.npz", "result.npz"))
def test_export_padding_requires_positive_zero(admission, child, artifact):
    root, code, receipt, runtime = child
    for name in ("A", "native_C"):
        runtime[name] = admission.array_identity(np.zeros((16, 32), np.float32))
    admission.admit_child_result(root, code, runtime, (32, 32, 32), 16)
    path = root / artifact
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if artifact == "source.npz":
        arrays["A"][16, 0] = -0.0
    else:
        arrays["actual_bits"][16, 0] = 0x8000
        arrays["expected_bits"][16, 0] = 0x8000
        arrays["target_C"][16, 0] = -0.0
    np.savez(path, **arrays)
    receipt["artifacts"][artifact] = digest(path)
    save_receipt(child)
    with pytest.raises(ValueError, match="padding|tail"):
        admission.admit_child_result(root, code, runtime, (32, 32, 32), 16)


def test_executable_checks_allowed_symlink_ancestor(admission, tmp_path):
    parent = tmp_path / "binary-parent"
    parent.symlink_to(Path(sys.executable).parent, target_is_directory=True)
    binding = admission.admit_executable(parent / Path(sys.executable).name)
    assert binding["symlinks"] >= 1
    assert binding["physical_binary"]["sha256"] == digest(Path(binding["physical_binary"]["path"]))


def test_parent_destination_refused_before_erased_component_inspection(admission, tmp_path, monkeypatch):
    entry = tmp_path / "python-entry"
    entry.symlink_to("uninspected-component/../python")
    observed = []
    original = os.lstat
    def checked(path, *args, **kwargs):
        assert "uninspected-component" not in Path(path).parts
        observed.append(str(path))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(admission.os, "lstat", checked)
    with pytest.raises(ValueError, match="Parent traversal"):
        admission.admit_executable(entry)
    assert str(tmp_path / "uninspected-component") not in observed
