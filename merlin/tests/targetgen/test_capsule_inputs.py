"""Input materialization has one host-only owner, without importing answer evaluators."""

from __future__ import annotations

import os
import struct
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from merlin.common.paths import python_import_roots
from merlin.targetgen import capsule_golden as golden
from merlin.targetgen import capsule_inputs as inputs


@pytest.fixture
def capsule():
    return {
        "inputs": [
            {"name": "A", "role": "input", "dtype": "f32", "shape": [1, 2]},
            {"name": "B", "role": "weight", "dtype": "bf16", "shape": [1, 2]},
            {"name": "Y", "role": "output", "dtype": "f32", "shape": [1, 2]},
        ]
    }


@pytest.fixture
def document():
    return {
        "outputs": {"Y": ["PRIVATE_ANSWER"]},
        "oracle_provenance": {
            "inputs": {
                "A": {"raw_hex": ["101", "ff"], "decoded": [[1.0, 2.0]], "shape": [1, 2]},
                "B": {"decoded": [[-1.0, 2.0]], "shape": [1, 2]},
                "SA_e8m0_codes": [[-1, 256], [3]],
                "metadata": {"private_intermediate": "PRIVATE_INTERMEDIATE"},
            }
        },
    }


@pytest.mark.parametrize(
    "name",
    [
        "capsule_stimulus_range",
        "materialize_capsule_leaves",
        "canonical_input_raws",
        "_leaf_dtype",
        "_encode_leaf",
        "_decoded_inputs",
        "canonical_input_values",
        "materialized_input_values",
        "_flatten_row_major",
        "mx_scale_codes",
    ],
)
def test_legacy_exports_are_exact_shared_implementations(name):
    assert getattr(golden, name) is getattr(inputs, name)


def test_projection_preserves_raw_precedence_shapes_and_scale_codes(monkeypatch, capsule, document):
    monkeypatch.setattr(golden, "_load_golden_yaml", lambda path: document)
    assert inputs.canonical_input_raws(capsule) == {"A": b"\x01\xff", "B": b"\x80\xbf\x00\x40"}
    assert inputs.canonical_input_values(capsule) == {
        "A": {"shape": [1, 2], "values": [1.0, 2.0]},
        "B": {"shape": [1, 2], "values": [-1.0, 2.0]},
    }
    assert inputs.mx_scale_codes(capsule) == {"SA_e8m0_codes": [255, 0, 3]}
    for name in ("golden", "mx_operands", "_load_golden_yaml", "compare"):
        assert not hasattr(inputs, name)


@pytest.mark.parametrize("document", [None, {}, [], "", False, 0])
def test_falsy_document_short_circuits_even_replaced_decoder(monkeypatch, document):
    monkeypatch.setattr(golden, "_load_golden_yaml", lambda path: document)

    def forbidden(value):
        raise AssertionError("decoder must not run without a document")

    monkeypatch.setattr(golden, "_decoded_inputs", forbidden)
    assert inputs.canonical_input_raws({}) == {}
    assert inputs.canonical_input_values({}) == {}
    assert inputs.mx_scale_codes({}) == {}


def test_truthy_document_empty_inputs_still_calls_legacy_decoder(monkeypatch):
    monkeypatch.setattr(golden, "_load_golden_yaml", lambda path: {"unrelated": True})
    monkeypatch.setattr(golden, "_decoded_inputs", lambda value: {"synthetic": [1]})
    monkeypatch.setattr(golden, "_encode_leaf", lambda *args: b"override")
    assert inputs.canonical_input_raws({}) == {"synthetic": b"override"}


@pytest.mark.parametrize("path", [None, ""])
def test_no_implicit_capsule_directory_and_loader_override_sees_exact_path(monkeypatch, tmp_path, path):
    seen = []

    def reader(value):
        seen.append(value)
        return None

    monkeypatch.setattr(golden, "_load_golden_yaml", reader)
    assert inputs.canonical_input_raws({"__dir__": str(tmp_path)}, path) == {}
    assert seen == [path]


@pytest.mark.parametrize(
    "document", [[1], "malformed", {"oracle_provenance": [1]}, {"oracle_provenance": {"inputs": [1]}}]
)
@pytest.mark.parametrize("name", ["canonical_input_raws", "canonical_input_values", "mx_scale_codes"])
def test_malformed_truthy_documents_preserve_attribute_errors(monkeypatch, document, name):
    monkeypatch.setattr(golden, "_load_golden_yaml", lambda path: document)
    with pytest.raises(AttributeError):
        getattr(inputs, name)({})


def test_invalid_loader_override_not_silently_ignored(monkeypatch):
    monkeypatch.setattr(golden, "_load_golden_yaml", None)
    with pytest.raises(TypeError):
        inputs.canonical_input_raws({})


def test_lossy_unknown_missing_and_exact_dtype_encoding(capsule):
    assert inputs._encode_leaf(capsule, "A", [1.0, -2.0]) == struct.pack("<ff", 1.0, -2.0)
    assert inputs._encode_leaf(capsule, "B", [0.1]) is None
    assert inputs._encode_leaf(capsule, "missing", [1.0]) is None
    capsule["inputs"][0]["dtype"] = "unknown"
    assert inputs._encode_leaf(capsule, "A", [1.0]) is None


def test_decoders_preserve_distinct_tuple_rules_and_dtype_override(monkeypatch):
    assert inputs._decoded_inputs({"A": {"decoded": [(1, 2), [3]]}}) == {"A": [(1, 2), 3]}
    assert inputs._flatten_row_major([(1, 2), [3]]) == [1, 2, 3]
    monkeypatch.setattr(golden, "_leaf_dtype", lambda *args: "f32")
    assert inputs._encode_leaf({}, "undeclared", [1.0]) == struct.pack("<f", 1.0)


def test_leaf_materializer_preserves_range_tensor_and_public_overrides(monkeypatch, capsule):
    calls = []

    def materialize(*args):
        calls.append(args)
        return args

    monkeypatch.setattr(golden, "capsule_stimulus_range", lambda cap: (-7, 4))
    monkeypatch.setattr(golden, "Tensor", SimpleNamespace(deterministic=materialize))
    assert set(inputs.materialize_capsule_leaves(capsule)) == {"A", "B"}
    assert calls == [("A", (1, 2), "f32", -7, 4), ("B", (1, 2), "bf16", -7, 4)]
    leaves = {
        "integer": SimpleNamespace(dtype="i8", shape=(2,), data=[1.0, -2.0]),
        "float": SimpleNamespace(dtype="f32", shape=(1,), data=[1]),
    }
    monkeypatch.setattr(golden, "materialize_capsule_leaves", lambda cap: leaves)
    assert inputs.selected_materialize_capsule_leaves({}) is leaves
    assert inputs.materialized_input_values({}) == {
        "integer": {"shape": [2], "values": [1, -2]},
        "float": {"shape": [1], "values": [1.0]},
    }
    monkeypatch.setattr(golden, "canonical_input_raws", lambda *args: {"custom": b"x"})
    assert inputs.selected_canonical_input_raws({}) == {"custom": b"x"}


def test_value_projection_preserves_flatten_override(monkeypatch, document):
    monkeypatch.setattr(golden, "_load_golden_yaml", lambda path: document)
    monkeypatch.setattr(golden, "_flatten_row_major", lambda value: ["override"])
    assert inputs.canonical_input_values({})["A"]["values"] == ["override"]


@pytest.mark.parametrize("body", ["", "null", "false", "{}", "[]"])
def test_cold_falsy_yaml_documents_are_empty(monkeypatch, tmp_path, body):
    monkeypatch.delitem(sys.modules, "merlin.targetgen.capsule_golden")
    (tmp_path / "golden.yaml").write_text(body, encoding="utf-8")
    for function in (inputs.canonical_input_raws, inputs.canonical_input_values, inputs.mx_scale_codes):
        assert function({}, tmp_path) == {}


@pytest.mark.parametrize("body, error", [("[", yaml.YAMLError), ("- truthy", AttributeError)])
def test_cold_malformed_yaml_not_normalized(monkeypatch, tmp_path, body, error):
    monkeypatch.delitem(sys.modules, "merlin.targetgen.capsule_golden")
    (tmp_path / "golden.yaml").write_text(body, encoding="utf-8")
    with pytest.raises(error):
        inputs.canonical_input_raws({}, tmp_path)


def test_cold_projected_values_do_not_include_answers(monkeypatch, tmp_path, document):
    monkeypatch.delitem(sys.modules, "merlin.targetgen.capsule_golden")
    (tmp_path / "golden.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")
    values = inputs.canonical_input_values({}, tmp_path)
    assert set(values) == {"A", "B"}
    assert "PRIVATE" not in repr(values)


def test_mx_derived_bundle_remains_evaluator_owned_with_second_read(monkeypatch):
    calls = []
    document = {
        "oracle_provenance": {
            "inputs": {
                "operand_codes": {"A_bytes": [1], "B_bytes": [2], "fmt": "fixture", "M": 1, "N": 1, "K": 1},
                "SA_e8m0_codes": [[3]],
                "SB_e8m0_codes": [[4]],
            }
        }
    }

    def reader(path):
        calls.append(path)
        return document

    monkeypatch.setattr(golden, "_load_golden_yaml", reader)
    result = golden.mx_operands({}, "fixture")
    assert calls == ["fixture", "fixture"]
    assert result["_scale_codes"] == {"SA_e8m0_codes": [3], "SB_e8m0_codes": [4]}
    monkeypatch.setattr(golden, "mx_scale_codes", lambda *args: {"override": [5]})
    assert golden.mx_operands({}, "fixture")["_scale_codes"] == {"override": [5]}


def test_new_owner_is_in_grading_source_commitment():
    from merlin.targetgen import tier_cache

    assert "merlin.targetgen.capsule_inputs" in tier_cache._GRADING_MODULES


def test_cold_core_debug_execution_without_evaluator(tmp_path, document):
    (tmp_path / "golden.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")
    script = r"""
import base64
import importlib.abc
import sys
from contextlib import nullcontext
from pathlib import Path
class BlockEvaluation(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("aet", "merlin_experiments")) or fullname in {
            "merlin.targetgen.capsule_runner", "merlin.targetgen.capsule_golden",
        }:
            raise AssertionError("optional evaluator imported: " + fullname)
sys.meta_path.insert(0, BlockEvaluation())
from merlin.targetgen import capsule_inputs as inputs, program_oracle as program
from merlin.targetgen import capsule_common, capsule_dram, dram_facts
from merlin.targetgen.rtl import mlc_bridge
root = Path(sys.argv[1])
cap = {"__dir__": str(root), "inputs": [
    {"name": "A", "role": "input", "dtype": "i8", "shape": [1, 2]},
    {"name": "Y", "role": "output", "dtype": "i8", "shape": [1, 2]},
]}
assert inputs.canonical_input_raws(cap, None) == {}
assert inputs.canonical_input_raws(cap, "") == {}
assert inputs.canonical_input_raws(cap, root / "missing") == {}
capsule_common.load_capsule = lambda path: cap
capsule_dram.output_tensor = lambda cap: cap["inputs"][1]
capsule_dram.inject_bases = lambda cb, cap, base: None
dram_facts.dram_base_for = lambda target: 0
cb = program.build_debug_cb("fixture", root)
assert base64.b64decode(cb["tensors"]["A"]["preload_b64"]) == b"\x01\xff"
assert "preload_b64" not in cb["tensors"]["Y"]
cb["commands"] = [
    {"opcode": "RES_PACK", "operands": {"src": "A", "dst": "handle"}},
    {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "handle"}},
]
prepared = program._inject_canonical_inputs("fixture", cb, root)
assert prepared["commands"][1]["operands"]["lhs"] == "A"
assert "data" in prepared["tensors"]["A"]
assert "data" not in prepared["tensors"]["Y"]
assert "data" not in cb["tensors"]["A"]
mlc_bridge.arc_available = lambda target: True
mlc_bridge.mlc_dir = lambda: root
program.modelir.importable = lambda path: nullcontext()
def run(prepared, target):
    assert "data" in prepared["tensors"]["A"]
    return {"outputs": {"Y": ["PRIVATE_ANSWER"]}, "correct": True,
            "metrics": {"cycles": 3}, "oracle": "synthetic"}
mlc_bridge.arc_run_command_buffer = run
result = program.run_command_buffer_debug("fixture", cb=cb, capsule_dir=root)
assert "PRIVATE_ANSWER" not in repr(result)
assert "outputs" not in result and "correct" not in result
assert result["metrics"] == {"cycles": 3}
assert result["n_output_tensors"] == 1
assert "merlin.targetgen.capsule_golden" not in sys.modules
assert "merlin.targetgen.capsule_runner" not in sys.modules
"""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(p) for p in python_import_roots()))
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
