"""Only positive, source-bound new references can qualify a captured forward."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from merlin.common.paths import repo_root
from merlin.compile.model_execution_inputs import strict_tree_sha256


def rules():
    path = repo_root() / "examples/gemmini/model-study/reference_admission.py"
    spec = importlib.util.spec_from_file_location("model_study_reference_admission_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def reference(tmp_path):
    module = rules()
    stage = tmp_path / "capture"
    stage.mkdir()
    (stage / "model.mlir").write_text("""module {
      func.func @forward(%a: tensor<2xf32>, %b: tensor<2xi1>) -> (tensor<2xf32>, tensor<2xi1>) {
        return %a, %b : tensor<2xf32>, tensor<2xi1>
      }
    }""")
    models, producer, catalog = (tmp_path / n for n in ("models.yaml", "reference.py", "catalog.json"))
    for p in (models, producer, catalog):
        p.write_text("fixture " + p.name)
    expected = tmp_path / "expected.npz"
    np.savez(expected, out0=np.array([-0.0, 1.125], np.float32), out1=np.array([False, True]))
    policy = {"atol": 2e-6, "rtol": 1e-5, "finite_required": True, "equal_nan": False}
    document = {
        "schema": module.SCHEMA,
        "status": "completed",
        "selected_model": "fixture_model",
        "selected_program": "fixture_program",
        "all_outputs_exact": True,
        "inputs_unchanged": True,
        "expected": module.pin(expected),
        "models": module.pin(models),
        "producer": module.pin(producer),
        "catalog": module.pin(catalog),
        "capture": {"path": str(stage), **strict_tree_sha256(stage)},
        "numeric_contract": {"version": "fixture_numerical_contract", "host_math": {"policy": "native"}},
        "comparison_policy": policy,
        "reference_vs_rewritten_exact": [True, True],
        "outputs": [{"shape": [2], "dtype": "float32"}, {"shape": [2], "dtype": "bool"}],
        "inputs": [module.pin(p) for p in (models, producer, catalog)],
    }
    selection = dict(
        reference=expected,
        reference_sha256=document["expected"]["sha256"],
        models=models,
        model="fixture_model",
        program="fixture_program",
        stage=stage,
        host_math_policy="native",
        comparison_policy=policy,
    )
    return module, document, selection


def test_positive_reference_preserves_signed_zero_and_all_declared_output_types(reference):
    module, document, selection = reference
    result = module.admit(document, **selection)
    assert result["status"] == "admitted" and result["expected"] == document["expected"]
    assert result["capture"] == document["capture"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "merlin.model_study.captured_reference.v1"),
        ("schema", None),
        ("status", "failed"),
        ("status", None),
        ("all_outputs_exact", False),
        ("all_outputs_exact", 1),
        ("inputs_unchanged", False),
        ("selected_model", "wrong"),
        ("selected_program", "wrong"),
        ("numeric_contract", {}),
        ("reference_vs_rewritten_exact", [True]),
        ("reference_vs_rewritten_exact", [True, 1]),
        ("reference_vs_rewritten_exact", [True, False]),
        ("outputs", []),
        ("inputs", []),
    ],
)
def test_missing_negative_or_incomplete_positive_proof_refuses(reference, field, value):
    module, document, selection = reference
    with pytest.raises(ValueError):
        module.admit({**document, field: value}, **selection)


@pytest.mark.parametrize("name", ["models", "producer", "catalog"])
def test_changed_metadata_or_producer_cannot_reuse_admission(reference, name):
    module, document, selection = reference
    Path(document[name]["path"]).write_text("changed")
    with pytest.raises(ValueError):
        module.admit(document, **selection)


def test_capture_mutation_and_different_expected_file_refuse(reference, tmp_path):
    module, document, selection = reference
    (selection["stage"] / "new-input.bin").write_bytes(b"changed capture membership")
    with pytest.raises(ValueError, match="captured source or inputs"):
        module.admit(document, **selection)
    other = tmp_path / "other.npz"
    np.savez(other, out0=np.zeros(2, np.float32), out1=np.zeros(2, np.bool_))
    with pytest.raises(ValueError, match="expected file"):
        module.admit(document, **{**selection, "reference": other})


@pytest.mark.parametrize(
    "payload",
    [
        {"out0": np.zeros(2, np.float32)},
        {"out0": np.zeros(2, np.float32), "wrong": np.zeros(2, np.bool_)},
        {"out0": np.zeros((1, 2), np.float32), "out1": np.zeros(2, np.bool_)},
        {"out0": np.zeros(2, np.float64), "out1": np.zeros(2, np.bool_)},
        {"out0": np.array([np.nan, 0], np.float32), "out1": np.zeros(2, np.bool_)},
        {"out0": np.zeros(2, np.float32), "out1": np.array([0, 1], np.uint8)},
        {"out0": np.zeros(2, np.float32), "out1": np.frombuffer(b"\x00\x02", np.bool_)},
    ],
)
def test_bound_but_invalid_reference_array_storage_refuses(reference, payload):
    module, document, selection = reference
    np.savez(selection["reference"], **payload)
    document = {**document, "expected": module.pin(selection["reference"])}
    selection = {**selection, "reference_sha256": document["expected"]["sha256"]}
    with pytest.raises(ValueError):
        module.admit(document, **selection)


def test_document_cannot_omit_selected_producer_or_repeat_input_binding(reference):
    module, document, selection = reference
    for inputs in [document["inputs"][:-1], document["inputs"] + document["inputs"][:1]]:
        with pytest.raises(ValueError):
            module.admit({**document, "inputs": inputs}, **selection)


@pytest.mark.parametrize(
    "change",
    [
        {"host_math_policy": "portable_exp_trig"},
        {"comparison_policy": {"atol": 0.5, "rtol": 1e-5, "finite_required": True, "equal_nan": False}},
        {"reference_sha256": "0" * 64},
    ],
)
def test_reference_cannot_change_selected_numeric_policy_or_external_digest(reference, change):
    module, document, selection = reference
    with pytest.raises(ValueError):
        module.admit(document, **{**selection, **change})


def test_reference_mutation_after_array_read_refuses(reference, monkeypatch):
    module, document, selection = reference
    original = np.load

    class ChangedAfterRead:
        def __enter__(self):
            self.archive = original(selection["reference"], allow_pickle=False)
            return self.archive

        def __exit__(self, *_):
            self.archive.close()
            with selection["reference"].open("ab") as stream:
                stream.write(b"changed after arrays were read")

    monkeypatch.setattr(np, "load", lambda *a, **kw: ChangedAfterRead())
    with pytest.raises(ValueError, match="changed during admission"):
        module.admit(document, **selection)
