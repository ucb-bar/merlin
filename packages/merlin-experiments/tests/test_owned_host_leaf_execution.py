"""Provided-host leaves preserve complete ABI inputs and never use candidate arithmetic."""

import json

import numpy as np
import pytest
import yaml

from merlin.targetgen import owned_host_leaf_execution as leaf


def test_only_reviewed_host_only_leaves_select_infrastructure():
    host = {
        "kind": "model_slice",
        "software_screen": {"status": "admitted", "decisions": [{"placement": "host", "status": "admitted"}]},
    }
    assert leaf.selected_host_leaf(host)
    assert not leaf.selected_host_leaf({**host, "kind": "model"})
    assert not leaf.selected_host_leaf(
        {
            **host,
            "software_screen": {
                "status": "admitted",
                "decisions": [{"placement": "accelerator", "status": "admitted"}],
            },
        }
    )
    assert not leaf.selected_host_leaf({"kind": "layer"})


def test_materialized_leaf_keeps_every_original_argument_and_full_output(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    source = root / "model.mlir"
    source.write_text("module { func.func @forward(%x: tensor<2xf32>) -> tensor<2xf32> { return %x : tensor<2xf32> } }")
    declaration = {"inputs": [{"name": "X", "shape": [2], "dtype": "f32"}]}
    (root / "capsule.yaml").write_text(yaml.safe_dump(declaration))
    (root / "golden.yaml").write_text(
        yaml.safe_dump(
            {
                "golden_source": "independent",
                "oracle_provenance": {"arg_order": ["X", "Y"], "inputs": {"X": {"shape": [2], "decoded": [1.0, -2.0]}}},
                "outputs": {"Y": [1.0, -2.0]},
            }
        )
    )
    monkeypatch.setattr(leaf, "_original_host_source", lambda *a, **k: (declaration, source, {"n_eligible": 0}))
    bound, certificate, identity = leaf.materialize_host_leaf({}, target="fixture", output=tmp_path / "materialized")
    assert certificate["n_eligible"] == 0
    assert (bound.root / "model.mlir").read_bytes() == source.read_bytes()
    assert json.loads((bound.root / "input_order.json").read_text()) == {"X": 0}
    assert json.loads((bound.root / "weights.safetensors.manifest.json").read_text()) == {
        "0": {"kind": "input", "name": "X"}
    }
    with np.load(bound.root / "inputs.npz", allow_pickle=False) as values:
        assert values.files == ["in0"]
        assert np.array_equal(values["in0"], [1.0, -2.0])
    with np.load(bound.root / "goldens.npz", allow_pickle=False) as values:
        assert values.files == ["Y"]
    assert len(bound.verify(bound.root)) == 8
    (bound.root / "model.mlir").write_text("another source")
    with pytest.raises(ValueError, match="changed"):
        bound.verify(bound.root)


@pytest.mark.parametrize("pure_movement", [False, True])
def test_runner_uses_provided_service_before_candidate_entrypoints(tmp_path, monkeypatch, pure_movement):
    from types import SimpleNamespace

    from merlin.targetgen import capsule_runner

    capsule = {
        "name": "provided",
        "kind": "model_slice",
        "software_screen": {"status": "admitted", "decisions": [{"placement": "host", "status": "admitted"}]},
    }
    if pure_movement:
        capsule["software_screen"]["decisions"] = []
        capsule["semantic"] = {"semantic_family": "movement", "eligible": False, "must_accelerate": False}
    cfg = SimpleNamespace(target="fixture", suite="fixtures", dtype="f32", force_match_policy=None)
    monkeypatch.setattr(capsule_runner, "run_entrypoints", lambda *a, **k: pytest.fail("candidate host arithmetic ran"))
    observed = []

    def provided(*args, **kwargs):
        observed.append((args, kwargs))
        return {"status": "incomplete", "scope": "provided host"}

    monkeypatch.setattr(leaf, "grade_owned_host_leaf", provided)
    result = capsule_runner.run_capsule(
        capsule, tmp_path, runs_root=tmp_path, config=cfg, oracle_adapters={}, pkg=object()
    )
    assert result["scope"] == "provided host"
    assert len(observed) == 1


def test_empty_compute_decisions_select_only_declared_nonaccelerated_movement():
    capsule = {
        "kind": "model_slice",
        "software_screen": {"status": "admitted", "decisions": []},
        "semantic": {"semantic_family": "movement", "eligible": False, "must_accelerate": False},
    }
    assert leaf.selected_host_leaf(capsule)
    for change in ({"eligible": True}, {"must_accelerate": True}, {"semantic_family": "contraction"}):
        assert not leaf.selected_host_leaf({**capsule, "semantic": {**capsule["semantic"], **change}})
    assert not leaf.selected_host_leaf({**capsule, "software_screen": {"status": "refused", "decisions": []}})
    assert not leaf.selected_host_leaf({**capsule, "software_screen": {"status": "admitted"}})


@pytest.mark.parametrize("pure_movement", [False, True])
def test_original_host_leaf_refuses_newly_eligible_source(tmp_path, monkeypatch, pure_movement):
    from merlin.targetgen import native_model_execution

    declaration = {
        "kind": "model_slice",
        "interface_mlir": "source.mlir",
        "operation": {"attributes": {}},
        "software_screen": {"status": "admitted", "decisions": [{"placement": "host", "status": "admitted"}]},
    }
    if pure_movement:
        declaration["software_screen"]["decisions"] = []
        declaration["semantic"] = {"semantic_family": "movement", "eligible": False, "must_accelerate": False}
    (tmp_path / "capsule.yaml").write_text(yaml.safe_dump(declaration))
    (tmp_path / "source.mlir").write_text("source bytes")
    monkeypatch.setattr(
        native_model_execution,
        "independent_frozen_source_eligibility",
        lambda *a, **k: {"regions": [{"target_eligible": True}]},
    )
    with pytest.raises(ValueError, match="eligible"):
        leaf._original_host_source({"__dir__": str(tmp_path)}, target="fixture")


def test_source_reverification_refuses_changed_full_golden(tmp_path):
    from merlin.targetgen.capsule_runner import _content_identity

    path = tmp_path / "golden.yaml"
    path.write_text("original")
    identity = _content_identity({"golden": path}, root=tmp_path)
    leaf._verify_source_identity({"__dir__": str(tmp_path)}, identity)
    path.write_text("changed")
    with pytest.raises(ValueError, match="full golden changed"):
        leaf._verify_source_identity({"__dir__": str(tmp_path)}, identity)
