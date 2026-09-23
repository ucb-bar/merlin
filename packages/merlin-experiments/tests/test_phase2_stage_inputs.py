"""Frozen objective selection uses explicit ownership, never checkout discovery."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase2 import stage_inputs as INPUTS
from merlin_experiments.phase2.contracts import StageGateError, exact_tree_record


def _inputs(tmp_path):
    source_root = tmp_path / "source-owner"
    relative = "examples/synthetic/phase0/capsules"
    snapshot = tmp_path / "functional-inputs"
    model = snapshot / "repo" / relative / "models" / "whole_model"
    model.mkdir(parents=True)
    descriptor = {
        "name": "whole_model",
        "kind": "model",
        "label": "public",
        "lanes": {"require": ["host", "accelerator"]},
        "required_oracle_tiers": ["L2"],
        "performance": {"global_objective": True},
    }
    (model / "capsule.yaml").write_text(yaml.safe_dump(descriptor))
    (model / "capsule.interface.mlir").write_text("module {}\n")
    functional = SimpleNamespace(bundle_input_snapshot={"path": str(snapshot)})
    target = SimpleNamespace(capsule_corpus=source_root / relative / "micro", performance_global_objective=None)
    return source_root, functional, target, model


def test_full_model_uses_frozen_bytes_under_explicit_source_owner(tmp_path, monkeypatch):
    source_root, functional, target, model = _inputs(tmp_path)
    unrelated = tmp_path / "unrelated-working-directory"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    selected = INPUTS.select_full_model_sentinel(functional, target, source_root=source_root)
    assert selected.source_dir == model
    assert selected.source_sha256 == exact_tree_record(model)["sha256"]
    assert selected.capsule == "whole_model"
    assert not source_root.exists()  # No live corpus is needed or consulted.


def test_corpus_outside_selected_source_owner_refuses(tmp_path):
    source_root, functional, target, _ = _inputs(tmp_path)
    target.capsule_corpus = tmp_path / "another-owner" / "micro"
    with pytest.raises(StageGateError, match="cannot be mapped"):
        INPUTS.select_full_model_sentinel(functional, target, source_root=source_root)


def test_explicit_objective_cannot_select_private_frozen_model(tmp_path):
    source_root, functional, target, model = _inputs(tmp_path)
    descriptor = yaml.safe_load((model / "capsule.yaml").read_text())
    descriptor["label"] = "hidden"
    (model / "capsule.yaml").write_text(yaml.safe_dump(descriptor))
    with pytest.raises(StageGateError, match="no public model"):
        INPUTS.select_full_model_sentinel(functional, target, source_root=source_root, objective_capsule="whole_model")


def test_e2e_prompt_destination_requires_exact_frozen_grant(tmp_path):
    source_root, functional, target, model = _inputs(tmp_path)
    snapshot = Path(functional.bundle_input_snapshot["path"])
    frozen = SimpleNamespace(root=snapshot, grants=())
    with pytest.raises(StageGateError, match="did not grant required path"):
        INPUTS.select_e2e_sentinel(functional, frozen, target, source_root=source_root)
    frozen.grants = (SimpleNamespace(destination=source_root, source=snapshot / "repo"),)
    selected = INPUTS.select_e2e_sentinel(functional, frozen, target, source_root=source_root)
    assert selected.capsule_path == str(source_root / model.relative_to(snapshot / "repo"))
    assert selected.frozen_source_path == str(model)
    assert selected.required_lanes == ("host", "accelerator")
