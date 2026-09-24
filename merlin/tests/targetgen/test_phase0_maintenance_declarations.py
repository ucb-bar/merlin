"""Maintenance tools follow selected declarations without reading hidden sidecars."""

import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from merlin_experiments.phase0.declarations import from_definition

from merlin.common.paths import repo_root


def _script(name):
    spec = importlib.util.spec_from_file_location(name, repo_root() / "build_tools/scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def selected(tmp_path, monkeypatch):
    synth = _script("synth_capsule_corpus")
    retire = _script("retire_hand_capsules")
    corpus = tmp_path / "corpus" / "isa"
    corpus.mkdir(parents=True)
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text(yaml.safe_dump({"target": "device", "capsule_corpus": str(corpus), "workload_spec": {}}))
    recipe = tmp_path / "custom-recipe.yaml"
    recipe.write_text("datapath: {}\ncapsules: []\n")
    (tmp_path / "performance.yaml").write_text("sweeps: []\n")
    requirement = tmp_path / "contract/capsules/conformance/device.yaml"
    requirement.parent.mkdir(parents=True)
    requirement.write_text("cells: []\n")
    definition = tmp_path / "experiment.yaml"
    definition.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": "fixture",
                "target": "device",
                "phases": {
                    "0": {
                        "adapter": "capsule_derivation",
                        "config": {
                            "descriptor": "descriptor.yaml",
                            "profile": "recipe-alias",
                            "recipe": recipe.name,
                            "performance_template": "performance.yaml",
                            "conformance_spec": str(requirement),
                            "synth_profile": "derived/custom.yaml",
                            "hidden_profile": "private-do-not-read.yaml",
                        },
                    }
                },
            }
        )
    )
    declaration = from_definition(definition)
    for tool in (synth, retire):
        monkeypatch.setattr(tool, "for_target", lambda selector: declaration)
        monkeypatch.setattr(tool, "all_declarations", lambda: (declaration,))
    monkeypatch.setattr(synth, "conformance_reference_dir", lambda: requirement.parent)
    monkeypatch.setattr(retire.CS, "derive_binding", lambda *_: object())
    return synth, retire, declaration, corpus


def test_synthesis_writes_artifact_and_checks_reference_without_overwriting(selected, tmp_path, monkeypatch, capsys):
    synth, _, declaration, _ = selected
    requirement = tmp_path / "contract/capsules/conformance/device.yaml"
    requirement.write_text("cells: []\n")
    monkeypatch.setattr(
        synth, "synthesize", lambda doc, **kwargs: {"capsules": [], "provenance": {"n_required_cells": 0}}
    )
    assert synth._targets([]) == ["device"]
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    assert synth.main(["--target", "recipe-alias", "--write", "--json"]) == 0
    generated = Path(json.loads(capsys.readouterr().out)[0]["output"])
    assert generated.is_relative_to(tmp_path / "out/artifacts/verification/device")
    assert generated.with_name("manifest.yaml").is_file()
    assert not declaration.synth_profile.exists()
    declaration.synth_profile.parent.mkdir(parents=True)
    declaration.synth_profile.write_bytes(generated.read_bytes())
    before = declaration.synth_profile.read_bytes()
    assert synth.main(["--target", "recipe-alias", "--check"]) == 0
    assert declaration.synth_profile.read_bytes() == before
    declaration.synth_profile.write_text("drift\n")
    assert synth.main(["--target", "recipe-alias", "--check"]) == 1
    assert declaration.synth_profile.read_text() == "drift\n"
    assert synth.main(["--target", "recipe-alias", "--write"]) == 0
    assert declaration.synth_profile.read_text() == "drift\n"
    assert not (tmp_path / "contract/capsules/profiles").exists()


def test_synthesis_can_select_new_generated_requirement_without_editing_example(
    selected, tmp_path, monkeypatch, capsys
):
    synth, _, declaration, _ = selected
    new_requirement = tmp_path / "out/artifacts/verification/device/new.yaml"
    new_requirement.parent.mkdir(parents=True)
    new_requirement.write_text("cells: []\napplication_demands: {status: not_declared}\n")
    monkeypatch.setattr(
        synth, "synthesize", lambda doc, **kwargs: {"capsules": [], "provenance": {"n_required_cells": 0}}
    )
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    assert synth.main(["--target", "device", "--conformance-spec", str(new_requirement), "--write", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)[0]
    generated = yaml.safe_load(Path(result["output"]).read_text())
    assert (
        generated["provenance"]["selected_inputs"]["conformance_spec_sha256"]
        == hashlib.sha256(new_requirement.read_bytes()).hexdigest()
    )
    assert declaration.conformance_spec != new_requirement


def test_incomplete_application_inventory_is_diagnostic_not_a_selectable_corpus(selected, monkeypatch, capsys):
    synth, _, declaration, _ = selected
    descriptor = yaml.safe_load(declaration.descriptor.read_text())
    descriptor["workload_spec"] = {"applications": ["model_a"]}
    declaration.descriptor.write_text(yaml.safe_dump(descriptor))
    declaration.conformance_spec.write_text("cells: []\napplication_demands: {status: incomplete}\n")
    monkeypatch.setattr(
        synth,
        "synthesize",
        lambda *_args, **_kwargs: {
            "capsules": [],
            "provenance": {"application_operation_plan": {"status": "unverified"}},
        },
    )
    monkeypatch.setattr(synth, "_ungradeable", lambda *_args: pytest.fail("diagnostic reached gradeability"))
    assert synth.main(["--target", "device", "--json"]) == 2
    result = json.loads(capsys.readouterr().out)[0]
    assert result["status"] == "incomplete_application_inventory"
    assert result["provenance"]["application_operation_plan"]["status"] == "unverified"


def test_unresolved_application_operation_blocks_generated_profile_selection(selected, monkeypatch, capsys):
    synth, _, declaration, _ = selected
    descriptor = yaml.safe_load(declaration.descriptor.read_text())
    descriptor["workload_spec"] = {"applications": ["model_a"]}
    declaration.descriptor.write_text(yaml.safe_dump(descriptor))
    declaration.conformance_spec.write_text(
        yaml.safe_dump(
            {
                "target": "device",
                "cells": [],
                "application_demands": {
                    "status": "inventoried",
                    "n_operations": 1,
                    "operation_groups": [
                        {
                            "operation": "aten.unmapped.default",
                            "mlir_operation": "linalg.generic",
                            "semantic_family": "contraction",
                            "operand_format": "int8",
                            "disposition": "hardware_admitted",
                            "count": 1,
                        }
                    ],
                },
            }
        )
    )
    monkeypatch.setattr(synth, "_ungradeable", lambda *_args: pytest.fail("blocked plan reached gradeability"))
    assert synth.main(["--target", "device", "--json", "--write"]) == 2
    result = json.loads(capsys.readouterr().out)[0]
    assert result["status"] == "unresolved_application_operations"
    plan = result["provenance"]["application_operation_plan"]
    assert plan["status"] == "blocked" and plan["blocked_operations"] == 1
    assert plan["obligations"][0]["obligation"] == "no_exact_generic_writer"
    assert not declaration.synth_profile.exists()


def test_binding_passes_exact_declared_inputs_and_never_enables_holdouts(selected, monkeypatch):
    from merlin_experiments.phase0 import profiles

    synth, _, declaration, _ = selected
    calls = []
    monkeypatch.setattr(
        profiles, "load_profile", lambda target, **kwargs: calls.append((target, kwargs)) or {"datapath": {}}
    )
    synth._binding("device")
    assert calls == [
        (
            declaration.profile,
            {
                "include_holdouts": False,
                "descriptor": declaration.descriptor,
                **{**declaration.profile_inputs(), "synth_profile": None},
            },
        )
    ]


def test_synthesis_applies_authored_policy_before_writing_and_refuses_stale_names(
    selected, tmp_path, monkeypatch, capsys
):
    synth, _, declaration, _ = selected
    requirement = tmp_path / "contract/capsules/conformance/device.yaml"
    requirement.write_text("cells: []\n")
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    derived = {"capsules": [{"name": "composition", "kind": "model"}], "provenance": {"n_required_cells": 0}}
    monkeypatch.setattr(synth, "synthesize", lambda *a, **kw: derived)
    declaration.recipe.write_text(
        yaml.safe_dump(
            {
                "synthesis_model_gates": {
                    "composition": {"after_op_pass_fraction": 0.0, "reason": "Reviewed scheduling policy"}
                }
            }
        )
    )
    assert synth.main(["--target", "device", "--write", "--json"]) == 0
    output = Path(json.loads(capsys.readouterr().out)[0]["output"])
    artifact = yaml.safe_load(output.read_text())
    assert artifact["capsules"][0]["gate"] == {"after_op_pass_fraction": 0.0}
    assert artifact["provenance"]["authored_model_gates"]["entries"][0]["name"] == "composition"
    assert "gate" not in derived["capsules"][0]
    before = set((tmp_path / "out").rglob("synth.yaml"))
    declaration.recipe.write_text(declaration.recipe.read_text().replace("composition:", "stale:"))
    assert synth.main(["--target", "device", "--write", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)[0]["status"] == "invalid_synthesis_policy"
    assert set((tmp_path / "out").rglob("synth.yaml")) == before
    assert not declaration.synth_profile.exists()


def test_authored_model_gates_preserve_existing_retained_policy():
    from merlin_experiments.phase0.declarations import all_declarations
    from merlin_experiments.phase0.synthesis_policy import apply_model_gates

    checked = 0
    for declaration in all_declarations():
        recipe = yaml.safe_load(declaration.recipe.read_bytes())
        if not recipe.get("synthesis_model_gates"):
            continue
        reference = yaml.safe_load(declaration.synth_profile.read_bytes())
        before = declaration.synth_profile.read_bytes()
        result = apply_model_gates(reference, declaration.recipe)
        assert result["capsules"] == reference["capsules"]
        assert declaration.synth_profile.read_bytes() == before
        checked += 1
    assert checked > 0, "authored model policy migration must exercise a retained reference"


def test_synthesis_check_requires_declared_reference(selected, monkeypatch):
    synth, _, declaration, _ = selected
    monkeypatch.setattr(synth, "for_target", lambda _: replace(declaration, synth_profile=None))
    monkeypatch.setattr(
        synth,
        "synth_for",
        lambda target: {
            "target": target,
            "status": "ok",
            "capsules": [],
            "provenance": {"n_required_cells": 0},
        },
    )
    assert synth.main(["--target", "device", "--check"]) == 1
    assert not declaration.synth_profile.exists()


def test_synthesis_write_does_not_require_or_create_reference(selected, monkeypatch, tmp_path, capsys):
    synth, _, declaration, _ = selected
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setattr(synth, "for_target", lambda _: replace(declaration, synth_profile=None))
    monkeypatch.setattr(
        synth,
        "synth_for",
        lambda target: {
            "target": target,
            "status": "ok",
            "capsules": [],
            "provenance": {"n_required_cells": 0},
        },
    )
    assert synth.main(["--target", "device", "--write", "--json"]) == 0
    assert Path(json.loads(capsys.readouterr().out)[0]["output"]).is_file()
    assert not declaration.synth_profile.exists()


def test_synthesis_cannot_combine_write_and_check(selected, monkeypatch):
    synth, *_ = selected
    monkeypatch.setattr(synth, "synth_for", lambda *a: pytest.fail("ambiguous mode reached synthesis"))
    with pytest.raises(SystemExit) as raised:
        synth.main(["--target", "device", "--write", "--check"])
    assert raised.value.code == 2


def test_retirement_without_declared_witness_keeps_hand_recipe(selected, monkeypatch):
    _, retire, declaration, corpus = selected
    member = corpus / "hand"
    member.mkdir()
    (member / "capsule.yaml").write_text("name: hand\nkind: op\n")
    declaration.recipe.write_text("datapath: {}\ncapsules: [{name: hand}]\n")
    monkeypatch.setattr(retire, "for_target", lambda _: replace(declaration, synth_profile=None))
    before = declaration.recipe.read_bytes()
    assert retire.classify("device")["verdicts"]["hand"][0] == retire.UNCOVERED
    assert retire.main(["--target", "device", "--write"]) == 0
    assert declaration.recipe.read_bytes() == before
    assert not (declaration.recipe.parent / "retired").exists()


def test_retirement_preserves_semantic_refusals_and_uses_public_selected_corpus(selected, monkeypatch):
    _, retire, declaration, corpus = selected
    names = ["covered", "scaled", "authored", "derived"]
    for name in names:
        member = corpus / name
        member.mkdir()
        cap = {"name": name, "kind": "op", "inputs": [{"shape": [3 if name == "scaled" else 2]}]}
        if name == "authored":
            cap["source_role"] = "handauthored_compiler_test"
        (member / "capsule.yaml").write_text(yaml.safe_dump(cap))
    hidden = corpus.parent / "hidden" / "secret"
    hidden.mkdir(parents=True)
    (hidden / "capsule.yaml").write_text("invalid: [")
    hidden_sidecar = declaration.hidden_profile
    hidden_sidecar.write_text("invalid: [")
    original_read = Path.read_text

    def read(path, *args, **kwargs):
        assert path != hidden_sidecar and hidden not in path.parents
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    declaration.recipe.write_text(yaml.safe_dump({"datapath": {}, "capsules": [{"name": n} for n in names[:-1]]}))
    declaration.synth_profile.parent.mkdir()
    declaration.synth_profile.write_text("capsules: [{name: derived}]\n")
    report = retire.classify("recipe-alias")
    assert report["verdicts"]["covered"] == (retire.COVERED, "derived")
    assert report["verdicts"]["scaled"] == (retire.COVERED_AT_SCALE, "derived")
    assert report["verdicts"]["authored"][0] == retire.UNCOVERED
    before = declaration.recipe.read_bytes()
    assert retire.main(["--target", "recipe-alias"]) == 0
    assert declaration.recipe.read_bytes() == before
    assert retire.main(["--target", "recipe-alias", "--write"]) == 0
    assert [e["name"] for e in yaml.safe_load(declaration.recipe.read_text())["capsules"]] == ["scaled", "authored"]
    archive = declaration.recipe.parent / "retired" / "recipe-alias.v0.yaml"
    assert yaml.safe_load(archive.read_text())["capsules"] == [{"name": "covered"}]
    assert "derived" in archive.with_name("RETIRED.md").read_text()
