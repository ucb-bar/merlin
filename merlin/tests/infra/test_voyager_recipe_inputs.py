"""Voyager experiment input routing only; no external compiler or hardware execution."""

import importlib.util
import sys

import pytest
import yaml
from merlin_experiments.phase0 import declarations as D
from merlin_experiments.spec import SpecError

from merlin.common.paths import repo_root


def load(name):
    path = repo_root() / "merlin/experiments/voyager_h2h/scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"fixture_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    corpus = tmp_path / "relocated-corpus"
    capsule = corpus / "isa" / "chosen"
    capsule.mkdir(parents=True)
    (capsule / "capsule.interface.mlir").write_text("// fixture only\n")
    (capsule / "capsule.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "chosen",
                "inputs": [
                    {"name": "a", "shape": [2, 3], "dtype": "i8"},
                    {"name": "b", "shape": [3, 4], "dtype": "i8"},
                ],
                "operation": {"attributes": {"lhs": "a", "weight": "b", "output_dtype": "i32"}},
            }
        )
    )
    recipe = tmp_path / "unrelated-name.yaml"
    recipe.write_text(yaml.safe_dump({"capsules": [{"name": "chosen", "cat": "isa", "op": "matmul"}]}))
    descriptor = tmp_path / "device.yaml"
    descriptor.write_text(yaml.safe_dump({"target": "device", "capsule_corpus": str(corpus / "isa")}))
    declaration = D.DerivationDeclaration(
        tmp_path / "definition.yaml",
        "example",
        "device",
        "alias",
        descriptor,
        recipe,
        tmp_path / "template.yaml",
        None,
        None,
        None,
    )
    monkeypatch.setattr(D, "for_target", lambda selector: declaration)
    monkeypatch.setattr(D, "from_definition", lambda path: declaration)
    return declaration, capsule


def test_h2h_uses_explicit_recipe_and_descriptor_corpus(inputs):
    declaration, capsule = inputs
    module = load("capsule_h2h")
    assert module._capsule_inputs("device", "alias", None) == {"chosen": capsule / "capsule.interface.mlir"}
    assert module._capsule_inputs("device", None, ["absent"], definition=declaration.definition) == {}


def test_bridge_preserves_capsule_arithmetic_from_declared_corpus(inputs):
    declaration, capsule = inputs
    module = load("build_bridge_package")
    [row] = module._declared_capsules("device", None, definition=declaration.definition)
    assert (row["M"], row["K"], row["N"], row["K_weight"]) == (2, 3, 4, 3)
    assert row["dtypes"] == ("i8", "i8")
    assert row["output_dtype"] == "i32"
    assert row["path"] == str(capsule / "capsule.yaml")


def test_both_readers_refuse_cross_target_selection(inputs):
    with pytest.raises(ValueError, match="different target"):
        load("capsule_h2h")._capsule_inputs("other", "alias", None)
    with pytest.raises(ValueError, match="different target"):
        load("build_bridge_package")._declared_capsules("other", "alias")


def test_bridge_ambiguity_refuses_before_provider_or_compiler_work(tmp_path, monkeypatch):
    module = load("build_bridge_package")

    def ambiguous(selector):
        raise SpecError("ambiguous declaration")

    def forbidden(*args, **kwargs):
        raise AssertionError("input refusal must precede external work")

    monkeypatch.setattr(D, "for_target", ambiguous)
    monkeypatch.setattr(module.provenance, "verify", forbidden)
    monkeypatch.setattr(module, "accelerator_config_for", forbidden)
    with pytest.raises(SpecError, match="ambiguous declaration"):
        module.main(["--target", "device", "--reference-package", str(tmp_path)])


@pytest.mark.parametrize("requested", ["device", "different-device"])
def test_inconsistent_descriptor_refuses_before_provider(inputs, tmp_path, monkeypatch, requested):
    declaration, _ = inputs
    document = yaml.safe_load(declaration.descriptor.read_text())
    document["target"] = "different-device"
    declaration.descriptor.write_text(yaml.safe_dump(document))
    module = load("build_bridge_package")

    def forbidden(*args, **kwargs):
        raise AssertionError("mismatched hardware identity must refuse before provider work")

    monkeypatch.setattr(module.provenance, "verify", forbidden)
    monkeypatch.setattr(module, "accelerator_config_for", forbidden)
    with pytest.raises(ValueError, match="different target"):
        module.main(["--target", requested, "--reference-package", str(tmp_path)])
    with pytest.raises(ValueError, match="different target"):
        load("capsule_h2h")._capsule_inputs(requested, "alias", None)
