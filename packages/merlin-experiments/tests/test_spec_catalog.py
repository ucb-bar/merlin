"""Catalog discovery is definition metadata, not execution composition."""

import pickle
import subprocess
import sys
from pathlib import Path

import pytest

from merlin_experiments.spec import catalog


def test_catalog_paths_are_relative_to_the_declaring_file(tmp_path, monkeypatch):
    source = tmp_path / "definitions" / "catalog.yaml"
    source.parent.mkdir()
    source.write_text("schema_version: 1\nexperiments:\n  sample: ../examples/sample/experiment.yaml\n")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert catalog(source) == {"sample": tmp_path / "examples/sample/experiment.yaml"}


def test_importing_catalog_does_not_import_cli_or_execution():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import merlin.development.definition as canonical; "
            "import merlin_experiments.spec as legacy; assert legacy is canonical; "
            "assert 'merlin_experiments.cli' not in sys.modules; "
            "assert 'merlin_experiments.runner' not in sys.modules; "
            "assert 'merlin_experiments.adapters' not in sys.modules; "
            "assert not {'torch', 'torchao', 'xdsl', 'jsonschema', 'aet', 'chia', 'ray'} & sys.modules.keys()",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cli_retains_the_same_catalog_function():
    from merlin_experiments.cli import catalog as cli_catalog

    from merlin.development import definition
    from merlin_experiments import load_spec, runner, spec

    assert cli_catalog is catalog is definition.catalog
    assert spec is definition
    assert load_spec is definition.load_spec
    assert runner.SpecError is definition.SpecError
    value = definition.ExperimentSpec(Path("/declared/experiment.yaml"), {"id": "sample", "target": "device"})
    restored = pickle.loads(pickle.dumps(value))
    assert type(restored) is type(value) and restored == value
    error = pickle.loads(pickle.dumps(definition.SpecError("preserved diagnostic")))
    assert type(error) is definition.SpecError and str(error) == "preserved diagnostic"
    assert pickle.loads(pickle.dumps(catalog)) is catalog
    assert definition.ExperimentSpec.__module__ == definition.SpecError.__module__ == "merlin_experiments.spec"


def test_legacy_global_patches_reach_canonical_declaration_algorithms(tmp_path, monkeypatch):
    from merlin.development import definition
    from merlin_experiments import spec

    calls = []

    def selected_yaml(path):
        calls.append(path)
        return {"schema_version": 1, "experiments": {"sample": "declared.yaml"}}

    monkeypatch.setattr(spec, "read_yaml", selected_yaml)
    selected = tmp_path / "catalog.yaml"
    assert definition.catalog(selected) == {"sample": tmp_path / "declared.yaml"}
    assert calls == [selected]
    # The old default-run globals remain the actual canonical function globals.
    assert definition.default_run_dir.__globals__ is vars(spec)
    assert definition.load_spec.__globals__ is vars(spec)


def test_loading_declarations_rechecks_the_same_lazy_adapter_registry(tmp_path, monkeypatch):
    from merlin.development import definition
    from merlin_experiments import adapters

    source = tmp_path / "definition.yaml"
    source.write_text("schema_version: 1\nid: sample\ntarget: device\nphases:\n  0:\n    adapter: declaration_fixture\n    config: {}\n")
    first = adapters.Adapter("declaration_fixture", "0", "unused.py")
    monkeypatch.setitem(adapters.ADAPTERS, first.name, first)
    assert definition.load_spec(source).document["phases"] == {
        "0": {"adapter": "declaration_fixture", "config": {}}
    }
    replacement = adapters.Adapter(
        "declaration_fixture", "0", "unused.py",
        options={"required": adapters.Option(required=True)},
    )
    monkeypatch.setitem(adapters.ADAPTERS, first.name, replacement)
    with pytest.raises(definition.SpecError, match="missing options"):
        definition.load_spec(source)
