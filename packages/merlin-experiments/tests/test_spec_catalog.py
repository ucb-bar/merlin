"""Catalog discovery is definition metadata, not execution composition."""

import subprocess
import sys

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
            "import sys; from merlin_experiments.spec import catalog; "
            "assert 'merlin_experiments.cli' not in sys.modules; "
            "assert 'merlin_experiments.runner' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cli_retains_the_same_catalog_function():
    from merlin_experiments.cli import catalog as cli_catalog

    assert cli_catalog is catalog
