"""All targets and the compatibility example use the same library preparation."""

import importlib.util
import subprocess
import sys

from merlin_experiments.phase1 import prepare_author_library as shared

from merlin.common.paths import repo_root


def test_example_is_only_a_compatibility_entrypoint():
    spec = importlib.util.spec_from_file_location(
        "library_preparation_example", repo_root() / "examples/gemmini/phase1/launch/prepare_author_library.py",
    )
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    assert example.prepare is shared.prepare
    assert example.main is shared.main
    assert example.reviewed_modules is shared.reviewed_modules


def test_shared_module_cli_can_be_called_without_a_target_backend():
    result = subprocess.run(
        [sys.executable, "-B", "-m", "merlin_experiments.phase1.prepare_author_library", "--help"],
        text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--library-version" in result.stdout
    assert "{8,16,17,18,19,20,21,22}" in result.stdout
    assert "--generated-package" in result.stdout
    assert "--reviewed-module" in result.stdout
