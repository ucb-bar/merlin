"""Codegen execution selection is declared separately from resource ownership."""

import socket
import subprocess

import pytest
import yaml

from merlin.targetgen.target_experiment import load_target_experiment


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("declaration loading must not execute or listen")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(socket.socket, "bind", refuse)


def load(tmp_path, **fields):
    path = tmp_path / "descriptor.yaml"
    path.write_text(yaml.safe_dump({"target": "experiment_fixture", **fields}))
    return load_target_experiment(path, source_root=tmp_path)


def test_explicit_codegen_backend_is_independent_of_resource_owner(tmp_path):
    experiment = load(
        tmp_path,
        preflight={"codegen_backend": "support_fixture"},
        backend_package_dir="unavailable/resource_owner",
    )
    assert experiment.preflight_codegen_backend == "support_fixture"
    assert experiment.target == "experiment_fixture"
    assert experiment.backend_package_dir == "unavailable/resource_owner"


def test_resource_owner_does_not_select_codegen_backend(tmp_path):
    experiment = load(tmp_path, backend_package_dir="unavailable/resource_owner")
    assert experiment.preflight_codegen_backend is None


@pytest.mark.parametrize("value", [None, "", "   ", False, 3, [], {}, ["support_fixture"]])
def test_present_invalid_codegen_backend_refuses(tmp_path, value):
    with pytest.raises(ValueError, match=r"preflight\.codegen_backend.*non-empty string"):
        load(tmp_path, preflight={"codegen_backend": value})
