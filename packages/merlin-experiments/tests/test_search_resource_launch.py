"""Capacity data cannot replace actual worker resource admission."""

import json
from pathlib import Path

import pytest
from merlin_experiments.phase2 import search_resource_launch as launch
from merlin_experiments.phase2.contracts import StageGateError
from merlin_experiments.phase2.search_configuration import read_search_configuration
from test_search_configuration import selected as selected


def test_soft_selection_does_not_issue_a_kernel_boundary(selected, monkeypatch):
    path, _document = selected

    def unexpected(**_kwargs):
        pytest.fail("capacity-only selection cannot issue resource proof")

    monkeypatch.setattr(launch, "prepare_external_resource_boundary", unexpected)
    assert launch.prepare_launch_boundary(read_search_configuration(path)) is None


def test_hard_selection_reopens_actual_mounts_and_refuses_unqualified_controls(selected, monkeypatch):
    path, document = selected
    document["resources"]["require_hard_limits"] = True
    path.write_text(json.dumps(document))
    original_read = Path.read_text
    mount_source = "actual selected mount source"
    observed = []

    def read(source, *args, **kwargs):
        return mount_source if str(source) == "/proc/self/mountinfo" else original_read(source, *args, **kwargs)

    def mounts(source):
        assert source == mount_source
        return (
            (1, "/", "/", "ext4", "ro", "ro"),
            (2, "/", "/work", "ext4", "rw", "rw"),
            (3, "/", "/proc", "proc", "rw", "rw"),
            (4, "/", "/dev/pts", "devpts", "rw", "rw"),
            (5, "/", "/tmp", "ext4", "rw", "rw"),
        )

    def refuse(**kwargs):
        observed.append(kwargs)
        raise StageGateError("actual controls are unqualified")

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(launch.kernel, "mounts", mounts)
    monkeypatch.setattr(launch, "prepare_external_resource_boundary", refuse)
    with pytest.raises(StageGateError, match="actual controls"):
        launch.prepare_launch_boundary(read_search_configuration(path))
    assert observed == [
        {
            "cpu_ids": tuple(document["resources"]["cpu_ids"]),
            "ram_bytes": document["resources"]["ram_bytes"],
            "disk_bytes": document["resources"]["disk_bytes"],
            "writable_roots": (Path("/work"), Path("/tmp")),
        }
    ]
