"""Portable device-capture inputs and graph-derived geometry, without device execution."""

import copy
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

from merlin.common.paths import repo_root


REFRESH = repo_root() / "examples/atlas/comparisons/refresh"
with patch.object(sys, "path", [str(REFRESH), *sys.path]):
    spec = importlib.util.spec_from_file_location("resnet_device_input_controls", REFRESH / "capture_resnet_device.py")
    capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capture)


def small_plan():
    return {
        "image": {"shape": [1, 5, 12, 10], "dtype": "float32"},
        "nodes": [
            {"node": "x", "operator": "placeholder", "output_shape": [1, 5, 12, 10]},
            {"node": "conv1", "operator": "Conv2d", "inputs": ["x"], "output_shape": [1, 64, 12, 10],
             "matrix_mkn": [120, 45, 64], "padded_mkn": [128, 64, 64],
             "geometry": {"kernel_size": [3, 3], "stride": [1, 1], "padding": [1, 1], "dilation": [1, 1]}},
            {"node": "maxpool", "operator": "MaxPool2d", "inputs": ["conv1"],
             "output_shape": [1, 64, 6, 5],
             "host_params": {"kernel": [3, 3], "stride": [2, 2], "padding": [1, 1],
                             "dilation": [1, 1], "ceil_mode": False}},
            {"node": "avgpool", "operator": "AdaptiveAvgPool2d", "inputs": ["maxpool"],
             "output_shape": [1, 64, 1, 1], "host_params": {"output_size": [1, 1]}},
            {"node": "fc", "operator": "Linear", "output_shape": [1, 17],
             "matrix_mkn": [1, 64, 17], "padded_mkn": [32, 64, 32]},
        ],
    }


def test_moved_symlink_inputs_are_canonical_and_bookended(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    payload = real / "payload.bin"
    payload.write_bytes(b"recorded payload")
    alias = tmp_path / "moved"
    alias.symlink_to(real, target_is_directory=True)
    pins = {}
    expected = capture.sha(payload.read_bytes())
    assert capture.read_source(alias / payload.name, pins, expected) == b"recorded payload"
    assert pins == {str(payload): expected}
    with pytest.raises(ValueError, match="output symlink"):
        capture.checked(alias / "output", symlinks=False)
    payload.write_bytes(b"changed payload")
    with pytest.raises(ValueError, match="source changed"):
        capture.read_source(alias / payload.name, pins)


def test_relative_payload_uses_its_manifest_directory(tmp_path, monkeypatch):
    manifest = tmp_path / "bundle" / "receipt.json"
    manifest.parent.mkdir()
    payload = manifest.parent / "values.bin"
    payload.write_bytes(b"relative data")
    monkeypatch.chdir(tmp_path.parent)
    assert capture.read_source("values.bin", {}, manifest=manifest) == b"relative data"


def test_legacy_root_remapping_is_component_based_and_longest_first(tmp_path):
    rules = capture.path_remaps([f"/retired={tmp_path / 'outer'}", f"/retired/plan={tmp_path / 'inner'}"])
    assert capture.source_path("/retired/plan/weight.bin", remaps=rules) == tmp_path / "inner/weight.bin"
    assert capture.source_path("/retired/reference/output.bin", remaps=rules) == tmp_path / "outer/reference/output.bin"
    assert capture.source_path("/retired-other/value.bin", remaps=rules) == Path("/retired-other/value.bin")
    with pytest.raises(ValueError, match="duplicate"):
        capture.path_remaps([f"/retired={tmp_path}", f"/retired={tmp_path}"])
    with pytest.raises(ValueError, match="absolute"):
        capture.path_remaps(["relative=/observed"])


def test_bad_expected_hash_is_rejected_without_provenance(tmp_path):
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"exact payload")
    pins = {}
    with pytest.raises(ValueError, match="digest differs"):
        capture.read_source(payload, pins, "0" * 64)
    assert pins == {}


def test_dimensions_follow_plan_without_image_pool_or_classifier_literals():
    assert capture.plan_dimensions(small_plan()) == {
        "image": [5, 12, 10], "maxpool": [64, 6, 5], "average_channels": 64,
        "classes": 17, "classifier_padded": [32, 64, 32], "pool_geometry": (3, 2, 1),
    }


@pytest.mark.parametrize("node,field,value", [
    ("x", "output_shape", [1, 3, 12, 10]),
    ("conv1", "matrix_mkn", [119, 45, 64]),
    ("conv1", "padded_mkn", [120, 64, 64]),
    ("maxpool", "output_shape", [1, 64, 5, 5]),
    ("avgpool", "output_shape", [1, 64, 2, 1]),
    ("fc", "matrix_mkn", [1, 32, 17]),
])
def test_inconsistent_plan_geometry_is_rejected(node, field, value):
    plan = copy.deepcopy(small_plan())
    next(item for item in plan["nodes"] if item["node"] == node)[field] = value
    with pytest.raises(ValueError):
        capture.plan_dimensions(plan)


def test_maxpool_uses_supplied_geometry_and_pads_only_physical_rows():
    source = np.arange(24, dtype=np.uint8).reshape(4, 6, 1).repeat(32, axis=2)
    raw = capture.maxpool(source.tobytes(), {"h": 4, "w": 6, "padded_channels": 32}, (2, 2, 0))
    result = np.frombuffer(raw, np.uint8).reshape(32, 32)
    np.testing.assert_array_equal(result[:6, 0], [7, 9, 11, 19, 21, 23])
    assert not result[6:].any()
