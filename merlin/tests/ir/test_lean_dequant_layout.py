"""Authoritative export tests for the proved dequantization layout checker, independently of C1."""

from __future__ import annotations

import json
import subprocess

import pytest
from test_lean_dequant_scalar import export_module, lowered

from merlin.common.paths import repo_root


def check_layout(document, kind, axis=0):
    driver = repo_root() / "merlin" / "tests" / "fixtures" / "lean_dequant_layout.lean"
    proc = subprocess.run(
        ["lake", "env", "lean", "--run", str(driver), str(document), kind, str(axis)],
        cwd=repo_root() / "merlin" / "lean",
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode in (0, 1), proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["scope"] == "tensor_layout_only"
    assert (proc.returncode == 0) == result["matches"]
    return result["matches"]


@pytest.mark.parametrize(
    "kind,shape,axis",
    [
        ("per_tensor", (), 0),
        ("per_tensor", (2,), 0),
        ("per_tensor", (2, 3), 0),
        ("per_tensor", (0, 3), 0),
        ("per_channel", (2,), 0),
        ("per_channel", (2, 3), 0),
        ("per_channel", (2, 3), 1),
        ("per_channel", (0, 3), 0),
        ("per_channel", (2, 3, 4), 2),
    ],
)
def test_actual_lowering_layout(tmp_path, lean_tools, kind, shape, axis):
    module, _ = lowered(kind, shape, axis)
    assert check_layout(export_module(module, tmp_path, lean_tools), kind, axis)


@pytest.mark.parametrize("mutation", ["input_map", "output_map", "symbols", "dimensions", "reduction", "map_count"])
def test_layout_checker_rejects_mutations(tmp_path, lean_tools, mutation):
    module, _ = lowered("per_tensor")
    document = export_module(module, tmp_path, lean_tools)
    data = json.loads(document.read_text())
    generic = next(
        op for op in data["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"] if op["name"] == "linalg.generic"
    )
    maps = generic["attrs"]["indexing_maps"]["elements"]
    if mutation == "input_map":
        maps[0], maps[1] = maps[1], maps[0]
    elif mutation == "output_map":
        maps[-1] = maps[1]
    elif mutation == "symbols":
        maps[1]["symbols"] = 1
    elif mutation == "dimensions":
        maps[1]["dims"] = 2
    elif mutation == "map_count":
        maps.pop()
    else:
        generic["attrs"]["iterator_types"]["elements"][0]["value"] = "reduction"
    document.write_text(json.dumps(data))
    assert not check_layout(document, "per_tensor")


def test_layout_checker_rejects_invalid_channel_axis(tmp_path, lean_tools):
    module, _ = lowered("per_channel")
    assert not check_layout(export_module(module, tmp_path, lean_tools), "per_channel", axis=1)
