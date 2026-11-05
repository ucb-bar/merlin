"""The scalar host recipe mints reproducible generated inputs, not a reviewed claim."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace

import pytest

from merlin.common.paths import repo_root
from merlin.common.tree_hash import hash_tree
from merlin.mining.registry import load_rvv_package
from merlin.targetgen.target_experiment import load_target_experiment


def test_scalar_host_package_mint_is_idempotent_and_unverified(tmp_path):
    root = repo_root()
    command = [
        sys.executable,
        str(root / "build_tools/scripts/mint_scalar_host_package.py"),
        str(root / "examples/gemmini/platform/scalar-host-recipe.yaml"),
    ]
    environment = {**os.environ, "MERLIN_OUT_ROOT": str(tmp_path / "out")}
    first = subprocess.run(command, env=environment, text=True, capture_output=True, check=True)
    package_dir = tmp_path / "out/artifacts/targets/host/gemmini_rocket_scalar_int8_v0"
    assert first.stdout.strip() == str(package_dir)
    digest = hash_tree(package_dir)
    second = subprocess.run(command, env=environment, text=True, capture_output=True, check=True)
    assert second.stdout.strip() == str(package_dir)
    assert hash_tree(package_dir) == digest

    package = load_rvv_package(package_dir)
    assert package.backend == "scalar"
    assert package.dtype_strategy == "int8_w8a8"
    assert package.schedule_text == ""
    assert package.manifest["status"] == "unverified"
    assert all(not instruction.startswith("v") for instruction in package.expected_instructions)

    # A new descriptor can select this package without an RVV schedule. The reviewed
    # Gemmini descriptor is deliberately left bound to its existing package.
    experimental = load_target_experiment(root / "examples/gemmini/target/descriptor.yaml")
    package_rel = "out/artifacts/targets/host/gemmini_rocket_scalar_int8_v0"
    scalar_lane = replace(
        experimental.host_lane,
        package=package_rel,
        requires_paths=("manifest.yaml", "knobs.yaml"),
        read_only=(package_rel + "/",),
        capability_spec=None,
        capability_spec_sha256=None,
    )
    _, identity = scalar_lane.resolve(root=tmp_path)
    assert identity["backend"] == "scalar"
    assert identity["schedule_file"] is None

    (package_dir / "knobs.yaml").write_text("modified\n", encoding="utf-8")
    changed = subprocess.run(command, env=environment, text=True, capture_output=True)
    assert changed.returncode != 0
    assert "differs from deterministic recipe" in changed.stderr


def test_scalar_package_loader_refuses_vector_march(tmp_path):
    root = repo_root()
    environment = {**os.environ, "MERLIN_OUT_ROOT": str(tmp_path / "out")}
    subprocess.run(
        [
            sys.executable,
            str(root / "build_tools/scripts/mint_scalar_host_package.py"),
            str(root / "examples/gemmini/platform/scalar-host-recipe.yaml"),
        ],
        env=environment,
        check=True,
        capture_output=True,
    )
    package_dir = tmp_path / "out/artifacts/targets/host/gemmini_rocket_scalar_int8_v0"
    knobs = package_dir / "knobs.yaml"
    knobs.write_text(knobs.read_text(encoding="utf-8").replace("rv64gc_zba_zbb_zbs_zfh", "rv64gcv"), encoding="utf-8")
    with pytest.raises(ValueError, match="non-vector -march"):
        load_rvv_package(package_dir)
