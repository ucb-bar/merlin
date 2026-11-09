#!/usr/bin/env python3
"""Mint an immutable scalar host recipe from target-owned inputs.

The package selects Merlin's existing generic linalg-to-loops lowering. It has
no transform schedule, target-specific kernel, or qualification claim.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from merlin.common.paths import artifacts_dir  # noqa: E402
from merlin.compiler.backend.host import load_host_package  # noqa: E402
from merlin.compiler.backend.isa import (
    require_host_isa,  # noqa: E402
    split_isa,  # noqa: E402
)


def mint(recipe_path: Path) -> Path:
    raw = recipe_path.read_bytes()
    recipe = yaml.safe_load(raw)
    if not isinstance(recipe, dict) or recipe.get("schema") != "merlin.scalar_host_recipe.v1":
        raise ValueError("scalar host recipe must declare merlin.scalar_host_recipe.v1")
    run_id = recipe.get("run_id")
    if (
        not isinstance(run_id, str)
        or not run_id
        or not ("a" <= run_id[0] <= "z")
        or any(not ("a" <= letter <= "z" or "0" <= letter <= "9" or letter == "_") for letter in run_id[1:])
    ):
        raise ValueError("scalar host recipe needs a safe run_id")
    dtype = recipe.get("dtype_strategy")
    if dtype not in {"fp32", "int8_w8a8", "bf16_f32acc", "fp16_f32acc"}:
        raise ValueError("scalar host recipe has unsupported dtype_strategy")
    march, host_isa = recipe.get("march"), recipe.get("host_isa")
    if not isinstance(march, str) or not isinstance(host_isa, str):
        raise ValueError("scalar host recipe requires explicit march and host_isa")
    required = split_isa(march)[1]
    if any(extension == "v" or extension.startswith(("zve", "zvl", "zv")) for extension in required):
        raise ValueError("scalar host recipe cannot request vector extensions")
    require_host_isa([f"-march={march}"], host_isa)

    manifest = {
        "target": "host",
        "run_id": run_id,
        "family": "scalar_linalg",
        "status": "unverified",
        "authoring": {
            "mode": "deterministic_generated_from_spec",
            "generated_by_agent": False,
            "author": "Merlin generic scalar lowering",
        },
        "inputs": {
            "recipe_sha256": hashlib.sha256(raw).hexdigest(),
            "lowering": "merlin.llvmlower.pipeline:_upstream_pipeline",
        },
        "outputs": {"knobs": "knobs.yaml"},
    }
    knobs = {
        "backend": "scalar",
        "dtype_strategy": dtype,
        "cflags": [f"-march={march}", "-mabi=lp64d", "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-builtin"],
        "expected_instructions": [],
    }
    files = {
        "manifest.yaml": yaml.safe_dump(manifest, sort_keys=False).encode(),
        "knobs.yaml": yaml.safe_dump(knobs, sort_keys=False).encode(),
    }
    parent = artifacts_dir() / "targets" / "host"
    parent.mkdir(parents=True, exist_ok=True)
    dest = parent / run_id
    if dest.exists():
        if (
            not dest.is_dir()
            or {p.name for p in dest.iterdir()} != set(files)
            or any((dest / name).read_bytes() != payload for name, payload in files.items())
        ):
            raise ValueError(f"existing scalar package differs from deterministic recipe: {dest}")
        return dest
    with tempfile.TemporaryDirectory(prefix=f".{run_id}-", dir=parent) as temporary:
        staging = Path(temporary)
        for name, payload in files.items():
            (staging / name).write_bytes(payload)
        package = load_host_package(staging)
        if package.backend != "scalar" or package.dtype_strategy != dtype:
            raise ValueError("minted scalar host package failed loader validation")
        staging.rename(dest)
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    args = parser.parse_args()
    print(mint(args.recipe))
