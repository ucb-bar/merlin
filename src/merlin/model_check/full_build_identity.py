"""Content correspondence for reusable whole-model builds, separate from qualification.

The identity binds direct selected tools, project source/runtime, capture bytes and
build settings. Observed compilation recipes additionally reopen actual explicit
producer inputs and outputs. Neither claims a hermetic transitive toolchain closure.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from merlin.model_check.cache import content_digest, package_digest, target_inputs


def file_identity(path: Path) -> dict:
    path = Path(shutil.which(str(path)) or path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": content_digest(path)}


def host_math_identity(policy: str) -> dict:
    """Bind the selected recipe's actual source bytes and flags, never an unselected C file."""
    import hashlib

    from merlin.runtime.host_math import host_math_recipe

    recipe = host_math_recipe(policy)
    return {
        "policy": policy,
        "source_sha256": hashlib.sha256(recipe.source.encode("utf-8")).hexdigest()
        if recipe.source is not None else None,
        "compile_flags": list(recipe.compile_flags),
        "link_flags": list(recipe.link_flags),
    }


def build_material(stage, target, label, submission, submission_sha256, settings, offload) -> dict:
    """Read build inputs once per build request, never once per recurrent input."""
    from merlin.common.paths import repo_root, runtime_dir
    from merlin.compile.model_execution_inputs import strict_tree_sha256
    from merlin.llvmlower import toolchain
    from merlin.runtime.backends import spike

    gcc = spike.gcc_path()
    tools = {
        "clang": toolchain.clang(),
        "mlir_optimizer": toolchain.llvm_install() / "bin" / "mlir-opt",
        "translate": toolchain.mlir_translate(),
        "optimizer": toolchain.llvm_opt(),
        "compiler_python": toolchain.compiler_python(),
        "gcc": gcc,
        "linker": gcc.with_name("riscv64-unknown-elf-ld"),
        "objcopy": toolchain.objcopy(),
        "nm": toolchain.nm(),
        "readelf": toolchain.readelf(),
    }
    selected = {}
    for name, path in tools.items():
        resolved = Path(shutil.which(str(path)) or path)
        selected[name] = file_identity(resolved) if resolved.is_file() else {"path": str(resolved), "missing": True}
    source = repo_root() / "src" / "merlin"
    # package_digest includes bytecode; select source files explicitly so importing
    # a module cannot change the identity of the same compiler implementation.
    project = {
        str(path.relative_to(source)): content_digest(path)
        for path in sorted(source.rglob("*.py"))
    }
    return {
        "schema": "merlin.model_check.full_build_material.v2",
        "scope": "project source and direct selected build inputs; transitive toolchain closure unknown",
        "stage": str(Path(stage).resolve()),
        "capture": strict_tree_sha256(Path(stage)),
        "target": target,
        "label": label,
        "submission_sha256": submission_sha256 if offload else None,
        "submission_content_sha256": package_digest(Path(submission)) if offload else None,
        "host_package_sha256": package_digest(settings.host_package),
        "settings": settings.record(),
        "host_math": host_math_identity(settings.host_math_policy),
        "board_catalog": file_identity(settings.board_catalog),
        "host_dts": file_identity(settings.host_dts),
        "source": project,
        "runtime_sha256": package_digest(runtime_dir()),
        "tools": selected,
        "compiler_driver": toolchain._env("MERLIN_COMPILER_DRIVER", "python"),
        "dotenv_sha256": content_digest(repo_root() / ".env") if (repo_root() / ".env").is_file() else None,
        "target_inputs": target_inputs(),
        "environment_sha256": content_digest_bytes(
            {name: value for name, value in os.environ.items() if name.startswith("MERLIN_")
             or name in {"PATH", "PYTHONPATH", "CFLAGS", "CXXFLAGS", "LDFLAGS", "CPATH", "LIBRARY_PATH"}}
        ),
        "offload": offload,
    }


def content_digest_bytes(value) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def verified_recipe(record: dict) -> dict:
    """Reject legacy/changed evidence instead of retrospectively minting custody."""
    from merlin.llvmlower.compilation_recipe import verify_completed_recipe

    evidence = record["compilation_recipe"]
    return verify_completed_recipe(
        Path(evidence["path"]), executable=Path(record["elf"]), expected_recipe_sha256=evidence["sha256"]
    )
