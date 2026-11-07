"""Content identity for reuse of compiled host-dispatch kernels.

The outlined kernel text is only the input to lowering. Optional passes,
Merlin's lowerer, its Python MLIR packages, and native compilers can change
the shared object without changing that text. Persistent caches must include
their observed bytes before a cached shared object can be loaded.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from merlin.common.paths import repo_root


def _file(path: Path) -> dict[str, Any]:
    actual = path.resolve(strict=True)
    if not actual.is_file():
        raise ValueError(f"host cache identity needs a regular file: {path}")
    digest = hashlib.sha256()
    with actual.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return {"path": str(actual), "bytes": actual.stat().st_size, "sha256": digest.hexdigest()}


def _tree(path: Path) -> dict[str, Any]:
    root = path.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"host cache identity needs a directory: {path}")
    digest = hashlib.sha256()
    count = 0
    for item in sorted(root.rglob("*")):
        if not item.is_file() or "__pycache__" in item.parts or item.suffix == ".pyc":
            continue
        relative = item.relative_to(root).as_posix()
        content = _file(item)
        digest.update(relative.encode() + b"\0")
        digest.update(content["sha256"].encode() + b"\0")
        count += 1
    if count == 0:
        raise ValueError(f"host cache identity has no source files: {path}")
    return {"path": str(root), "files": count, "sha256": digest.hexdigest()}


def _python_packages(interpreter: Path) -> dict[str, Path]:
    code = (
        "import importlib.util,json; "
        "names=('torch_mlir','xdsl','numpy'); "
        "print(json.dumps({n:list(importlib.util.find_spec(n).submodule_search_locations)[0] "
        "for n in names}))"
    )
    result = subprocess.run([str(interpreter), "-c", code], capture_output=True, text=True, check=True, timeout=30)
    return {name: Path(path) for name, path in json.loads(result.stdout).items()}


def host_cache_identity(
    *,
    source_root: Path | None = None,
    tool_paths: Mapping[str, Path] | None = None,
    package_paths: Mapping[str, Path] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Hash effective source, selected compiler bytes, and lowering switches.

    Optional arguments allow tiny synthetic mutation tests. The production
    defaults inspect current bytes, including dirty source edits and resolved
    compiler executables, rather than trusting a Git HEAD or version string.
    """
    from merlin.llvmlower import optional_passes, toolchain

    env = os.environ if environ is None else environ
    source = Path(source_root) if source_root is not None else repo_root() / "src" / "merlin"
    if tool_paths is None:
        selected = {
            "clang": toolchain.clang(),
            "mlir_translate": toolchain.mlir_translate(),
            "compiler_python": toolchain.compiler_python(),
            "system_cc": Path(shutil.which("cc") or "cc"),
        }
        if llc := toolchain.host_llc():
            selected["host_llc"] = llc
    else:
        selected = dict(tool_paths)
    if not selected:
        raise ValueError("host cache identity needs compiler executables")
    packages = _python_packages(selected["compiler_python"]) if package_paths is None else dict(package_paths)
    if not packages:
        raise ValueError("host cache identity needs compiler Python packages")
    switches = {optional_passes.ENV: env.get(optional_passes.ENV, "")}
    pass_switches = {entry.switch for entry in optional_passes.REGISTRY if entry.switch}
    switches.update({key: env.get(key, "") for key in sorted(pass_switches)})
    switches["MERLIN_TORCHAO_CPU_SOURCE_COMMIT"] = env.get("MERLIN_TORCHAO_CPU_SOURCE_COMMIT", "")
    switches["MERLIN_TORCHAO_CPU_CAPABILITY"] = env.get("MERLIN_TORCHAO_CPU_CAPABILITY", "")
    # This host-only dispatch policy changes completed values without changing
    # the outlined kernel text or the target object. Strict replay must still
    # distinguish its selected numerical path.
    switches["MERLIN_TORCHAO_CPU_VML_TRIG"] = env.get("MERLIN_TORCHAO_CPU_VML_TRIG", "")
    inputs: dict[str, Any] = {
        "schema": "merlin.host_cache_identity.v1",
        "merlin_source": _tree(source),
        "tools": {name: _file(Path(path)) for name, path in sorted(selected.items())},
        "packages": {name: _tree(Path(path)) for name, path in sorted(packages.items())},
        "switches": switches,
        "host_machine": platform.machine(),
    }
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"sha256": digest, "inputs": inputs}
