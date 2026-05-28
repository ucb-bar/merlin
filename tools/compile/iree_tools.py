"""Locate iree-compile / iree-import-onnx binaries + ONNX-to-MLIR import.

Helpers extracted from `cli.py` so the `main()` flow can stay focused on
compose-flags-and-invoke logic. Used by `cli.main` and `radiance` (radiance
has its own iree-compile resolution because of historical default-build-dir
quirks; if those converge later, point both at `get_iree_tool`).
"""

from __future__ import annotations

import pathlib
import sys

import utils


def get_iree_tool(tool_name: str, build_dir_name: str) -> pathlib.Path:
    """Find an iree-* binary, preferring in-tree builds over installs.

    Resolution order:
      1. `build/<build_dir_name>/tools/<tool_name>` (in-tree build)
      2. `build/<build_dir_name>/install/bin/<tool_name>` (install in same tree)
      3. `build/host-merlin-release/tools/<tool_name>` (merlin fallback)
      4. `build/host-merlin-release/install/bin/<tool_name>` (merlin install fallback)
      5. `<conda_env>/bin/<tool_name>` (last resort)

    Exits with code 1 if nothing resolves.
    """
    primary_build_tool = utils.REPO_ROOT / "build" / build_dir_name / "tools" / tool_name
    if primary_build_tool.exists():
        return primary_build_tool

    primary_install_tool = utils.REPO_ROOT / "build" / build_dir_name / "install" / "bin" / tool_name
    if primary_install_tool.exists():
        return primary_install_tool

    fallback_build_tool = utils.REPO_ROOT / "build" / "host-merlin-release" / "tools" / tool_name
    if fallback_build_tool.exists():
        return fallback_build_tool

    fallback_install_tool = utils.REPO_ROOT / "build" / "host-merlin-release" / "install" / "bin" / tool_name
    if fallback_install_tool.exists():
        return fallback_install_tool

    env_tool = pathlib.Path(sys.executable).parent / tool_name
    if env_tool.exists():
        return env_tool

    utils.eprint(f"❌ Error: {tool_name} not found in build/{build_dir_name} or environment.")
    sys.exit(1)


def import_onnx(onnx_path: pathlib.Path, mlir_path: pathlib.Path, build_dir: str, dry_run: bool) -> None:
    """ONNX → MLIR via iree-import-onnx. Exits on failure."""
    import_tool = get_iree_tool("iree-import-onnx", build_dir)
    print(f"  📥 Importing ONNX to MLIR using {import_tool.parent.parent.parent.name}...")
    cmd = [str(import_tool), str(onnx_path), "--opset-version", "17", "-o", str(mlir_path)]
    if utils.run(cmd, dry_run=dry_run) != 0:
        utils.eprint("❌ ONNX Import failed.")
        sys.exit(1)
