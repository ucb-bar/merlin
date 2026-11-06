"""Resolve the separate compiler checkout or installed compiler package."""

import os
from pathlib import Path
import subprocess
import sys

MERLIN_ROOT = Path(__file__).resolve().parents[4]
QWEN_ROOT = Path(__file__).resolve().parent


def bootstrap_compiler(explicit=None):
    """Prefer an explicit root, ATLAS_MLIR_ROOT, a sibling, then installation."""
    requested = explicit or os.environ.get("ATLAS_MLIR_ROOT")
    root = Path(requested).expanduser().resolve() if requested else MERLIN_ROOT.parent / "atlas-mlir"
    if requested and not (root / "python/atlas_mlir").is_dir():
        raise RuntimeError(f"Compiler source package is missing under {root}")
    if (root / "python/atlas_mlir").is_dir():
        os.environ["ATLAS_MLIR_ROOT"] = str(root)
        if str(root / "python") not in sys.path:
            sys.path.insert(0, str(root / "python"))
    try:
        from atlas_mlir.toolchain import data_root
    except ModuleNotFoundError as error:
        raise RuntimeError("Set ATLAS_MLIR_ROOT or install atlas-mlir before capture/compilation") from error
    support = data_root()
    if str(support / "tools") not in sys.path:
        sys.path.insert(0, str(support / "tools"))
    return support


def observed_revision(root):
    """Record the supplied source identity without enforcing historical pins."""
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None
