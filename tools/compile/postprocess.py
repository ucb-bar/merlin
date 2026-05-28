"""Post-compile artifact handling — zip benchmark bundles, place outputs.

Lightweight helpers used after `iree-compile` produces a `.vmfb`.
"""

from __future__ import annotations

import pathlib
import zipfile


def zip_artifacts(zip_path: pathlib.Path, sources_dir: pathlib.Path, vmfb_dir: pathlib.Path) -> None:
    """Flatten benchmark MLIR + VMFB artifacts into a single zip."""
    print("  📦 Zipping benchmark artifacts...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        if sources_dir.exists():
            for f in sources_dir.glob("*.mlir"):
                zf.write(f, f.name)
        if vmfb_dir.exists():
            for f in vmfb_dir.glob("*.vmfb"):
                zf.write(f, f.name)
    print(f"  ✅ Created Flattened Archive: {zip_path}")
