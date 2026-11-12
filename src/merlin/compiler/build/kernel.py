"""Snapshot a selected matrix kernel and its derived header for one build.

Ported from OPU-VME 80e00803d; hardware values remain caller-supplied.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.compiler.selection.kernels import KernelSelectionError, SelectedKernel, _sha256, reference_source


@dataclass(frozen=True)
class KernelChoice:
    """What a package's ``build_object(kernel=...)`` receives for a hand-written kernel: the selection and
    the header text written for this unit and configuration. The package calls :meth:`prepare`."""

    selected: SelectedKernel
    header_text: str

    def prepare(self, work: str | Path, *, scalar_tile: bool) -> PreparedKernel:
        return prepare(self.selected, work, header_text=self.header_text, scalar_tile=scalar_tile)

    def record(self) -> dict[str, Any]:
        return self.selected.record()


@dataclass(frozen=True)
class PreparedKernel:
    """What a package's ``build_object`` hands :func:`merlin.compiler.lower.matrix.emit_translation_unit`."""

    source: str
    function: str
    files: dict[str, str]
    record: dict[str, Any]


KERNEL_DIR = "kernel"


def prepare(kernel: SelectedKernel, work: str | Path, *, header_text: str, scalar_tile: bool) -> PreparedKernel:
    """Snapshot a hand-written kernel beside the shim source, with the generated header, and return the text
    the shim pastes as its kernel.

    The shipped build includes ``kernel/<file>`` (the snapshot), so the compiler's diagnostics name the
    kernel's own lines and the bytes compiled are the bytes hashed. Headers beside the kernel in its source
    directory are snapshotted too (a kernel may include its own helpers). The stand-in twin
    (``scalar_tile``) compiles :func:`reference_source` under the kernel's function name instead.
    """
    if kernel.generated:
        raise ValueError("prepare() is for a hand-written kernel; the generated one is the package's own path")
    work = Path(work)
    snap = work / KERNEL_DIR
    if snap.exists():
        shutil.rmtree(snap)
    snap.mkdir(parents=True)
    src = Path(str(kernel.entry.file))
    files: dict[str, str] = {}
    for helper in sorted(src.parent.glob("*.h")):
        if helper.name == kernel.header:
            raise KernelSelectionError(
                f"{helper} has the generated header's name {kernel.header!r}; remove it (the header is written at "
                "build time from the derived encodings)"
            )
        shutil.copy2(helper, snap / helper.name)
        files[helper.name] = _sha256(snap / helper.name)
    shutil.copy2(src, snap / src.name)
    files[src.name] = _sha256(snap / src.name)
    if files[src.name] != kernel.sha256:
        raise KernelSelectionError(f"{src} changed after it was selected (sha256 {kernel.sha256} -> {files[src.name]})")
    (snap / kernel.header).write_text(header_text, encoding="utf-8")
    files[kernel.header] = _sha256(snap / kernel.header)
    function = str(kernel.function)
    if scalar_tile:
        text = reference_source(function)
    else:
        text = (
            f"/* the selected matrix kernel {kernel.name!r}: {src} (sha256 {kernel.sha256}) */\n"
            f'#include "{KERNEL_DIR}/{src.name}"\n'
        )
    record = {
        **kernel.record(),
        "snapshot": {k: v for k, v in sorted(files.items())},
        "compiled": "reference" if scalar_tile else "kernel",
    }
    return PreparedKernel(source=text, function=function, files=files, record=record)
