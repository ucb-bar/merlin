"""Independent host LLVM transform dispatch with immutable source ownership.

No target backend, numerical oracle, corpus or measured cost is selected here.
The selected callback or chain owns its semantic evidence.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .host_transform_chain import HostLLVMTransformChain


class HostLLVMTransformError(RuntimeError):
    pass


def _transform_host_ir(
    source: Path,
    workdir: Path,
    transform: Callable[[Path, Path], Path] | None,
    *,
    chain: HostLLVMTransformChain | None = None,
) -> tuple[Path, dict | None]:
    """Keep target-selected late legalization inside the normal object build.

    The source lowering artifact remains immutable. Clang verifies the selected
    LLVM IR, and its compiled object participates in the existing build hash.
    The optional callback owns its semantic proof and supporting artifacts.
    """
    if chain is not None:
        from .host_transform_chain import apply_host_transform_chain

        return apply_host_transform_chain(
            source,
            workdir,
            chain,
            terminal=(lambda current, work: _transform_host_ir(current, work, transform))
            if transform is not None
            else None,
        )
    if transform is None:
        return source, None
    import hashlib

    before = source.read_bytes()
    workdir.mkdir(parents=True, exist_ok=True)
    selected = Path(transform(source, workdir)).resolve()
    if source.read_bytes() != before:
        raise HostLLVMTransformError("host LLVM transform modified its source lowering artifact")
    if selected != source.resolve() and not selected.is_relative_to(workdir.resolve()):
        raise HostLLVMTransformError("host LLVM transform output must be retained in its workdir")
    if not selected.is_file():
        raise HostLLVMTransformError("host LLVM transform returned no LLVM IR file")
    return selected, {
        "source_path": str(source.resolve()),
        "source_sha256": hashlib.sha256(before).hexdigest(),
        "selected_path": str(selected),
        "selected_sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
    }
