"""QNN-specific tail of `kernels.core.precompile`.

The core precompile dispatcher recognizes `source_lang == "qnn-context-binary"`
as a hook point and delegates here. This module owns:

- QNN target classification (`is_qnn_target`)
- Validation-backend + board-backend mapping per QNN target
- The board-build vs host-build env switch
- The `.qnn-ctx` pre-built passthrough cache
- The `.qnn.cpp` → `build_qnn_kernel(_on_board)` compile flow

Lifted out of `kernels/core/precompile.py` so the core stays
backend-agnostic. To add another vendor SDK with the same shape (a
pre-built context-binary or source-compile path), create a sibling
`<backend>/precompile_extras.py` and add the dispatch in
`core.precompile._dispatch_extras`.
"""

from __future__ import annotations

import os
import pathlib

from . import build as _qnn_build

# Map our HAL target keys to the QNN backend identifier qnn_build uses
# during context-binary generation. The host-side default is `cpu` because
# QnnContext_getBinary on the host CPU backend accepts the shapes/ops we
# care about; switch to gpu/hta to bake backend-specific graph
# optimisations into the .qnn-ctx (the .qnn-ctx is still target-agnostic
# at runtime — the runtime HAL driver picks the real backend when loading).
_QNN_TARGET_VALIDATION_BACKEND: dict[str, str] = {
    "qnn": "cpu",
    "qnn-gpu": "cpu",
    "qnn-hta": "cpu",
}

# Map manifest target → `BoardBuildConfig.target_backend` for the
# board-side build path (`QNN_USE_BOARD_BUILD=1`).
_QNN_TARGET_BOARD_BACKEND: dict[str, str] = {
    "qnn": "gpu",
    "qnn-gpu": "gpu",
    "qnn-hta": "hta",
}


def is_qnn_target(target: str) -> bool:
    """True iff `target` is a QNN HAL target."""
    return target.startswith("qnn-") or target == "qnn"


def should_build_on_board() -> bool:
    """Whether to compile QNN kernels via the board-side path
    (`qnn.build.build_qnn_kernel_on_board`) instead of the host-side
    `qnn-context-binary-generator --backend libQnnCpu.so`.

    Required when the kernel uses fp16, uint8, or any other dtype that
    libQnnCpu rejects but the real backend (GPU/HTA) accepts. Opt in via:

        QNN_USE_BOARD_BUILD=1     enable
        QNN_BOARD_HOST=qdev       ssh host (default 'qdev')
        QNN_BOARD_QAIRT_ROOT=...  on-board QAIRT staging dir
                                  (default '/tmp/qnn_probe')
    """
    return os.environ.get("QNN_USE_BOARD_BUILD", "0").lower() in {"1", "true", "yes"}


def compile_qnn_kernel(
    kernel,
    target: str,
    source_bytes: bytes,
    cache_dir: pathlib.Path,
    *,
    force: bool,
    cache_key,
):
    """Compile a `qnn-context-binary` kernel into a `.qnn-ctx` ObjectArtifact.

    Two source shapes are accepted:
    - `<name>.qnn-ctx`: pre-built blob; cached via passthrough copy.
    - `<name>.qnn.cpp` (or `.cpp`): compiled via the QNN SDK (host or
      board path, depending on `QNN_USE_BOARD_BUILD`).

    Raises `RuntimeError` if `target` isn't a QNN target or if the source
    extension is unrecognized.

    `cache_key` is `kernels.core.precompile._cache_key` — passed in as a
    parameter so this module doesn't have to import a private name from
    core.
    """
    # Late import to avoid circularity (core.precompile imports us).
    from ..core.precompile import ObjectArtifact

    if not is_qnn_target(target):
        raise RuntimeError(
            f"kernel '{kernel.name}': source_lang=qnn-context-binary "
            f"requires a QNN target (qnn-gpu / qnn-hta), got '{target}'"
        )

    # Passthrough: pre-built .qnn-ctx blob.
    if kernel.source.suffix == ".qnn-ctx":
        key = cache_key(source_bytes, "qnn-passthrough", [])
        dst = cache_dir / f"{kernel.name}.{key}.qnn-ctx"
        if force or not dst.exists():
            dst.write_bytes(source_bytes)
        return ObjectArtifact(
            kernel.name,
            target,
            dst,
            spirv=False,
            qnn_context=True,
        )

    # Compile a .qnn.cpp source through the QNN SDK.
    if kernel.source.suffix in {".cpp", ".qnn.cpp"} or kernel.source.name.endswith(".qnn.cpp"):
        if should_build_on_board():
            board_backend = _QNN_TARGET_BOARD_BACKEND.get(target, "gpu")
            bcfg = _qnn_build.BoardBuildConfig.from_env(
                ssh_host=os.environ.get("QNN_BOARD_HOST", "qdev"),
                board_qairt_root=os.environ.get(
                    "QNN_BOARD_QAIRT_ROOT",
                    "/tmp/qnn_probe",
                ),
                target_backend=board_backend,
            )
            qnn_ctx_path = _qnn_build.build_qnn_kernel_on_board(
                kernel.source,
                kernel.name,
                cache_dir,
                bcfg,
            )
        else:
            backend_for_validation = _QNN_TARGET_VALIDATION_BACKEND.get(target, "cpu")
            cfg = _qnn_build.QnnBuildConfig.from_env(backend=backend_for_validation)
            qnn_ctx_path = _qnn_build.build_qnn_kernel(
                kernel.source,
                kernel.name,
                cache_dir,
                cfg,
            )
        return ObjectArtifact(
            kernel.name,
            target,
            qnn_ctx_path,
            spirv=False,
            qnn_context=True,
        )

    raise RuntimeError(
        f"kernel '{kernel.name}': source_lang=qnn-context-binary "
        f"expects either a .qnn.cpp source (built via the QNN SDK) "
        f"or a pre-built .qnn-ctx blob; got '{kernel.source.name}'"
    )
