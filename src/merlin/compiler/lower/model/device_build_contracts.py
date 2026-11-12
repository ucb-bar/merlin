"""Device source entry conventions and post-offload transform selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

FROM_GROUP = "stated_group"


FROM_EXTENTS = "bare_extents"


def rename_object_command(objcopy, original, selected, raw, output):
    """Expose only the selected entry; retain unresolved runtime imports.

    LLVM applies symbol visibility before renaming; GNU applies it afterward.
    Naming both sides preserves the entry with either tool, and privatizes helpers.
    """
    return [
        str(objcopy),
        f"--redefine-sym={original}={selected}",
        f"--keep-global-symbol={original}",
        f"--keep-global-symbol={selected}",
        str(raw),
        str(output),
    ]


def executable_kernel_entry(device: str) -> str:
    """The selected executable entry, independently of logical pointer-layout names."""
    from merlin.runtime.backends import base as backends

    return backends.harness_build_recipe(device).require_kernel_stack_frame().entry_symbol


def kernel_symbol(base: str, index: int) -> str:
    """The per-signature kernel symbol. Distinct by construction; see the module docstring."""
    return f"{base}_{int(index)}"


def kernel_entry(
    symbol: str, extents: Sequence[int], stated: Mapping[str, Any] | None, device: str
) -> tuple[dict[str, Any] | None, str, str]:
    """``(entry, provenance, refusal)`` -- what a backend is asked to emit for one routed symbol.

    Two sources, named apart because they compute different functions. A STATED group program is
    carried verbatim: it is the layer's own program -- epilogue, requantize multiplier, convolution
    geometry, committed type -- and rebuilding any of it from the extents would emit a kernel that
    computes something else. Without one the entry is synthesized as a bare contraction of those
    extents, which is honest for a caller that routed exactly the contraction.

    THE EXTENT TRIPLE IS REQUIRED ONLY OF THE SYNTHESIZED FORM. A statement carries its own shape --
    a convolution states taps and strides, an integer sum reduces over nothing -- so demanding
    ``M x K x N`` of one would decline exactly the layers a whole-program route exists to carry
    (measured on a captured ResNet-50: 54 of its 69 routed groups are convolutions). A separate
    function so that rule is reachable without a package and a toolchain: inside the build loop it
    sits behind a package load, and a test of it there passes whether or not the rule is present.
    """
    key = tuple(int(v) for v in extents)
    if stated is not None:
        entry = dict(stated)
        entry.setdefault("name", symbol)
        return entry, FROM_GROUP, ""
    if len(key) not in (3, 4):
        return None, FROM_EXTENTS, f"signature {key} has neither 3 nor 4 extents; no kernel shape for it"
    m, n, k = key[-3:]
    return (
        {
            "name": symbol,
            "op": "matmul",
            "kind": "op",
            "source_role": "mesh_tile_synthesized",
            "source_reference": f"offloaded layer {m}x{k}x{n} for {device}",
            "M": m,
            "K": k,
            "N": n,
        },
        FROM_EXTENTS,
        "",
    )


def _flags(codegen_target: str, cflags: Sequence[str] | None = None) -> list[str]:
    """Compile flags for the device objects: the CALLER's when it supplied them.

    The defaults name an ISA (`-march=rv64gcv`), and a default ISA is an assumption about the
    hardware. Baking it in here meant the device shim was compiled with the vector extension for a
    core that does not have one: the whole-model image trapped mid-run on its first `vsetvli`
    (`mcause=2`, mtval opcode 0x57) on a Rocket whose own DTS reads
    `rv64imafdcbzicsr_..._xrocket` -- no `v`. The kernels themselves came out clean because they are
    translated from the target's own lowering; only this C shim was compiled against the default.
    The matrix/OPU shim path already took its flags from the caller; this one now does too."""
    if cflags:
        return list(cflags)
    from merlin.compiler.lower.model.codegen import RISCV_FLAGS, X86_FLAGS

    return list(RISCV_FLAGS if codegen_target == "riscv" else X86_FLAGS)


def apply_post_offload_transform(routing, prepared: Path, sidecar: Path, workdir: Path) -> Path:
    """Compatibility entry for the ordinary preparation-owned host ABI rewrite."""
    from merlin.compiler.lower.model.device_preparation import apply_post_offload_transform as prepare

    return prepare(routing, prepared, sidecar, workdir)


rename_object_command.__module__ = "merlin.compiler.lower.model.device_build"


executable_kernel_entry.__module__ = "merlin.compiler.lower.model.device_build"


kernel_symbol.__module__ = "merlin.compiler.lower.model.device_build"


kernel_entry.__module__ = "merlin.compiler.lower.model.device_build"


_flags.__module__ = "merlin.compiler.lower.model.device_build"


apply_post_offload_transform.__module__ = "merlin.compiler.lower.model.device_build"
