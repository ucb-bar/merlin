"""Bind selected matrix kernels to an explicitly selected provider.

Ported from OPU-VME 80e00803d; hardware values remain caller-supplied.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.compiler.build.kernel import KernelChoice
from merlin.compiler.selection.kernels import SelectedKernel, check_signatures, resolve

from .headers import unit_header


def package_root(support_target: str) -> Path:
    """The selected support package's root (where its ``kernels/`` live), resolved like its plugin."""
    from merlin.targetgen.plugins import resolve_support

    return Path(resolve_support(support_target).base)


def for_routing(routing: Any) -> SelectedKernel:
    """The kernel a ``MatrixRouting`` (``support_target``, ``unit``, ``kernel``) builds with, validated.

    With no kernel named, a package that cannot be resolved here still has its default -- its plugin's own
    build, which fails on its own terms if the package is missing -- but a NAMED kernel never falls back.
    """
    from merlin.targetgen.plugins import PluginError

    spec = getattr(routing, "kernel", None)
    target = getattr(routing, "support_target", None)
    try:
        root: Path | None = package_root(target) if target or spec else None
    except (PluginError, FileNotFoundError):
        if spec:
            raise
        root = None
    return resolve(spec, package_root=root, unit=routing.unit)


def build_kwargs(routing: Any, signatures: Mapping[str, Sequence[int]]) -> dict[str, Any]:
    """``{"kernel": KernelChoice}`` for the provider's ``build_object`` after the constraint check, or ``{}``
    when the package's generated kernel is selected -- so that build is exactly the one it always was."""
    chosen = for_routing(routing)
    if chosen.generated:
        return {}
    check_signatures(chosen, signatures, tile_edge=routing.tile_edge())
    provider = routing.provider()
    text = unit_header(
        provider, unit=routing.unit, config=routing.config, header=chosen.header, prefix=chosen.header_prefix
    )
    return {"kernel": KernelChoice(chosen, text)}
