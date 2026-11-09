"""Compatibility surface for the shared matrix-kernel selection/build protocol."""

from merlin.compiler.backend.headers import emit_header, unit_header
from merlin.compiler.backend.kernel import build_kwargs, for_routing, package_root
from merlin.compiler.build.kernel import KernelChoice, PreparedKernel, prepare
from merlin.compiler.selection.kernels import (
    ABI_SIGNATURE,
    REFERENCE_FUNCTION,
    REGISTRY_FILE,
    REGISTRY_SCHEMA,
    Constraints,
    KernelConstraintError,
    KernelEntry,
    KernelSelectionError,
    Registry,
    SelectedKernel,
    check_signatures,
    load_registry,
    reference_source,
    resolve,
)

__all__ = [
    "ABI_SIGNATURE",
    "Constraints",
    "KernelChoice",
    "KernelConstraintError",
    "KernelEntry",
    "KernelSelectionError",
    "PreparedKernel",
    "REFERENCE_FUNCTION",
    "REGISTRY_FILE",
    "REGISTRY_SCHEMA",
    "Registry",
    "SelectedKernel",
    "check_signatures",
    "emit_header",
    "load_registry",
    "prepare",
    "reference_source",
    "resolve",
    "unit_header",
    "package_root",
    "for_routing",
    "build_kwargs",
]
