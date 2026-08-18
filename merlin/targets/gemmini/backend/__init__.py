"""Gemmini reference backend — the OUT-OF-TREE package home (evicted from ``runtime/backends/``).

Loaded by :func:`merlin.runtime.backends.base._load_oot_backend` as ``merlin._oot_backends.gemmini`` because
the gemmini target contract declares ``plugin.backend: backend`` and the discovery pass now walks curated
reference targets (see base._oot_backend_modules). Registering under THIS PACKAGE's name makes
``base.get_backend("gemmini")`` return this package — exposing both the public backend API (re-exported from
``.gemmini``) and the codegen submodules callers reach (``gem.gemmini_codegen._ceil_dim`` etc.).

The three modules keep their sibling ``from .gemmini_codegen import ...`` relative imports (they load as
submodules of this package); their PARENT imports were rewritten absolute (``merlin.runtime.*``) so they resolve
out-of-tree.
"""
from __future__ import annotations

import importlib

# Import ``gemmini`` — runs its module-level ``register(...)`` and re-exports its public API onto the package
# so ``gem.available`` / ``gem.parse_output`` / ``gem.compile_command_buffer`` / ``gem.GemminiError`` resolve.
# NB: ``gemmini`` pulls in ``gemmini_codegen`` (its sibling) but NOT ``gemmini_codegen_mlir`` — which runs a
# module-level ``_isa = _load_isa()`` (needs the target's RTL facts / chipyard). So REGISTRATION stays free of
# that dependency; a host without the RTL toolchain still discovers + registers the gemmini backend.
from . import gemmini  # noqa: F401
from .gemmini import *  # noqa: F401,F403
from .gemmini import _common_dir  # noqa: F401 — a private helper a test reaches (not caught by *)

from merlin.runtime.backends.base import BackendInfo, BackendKind, TargetClass, register

# Re-register under the PACKAGE name (gemmini.py already registered under its own submodule name when imported
# above; registration is last-wins per name), so get_backend("gemmini") re-resolves THIS package — which
# exposes the public API above and the codegen submodules below.
register(BackendInfo("gemmini", TargetClass.NPU, BackendKind.KERNEL, __name__))


def __getattr__(name: str):
    """Expose the codegen submodules LAZILY (``gem.gemmini_codegen`` / ``gem.gemmini_codegen_mlir``) so that
    importing this package to register the backend does NOT eagerly import ``gemmini_codegen_mlir`` (whose
    module-level fact-load needs the RTL toolchain). Callers that reach the codegen internals trigger the load
    on first access — exactly when they already need the toolchain anyway."""
    if name in ("gemmini_codegen", "gemmini_codegen_mlir"):
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
