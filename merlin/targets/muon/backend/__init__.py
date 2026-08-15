"""Muon SIMT reference backend — the OUT-OF-TREE package home (evicted from ``runtime/backends/``).

Loaded by :func:`merlin.runtime.backends.base._load_oot_backend` as ``merlin._oot_backends.muon`` because
the muon target contract declares ``plugin.backend: backend`` and the discovery pass walks curated
reference targets (see base._oot_backend_modules). Registering under THIS PACKAGE's name makes
``base.get_backend("muon")`` return this package — exposing both the public backend API (re-exported from
``.muon``) and the codegen/oracle/introspect/CLI submodules callers reach
(``get_backend("muon").muon_oracles`` / ``.muon_introspect`` / ``.muon_codegen_mlir`` / ...).

Unlike the saturn reference package, ``muon.py`` STILL self-registers at module import (a module-level
``register(BackendInfo("muon", ...))``); the re-registration below is last-wins so ``get_backend("muon")``
re-resolves THIS package rather than the ``.muon`` submodule.

The relocated modules keep their sibling ``from .muon import ...`` relative imports (they load as
submodules of this package); their PARENT imports were rewritten absolute (``merlin.runtime.*`` /
``merlin.targetgen.*`` / ``merlin.llvmlower.*``) so they resolve out-of-tree.
"""
from __future__ import annotations

import importlib

# Import ``muon`` — runs its module-level ``register(...)`` and re-exports its public API onto the package
# so ``get_backend("muon").available`` / ``.compile_kernel_forkfree`` / ``.run_elf`` / ``.MuonUnavailable``
# resolve. NB: importing ``muon`` does NOT eagerly pull in ``muon_codegen_mlir`` (whose lowering path needs
# the MLIR->LLVM + RTL-derived transcode toolchain); REGISTRATION stays free of that dependency.
from . import muon  # noqa: F401
from .muon import *  # noqa: F401,F403

from merlin.runtime.backends.base import BackendInfo, BackendKind, TargetClass, register

# Re-register under the PACKAGE name (muon.py already registered under its own submodule name when imported
# above; registration is last-wins per name), so get_backend("muon") re-resolves THIS package — which
# exposes the public API above and the submodules below.
register(BackendInfo("muon", TargetClass.GPU, BackendKind.KERNEL, __name__))

#: Submodules exposed LAZILY (``get_backend("muon").muon_oracles`` etc.) so importing this package to
#: register the backend does NOT eagerly import the heavy codegen/oracle paths (their fact-load /
#: MLIR->LLVM toolchain). Callers that reach them trigger the load on first access — exactly when they
#: already need the toolchain anyway.
_LAZY_SUBMODULES = (
    "muon_codegen", "muon_codegen_mlir", "muon_bsp", "muon_link", "muon_harness",
    "muon_oracles", "muon_introspect", "gen_muon_digest", "muon_capsule_runner",
    "muon_mx_codegen",
)


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
