"""Saturn RVV/CPU reference backend — the OUT-OF-TREE package home (evicted from ``runtime/backends/``).

Loaded by :func:`merlin.runtime.backends.base._load_oot_backend` as ``merlin._oot_backends.saturn`` because
the saturn target contract declares ``plugin.backend: backend`` and the discovery pass walks curated
reference targets (see base._oot_backend_modules). Registering under THIS PACKAGE's name makes
``base.get_backend("saturn_vec")`` return this package — exposing both the public backend API (re-exported
from ``.saturn_vec``) and the codegen/MLIR submodules callers reach
(``get_backend("saturn_vec").saturn_vec_codegen`` / ``.saturn_vec_mlir``).

The three modules keep their sibling ``from .saturn_vec_codegen import ...`` relative imports (they load as
submodules of this package); their PARENT imports were rewritten absolute (``merlin.runtime.*`` /
``merlin.llvmlower.*``) so they resolve out-of-tree.
"""
from __future__ import annotations

import importlib

# Import ``saturn_vec`` — re-exports its public API onto the package so ``sv.available`` /
# ``sv.run_command_buffer`` / ``sv.compile_command_buffer`` / ``sv.parse_output`` / ``sv.SaturnVecError``
# resolve. NB: ``saturn_vec`` pulls in ``saturn_vec_codegen`` (its sibling) but NOT ``saturn_vec_mlir``
# (whose lowering path needs the MLIR→LLVM toolchain); so REGISTRATION stays free of that dependency.
from . import saturn_vec  # noqa: F401
from .saturn_vec import *  # noqa: F401,F403

from merlin.runtime.backends.base import BackendInfo, BackendKind, TargetClass, register

# ``saturn_vec.py`` no longer self-registers (it relied on the core base.py seed, now deleted); this is
# the sole registration. Register under the PACKAGE name so get_backend("saturn_vec") resolves THIS
# package — which exposes the public API above and the codegen/MLIR submodules below.
register(BackendInfo("saturn_vec", TargetClass.CPU, BackendKind.KERNEL, __name__))


def __getattr__(name: str):
    """Expose the codegen/MLIR submodules LAZILY (``sv.saturn_vec_codegen`` / ``sv.saturn_vec_mlir``) so
    importing this package to register the backend does NOT eagerly import ``saturn_vec_mlir`` (whose
    lowering path needs the MLIR→LLVM toolchain). Callers that reach those internals trigger the load on
    first access — exactly when they already need the toolchain anyway."""
    if name in ("saturn_vec_codegen", "saturn_vec_mlir"):
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
