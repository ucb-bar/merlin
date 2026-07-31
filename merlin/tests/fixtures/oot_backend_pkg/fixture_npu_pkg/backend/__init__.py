"""Synthetic MULTI-MODULE out-of-tree runtime backend for the OOT package-discovery regression test.

Unlike the single-file ``fixture_npu`` backend, this one spans two relative-import-coupled modules
(this ``__init__`` + ``capabilities``), mirroring an evicted accelerator backend whose implementation
is split across ``backend.py`` + ``codegen.py`` and therefore cannot be inlined into one file. Its
package's contract names it via ``plugin.backend: backend`` (a DIRECTORY), so base._load_oot_backend
loads it as ``merlin._oot_backends.fixture_npu_pkg`` WITH ITS OWN ``__path__`` — which is exactly what
lets the relative import below resolve out-of-tree. At import time it self-registers with the runtime
backend registry, exactly as an in-tree backend would; it carries no real toolchain (the test only
proves the discover -> import-as-package -> register plumbing).
"""
from __future__ import annotations

from merlin.runtime.backends import base
from merlin.runtime.backends.base import BackendInfo

from .capabilities import BACKEND_CLASS, BACKEND_KIND  # RELATIVE import — the thing under test

# Module-level self-registration: BackendInfo.module is this package's own synthetic __name__
# (``merlin._oot_backends.fixture_npu_pkg``), so base.get_backend re-resolves it from sys.modules
# without the core knowing the package layout.
base.register(BackendInfo("fixture_npu_pkg", BACKEND_CLASS, BACKEND_KIND, __name__))
