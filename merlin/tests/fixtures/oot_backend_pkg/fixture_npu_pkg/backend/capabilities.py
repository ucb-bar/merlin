"""Sibling module of the fixture package backend. Its only job is to be pulled in via a RELATIVE
import from the package ``__init__`` (``from .capabilities import ...``), so the regression test
proves a MULTI-MODULE out-of-tree backend package loads with its intra-package relative imports
intact. If base._load_oot_backend loaded the package's ``__init__`` as a bare single file (no
``__path__``), the relative import would raise and the backend would fail to register.
"""
from __future__ import annotations

from merlin.runtime.backends.base import BackendKind, TargetClass

# Consumed by the package __init__ via a relative import.
BACKEND_CLASS = TargetClass.NPU
BACKEND_KIND = BackendKind.KERNEL
