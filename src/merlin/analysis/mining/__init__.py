"""Schedule-package preparation, fork lineage and reference observations.

Host package loading is shared compiler infrastructure. The historical RVV
names below retain that class and loader identity; target implementation comes
from an explicitly selected OOT provider.
"""

from __future__ import annotations

from merlin.compiler.backend.host import (
    HostPackage as RvvPackage,
)
from merlin.compiler.backend.host import (
    default_run,
)
from merlin.compiler.backend.host import (
    load_host_package as load_rvv_package,
)

# ---- mining ------------------------------------------------------------------------------------

__all__ = [
    "RvvPackage",
    "default_run",
    "load_rvv_package",
]
