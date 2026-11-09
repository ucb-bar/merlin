"""Temporary compatibility imports for host code-generation packages.

The implementation is owned by :mod:`merlin.compiler.backend.host`. Historical mining
consumers retain the same class and function objects while their imports migrate. The
serialized ``rvv_package_manifest`` name remains supported for scalar and RVV packages.
"""

from merlin.compiler.backend.host import (
    DTYPE_STRATEGIES as DTYPE_STRATEGIES,
)
from merlin.compiler.backend.host import (
    HostPackage as RvvPackage,
)
from merlin.compiler.backend.host import (
    _check_cflags as _check_cflags,
)
from merlin.compiler.backend.host import (
    default_run as default_run,
)
from merlin.compiler.backend.host import (
    load_host_package as load_rvv_package,
)
from merlin.compiler.backend.host import (
    resolve_features as _resolve_features,
)

__all__ = ["DTYPE_STRATEGIES", "RvvPackage", "_check_cflags", "_resolve_features", "default_run", "load_rvv_package"]
