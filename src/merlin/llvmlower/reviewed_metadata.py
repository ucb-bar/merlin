"""One original qualification check around a trusted metadata operation.

This coalesces repeated historical-owner checks in nested registry getters.
It never caches current program, source, numerical, execution or timing checks.
The caller must finish the operation before consuming or publishing its result.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import weakref
from dataclasses import dataclass

_ACTIVE = contextvars.ContextVar("original_registry_metadata_operation", default=None)
_ISSUED = weakref.WeakSet()


@dataclass(frozen=True, eq=False)
class _Operation:
    registry: object
    capabilities: tuple
    records: list
    identity: str

    def check(self):
        if self not in _ISSUED:
            raise ValueError("registry metadata operation is not its original live scope")
        registry = self.registry
        if (
            type(registry._capabilities) is not tuple
            or len(registry._capabilities) != len(self.capabilities)
            or any(a is not b for a, b in zip(registry._capabilities, self.capabilities, strict=True))
            or registry._records != self.records
            or registry._identity != self.identity
        ):
            raise ValueError("original registry metadata changed during its checked operation")


def _metadata_verified(capability):
    """Called only after the capability's unchanged structural checks pass."""
    operation = _ACTIVE.get()
    if operation is None:
        return False
    if type(operation) is not _Operation:
        raise ValueError("registry metadata operation is not its original live scope")
    operation.check()
    return any(capability is original for original in operation.capabilities)


@contextlib.contextmanager
def scoped_registry_metadata_verification(registry):
    """Check original owners before and after one nested metadata operation.

    Only an actual live registry can enter. Every getter retains its original
    object, callback/code/source, binding, identity and record checks. A nested
    different registry gets its own fresh checks. No owner or record can be
    issued from this scope, and a failed final recheck prevents consumption.
    """
    from .reviewed_pass_capabilities import ReviewedPassCapabilities

    if type(registry) is not ReviewedPassCapabilities:
        raise ValueError("registry metadata verification needs the original live registry")
    current = _ACTIVE.get()
    if current is not None and type(current) is not _Operation:
        raise ValueError("registry metadata operation is not its original live scope")
    if current is not None and current.registry is registry:
        current.check()
        try:
            yield
        finally:
            current.check()
        return

    # Disabling an outer scope is essential when registries share a capability:
    # both boundaries of the newly selected registry must still be fresh.
    disabled = _ACTIVE.set(None)
    try:
        registry.revalidate()
        operation = _Operation(registry, registry._capabilities, copy.deepcopy(registry._records), registry._identity)
        _ISSUED.add(operation)
        token = _ACTIVE.set(operation)
        primary = None
        try:
            yield
        except BaseException as error:
            primary = error
            raise
        finally:
            _ACTIVE.reset(token)
            try:
                operation.check()
                registry.revalidate()
            except BaseException as error:
                if primary is None:
                    raise
                primary.add_note("Original registry final metadata/qualification check failed: " + str(error))
            finally:
                _ISSUED.discard(operation)
    finally:
        _ACTIVE.reset(disabled)
