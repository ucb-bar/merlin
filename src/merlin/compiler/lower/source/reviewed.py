"""Explicit original-host selection of independently reviewed pass capabilities.

This registry transports existing owners; it issues no execution, numerical,
resource or timing qualification. It is host infrastructure outside the shared
public SDK. A source declaration or a registry record cannot construct an owner
inside a compiler sandbox or extend a frozen library/resource grant.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import hashlib
import inspect
import json
import weakref
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MethodType

from merlin.common.digest import sha256_file
from merlin.common.process.cpu_pool import AUTHOR, cpu_priority

# ---- reviewed_pass_capabilities ----------------------------------------------------------------


def _identity(value):
    if type(value) is not str or not value or len(value) > 512 or any(ord(c) < 32 for c in value):
        raise ValueError("pass capability requires an explicit bounded original identity")
    return value


def _mechanism(value):
    _identity(value)
    if (
        len(value) > 96
        or not value[0].isalpha()
        or any(not (c.isascii() and (c.isalnum() or c in "_.-")) for c in value)
    ):
        raise ValueError("pass capability requires an explicit mechanism key")
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class PassCapabilityBinding:
    """Original exact identities, not target names or approximate compatibility.

    The original owner supplies all five identities. The registry neither
    derives them from labels nor fills missing runtime/library/numerical data.
    """

    hardware: str
    software: str
    numerical_policy: str
    runtime: str
    library: str

    def __post_init__(self):
        for value in asdict(self).values():
            _identity(value)


def _verification(owner):
    callback = getattr(owner, "revalidate", None)
    if type(callback) is not MethodType or callback.__self__ is not owner:
        raise ValueError("pass capability requires its original Python qualification owner")
    source = inspect.getsourcefile(callback.__func__)
    if source is None or Path(source).is_symlink() or not Path(source).is_file():
        raise ValueError("pass capability verification source is absent or linked")
    return callback.__func__, callback.__func__.__code__, Path(source).resolve(), sha256_file(Path(source))


class ReviewedPassCapability:
    """A live original owner supplied by the host, never a copied JSON verdict.

    The caller must obtain the owner from its independent qualification route.
    This wrapper pins that owner and its verification implementation; it cannot
    establish the owner's claims. Candidate code receives no registry-issuance
    or resource permissions through this module.
    """

    def __init__(self, mechanism, binding, owner):
        if type(binding) is not PassCapabilityBinding:
            raise ValueError("pass capability requires an exact original binding")
        self._mechanism, self._binding, self._owner = _mechanism(mechanism), binding, owner
        self._original_owner = owner
        self._original_binding = asdict(binding)
        self._owner_identity = _identity(getattr(owner, "identity", None))
        self._verification = _verification(owner)
        self._record = dict(
            schema="merlin.reviewed_pass_capability.v1",
            mechanism=mechanism,
            binding=asdict(binding),
            owner_identity=self._owner_identity,
            verifier_source_sha256=self._verification[3],
            authority="original owner required; registry issues no qualification",
        )
        self._identity = _digest(self._record)
        self.revalidate()

    @property
    def identity(self):
        self.revalidate()
        return self._identity

    @property
    def mechanism(self):
        self.revalidate()
        return self._mechanism

    @property
    def binding(self):
        self.revalidate()
        return self._binding

    @property
    def owner(self):
        self.revalidate()
        return self._owner

    def record(self):
        self.revalidate()
        return copy.deepcopy(self._record)

    def revalidate(self):
        function, code, source, digest = self._verification
        if (
            type(self) is not ReviewedPassCapability
            or self._owner is not self._original_owner
            or type(self._binding) is not PassCapabilityBinding
            or asdict(self._binding) != self._original_binding
            or _verification(self._owner) != (function, code, source, digest)
            or getattr(self._owner, "identity", None) != self._owner_identity
            or self._mechanism != self._record["mechanism"]
            or _digest(self._record) != self._identity
        ):
            raise ValueError("original pass capability owner, binding or implementation changed")
        # The original callback checks execution/effect/resource/qualification
        # evidence. A digest of a copied receipt never replaces that live owner.

        if not _metadata_verified(self):
            self._owner.revalidate()
        if (
            _verification(self._owner) != (function, code, source, digest)
            or getattr(self._owner, "identity", None) != self._owner_identity
        ):
            raise ValueError("original pass capability changed during qualification recheck")


class ReviewedPassCapabilities:
    """Bounded explicit registry; absent or ambiguous capability never enables work."""

    def __init__(self, capabilities=()):
        if type(capabilities) is not tuple or len(capabilities) > 128:
            raise ValueError("pass registry requires a bounded immutable original owner roster")
        self._capabilities = capabilities
        self._records = []
        keys = set()
        for capability in capabilities:
            if type(capability) is not ReviewedPassCapability:
                raise ValueError("pass registry accepts only original reviewed owners")
            record = capability.record()
            key = (record["mechanism"], _digest(record["binding"]))
            if key in keys:
                raise ValueError("pass registry has ambiguous mechanism/binding owners")
            keys.add(key)
            self._records.append(record)
        self._identity = _digest(self._records)
        self.revalidate()

    @property
    def identity(self):
        self.revalidate()
        return self._identity

    def revalidate(self):
        if (
            type(self) is not ReviewedPassCapabilities
            or type(self._capabilities) is not tuple
            or [capability.record() for capability in self._capabilities] != self._records
            or _digest(self._records) != self._identity
        ):
            raise ValueError("original reviewed pass registry changed")

    def record(self):
        self.revalidate()
        return dict(
            schema="merlin.reviewed_pass_capabilities.v1",
            identity=self._identity,
            capabilities=copy.deepcopy(self._records),
            missing_capability="disabled; no nearest-target or source-only fallback",
        )

    def resolve(self, mechanism, binding):
        """Reopen every owner, then require an exact original mechanism/binding."""
        self.revalidate()
        _mechanism(mechanism)
        if type(binding) is not PassCapabilityBinding:
            raise ValueError("pass registry lookup requires an exact original binding")
        rows = [
            capability
            for capability in self._capabilities
            if capability.mechanism == mechanism and capability.binding == binding
        ]
        if len(rows) > 1:
            raise ValueError("pass registry has ambiguous original owners")
        return rows[0] if rows else None


# ---- reviewed_metadata -------------------------------------------------------------------------
# One original qualification check around a trusted metadata operation.
#
# This coalesces repeated historical-owner checks in nested registry getters.
# It never caches current program, source, numerical, execution or timing checks.
# The caller must finish the operation before consuming or publishing its result.

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
        # Only the blocking owner-validation leaves outrank grading work.
        # The metadata operation body retains its caller priority and environment.
        with cpu_priority(AUTHOR, export=False):
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
                with cpu_priority(AUTHOR, export=False):
                    registry.revalidate()
            except BaseException as error:
                if primary is None:
                    raise
                primary.add_note("Original registry final metadata/qualification check failed: " + str(error))
            finally:
                _ISSUED.discard(operation)
    finally:
        _ACTIVE.reset(disabled)
