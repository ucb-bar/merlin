"""Explicit original-host selection of independently reviewed pass capabilities.

This registry transports existing owners; it issues no execution, numerical,
resource or timing qualification. It is host infrastructure outside the shared
public SDK. A source declaration or a registry record cannot construct an owner
inside a compiler sandbox or extend a frozen library/resource grant.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MethodType

from merlin.common.digest import sha256_file


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
