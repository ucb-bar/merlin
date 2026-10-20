"""Bounded source-pass choices, separated from original caller permissions.

This data protocol issues no source, numerical, runtime or import authority.
The original host supplies the current source, permitted passes and construction
ceiling at consumption. Existing host-request schemas and SDK inventories are
unchanged. Execution still needs each ordinary pass's independent checks.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

SCHEMA = "merlin.source_optimization_request.v1"
PACKING_PASS = "automatic-host-permutations"
GATHER_PASS = "automatic-host-gathers"
_PASSES = (PACKING_PASS, GATHER_PASS)
_FIELDS = ("schema", "source_sha256", "producer_packing", "padding_gather", "maximum_added_operations")
_MAXIMUM_OPERATIONS = 1 << 20


def _budget(value):
    if type(value) is not int or not 0 <= value <= _MAXIMUM_OPERATIONS:
        raise ValueError("source optimization construction budget must be a bounded nonnegative integer")


@dataclass(frozen=True)
class SourceOptimizationRequest:
    """Explicit choices for one exact current source, never an admission.

    ``maximum_added_operations`` bounds construction separately for each
    selected pass. It is not net growth, a shared work budget, memory capacity
    or permission to duplicate scalar arithmetic. Zero permits no construction.
    """

    source_sha256: str
    producer_packing: bool
    padding_gather: bool
    maximum_added_operations: int
    schema: str = SCHEMA

    def __post_init__(self):
        if type(self.schema) is not str or self.schema != SCHEMA:
            raise ValueError("unknown source optimization request schema")
        if (
            type(self.source_sha256) is not str
            or len(self.source_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.source_sha256)
        ):
            raise ValueError("source optimization requires an exact SHA256 identity")
        if type(self.producer_packing) is not bool or type(self.padding_gather) is not bool:
            raise ValueError("source optimization choices must be explicit booleans")
        _budget(self.maximum_added_operations)

    def record(self):
        self.__post_init__()
        return {name: getattr(self, name) for name in _FIELDS}

    def bind(self, current_source, *, permitted_passes, maximum_added_operations):
        """Return only checked optional-pass states and the bounded request limit.

        Caller arguments must come from the live original host selection, not
        this record. Source hashes establish correspondence, not authorization.
        Both known states are explicit, overriding their environment defaults;
        unrelated ambient pass choices and all legality gates remain external.
        This neither installs a selection nor runs a pass or callback.
        """
        self.__post_init__()
        if type(current_source) is not str:
            raise ValueError("source optimization requires the exact current source text")
        if hashlib.sha256(current_source.encode()).hexdigest() != self.source_sha256:
            raise ValueError("source optimization request does not bind the current source")
        if (
            type(permitted_passes) is not tuple
            or any(type(name) is not str or name not in _PASSES for name in permitted_passes)
            or len(set(permitted_passes)) != len(permitted_passes)
        ):
            raise ValueError("source optimization requires a closed original permission tuple")
        _budget(maximum_added_operations)
        if self.maximum_added_operations > maximum_added_operations:
            raise ValueError("source optimization request exceeds its original construction ceiling")
        enabled = tuple(name for name, choice in zip(_PASSES, (self.producer_packing, self.padding_gather)) if choice)
        if any(name not in permitted_passes for name in enabled):
            raise ValueError("source optimization choice lacks original permission")
        from .optional_passes import Selection

        selection = Selection.of(enable=enabled, disable=tuple(name for name in _PASSES if name not in enabled))
        return selection, self.maximum_added_operations


def request_from_record(record):
    """Read only the closed data schema; unknown authority fields refuse."""
    if type(record) is not dict or set(record) != set(_FIELDS):
        raise ValueError("source optimization request has unknown or missing fields")
    return SourceOptimizationRequest(**record)
