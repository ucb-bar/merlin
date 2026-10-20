"""Original-host choices bound later to one actual prepared compilation input.

This is an engineering policy, not a pass qualification, restored capability or
sandbox grant. The original host supplies both permissions and ceilings before
preparation. The ordinary build derives the existing exact-SHA request once;
static request semantics and independently checked pass legality are unchanged.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .source_optimization_request import _PASSES, SourceOptimizationRequest, _budget
from .source_optimization_selection import SourceOptimizationSelection

SCHEMA = "merlin.source_optimization_policy.v1"


@dataclass(frozen=True)
class SourceOptimizationPolicy:
    """Fixed choices without an invented source identity or baseline execution.

    The construction budget applies separately to each selected source pass. It
    grants neither memory capacity nor permission to duplicate scalar work.
    This original-host object is never reconstructed from a build receipt.
    """

    producer_packing: bool
    padding_gather: bool
    maximum_added_operations: int
    permitted_passes: tuple[str, ...]
    original_maximum_added_operations: int
    schema: str = SCHEMA

    def __post_init__(self):
        if type(self.schema) is not str or self.schema != SCHEMA:
            raise ValueError("unknown original source optimization policy schema")
        if type(self.producer_packing) is not bool or type(self.padding_gather) is not bool:
            raise ValueError("original source policy requires explicit boolean choices")
        if (
            type(self.permitted_passes) is not tuple
            or any(type(name) is not str or name not in _PASSES for name in self.permitted_passes)
            or len(set(self.permitted_passes)) != len(self.permitted_passes)
        ):
            raise ValueError("original source policy requires a closed permission tuple")
        _budget(self.maximum_added_operations)
        _budget(self.original_maximum_added_operations)
        if self.maximum_added_operations > self.original_maximum_added_operations:
            raise ValueError("original source policy exceeds its construction ceiling")
        for name, enabled in zip(_PASSES, (self.producer_packing, self.padding_gather), strict=True):
            if enabled and name not in self.permitted_passes:
                raise ValueError("original source policy choice lacks original permission")

    def record(self):
        self.__post_init__()
        return {
            "schema": self.schema,
            "producer_packing": self.producer_packing,
            "padding_gather": self.padding_gather,
            "maximum_added_operations": self.maximum_added_operations,
            "original_permitted_passes": list(self.permitted_passes),
            "original_maximum_added_operations": self.original_maximum_added_operations,
        }

    def bind(self, source):
        """Derive ordinary request data; the build owns the one-time binding."""
        self.__post_init__()
        if type(source) is not str:
            raise ValueError("original source policy requires exact prepared source text")
        request = SourceOptimizationRequest(
            hashlib.sha256(source.encode()).hexdigest(),
            self.producer_packing,
            self.padding_gather,
            self.maximum_added_operations,
        )
        selection = SourceOptimizationSelection(request, self.permitted_passes, self.original_maximum_added_operations)
        self.verify_selection(selection, source)
        return selection

    def verify_selection(self, selection, source):
        """Check correspondence using ORIGINAL limits, not returned permissions."""
        self.__post_init__()
        if type(selection) is not SourceOptimizationSelection:
            raise ValueError("original source policy requires its exact bound selection")
        request = selection.request
        if (
            type(request) is not SourceOptimizationRequest
            or request.producer_packing is not self.producer_packing
            or request.padding_gather is not self.padding_gather
            or request.maximum_added_operations != self.maximum_added_operations
            or selection.permitted_passes != self.permitted_passes
            or selection.maximum_added_operations != self.original_maximum_added_operations
        ):
            raise ValueError("bound source choices differ from original policy")
        selection.bind(source)
