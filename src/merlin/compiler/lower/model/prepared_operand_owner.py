"""Explicit, invocation-local ownership for immutable prepared tensor operands.

This is a proof/placement prototype, not a cache or permission to delete source
operations. A lowering must validate the representation producer and borrowed
consumer effects independently before using the returned storage plan. Source
SSA equivalence alone never establishes physical buffer lifetime or purity.
"""

from __future__ import annotations

from dataclasses import dataclass

from merlin.compiler.lower.model.ordered_fma_groups import _snapshot
from merlin.compiler.lower.model.tensor_preparation_identity import validate_tensor_preparation_opportunity


def _pin(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("prepared representation requires a complete proof identity")


@dataclass(frozen=True)
class PreparedStorageSpan:
    """Bytes and alignment are derived by the selected representation producer."""

    name: str
    bytes: int
    alignment: int

    def __post_init__(self):
        if not self.name or type(self.bytes) is not int or not 0 < self.bytes < 1 << 63:
            raise ValueError("invalid prepared storage extent")
        if type(self.alignment) is not int or not 0 < self.alignment < 1 << 63 or self.alignment & (self.alignment - 1):
            raise ValueError("prepared storage alignment must be a power of two")


@dataclass(frozen=True)
class PreparedRepresentation:
    """Format includes axis order, dtype, radix, scales and reconstruction order.

    Numeric and effect evidence must cover source RNE, finite admission,
    initialization, immutable prepared/source spans, and synchronous noescape
    consumption. These pins name separately checked evidence; they are not a
    substitute for checking it in the normal compiler/provider binding.
    """

    format_sha256: str
    numeric_sha256: str
    effects_sha256: str
    spans: tuple[PreparedStorageSpan, ...]

    def __post_init__(self):
        for pin in (self.format_sha256, self.numeric_sha256, self.effects_sha256):
            _pin(pin)
        if not self.spans or len({span.name for span in self.spans}) != len(self.spans):
            raise ValueError("prepared spans must have distinct ownership names")

    def layout(self):
        offset, result = 0, []
        for span in self.spans:
            offset = (offset + span.alignment - 1) & -span.alignment
            end = offset + span.bytes
            if end >= 1 << 63:
                raise ValueError("prepared storage exceeds signed index range")
            result.append((span.name, offset, end))
            offset = end
        return tuple(result), offset, max(span.alignment for span in self.spans)


class PreparedOperandOwner:
    """Linear proof-state model for one owner and one source invocation epoch.

    No addresses or tensor values are used as identity keys. Each consumer must
    present its actual source operation and the original epoch capability.
    Invalidation is terminal; a later invocation requires a fresh owner.
    """

    def __init__(self, opportunity, representation, *, validate_effects):
        validate_tensor_preparation_opportunity(opportunity)
        if opportunity.format_sha256 != representation.format_sha256:
            raise ValueError("source and physical preparation formats disagree")
        positions = {op: i for i, op in enumerate(opportunity.owner_block.ops)}
        ordered = [positions[op] for op in opportunity.consumers]
        if ordered != sorted(set(ordered)):
            raise ValueError("prepared consumers must be unique and in source order")
        # The binding-specific validator must actually check provider evidence.
        # Refusal propagates; no bool option silently authorizes purity.
        if validate_effects(opportunity, representation) is not None:
            raise ValueError("effect validator must raise on refusal, not return a permission flag")
        self.opportunity = opportunity
        self.representation = representation
        self.layout, self.capacity, self.alignment = representation.layout()
        self._block = tuple(_snapshot(op) for op in opportunity.owner_block.ops)
        self._epoch = object()
        self._next = 0
        self._active = True
        self._storage_binding = None

    @property
    def epoch(self):
        return self._epoch

    def invalidate(self):
        self._active = False

    def consume(self, consumer, *, epoch, representation, capacity, alignment, spans, storage_base, live_read_ranges):
        if not self._active or epoch is not self._epoch:
            raise ValueError("stale prepared owner epoch")
        try:
            validate_tensor_preparation_opportunity(self.opportunity)
            if tuple(_snapshot(op) for op in self.opportunity.owner_block.ops) != self._block:
                raise ValueError("prepared owner source schedule changed")
            if representation != self.representation:
                raise ValueError("prepared representation changed")
            if consumer is not self.opportunity.consumers[self._next]:
                raise ValueError("prepared consumer order or ownership mismatch")
            if (
                type(capacity) is not int
                or capacity < self.capacity
                or type(alignment) is not int
                or alignment < self.alignment
                or alignment % self.alignment
            ):
                raise ValueError("prepared allocation is insufficient")
            if tuple(spans) != self.layout:
                raise ValueError("prepared spans overlap or differ from proved layout")
            if (
                type(storage_base) is not int
                or storage_base <= 0
                or storage_base % self.alignment
                or storage_base + capacity >= 1 << 64
            ):
                raise ValueError("prepared allocation address is invalid")
            live_read_ranges = tuple(live_read_ranges)
            if not live_read_ranges:
                raise ValueError("source allocation disjointness is unknown")
            for lo, hi in live_read_ranges:
                if type(lo) is not int or type(hi) is not int or not 0 < lo < hi < 1 << 64:
                    raise ValueError("source allocation interval is invalid")
                if storage_base < hi and lo < storage_base + capacity:
                    raise ValueError("prepared allocation aliases live source storage")
            binding = (storage_base, capacity, alignment, live_read_ranges)
            if self._storage_binding is not None and binding != self._storage_binding:
                raise ValueError("prepared storage owner changed within epoch")
            self._storage_binding = binding
        except Exception:
            self.invalidate()
            raise
        self._next += 1
        if self._next == len(self.opportunity.consumers):
            self.invalidate()
        return self.layout
