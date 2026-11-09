"""Original allocator effects and per-island packing obligations.

These immutable declarations are premises, not runtime or compiler proofs.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PackingAllocationEffects:
    """Explicit original runtime premises, never inferred from symbol spellings.

    Fresh private allocation means a disjoint contiguous object of the complete
    requested byte extent. Success includes non-null, representable storage.
    The separately selected pointer width must match every address conversion.
    Releases are possible only through the selected deallocator. These are
    original qualification premises; this record proves none of them.
    """

    allocator_symbol: str
    deallocator_symbol: str
    qualification_identity: str
    pointer_bits: int
    fresh_private_allocations: bool = field(kw_only=True)
    successful_allocations: bool = field(kw_only=True)
    releases_only_via_deallocator: bool = field(kw_only=True)

    def __post_init__(self):
        if (
            any(
                type(s) is not str or not s or len(s) > 4096
                for s in (self.allocator_symbol, self.deallocator_symbol, self.qualification_identity)
            )
            or self.allocator_symbol == self.deallocator_symbol
            or type(self.pointer_bits) is not int
            or self.pointer_bits not in (32, 64)
            or any(
                v is not True
                for v in (
                    self.fresh_private_allocations,
                    self.successful_allocations,
                    self.releases_only_via_deallocator,
                )
            )
        ):
            raise ValueError("packing storage needs original complete allocator effects")


@dataclass(frozen=True)
class PackingStorageUse:
    wrapper_symbol: str
    argument: int

    def __post_init__(self):
        if (
            type(self.wrapper_symbol) is not str
            or not self.wrapper_symbol
            or type(self.argument) is not int
            or self.argument < 0
        ):
            raise ValueError("packing storage requires an original positional call argument")


@dataclass(frozen=True)
class PackingStorageRequest:
    """One original island's use roster and logical-payload budget."""

    uses: tuple[PackingStorageUse, ...]
    max_payload_bytes: int

    def __post_init__(self):
        if (
            type(self.uses) is not tuple
            or not self.uses
            or any(type(use) is not PackingStorageUse for use in self.uses)
            or type(self.max_payload_bytes) is not int
            or self.max_payload_bytes <= 0
            or len({(use.wrapper_symbol, use.argument) for use in self.uses}) != len(self.uses)
        ):
            raise ValueError("packing storage needs an immutable nonempty island request and positive budget")
        for use in self.uses:
            use.__post_init__()
