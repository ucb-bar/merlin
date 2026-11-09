"""Operation offers and the existing explicitly selected matrix implementation.

The offers/lower split adapts OPU-VME's extension_offers at revision
2f513d484f5d68016811139344ffdfbf8d23cd95. This branch's original selector is
Boolean and has no cost model: its offered costs remain unknown, never zero.

These are provider claims for one queried operation descriptor. They do not
establish numerical legality, hardware capability, source-IR identity, resource
independence, or observed execution. Existing provider and build admission stays
mandatory. This is not the full heterogeneous region proposal protocol.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OperationOffer:
    """One provider's declared implementation choice, with an optional cycle estimate."""

    owner: str
    unit: str
    estimated_cycles: float | None = None
    detail: Any = None

    def __post_init__(self) -> None:
        for name, value in (("owner", self.owner), ("unit", self.unit)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"an offer requires a nonempty {name}")
        cost = self.estimated_cycles
        if cost is not None and (not math.isfinite(cost) or cost < 0):
            raise ValueError("an offer needs finite nonnegative estimated cycles or None")


@dataclass(frozen=True)
class MatrixRouting:
    """Which matrix extension a build routes contractions to, and in which configuration.

    Carried as one object so the tile edge has exactly ONE source. The alternative -- deriving the edge
    here from the caller's ``vlen`` and again at object-build time from the config's Scala -- is two
    statements of the same fact that can disagree, and the disagreement would be silent: the selector
    would choose contractions for a geometry the compiled kernel does not have.

    ``select`` overrides the default tile-filling decision, which is the seam the cost model and the
    e-graph plug into. Nothing here decides profitability on its own.

    ``support_target`` selects executable OOT support on ``MERLIN_TARGET_PATH``;
    it is not inferred from ``unit`` or ``config``. The provider declares
    ``plugin.matrix_lowering`` with geometry, selector, rewrite_prepared_file,
    load_signatures and build_object callables. Rewrite reports preserve count,
    signatures, skipped and sub_tile(); builds expose object_path, tile_edge,
    scalar_tile, scratch_bytes, parallel_tiles and to_dict().
    """

    support_target: str  # explicitly selected OOT support identity
    unit: str  # a unit declared by that provider
    config: str  # the elaborated hardware configuration
    select: Callable[[Any], bool] | None = None
    kernel: str | None = None  # explicit registry name or source path; None retains the provider default

    def __post_init__(self) -> None:
        for name, value in self.identity().items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"matrix routing requires an explicit nonempty {name}")
        if self.kernel is not None and (not isinstance(self.kernel, str) or not self.kernel.strip()):
            raise ValueError("matrix kernel selection must be a nonempty name or source path")

    def provider(self):
        """Resolve executable support explicitly, never from the unit's name."""
        from merlin.targetgen.plugins import PluginError, load_declared

        provider = load_declared(self.support_target, "matrix_lowering")
        required = ("geometry", "selector", "rewrite_prepared_file", "load_signatures", "build_object")
        missing = [name for name in required if not callable(getattr(provider, name, None))]
        if missing:
            raise PluginError(f"{self.support_target!r}: matrix_lowering lacks callable {missing}")
        return provider

    def identity(self) -> dict[str, str]:
        return {"support_target": self.support_target, "unit": self.unit, "config": self.config}

    def tile_edge(self) -> int:
        """The edge, read from this configuration's own declaration.

        Through the contract's own accessor rather than re-resolving the paths here: this used to rebuild
        them, which meant two places had to agree about where a unit's configs live -- and when the contract
        grew a second declaration site they stopped agreeing.
        """
        return self.provider().geometry(unit=self.unit, config=self.config)[0]

    def selector(self) -> Callable[[Any], bool]:
        return self.select if self.select is not None else self.provider().selector(self.tile_edge())


class MatrixExtension:
    """Bind the current matrix provider and expose its actual selection as offers.

    Provider resolution, the original selector and configuration geometry are
    evaluated in the same order as the preparation caller previously used.
    Every query invokes the original selector exactly once; no answers are
    cached by shapes or used to infer support for another operation.
    """

    def __init__(self, routing: MatrixRouting) -> None:
        self.routing = routing
        self.provider = routing.provider()
        self._selector = routing.selector()
        self.tile_edge = routing.tile_edge()

    def offers(self, operation: Any) -> tuple[OperationOffer, ...]:
        if self._selector(operation):
            return (OperationOffer(self.routing.support_target, self.routing.unit, detail=operation),)
        return ()

    def __call__(self, operation: Any) -> bool:
        return bool(self.offers(operation))

    def __getattr__(self, name: str) -> Any:
        # Some explicit selectors expose additional provider-observed metadata.
        # Forward it to the original object, as the OPU adapter does.
        return getattr(self._selector, name)

    def lower(self, prepared: Any, work: Any) -> Any:
        return self.provider.rewrite_prepared_file(prepared, work, select=self, tile_edge=self.tile_edge)
