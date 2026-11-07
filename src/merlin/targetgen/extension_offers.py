"""Fixed-stage operation offers over a selected fallback backend.

An extension prices an operation and lowers only selected work. The fallback
backend receives everything else. This module arbitrates offers; it does not
estimate hardware costs or import an accelerator package.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

API_VERSION = 1


class OfferError(ValueError):
    """An extension offer is invalid or routing would be ambiguous."""


@dataclass(frozen=True)
class OperationOffer:
    owner: str
    unit: str
    estimated_cycles: float
    detail: Any = None

    def __post_init__(self) -> None:
        if not self.owner or not self.unit:
            raise OfferError("an offer needs its declaring provider and unit")
        if not math.isfinite(self.estimated_cycles) or self.estimated_cycles < 0:
            raise OfferError("an offer needs finite nonnegative estimated cycles")


class ExtensionV1(Protocol):
    """The two executable stages supplied by a unit extension."""

    API_VERSION: int

    def offers(
        self, operation: Any, *, layout_elements: dict[str, int] | None = None
    ) -> tuple[OperationOffer, ...]: ...

    def lower(self, prepared: Any, work: Any, *, select: Callable[[Any], bool]) -> Any: ...


def select_offer(offers: tuple[OperationOffer, ...], *, fallback_cycles: float) -> OperationOffer | None:
    """Select the cheapest offer below fallback; refuse tied extension costs.

    The caller's cost model supplies all estimates. Equal cost with fallback
    leaves the operation on fallback, preserving the current matrix policy.
    """
    if not math.isfinite(fallback_cycles) or fallback_cycles < 0:
        raise OfferError("fallback needs finite nonnegative estimated cycles")
    if not offers:
        return None
    unique: set[tuple[str, str]] = set()
    for offer in offers:
        key = (offer.owner, offer.unit)
        if key in unique:
            raise OfferError(f"duplicate offer from {offer.owner}/{offer.unit}")
        unique.add(key)
    best_cost = min(o.estimated_cycles for o in offers)
    if best_cost >= fallback_cycles:
        return None
    best = [o for o in offers if o.estimated_cycles == best_cost]
    if len(best) != 1:
        raise OfferError(f"equal-cost offers require explicit selection: {[(o.owner, o.unit) for o in best]}")
    return best[0]


class MatrixExtension:
    """Live adapter from the existing matrix plugin to the extension stages.

    Geometry and costs remain in the provider and current matrix cost model.
    The adapter changes neither a route decision nor its reported estimate.
    """

    API_VERSION = API_VERSION

    def __init__(self, routing: Any) -> None:
        from merlin.llvmlower.matrix_route_cost import routing_selector

        self.routing = routing
        self.provider = routing.provider()
        version = getattr(self.provider, "API_VERSION", API_VERSION)  # explicit legacy package adapter
        if version != API_VERSION:
            raise OfferError(f"{routing.support_target}: extension API version {version!r}; need {API_VERSION}")
        self._selector = routing.select if routing.select is not None else routing_selector(routing)

    def offers(self, operation: Any, *, layout_elements: dict[str, int] | None = None) -> tuple[OperationOffer, ...]:
        if callable(getattr(self._selector, "decide", None)):
            decision = self._selector.decide(operation, layout_elements=layout_elements)
            return (
                OperationOffer(
                    self.routing.support_target,
                    self.routing.unit,
                    float(decision.chosen.unit_cycles),
                    decision,
                ),
            )
        if self._selector(operation):
            # An explicitly selected boolean policy has no cycle estimate. A
            # zero-cost offer records that explicit policy without inventing one.
            return (OperationOffer(self.routing.support_target, self.routing.unit, 0.0),)
        return ()

    def lower(self, prepared: Any, work: Any, *, select: Callable[[Any], bool]) -> Any:
        return self.provider.rewrite_prepared_file(prepared, work, select=select, tile_edge=self.routing.tile_edge())

    def selector(self) -> Any:
        if callable(getattr(self._selector, "decide", None)):
            return _CostedSelector(self)
        return _BooleanSelector(self)


class _BooleanSelector:
    def __init__(self, extension: MatrixExtension) -> None:
        self.extension = extension

    def __call__(self, operation: Any) -> bool:
        return bool(self.extension.offers(operation))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.extension._selector, name)


class _CostedSelector(_BooleanSelector):
    def decide(self, operation: Any, *, layout_elements: dict[str, int] | None = None) -> Any:
        offer = self.extension.offers(operation, layout_elements=layout_elements)[0]
        decision = offer.detail
        selected = select_offer((offer,), fallback_cycles=float(decision.chosen.vector_cycles))
        if (selected is not None) != bool(decision.route):
            raise OfferError("extension selection differs from the provider's cost-model decision")
        return decision

    def __call__(self, operation: Any) -> bool:
        return bool(self.decide(operation).route)
