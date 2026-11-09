"""Transitional identities for placement's compiler-owned implementation.

Retained for existing provider/test imports; new callers use compiler.planning.
Remove after those consumers migrate. No selection policy is implemented here.
"""

from merlin.compiler.planning.candidates import HOST_DEVICE, host_units, units_for
from merlin.compiler.planning.cost import _declared_rate, measured_cost_for
from merlin.compiler.planning.placement import Placed, Placement, device_selector, place

__all__ = [
    "HOST_DEVICE",
    "Placement",
    "Placed",
    "device_selector",
    "host_units",
    "place",
    "units_for",
    "measured_cost_for",
    "_declared_rate",
]
