"""Enumerate source contractions without granting numerical conversion.

Hardware facts remain in ``system.offload``. This shared planner query checks
source shape and initialization; a backend still owns conversion admission.
"""

from __future__ import annotations

from typing import Any

__all__ = ["catalog_source_contractions"]


def catalog_source_contractions(
    module, device_name: str, source_dtypes: tuple[str, str, str], *, require_zero_init: bool = True
) -> list[tuple[Any, Any]]:
    """Structural candidates for an explicit source-bound conversion catalog.

    The catalog's source ABI is checked separately from the device's native
    datapaths. A target-owned conversion contract names one or more native
    target triples; this query must not pretend its f32 (or other source)
    tensors are natively consumed by the device. Without native device facts,
    even an explicit catalog has no grounded target to route to.
    """
    from merlin.system.offload import device_dtype_triples

    if len(source_dtypes) != 3 or not all(isinstance(value, str) and value for value in source_dtypes):
        raise ValueError("catalog source dtypes must be an explicit triple")
    if not device_dtype_triples(device_name):
        return []
    return contractions_for_dtypes(module, device_name, (source_dtypes,), require_zero_init=require_zero_init)


def contractions_for_dtypes(module, device_name, triples, *, require_zero_init: bool):
    from merlin.compiler.ir.contraction_shapes import observe_contractions
    from merlin.system.offload import device_contraction_ranks, why_not

    ranks = device_contraction_ranks(device_name)

    picked: list[tuple[Any, Any]] = []
    for op, shape in observe_contractions(module):
        if why_not(shape, triples=triples, ranks=ranks) is not None:
            continue
        if require_zero_init:
            from merlin.compiler.ir.contraction_shapes import zero_initialised

            if not zero_initialised(op):
                continue
        picked.append((op, shape))
    return picked
