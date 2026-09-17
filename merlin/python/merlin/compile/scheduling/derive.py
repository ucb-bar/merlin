"""Derive a block-schedule :class:`~merlin.compile.scheduling.block_schedule.Geometry` from a target's facts.

Kept apart from ``block_schedule.py`` so that module imports nothing from merlin and a generated backend
can vendor it verbatim; this is the one place the pass meets :mod:`merlin.targetgen.address_space`.
"""

from __future__ import annotations

from typing import Any

from merlin.targetgen.address_space import ADDRESSABLE, accumulator_kind, operand_store

from .block_schedule import BlockScheduleError, Geometry


def geometry_from_address_space(space: Any) -> Geometry:
    """Derive from :class:`~merlin.targetgen.address_space.AddressSpace`, or refuse.

    Refuses rather than defaults on: no array geometry, a non-square array (this weight-stationary
    model has one square block edge, and which edge a rectangular array's block spans is a choice
    no fact here makes), a missing operand or accumulator store, and a store whose row count the
    facts could not derive. Stores are resolved to their roles by ROW WIDTH
    (:func:`~merlin.targetgen.address_space.operand_store`,
    :func:`~merlin.targetgen.address_space.accumulator_store`), never by the name an extractor
    happened to give them, and a refusal quotes the resolver's reason.
    """
    if space.array_rows is None or space.array_cols is None:
        raise BlockScheduleError(
            f"{space.target!r}: no array geometry in its facts, so there is no block edge to "
            f"schedule in (unknowns: {list(space.unknown_quantities())})"
        )
    if space.array_rows != space.array_cols:
        raise BlockScheduleError(
            f"{space.target!r}: a {space.array_rows}x{space.array_cols} array is not square; this "
            "pass schedules one square block edge and will not choose an edge for you"
        )
    kind = accumulator_kind(space)
    resolved = {"operand": operand_store(space), "accumulator": kind}
    if resolved["operand"].store is None:
        raise BlockScheduleError(f"{space.target!r}: no operand store to schedule into: {resolved['operand'].reason}")
    if kind.kind != ADDRESSABLE:
        # An accumulator inside the compute element is a real design, not a defect in the facts --
        # but this pass addresses accumulator ROWS, and such an accumulator has none to address.
        raise BlockScheduleError(
            f"{space.target!r}: its accumulator is {kind.kind} ({kind.reason}); this pass schedules "
            "into an addressable accumulator store"
        )
    for role, resolution in resolved.items():
        if resolution.store.total_rows is None:
            raise BlockScheduleError(
                f"{space.target!r}: the row count of its {role} store "
                f"{resolution.store.name!r} is UNKNOWN "
                f"({[u.reason for u in space.unknowns if u.store == resolution.store.name]})"
            )
    operand, accumulator = resolved["operand"].store, resolved["accumulator"].store
    if operand.row_elems is not None and operand.row_elems != space.array_rows:
        raise BlockScheduleError(
            f"{space.target!r}: its operand row spans {operand.row_elems} elements but its array "
            f"edge is {space.array_rows}; a block cannot be one row wide and another edge tall"
        )
    return Geometry(
        block=space.array_rows,
        operand_rows=operand.total_rows,
        operand_bank_rows=operand.depth,
        accumulator_rows=accumulator.total_rows,
        separate_accumulator_space=bool(space.separate_accumulator_space),
        sources={
            "facts": space.sources.get("facts", "derive_address_space"),
            "block": f"arrays[{space.array_name!r}] edge",
            "operand_rows": (f"{operand.name}.total_rows (operand store, {resolved['operand'].basis})"),
            "operand_bank_rows": f"{operand.name}.depth",
            "accumulator_rows": (
                f"{accumulator.name}.total_rows (addressable accumulator "
                f"linked to datapath {kind.datapath!r}, {kind.dtype})"
            ),
        },
    )
