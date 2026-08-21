"""Find gathers whose row index is a VALUE of a bundle input, so the table can be specialized.

WHY THIS EXISTS. A model2MLIR embedding lookup lowers to a `linalg.generic` whose body indexes a
weight table with the *value* of an input element::

    ^bb0(%id: i64, %out: f32):
      %r = arith.index_cast %id : i64 to index      # row  = the token id VALUE
      %c = linalg.index 2 : index                   # col  = the full trailing extent
      %v = tensor.extract %table[%r, %c]
      linalg.yield %v

For a bundle whose inputs are fixed, only the rows named by those values are ever read. On
gemma2_2b_int8_full_seq8 that is 8 of 256000 rows: the table is 2250.0 MiB of the bundle's 4754.9 MiB
and 99.997% of it is never touched. Dropping the rest is what brought the image under the addressing
limit that kept the whole model from running at all.

THIS IS INPUT SPECIALIZATION, NOT DEAD-CODE ELIMINATION — the distinction is the whole point. The row
indices are not compile-time constants; they are runtime values that this bundle happens to pin. An
earlier note in the one-off slicer called this "DCE over a gather with compile-time-constant indices",
which is wrong and dangerous: acting on that name would license baking input values into a model a
caller expects to be general. So the opportunity is reported, never applied silently, and the caller
that applies it is responsible for recording the specialization in the bundle manifest.

SOUNDNESS. Rewriting the table to its dense subset means the stored index values must be renumbered
to their new positions. That is value-preserving only if the gather is the ONLY consumer of the index
tensor — any other consumer (an attention mask derived from token ids, a second lookup) would silently
receive renumbered tokens and be wrong with no numerical tell. :func:`find_gather_specializations`
therefore checks the use count and reports a single-use failure as a REJECTION rather than an
opportunity.

Everything here is structural (xDSL ops and SSA uses). Nothing is matched against printed IR text:
the index tensor in the model this was built for is `%658`, and a textual search for that also matches
`%6580`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ...common import mlir_query as mq


@dataclass(frozen=True)
class GatherSpecialization:
    """One table that can be reduced to the rows a fixed input actually names."""

    table_arg: int              # func-arg index of the gathered table
    table_shape: list[int]
    table_dtype: str
    index_arg: int              # func-arg index of the input whose VALUES index the table
    index_shape: list[int]
    row_dim: int                # which table dimension the value indexes
    generic_index: int          # position of the owning linalg.generic, for reporting

    @property
    def rows(self) -> int:
        return self.table_shape[self.row_dim]


@dataclass(frozen=True)
class GatherRejection:
    """A gather that ALMOST matched, with why it did not.

    Reported rather than dropped: a near-miss is either a soundness hazard (the index tensor has
    other consumers) or a pattern this analysis should learn to handle. Silently returning fewer
    opportunities would make both invisible.
    """

    table_arg: int | None
    reason: str


def _func(module: Any, func_name: str):
    for fn in mq.walk(module, "func.func"):
        if mq.op_name(fn) == "func.func":
            name = fn.properties.get("sym_name") or fn.attributes.get("sym_name")
            if name is None or func_name in str(name):
                return fn
    return None


def _block_arg_index(value, block) -> int | None:
    """Position of `value` in `block`'s argument list, or None if it is not one of them."""
    for i, a in enumerate(block.args):
        if a is value:
            return i
    return None


def _defining_op(value):
    """The operation that produced `value`, or None when it is a block argument."""
    return getattr(value, "owner", None) if not hasattr(value, "index_of_arg") else None


def find_gather_specializations(
    module: Any, func_name: str = "forward"
) -> tuple[list[GatherSpecialization], list[GatherRejection]]:
    """Report every value-indexed gather in `func_name`, split into safe and rejected.

    Returns ``(specializations, rejections)``. A caller may only act on the first list, and should
    surface the second rather than discarding it.
    """
    fn = _func(module, func_name)
    if fn is None:
        return [], [GatherRejection(None, f"no func.func matching {func_name!r}")]

    fn_block = fn.body.blocks[0]
    found: list[GatherSpecialization] = []
    rejected: list[GatherRejection] = []

    for g_i, generic in enumerate(mq.walk(module, "linalg.generic")):
        if not generic.regions or not generic.regions[0].blocks:
            continue
        body = generic.regions[0].blocks[0]

        for op in body.ops:
            if mq.op_name(op) != "tensor.extract":
                continue
            operands = list(op.operands)
            if len(operands) < 2:
                continue
            table, indices = operands[0], operands[1:]

            table_arg = _block_arg_index(table, fn_block)
            if table_arg is None:
                # Gathering from a computed tensor, not a bundle weight. Not this analysis's job,
                # and not a hazard, so not worth reporting as a near-miss.
                continue

            # Exactly one index must be a value read out of the generic's input; the rest must be
            # the loop's own induction (`linalg.index`), i.e. the full extent of that dimension is
            # swept and no row is partially read.
            value_dims: list[tuple[int, int]] = []   # (table dim, generic operand index)
            swept: list[int] = []
            unknown: list[str] = []
            for dim, idx in enumerate(indices):
                producer = getattr(idx, "owner", None)
                pname = mq.op_name(producer) if producer is not None else "<block-arg>"
                if pname == "linalg.index":
                    swept.append(dim)
                elif pname == "arith.index_cast":
                    src = list(producer.operands)[0]
                    body_arg = _block_arg_index(src, body)
                    if body_arg is None:
                        unknown.append(f"dim {dim}: index_cast of a non-block-argument")
                    else:
                        value_dims.append((dim, body_arg))
                else:
                    unknown.append(f"dim {dim}: produced by {pname}")

            if unknown:
                rejected.append(GatherRejection(
                    table_arg, f"table arg {table_arg}: unhandled index provenance — "
                               + "; ".join(unknown)))
                continue
            if len(value_dims) != 1:
                rejected.append(GatherRejection(
                    table_arg, f"table arg {table_arg}: {len(value_dims)} value-indexed dimensions "
                               "(exactly 1 is handled)"))
                continue
            if len(swept) != len(indices) - 1:
                rejected.append(GatherRejection(
                    table_arg, f"table arg {table_arg}: {len(swept)} swept dimensions for "
                               f"{len(indices)} indices — some extent is not fully read"))
                continue

            row_dim, body_arg = value_dims[0]

            # The body argument maps positionally onto the generic's operands (ins then outs), so
            # this is the tensor the index VALUES come from.
            g_operands = list(generic.operands)
            if body_arg >= len(g_operands):
                rejected.append(GatherRejection(
                    table_arg, f"table arg {table_arg}: body arg {body_arg} has no matching operand"))
                continue
            index_tensor = g_operands[body_arg]
            index_arg = _block_arg_index(index_tensor, fn_block)
            if index_arg is None:
                rejected.append(GatherRejection(
                    table_arg, f"table arg {table_arg}: indices come from a computed tensor, not a "
                               "bundle input, so they are not fixed by the bundle"))
                continue

            # THE SOUNDNESS CONDITION. Specializing renumbers the stored index values, so the gather
            # must be the only thing that reads them.
            n_uses = len(list(index_tensor.uses))
            if n_uses != 1:
                rejected.append(GatherRejection(
                    table_arg, f"table arg {table_arg}: index arg {index_arg} has {n_uses} consumers; "
                               "renumbering its values would corrupt the others"))
                continue

            # Likewise the table itself: another reader would see a table that no longer has the
            # rows it expects.
            t_uses = len(list(table.uses))
            if t_uses != 1:
                rejected.append(GatherRejection(
                    table_arg, f"table arg {table_arg}: table has {t_uses} consumers; slicing it "
                               "would break the others"))
                continue

            t_shape, t_dtype = mq.type_shape_dtype(table.type)
            i_shape, _ = mq.type_shape_dtype(index_tensor.type)
            found.append(GatherSpecialization(
                table_arg=table_arg, table_shape=list(t_shape), table_dtype=t_dtype,
                index_arg=index_arg, index_shape=list(i_shape), row_dim=row_dim,
                generic_index=g_i))

    return found, rejected


def kept_rows(spec: GatherSpecialization, index_values: Iterable[int]
              ) -> tuple[list[int], list[int]]:
    """Rows to keep, and the renumbered index values that address them.

    ``kept[k]`` is an original row of the table and every original value ``v`` becomes the position
    of ``v`` in ``kept``. Duplicate values collapse onto one kept row, so a repeated token costs one
    row rather than one row per occurrence.

    Raises on a value outside the table: an out-of-range index is a broken bundle, and specializing
    it would turn a detectable fault into a silent read of the wrong row.
    """
    values = [int(v) for v in index_values]
    rows = spec.rows
    bad = [v for v in values if v < 0 or v >= rows]
    if bad:
        raise ValueError(
            f"index values out of range for a {rows}-row table: {sorted(set(bad))[:8]}")
    kept = sorted(set(values))
    position = {v: i for i, v in enumerate(kept)}
    return kept, [position[v] for v in values]
