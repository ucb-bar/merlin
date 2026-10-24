"""Which capsules one (family, dtype, alignment) conformance cell needs: one per declared capability row.

A family one unit declares more than once -- fused behind a producer, and standalone for a narrower
reduction kind -- folds into a capability whose ``composed_with`` intersects to nothing and whose
``kinds`` union to everything, a view no single row declares. The eligibility oracle checks the rows
any-of, so synthesis covers each of them; :mod:`merlin.targetgen.corpus_synth` writes the entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from merlin.targetgen import semantic_families as _sf


@dataclass(frozen=True)
class CellForm:
    """One capsule a cell needs: the op (``None`` when nothing can write it), its fused epilogue, and
    the declared capability row it exercises (``composed`` for a fused-only row, ``kinds`` for a
    kind-restricted standalone row)."""

    op: str | None
    epilogue: tuple[str, ...] = ()
    composed: tuple[str, ...] = ()
    kinds: tuple[str, ...] = ()

    def qualifier(self) -> str:
        if self.kinds:
            return "_".join(self.kinds)
        return "fused" if self.composed else "standalone"

    def row_note(self) -> str:
        if self.composed:
            return f", row fused with {list(self.composed)}"
        return f", standalone row for {list(self.kinds)} reductions" if self.kinds else ""


def _declared_rows(cap: Any, dtype: str) -> tuple:
    """The capability rows ``cap`` was folded from that admit ``dtype`` -- every row when none does.

    READ PER ROW, NOT FOLDED. A family one unit declares twice (fused behind a producer, and standalone
    for a narrower reduction kind) folds into a view that intersects ``composed_with`` to nothing and
    unions ``kinds`` to everything, so the folded view claims an unrestricted standalone family that no
    row declares. The eligibility oracle checks the rows any-of; synthesis has to cover each of them.
    """
    if cap is None:
        return ()
    from merlin.targetgen.eligibility import _dtype_ok

    rows = tuple(cap.rows()) if hasattr(cap, "rows") else (cap,)
    return tuple(r for r in rows if _dtype_ok(dtype, tuple(getattr(r, "dtypes", ()) or ()))) or rows


def _row_op(family: str, kinds: tuple, gradeable: set[str], dtype: str) -> str | None:
    """The op exercising a STANDALONE row: any op of ``family``, or, when the row restricts its reduction
    kinds, only an op whose reduction kind is one of them (never guessed from the op's spelling)."""
    from .corpus_synth import op_for_family

    if not kinds:
        return op_for_family(family, admitted_ops=gradeable, dtype=dtype)
    admitted = {op for op in gradeable if _sf.reduction_kind_of_op(op) in kinds}
    return op_for_family(family, admitted_ops=admitted, dtype=dtype)


def cell_forms(family: str, dtype: str, cap: Any, pool: set[str]) -> list[CellForm]:
    """The capsules one (family, dtype, alignment) cell needs: one per declared capability row.

    A fused-only row (``composed_with`` a contraction) is satisfied by a contraction carrying the family
    as an epilogue; a standalone row by a standalone op of the family, restricted to the row's reduction
    kinds when it declares any. A family declared by one row (or not at all) needs exactly the one
    capsule it always did, including the standalone fallback when no carrier can be built.
    """
    # The CELL'S dtype narrows the pool twice over, and both narrowings are needed: what the GOLDEN
    # ENGINE can grade at this dtype (ops_gradeable_at) and, within that, what a WRITER can express (the
    # dtype-aware ranking). Buildable is not gradeable and gradeable is not writable; choosing on any one
    # of the three alone produced an entry that died in the other.
    from .corpus_synth import _fused_carrier, ops_gradeable_at

    gradeable = ops_gradeable_at(dtype, pool)
    rows = _declared_rows(cap, dtype)
    if len(rows) <= 1:
        # `composed_with` is a tuple ATTRIBUTE, not a method. Calling it raised TypeError, a broad
        # `except` swallowed it, and every family looked standalone -- so the fused-only case silently
        # took the branch that produces a capsule the eligibility oracle refuses. Read the attribute,
        # and let anything unexpected surface.
        composed = tuple(getattr(rows[0], "composed_with", ()) or ()) if rows else ()
        kinds = tuple(getattr(rows[0], "kinds", ()) or ()) if rows else ()
        fused = _fused_carrier(family, composed, gradeable) if composed else None
        if fused is not None:
            return [CellForm(fused[0], tuple(fused[1]), composed)]
        return [CellForm(_row_op(family, kinds, gradeable, dtype), (), (), kinds)]
    forms: list[CellForm] = []
    for row in rows:
        composed = tuple(getattr(row, "composed_with", ()) or ())
        if composed:
            fused = _fused_carrier(family, composed, gradeable)
            form = CellForm(fused[0], tuple(fused[1]), composed) if fused else CellForm(None, (), composed)
        else:
            kinds = tuple(getattr(row, "kinds", ()) or ())
            form = CellForm(_row_op(family, kinds, gradeable, dtype), (), (), kinds)
        # Two rows one capsule already satisfies need it once: same op, same epilogue.
        if form.op is not None and any((f.op, f.epilogue) == (form.op, form.epilogue) for f in forms):
            continue
        forms.append(form)
    return forms
