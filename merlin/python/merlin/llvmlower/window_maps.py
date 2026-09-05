"""Why a windowed (pooling / convolution) ``linalg`` op cannot be lowered, said in terms of the op.

MLIR's ``linalg`` verifier requires the CONCATENATION of an op's indexing maps to be invertible in
the projected-permutation sense: ``inversePermutation`` builds the inverse by looking for every
iteration dim as a BARE result expression, and returns null when a dim appears only inside a
compound expression. A strided window read

    (d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 * 2 + d4, d3 * 2 + d5)

uses the window dims ``d4``/``d5`` nowhere else, so an op carrying only that map and an
``(d0, d1, d2, d3)`` output map is rejected at PARSE time -- before any pass runs -- with
``'linalg.generic' op invalid indexing maps are non-invertible``. The failure therefore surfaces as
an unreadable "upstream lowering failed" dump from the pipeline's own MLIR reader, hundreds of lines
from the op that caused it.

The rejection is not arbitrary. Nothing downstream can pick the reduction's trip count either: a
114-wide padded input at stride 2 producing 56 outputs is consistent with BOTH a 3-tall and a 4-tall
window, and the two compute different maxima. The extent is a fact of the captured model that the
map alone does not carry, which is exactly why the upstream ``linalg.pooling_*`` named ops take a
shape-only window operand ``K``: it pins the extent and makes the concatenated map invertible in one
move. A capture that emits the windowed reduction as a bare two-operand ``linalg.generic`` has
dropped that fact, and the repair belongs in the producer -- there is no sound way to reconstruct it
here, and guessing would silently change the numbers.

This module does not repair anything. It NAMES the defect: which op, which iteration dims are
unbound, and which captured layer it came from, so the diagnosis costs one line instead of a
bisection. It is consulted only after a lowering has already failed, so it costs nothing on the
path that works.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class UnboundWindow:
    """One op whose iteration space is not recoverable from its indexing maps."""

    op: str                                  # the op's real name, e.g. "linalg.generic"
    dims: tuple[int, ...]                    # iteration dims that never appear as a bare dim
    maps: tuple[str, ...]                    # the op's indexing maps, as printed
    prov: dict[str, str] = field(default_factory=dict)   # prov.* tags of the offending op

    def describe(self) -> str:
        names = ", ".join(f"d{d}" for d in self.dims)
        where = " ".join(f"{k}={v}" for k, v in sorted(self.prov.items()) if v)
        head = f"{self.op}: iteration dim(s) {names} never appear as a bare dim in any indexing map"
        body = "\n".join(f"      {m}" for m in self.maps)
        tail = f"\n    captured from: {where}" if where else ""
        return f"  - {head}\n{body}{tail}"


#: prov keys worth quoting back: they name the captured layer the op came from.
_PROV_KEYS = ("prov.fqn", "prov.aten", "prov.op", "prov.region_id")


def _indexing_maps(op) -> tuple[Any, ...] | None:
    """The op's ``indexing_maps``, whether carried as a property or an attribute (both spellings
    occur: MLIR moved linalg's to inherent properties, xDSL round-trips either)."""
    for table in (getattr(op, "properties", None), getattr(op, "attributes", None)):
        if not table:
            continue
        maps = table.get("indexing_maps")
        if maps is None:
            continue
        entries = getattr(maps, "data", None)
        if entries is None:
            return None
        out = []
        for entry in entries:
            affine = getattr(entry, "data", None)
            if affine is None or not hasattr(affine, "results"):
                return None
            out.append(affine)
        return tuple(out)
    return None


def unbound_iteration_dims(maps) -> tuple[int, ...]:
    """Iteration dims of ``maps`` that no map states as a bare dim, in order.

    This is precisely the predicate ``mlir::inversePermutation`` fails on, restated: it seeds the
    inverse from bare ``AffineDimExpr`` results only, and yields null unless every input dim was
    seeded. An empty result means the op's maps ARE invertible.
    """
    from xdsl.ir.affine import AffineDimExpr

    if not maps:
        return ()
    n_dims = max(int(m.num_dims) for m in maps)
    bound: set[int] = set()
    for m in maps:
        for result in m.results:
            if isinstance(result, AffineDimExpr):
                bound.add(int(result.position))
    return tuple(d for d in range(n_dims) if d not in bound)


def unbound_windows(src: Any) -> list[UnboundWindow]:
    """Every op in ``src`` (module, MLIR text, or path) whose iteration space is unbound."""
    from ..common.mlir_query import op_name, parse, walk

    found: list[UnboundWindow] = []
    for op in walk(parse(src)):
        maps = _indexing_maps(op)
        if not maps:
            continue
        dims = unbound_iteration_dims(maps)
        if not dims:
            continue
        attrs = getattr(op, "attributes", {}) or {}
        prov = {k.split(".", 1)[1]: getattr(attrs[k], "data", str(attrs[k]))
                for k in _PROV_KEYS if k in attrs}
        found.append(UnboundWindow(op=op_name(op), dims=dims,
                                   maps=tuple(str(m) for m in maps), prov=prov))
    return found


def explain(src: Any) -> str | None:
    """An actionable diagnosis of ``src``, or ``None`` when its iteration spaces are all bound.

    Never raises: it runs on a module a lowering has ALREADY rejected, so it must not replace one
    failure with another. An unparseable module simply yields no diagnosis.
    """
    try:
        found = unbound_windows(src)
    except Exception:                       # noqa: BLE001 -- a diagnosis must never mask the real error
        return None
    if not found:
        return None
    lines = [
        f"{len(found)} op(s) have an iteration space their indexing maps do not bind, which MLIR "
        "rejects when it parses the module (\"invalid indexing maps are non-invertible\"):",
        *(w.describe() for w in found),
        "",
        "A dim used only inside a compound result (a strided window read such as `d2 * 2 + d4`) "
        "pins neither the inverse map nor the loop's trip count, so the op cannot be verified or "
        "lowered, and the extent cannot be recovered from the shapes: several window sizes fit the "
        "same operand and output shapes and compute different results. Give the op a shape-only "
        "window operand mapped `(d4, d5)` -- what `linalg.pooling_*` calls K -- at the point that "
        "still knows the window, i.e. in the capture. Re-export the bundle rather than patching "
        "this module: an extent guessed here would change the numbers silently.",
    ]
    return "\n".join(lines)
