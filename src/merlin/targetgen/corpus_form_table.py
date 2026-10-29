"""Declared capsule forms -> the BUILDERS entry that exercises each one.

A spec row of a family may declare ``forms: [slice, pad_reflect, ...]``. Each declared form is one
named, shape-general capsule: this table says which BUILDERS op writes it and which entry fields
(beyond the cell's extents) make it that form. The spec's form rows therefore decide which capsules
exist; nothing here is per overload or per ATen op.

API (other families add rows, nothing else changes):

* ``FormEntry(op, make, primitives)``: ``op`` is a BUILDERS key; ``make(m, n)`` returns the entry fields for
  a rank-2 source of ``m`` x ``n`` (the cell's resolved extents); ``primitives`` are the hardware forms
  (the contract's ``semantic_capabilities[].forms`` vocabulary) the form is composed of.
* ``FORM_TABLE[(family, form)]``; ``entry_fields(family, form, extents)``; ``unsupported_primitives``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class FormEntry:
    op: str
    make: Callable[[int, int], dict[str, Any]]
    primitives: tuple[str, ...] = ()


def _slice(m: int, n: int) -> dict[str, Any]:
    # Rows 1.. at stride 2 (an arbitrary row stride), columns 1.. contiguous (no column stride in the DMA).
    ro = 1 if m > 1 else 0
    co = 1 if n > 1 else 0
    return {
        "form": "slice", "shape": [m, n], "offsets": [ro, co],
        "sizes": [-(-(m - ro) // 2), n - co], "strides": [2, 1],
    }  # fmt: skip


def _flip(m: int, n: int) -> dict[str, Any]:
    return {
        "form": "slice", "shape": [m, n], "offsets": [m - 1, n - 1], "sizes": [m, n], "strides": [-1, -1],
    }  # fmt: skip


def _pad(mode: str) -> Callable[[int, int], dict[str, Any]]:
    def make(m: int, n: int) -> dict[str, Any]:
        lo_r, lo_c = (1 if m > 1 else 0), (1 if n > 1 else 0)
        out: dict[str, Any] = {
            "form": "pad", "shape": [m, n], "low": [lo_r, lo_c], "high": [lo_r, lo_c], "mode": mode,
        }  # fmt: skip
        if mode == "constant":
            out["value_code"] = 0
        return out

    return make


def _broadcast(m: int, n: int) -> dict[str, Any]:
    return {"form": "broadcast", "shape": [1, n], "out_shape": [m, n]}


def _repeat(m: int, n: int) -> dict[str, Any]:
    # OUTER repeat only: the innermost count stays 1 (the spec row admits broadcasting none/outer).
    return {"form": "repeat", "shape": [m, n], "repeats": [2, 1]}


def _permute(m: int, n: int) -> dict[str, Any]:
    return {"form": "permute", "shape": [m, n], "perm": [1, 0]}


def _concat(m: int, n: int) -> dict[str, Any]:
    return {"sources": ["X0", "X1"], "shapes": [[m, n], [m, n]], "axis": 0}


def _insert_slice(m: int, n: int) -> dict[str, Any]:
    ro = 1 if m > 1 else 0
    rows = max(1, (m - ro + 1) // 2)
    return {
        "dest": "D", "src": "S", "dest_shape": [m, n], "src_shape": [rows, n],
        "offsets": [ro, 0], "strides": [2 if rows > 1 else 1, 1],
    }  # fmt: skip


def _fill(code: int) -> Callable[[int, int], dict[str, Any]]:
    def make(m: int, n: int) -> dict[str, Any]:
        return {"shape": [m, n], "value_code": code}

    return make


def _scalar_affine(weight: int, bias: int) -> Callable[[int, int], dict[str, Any]]:
    """A scalar-immediate affine map ``y = weight * x + bias`` (literal immediates; one tensor operand).

    ``add.Scalar`` is weight 1 / bias +c, ``sub.Scalar`` is weight 1 / bias -c (written ``subi``), ``mul.Scalar``
    is weight w / bias 0. The immediates are small literals: the weight a signed int8 (mesh / load-path
    multiplier), the bias an int32 accumulator-seed constant."""

    def make(m: int, n: int) -> dict[str, Any]:
        return {"weight": weight, "bias": bias}

    return make


def _select(m: int, n: int) -> dict[str, Any]:
    # select.int on dim 0 at an interior index: ONE row, the selected dim kept at size 1 (rank-reducing
    # reshape of the result is a representation change, a separate reshape form).
    idx = m // 2
    return {"form": "slice", "shape": [m, n], "offsets": [idx, 0], "sizes": [1, n], "strides": [1, 1]}


def _split_chunk(m: int, n: int) -> dict[str, Any]:
    # split_with_sizes along dim 0, the SECOND chunk (the first one starts at offset 0; this one does not).
    first = max(1, m // 2)
    return {"form": "slice", "shape": [m, n], "offsets": [first, 0], "sizes": [max(1, m - first), n], "strides": [1, 1]}


FORM_TABLE: dict[tuple[str, str], FormEntry] = {
    # elementwise_map standalone affine forms: the contract states `out = sat_int8(RNE(s_o * act(sum_i w_i * x_i + c)))`
    # with one broadcast constant c; the hardware primitives are the unit-scale accumulating load (c as the
    # accumulator seed) and the weighted integer mesh (w), both documented on the spec row, not on the contract's
    # movement `forms` vocabulary, so none is listed here.
    ("elementwise_map", "scalar_add"): FormEntry("scalar_affine", _scalar_affine(1, 3)),
    ("elementwise_map", "scalar_sub"): FormEntry("scalar_affine", _scalar_affine(1, -3)),
    ("elementwise_map", "scalar_mul"): FormEntry("scalar_affine", _scalar_affine(3, 0)),
    ("movement", "slice"): FormEntry("movement", _slice, ("copy",)),
    ("movement", "flip"): FormEntry("movement", _flip, ("copy", "permutation")),
    ("movement", "pad_constant"): FormEntry("movement", _pad("constant"), ("copy", "zero_fill")),
    ("movement", "pad_reflect"): FormEntry("movement", _pad("reflect"), ("copy", "permutation")),
    ("movement", "pad_replicate"): FormEntry("movement", _pad("replicate"), ("copy", "broadcast")),
    ("movement", "broadcast"): FormEntry("movement", _broadcast, ("broadcast",)),
    ("movement", "repeat"): FormEntry("movement", _repeat, ("copy", "broadcast")),
    ("movement", "permute"): FormEntry("movement", _permute, ("permutation",)),
    ("movement", "concat"): FormEntry("concat", _concat, ("copy",)),
    ("movement", "select"): FormEntry("movement", _select, ("copy",)),
    ("movement", "split_chunk"): FormEntry("movement", _split_chunk, ("copy",)),
    ("movement", "fill"): FormEntry("fill", _fill(5), ("broadcast",)),
    ("movement", "fill_zero"): FormEntry("fill", _fill(0), ("zero_fill",)),
    ("movement", "insert_slice"): FormEntry("insert_slice", _insert_slice, ("copy",)),
}


def entry_fields(family: str, form: str, m: int, n: int) -> dict[str, Any]:
    """The entry fields that make a cell's capsule the declared ``form`` (KeyError when undeclared in the table)."""
    return FORM_TABLE[(family, form)].make(m, n)


def unsupported_primitives(family: str, form: str, declared_primitives: tuple[str, ...]) -> tuple[str, ...]:
    """Primitives ``form`` is composed of that the contract does not declare (empty = fully supported)."""
    needed = FORM_TABLE[(family, form)].primitives
    return tuple(p for p in needed if p not in declared_primitives)


def _tile_expr(token: Any, tile: int) -> int:
    """``tile``, ``tile/d``, ``tile-k``, ``a*tile``, ``a*tile-k`` or an int, resolved structurally."""
    if isinstance(token, int):
        return token
    text = str(token).strip().replace(" ", "")
    head, minus, delta = text.partition("-")
    k = int(delta) if minus else 0
    if head.startswith("tile/"):
        return tile // int(head.partition("/")[2]) - k
    mult, star, rest = head.partition("*")
    if star and rest == "tile":
        return int(mult) * tile - k
    if head == "tile":
        return tile - k
    return int(head) - k


def resolve_extents(entry: dict, probes: list[dict]) -> tuple[int, int]:
    """The (rows, columns) a rank-2 form capsule gets: the cell's own M and N at the target tile edge."""
    tile = next((int(p.get("edge") or 0) for p in probes if str((p or {}).get("boundary") or "") == "tile_edge"), 0)
    return _tile_expr(entry["M"], tile), _tile_expr(entry["N"], tile)


def declared_forms(software_spec: dict | None, contract: dict | None) -> tuple[dict[str, list[str]], list[dict]]:
    """``({family: [form, ...]}, refused)`` from the software spec's REVIEWED rows' ``forms``.

    A form is kept only when the table knows it and every hardware primitive it is composed of is among
    the contract's declared ``forms`` for that family (the RTL-backed vocabulary); anything else is
    returned in ``refused`` with its reason rather than silently dropped."""
    hw: dict[str, set[str]] = {}
    for unit in (contract or {}).get("compute_units") or ():
        for cap in unit.get("semantic_capabilities") or ():
            hw.setdefault(str(cap.get("family")), set()).update(cap.get("forms") or ())
    kept: dict[str, list[str]] = {}
    refused: list[dict] = []
    for row in (software_spec or {}).get("operations") or ():
        if not isinstance(row, dict) or not row.get("forms") or row.get("status", "reviewed") != "reviewed":
            continue
        for family in row.get("families") or ([row["family"]] if row.get("family") else []):
            for form in row["forms"]:
                entry = FORM_TABLE.get((family, form))
                if entry is None:
                    refused.append({"family": family, "form": form, "reason": "no capsule writer for this form"})
                    continue
                missing = unsupported_primitives(family, form, tuple(hw.get(family, ())))
                if missing:
                    refused.append({"family": family, "form": form, "reason": f"contract lacks primitives {list(missing)}"})
                elif form not in kept.setdefault(family, []):
                    kept[family].append(form)
    return kept, refused
