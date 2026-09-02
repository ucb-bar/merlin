"""Legal VALUE domains for a target's config fields, derived from its own RTL facts.

The derived encoder already makes an illegal opcode and a use-before-config structurally impossible. It
says nothing about whether a config *value* is right, and that is the gap a whole class of failures
lives in: the program decodes cleanly, every instruction is of the right class, the functional tiers
agree, and only the RTL disagrees -- a submission passed L0/L1/L2 and collapsed to 1/23 on the RTL with
nothing in between to point at. A value out of its hardware's range is detectable BEFORE any of that,
and this is what makes it detectable.

What is derivable is bounded by what the target's facts actually contain, and that varies enormously:
one target discovers its arrays, memories and datapaths; another discovers only an array; a third only
a memory whose depth is unknown; a fourth nothing at all. So this module reports the domains it CAN
derive and, separately, what it could not -- an absent domain is never a permissive one. A checker that
silently passes a field it cannot bound is worse than no checker, because it is cited as evidence.

Only extents the hardware genuinely fixes become domains:

* a memory's ``depth`` bounds a ROW index, its ``bytes`` a byte offset;
* an array's ``rows``/``cols`` bound an INDEX into that array;
* a datapath's dtype bounds a value stored on it.

Deliberately NOT derived: a tile dimension from the array size. A tile larger than the mesh is legal --
it gets tiled -- so bounding one by the other would reject correct programs. The array bound is an
index bound, and it is named that way.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Domain:
    """A closed interval a field's value must lie in, and the RTL fact that fixes it."""
    name: str
    lo: int
    hi: int
    unit: str
    evidence: str

    def contains(self, value: int) -> bool:
        return self.lo <= int(value) <= self.hi


def _int_dtype_range(dtype: str) -> tuple[int, int] | None:
    """``(lo, hi)`` for a signed/unsigned integer dtype token, or None when it is not one this can
    bound. Parsed structurally from the token's own width -- never a table of known dtypes."""
    t = (dtype or "").strip().lower()
    if not t or t[0] not in ("i", "u"):
        return None
    width = t[1:]
    if not width.isdigit():
        return None
    bits = int(width)
    if bits <= 0 or bits > 64:
        return None
    if t[0] == "u":
        return 0, (1 << bits) - 1
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def derive_domains(target: str) -> dict[str, Domain]:
    """Every value domain this target's facts fix, keyed by ``<resource>.<unit>``."""
    from .rtl import facts as _facts

    f = (_facts.load_facts(target) or {}).get("facts") or {}
    out: dict[str, Domain] = {}

    for mem in f.get("memories") or []:
        name = mem.get("name")
        if not name:
            continue
        depth, nbytes = mem.get("depth"), mem.get("bytes")
        if isinstance(depth, int) and depth > 0:
            out[f"{name}.row"] = Domain(f"{name}.row", 0, depth - 1, "row",
                                        f"memories.{name}.depth={depth}")
        if isinstance(nbytes, int) and nbytes > 0:
            out[f"{name}.byte"] = Domain(f"{name}.byte", 0, nbytes - 1, "byte",
                                         f"memories.{name}.bytes={nbytes}")

    for arr in f.get("arrays") or []:
        name = arr.get("name")
        if not name:
            continue
        for axis in ("rows", "cols"):
            n = arr.get(axis)
            if isinstance(n, int) and n > 0:
                key = f"{name}.{axis[:-1]}"          # rows -> row, cols -> col
                out[key] = Domain(key, 0, n - 1, "index", f"arrays.{name}.{axis}={n}")

    for dp in f.get("datapaths") or []:
        name, dtype = dp.get("name"), dp.get("dtype")
        rng = _int_dtype_range(dtype or "")
        if name and rng:
            out[f"{name}.value"] = Domain(f"{name}.value", rng[0], rng[1], "value",
                                          f"datapaths.{name}.dtype={dtype}")
    return out


def undecidable(target: str) -> list[str]:
    """Resources present in the facts whose extent could NOT be bounded, so a caller can see the
    checker's coverage rather than read silence as approval."""
    from .rtl import facts as _facts

    f = (_facts.load_facts(target) or {}).get("facts") or {}
    gaps: list[str] = []
    for mem in f.get("memories") or []:
        n = mem.get("name") or "?"
        if not isinstance(mem.get("depth"), int):
            gaps.append(f"memories.{n}.depth unknown — no row bound")
        if not isinstance(mem.get("bytes"), int):
            gaps.append(f"memories.{n}.bytes unknown — no byte bound")
    for dp in f.get("datapaths") or []:
        if not _int_dtype_range(dp.get("dtype") or ""):
            gaps.append(f"datapaths.{dp.get('name') or '?'}.dtype={dp.get('dtype')!r} — not an integer width")
    if not (f.get("memories") or f.get("arrays") or f.get("datapaths")):
        gaps.append("no arrays, memories or datapaths discovered — nothing is bounded for this target")
    return gaps


def check(target: str, values: dict[str, int]) -> dict[str, Any]:
    """Check ``{domain_name: value}`` against the derived domains.

    Three outcomes per field, and the third is the point: ``ok`` in range, ``violation`` out of range
    with the fact that bounds it, and ``unbounded`` for a field whose domain this target's facts do not
    fix. An unbounded field is reported, never silently accepted.
    """
    domains = derive_domains(target)
    ok, bad, unbounded = [], [], []
    for field, value in (values or {}).items():
        d = domains.get(field)
        if d is None:
            unbounded.append({"field": field, "value": value,
                              "reason": f"no derived domain for {field!r} on {target}"})
        elif d.contains(value):
            ok.append({"field": field, "value": value, "domain": [d.lo, d.hi], "evidence": d.evidence})
        else:
            bad.append({"field": field, "value": value, "domain": [d.lo, d.hi], "unit": d.unit,
                        "evidence": d.evidence,
                        "detail": f"{field}={value} outside [{d.lo}, {d.hi}] fixed by {d.evidence}"})
    return {"target": target, "ok": ok, "violations": bad, "unbounded": unbounded,
            "checked": len(values or {}), "coverage_gaps": undecidable(target)}
