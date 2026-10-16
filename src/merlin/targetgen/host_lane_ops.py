"""The host operations real captures contain, per ``(family, dtype)``, with their frontend identity.

The host-lane requirement says WHICH ``(family, dtype)`` work a target must leave on the host. A probe
capsule for that pair has to be built from an operation that work actually consists of: a probe
written with a stock op of the same family (``gelu`` for an elementwise pair, ``reduce_sum`` for a
reduction) exercises a program no captured model contains, and it falls outside reviewed host
declarations that name only the operations the captures carry. So the candidates are read from the
captures themselves -- each region's model2MLIR op (``prov.op``, the same vocabulary as Merlin's op
names) and its frontend operator (``prov.aten``) -- and a probe is chosen only from those.

Nothing here names a target or an operation; it counts what the captured regions say they are.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def pair_key(family: str, dtype: str) -> str:
    return f"{family}/{dtype}"


def observed_host_ops(captures: Mapping[str, str | Path], pairs: Iterable[tuple[str, str]]) -> dict[str, list]:
    """``"family/dtype" -> [{op, frontend_op, n_regions}]`` for the requested pairs, most frequent first.

    A region whose provenance names no op is not counted: an anonymous region cannot be reproduced by
    a probe, and guessing its op would invent the very stock program this exists to avoid. An
    unreadable capture contributes nothing (it is reported by the pair census that reads it too).
    """
    from merlin.targetgen import model_coverage as mc
    from merlin.targetgen.conformance import capsule_dtype

    wanted = {(str(family), str(dtype)) for family, dtype in pairs}
    counts: dict[tuple[str, str], Counter] = {pair: Counter() for pair in wanted}
    layouts: dict[tuple[str, str, str, str | None], Counter] = {}
    for _label, path in sorted((captures or {}).items()):
        try:
            module = mc.load_module(path)
            sources = list(mc.region_sources(module))
        except Exception:  # noqa: BLE001 -- an unreadable capture is evidence neither way
            continue
        for family, dtype, op, frontend, layout in _transpose_layouts(module):
            try:
                dtype = capsule_dtype(str(dtype))
            except Exception:  # noqa: BLE001 -- an unmappable token stays as spelled
                dtype = str(dtype)
            if (family, dtype) in counts:
                layouts.setdefault((family, dtype, op, frontend), Counter())[layout] += 1
        for family, dtype, op, frontend in sources:
            if not family or not dtype or not op:
                continue
            try:
                dtype = capsule_dtype(str(dtype))
            except Exception:  # noqa: BLE001 -- an unmappable token stays as spelled
                dtype = str(dtype)
            key = (str(family), str(dtype))
            if key in counts:
                counts[key][(str(op), frontend)] += 1

    def row(pair, op, frontend, n):
        found = layouts.get((*pair, op, frontend))
        out = {"op": op, "frontend_op": frontend, "n_regions": n}
        if found:
            # The exact (input shape, axis order) of each observed region, so a probe reproduces a
            # layout the captures carry: most frequent first, the higher rank first among equals.
            out["layouts"] = [
                {"shape": list(shape), "permutation": list(order), "n_regions": count}
                for (shape, order), count in sorted(found.items(), key=lambda item: (-item[1], -len(item[0][0])))
            ]
        return out

    return {
        pair_key(*pair): [
            row(pair, op, frontend, n)
            for (op, frontend), n in sorted(counter.items(), key=lambda item: (-item[1], item[0][0]))
        ]
        for pair, counter in sorted(counts.items())
    }


def _transpose_layouts(module):
    """``(family, dtype, prov.op, prov.aten, (input shape, permutation))`` of every permutation region."""
    from merlin.common import mlir_query as mq
    from merlin.targetgen import model_coverage as mc
    from merlin.xdsl_dialects.lowering.group_command import _attr_ints

    for op in module.walk():
        if op.name != "linalg.transpose":
            continue
        order = _attr_ints(op, "permutation")
        name = mc._attr_str(op, "prov.op")  # noqa: SLF001 -- the coverage module's provenance reader
        if not order or not name or not op.operands:
            continue
        shape, _dtype = mq.type_shape_dtype(op.operands[0].type)
        if len(shape) != len(order) or any(not isinstance(d, int) or d < 1 for d in shape):
            continue
        yield (
            mc.region_family(op, mc._short_op(op.name)),  # noqa: SLF001
            mc._elem_dtype(op),  # noqa: SLF001
            name,
            mc._attr_str(op, "prov.aten"),  # noqa: SLF001
            (tuple(int(d) for d in shape), tuple(int(a) for a in order)),
        )


def choose(observed: Iterable[Mapping[str, Any]] | None, writable: Iterable[str]) -> dict | None:
    """The most frequent observed host op a writer can express, or ``None`` when none can.

    ``None`` is reported by the caller as an uncovered pair; substituting a writable op that the
    captures do not contain is exactly the failure this module replaces.
    """
    allowed = set(writable)
    for row in observed or ():
        if row.get("op") in allowed:
            return dict(row)
    return None
