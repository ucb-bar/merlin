#!/usr/bin/env python
"""Census the `math.*` population of a capture's PREPARED module, ranked by ELEMENTS PER INFERENCE.

WHY RANKED BY ELEMENTS AND NOT BY OP COUNT. A linked-ELF instruction census groups the emitted code
into structural classes (`build_tools/scripts/scalar_remainder.py`), and its "transcendental" class
is defined as "a `linalg.generic` with no reduction dim and ANY `math.*` op in its body". That is a
useful class for deciding what the vectorize tagger refuses, but it is a poor guide to WHICH math op
to attack: it says nothing about which op is in the body, and an op that appears in three generics
processing 45M elements matters more than one in thirty generics processing a thousand. This tool
answers the second question -- for each `math.*` op, how many generics carry it, how many ELEMENTS
they process per inference (iteration-space volume times the op's multiplicity in the body), and
whether they are a REDUCTION (an amax / softmax / norm accumulate) or an ELEMENTWISE map.

It also splits each row by DATA DEPENDENCE: work reachable from a `@forward` activation argument
versus work that depends only on weights and is therefore recomputed identically on every inference.

Everything is read off the module's own IR -- iterator types, indexing maps and operand shapes -- so
no model, shape, dtype or target is named, and a capture that spells its quantization differently is
counted the same way.

    PYTHONPATH=merlin/python .venv/bin/python build_tools/scripts/model_math_census.py \\
        --bundle out/artifacts/recaptures/lstmnetvit_int8_w8a8_consistent --int8
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


def iteration_volume(op) -> "int | None":
    """The op's iteration-space volume, DERIVED from its own indexing maps and operand shapes.

    An operand dimension addressed by a bare dim expression pins that loop's extent. A dim no
    operand pins that way (a compound or constant result expression) leaves the volume undetermined,
    and this returns None rather than guessing -- an undercounted denominator is how a census
    quietly understates the thing it was built to rank.
    """
    from xdsl.ir.affine import AffineDimExpr
    maps = op.properties.get("indexing_maps")
    if maps is None:
        return None
    bounds: dict[int, int] = {}
    for affine, value in zip(maps.data, op.operands):
        try:
            shape = list(value.type.get_shape())
        except Exception:                                          # noqa: BLE001
            continue
        results = affine.data.results
        if len(results) != len(shape):
            continue
        for result, extent in zip(results, shape):
            if isinstance(result, AffineDimExpr):
                bounds.setdefault(result.position, extent)
    n_loops = len(list(op.properties.get("iterator_types").data)) if \
        op.properties.get("iterator_types") is not None else 0
    if not n_loops or len(bounds) < n_loops:
        return None
    volume = 1
    for i in range(n_loops):
        volume *= bounds[i]
    return volume


def _forward_block(module):
    for op in module.walk():
        if op.name in ("func.func", "builtin.func") and \
                "forward" in str(op.properties.get("sym_name", "")):
            return op.regions[0].blocks[0]
    return None


def activation_dependence(module, n_activation_args: int):
    """A predicate: does this value's def-chain reach one of `@forward`'s ACTIVATION arguments?

    The weight tensors are lifted to TRAILING `@forward` arguments (`llvmlower.qinner.lift`), so the
    first `n_activation_args` arguments are the real inputs and the rest are weights. Work that
    reaches none of the first group is identical on every inference.
    """
    block = _forward_block(module)
    if block is None:
        return None
    args = list(block.args)
    activation = {id(a) for a in args[:n_activation_args]}
    memo: dict[int, bool] = {}

    def reaches(value) -> bool:
        key = id(value)
        if key in memo:
            return memo[key]
        memo[key] = False                       # cycle guard: an unresolved value is not activation
        if key in activation:
            memo[key] = True
            return True
        owner = getattr(value, "owner", None)
        if owner is None or not hasattr(owner, "operands"):
            return False
        result = any(reaches(o) for o in owner.operands)
        memo[key] = result
        return result

    return reaches


def census(module, n_activation_args: int) -> dict:
    reaches = activation_dependence(module, n_activation_args)
    rows: list[dict] = []
    for op in module.walk():
        if getattr(op, "name", None) != "linalg.generic":
            continue
        region = op.regions[0] if op.regions else None
        body = region.blocks[0] if region and region.blocks else None
        if body is None:
            continue
        math_ops = [inner.name for inner in body.ops if inner.name.startswith("math.")]
        if not math_ops:
            continue
        iters = op.properties.get("iterator_types")
        kinds = [str(e) for e in iters.data] if iters is not None else []
        kind = "REDUCTION" if any("reduction" in k for k in kinds) else "ELEMENTWISE"
        depends = ("UNKNOWN" if reaches is None else
                   ("ACTIVATION" if any(reaches(o) for o in op.operands)
                    else "WEIGHT_INVARIANT"))
        volume = iteration_volume(op)
        counts: dict[str, int] = {}
        for m in math_ops:
            counts[m] = counts.get(m, 0) + 1
        rows.append({"kind": kind, "depends": depends, "volume": volume,
                     "rank": len(kinds), "body": counts,
                     "iterators": "".join("R" if "reduction" in k else "P" for k in kinds)})

    agg: dict[tuple, dict] = {}
    for row in rows:
        for name, mult in row["body"].items():
            key = (name, row["kind"], row["depends"])
            slot = agg.setdefault(key, {"generics": 0, "elements": 0, "volume_unknown": 0})
            slot["generics"] += 1
            if row["volume"] is None:
                slot["volume_unknown"] += 1
            else:
                slot["elements"] += row["volume"] * mult
    total = sum(v["elements"] for v in agg.values()) or 1
    ranked = [{"math_op": k[0], "kind": k[1], "depends": k[2],
               "generics": v["generics"], "elements_per_inference": v["elements"],
               "share_pct": round(100.0 * v["elements"] / total, 3),
               "volume_unknown": v["volume_unknown"]}
              for k, v in sorted(agg.items(), key=lambda kv: -kv[1]["elements"])]
    layouts: dict[str, int] = {}
    for row in rows:
        if row["kind"] == "REDUCTION":
            layouts[row["iterators"]] = layouts.get(row["iterators"], 0) + 1
    return {"total_elements_per_inference": sum(v["elements"] for v in agg.values()),
            "ranked": ranked, "reduction_iterator_layouts": layouts,
            "n_carrying_generics": len(rows)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--int8", action="store_true", help="prepare with the int8 compute datapath")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)

    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.runtime.backends import zephyr_model as zm

    order = a.bundle / "input_order.json"
    n_act = len(json.loads(order.read_text())) if order.is_file() else 0
    work = Path(tempfile.mkdtemp(prefix="math_census_"))
    prepared = zm._prepare_model_mlir(a.bundle / "model.mlir", work, int8_compute=a.int8)
    result = census(parse_mlir_file(prepared), n_act)
    result["bundle"] = str(a.bundle)
    result["n_activation_args"] = n_act
    result["prepared"] = str(prepared)

    print(f"\n{a.bundle.name}  ({result['n_carrying_generics']} generics carry a math.* op; "
          f"{result['total_elements_per_inference']:,} elements/inference)\n")
    print(f"{'math op':<18} {'kind':<12} {'data dependence':<18} {'#gen':>6} "
          f"{'elements/inf':>16} {'share':>8}")
    for r in result["ranked"]:
        print(f"{r['math_op']:<18} {r['kind']:<12} {r['depends']:<18} {r['generics']:>6} "
              f"{r['elements_per_inference']:>16,} {r['share_pct']:>7}%")
    print(f"\nreduction iterator layouts (P=parallel, R=reduction, in loop order): "
          f"{result['reduction_iterator_layouts']}")
    if a.json:
        a.json.write_text(json.dumps(result, indent=2))
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
