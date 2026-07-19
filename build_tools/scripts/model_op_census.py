#!/usr/bin/env python
"""Work-weighted op census over the recaptured model bundles.

The XNNPACK catalog (`xnnpack_kernel_catalog.py`) says WHAT expert kernels exist; it cannot say
which of them MATTER. This script answers the other half: which op families the models we actually
run on the board actually execute, weighted by work done rather than by op count.

Weighting. For every linalg op we recover the ITERATION SPACE and multiply it by the number of
scalar arithmetic ops in the body:

    work = (product of iteration-space extents) * (arith/math ops in the region)

The iteration-space extents come from the indexing maps: for each iteration dim `d`, its extent is
the operand-shape entry at any map result that is the bare `AffineDimExpr(d)`. Compound results
(e.g. `d2*16 + d5` in a conv) are skipped -- the bare occurrences of the same dims in the *filter*
map supply them, so a conv's full 7-deep loop nest is still recovered exactly. This is the real
loop-nest size, not an output-element proxy: a matmul's K and a conv's reduction window count.

`bytes_moved` is the complementary axis (sum of operand + result footprints). Ops with a low
work/bytes ratio are memory-bound and rank differently from the flop ranking -- both are reported,
because an elementwise family can be irrelevant on flops and dominant on traffic.

Ops are grouped by the `prov.op` provenance attribute the capture pipeline stamps on every region
(matmul / softmax / gelu / conv2d / rmsnorm / ...), which is the semantic op the model author wrote,
falling back to the linalg op name when a region is unstamped.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from xdsl.ir.affine import AffineDimExpr

from merlin.common import mlir_query as mq
from merlin.common.paths import artifacts_dir

# Named linalg ops carry no region, so their body arithmetic is implicit. Value = scalar arith ops
# performed per iteration-space point.
_NAMED_BODY_OPS = {
    "linalg.matmul": 2,          # mul + add
    "linalg.batch_matmul": 2,
    "linalg.matvec": 2,
    "linalg.fill": 0,            # a store, no arithmetic
    "linalg.copy": 0,
    "linalg.transpose": 0,       # pure movement
    "linalg.broadcast": 0,
    "linalg.reduce": 1,
}

# Named contractions whose iteration space is NOT just the result shape: the reduction dim has to
# come from an input. Value = (operand index, dim index within that operand).
_NAMED_EXTRA_DIM = {
    "linalg.matmul": (0, -1),        # K = lhs last dim
    "linalg.batch_matmul": (0, -1),
    "linalg.matvec": (0, -1),
}

_ELEM_BYTES = {"f32": 4, "f64": 8, "f16": 2, "bf16": 2, "i64": 8, "i32": 4,
               "i16": 2, "i8": 1, "i1": 1, "index": 8}


def _shape_of(t) -> tuple[list[int], str]:
    try:
        return mq.type_shape_dtype(t)
    except Exception:  # noqa: BLE001 - non-tensor operand (scalar/index): no shape, no footprint
        return [], ""


def _iteration_space_generic(op) -> tuple[int, bool]:
    """(product of iteration-space extents, complete?) recovered from the indexing maps."""
    maps = op.properties.get("indexing_maps") or op.attributes.get("indexing_maps")
    if maps is None:
        return 0, False
    maps = list(maps.data)
    tensors = [*op.operands, *op.results]
    extents: dict[int, int] = {}
    ndims = 0
    for amap_attr, tensor in zip(maps, tensors):
        amap = amap_attr.data
        ndims = max(ndims, amap.num_dims)
        shape, _ = _shape_of(tensor.type)
        if len(shape) != len(amap.results):
            continue
        for res, extent in zip(amap.results, shape):
            if isinstance(res, AffineDimExpr):
                extents[res.position] = max(extents.get(res.position, 0), extent)
    if ndims == 0:
        return 0, False
    prod = 1
    for v in extents.values():
        prod *= max(v, 1)
    # An iteration dim that only ever appears inside a compound expression cannot be pinned down;
    # flag it rather than silently understating the work.
    return prod, len(extents) == ndims


def _body_arith_ops(op) -> int:
    """Count real arithmetic in the region (yields/constants/index reads are not work)."""
    n = 0
    for region in op.regions:
        for inner in region.walk():
            name = mq.op_name(inner)
            if name.startswith(("arith.", "math.")) and not name.endswith(".constant"):
                n += 1
    return n


def _footprint_bytes(op) -> int:
    total = 0
    for tensor in [*op.operands, *op.results]:
        shape, dtype = _shape_of(tensor.type)
        if not shape:
            continue
        elems = 1
        for d in shape:
            elems *= max(d, 1)
        total += elems * _ELEM_BYTES.get(dtype, 4)
    return total


def _named_iters(op, name: str) -> int:
    """Iteration space of a named linalg op (no indexing maps to read)."""
    if name == "linalg.reduce" and op.operands:
        in_shape, _ = _shape_of(op.operands[0].type)
        iters = 1
        for d in in_shape:
            iters *= max(d, 1)
        return iters
    res_shape: list[int] = []
    if op.results:
        res_shape, _ = _shape_of(op.results[0].type)
    iters = 1
    for d in res_shape:
        iters *= max(d, 1)
    extra = _NAMED_EXTRA_DIM.get(name)
    if extra is not None and op.operands:
        lhs_shape, _ = _shape_of(op.operands[extra[0]].type)
        if lhs_shape:
            iters *= max(lhs_shape[extra[1]], 1)
    return iters


def census_bundle(mlir: Path) -> dict:
    module = mq.parse(mlir)
    per_op: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "work": 0, "bytes": 0, "partial": 0,
                 "linalg_ops": defaultdict(int), "shapes": defaultdict(int)})
    for op in module.walk():
        name = mq.op_name(op)
        if not name.startswith("linalg.") or name in ("linalg.yield", "linalg.index"):
            continue
        complete = True
        if name == "linalg.generic":
            iters, complete = _iteration_space_generic(op)
            body = _body_arith_ops(op)
        else:
            iters = _named_iters(op, name)
            body = _NAMED_BODY_OPS.get(name, 1)
        prov = mq.provenance(op)
        family = prov.get("prov.op") or prov.get("prov.family") or name.split(".", 1)[1]
        rec = per_op[family]
        rec["count"] += 1
        rec["work"] += iters * body
        rec["bytes"] += _footprint_bytes(op)
        rec["partial"] += 0 if complete else 1
        rec["linalg_ops"][name] += 1
        if op.results:
            shape, dtype = _shape_of(op.results[0].type)
            if shape:
                rec["shapes"]["x".join(str(d) for d in shape) + dtype] += 1
    out = {}
    for fam, rec in per_op.items():
        top = sorted(rec["shapes"].items(), key=lambda kv: -kv[1])[:3]
        out[fam] = {"count": rec["count"], "work": rec["work"], "bytes": rec["bytes"],
                    "partial_iterspace": rec["partial"], "linalg_ops": dict(rec["linalg_ops"]),
                    "top_shapes": [{"shape": s, "n": n} for s, n in top]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", default="", help="comma-separated bundle names (default: *_full)")
    ap.add_argument("--suffix", default="_full", help="bundle-name suffix to auto-select")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    root = Path(artifacts_dir()) / "recaptures"
    if a.bundles:
        names = a.bundles.split(",")
    else:
        names = sorted(p.name for p in root.iterdir()
                       if p.is_dir() and (p / "model.mlir").is_file()
                       and p.name.endswith(a.suffix))

    per_model: dict[str, dict] = {}
    for name in names:
        mlir = root / name / "model.mlir"
        if not mlir.is_file():
            print(f"skip {name}: no model.mlir")
            continue
        try:
            per_model[name] = census_bundle(mlir)
        except Exception as e:  # noqa: BLE001 - one unparseable bundle must not sink the census
            print(f"skip {name}: parse failed: {type(e).__name__}: {str(e)[:160]}")
            continue
        tw = sum(r["work"] for r in per_model[name].values())
        print(f"{name}: {len(per_model[name])} families, work={tw:,}")

    # Cross-model rollup: each model contributes its SHARE of work, so a single huge model cannot
    # define the ranking for all of them. Ranking by mean share answers "how much of a typical
    # model's work is this family", which is the question the kernel race needs answered. Models
    # lacking a family contribute 0 to its mean, so a family present in one model of ten cannot
    # outrank one present in all ten.
    shares: dict[str, list[float]] = defaultdict(list)
    byte_shares: dict[str, list[float]] = defaultdict(list)
    totals: dict[str, int] = defaultdict(int)
    counts: dict[str, int] = defaultdict(int)
    models_with: dict[str, int] = defaultdict(int)
    for name, fams in per_model.items():
        tw = sum(r["work"] for r in fams.values()) or 1
        tb = sum(r["bytes"] for r in fams.values()) or 1
        for fam, rec in fams.items():
            shares[fam].append(rec["work"] / tw)
            byte_shares[fam].append(rec["bytes"] / tb)
            totals[fam] += rec["work"]
            counts[fam] += rec["count"]
            models_with[fam] += 1
    n_models = len(per_model) or 1
    ranking = [{"family": fam,
                "mean_work_share": sum(sl) / n_models,
                "max_work_share": max(sl),
                "mean_bytes_share": sum(byte_shares[fam]) / n_models,
                "models_with": models_with[fam],
                "n_models": n_models,
                "total_work": totals[fam],
                "total_op_count": counts[fam]}
               for fam, sl in shares.items()]
    ranking.sort(key=lambda r: -r["mean_work_share"])

    out = Path(a.out) if a.out else Path(artifacts_dir()) / "ceiling" / "model_op_census.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"models": per_model, "ranking": ranking,
                               "n_models": len(per_model), "bundles": list(per_model)}, indent=2))
    print(f"\nwrote {out}\n")
    print(f"{'family':<26}{'work share':>12}{'bytes share':>13}{'max':>9}{'models':>8}{'ops':>9}")
    for r in ranking[:30]:
        print(f"{r['family']:<26}{r['mean_work_share']*100:>11.2f}%"
              f"{r['mean_bytes_share']*100:>12.2f}%{r['max_work_share']*100:>8.1f}%"
              f"{r['models_with']:>8}{r['total_op_count']:>9}")


if __name__ == "__main__":
    main()
