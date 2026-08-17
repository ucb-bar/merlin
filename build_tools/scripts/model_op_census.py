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

The work/bytes weighting itself lives in `merlin.kernels.work` -- the per-contraction census
(`merlin.kernels.census`) needs the identical proxy, and two copies of a cost model drift into two
different rankings of the same model.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from merlin.common import mlir_query as mq
from merlin.common.paths import artifacts_dir
from merlin.kernels import work as wk


def _shape_of(t) -> tuple[list[int], str]:
    try:
        return mq.type_shape_dtype(t)
    except Exception:  # noqa: BLE001 - non-tensor operand (scalar/index): no shape, no footprint
        return [], ""


def census_bundle(mlir: Path) -> dict:
    module = mq.parse(mlir)
    per_op: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "work": 0, "bytes": 0, "partial": 0,
                 "linalg_ops": defaultdict(int), "shapes": defaultdict(int)})
    for op in module.walk():
        name = mq.op_name(op)
        if not wk.is_weighable(op):
            continue
        work, complete = wk.work_of(op)
        prov = mq.provenance(op)
        family = prov.get("prov.op") or prov.get("prov.family") or name.split(".", 1)[1]
        rec = per_op[family]
        rec["count"] += 1
        rec["work"] += work
        rec["bytes"] += wk.footprint_bytes(op)
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
