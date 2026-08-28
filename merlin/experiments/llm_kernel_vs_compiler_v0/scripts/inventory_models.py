#!/usr/bin/env python3
"""Inventory a captured model: every op, its shapes, its cost, its family.

This is the study's denominator-side measurement. It reads ``model.mlir`` only -- no weights are
loaded -- so a 4 GB capture inventories in seconds.

Two properties matter more than completeness here:

**An op that cannot be priced is UNKNOWN, never zero.** A capture can contain an opaque
``func.call`` -- an op the importer could not decompose. It is fully typed, so its shapes are known
even when its cost formula is not. Counting it as zero work would silently shrink the workload and,
worse, shrink it non-uniformly: SmolVLA's single opaque op is its SigLIP patch embedding at ~1.2
GFLOP, which is not a rounding error. Such ops are reported in their own bucket with their signature,
and the manifest carries ``priced_fraction`` so any downstream weighting can see what it is missing.

Work is counted in the unit ``kernels.work.work_of`` uses -- iteration space x body arithmetic ops,
i.e. FLOPs rather than MACs -- and opaque ops are converted into that unit so the two are additive.

**Work is a lower bound where the IR says so.** ``kernels.work.work_of`` returns a ``complete`` flag
that is False when a dimension only appears compounded in an indexing map. That flag is propagated
per row and summarised, rather than being flattened into a number that looks exact.

Usage:
    inventory_models.py --model tiny_llama [--variant fp32] --out <dir>
    inventory_models.py --all --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

#: Arithmetic ops per multiply-accumulate. ``kernels.work.work_of`` prices a linalg op as
#: iteration-space x body arith ops, and a contraction body is a multiply and an add -- so its unit
#: is FLOPs, not MACs. An opaque op priced in MACs would therefore be weighted HALF as heavily as an
#: identical op the importer happened to decompose, which is a bias that tracks how well the importer
#: did rather than anything about the workload.
_ARITH_PER_MAC = 2


# Opaque placeholders are typed but unpriced. A formula per known family lets the big ones be
# priced anyway; anything absent here stays UNKNOWN rather than becoming zero.
def _conv_macs(arg_shapes: list[list[int]], out_shape: list[int]) -> int | None:
    """Convolution cost from its signature alone, in the SAME unit as ``work_of`` (FLOPs).

    |out| x (C_in/groups x kernel spatial) MACs, scaled by :data:`_ARITH_PER_MAC`. Reads the weight
    operand (out_channels, in_channels/groups, *kernel) for the per-output-element dot length;
    groups are already folded into the weight's second dimension, so no group argument is needed.
    """
    if len(arg_shapes) < 2 or not out_shape:
        return None
    w = arg_shapes[1]
    if len(w) < 3:
        return None
    dot = 1
    for d in w[1:]:
        dot *= d
    n_out = 1
    for d in out_shape:
        n_out *= d
    return n_out * dot * _ARITH_PER_MAC


_OPAQUE_PRICERS = {"convolution": _conv_macs}


def _opaque_family(callee: str) -> str:
    """Best-effort family for an opaque callee name, by whole-token match.

    Substring matching would let `aten_convolution_default` and a hypothetical
    `aten_deconvolution` collide, so the name is split on '_' and matched token-wise.
    """
    tokens = {t for t in callee.lower().split("_") if t}
    for fam in _OPAQUE_PRICERS:
        if fam in tokens:
            return fam
    return "unknown"


def inventory(mlir_path: Path) -> dict:
    """Per-op inventory of one captured model, read from its MLIR alone."""
    from merlin.common import mlir_query as mq
    from merlin.kernels import work as wk

    text = mlir_path.read_text()
    module = mq.parse(text)

    rows: list[dict] = []
    incomplete = 0
    for op in module.walk():
        # is_weighable excludes the ops INSIDE a linalg body (linalg.yield, linalg.index): they are
        # part of one op's payload, not ops in their own right, and counting them would both inflate
        # the op count and let a body op claim a family of its own.
        if not wk.is_weighable(op):
            continue
        name = mq.op_name(op)
        # provenance() returns keys WITH the "prov." prefix. Reading them without it returned empty
        # for every row and silently fell back to the MLIR op name as the family, which reads as
        # "this model has no semantic families" rather than as a bug.
        prov = {k[len("prov."):]: v for k, v in mq.provenance(op).items()}
        w, complete = wk.work_of(op)
        if not complete:
            incomplete += 1
        shapes, dtypes = [], []
        for res in getattr(op, "results", []):
            try:
                shp, dt = mq.type_shape_dtype(res.type)
                shapes.append(shp)
                dtypes.append(dt)
            except Exception:      # a result whose type is not a shaped tensor tells us nothing
                continue
        rows.append({
            "mlir_op": name,
            "family": prov.get("family") or "",
            "op": prov.get("op") or "",
            "role": prov.get("role") or "",
            "region_id": prov.get("region_id") or "",
            "work": int(w),
            "work_complete": bool(complete),
            "bytes": int(wk.footprint_bytes(op)),
            "result_shapes": shapes,
            "result_dtypes": dtypes,
        })

    opaque = _opaque_rows(text)

    priced = sum(r["work"] for r in rows) + sum(o["work"] or 0 for o in opaque)
    unpriced_ops = [o for o in opaque if o["work"] is None]
    by_family: Counter = Counter()
    for r in rows:
        by_family[r["op"] or r["family"] or r["mlir_op"]] += r["work"]
    for o in opaque:
        by_family[f"opaque:{o['family']}"] += (o["work"] or 0)

    return {
        "source_mlir": str(mlir_path),
        "n_linalg_ops": len(rows),
        "n_opaque_ops": len(opaque),
        "n_unpriced_ops": len(unpriced_ops),
        "n_incomplete_work": incomplete,
        "total_work": priced,
        "priced_fraction": (
            None if unpriced_ops else 1.0
        ),
        "work_by_family": dict(by_family.most_common()),
        "opaque": opaque,
        "rows": rows,
    }


def _opaque_rows(text: str) -> list[dict]:
    """Opaque ``func.call`` placeholders, with shapes recovered from their private declaration.

    Parsed structurally (``str.split``/``partition``), not by pattern matching: a too-narrow pattern
    silently drops a validly-spelled declaration, and a dropped op here reads as a smaller workload.
    """
    decls: dict[str, tuple[list[list[int]], list[int]]] = {}
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("func.func private @"):
            continue
        name, _, rest = s[len("func.func private @"):].partition("(")
        args_txt, _, ret_txt = rest.rpartition(") ->")
        decls[name.strip()] = (_shapes_in(args_txt), (_shapes_in(ret_txt) or [[]])[0])

    counts: Counter = Counter()
    for line in text.splitlines():
        if "func.call @" in line:
            callee = line.split("func.call @", 1)[1].split("(", 1)[0].strip()
            counts[callee] += 1

    out: list[dict] = []
    for callee, n in sorted(counts.items()):
        arg_shapes, ret_shape = decls.get(callee, ([], []))
        fam = _opaque_family(callee)
        pricer = _OPAQUE_PRICERS.get(fam)
        per = pricer(arg_shapes, ret_shape) if pricer else None
        out.append({
            "callee": callee,
            "count": n,
            "family": fam,
            "arg_shapes": arg_shapes,
            "result_shape": ret_shape,
            "work": (per * n) if per is not None else None,
            "priced": per is not None,
            "note": (
                "priced from the declared signature" if per is not None else
                "UNPRICED: no cost formula for this family -- work is UNKNOWN, not zero"
            ),
        })
    return out


def _shapes_in(txt: str) -> list[list[int]]:
    """Every ``tensor<AxBx...xTYPE>`` shape in a type list, in order."""
    shapes: list[list[int]] = []
    for chunk in txt.split("tensor<")[1:]:
        body = chunk.split(">", 1)[0]
        dims: list[int] = []
        for part in body.split("x")[:-1]:      # last part is the element type
            part = part.strip()
            if part.lstrip("-").isdigit():
                dims.append(int(part))
            else:
                dims = []                      # dynamic or unparsable -> no shape claim
                break
        if dims:
            shapes.append(dims)
    return shapes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", default=None)
    ap.add_argument("--variant", default="fp32")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--mlir", type=Path, default=None,
                    help="inventory this MLIR directly, bypassing capture resolution")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    from merlin.baselines import bundle as B

    jobs: list[tuple[str, Path]] = []
    if a.mlir:
        jobs.append((a.mlir.stem, a.mlir))
    else:
        names = sorted(B.available_models()) if a.all else (a.model or [])
        if not names:
            raise SystemExit("nothing to do: pass --model, --all or --mlir")
        for m in names:
            for variant in (a.variant, "int8", "fp32"):   # first variant that is actually present
                b = B.resolve(m, variant)
                if b.mlir.exists():
                    jobs.append((f"{m}_{variant}", b.mlir))
                    break
            else:
                print(f"  !! {m}: no capture with an MLIR on disk -- skipped")

    a.out.mkdir(parents=True, exist_ok=True)
    for label, mlir in jobs:
        inv = inventory(mlir)
        dest = a.out / f"{label}.workload.json"
        dest.write_text(json.dumps(inv, indent=1))
        warn = ""
        if inv["n_unpriced_ops"]:
            warn = f"  ⚠️ {inv['n_unpriced_ops']} UNPRICED opaque op(s)"
        print(f"  {label:28s} ops={inv['n_linalg_ops']:>5d} opaque={inv['n_opaque_ops']:<3d} "
              f"work={inv['total_work']:>16,}{warn}")
        print(f"      -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
