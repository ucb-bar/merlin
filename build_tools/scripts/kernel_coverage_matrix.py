#!/usr/bin/env python
"""Coverage matrix: op family -> model relevance -> XNNPACK family -> our board result.

Joins three sources into one honest matrix:
  1. model_op_census.json   -- which op families our models actually execute, work/byte weighted
     (build_tools/scripts/model_op_census.py)
  2. kernel_catalog.json     -- the 637-kernel / 115-family XNNPACK surface + mapped/partial/
     expert-only classification (build_tools/scripts/xnnpack_kernel_catalog.py)
  3. the cross_framework_ops board JSONL(s) -- XNNPACK-vs-OUR-codegen races on the real K1
     (build_tools/scripts/k1_cross_framework_ops.py)

The point is honesty about WHERE WE STAND across the expert surface, not a leaderboard: a family we
could not race shows as an explicit not_run with the reason, and a family that dominates model work
but has no XNNPACK primitive (attention, gather) is called out as such. Breadth + honest gaps beats
a few deep wins.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from merlin.common.paths import artifacts_dir

# Census prov.op family -> (XNNPACK catalog family, board op key, note). board op key is the `op`
# field the race JSONL uses (None if we cannot race it). A curated join: the census names the op the
# model author wrote; the catalog names the expert kernel; the race names the cell.
FAMILY_MAP: dict[str, tuple[str | None, str | None, str]] = {
    # contractions -- the bulk of model WORK; all lower to a GEMM on our side.
    "matmul":       ("f32-gemm", "f32_gemm", "raced as f32_gemm (7x4v)"),
    "addmm":        ("f32-gemm", "f32_gemm", "linear+bias == GEMM; raced as f32_gemm"),
    "batch_matmul": ("f32-gemm", "attention_bmm", "no XNN batch primitive; ours-vs-ours bmm cell"),
    "linear":       ("f32-gemm", "f32_gemm", "== GEMM"),
    "conv2d":       ("f32-igemm/f32-dwconv", "conv2d/dwconv", "regular=im2col->GEMM; depthwise raced sep."),
    "sdpa":         (None, "attention_bmm", "no XNN/BLAS attention primitive; ours-vs-ours"),
    # elementwise + reductions -- the bulk of model TRAFFIC.
    "mul":          ("f32-vbinary", "vbinary_mul", "raced"),
    "add":          ("f32-vbinary", "vbinary_add", "raced"),
    "sub":          ("f32-vbinary", "vbinary_add", "same kernel family (vsub)"),
    "div":          ("f32-vbinary", None, "f32-vdiv exists; not raced (add/mul cover the family)"),
    "elementwise":  ("f32-vbinary", "vbinary_mul", "generic elementwise -> vbinary family"),
    "minmax":       ("f32-vclamp", "clamp", "clamp/relu; raced as clamp"),
    "reduce":       ("f32-rsum/f32-rminmax", "reduce_sum", "raced (sum + max)"),
    "reduce_mean":  ("f32-rsum", "reduce_sum", "mean == rsum * 1/N"),
    "transpose":    ("x32-transposec", "transpose", "largest byte-traffic family; raced"),
    # activations.
    "gelu":         ("f32-vgelu", "gelu", "raced"),
    "sigmoid":      ("f32-vsigmoid", "sigmoid", "raced"),
    "silu":         ("f32-vsigmoid", None, "x*sigmoid; f32-vsigmoid closest; not raced separately"),
    # transcendental / norm building blocks with an XNN kernel we did not race.
    "softmax":      ("f32-raddstoreexpminusmax", None, "expert-only fused exp-max-sum; composite on our side"),
    "exp":          ("f32-vexp", None, "catalog partial (softmax path)"),
    "pow":          ("f32-vsqrt", None, "pow(.,0.5)->vsqrt; general pow no XNN kernel"),
    "rsqrt":        ("f32-vrsqrt", None, "rmsnorm; catalog mapped; not raced (low work share)"),
    "abs":          ("f32-vabs", None, "f32-vunary; not raced"),
    "layer_norm":   (None, None, "composite (rsum+rsqrt+vbinary); no single XNN kernel"),
    "neg":          ("f32-vneg", None, "f32-vunary; not raced"),
    # ops with NO XNNPACK vector primitive -- honest structural gaps.
    "round":        ("f32-vrnd", None, "expert-only; no Merlin equivalent"),
    "sin":          ("f32-vsin", None, "expert-only; RoPE"),
    "cos":          ("f32-vcos", None, "expert-only; RoPE"),
    "embedding":    (None, None, "gather; no XNN vector primitive"),
    "index_gather": (None, None, "gather; no XNN vector primitive"),
    "select":       (None, None, "predicated select; no XNN primitive"),
    "dtype_cast":   ("f32-qs8-vcvt", None, "convert family; expert-only"),
    "arange":       (None, None, "iota; no XNN primitive"),
    "fill":         (None, None, "memset; no compute"),
    "expand":       (None, None, "broadcast/copy; no XNN compute primitive"),
    "compare":      (None, None, "predicate; no XNN primitive"),
    "bitwise":      (None, None, "no XNN f32 primitive"),
    "aten_max_dim": ("f32-rminmax", None, "argmax-style reduce; not raced"),
    "generic":      (None, None, "unclassified generic"),
}


def _load_results(paths: list[Path]) -> dict[str, list[dict]]:
    """op -> list of rows (source split lives in each row's `source` field)."""
    by_op: dict[str, list[dict]] = defaultdict(list)
    for p in paths:
        if not p.is_file():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_op[row.get("op", "?")].append(row)
    return by_op


def _best(rows: list[dict], pred) -> dict | None:
    cand = [r for r in rows if pred(r) and r.get("status") == "pass" and r.get("ticks")]
    if not cand:
        return None
    return min(cand, key=lambda r: r["ticks"])


def _race_summary(op: str, by_op: dict[str, list[dict]]) -> dict:
    """Summarize the XNNPACK-vs-ours race for one board op across all its shapes."""
    rows = by_op.get(op, [])
    if not rows:
        return {"raced": False}
    out = {"raced": True, "n_rows": len(rows), "shapes": []}
    # group by shape signature
    def shape_key(r):
        for keys in (("M", "N", "K"), ("R", "C"), ("M", "N"), ("size_n",)):
            if all(k in r for k in keys):
                return "x".join(str(r[k]) for k in keys)
        return "?"
    shapes = sorted({shape_key(r) for r in rows})
    n_pass = n_notrun = 0
    for sh in shapes:
        srows = [r for r in rows if shape_key(r) == sh]
        xnn = _best(srows, lambda r: r.get("source") == "xnnpack")
        ours = _best(srows, lambda r: str(r.get("source", "")).startswith("ours"))
        blockers = sorted({r.get("blocker", "")[:120] for r in srows
                           if r.get("status") == "not_run"} - {""})
        cell = {"shape": sh,
                "xnn_ticks": xnn["ticks"] if xnn else None,
                "xnn_instret": xnn.get("instret") if xnn else None,
                "ours_ticks": ours["ticks"] if ours else None,
                "ours_instret": ours.get("instret") if ours else None,
                "ours_source": ours.get("source") if ours else None}
        if xnn and ours:
            cell["ratio_ours_over_xnn"] = round(ours["ticks"] / xnn["ticks"], 2)
            n_pass += 1
        else:
            cell["blockers"] = blockers
            n_notrun += 1
        out["shapes"].append(cell)
    out["n_shapes_raced"] = n_pass
    out["n_shapes_blocked"] = n_notrun
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default="")
    ap.add_argument("--catalog", default="")
    ap.add_argument("--results", default="", help="comma-separated race JSONL paths")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    ceiling = Path(artifacts_dir()) / "ceiling"
    census = json.loads(Path(a.census or ceiling / "model_op_census.json").read_text())
    catalog = json.loads(Path(a.catalog or ceiling / "kernel_catalog.json").read_text())
    cat_status = {}
    for row in catalog["rows"]:
        cat_status.setdefault(row["family"], row["status"])

    if a.results:
        rpaths = [Path(p) for p in a.results.split(",")]
    else:
        mdir = Path(artifacts_dir()) / "measurements" / "k1_spacemit" / "gemm"
        rpaths = [mdir / "cross_framework_ops_broadened_brd.jsonl",
                  mdir / "cross_framework_ops_k1.jsonl"]
    by_op = _load_results(rpaths)

    ranking = census["ranking"]
    matrix = []
    for r in ranking:
        fam = r["family"]
        xnn_fam, board_op, note = FAMILY_MAP.get(fam, (None, None, "unmapped"))
        status = None
        if xnn_fam:
            status = cat_status.get(xnn_fam.split("/")[0], "?")
        race = _race_summary(board_op, by_op) if board_op else {"raced": False}
        matrix.append({"family": fam,
                       "mean_work_share": r["mean_work_share"],
                       "mean_bytes_share": r["mean_bytes_share"],
                       "models_with": r["models_with"],
                       "n_models": r.get("n_models"),
                       "xnn_family": xnn_fam, "catalog_status": status,
                       "board_op": board_op, "note": note, "race": race})

    out = Path(a.out) if a.out else ceiling / "kernel_coverage_matrix.json"
    out.write_text(json.dumps({"matrix": matrix, "n_models": census.get("n_models")}, indent=2))

    # ---- markdown ----
    md = ["# Kernel coverage matrix — model relevance vs the XNNPACK expert surface", "",
          f"Model census over {census.get('n_models')} `*_full` recapture bundles "
          "(work = iteration-space x body-arith; bytes = operand+result footprint). "
          "Race = XNNPACK RVV ukernel vs OUR codegen on the K1, same shape, correctness-gated, "
          "min-of-reps rdtime ticks. `ratio` = ours_ticks / xnn_ticks (>1 = XNNPACK faster).", "",
          "## Families ranked by model WORK share", "",
          "| family | work% | bytes% | models | XNN family | catalog | raced | ours/xnn (by shape) |",
          "|---|--:|--:|--:|---|---|:--:|---|"]
    for m in matrix:
        race = m["race"]
        if race.get("raced") and race.get("shapes"):
            cells = []
            for s in race["shapes"]:
                if s.get("ratio_ours_over_xnn") is not None:
                    cells.append(f"{s['shape']}:{s['ratio_ours_over_xnn']}x")
                else:
                    cells.append(f"{s['shape']}:not_run")
            raced_mark = "yes"
            ratio_str = ", ".join(cells)
        elif m["board_op"]:
            raced_mark = "no"
            ratio_str = "(op known, no result rows)"
        else:
            raced_mark = "n/a"
            ratio_str = m["note"]
        md.append(f"| {m['family']} | {m['mean_work_share']*100:.2f} | "
                  f"{m['mean_bytes_share']*100:.2f} | {m['models_with']} | "
                  f"{m['xnn_family'] or '—'} | {m['catalog_status'] or '—'} | "
                  f"{raced_mark} | {ratio_str} |")
    md += ["", "## Honest gaps (families we could not race, and why)", ""]
    for m in matrix:
        if not m["race"].get("raced"):
            md.append(f"- **{m['family']}** ({m['mean_work_share']*100:.2f}% work, "
                      f"{m['mean_bytes_share']*100:.2f}% bytes): {m['note']}")
    md_path = out.with_suffix(".md")
    md_path.write_text("\n".join(md) + "\n")
    print(f"wrote {out}\nwrote {md_path}")
    print("\n".join(md[:8 + len(matrix)]))


if __name__ == "__main__":
    main()
