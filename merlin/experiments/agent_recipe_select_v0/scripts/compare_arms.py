"""Head-to-head table: the recipe arm (merlin compiler + agent) vs the AutoComp arm.

WHY THIS IS NOT A SPEEDUP CHART. The two arms do not share a baseline and cannot be ranked by their
own improvement factors:

  * the RECIPE arm starts from the frozen compiler's DEFAULT recipe for that shape;
  * the AUTOCOMP arm starts from a hand-written Gemmini C SEED kernel.

On real ResNet-50 layers that seed starts well AHEAD of the compiler default, so "2.08x over my own
baseline" and "1.3x over mine" are answers to different questions. The only comparable column is
ABSOLUTE CYCLES on the same oracle, which is why `faster_x` below is computed from cycles and the
per-arm improvement factors are reported beside it rather than instead of it.

Both arms are measured on GSIM. FireSim cycles are a different column entirely and never enter here
(the same capsule reads 510 there and 317 under Verilator).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _track as T                                                        # noqa: E402

T.assert_right_merlin()
RUNS = T.REPO / "out" / "runs" / "gemmini" / "recipe-select"


def _shape_of(name: str) -> str:
    """`resnet50_v1_5__model_fc__1x112x2048[.mlir]` -> the workload key both arms share."""
    return name[:-5] if name.endswith(".mlir") else name


def recipe_results() -> dict[str, dict]:
    """Parse the recipe arm's per-shape summary line out of each campaign log."""
    out: dict[str, dict] = {}
    for camp in sorted(RUNS.glob("campaign_recipe_*")):
        mf = camp / "campaign.jsonl"
        if not mf.exists():
            continue
        for line in mf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("rc") != 0:
                continue
            log = Path(rec["log"])
            if not log.exists():
                continue
            # "baseline 14493 -> best 6955 (2.0838x) in 16 candidates (0 invalid, 1 duplicate)"
            for ln in log.read_text(encoding="utf-8", errors="replace").splitlines():
                if "baseline" in ln and "-> best" in ln:
                    tok = ln.split()
                    base = int(tok[tok.index("baseline") + 1])
                    best = int(tok[tok.index("best") + 1])
                    # "(0 invalid," -> the count carries the opening paren; strip it.
                    n_inv = int(tok[tok.index("invalid,") - 1].lstrip("("))
                    out[_shape_of(rec["workload"])] = {
                        "baseline": base, "best": best, "invalid": n_inv,
                        "wall_s": rec.get("wall_s"), "run": camp.name}
    return out


def autocomp_results() -> dict[str, dict]:
    """Seed latency and best CORRECT candidate latency, per shape, from the eval-results dirs."""
    out: dict[str, dict] = {}
    for d in sorted(RUNS.glob("*_autocomp_census_*")):
        if not d.is_dir():
            continue
        tag = d.name.split("census_", 1)[1].rsplit("_seed", 1)[0]
        seed: int | None = None
        best: int | None = None
        n_cand = n_correct = n_compiled_wrong = n_uncompiled = 0
        for f in sorted(d.glob("eval-results-iter-*/*_result.txt")):
            try:
                j = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            lat = j.get("latency")
            correct, compiled = bool(j.get("correct")), bool(j.get("compiled"))
            if f.parent.name.endswith("-0"):          # iteration 0 IS the seed, not a candidate
                if correct and isinstance(lat, int):
                    seed = lat if seed is None else min(seed, lat)
                continue
            n_cand += 1
            if correct and isinstance(lat, int):
                n_correct += 1
                best = lat if best is None else min(best, lat)
            elif compiled:
                n_compiled_wrong += 1
            else:
                n_uncompiled += 1
        if seed is None and best is None:
            continue
        # AutoComp keeps the seed when nothing beats it, so the arm's result is the better of the two.
        arm_best = seed if best is None else min(best, seed if seed is not None else best)
        out[tag] = {"baseline": seed, "best": arm_best, "candidates": n_cand,
                    "correct": n_correct, "compiled_wrong": n_compiled_wrong,
                    "uncompiled": n_uncompiled, "run": d.name,
                    "complete": (d / "run_metrics.json").exists() and n_cand > 0}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="emit rows as JSON instead of a table")
    ap.add_argument("--all", action="store_true", help="include shapes only one arm has run")
    a = ap.parse_args(argv)

    rec, aut = recipe_results(), autocomp_results()
    keys = sorted(set(rec) | set(aut)) if a.all else sorted(set(rec) & set(aut))

    rows = []
    for k in keys:
        r, u = rec.get(k), aut.get(k)
        row = {"shape": k}
        row["recipe_baseline"] = r["baseline"] if r else None
        row["recipe_best"] = r["best"] if r else None
        row["recipe_x_over_own_baseline"] = round(r["baseline"] / r["best"], 3) if r else None
        row["autocomp_baseline"] = u["baseline"] if u else None
        row["autocomp_best"] = u["best"] if u else None
        row["autocomp_x_over_own_baseline"] = (
            round(u["baseline"] / u["best"], 3) if u and u["baseline"] and u["best"] else None)
        # THE comparable number: absolute cycles, same oracle. >1 means the recipe arm is faster.
        if r and u and u["best"]:
            row["recipe_vs_autocomp_x"] = round(u["best"] / r["best"], 3)
            row["winner"] = "recipe" if r["best"] < u["best"] else "autocomp"
        else:
            row["recipe_vs_autocomp_x"] = None
            row["winner"] = None
        row["autocomp_candidates"] = u["candidates"] if u else None
        row["autocomp_correct"] = u["correct"] if u else None
        row["recipe_invalid"] = r["invalid"] if r else None
        rows.append(row)

    if a.json:
        print(json.dumps(rows, indent=1))
        return 0

    if not rows:
        print("no shapes completed on both arms yet")
        return 0

    hdr = (f"{'shape':46} | {'recipe base':>11} {'recipe best':>11} {'x':>6} | "
           f"{'ac base':>8} {'ac best':>8} {'x':>6} | {'recipe vs ac':>13} {'winner':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        def f(v, w, dp=0):
            return (f"{v:>{w}.{dp}f}" if isinstance(v, float) else
                    f"{v:>{w}}" if v is not None else f"{'-':>{w}}")
        vs = r["recipe_vs_autocomp_x"]
        vstxt = (f"{vs:.2f}x faster" if vs and vs >= 1 else
                 f"{1/vs:.2f}x slower" if vs else "-")
        print(f"{r['shape'][:46]:46} | {f(r['recipe_baseline'],11)} {f(r['recipe_best'],11)} "
              f"{f(r['recipe_x_over_own_baseline'],6,2)} | {f(r['autocomp_baseline'],8)} "
              f"{f(r['autocomp_best'],8)} {f(r['autocomp_x_over_own_baseline'],6,2)} | "
              f"{vstxt:>13} {str(r['winner'] or '-'):>9}")

    both = [r for r in rows if r["recipe_vs_autocomp_x"]]
    if both:
        wins = sum(1 for r in both if r["winner"] == "recipe")
        gm = 1.0
        for r in both:
            gm *= r["recipe_vs_autocomp_x"]
        gm **= 1.0 / len(both)
        print(f"\nrecipe wins {wins}/{len(both)} shapes; geomean recipe-vs-autocomp {gm:.3f}x "
              f"({'recipe faster' if gm >= 1 else 'autocomp faster'})")
        print("cycles are GSIM for both arms. 'x over own baseline' is NOT comparable across arms: "
              "the\nrecipe baseline is the compiler default, AutoComp's is a hand-written C seed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
