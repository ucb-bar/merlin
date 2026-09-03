"""Turn two captured models into the workload list both arms search over.

WHY THIS EXISTS. v0's workloads were three ``(M, N, K)`` tuples typed into three different scripts.
That is fine for a go/no-go and useless for a generalization claim: the shapes were chosen because
they made the lever observable, which is the definition of fitting the test to the answer. Here the
shapes come from ResNet-50 and TinyLlama, and nothing about them was chosen by us.

WHAT THE CENSUS DOES NOT ANSWER, and this does. ``merlin.kernels.census`` reports legality at the
scope ``op_name+element_types`` -- it asks "is this op and dtype routable to the mesh at all", and for
ResNet-50 the answer is 54/54 legal. That is true and it is not the question a compiler has to
answer. The question is whether the LOWERING can emit it, and the frozen lowering stages both operand
grids whole and keeps every output tile live across the K reduction, so it needs

    Kt*(Mt + Nt) <= spad_rows/dim   AND   Mt*Nt <= acc_rows/dim

and enforces NEITHER on the matmul path: past either bound the weight base goes negative and the
accumulator index runs off the end, which is a silent wrong answer rather than a refusal. The verdict
here is not re-derived from those inequalities -- it is delegated to the compiler's own
``lowering.recipe.fit``, so this script cannot drift from what the compiler will actually do.

CLAIM-MODEL RULE (``merlin/contract/claim_models.yaml``). ``resnet50_v1_5`` and ``tiny_llama`` are
declared claim models: they may be COMPILED and GRADED, and their census may never enter requirement
derivation, corpus synthesis, or capsule selection. So this file is a GRADING input only. Nothing
downstream may fit a rule on it, and the recipe space must be frozen (see ``_track.mint_fork``) before
it is read. The freeze digest is recorded beside the output for exactly that reason.

SIZING. Real shapes are also unaffordable: at the measured GSIM throughput one full-resolution
ResNet-50 pass is ~68M cycles, ~100 h of simulation. So a shape too big for the per-candidate budget
is CLAMPED the way ``targetgen.applications.size_class`` clamps -- **K is preserved** (it is where
accumulation, residency and spill behaviour live, and it is what makes a shape a member of its class)
and only the parallel extents move. The true shape is kept on the row beside the clamped one, never
overwritten, so a reader can always see what was actually asked for.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _track as T                                                        # noqa: E402

T.assert_right_merlin()

from merlin.common.paths import artifacts_dir                             # noqa: E402

#: MEASURED GSIM throughput on this lowering, from v0's own runs (frozen default recipe):
#: 32^3 -> 42.0, 64^3 -> 75.6, 16x512x256 -> 46.3 MACs/cycle. The LOW end is used to size, because
#: sizing must not under-predict the bill. This is a calibration band, NOT a cost model: the repo's
#: fitted gemmini cost model was falsified against these same measurements (-42% to +111%).
MACS_PER_CYCLE_LOW = 42.0
MACS_PER_CYCLE_HIGH = 76.0

#: MEASURED wall law for one candidate on GSIM, fitted over 172 evaluated candidates in v0:
#: eval_seconds ~= 70 + 0.006 * cycles. The intercept is the ELF build; the slope is the simulation.
EVAL_FIXED_S = 70.0
EVAL_S_PER_CYCLE = 0.006


def _geometry(pkg: Path) -> dict:
    """DIM / SPAD_ROWS / ACC_ROWS read from the compiler that will emit, not restated here."""
    sys.path.insert(0, str(pkg / "mlir_oot"))
    from lowering.isa import ACC_ROWS, DIM, SPAD_ROWS                     # noqa: PLC0415
    return {"dim": DIM, "spad_rows": SPAD_ROWS, "acc_rows": ACC_ROWS}


def _fit(pkg: Path, m: int, n: int, k: int, geom: dict):
    """Does the whole shape fit both stores AT ONCE -- the pre-blocking bound."""
    sys.path.insert(0, str(pkg / "mlir_oot"))
    from lowering.recipe import Recipe, fit                               # noqa: PLC0415
    return fit(Recipe(), m=m, n=n, k=k, **geom)


def _plan(pkg: Path, m: int, n: int, k: int, geom: dict):
    """Can the compiler EMIT the shape at all -- i.e. does a legal cut exist?

    Reported beside `_fit` rather than instead of it, because they answer different questions and the
    pair is the capability result: how many of a real model's shapes the compiler could emit before,
    and how many after.
    """
    sys.path.insert(0, str(pkg / "mlir_oot"))
    from lowering.recipe import Recipe, blocks                            # noqa: PLC0415
    return blocks(Recipe(), m=m, n=n, k=k, **geom)


def _shrink(current: int, want: float, tile: int) -> int:
    """Shrink ``current`` towards ``want``, rounded down to whole tiles -- and NEVER grow it.

    The tile floor in ``applications.py`` reads ``max(tile, ...)``, which is right when clamping a
    large extent and WRONG here: TinyLlama's real shapes have M=8 and ResNet-50's classifier has M=1,
    and a floor of one tile silently rounds those UP. That is not a smaller version of the workload,
    it is a different one -- and M=1 decode is precisely the regime this experiment should cover. So
    the result is bounded above by what came in, and a sub-tile extent is left exactly as it is.
    """
    if current <= tile:
        return current
    stepped = (int(want) // tile) * tile
    return max(tile, min(current, stepped))


def clamp_to_budget(m: int, n: int, k: int, *, geom: dict, budget_cycles: float) -> tuple[int, int]:
    """Shrink the PARALLEL extents until the shape fits the per-candidate cycle budget. K is fixed.

    Shrinking proportionally rather than along one axis keeps the shape's aspect -- an N-heavy layer
    stays N-heavy, which is the property the residency lever reads. A shape already inside the budget
    is returned unchanged.
    """
    tile = geom["dim"]
    macs = m * n * k
    worst_cycles = macs / MACS_PER_CYCLE_LOW
    if worst_cycles <= budget_cycles:
        return m, n
    scale = math.sqrt(budget_cycles / worst_cycles)
    return _shrink(m, m * scale, tile), _shrink(n, n * scale, tile)


def fit_to_capacity(m: int, n: int, k: int, *, pkg: Path, geom: dict) -> tuple[int, int, str]:
    """Shrink the parallel extents until the frozen lowering can actually emit the shape.

    This is the CURRENT compiler's limit, and shrinking to meet it is how a shape becomes measurable
    at all today. It is deliberately reported separately from the budget clamp: one says "the machine
    cannot hold this", the other says "we cannot afford to simulate this", and conflating them would
    hide the capability gap this experiment exists to measure.
    """
    tile = geom["dim"]
    cm, cn = m, n
    for _ in range(64):
        verdict = _fit(pkg, cm, cn, k, geom)
        if verdict.ok:
            return cm, cn, ""
        if cm <= tile and cn <= tile:
            return cm, cn, verdict.reason
        # Halve the larger extent: the binding store is whichever grid dominates.
        if cn >= cm:
            cn = _shrink(cn, cn / 2, tile)
        else:
            cm = _shrink(cm, cm / 2, tile)
    return cm, cn, "did not converge to a fitting shape"


def rows_from_census(census_json: Path, pkg: Path, *, budget_s: float) -> list[dict]:
    payload = json.loads(census_json.read_text(encoding="utf-8"))
    geom = _geometry(pkg)
    budget_cycles = max(0.0, (budget_s - EVAL_FIXED_S)) / EVAL_S_PER_CYCLE
    out: list[dict] = []

    for model in payload["models"]:
        name = model["model"]
        # Group by the SHAPE, not the layer: 44 TinyLlama q_projs are one workload evaluated once and
        # counted 44 times. Counting them as 44 workloads would let one shape's result dominate a
        # geomean purely by how often the model happens to call it.
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for r in model.get("rows", []):
            par, red = r.get("parallel") or [], r.get("reduction") or []
            if len(par) != 2 or len(red) != 1:
                out.append({"model_id": name, "layer_fqn": r.get("key", ""),
                            "shape_rank": "", "M": "", "N": "", "K": "",
                            "dtype_in": ",".join(r.get("dtypes") or []),
                            "invocation_count": 1, "macs": "", "mac_share": "",
                            "expressible": "no", "inexpressible_reason":
                                f"rank {len(par)}x{len(red)} contraction: this lowering emits 2-D "
                                f"matmul only ({r.get('op_class')})",
                            "census_verdict": (r.get("legality") or {}).get("verdict", ""),
                            "eval_M": "", "eval_N": "", "eval_K": "", "sizing": "not_expressible",
                            "true_M": par[0] if len(par) > 0 else "",
                            "true_N": par[1] if len(par) > 1 else "",
                            "true_K": red[0] if red else "",
                            "pred_cycles_low": "", "pred_cycles_high": "", "pred_eval_s": ""})
                continue
            groups[(par[0], par[1], red[0], tuple(r.get("dtypes") or []))].append(r)

        total_macs = sum(m_ * n_ * k_ * len(rs) for (m_, n_, k_, _), rs in groups.items()) or 1
        ranked = sorted(groups.items(), key=lambda kv: -(kv[0][0] * kv[0][1] * kv[0][2] * len(kv[1])))
        for rank, ((m_, n_, k_, dts), rs) in enumerate(ranked):
            macs = m_ * n_ * k_
            verdict = _fit(pkg, m_, n_, k_, geom)
            plan = _plan(pkg, m_, n_, k_, geom)
            cap_m, cap_n, cap_fail = fit_to_capacity(m_, n_, k_, pkg=pkg, geom=geom)
            bud_m, bud_n = clamp_to_budget(cap_m, cap_n, k_, geom=geom, budget_cycles=budget_cycles)
            sizing = ("true_shape" if (bud_m, bud_n) == (m_, n_)
                      else "clamped_capacity+budget" if (cap_m, cap_n) != (m_, n_)
                      else "clamped_budget")
            ev_macs = bud_m * bud_n * k_
            lo = ev_macs / MACS_PER_CYCLE_HIGH
            hi = ev_macs / MACS_PER_CYCLE_LOW
            out.append({
                "model_id": name,
                "layer_fqn": rs[0].get("key", ""),
                "shape_rank": rank,
                "M": bud_m, "N": bud_n, "K": k_,
                "dtype_in": ",".join(dts),
                "invocation_count": len(rs),
                "macs": macs,
                "mac_share": round(macs * len(rs) / total_macs, 6),
                "fits_without_cutting": "yes" if verdict.ok else "no",
                "why_cutting_needed": "" if verdict.ok else verdict.reason,
                "expressible": "yes" if plan.ok else "no",
                "inexpressible_reason": "" if plan.ok else plan.reason,
                "derived_block_m": plan.bm, "derived_block_n": plan.bn,
                "derived_block_k": plan.bk, "n_blocks": plan.n_blocks,
                "census_verdict": (rs[0].get("legality") or {}).get("verdict", ""),
                "eval_M": bud_m, "eval_N": bud_n, "eval_K": k_,
                "sizing": sizing,
                "true_M": m_, "true_N": n_, "true_K": k_,
                "capacity_fit_M": cap_m, "capacity_fit_N": cap_n,
                "capacity_fit_failed": cap_fail,
                "pred_cycles_low": int(lo), "pred_cycles_high": int(hi),
                "pred_eval_s": round(EVAL_FIXED_S + EVAL_S_PER_CYCLE * hi, 1),
            })
    return out


def _capability_report(rows: list[dict], census: Path, pkg: Path, digest: str,
                       budget_s: float) -> str:
    """The before/after the whole extension turns on, written so a reader can check it.

    Deliberately separates three things this repo has confused before: what the CENSUS says is legal
    (op and dtype routable), what the LOWERING could emit (both capacity bounds, one block), and what
    it can emit NOW (a legal cut exists). Only the third is a statement about today's compiler.
    """
    L = ["# Model kernel census and the capability it exposes", "",
         f"- census: `{census}`",
         f"- compiler package: `{pkg.name}` (source digest `{digest[:16]}`)",
         f"- per-candidate GSIM wall budget used for sizing: {budget_s:.0f} s", "",
         "## Expressibility, per model", "",
         "| model | distinct shapes | fit whole (pre-blocking) | emittable (with blocking) | "
         "MACs covered |", "|---|---|---|---|---|"]
    for model in sorted({r["model_id"] for r in rows}):
        mr = [r for r in rows if r["model_id"] == model and r["M"] != ""]
        whole = [r for r in mr if r["fits_without_cutting"] == "yes"]
        expr = [r for r in mr if r["expressible"] == "yes"]
        L.append(f"| {model} | {len(mr)} | {len(whole)}/{len(mr)} | {len(expr)}/{len(mr)} | "
                 f"{sum(r['mac_share'] for r in expr):.1%} |")
    L += ["",
          "The census's own legality column reports every one of these as legal. Its scope is",
          "`op_name+element_types` -- *is this op and dtype routable to the mesh at all* -- which is a",
          "different question from *can the lowering emit it*, and only the second is about the",
          "compiler. Both bounds have to hold for a single block:",
          "",
          "> operand store `Kt*(Mt+Nt) <= spad_rows/dim`  **and**  accumulator `Mt*Nt <= acc_rows/dim`",
          "",
          "and the accumulator one binds first and binds everywhere, which is why the pre-blocking",
          "column is zero rather than small.",
          "",
          "## What the frozen compiler did with a shape past the bound",
          "",
          "It had no capacity predicate at all, so the answer was not a refusal. MEASURED, two ways:",
          "",
          "* `PR06_spills_k8208` (16x16x8208, a 32-row A/B overlap) **passes** at 28118 cycles. At",
          "  Nt=1 each tile is written and consumed in the same iteration, so the aliased partners are",
          "  never live together. Accidentally safe.",
          "* 16x512x528 computes a NEGATIVE weight base (-512) and **does not halt**: stopped after",
          "  13 minutes having written 557 MB to the console with no `METRIC`/`DONE`. The same shape",
          "  under blocking is bit-exact at 93366 cycles, cut into 2 K-blocks.",
          "",
          "So the defect is not 'wrong answers'. It is that nothing distinguished the two cases.",
          "",
          "## Sizing",
          "",
          "K is preserved on every row -- it is where accumulation, residency and spill behaviour live",
          "-- and only the parallel extents are clamped, to the per-candidate wall budget above. The",
          "true shape is kept beside the clamped one on every row, never overwritten. Sub-tile extents",
          "(ResNet-50's M=1 classifier, TinyLlama's M=8 at sequence 8) are left exactly as they are:",
          "a one-tile floor would round them UP, which is a different workload, not a smaller one.",
          "",
          "## Claim-model rule",
          "",
          "`merlin/contract/claim_models.yaml` declares both models as claim models: they may be",
          "compiled and graded, and their census may never enter requirement derivation, corpus",
          "synthesis or capsule selection. This file is a GRADING input only. The recipe space and",
          "the derived blocking rule are frozen in the package named above, whose digest predates",
          "this artifact.",
          ""]
    return "\n".join(L)

FIELDS = ["model_id", "layer_fqn", "shape_rank", "M", "N", "K", "dtype_in", "invocation_count",
          "macs", "mac_share", "fits_without_cutting", "why_cutting_needed",
          "expressible", "inexpressible_reason", "census_verdict",
          "derived_block_m", "derived_block_n", "derived_block_k", "n_blocks",
          "eval_M", "eval_N", "eval_K", "sizing", "true_M", "true_N", "true_K",
          "capacity_fit_M", "capacity_fit_N", "capacity_fit_failed",
          "pred_cycles_low", "pred_cycles_high", "pred_eval_s"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--census", default="", help="workload_census.json (default: newest under "
                                                 "out/artifacts/target-evolution/gemmini/v1)")
    ap.add_argument("--package", default="", help="minted compiler package (default: mint from WORK)")
    ap.add_argument("--budget-s", type=float, default=300.0,
                    help="per-candidate GSIM wall budget that sizing must respect")
    ap.add_argument("--out", default="", help="where to write kernel_census.csv (default: stdout summary only)")
    ap.add_argument("--product", action="store_true",
                    help="write a versioned product under out/artifacts/recipe-select/gemmini/v2/ "
                         "with the census, the capability result and the freeze record")
    ap.add_argument("--version", type=int, default=2)
    a = ap.parse_args(argv)

    if a.census:
        census = Path(a.census)
    else:
        root = artifacts_dir() / "target-evolution/gemmini/v1"
        cands = sorted(root.glob("*/workload_census.json"))
        if not cands:
            raise SystemExit(f"no workload_census.json under {root}; run merlin.kernels.census first")
        census = cands[-1]

    pkg = Path(a.package) if a.package else T.mint_fork()
    digest = T.assert_package_frozen(pkg)

    rows = rows_from_census(census, pkg, budget_s=a.budget_s)

    dest = Path(a.out) if a.out else None
    if a.product:
        from merlin.common.artifacts import new_product                   # noqa: PLC0415
        prod = new_product("recipe-select", version=a.version, target="gemmini",
                           notes="model kernel census + expressibility, ResNet-50 and TinyLlama")
        dest = Path(prod.add_artifact("kernel_census.csv"))
        Path(prod.add_artifact("capability.md")).write_text(
            _capability_report(rows, census, pkg, digest, a.budget_s), encoding="utf-8")
        prod.write_manifest()
        print(f"product: {prod.path}")
    if dest:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        (dest.parent / "census_provenance.json").write_text(json.dumps({
            "census": str(census),
            "package": pkg.name,
            "package_source_digest": digest,
            "budget_s": a.budget_s,
            "calibration": {"macs_per_cycle_low": MACS_PER_CYCLE_LOW,
                            "macs_per_cycle_high": MACS_PER_CYCLE_HIGH,
                            "eval_fixed_s": EVAL_FIXED_S, "eval_s_per_cycle": EVAL_S_PER_CYCLE,
                            "source": "measured on 172 v0 candidates; NOT the falsified cost model"},
            "claim_model_rule": ("resnet50_v1_5 and tiny_llama are claim models: this file is a "
                                 "GRADING input only and may not fit any rule"),
            "engine_note": T.ENGINE_NOTE,
        }, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {dest} ({len(rows)} rows)")

    for model in sorted({r["model_id"] for r in rows}):
        mr = [r for r in rows if r["model_id"] == model]
        shapes = [r for r in mr if r["M"] != ""]
        whole = [r for r in shapes if r["fits_without_cutting"] == "yes"]
        expr = [r for r in shapes if r["expressible"] == "yes"]
        w_share = sum(r["mac_share"] for r in whole)
        share = sum(r["mac_share"] for r in expr)
        nt_gt1 = [r for r in shapes if r["eval_N"] != "" and r["eval_N"] > 16]
        print(f"\n{model}")
        print(f"  distinct shapes            {len(shapes)}   (+{len(mr) - len(shapes)} non-2-D contractions)")
        print(f"  fit whole (pre-blocking)   {len(whole)}/{len(shapes)}  = {w_share:.1%} of model MACs")
        print(f"  EMITTABLE (with blocking)  {len(expr)}/{len(shapes)}  = {share:.1%} of model MACs")
        print(f"  measurable after sizing    {len(shapes)}  (K preserved on every one)")
        print(f"  with Nt>1 after sizing     {len(nt_gt1)}  <- shapes that can see the residency lever")
        print(f"  predicted GSIM wall        {sum(r['pred_eval_s'] for r in shapes) / 60:.1f} min "
              f"for one pass over all shapes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
