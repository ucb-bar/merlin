"""Milestone 0 — price the recipe surface with the calibrated cost model, before any simulator runs.

The question this answers is the experiment's go/no-go: does selecting a recipe over the FROZEN
gemmini backend change the emitted code enough to move cycles, and does the winner change with shape?
It is answered here for free, because the cost model is a linear function of command counts and the
counts follow in closed form from the loop nest in the frozen lowering.

WHAT THIS CAN AND CANNOT SEE, stated up front because it decides which levers need a simulator:

* :class:`~merlin.cost_model.gemmini.GemminiCostModel` is ``const + sum(coeff[e] * n_e)`` over
  ``(config, mvin_A, mvin2_B, compute, mvout, fence)`` and its own metadata calls it
  "L2.5 calibrated (linear, serial; no overlap)". So it prices a recipe ONLY through command COUNTS.
* Therefore ``activation_residency`` and ``config_policy`` are model-VISIBLE: they delete MVINs and
  CONFIG_LDs outright.
* ``drain`` (inline vs deferred MVOUT) is model-BLIND BY CONSTRUCTION. The MVOUT count is an
  invariant (``merlin/tests/targetgen/test_rtl_filecheck.py`` asserts
  ``MVOUT_COUNT == ceil(M/DIM)*ceil(N/DIM)``), so a reordering changes no count and the model must
  predict exactly zero delta. That is not evidence the lever is inert -- it is the model declining to
  price accumulator lifetime and DMA clustering, which are precisely the overlap effects it excludes.
* The B-allocation values (``b_k_panel`` / ``b_n_panel``) are likewise model-blind: each B tile is
  already MVIN'd exactly once in the frozen schedule, so those values change scratchpad ADDRESSES and
  the capacity relation, not traffic. Their value is EXPRESSIVENESS -- they are what lets a shape past
  the capacity cliff compile at all -- and it is reported as a fit/no-fit verdict, never as cycles.

A model-blind cell is emitted with ``model_visible=false`` and a stated reason rather than a 0.0
delta, because "the model predicts no change" and "the model cannot see this change" are different
claims and only the first would be a result.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path


def _repo_root() -> Path:
    """Discover the repo root, never a hardcoded parents[N] (see merlin/experiments/AGENT.md)."""
    here = Path(__file__).resolve()
    for cand in (here, *here.parents):
        if (cand / "merlin" / "python").is_dir():
            return cand
    raise SystemExit("could not locate repo root (no merlin/python above this file)")


REPO = _repo_root()
sys.path.insert(0, str(REPO / "merlin" / "python"))

from merlin.cost_model.gemmini import GemminiCostModel          # noqa: E402
from merlin.common.artifacts import new_product                 # noqa: E402
from merlin.common import provenance as PROV                    # noqa: E402


# --------------------------------------------------------------------------- the frozen machine

#: Tile dim and operand-store depth. DERIVED at run time from the RTL fact bundle -- never written
#: down here. `tile_geometry` fails closed if the array is not discoverable, and SPAD_ROWS comes from
#: the scratchpad memory the same bundle reports, so a different Gemmini elaboration re-prices itself.
def machine_facts(target: str) -> tuple[int, int]:
    from merlin.perf.workload_gen import tile_geometry
    from merlin.targetgen.rtl import facts as rtl_facts

    geom = tile_geometry(target)
    dim = geom.rows
    if geom.rows != geom.cols:
        raise SystemExit(f"non-square array {geom.rows}x{geom.cols}: this pricing assumes DIM square")
    body = rtl_facts.load_facts(target)
    mems = (body.get("facts") or body).get("memories") or []
    spad = next((m for m in mems if m.get("name") == "scratchpad"), None)
    if spad is None or not spad.get("bytes"):
        raise SystemExit("scratchpad capacity not derivable from rtl facts -- refusing to assume one")
    return dim, int(spad["bytes"]) // dim


# --------------------------------------------------------------------------- the recipe surface

WAVE_A = {
    "activation_residency": ["per_tile", "panel"],
    "drain": ["inline", "deferred"],
}
#: The fallback axis (plan: used only if a wave-A lever proves inert). Model-VISIBLE, so priced here.
WAVE_A_PLUS = dict(WAVE_A, config_policy=["per_mvin", "on_change"])

DEFAULT_RECIPE = {"activation_residency": "per_tile", "drain": "inline", "config_policy": "per_mvin"}

#: Why a value cannot be priced by a count-only model. Keyed by (dimension, value).
MODEL_BLIND = {
    ("drain", "deferred"):
        "reordering only: MVOUT count is invariant (test_rtl_filecheck asserts Mt*Nt), so a count-only "
        "linear model predicts exactly 0 delta; accumulator lifetime and DMA clustering are the "
        "overlap effects this model's own metadata excludes",
}


@dataclass(frozen=True)
class Workload:
    wid: str
    M: int
    N: int
    K: int
    why: str

    def tiles(self, dim: int) -> tuple[int, int, int]:
        return (math.ceil(self.M / dim), math.ceil(self.N / dim), math.ceil(self.K / dim))

    def macs(self) -> int:
        return self.M * self.N * self.K


WORKLOADS = [
    Workload("w1_small",    32,  32,  32, "little headroom -- the negative control"),
    Workload("w2_medium",   64,  64,  64, "square, mid-size"),
    Workload("w3_n_heavy",  16, 512, 256, "A-reload dominates; just inside the capacity cliff"),
    Workload("w4_over_cap", 32, 512, 512, "past the cliff: the frozen default cannot stage it"),
]


def histogram(w: Workload, recipe: dict, dim: int) -> dict[str, float]:
    """Command counts for one (workload, recipe), in closed form from the frozen loop nest.

    Derived from `lowering/isa.py:137-189` (the live non-transposed path) and `build_trace:446-447`
    (the leading FENCE+FLUSH). PRELOAD is 1:1 with COMPUTE there, which is exactly the fold the cost
    model documents, so PRELOAD is not a separate regressor.
    """
    Mt, Nt, Kt = w.tiles(dim)
    resident = recipe["activation_residency"]
    cfg_policy = recipe.get("config_policy", "per_mvin")

    # A-tile transfers. `per_tile` reloads A[mi,kk] once per N block (the MVIN sits in the innermost
    # M loop, under N); `panel` hoists it to once per (kk,mi). Both keep the TILE shape, so both stay
    # inside the model's calibration domain -- a full K-row-panel MVIN would not (no byte term).
    mvin_a = Kt * Nt * Mt if resident == "per_tile" else Kt * Mt
    mvin_b = Kt * Nt                      # each B tile is MVIN'd once in every variant

    # CONFIG_LD: one before every MVIN in the frozen emitter, or one per stride per channel when the
    # emitter re-emits only on change. Plus the per-matmul CONFIG_EX + CONFIG_ST prologue.
    config_ld = (mvin_a + mvin_b) if cfg_policy == "per_mvin" else 2
    config = 2 + config_ld

    return {
        "config": config,
        "mvin_A": mvin_a,
        "mvin2_B": mvin_b,
        "compute": Kt * Nt * Mt,
        "mvout": Mt * Nt,                 # invariant across recipes, by contract
        "fence": 1,                       # build_trace's leading FENCE
    }


def staged_rows(w: Workload, recipe: dict, dim: int, spad_rows: int) -> tuple[int, bool, str]:
    """Operand-store rows the recipe needs live, and whether they fit. The frozen emitter stages the
    WHOLE A and B grids (`isa.py:133-134`), which is why a large shape collides rather than spills."""
    Mt, Nt, Kt = w.tiles(dim)
    a_rows = Mt * Kt * dim
    b_rows = Kt * Nt * dim
    total = a_rows + b_rows
    fits = total <= spad_rows
    reason = "" if fits else (
        f"A({a_rows}) + B({b_rows}) = {total} rows exceeds the {spad_rows}-row operand store: the "
        f"frozen lowering stages both grids whole, so this shape collides (Kt*(Mt+Nt)={Kt*(Mt+Nt)} > "
        f"{spad_rows // dim})")
    return total, fits, reason


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--target", default="gemmini")
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--no-product", action="store_true",
                    help="print only; do not mint an out/artifacts product dir")
    args = ap.parse_args(argv)

    dim, spad_rows = machine_facts(args.target)
    model = GemminiCostModel.load()
    band_pct = 100.0 * model.error.get("mape", 0.0)

    dims = WAVE_A_PLUS
    names = list(dims)
    combos: list[dict] = [{}]
    for d in names:
        combos = [dict(c, **{d: v}) for c in combos for v in dims[d]]

    rows = []
    for w in WORKLOADS:
        Mt, Nt, Kt = w.tiles(dim)
        base_hist = histogram(w, DEFAULT_RECIPE, dim)
        base_cyc, _ = model.predict_with_band(base_hist)
        for recipe in combos:
            hist = histogram(w, recipe, dim)
            cyc, band = model.predict_with_band(hist)
            total_rows, fits, fit_reason = staged_rows(w, recipe, dim, spad_rows)
            blind = [MODEL_BLIND[(d, v)] for d, v in recipe.items() if (d, v) in MODEL_BLIND]
            is_default = recipe == DEFAULT_RECIPE
            rows.append({
                "workload": w.wid, "M": w.M, "N": w.N, "K": w.K,
                "Mt": Mt, "Nt": Nt, "Kt": Kt, "macs": w.macs(),
                **{f"recipe_{k}": v for k, v in recipe.items()},
                "is_default": is_default,
                **{f"n_{k}": int(v) for k, v in hist.items()},
                "predicted_cycles": round(cyc, 1),
                "predicted_band": round(band, 1),
                "speedup_vs_default": round(base_cyc / cyc, 4) if cyc > 0 else "",
                "delta_pct_vs_default": round(100.0 * (base_cyc - cyc) / base_cyc, 2),
                "exceeds_band": abs(100.0 * (base_cyc - cyc) / base_cyc) > band_pct,
                "model_visible": not blind,
                "model_blind_reason": " | ".join(blind),
                "staged_rows": total_rows, "fits_operand_store": fits, "fit_reason": fit_reason,
            })

    # ---- report
    print(f"machine (derived): DIM={dim}  scratchpad_rows={spad_rows}  "
          f"capacity relation Kt*(Mt+Nt) <= {spad_rows // dim}")
    print(f"cost model: MAPE band +/-{band_pct:.2f}%  n_points={model.error.get('n_points')}  "
          f"fidelity={model.meta.get('fidelity')!r}\n")

    visible = [r for r in rows if r["model_visible"]]
    print("MODEL-VISIBLE cells (activation_residency x config_policy), cycles vs the frozen default:")
    hdr = f"{'workload':<13}{'resident':<10}{'config':<11}{'cycles':>10}{'delta%':>9}{'>band':>7}{'fits':>6}"
    print(hdr); print("-" * len(hdr))
    for r in visible:
        print(f"{r['workload']:<13}{r['recipe_activation_residency']:<10}"
              f"{r['recipe_config_policy']:<11}{r['predicted_cycles']:>10.0f}"
              f"{r['delta_pct_vs_default']:>9.1f}{str(r['exceeds_band']):>7}"
              f"{str(r['fits_operand_store']):>6}")

    print("\nper-workload best model-visible recipe:")
    winners = {}
    for w in WORKLOADS:
        cand = [r for r in visible if r["workload"] == w.wid and r["fits_operand_store"]]
        if not cand:
            print(f"  {w.wid:<13} NO FITTING RECIPE -- {[r['fit_reason'] for r in rows if r['workload']==w.wid][0]}")
            continue
        best = min(cand, key=lambda r: r["predicted_cycles"])
        winners[w.wid] = (best["recipe_activation_residency"], best["recipe_config_policy"])
        print(f"  {w.wid:<13} {best['recipe_activation_residency']}+{best['recipe_config_policy']}"
              f"  {best['predicted_cycles']:.0f} cyc  ({best['delta_pct_vs_default']:+.1f}% vs default)")

    spread = {}
    for w in WORKLOADS:
        cand = [r["predicted_cycles"] for r in visible
                if r["workload"] == w.wid and r["fits_operand_store"]]
        if cand:
            spread[w.wid] = 100.0 * (max(cand) - min(cand)) / max(cand)
    over = [k for k, v in spread.items() if v > band_pct]
    print(f"\nrecipe spread per workload (max-min)/max: "
          + ", ".join(f"{k} {v:.1f}%" for k, v in spread.items()))
    print(f"GATE: spread exceeds the {band_pct:.2f}% band on {len(over)}/{len(spread)} workloads "
          f"({', '.join(over) if over else 'none'}) -> {'PROCEED' if len(over) >= 2 else 'STOP, widen the space'}")

    blind_dims = sorted({d for (d, _v) in MODEL_BLIND})
    print(f"\nNOT priced here (needs the simulator): {', '.join(blind_dims)} -- "
          f"{len(rows) - len(visible)}/{len(rows)} cells. Reason is recorded per row, never as a 0 delta.")
    nofit = [w.wid for w in WORKLOADS
             if not any(r["fits_operand_store"] for r in rows if r["workload"] == w.wid and r["is_default"])]
    if nofit:
        print(f"Frozen default does NOT fit the operand store for: {', '.join(nofit)} "
              f"-> an expressiveness result, reported separately from any cycle claim.")

    if args.no_product:
        return 0

    # ---- provenance. This artifact claims no hardware verdict, but every number in it is derived
    # from bytes: the calibrated coefficients, and the FROZEN lowering whose loop nest the closed-form
    # counts above transcribe. Digesting `isa.py` is the load-bearing part -- if that file moves, the
    # formulas are stale and the CSV is wrong while still looking right.
    frozen_isa = (REPO / "out/artifacts/targets" / args.target
                  / "gemmini_xdsl_rtl_v0/mlir_oot/lowering/isa.py")
    sources = [p for p in (frozen_isa, Path(GemminiCostModel.DEFAULT_ARTIFACT), Path(__file__))
               if p.exists()]
    pins = {}
    try:
        pins["gemmini_rtl"] = PROV.verify("gemmini_rtl")
    except Exception as exc:                      # pin unlocatable: record the gap, never assume ok
        print(f"\nprovenance: gemmini_rtl NOT verified -- {type(exc).__name__}: {exc}")
    prov = PROV.record(pins=pins, sources=sources)
    if pins:
        v = pins["gemmini_rtl"]
        print(f"\nprovenance: gemmini_rtl ok={v.ok}"
              + ("" if v.ok else f" drift={list(v.drift)[:2]}"))

    prod = new_product("recipe-select", version=args.version, target=args.target,
                       notes="milestone 0: cost-model pricing of the recipe surface, no simulator")
    csv_path = prod.add_artifact("costmodel_surface.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    meta_path = prod.add_artifact("costmodel_surface.json")
    meta_path.write_text(json.dumps({
        "target": args.target, "dim": dim, "scratchpad_rows": spad_rows,
        "cost_model": {"const": model.const, "coeff": model.coeff, "error": model.error,
                       "meta": model.meta, "band_pct": band_pct},
        "default_recipe": DEFAULT_RECIPE,
        "dimensions": dims,
        "model_blind": {f"{d}={v}": r for (d, v), r in MODEL_BLIND.items()},
        "workloads": [asdict(w) for w in WORKLOADS],
        "spread_pct": spread, "workloads_over_band": over,
        "gate": "PROCEED" if len(over) >= 2 else "STOP",
        "provenance": prov,
        "winners": {k: {"activation_residency": a, "config_policy": c} for k, (a, c) in winners.items()},
    }, indent=1), encoding="utf-8")
    prod.write_manifest()
    print(f"\nproduct: {prod.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
