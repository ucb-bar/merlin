"""Does the compiler change GENERALIZE? Fit a rule on the sweep shapes, test it on shapes it never saw.

WHY THIS EXISTS. The sweep found that one recipe value (`activation_residency=panel`) wins on every
shape it was measured on. A recipe that always wins is not something to select per workload -- it is a
compiler DEFAULT that is wrong, and the deliverable is to change the compiler. But "it won on the
three shapes I chose" is exactly the claim that overfits, so the rule is stated first and then run on
a held-out set drawn to include the cases most likely to break it:

* ``Nt == 1`` -- no N sweep, so the saving `Mt*Kt*(Nt-1)` is identically ZERO. If the rule still
  prefers `panel` here it is preferring a change that buys nothing, and if `panel` LOSES here the rule
  is actively harmful. Two held-out shapes probe this at very different Mt.
* shapes whose operand grids do NOT fit -- where `panel` is not merely worse but inexpressible, and
  the rule must fall back rather than emit something the machine cannot hold.
* a mid-range shape with non-power-of-two tiles, so the rule is not accidentally reading a divisibility
  property of the fitting set.

THE RULE, stated before the evidence was collected so it could not be tuned to it -- and then
SIMPLIFIED by a finding rather than by taste:

    proposed:  activation_residency = panel if Kt*(Mt+Nt) <= operand_rows/DIM else per_tile
    actual:    activation_residency = panel, unconditionally

The capacity predicate does not discriminate. Both residency values reserve the SAME rows -- the
frozen lowering stages the whole activation grid either way, and `panel` changes only how many times a
tile is TRANSFERRED. So a shape that defeats one defeats the other, and there is no shape where
`per_tile` is preferable: never faster, never more expressible. That makes the change a plain default
flip, which is the stronger claim because there is no predicate left to get wrong.

VERDICT SHAPE. The rule PASSES only if, on every held-out shape, it is no worse than either fixed
policy. "Ties on the zero-saving shapes and wins elsewhere" is the expected pass; a single held-out
loss falsifies it and the default must not change.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for cand in (here, *here.parents):
        if (cand / "merlin" / "python").is_dir():
            return cand
    raise SystemExit("could not locate repo root")


REPO = _repo_root()
sys.path.insert(0, str(REPO / "merlin" / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from merlin.targetgen import oot_runner as OOT      # noqa: E402
from merlin.common.artifacts import new_product     # noqa: E402
import _track as T                                  # noqa: E402
from merlin.common import provenance as PROV        # noqa: E402

FORK = REPO / "out/artifacts/targets/gemmini/gemmini_xdsl_recipe_v0"
GSIM_EMU = ("/scratch/agustin/tmp/gsim_cert_serialclk_v1/"
            "emu_gemmini_gsim_serialclk_v1_filtered_final")
GSIM_SHA = "fb356ede610fb5f5ecbe2edb61dfd9a5a196293408a5ea02f34f919b5e39916b"
DIM = 16

#: Shapes used to FIT the rule -- listed so the held-out set can be checked disjoint from them.
FITTING = {(32, 32, 32), (64, 64, 64), (16, 512, 256)}

#: HELD OUT. Never measured before this script runs. Each names what it is designed to break.
HELDOUT = [
    ("h1_no_n_sweep",       16,  16, 128, "Nt=1: the saving is identically zero, so `panel` must not "
                                          "be preferred and must not lose"),
    ("h2_tall_m_no_sweep", 128,  16, 128, "Nt=1 again but Mt=8: probes whether the zero-saving case "
                                          "depends on M"),
    ("h3_odd_tiles",        48,  96,  48, "Mt=3, Nt=6, Kt=3: non-power-of-two tiles, so the rule "
                                          "cannot be reading a divisibility property"),
    ("h4_wide_n",           16, 256, 128, "Mt=1, Nt=16, Kt=8: the regime the fitting set says should "
                                          "win big, at a shape it never saw"),
    ("h5_past_capacity",    32, 512, 512, "the operand grids do not fit: `panel` is inexpressible and "
                                          "the rule must fall back rather than emit it"),
]

IFACE = """module attributes {{merlin_iface.version = "0.1", merlin_iface.target = "gemmini", \
merlin_iface.abi_version = "0.1"}} {{
  %W = merlin_iface.tensor {{name = "W", role = "weight"}} : tensor<{K}x{N}xi8>
  %A0 = merlin_iface.tensor {{name = "A0", role = "input"}} : tensor<{M}x{K}xi8>
  %W_res = merlin_iface.resident_pack %W {{layout = "packed_rhs"}} : (tensor<{K}x{N}xi8>) \
-> !merlin_iface.resident
  %acc0 = merlin_iface.matmul %A0, %W_res : (tensor<{M}x{K}xi8>, !merlin_iface.resident) \
-> !merlin_iface.acc<i32>
  %Y0 = merlin_iface.commit %acc0 {{name = "Y0", epilogue = [], output_dtype = "i32"}} : \
(!merlin_iface.acc<i32>) -> tensor<{M}x{N}xi32>
  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
}}
"""


def _recipe_mod():
    sys.path.insert(0, str(FORK / "mlir_oot" / "lowering"))
    import recipe                                                    # noqa: PLC0415
    return recipe


def rule(m: int, n: int, k: int, *, dim: int, spad_rows: int, acc_rows: int) -> tuple[str, str]:
    """THE RULE under test -- delegated to the COMPILER's own ``resolve_auto`` so there is one
    authority for what the proposed default means. Returns (value, why).

    ⚠️ The rule got SIMPLER than the docstring at the top of this file first proposed, and the reason
    is a finding rather than a simplification. The intended predicate was "panel where both operand
    grids fit, per_tile otherwise". It does not discriminate: both residency values reserve the SAME
    rows (the frozen lowering stages the whole activation grid either way -- only the number of
    TRANSFERS differs), so a shape that defeats one defeats the other. What is left is a plain default
    flip, which is a stronger claim: there is no predicate to get wrong and nothing to select.

    A shape past the operand-store bound is therefore NOT a case this rule handles by falling back --
    it is a compiler coverage gap needing a blocked-residency value the surface does not yet have, and
    the verdict logic below scores it as inexpressible rather than as evidence about the rule.
    """
    R = _recipe_mod()
    chosen = R.resolve_auto(m, n, k, dim=dim, spad_rows=spad_rows)
    mt, nt, kt = -(-m // dim), -(-n // dim), -(-k // dim)
    f = R.fit(R.Recipe(activation_residency=chosen), m=m, n=n, k=k, dim=dim,
              spad_rows=spad_rows, acc_rows=acc_rows)
    if not f.ok:
        return chosen, f"the shape is inexpressible on this surface: {f.reason.split(':')[0]}"
    if nt == 1:
        return chosen, ("Nt=1, so the saving Mt*Kt*(Nt-1) is exactly zero -- the rule must TIE here, "
                        "and a loss would falsify it")
    return chosen, f"saves Mt*Kt*(Nt-1) = {mt * kt * (nt - 1)} activation transfers"


def evaluate(mlir: Path, recipe: dict | None, run_id: str, engine: str, timeout: int) -> dict:
    prev = os.environ.get("MERLIN_CODEGEN_RECIPE")
    if recipe is None:
        os.environ.pop("MERLIN_CODEGEN_RECIPE", None)
    else:
        os.environ["MERLIN_CODEGEN_RECIPE"] = json.dumps(recipe)
    t0 = time.time()
    try:
        res = OOT.certify(FORK, mlir, runs_root=T.RUNS, run_id=run_id,
                          simulator=engine, target="gemmini", timeout=timeout)
        err = None
    except Exception as exc:
        res, err = {}, f"{type(exc).__name__}: {exc}"
    wall = time.time() - t0
    if prev is None:
        os.environ.pop("MERLIN_CODEGEN_RECIPE", None)
    else:
        os.environ["MERLIN_CODEGEN_RECIPE"] = prev
    o = (res or {}).get("oracle") or {}
    return {"cycles": o.get("cycles"), "correct": (res or {}).get("status") == "pass",
            "status": (res or {}).get("status"), "wall_s": round(wall, 2), "error": err}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--engine", default="gsim")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--concurrency", type=int, default=1,
                    help="how many sims were running alongside; recorded, never assumed")
    ap.add_argument("--version", type=int, default=1)
    args = ap.parse_args(argv)
    T.assert_frozen_intact()   # this track never edits the champion; prove it

    if args.engine == "gsim":
        got = PROV.file_digest(Path(GSIM_EMU))
        if got != GSIM_SHA:
            raise SystemExit(f"gsim emu digest {got} != certified {GSIM_SHA}")
        os.environ["MERLIN_GEMMINI_GSIM_EMU"] = GSIM_EMU
        os.environ["MERLIN_GEMMINI_GSIM_MAXCYCLES"] = "100000000"

    # The held-out set must share no shape with the fitting set, or "generalises" means nothing.
    heldout_shapes = {(m, n, k) for _hid, m, n, k, _why in HELDOUT}
    overlap = FITTING & heldout_shapes
    if overlap:
        raise SystemExit(f"held-out set overlaps the fitting set at {sorted(overlap)}: a rule cannot "
                         f"be tested on shapes it was fitted on")

    from merlin.perf.workload_gen import tile_geometry
    from merlin.targetgen.rtl import facts as rtl_facts
    geom = tile_geometry("gemmini")
    body = rtl_facts.load_facts("gemmini")
    mems = (body.get("facts") or body).get("memories") or []
    spad = next(x for x in mems if x["name"] == "scratchpad")["bytes"] // geom.rows
    acc = next(x for x in mems if x["name"] == "accumulator")["bytes"] // (geom.rows * 4)

    work = REPO / "out" / "build" / "recipe_select_heldout"
    work.mkdir(parents=True, exist_ok=True)

    rows = []
    for hid, M, N, K in [(h[0], h[1], h[2], h[3]) for h in HELDOUT]:
        why = next(h[4] for h in HELDOUT if h[0] == hid)
        mlir = work / f"{hid}.interface.mlir"
        mlir.write_text(IFACE.format(M=M, N=N, K=K), encoding="utf-8")
        chosen, reason = rule(M, N, K, dim=geom.rows, spad_rows=spad, acc_rows=acc)
        mt, nt, kt = -(-M // geom.rows), -(-N // geom.rows), -(-K // geom.rows)
        row = {"heldout": hid, "M": M, "N": N, "K": K, "Mt": mt, "Nt": nt, "Kt": kt,
               "probes": why, "rule_choice": chosen, "rule_reason": reason,
               "engine": args.engine, "concurrency": args.concurrency, "arms": {}}
        R = _recipe_mod()
        for arm in ("per_tile", "panel"):
            f = R.fit(R.Recipe(activation_residency=arm), m=M, n=N, k=K, dim=geom.rows,
                      spad_rows=spad, acc_rows=acc)
            if not f.ok:
                row["arms"][arm] = {"legal": False, "reason": f.reason, "cycles": None}
                print(f"{hid:<20} {arm:<9} ILLEGAL: {f.reason[:70]}", flush=True)
                continue
            rec = None if arm == "per_tile" else {"activation_residency": "panel",
                                                  "drain": "inline"}
            r = evaluate(mlir, rec, f"ho_{hid}_{arm}", args.engine, args.timeout)
            r["legal"] = True
            row["arms"][arm] = r
            print(f"{hid:<20} {arm:<9} cycles={r['cycles']} correct={r['correct']} "
                  f"wall={r['wall_s']}s", flush=True)
        rows.append(row)

    print("\n================ HELD-OUT VERDICT ================")
    losses, ties, wins, undet, inexpressible = [], [], [], [], []
    for r in rows:
        legal = {a: v for a, v in r["arms"].items() if v.get("legal")}
        cyc = {a: v["cycles"] for a, v in legal.items()
               if isinstance(v.get("cycles"), int) and v.get("correct")}
        pick = r["rule_choice"]
        if pick not in cyc:
            # Three different things land here and only one is a rule failure. Collapsing them would
            # let a COMPILER COVERAGE gap read as a falsified rule, which would wrongly block a
            # default change that the evidence actually supports.
            any_legal = any(v.get("legal") for v in r["arms"].values())
            if not any_legal:
                # No value of this dimension can express the shape: the frozen lowering stages both
                # operand grids whole, so nothing on the wave-A surface fits. That is a gap in what
                # the COMPILER can emit (it needs a blocked-residency value), not a wrong rule.
                inexpressible.append((r["heldout"], r["arms"]["panel"]["reason"]))
                verdict = "N/A  "
            elif not legal.get(pick, {}).get("legal", False):
                losses.append((r["heldout"], "the rule chose a value that is illegal here while a "
                                             "legal alternative existed -- rule is WRONG"))
                verdict = "WRONG"
            else:
                undet.append((r["heldout"], "the rule's choice is legal but produced no usable "
                                            "cycle count"))
                verdict = "UNDET"
            print(f"  {r['heldout']:<20} rule={pick:<9} {verdict} "
                  f"{'no wave-A value fits this shape' if not any_legal else ''}")
            continue
        best = min(cyc, key=lambda a: cyc[a])
        delta = {a: cyc[a] for a in cyc}
        margin = (max(cyc.values()) - min(cyc.values()))
        verdict = "WINS " if best == pick and margin else ("TIES " if margin == 0 else "LOSES")
        if verdict == "LOSES":
            losses.append((r["heldout"], f"rule picked {pick} at {cyc[pick]}, best was {best} "
                                         f"at {cyc[best]}"))
        elif verdict == "TIES ":
            ties.append(r["heldout"])
        else:
            wins.append((r["heldout"], f"{100.0 * margin / max(cyc.values()):.1f}% better"))
        print(f"  {r['heldout']:<20} rule={pick:<9} {verdict} {delta}")

    # A shape no wave-A value can express does not bear on whether the rule RANKS correctly, so it
    # is excluded from the verdict and reported as the coverage gap it is.
    ok = not losses and not undet
    print(f"\n  wins={len(wins)} ties={len(ties)} losses={len(losses)} "
          f"undetermined={len(undet)} inexpressible={len(inexpressible)}")
    for name, why in inexpressible:
        print(f"    - {name}: NOT a rule failure -- no wave-A value fits. {why[:110]}")
    for name, d in losses + undet:
        print(f"    ! {name}: {d}")
    print(f"  RULE {'GENERALIZES -- the default may change' if ok else 'FALSIFIED -- do NOT change the default'}")

    prod = new_product("recipe-select", version=args.version, target="gemmini",
                       notes="held-out generalization of the panel-if-fits rule")
    out = prod.add_artifact("heldout_rule.json")
    out.write_text(json.dumps({
        "rule": ("activation_residency = panel, unconditionally on every shape this lowering can "
                 "emit. The capacity predicate originally proposed here does NOT discriminate: both "
                 "residency values reserve the same rows (the lowering stages the whole activation "
                 "grid either way; only the TRANSFER count differs), so a shape that defeats one "
                 "defeats the other."),
        "rule_supersedes": ("panel if Kt*(Mt+Nt) <= operand_rows/DIM else per_tile -- falsified by "
                            "the footprint arithmetic, not by the cycle measurements"),
        "fitting_shapes": sorted(FITTING), "engine": args.engine,
        "machine": {"dim": geom.rows, "operand_rows": spad, "accumulator_rows": acc},
        "rows": rows, "wins": wins, "ties": ties, "losses": losses, "undetermined": undet,
        "inexpressible": inexpressible, "generalizes": ok,
        "inexpressible_means": ("no value of this dimension can express the shape -- a gap in what "
                                "the compiler can EMIT (it needs a blocked-residency value), not "
                                "evidence about the rule; excluded from the verdict"),
        "citation_constraint": ("cycles describe GemminiGsimSerialClkConfig; its accelerator modules "
                               "are identical to stock GemminiRocketConfig (only ClockSourceAtFreqMHz "
                               "x2 and one IO cell differ), but the engines were measured to disagree "
                               "by +1/+6 cycles, so these are not Verilator-equivalent numbers"),
        "provenance": PROV.record(pins={}, sources=[FORK / "mlir_oot/lowering/isa.py",
                                                    FORK / "mlir_oot/lowering/recipe.py",
                                                    Path(__file__)],
                                  artifacts={"gsim_emu": GSIM_EMU}),
    }, indent=1), encoding="utf-8")
    prod.write_manifest()
    print(f"\nproduct: {prod.path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
