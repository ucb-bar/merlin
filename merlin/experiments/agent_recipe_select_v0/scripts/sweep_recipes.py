"""Measure the recipe surface for real: emitted-code deltas plus GSIM cycles, per shape.

This replaces the cost model as the source of truth, because the cost model was FALSIFIED against
measured cycles on two certified capsules -- it predicted 174 vs 302 measured (-42%) on A2 and 1103 vs
604 (+83%) on PK03_k128, against a declared +/-8.15% band, in BOTH directions. Its own metadata says
"linear, serial; no overlap", and the +83% at Kt=8 is what a machine that overlaps DMA with compute
looks like to a model that sums them. So a predicted saving from deleting transfers is not evidence
that the transfers cost anything; only measured cycles are.

TWO CHECKS PER CELL, because either alone can lie:

* the **emitted-code delta** (instruction class histogram from the emitted ``.insn`` stream). A recipe
  whose histogram is unchanged is an inert lever no matter what the cycles do -- the repo has been
  bitten by levers that changed the schedule text and not the code.
* the **measured cycles** on elaborated RTL. A histogram delta with no cycle delta is the opposite
  failure: real code change, no economic change, which is a result and not a bug.

CITATION CONSTRAINT. GSIM here simulates ``GemminiGsimSerialClkConfig``; the frozen package's
certified numbers came from stock ``GemminiRocketConfig`` under Verilator, and the two DISAGREE (302 vs
303, 604 vs 610 -- measured). So every number this script emits describes the serial-clock elaboration
and may not be quoted as Verilator-equivalent. One engine per comparison, named on every row.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import hashlib
from collections import Counter
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
TOOL = FORK / "mlir_oot" / "gemmini_opt.py"

GSIM_EMU = ("/scratch/agustin/tmp/gsim_cert_serialclk_v1/"
            "emu_gemmini_gsim_serialclk_v1_filtered_final")
GSIM_SHA = "fb356ede610fb5f5ecbe2edb61dfd9a5a196293408a5ea02f34f919b5e39916b"

#: Shapes with Nt > 1, which is what `activation_residency` needs to be observable at all. The four
#: certified GSIM shapes are all m=n=16 (Mt=Nt=1), where `panel` and `per_tile` emit the SAME code --
#: so the certified set cannot see this lever and every shape here is deliberately outside it.
WORKLOADS = {
    "w1_small":   (32, 32, 32),
    "w2_medium":  (64, 64, 64),
    "w3_n_heavy": (16, 512, 256),
}

RECIPES = [
    {"activation_residency": "per_tile", "drain": "inline"},      # the frozen default
    {"activation_residency": "per_tile", "drain": "deferred"},
    {"activation_residency": "panel",    "drain": "inline"},
    {"activation_residency": "panel",    "drain": "deferred"},
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

#: RoCC funct -> class name, taken from the emitter's own FUNCT table rather than re-asserted here.
def _funct_names() -> dict[int, str]:
    sys.path.insert(0, str(FORK / "mlir_oot"))
    from lowering.isa import FUNCT                                    # noqa: PLC0415
    out: dict[int, str] = {}
    for name, f in FUNCT.items():
        out.setdefault(f, name)
    out[0] = "CONFIG"
    return out


def emit(mlir: Path, recipe: dict | None) -> str:
    env = dict(os.environ)
    env.pop("MERLIN_CODEGEN_RECIPE", None)
    if recipe is not None:
        env["MERLIN_CODEGEN_RECIPE"] = json.dumps(recipe)
    r = subprocess.run([sys.executable, "gemmini_opt.py", "--convert-iface-to-gemmini",
                        "--emit-target-artifact", str(mlir)],
                       cwd=str(FORK / "mlir_oot"), capture_output=True, text=True, env=env,
                       timeout=600)
    if r.returncode != 0:
        raise SystemExit(f"emit failed for {recipe}: {r.stderr[-600:]}")
    return r.stdout


def histogram(artifact: str, names: dict[int, str]) -> dict[str, int]:
    """Count instruction classes STRUCTURALLY from the emitted asm -- no regex, per the repo rule.

    Each accelerator instruction is one ``.insn r <opcode>, <funct3>, <funct>, ...`` inside an
    ``llvm.inline_asm``; the funct field is the third comma-separated token of the .insn text.
    """
    counts: Counter[str] = Counter()
    for line in artifact.splitlines():
        if "llvm.inline_asm" not in line:
            continue
        if '"fence"' in line:
            counts["FENCE"] += 1
            continue
        _, _, rest = line.partition(".insn ")
        if not rest:
            continue
        body = rest.split('"', 1)[0]
        parts = [p.strip() for p in body.split(",")]
        if len(parts) < 3 or not parts[2].startswith("0x"):
            continue
        counts[names.get(int(parts[2], 16), f"funct_{parts[2]}")] += 1
    return dict(counts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--engine", default="gsim")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--static-only", action="store_true",
                    help="emitted-code deltas only; skip the simulator")
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--workloads", default="", help="comma-separated subset of the workload ids")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="how many simulators ran alongside this sweep. RECORDED, never assumed: "
                         "cycles are concurrency-invariant but wall times are not (6.3x measured), "
                         "so a wall number without its concurrency is unusable")
    ap.add_argument("--all-points", action="store_true",
                    help="enumerate the compiler's FULL catalog instead of the wave-A four; used to "
                         "establish bit-exactness of every exposed value before any search runs")
    args = ap.parse_args(argv)
    T.assert_frozen_intact()   # this track never edits the champion; prove it

    got = PROV.file_digest(Path(GSIM_EMU)) if Path(GSIM_EMU).exists() else None
    if args.engine == "gsim" and got != GSIM_SHA:
        raise SystemExit(f"gsim emu digest {got} != certified {GSIM_SHA}; refusing to cite it")
    os.environ["MERLIN_GEMMINI_GSIM_EMU"] = GSIM_EMU
    os.environ["MERLIN_GEMMINI_GSIM_MAXCYCLES"] = "100000000"

    names = _funct_names()

    global RECIPES, WORKLOADS
    if args.all_points:
        # The legality set comes from the COMPILER's catalog, not from a list written here, so a
        # value added to the surface cannot escape the bit-exactness gate by being forgotten.
        sys.path.insert(0, str(FORK / "mlir_oot" / "lowering"))
        import recipe as _R                                            # noqa: PLC0415
        cat = _R.catalog(m=64, n=64, k=64, dim=16, spad_rows=16384, acc_rows=1024)
        RECIPES = [_R.DEFAULTS] + [p["recipe"] for p in cat["points"]
                                   if p["legal"] and p["recipe"] != _R.DEFAULTS]
        print(f"enumerating the compiler's full catalog: {len(RECIPES)} points "
              f"(default first, so every delta is measured against it)")
    if args.workloads:
        want = {w.strip() for w in args.workloads.split(",") if w.strip()}
        missing = want - set(WORKLOADS)
        if missing:
            raise SystemExit(f"unknown workload id(s) {sorted(missing)}; have {sorted(WORKLOADS)}")
        WORKLOADS = {k: v for k, v in WORKLOADS.items() if k in want}
    work = REPO / "out" / "build" / "recipe_select_workloads"
    work.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for wid, (M, N, K) in WORKLOADS.items():
        mlir = work / f"{wid}.interface.mlir"
        mlir.write_text(IFACE.format(M=M, N=N, K=K), encoding="utf-8")
        base_hist = None
        base_cyc = None
        for idx, recipe in enumerate(RECIPES):
            art = emit(mlir, None if idx == 0 else recipe)
            hist = histogram(art, names)
            # Order-SENSITIVE identity. A histogram counts instructions and therefore cannot see a
            # pure reordering -- and `drain` is exactly a reordering (the store count is an invariant),
            # so judging it by histogram would report a real code change as an inert lever.
            digest = hashlib.sha256(art.encode()).hexdigest()[:16]
            if idx == 0:
                base_hist, base_digest = hist, digest
            row = {
                "workload": wid, "M": M, "N": N, "K": K,
                "Mt": -(-M // 16), "Nt": -(-N // 16), "Kt": -(-K // 16),
                "recipe": recipe, "is_default": idx == 0,
                "instr_histogram": hist,
                "n_instr": sum(hist.values()),
                "artifact_digest": digest,
                "code_differs_from_default": digest != base_digest,
                "instr_counts_differ": hist != base_hist,
                "engine": args.engine,
            }
            if not args.static_only:
                # The run id must name the WHOLE recipe. It used to spell only residency+drain,
                # which was unique while those were the only two dimensions -- adding
                # `config_policy` silently collapsed 20 points onto 10 ids, so half the cells would
                # have overwritten each other and been reported as measured. Content-addressing the
                # recipe makes a collision impossible by construction.
                rkey = hashlib.sha256(
                    json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:8]
                run_id = f"sw_{wid}_{rkey}"
                env_prev = os.environ.get("MERLIN_CODEGEN_RECIPE")
                if idx == 0:
                    os.environ.pop("MERLIN_CODEGEN_RECIPE", None)
                else:
                    os.environ["MERLIN_CODEGEN_RECIPE"] = json.dumps(recipe)
                t0 = time.time()
                try:
                    res = OOT.certify(FORK, mlir, runs_root=T.RUNS, run_id=run_id,
                                      simulator=args.engine, target="gemmini",
                                      timeout=args.timeout)
                    err = None
                except Exception as exc:
                    res, err = {}, f"{type(exc).__name__}: {exc}"
                wall = time.time() - t0
                if env_prev is None:
                    os.environ.pop("MERLIN_CODEGEN_RECIPE", None)
                else:
                    os.environ["MERLIN_CODEGEN_RECIPE"] = env_prev
                oracle = (res or {}).get("oracle") or {}
                row.update({
                    "status": (res or {}).get("status"), "cycles": oracle.get("cycles"),
                    "oracle_kind": oracle.get("kind"),
                    "derived_from_rtl": oracle.get("derived_from_rtl"),
                    "correct": (res or {}).get("status") == "pass",
                    "wall_s": round(wall, 2), "error": err,
                    "concurrency": args.concurrency,
                })
                if idx == 0 and isinstance(row["cycles"], int):
                    base_cyc = row["cycles"]
                if isinstance(row.get("cycles"), int) and base_cyc:
                    row["speedup_vs_default"] = round(base_cyc / row["cycles"], 4)
                    row["delta_pct"] = round(100.0 * (base_cyc - row["cycles"]) / base_cyc, 2)
            rows.append(row)
            tag = "default" if idx == 0 else f"{recipe['activation_residency']}/{recipe['drain']}"
            print(f"{wid:<12} {tag:<20} instr={row['n_instr']:<7} "
                  f"code_delta={str(row['code_differs_from_default']):<6} "
                  f"cycles={row.get('cycles')} status={row.get('status')} "
                  f"wall={row.get('wall_s')}s", flush=True)

    print("\n================ EMITTED-CODE DELTA (static, free) ================")
    for wid in WORKLOADS:
        d = [r for r in rows if r["workload"] == wid]
        base = d[0]["instr_histogram"]
        print(f"  {wid}: default {base}")
        for r in d[1:]:
            diff = {k: r["instr_histogram"].get(k, 0) - base.get(k, 0)
                    for k in set(base) | set(r["instr_histogram"])
                    if r["instr_histogram"].get(k, 0) != base.get(k, 0)}
            tag = f"{r['recipe']['activation_residency']}/{r['recipe']['drain']}"
            if not r["code_differs_from_default"]:
                verdict = "INERT -- emitted code byte-identical to the default"
            elif diff:
                verdict = f"counts change: {diff}"
            else:
                verdict = ("REORDERED ONLY -- same instruction multiset, different order "
                           "(the store-count invariant holds, so this is the intended shape)")
            print(f"    {tag:<20} {verdict}")

    if not args.static_only:
        print("\n================ MEASURED CYCLES ================")
        for wid in WORKLOADS:
            d = [r for r in rows if r["workload"] == wid]
            print(f"  {wid} (Mt,Nt,Kt = {d[0]['Mt']},{d[0]['Nt']},{d[0]['Kt']}):")
            for r in d:
                tag = "default" if r["is_default"] else \
                      f"{r['recipe']['activation_residency']}/{r['recipe']['drain']}"
                print(f"    {tag:<20} cycles={str(r.get('cycles')):<9} "
                      f"{('%+.1f%%' % r['delta_pct']) if r.get('delta_pct') is not None else '':<9}"
                      f" correct={r.get('correct')}")
            good = [r for r in d if isinstance(r.get("cycles"), int) and r.get("correct")]
            if good:
                best = min(good, key=lambda r: r["cycles"])
                tag = "default" if best["is_default"] else \
                      f"{best['recipe']['activation_residency']}/{best['recipe']['drain']}"
                print(f"    -> best: {tag}")

    prod = new_product("recipe-select", version=args.version, target="gemmini",
                       notes=f"wave-A recipe sweep, engine={args.engine}, "
                             f"GemminiGsimSerialClkConfig (NOT verilator-equivalent)")
    out = prod.add_artifact("recipe_sweep.json")
    out.write_text(json.dumps({
        "engine": args.engine, "gsim_emu": GSIM_EMU, "gsim_sha256": GSIM_SHA,
        "config": "chipyard.harness.TestHarness.GemminiGsimSerialClkConfig",
        "citation_constraint": ("cycles describe the serial-clock elaboration; measured to disagree "
                               "with stock GemminiRocketConfig under Verilator (302 vs 303, 604 vs "
                               "610), so they may not be quoted as Verilator-equivalent"),
        "cost_model_falsified": {"A2": {"pred": 174, "measured": 302},
                                 "PK03_k128": {"pred": 1103, "measured": 604},
                                 "declared_band_pct": 8.15},
        "fork": str(FORK), "rows": rows,
        "provenance": PROV.record(pins={}, sources=[FORK / "mlir_oot/lowering/isa.py",
                                                    FORK / "mlir_oot/lowering/recipe.py",
                                                    Path(__file__)],
                                  artifacts={"gsim_emu": GSIM_EMU}),
    }, indent=1), encoding="utf-8")
    prod.write_manifest()
    print(f"\nproduct: {prod.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
