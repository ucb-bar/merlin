"""Pin the oracle: run the SAME ELF on Verilator and GSIM, compare CYCLES and measure wall time.

Two open questions block the experiment's budget and its metric, and one run answers both.

1. **CYCLES AGREEMENT IS NOT CERTIFIED.** The gemmini GSIM equivalence certificate
   (`out/artifacts/perf-bench/gemmini/tuning_certificate_v1/`) has 4 members, every one carrying
   ``evidence: output_bytes`` / ``bytes_match: true`` and **no ``cycles`` field on either side**. Its
   ``fidelity: elaborated_rtl_cycle_accurate`` labels the ENGINE CLASS, not an agreement between the
   two engines -- and the two engines are different elaborations (``GemminiGsimSerialClkConfig`` vs
   stock ``GemminiRocketConfig``), so equal cycle counts are not implied. An experiment whose only
   metric is cycles cannot inherit a bytes certificate.
2. **WALL COST IS CONTESTED BY ~229x.** ``capsule_runner.py:3354-3362`` records gemmini Verilator at
   0.255 ms/cycle (median 0.276 s/capsule, 42 samples); a peer session measured ~0.0584 s/cycle plus a
   ~149 s fixed term at m=n=16. The second makes a 69k-cycle workload ~67 min instead of ~18 s, which
   decides whether the planned shapes are affordable at all.

Two shapes are run, not one, because a single point cannot separate a rate from a fixed term -- the
same "at least two points per fitted parameter" rule `merlin/python/merlin/perf/AGENT.md` states.

Nothing here is a verdict on the recipe surface; it measures the instrument.
"""
from __future__ import annotations

import argparse
import json
import os
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

from merlin.targetgen import oot_runner as OOT          # noqa: E402
from merlin.common.artifacts import new_product          # noqa: E402
import _track as T                                       # noqa: E402
from merlin.common import provenance as PROV             # noqa: E402

FROZEN = REPO / "out/artifacts/targets/gemmini/gemmini_xdsl_rtl_v0"

#: Two shapes inside the GSIM certificate's own covered set (m=n=16), so a cycles disagreement here
#: cannot be blamed on running outside what was certified. Both are single-output-tile, which is also
#: exactly where `activation_residency` is a no-op -- that is the certificate's limitation, recorded.
CASES = [
    ("A2_single_tile_matmul", REPO / "merlin/contract/capsules/isa/A2_single_tile_matmul", 302),
    ("PK03_k128",             REPO / "merlin/contract/capsules/_perf/PK03_k128",           None),
]

_GSIM_DIR = Path("/scratch/agustin/tmp/gsim_cert_serialclk_v1")

#: TWO gsim binaries. The OPERATIVE 4-member certificate
#: (``out/artifacts/perf-bench/gemmini/tuning_certificate_v1/gsim_equivalence_certificate.json``)
#: pins ``gsim_binary`` = ``..._v1_filtered_final`` (fb356ede), and that is also the binary in
#: production -- so certified and in-use agree. Do not be misled by
#: ``/scratch/agustin/tmp/gsim_cert_serialclk_v1/certificate.stdout``: that is an EARLIER 1-member
#: intermediate (A2 only) which pins ``..._v1_final`` (ae599b04), and reading it as "the certificate"
#: inverts the attribution. ``_final`` is byte-identical to the unsuffixed build. The uncertified
#: ``_final`` is kept here as a third arm purely to see whether "filtered" moved the cycle count.
GSIM_BINARIES = [
    ("gsim_certified", _GSIM_DIR / "emu_gemmini_gsim_serialclk_v1_filtered_final",
     "fb356ede610fb5f5ecbe2edb61dfd9a5a196293408a5ea02f34f919b5e39916b",
     "certified by the operative 4-member certificate, and the binary in production"),
    ("gsim_unfiltered", _GSIM_DIR / "emu_gemmini_gsim_serialclk_v1_final",
     "ae599b04dc3ea548d5a7653fb7b3b7eeb48de6fc31e0a747095a85f752c03197",
     "the pre-filter build; UNCERTIFIED, measured only as a control on what filtering changed"),
]
GSIM_MAXCYCLES = "100000000"


def _iface(capsule_dir: Path) -> Path:
    cand = capsule_dir / "capsule.interface.mlir"
    if not cand.exists():
        raise SystemExit(f"no capsule.interface.mlir under {capsule_dir}")
    return cand


def run_cell(engine: str, name: str, capsule_dir: Path, runs_root: Path, timeout: int) -> dict:
    """One (engine, capsule) cell. Records wall time beside cycles: cycles are
    concurrency-invariant, wall times are not, so a wall number without its concurrency is unusable."""
    run_id = f"pin_{engine}_{name}"
    t0 = time.time()
    try:
        res = OOT.certify(FROZEN, _iface(capsule_dir), runs_root=runs_root, run_id=run_id,
                          simulator=engine, target="gemmini", timeout=timeout)
        err = None
    except Exception as exc:                       # only an internal harness bug reaches here
        res, err = {}, f"{type(exc).__name__}: {exc}"
    wall = time.time() - t0
    oracle = (res or {}).get("oracle") or {}
    return {
        "engine": engine, "capsule": name, "wall_s": round(wall, 2),
        "status": (res or {}).get("status"),
        "cycles": oracle.get("cycles"),
        "oracle_kind": oracle.get("kind"),
        "derived_from_rtl": oracle.get("derived_from_rtl"),
        "cycle_accurate": oracle.get("cycle_accurate"),
        "oracle_result": oracle.get("result"),
        "error": err,
        "concurrency": 1,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--engines", default="verilator,gsim")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--no-product", action="store_true")
    args = ap.parse_args(argv)
    T.assert_frozen_intact()   # this track never edits the champion; prove it

    gsim_state = []
    for label, path, sha, why in GSIM_BINARIES:
        if not path.exists():
            gsim_state.append((label, path, False, f"absent at {path}", why)); continue
        got = PROV.file_digest(path)
        ok = got == sha
        gsim_state.append((label, path, ok, f"digest {got}" + ("" if ok else f" != declared {sha}"), why))
    for label, _p, ok, note, why in gsim_state:
        print(f"{label:<16} {'OK     ' if ok else 'PROBLEM'} {note}  ({why})")
    print("certified binary == binary in production (per the operative 4-member certificate); "
          "the unfiltered build is a control arm, not a candidate")

    runs_root = T.RUNS
    rows = []
    wanted = [e.strip() for e in args.engines.split(",") if e.strip()]
    for engine in wanted:
        variants = ([(engine, None, True, "", "")] if engine != "gsim"
                    else [(l, p, o, n, w) for l, p, o, n, w in gsim_state])
        for label, emu_path, ok, note, _why in variants:
            if emu_path is not None:
                if not ok:
                    print(f"SKIP {label}: {note} -- refusing to cite a binary whose bytes are unconfirmed")
                    rows.append({"engine": label, "capsule": "*", "error": note, "skipped": True})
                    continue
                os.environ["MERLIN_GEMMINI_GSIM_EMU"] = str(emu_path)
                os.environ["MERLIN_GEMMINI_GSIM_MAXCYCLES"] = GSIM_MAXCYCLES
            for name, cdir, expect in CASES:
                print(f"\n--- {label} x {name}"
                      + (f" (frozen manifest records ~{expect} cyc)" if expect else ""), flush=True)
                row = run_cell("gsim" if emu_path is not None else engine, name, cdir,
                               runs_root, args.timeout)
                row["engine"] = label
                row["emu"] = str(emu_path) if emu_path else None
                row["expected_cycles_from_manifest"] = expect
                rows.append(row)
                print(f"    status={row['status']} cycles={row['cycles']} wall={row['wall_s']}s"
                      f" kind={row['oracle_kind']} err={row['error']}", flush=True)

    # ---- the two questions
    print("\n================ CYCLES AGREEMENT ================")
    byc: dict[str, dict[str, object]] = {}
    for r in rows:
        if r.get("skipped"):
            continue
        byc.setdefault(r["capsule"], {})[r["engine"]] = r["cycles"]
    agree = {}
    for cap, per in byc.items():
        v, g = per.get("verilator"), per.get("gsim_certified")
        if isinstance(v, int) and isinstance(g, int):
            agree[cap] = (v == g)
            rel = 100.0 * abs(g - v) / v if v else float("inf")
            print(f"  {cap:<24} verilator={v:<8} gsim={g:<8} "
                  f"{'AGREE' if v == g else f'DISAGREE by {rel:.1f}%'}")
        else:
            print(f"  {cap:<24} verilator={v} gsim={g} -> UNDETERMINABLE (a missing cycle count is "
                  f"not an agreement)")
    if agree and all(agree.values()):
        print("  => GSIM cycles are usable as the experiment's metric on these shapes.")
    elif agree:
        print("  => GSIM cycles DIVERGE from Verilator. The two engines are different elaborations; "
              "pick ONE engine for every cited number and never mix them in one comparison.")

    print("\n================ WALL COST (rate + fixed term) ================")
    for engine in sorted({r["engine"] for r in rows if not r.get("skipped")}):
        pts = [(r["cycles"], r["wall_s"]) for r in rows
               if r["engine"] == engine and isinstance(r.get("cycles"), int)]
        print(f"  {engine}: " + ", ".join(f"{c} cyc -> {w}s" for c, w in pts))
        if len(pts) >= 2:
            (c0, w0), (c1, w1) = sorted(pts)
            if c1 != c0:
                rate = (w1 - w0) / (c1 - c0)
                fixed = w0 - rate * c0
                print(f"    fitted: {rate*1000:.4f} ms/cycle marginal, {fixed:.1f}s fixed term "
                      f"(2 points, so exactly determined -- not a validated law)")
                for cyc, label in ((7322, "w2_medium"), (68917, "w3_n_heavy"), (238035, "w4_over_cap")):
                    print(f"      projected {label:<12} ({cyc:>6} cyc): {(fixed + rate*cyc)/60:.1f} min/candidate")
        else:
            print("    only one usable point: a rate and a fixed term cannot both be fitted from it")

    if args.no_product:
        return 0
    prod = new_product("recipe-select", version=args.version, target="gemmini",
                       notes="oracle pinning: cycles agreement verilator-vs-gsim + measured wall cost")
    out = prod.add_artifact("oracle_cost.json")
    out.write_text(json.dumps({
        "rows": rows, "cycles_agreement": agree,
        "gsim": {"binaries": [{"label": l, "path": str(p), "digest_ok": o, "note": n, "why": w}
                              for l, p, o, n, w in gsim_state],
                 "maxcycles": GSIM_MAXCYCLES,
                 "config": "chipyard.harness.TestHarness.GemminiGsimSerialClkConfig"},
        "verilator": {"config": "GemminiRocketConfig"},
        "frozen_package": str(FROZEN),
        "caveat": ("the GSIM equivalence certificate covers output BYTES on 4 single-output-tile "
                   "shapes (m=n=16), not cycles, and m=n=16 is exactly where activation_residency "
                   "is a no-op; every experiment workload is outside that certified set"),
        "provenance": PROV.record(pins={}, sources=[FROZEN / "mlir_oot/lowering/isa.py", Path(__file__)]),
    }, indent=1), encoding="utf-8")
    prod.write_manifest()
    print(f"\nproduct: {prod.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
