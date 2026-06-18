#!/usr/bin/env python3
"""CAPSTONE: lower + execute EVERY model through the int8 (W8A8) datapath on the SpacemiT K1 RVV
multicore board, and gate correctness against the torch goldens.

For each int8 bundle (small -> large): build a K1 Linux rv64gcv binary via the int8 package
(rvvgen.k1.build_k1_binary, which threads int8_compute=True), scp + run on the board, parse
OUT/METRIC/DONE, gate the output prefix vs golden_w8a8 (T1) and the fp32 golden (T2), objdump the
model object to confirm genuine integer RVV was emitted, and record a resumable JSONL ledger +
coverage table. The board run cleans its own /tmp (run_on_k1); we clean the host work dir per model
to bound disk (big int8 binaries embed the weight blob).

pi05 EXCLUDED (16G; its int8 capture is unquantized fp32 — task #61). groot (1.8G) is marginal vs
the board's 1.9G /tmp; it runs last and a fit failure is recorded honestly, not hidden.

Usage:
  MERLIN_K1_HOST=root@10.44.97.186 .venv/bin/python build_tools/scripts/k1_int8_model_sweep.py \
      --ledger /scratch/agustin/tmp/k1_int8.jsonl
"""
import argparse, json, shutil, sys, time, traceback
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "merlin" / "python"))
from merlin.rvvgen import load_rvv_package           # noqa: E402
from merlin.rvvgen import k1 as k1mod                  # noqa: E402
from merlin.runtime.backends import zephyr_model as zm  # noqa: E402
from merlin.llvmlower import custom_isa                 # noqa: E402

# small -> large (bundle sizes from inventory); pi05 excluded.
DEFAULT_ORDER = ["small_llama", "bitvla", "openvla", "rdt2", "rdt", "xr0",
                 "tiny_llama", "smolvla", "molmoact", "groot_n1d7"]
PKG = ROOT / "generated_targets" / "rvv" / "hand_v0_int8"
RVV_INT = ("vmul.vv", "vwmacc", "vmacc", "vsext", "vle8", "vadd.vv")


def already(ledger: Path, bundle: str) -> bool:
    if not ledger.is_file():
        return False
    for ln in ledger.read_text().splitlines():
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if r.get("bundle") == bundle and r.get("ran"):
            return True
    return False


def objdump_int_rvv(model_o: Path) -> dict:
    if not model_o.is_file():
        return {"present": False}
    try:
        out = custom_isa.disassemble(model_o)
    except Exception as e:  # noqa: BLE001
        return {"present": True, "error": str(e)[:120]}
    counts = {op: out.count(op) for op in RVV_INT}
    counts["any_int_rvv"] = any(out.count(op) for op in RVV_INT)
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundles", nargs="*")
    ap.add_argument("--ledger", default="/scratch/agustin/tmp/k1_int8.jsonl")
    ap.add_argument("--workroot", default="/scratch/agustin/tmp/k1_int8_work")
    ap.add_argument("--host", default=None, help="root@<ip> (else env MERLIN_K1_HOST)")
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.host:
        k1mod.K1_HOST = a.host
    if not k1mod.K1_HOST:
        print("ERROR: set --host or MERLIN_K1_HOST", flush=True)
        return 2
    if not k1mod.available():
        print(f"ERROR: K1 unavailable (host={k1mod.K1_HOST}, toolchain={k1mod.toolchain_cc()})",
              flush=True)
        return 2
    pkg = load_rvv_package(PKG)
    ledger = Path(a.ledger)
    names = a.bundles or DEFAULT_ORDER

    for name in names:
        bundle = name if name.endswith("_consistent") else f"{name}_int8_consistent"
        if not a.force and already(ledger, bundle):
            print(f"SKIP {bundle} (done)", flush=True)
            continue
        mdir = ROOT / "output" / bundle
        rec = {"bundle": bundle, "t": time.strftime("%Y-%m-%dT%H:%M:%S"), "target": "k1"}
        if not (mdir / "model.mlir").is_file():
            rec.update(ran=False, error="bundle missing")
            _append(ledger, rec); print(f"MISS {bundle}", flush=True); continue
        work = Path(a.workroot) / bundle
        t0 = time.time()
        try:
            refs = {"fp32": np.load(mdir / "golden.npy")}
            w8 = mdir / "golden_w8a8.npy"
            if w8.is_file():
                refs["w8a8"] = np.load(w8)
            res = k1mod.run_on_k1(mdir, work, pkg, timeout=a.timeout)
            gate = zm._gate(res["prefix"], refs)
            rec.update(ran=True, gate_ok=bool(gate.get("ok")),
                       cos=gate.get("cos"), rel=gate.get("rel"),
                       w8a8_cos=gate.get("w8a8_cos"), w8a8_rel=gate.get("w8a8_rel"),
                       fp32_cos=gate.get("fp32_cos"), fp32_argmax=gate.get("fp32_argmax"),
                       vlen=res.get("vlen"),
                       wall_ns=res.get("metrics", {}).get("wall_ns"),
                       cycles_est=res.get("metrics", {}).get("cycles"),
                       tiers=list(refs))
            # run_on_k1 builds into work/<mode>/ (v|omp|scalar); find the model.o that was used.
            mo = next(iter(sorted(work.rglob("model.o"), key=lambda p: p.stat().st_mtime,
                                  reverse=True)), work / "model.o")
            rec["int_rvv"] = objdump_int_rvv(mo)
            print(f"K1 {bundle}: ok={rec['gate_ok']} cos={rec.get('cos')} "
                  f"int_rvv={rec['int_rvv'].get('any_int_rvv')} wall_ms="
                  f"{(rec.get('wall_ns') or 0)/1e6:.1f} vlen={rec.get('vlen')}", flush=True)
        except Exception as e:  # noqa: BLE001
            rec.update(ran=False, error=f"{type(e).__name__}: {str(e).splitlines()[0][:300]}")
            print(f"K1 {bundle} ERR: {rec['error']}", flush=True)
            traceback.print_exc()
        rec["build_run_s"] = round(time.time() - t0, 1)
        _append(ledger, rec)
        shutil.rmtree(work, ignore_errors=True)   # bound host disk (big int8 binaries)

    _coverage(ledger, names)
    return 0


def _append(ledger: Path, rec: dict) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def _coverage(ledger: Path, names: list) -> None:
    rows = {}
    for ln in ledger.read_text().splitlines() if ledger.is_file() else []:
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        rows[r["bundle"]] = r
    print("\n=== K1 int8 coverage ===", flush=True)
    print(f"{'model':<32} ran   gate  cos       int_rvv  wall_ms", flush=True)
    for name in names:
        b = name if name.endswith("_consistent") else f"{name}_int8_consistent"
        r = rows.get(b)
        if not r:
            print(f"{b:<32} -", flush=True); continue
        if not r.get("ran"):
            print(f"{b:<32} FAIL  -     -         -        {r.get('error','')[:40]}", flush=True)
            continue
        print(f"{b:<32} yes   {str(r.get('gate_ok')):<5} {str(r.get('cos'))[:8]:<9} "
              f"{str(r.get('int_rvv',{}).get('any_int_rvv')):<8} "
              f"{(r.get('wall_ns') or 0)/1e6:.1f}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
