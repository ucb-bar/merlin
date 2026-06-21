#!/usr/bin/env python3
"""Bring up EVERY model on the int8 INTEGER-compute datapath on the RVV vector path, on
spike (functional, no FPGA contention — the Saturn-OPU hang is FASED-specific, so spike
validates the identical `-march=rv64gcv` image's correctness + codegen).

For each int8 bundle: build backend=rvv int8_compute=True, run on spike (-p<harts>), gate
multi-tier (T1 W8A8 cos>0.999&rel<1e-2 if golden_w8a8.npy present; T2 fp32 cos>0.99&argmax),
and objdump the model object to assert genuine integer RVV (vmul.vv/vadd.vv/vmacc/vwmacc on
vle8-loaded i8) — i.e. it actually vectorized, not scalar.

Resumable JSONL ledger. Usage:
    .venv/bin/python build_tools/scripts/spike_int8_rvv_sweep.py [BUNDLE ...] \
        --ledger /scratch/agustin/tmp/spike_int8_rvv.jsonl
With no BUNDLE args, sweeps the default fitting->large order below.
"""
import argparse, json, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "merlin" / "python"))
from merlin.runtime.backends import zephyr_model as zm  # noqa: E402

# small -> large; pi05 excluded (task #61: its int8 capture is unquantized fp32, not a real
# int8 test). The 1B-class (rdt/smolvla/groot/molmoact) are functional-slow on spike but run.
DEFAULT_ORDER = [
    "small_llama", "tiny_llama", "bitvla", "xr0", "openvla", "rdt2",
    "rdt", "smolvla", "groot_n1d7", "molmoact",
]

RVV_INT_OPS = ("vmacc", "vwmacc", "vmul.vv", "vadd.vv", "vsext", "vle8")


def objdump_rvv(model_o: Path) -> dict:
    """Count integer-RVV evidence in the compiled model object."""
    if not model_o.is_file():
        return {"present": False}
    try:
        objdump = zm._spike.gcc_path().with_name("riscv64-unknown-elf-objdump")
        out = subprocess.run([str(objdump), "-d", str(model_o)], capture_output=True,
                             text=True, timeout=300).stdout
    except Exception as e:
        return {"present": True, "error": str(e)[:120]}
    counts = {op: out.count(op) for op in RVV_INT_OPS}
    counts["vsetvli"] = out.count("vsetvli") + out.count("vsetivli")
    counts["any_rvv"] = sum(out.count(op) for op in RVV_INT_OPS) > 0
    return counts


def already_done(ledger: Path, bundle: str) -> bool:
    if not ledger.is_file():
        return False
    for line in ledger.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("bundle") == bundle and r.get("ran"):
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundles", nargs="*")
    ap.add_argument("--ledger", default="/scratch/agustin/tmp/spike_int8_rvv.jsonl")
    ap.add_argument("--workroot", default="/scratch/agustin/tmp/merlin_spike_rvv")
    ap.add_argument("--harts", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=10800)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ledger = Path(args.ledger)

    names = args.bundles or DEFAULT_ORDER
    for name in names:
        bundle = name if name.endswith("_consistent") else f"{name}_int8_consistent"
        if not args.force and already_done(ledger, bundle):
            print(f"SKIP {bundle} (already ran)", flush=True)
            continue
        mdir = ROOT / "output" / bundle
        rec = {"bundle": bundle, "t": time.strftime("%Y-%m-%dT%H:%M:%S"), "backend": "rvv"}
        work = Path(args.workroot) / bundle
        try:
            golden = np.load(mdir / "golden.npy")
            refs = {"fp32": golden}
            w8a8 = mdir / "golden_w8a8.npy"
            if w8a8.is_file():
                refs["w8a8"] = np.load(w8a8)
            rec["tiers"] = list(refs)
            r = zm.build_and_run(mdir, work, board="spike_riscv64", backend="rvv",
                                 int8_compute=True, references=refs, harts=args.harts,
                                 timeout=args.timeout)
            rec["ran"] = True
            rec.update(cos=r.get("cos"), rel=r.get("rel"), ok=bool(r.get("ok")),
                       w8a8_cos=r.get("w8a8_cos"), w8a8_rel=r.get("w8a8_rel"),
                       fp32_cos=r.get("fp32_cos"), fp32_argmax=r.get("fp32_argmax"),
                       cycles=r.get("metrics", {}).get("cycles"))
            rec["rvv"] = objdump_rvv(work / "model.o")
            print(f"SPIKE {bundle}: cos={rec.get('cos')} ok={rec['ok']} "
                  f"any_rvv={rec['rvv'].get('any_rvv')} cyc={rec.get('cycles')}", flush=True)
        except Exception as e:
            rec["ran"] = False
            rec["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
            print(f"SPIKE {bundle} ERR: {rec['error']}", flush=True)
            traceback.print_exc()
        with ledger.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
