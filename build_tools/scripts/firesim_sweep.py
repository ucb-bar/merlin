#!/usr/bin/env python3
"""Build chipyard_riscv64 scalar Zephyr images for a set of model bundles and run each on
FireSim through the firesim-queue (single FPGA -> the queue serializes them), gating cos
against golden.npy. Resumable: results are appended to a JSONL ledger and already-passed
(bundle, dtype) pairs are skipped on re-run.

Usage:
    .venv/bin/python build_tools/scripts/firesim_sweep.py BUNDLE [BUNDLE ...]
    .venv/bin/python build_tools/scripts/firesim_sweep.py --ledger /tmp/fs_sweep.jsonl ...

Each BUNDLE is a directory name under output/ (e.g. rdt2_int8_consistent). The build runs
locally; the run is submitted to the queue via zephyr_model.run_on_firesim (FIRESIM_QUEUE=1).
"""
import argparse, json, shutil, sys, time, traceback
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "merlin" / "python"))
from merlin.runtime.backends import zephyr_model as zm  # noqa: E402


def already_done(ledger: Path, bundle: str) -> bool:
    if not ledger.is_file():
        return False
    for line in ledger.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("bundle") == bundle and r.get("ok"):
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundles", nargs="+")
    ap.add_argument("--ledger", default="/tmp/fs_sweep.jsonl")
    # Build scratch goes on /scratch (3 TB), NOT /tmp (the small shared root disk) — a
    # multi-GB external-weights image in /tmp pressures root, where /home caches already
    # sit at ~94%. /scratch is the project's filesystem with terabytes free.
    ap.add_argument("--workroot", default="/scratch/agustin/tmp/merlin_fs")
    ap.add_argument("--timeout", type=int, default=5400)
    ap.add_argument("--force", action="store_true", help="re-run even if already passed")
    args = ap.parse_args()
    ledger = Path(args.ledger)

    for bundle in args.bundles:
        if not args.force and already_done(ledger, bundle):
            print(f"SKIP {bundle} (already passed in ledger)", flush=True)
            continue
        mdir = ROOT / "output" / bundle
        rec = {"bundle": bundle, "t": time.strftime("%Y-%m-%dT%H:%M:%S")}
        try:
            golden = np.load(mdir / "golden.npy")
            # 1. build the chipyard scalar image locally
            b = zm.build_app(mdir, f"{args.workroot}/{bundle}", board="chipyard_riscv64",
                             backend="scalar", cpus=2)
            rec["ram_mb"] = b["ram_bytes"] // (1024 * 1024)
            print(f"BUILT {bundle} ram={rec['ram_mb']}MB -> submitting to firesim-queue",
                  flush=True)
            # 2. run on FireSim through the queue, gate cos
            r = zm.run_on_firesim(b["elf"], reference=golden, queue=True,
                                  timeout=args.timeout)
            rec.update(cos=r.get("cos"), rel=r.get("rel"), ok=bool(r.get("ok")),
                       cycles=r.get("metrics", {}).get("cycles"))
            # cos covers only the dumped prefix (<=4096). For larger outputs the model
            # also emits a full-output SUM and a full per-row ARGMAX — gate those too so
            # the WHOLE output is validated, not just its first 4096 elements.
            gflat = golden.astype(np.float32).ravel()
            if gflat.size > 4096:
                checks = []
                if r.get("sum") is not None:
                    g_sum = float(gflat.sum())
                    sum_rel = abs(float(r["sum"]) - g_sum) / max(1e-6, abs(g_sum))
                    rec["sum_rel"] = sum_rel
                    # The SUM check is only meaningful when the true sum is a well-conditioned
                    # aggregate. For mixed-sign outputs (logits) it can be a near-zero residual
                    # of large-magnitude cancellation — bitvla's golden sums to ~1e-4 of its
                    # total |mass|, so a rel-0.004 per-element diff blows sum_rel up to ~0.4
                    # despite cos=0.99999 + 100% argmax. Gate sum only when |sum| is a real
                    # fraction of the L2 magnitude; otherwise ARGMAX governs the full output.
                    l2 = float(np.sqrt((gflat.astype(np.float64) ** 2).sum()))
                    well_conditioned = abs(g_sum) >= 0.1 * l2
                    rec["sum_conditioned"] = well_conditioned
                    if well_conditioned:
                        checks.append(sum_rel < 1e-3)
                if r.get("argmax") is not None:
                    last = golden.shape[-1]
                    g_arg = golden.astype(np.float32).reshape(-1, last).argmax(1)
                    hw = np.asarray(r["argmax"]); k = min(len(hw), len(g_arg))
                    frac = float((hw[:k] == g_arg[:k]).mean()) if k else 0.0
                    rec["argmax_match"] = frac
                    checks.append(frac > 0.999)
                if checks:
                    rec["ok"] = bool(rec["ok"] and all(checks))
            print(f"FSIM {bundle}: cos={rec['cos']:.7f} ok={rec['ok']} "
                  f"sum_rel={rec.get('sum_rel')} argmax={rec.get('argmax_match')} "
                  f"cyc={rec['cycles']}", flush=True)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
            rec["ok"] = False
            print(f"FSIM {bundle} ERR: {rec['error']}", flush=True)
            traceback.print_exc()
        with ledger.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        # The elf is staged into the firesim deploy dir by run_firesim, so the multi-GB
        # local build tree is no longer needed — drop it to keep disk from filling during
        # a long sweep. Keep it only when the build itself failed (no ram_mb), for debug.
        wd = Path(args.workroot) / bundle
        if rec.get("ram_mb") is not None and wd.is_dir():
            shutil.rmtree(wd, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
