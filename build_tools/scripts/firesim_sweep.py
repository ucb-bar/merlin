#!/usr/bin/env python3
"""Build chipyard_riscv64 scalar Zephyr images for a set of model bundles and run each on
FireSim through the firesim-queue (single FPGA -> the queue serializes them), gating cos
against golden.npy. Resumable: results are appended to a JSONL ledger and already-passed
(bundle, dtype) pairs are skipped on re-run.

Usage:
    .venv/bin/python build_tools/scripts/firesim_sweep.py BUNDLE [BUNDLE ...]
    .venv/bin/python build_tools/scripts/firesim_sweep.py --ledger /tmp/fs_sweep.jsonl ...

Each BUNDLE is a directory name under out/artifacts/recaptures/ (e.g. rdt2_int8_consistent). The build runs
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


def cycle_report(ledger: Path) -> int:
    """Pair <model>_int8 vs <model>_fp32 cycle counts -> speedup table, PER BACKEND (scalar vs
    rvv are not comparable — the int8 win lives on the Saturn-OPU vector tile). Uses the most
    recent row per (model, dtype, backend)."""
    if not ledger.is_file():
        print(f"no ledger at {ledger}"); return 1
    # (backend, model) -> {dtype: cycles}; last row wins (ledger is append-only / time-ordered)
    cyc: dict[tuple, dict[str, int]] = {}
    for line in ledger.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("cycles") is None:
            continue
        b = r.get("bundle", "")
        backend = r.get("backend", "scalar")
        parts = b.split("_")
        if "consistent" in parts:
            parts = parts[:parts.index("consistent")]
        dtype = parts[-1] if parts else ""
        model = "_".join(parts[:-1]) if len(parts) > 1 else b
        cyc.setdefault((backend, model), {})[dtype] = int(r["cycles"])
    print(f"{'backend':8s} {'model':18s} {'fp32_cycles':>14s} {'int8_cycles':>14s} {'speedup':>9s}")
    for (backend, model) in sorted(cyc):
        d = cyc[(backend, model)]
        fp, i8 = d.get("fp32"), d.get("int8")
        sp = f"{fp / i8:.2f}x" if (fp and i8) else "-"
        print(f"{backend:8s} {model:18s} {str(fp or '-'):>14s} {str(i8 or '-'):>14s} {sp:>9s}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundles", nargs="*")
    ap.add_argument("--ledger", default="/tmp/fs_sweep.jsonl")
    ap.add_argument("--int8", action="store_true",
                    help="run the W8A8 INTEGER-compute datapath (i8xi8->i32 + integer "
                         "nonlinears) and gate multi-tier vs golden_w8a8.npy + golden.npy")
    ap.add_argument("--report", action="store_true",
                    help="print the int8-vs-fp32 cycle/speedup table from the ledger and exit")
    ap.add_argument("--backend", default="scalar", choices=("scalar", "rvv"),
                    help="scalar (Gemmini tile 0) or rvv (Saturn-OPU vector tile 1). The int8 "
                         "throughput win lives on rvv — i8 lanes pack ~4x an f32 lane.")
    ap.add_argument("--rvv-hart", type=int, default=1,
                    help="hart the rvv model object runs on (Saturn-OPU tile = hart 1)")
    # Build scratch goes on /scratch (3 TB), NOT /tmp (the small shared root disk) — a
    # multi-GB external-weights image in /tmp pressures root, where /home caches already
    # sit at ~94%. /scratch is the project's filesystem with terabytes free.
    ap.add_argument("--workroot", default="/path/to/tmp/merlin_fs")
    ap.add_argument("--timeout", type=int, default=5400)
    ap.add_argument("--force", action="store_true", help="re-run even if already passed")
    args = ap.parse_args()
    ledger = Path(args.ledger)

    if args.report:
        return cycle_report(ledger)

    for bundle in args.bundles:
        if not args.force and already_done(ledger, bundle):
            print(f"SKIP {bundle} (already passed in ledger)", flush=True)
            continue
        mdir = ROOT / "artifacts" / "recaptures" / bundle
        rec = {"bundle": bundle, "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "int8_compute": bool(args.int8), "backend": args.backend}
        try:
            golden = np.load(mdir / "golden.npy")
            # 1. build the chipyard image locally (int8: real W8A8 integer datapath). The rvv
            #    backend targets the Saturn-OPU vector tile (hart 1) where i8xi8->i32 packs ~4x
            #    the lanes of f32 — that is where the int8 throughput win actually shows up.
            b = zm.build_app(mdir, f"{args.workroot}/{bundle}", board="chipyard_riscv64",
                             backend=args.backend, rvv_hart=args.rvv_hart, cpus=2,
                             int8_compute=args.int8)
            rec["ram_mb"] = b["ram_bytes"] // (1024 * 1024)
            print(f"BUILT {bundle} ram={rec['ram_mb']}MB -> submitting to firesim-queue",
                  flush=True)
            # 2. run on FireSim through the queue, gate cos. For int8, gate multi-tier vs the
            #    W8A8 reference (T1) + the fp32 golden (T2); golden_w8a8.npy is generated by
            #    run_model(int8_compute=True) (the host W8A8 sim) and may be absent.
            if args.int8:
                w8a8_path = mdir / "golden_w8a8.npy"
                refs = {"fp32": golden}
                if w8a8_path.is_file():
                    refs["w8a8"] = np.load(w8a8_path)
                r = zm.run_on_firesim(b["elf"], references=refs, queue=True, timeout=args.timeout)
                rec.update(w8a8_cos=r.get("w8a8_cos"), w8a8_rel=r.get("w8a8_rel"),
                           fp32_cos=r.get("fp32_cos"), fp32_argmax=r.get("fp32_argmax"))
            else:
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
