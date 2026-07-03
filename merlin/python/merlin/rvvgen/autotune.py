"""``merlin-rvv-autotune`` — AUTOMATICALLY enumerate, build, benchmark and rank impr forks.

This is what should have produced the combined (vfmacc+LMUL) and tiled forks without hand-minting:
given a set of forkable compiler features (from the action catalog / registry), it enumerates
candidate forks (the empty baseline, every single feature, and bounded combinations), mints each as
an isolated ``impr_<target>`` fork off the FROZEN baseline, builds it with the features actually
applied, benchmarks on real K1 silicon (N runs, min wall), gates correctness vs golden, and ranks
by measured cycles. The frozen baseline is always candidate 0 — every fork is measured against it.

Deterministic + versioned: writes an ``autotune_<target>_v{V}_{ts}/`` run with the full ranking, so
the search is auditable and the winner is evidence-backed, not asserted.
"""
from __future__ import annotations

import argparse
import itertools
import shutil
import statistics
import time
from pathlib import Path

import numpy as np
import yaml

from ..runtime.backends import zephyr_model as zm
from . import k1 as k1mod
from .registry import load_rvv_package
from merlin.common.paths import repo_root

_REPO = Path(__file__).resolve().parents[4]


def enumerate_candidates(features: list[str], max_combo: int = 2) -> list[frozenset]:
    """Baseline (∅) + every single feature + combinations up to ``max_combo``."""
    cands: list[frozenset] = [frozenset()]
    for r in range(1, max_combo + 1):
        for combo in itertools.combinations(sorted(features), r):
            cands.append(frozenset(combo))
    return cands


def _mint_fork(base_dir: Path, out_dir: Path, feats: frozenset, ts: str, idx: int) -> Path:
    fork = out_dir / f"impr_auto_{idx}_{ts}"
    shutil.rmtree(fork, ignore_errors=True)
    shutil.copytree(base_dir, fork)
    k = fork / "knobs.yaml"
    k.write_text(k.read_text() + f"\ncompiler_features: [{', '.join(sorted(feats))}]\n")
    m = fork / "manifest.yaml"
    m.write_text(m.read_text().replace("run_id: hand_v0", f"run_id: {fork.name}")
                 .replace("status: spike_verified", "status: experimental")
                 + f"\nlineage: {{parent_run_id: hand_v0, features: [{', '.join(sorted(feats))}]}}\n")
    return fork


def _bench_k1(pkg, model_dir: Path, work: Path, golden, n_runs: int) -> dict:
    """Build (features applied) -> deploy -> run N times -> {min_wall, median, cos} (honest None on fail)."""
    try:
        binary = k1mod.build_k1_binary(model_dir, work, pkg)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"build: {type(e).__name__}: {str(e)[:120]}"}
    remote = f"/tmp/autotune_{work.name}_merlin_k1"
    try:
        k1mod._run(["scp", "-i", k1mod.K1_SSH_KEY, "-o", "BatchMode=yes",
                    "-o", "StrictHostKeyChecking=no", str(binary), f"{k1mod.K1_HOST}:{remote}"])
        k1mod._ssh(f"chmod +x {remote}", timeout=30)
        walls, cos = [], None
        for _ in range(n_runs):
            proc = k1mod._ssh(remote, timeout=300)
            res = zm._parse_console(proc.stdout + proc.stderr, proc.returncode)
            w = res.get("metrics", {}).get("wall_ns")
            if w:
                walls.append(w)
            if cos is None:
                g = zm._gate(res["prefix"], {"fp32": golden})
                cos = g.get("fp32_cos") or g.get("cos")
        k1mod._ssh(f"rm -f {remote}", timeout=30)
        if not walls:
            return {"ok": False, "reason": "no wall_ns parsed"}
        return {"ok": True, "min_wall": min(walls), "median_wall": statistics.median(walls), "cos": cos}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"run: {type(e).__name__}: {str(e)[:120]}"}


def autotune(target: str, base_pkg_dir: str | Path, model_dir: str | Path, features: list[str], *,
             out_root: str | Path | None = None, fork_root: str | Path | None = None,
             max_combo: int = 2, n_runs: int = 3, cos_gate: float = 0.999) -> Path:
    base_dir = Path(base_pkg_dir)
    model_dir = Path(model_dir)
    golden = np.load(model_dir / "golden.npy")
    fork_root = Path(fork_root) if fork_root else _REPO / "artifacts/targets" / target
    out_root = Path(out_root) if out_root else _REPO / "artifacts" / "kernel-mining" / target
    ts = time.strftime("%Y%m%dT%H%M%S")
    cands = enumerate_candidates(features, max_combo=max_combo)
    results = []
    for i, feats in enumerate(cands):
        fork = _mint_fork(base_dir, fork_root, feats, ts, i)
        pkg = load_rvv_package(fork)
        work = repo_root() / "build" / "tmp" / f"autotune_{target}_{i}_{ts}"
        shutil.rmtree(work, ignore_errors=True)
        b = _bench_k1(pkg, model_dir, work, golden, n_runs)
        b["features"] = sorted(feats)
        b["fork"] = fork.name
        results.append(b)
        print(f"[{i}] features={sorted(feats) or '(baseline)'}: "
              + (f"min_wall={b['min_wall']} cos={b['cos']}" if b["ok"] else f"FAIL {b['reason']}"),
              flush=True)
    # rank correct + measured by min wall; baseline is the ∅ candidate
    base = next((r for r in results if r.get("features") == [] and r["ok"]), None)
    ok = [r for r in results if r["ok"] and (r["cos"] or 0) >= cos_gate]
    ok.sort(key=lambda r: r["min_wall"])
    for r in results:
        if r["ok"] and base and base["min_wall"]:
            r["speedup_vs_baseline"] = round(base["min_wall"] / r["min_wall"], 3)
            r["correct"] = (r["cos"] or 0) >= cos_gate
    v = len(list(out_root.glob(f"autotune_{target}_v*"))) + 1
    run_dir = out_root / f"autotune_{target}_v{v}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "ranking.yaml").write_text(yaml.safe_dump({
        "target": target, "op_dir": str(model_dir), "n_runs": n_runs, "created": ts,
        "candidates": results,
        "ranked_correct": [{"features": r["features"], "min_wall": r["min_wall"],
                            "speedup": r.get("speedup_vs_baseline"), "cos": r["cos"]} for r in ok],
    }, sort_keys=False))
    print(f"\n=== autotune ranking (correct, by K1 min wall) -> {run_dir.name} ===")
    for r in ok:
        print(f"  {r.get('speedup_vs_baseline'):>6}x  {r['features'] or '(baseline)'}")
    return run_dir


def beam_search(target: str, base_pkg_dir: str | Path, model_dir: str | Path, features: list[str], *,
                width: int = 2, depth: int = 3, n_runs: int = 3, cos_gate: float = 0.999,
                fork_root: str | Path | None = None, out_root: str | Path | None = None) -> Path:
    """Go-wide-then-combine beam over the forkable features, ranked by MEASURED K1 cycles.

    gen 0: baseline + every single feature (go wide). Keep the top-``width`` correct survivors.
    gen d>0: extend each survivor by ONE not-yet-present feature (combine), benchmark, re-rank,
    keep top-``width``. Stop when a generation yields nothing new or no improvement over the best.
    Only promising branches are explored — so it scales to many features without the all-pairs blow-up.
    """
    base_dir = Path(base_pkg_dir); model_dir = Path(model_dir)
    golden = np.load(model_dir / "golden.npy")
    fork_root = Path(fork_root) if fork_root else _REPO / "artifacts/targets" / target
    out_root = Path(out_root) if out_root else _REPO / "artifacts" / "kernel-mining" / target
    ts = time.strftime("%Y%m%dT%H%M%S")
    evaluated: dict[frozenset, dict] = {}
    counter = [0]

    def evalc(feats: frozenset) -> dict:
        if feats in evaluated:
            return evaluated[feats]
        fork = _mint_fork(base_dir, fork_root, feats, ts, counter[0]); counter[0] += 1
        pkg = load_rvv_package(fork)
        work = repo_root() / "build" / "tmp" / f"beam_{target}_{counter[0]}_{ts}"
        shutil.rmtree(work, ignore_errors=True)
        r = _bench_k1(pkg, model_dir, work, golden, n_runs)
        r["features"] = sorted(feats); r["fork"] = fork.name
        r["correct"] = bool(r.get("ok") and (r.get("cos") or 0) >= cos_gate)
        evaluated[feats] = r
        print(f"  eval {sorted(feats) or '(baseline)'}: "
              + (f"min_wall={r['min_wall']} cos={r['cos']} correct={r['correct']}"
                 if r["ok"] else f"FAIL {r['reason']}"), flush=True)
        return r

    base = evalc(frozenset())
    base_wall = base.get("min_wall")
    gen_log = []
    # generation 0 — go wide
    g0 = [evalc(frozenset([f])) for f in features]
    pool = [r for r in g0 if r["correct"]]
    pool.sort(key=lambda r: r["min_wall"])
    survivors = pool[:width]
    gen_log.append({"gen": 0, "survivors": [r["features"] for r in survivors]})
    best = min([r for r in ([base] + g0) if r.get("ok") and r["correct"]],
               key=lambda r: r["min_wall"], default=base)
    for d in range(1, depth):
        cands = {frozenset(p["features"]) | {f}
                 for p in survivors for f in features if f not in p["features"]}
        cands = [c for c in cands if c not in evaluated]
        if not cands:
            break
        rs = [evalc(c) for c in cands]
        allcorrect = [r for r in (survivors + rs) if r["correct"]]
        allcorrect.sort(key=lambda r: r["min_wall"])
        survivors = allcorrect[:width]
        gen_best = min([r for r in rs if r["correct"]], key=lambda r: r["min_wall"], default=None)
        gen_log.append({"gen": d, "survivors": [r["features"] for r in survivors]})
        if gen_best and gen_best["min_wall"] < best["min_wall"]:
            best = gen_best
        else:
            break  # no improvement this generation -> stop
    v = len(list(out_root.glob(f"beam_{target}_v*"))) + 1
    run_dir = out_root / f"beam_{target}_v{v}_{ts}"; run_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted([r for r in evaluated.values() if r["correct"]], key=lambda r: r["min_wall"])
    for r in ranked:
        r["speedup_vs_baseline"] = round(base_wall / r["min_wall"], 3) if base_wall else None
    (run_dir / "beam_tree.yaml").write_text(yaml.safe_dump({
        "target": target, "op_dir": str(model_dir), "width": width, "depth": depth, "created": ts,
        "generations": gen_log, "n_evaluated": len(evaluated),
        "best": {"features": best["features"], "min_wall": best.get("min_wall"),
                 "speedup": round(base_wall / best["min_wall"], 3) if base_wall and best.get("min_wall") else None},
        "ranked": [{"features": r["features"], "min_wall": r["min_wall"],
                    "speedup": r.get("speedup_vs_baseline"), "cos": r["cos"]} for r in ranked],
    }, sort_keys=False))
    print(f"\n=== beam result -> {run_dir.name} (evaluated {len(evaluated)}, not the full {2**len(features)}) ===")
    print(f"  BEST: {best['features'] or '(baseline)'}  "
          f"{round(base_wall/best['min_wall'],3) if base_wall and best.get('min_wall') else '?'}x")
    for r in ranked[:width + 1]:
        print(f"   {r.get('speedup_vs_baseline')}x  {r['features'] or '(baseline)'}")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default="rvv")
    ap.add_argument("--base", default="artifacts/targets/rvv/hand_v0")
    ap.add_argument("--workload", required=True)
    ap.add_argument("--features", required=True, help="comma list of registered impr features")
    ap.add_argument("--max-combo", type=int, default=2)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--beam", action="store_true", help="beam search (go wide, keep top-k, combine)")
    ap.add_argument("--width", type=int, default=2)
    ap.add_argument("--depth", type=int, default=3)
    a = ap.parse_args(argv)
    if a.beam:
        beam_search(a.target, a.base, a.workload, a.features.split(","),
                    width=a.width, depth=a.depth, n_runs=a.runs)
    else:
        autotune(a.target, a.base, a.workload, a.features.split(","),
                 max_combo=a.max_combo, n_runs=a.runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
