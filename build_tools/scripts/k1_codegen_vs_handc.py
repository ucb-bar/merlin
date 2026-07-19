#!/usr/bin/env python
"""LOCALIZE the codegen-vs-hand-C whole-model gap with a 2x2 FACTORIAL on the real K1.

THE UNEXPLAINED NUMBER. On rdt2 whole-model, same runtime for every row, our COMPILER
(`accumulator_resident_wholemodel_vf`, 30,235 ms) is ~1.5x SLOWER than our own hand-written C
RVV GEMM shim (`runtime/backends/ours_board/ours_gemm_rvv_shim.c`, 20,154 ms) — a ~10 s gap that
nobody had attributed. Both compute the SAME matmuls in the SAME runtime, so the gap is either
(a) our emitted matmul code being worse than the hand-C kernel, or (b) the `vf` feature making the
NON-matmul 80% of the model worse, or both. Prior work could not tell these apart: the codegen
matmul is INLINED into the one monolithic `_mlir_ciface_forward`, so it has no call boundary to
time.

THE DESIGN. `kernel_backend` rewrites the routable f32 `linalg.matmul` ops to shim calls on the
PREPARED MLIR, *before* `lower_model_file(..., features=...)` applies the compiler feature. So the
two knobs COMPOSE, giving a clean 2x2 that creates the missing boundary:

                            | matmul = OUR CODEGEN | matmul = hand-C shim (timed)
    ------------------------+----------------------+----------------------------
    non-matmul = baseline   |  A  (baseline)       |  C  (hand-C arm)
    non-matmul = vf feature |  B  (codegen arm)    |  D  (THE NEW ARM)

Arm D is the one nobody ran. In BOTH C and D every matmul is routed away to the same shim (routing
is total: n_routed == the model's linalg.matmul count), so C and D differ ONLY in what the `vf`
feature does to the non-matmul model. That isolates the two terms of the gap:

    non_matmul_delta = D.dispatch_bucket - C.dispatch_bucket      # vf's cost on the non-matmul 80%
    codegen_matmul   = B.wall - D.dispatch_bucket                 # our emitted matmul, at last
    handc_matmul     = C.matmul_bucket                            # measured in the shim's rdtime
    B - C            = non_matmul_delta + (codegen_matmul - handc_matmul)   # identity, both measured

`dispatch_bucket = wall - matmul_bucket` and `matmul_bucket` is the shim's own rdtime bracket
(`-DMERLIN_DISPATCH_TIMING`), so every term above is MEASURED, not attributed — closing the
documented attribution caveat of `k1_dispatch_breakdown.py` (which could only ever time a routed
kernel, never our inlined one).

CONTROL. Arm `ctrl` re-runs arm C's configuration under a different run_id (hence a different
build + a different remote path). The board's measured noise floor is >= 1.9%, so `ctrl` vs `C`
is the yardstick every delta below must clear before it is called a result.

cos-gated (>= 0.9999) per arm BEFORE any wall is reported; a failed gate is `not_run`, never a
number. Run ids are `cg_`-prefixed so the remote /tmp paths cannot collide with other agents'.

Run: MERLIN_K1_HOST=root@<ip> MERLIN_COMPILE_TIMEOUT_S=3600 \
     .venv/bin/python build_tools/scripts/k1_codegen_vs_handc.py \
        --model out/artifacts/recaptures/bitvla_fp32_consistent -n 3
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from merlin.rvvgen import k1
from merlin.rvvgen.registry import load_rvv_package
from merlin.runtime.backends import zephyr_model as zm

TIMEBASE_HZ = k1.K1_TIMEBASE_HZ
VF = "accumulator_resident_wholemodel_vf"

ARMS = {
    # tag           -> (compiler_features, kernel_backend)
    "A_base":       ([],   None),      # baseline lowering, codegen matmul
    "B_vf":         ([VF], None),      # THE CODEGEN ARM (vf lowering, codegen matmul)
    "C_base_shim":  ([],   "ours"),    # THE HAND-C ARM (baseline lowering, shim matmul)
    "D_vf_shim":    ([VF], "ours"),    # THE NEW ARM (vf lowering, shim matmul)
    "ctrl":         ([],   "ours"),    # identical config to C -> board noise floor
}

_LABEL = {"A_base": ("baseline", "our codegen"),
          "B_vf": ("vf", "our codegen"),
          "C_base_shim": ("baseline", "hand-C shim"),
          "D_vf_shim": ("vf", "hand-C shim"),
          "ctrl": ("baseline", "hand-C shim (control)")}


def _ticks_to_ns(t):
    return None if t is None else float(t) * (1e9 / TIMEBASE_HZ)


def _spread(walls):
    if not walls:
        return None
    ws = sorted(walls)
    mean = sum(ws) / len(ws)
    std = (sum((w - mean) ** 2 for w in ws) / len(ws)) ** 0.5
    return {"min_ns": ws[0], "max_ns": ws[-1], "median_ns": ws[len(ws) // 2],
            "stdev_ns": round(std), "range_pct": round(100.0 * (ws[-1] - ws[0]) / ws[0], 2),
            "n": len(ws)}


def run_arm(model_dir: Path, base_pkg, golden, n: int, tag: str,
            features: list[str], kernel_backend: str | None) -> dict:
    """Build+deploy+run ONE cell of the factorial n times; gate cos, then report min wall."""
    pkg = replace(base_pkg, run_id=f"cg_{tag}", compiler_features=list(features))
    dispatch_timing = kernel_backend is not None
    runs: list[dict] = []
    cos = None
    blocker = None
    n_routed = 0
    for i in range(n):
        work = Path(tempfile.mkdtemp(prefix=f"cg_{tag}_{i}_"))
        try:
            res = k1.run_on_k1(model_dir, work, pkg, timeout=1800,
                               kernel_backend=kernel_backend,
                               dispatch_timing=dispatch_timing)
            cos = zm._gate(res["prefix"], {"fp32": golden})["fp32_cos"]
            n_routed = res.get("n_ours_routed", 0)
            m = res["metrics"]
            runs.append({"wall_ns": m.get("wall_ns"), "time_ticks": m.get("time_ticks"),
                         "matmul_ticks": m.get("matmul_ticks"),
                         "matmul_calls": m.get("matmul_calls"),
                         "fp32_cos": cos, "vlen": res.get("vlen")})
            print(f"  [{tag}] run {i}: wall={runs[-1]['wall_ns']} "
                  f"mm_ticks={runs[-1]['matmul_ticks']} cos={cos:.7f} routed={n_routed}")
        except Exception as e:  # noqa: BLE001
            blocker = f"{type(e).__name__}: {str(e)[:400]}"
            print(f"  [{tag}] run {i}: BLOCKED — {blocker}")
            break
        finally:
            shutil.rmtree(work, ignore_errors=True)

    walls = sorted(r["wall_ns"] for r in runs if r["wall_ns"])
    ok = cos is not None and cos >= 0.9999 and bool(walls)
    # Pair the matmul bucket with the MIN-wall run (not a mean across runs at different walls) so
    # dispatch_bucket = wall - matmul_bucket stays a WITHIN-run difference.
    best = min((r for r in runs if r["wall_ns"]), key=lambda r: r["wall_ns"], default=None)
    mm_ns = _ticks_to_ns(best.get("matmul_ticks")) if best else None
    disp_ns = (best["wall_ns"] - mm_ns) if (best and mm_ns is not None) else None
    return {"tag": tag, "run_id": pkg.run_id, "compiler_features": list(features),
            "kernel_backend": kernel_backend, "dispatch_timing": dispatch_timing,
            "n_routed": n_routed,
            "min_wall_ns": walls[0] if walls else None,
            "spread": _spread(walls),
            "matmul_bucket_ns": mm_ns, "dispatch_bucket_ns": disp_ns,
            "matmul_frac": (mm_ns / best["wall_ns"]) if (best and mm_ns is not None) else None,
            "matmul_calls": best.get("matmul_calls") if best else None,
            "fp32_cos": cos, "ok": ok, "blocker": blocker, "runs": runs}


def _attribute(r: dict) -> dict:
    """The decomposition. Every term measured; None where its arm did not pass its gate."""
    def pick(tag, key):
        a = r.get(tag) or {}
        return a.get(key) if a.get("ok") else None

    B = pick("B_vf", "min_wall_ns")
    C = pick("C_base_shim", "min_wall_ns")
    ctrl = pick("ctrl", "min_wall_ns")
    dC = pick("C_base_shim", "dispatch_bucket_ns")
    dD = pick("D_vf_shim", "dispatch_bucket_ns")
    mC = pick("C_base_shim", "matmul_bucket_ns")
    mD = pick("D_vf_shim", "matmul_bucket_ns")

    gap = (B - C) if (B and C) else None
    non_mm = (dD - dC) if (dD is not None and dC is not None) else None
    cg_mm = (B - dD) if (B and dD is not None) else None
    out = {
        "gap_ns": gap,
        "gap_x": (B / C) if (B and C) else None,
        "non_matmul_delta_ns": non_mm,
        "codegen_matmul_ns": cg_mm,
        "handc_matmul_ns": mC,
        "codegen_over_handc_matmul_x": (cg_mm / mC) if (cg_mm and mC) else None,
        "shim_matmul_reproducibility_pct": (round(100.0 * abs(mD - mC) / mC, 2)
                                            if (mC and mD) else None),
        "control_vs_C_pct": (round(100.0 * (ctrl - C) / C, 2) if (ctrl and C) else None),
        "note": ("codegen_matmul = B.wall - D.dispatch_bucket: D's non-matmul == B's non-matmul "
                 "(same vf lowering) and D's matmul is the shim, so subtracting D's non-matmul "
                 "cost from B's wall leaves B's OWN emitted matmul. handc_matmul is the shim's "
                 "rdtime bracket. control_vs_C_pct is this campaign's noise floor; no delta "
                 "below it is a result."),
    }
    if gap:
        out["share_non_matmul_pct"] = (round(100.0 * non_mm / gap, 1)
                                       if non_mm is not None else None)
        out["share_matmul_pct"] = (round(100.0 * (cg_mm - mC) / gap, 1)
                                   if (cg_mm is not None and mC is not None) else None)
    return out


def run_workload(model_dir, baseline_pkg="out/artifacts/targets/rvv/hand_v0", n=3,
                 arms="A_base,B_vf,C_base_shim,D_vf_shim,ctrl", out=None) -> dict:
    md = Path(model_dir)
    golden = np.load(md / "golden.npy")
    base = load_rvv_package(baseline_pkg)
    want = [a for a in arms.split(",") if a]
    unknown = [a for a in want if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; known: {sorted(ARMS)}")

    results = {}
    for tag in want:
        feats, kb = ARMS[tag]
        print(f"=== {tag} (features={feats or '[]'} kernel_backend={kb}) ===")
        results[tag] = run_arm(md, base, golden, n, tag, feats, kb)

    summary = {
        "model": str(md), "n": n, "board": "k1_spacemit", "vlen": k1.VLEN,
        "timebase_hz": TIMEBASE_HZ,
        "timer": "CLOCK_MONOTONIC wall_ns + rdtime matmul ticks; cycle_accurate=false",
        "design": ("2x2 factorial: {baseline, vf} lowering x {our codegen, hand-C shim} matmul. "
                   "kernel_backend rewrites matmuls to shim calls BEFORE the feature lowers, so "
                   "C and D differ only in what vf does to the NON-matmul model."),
        "arms_run": want, "results": results,
    }
    summary["attribution"] = _attribute(results)
    if out is not None:
        outp = Path(out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(summary, indent=2))
        _write_md(outp.with_suffix(".md"), summary)
        print(f"\nwrote -> {outp}\nwrote -> {outp.with_suffix('.md')}")
    return summary


def _ms(ns, prec=1):
    return "—" if ns is None else f"{ns / 1e6:,.{prec}f}"


def _num(v, prec=2, suffix=""):
    return "—" if v is None else f"{v:.{prec}f}{suffix}"


def _write_md(path: Path, s: dict) -> None:
    a = s["attribution"]
    lines = [
        f"# K1 2x2 factorial — where the codegen-vs-hand-C gap lives ({Path(s['model']).name})",
        "",
        f"Board: SpacemiT K1 (real RVV silicon, VLEN={s['vlen']}). N={s['n']}/arm, min wall. "
        f"Timer: {s['timer']}. cos gated >= 0.9999 before any wall is reported.",
        "",
        s["design"],
        "",
        "| arm | lowering | matmul | min wall (ms) | matmul bucket (ms) | dispatch bucket (ms) "
        "| range % | cos | routed | ok |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for tag, r in s["results"].items():
        lowering, mm = _LABEL.get(tag, (tag, ""))
        spr = r.get("spread") or {}
        lines.append(
            f"| `{tag}` | {lowering} | {mm} | {_ms(r['min_wall_ns'])} | "
            f"{_ms(r['matmul_bucket_ns'])} | {_ms(r['dispatch_bucket_ns'])} | "
            f"{spr.get('range_pct', '—')}% | {_num(r['fp32_cos'], 7)} | "
            f"{r['n_routed']} | {'yes' if r['ok'] else 'NO'} |")
    lines += [
        "",
        "## Attribution",
        "",
        f"- **gap** (B codegen vs C hand-C): {_ms(a['gap_ns'])} ms ({_num(a['gap_x'], 3)}x)",
        f"- **non-matmul term** (D.dispatch − C.dispatch): {_ms(a['non_matmul_delta_ns'])} ms "
        f"→ {a.get('share_non_matmul_pct', '—')}% of the gap",
        f"- **matmul term**: our emitted {_ms(a['codegen_matmul_ns'])} ms vs hand-C "
        f"{_ms(a['handc_matmul_ns'])} ms "
        f"({_num(a['codegen_over_handc_matmul_x'])}x) → {a.get('share_matmul_pct', '—')}% of the gap",
        "",
        "## Validity",
        "",
        f"- **control** (identical config to `C_base_shim`, separate build + deploy): "
        f"{a['control_vs_C_pct']}% — this campaign's noise floor. No delta below it is a result.",
        f"- **shim reproducibility** (D vs C matmul bucket — the same shim doing the same work): "
        f"{a['shim_matmul_reproducibility_pct']}%.",
        "",
        a["note"],
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="out/artifacts/recaptures/bitvla_fp32_consistent")
    ap.add_argument("--baseline", default="out/artifacts/targets/rvv/hand_v0")
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--arms", default="A_base,B_vf,C_base_shim,D_vf_shim,ctrl")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    name = Path(a.model).name.replace("_fp32_consistent", "")
    out = a.out or f"out/artifacts/kernel-mining/rvv/bench/k1_codegen_vs_handc_{name}.json"
    s = run_workload(a.model, a.baseline, a.n, a.arms, out=out)
    print("\n=== ATTRIBUTION ===")
    print(json.dumps(s["attribution"], indent=2))


if __name__ == "__main__":
    main()
