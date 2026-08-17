#!/usr/bin/env python
"""K1 WHOLE-MODEL PER-OP PROFILE: attribute model wall time to the ops the model executes.

WHY. The only whole-model instrumentation this repo had was a TWO-WAY split — a matmul bucket
(``-DMERLIN_DISPATCH_TIMING`` inside the routed GEMM shim) vs "everything else". Measured on the
K1, matmul is 1.3-6 % of a model once the kernel is fast (dispatch_breakdown_measured.json), so
94-97 % of model time had never been attributed to anything. This driver produces the missing
attribution and ranks what is left on the table by MEASURED milliseconds.

METHOD (default-OFF; the un-instrumented build is byte-identical).
``merlin.llvmlower.op_profile`` interleaves a one-instruction-cost ``@merlin_prof_mark(id)`` call
between the top-level ops of ``func.func @forward`` (plus a sentinel before ``func.return``); the
shim ``runtime/c/merlin_op_prof.c`` samples ``rdtime`` at each mark and credits the elapsed ticks
to the previous op's id. ``build_k1_binary(..., op_profile=True)`` splices this in AFTER any
kernel-backend rewrite and links the shim; the harness prints ``PROF <id> <ticks> <hits>``.
``run_on_k1`` joins those against the id->op table (``prov.*`` provenance) it emitted at build.

Each op record carries the cross-compiler JOIN KEY (:func:`op_profile.join_key`): ``prov.fqn`` when
present (the deepest ``nn.Module`` path that aligns a Merlin region with the SAME layer in an
ExecuTorch/GGUF/ONNX export — see ``baselines/contract.py`` / ``_et_export.py``), else
``prov.region_id``, else the MLIR op name. So this breakdown is a framework-agnostic
graph-node -> IR -> emitted-asm -> measured-cycles chain, not a Merlin-only scheme.

PERTURBATION GUARD (the profiler must not move what it measures). Every model is ALSO built and run
WITHOUT instrumentation (byte-identical baseline path). We report the profiled-vs-unprofiled median
wall delta and FAIL the perturbation check if it exceeds the board noise floor (--noise, default
1.9 %). A breakdown whose own instrumentation moved the wall past the floor is reported as
``perturbation_ok=false`` and must not be trusted.

FAIL-CLOSED. cos is gated (>= --cos, default 0.9999) against golden.npy every run; a run that does
not gate is ``not_run`` and its wall is never recorded.

Run (board serialized internally — do NOT wrap in board_lock):
  MERLIN_COMPILE_TIMEOUT_S=3600 .venv/bin/python build_tools/scripts/k1_op_profile.py \
      --model out/artifacts/recaptures/bitvla_fp32_consistent -n 3
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from merlin.common.paths import artifacts_dir
from merlin.llvmlower.op_profile import join_key
from merlin.rvvgen import k1
from merlin.rvvgen.registry import load_rvv_package
from merlin.runtime.backends import zephyr_model as zm

TIMEBASE_HZ = k1.K1_TIMEBASE_HZ  # rdtime tick rate (24 MHz)
NS_PER_TICK = 1e9 / TIMEBASE_HZ

# The default K1 RVV pipeline (llvmlower.pipeline.build_rvv_pipeline) vectorizes ONLY the
# contraction ops (linalg.matmul / linalg.batch_matmul); every other family goes through
# convert-linalg-to-loops -> scalar. So "is this op vectorized?" is answered by its family, and
# the sum of non-contraction ticks IS the scalar work left on the table.
_VECTORIZED_FAMILIES = {"contraction"}


def _ticks_to_ms(ticks: float) -> float:
    return ticks * NS_PER_TICK / 1e6


def _rvv_coverage(binary: Path) -> dict | None:
    """Disassemble the built binary and report the STATIC vector coverage of the compiler-emitted
    ``forward`` symbol (cross-check for the family-based ``vectorized`` intent flag). This is a
    static instruction count, not a dynamic one — but a low coverage on ``forward`` proves the
    emitted compute is scalar-dominated regardless of what family a op claims. Best-effort: returns
    None if objdump/the binary is unreadable (honest, never a silent 'fully vectorized')."""
    try:
        from merlin.baselines.rvv_audit import audit_binary
        rep = audit_binary(binary)
        fwd = rep.by_symbol.get("forward")
        if fwd is None:
            return None
        return {"symbol": "forward", "vector_insns": fwd.vector,
                "scalar_compute_insns": fwd.scalar_compute,
                "static_vector_coverage": fwd.coverage,
                "note": ("STATIC instruction coverage of the emitted `forward` (one inlined "
                         "function). Low coverage => the compute is scalar-dominated (addressing, "
                         "memrefCopy, tails) even for ops whose family is 'contraction'.")}
    except Exception:  # noqa: BLE001
        return None


def _median(xs: list[int]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return float(s[n // 2]) if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def run_profiled(model_dir: Path, pkg, golden: np.ndarray, n: int, work_root: Path,
                 kernel_backend: str | None = None, cos_th: float = 0.9999,
                 rel_th: float = 1e-2) -> dict:
    """Build+run the instrumented binary n times; gate cos; accumulate per-op ticks.

    The per-op ticks are SUMMED across the gated runs then averaged, so a single op faster than
    one rdtime tick (~41.7 ns) still accumulates signal across runs. Returns the per-op table with
    averaged ticks/ms plus the run-level walls and profiler-coverage metrics.
    """
    walls: list[int] = []
    time_ticks: list[int] = []
    prof_totals: list[int] = []
    per_op_ticks: dict[int, int] = defaultdict(int)
    per_op_hits: dict[int, int] = defaultdict(int)
    table: list[dict] | None = None
    gated = 0
    blocker = None
    for i in range(n):
        work = work_root / f"on_{i}"
        try:
            res = k1.run_on_k1(model_dir, work, pkg, timeout=1800, op_profile=True,
                               kernel_backend=kernel_backend)
            g = zm._gate(res["prefix"], {"fp32": golden})
            cos = g["fp32_cos"]
            rel = g.get("fp32_rel")
            # Fail-closed per-ELEMENT gate: cos (aggregate) AND max per-element relative error. cos
            # alone was measured to accept a kernel 1209% wrong per-element (rel~12), and a mis-fused
            # transpose is a silent numerical bug — so require BOTH. rel_th (default 1e-2) sits well
            # below that failure while passing legitimate baselines (openvla rel 1e-6; the bitvla fp32
            # recapture rel 3e-3), so it only rejects a genuinely wrong emitted kernel.
            if cos is None or cos < cos_th or (rel is not None and rel > rel_th):
                blocker = f"gate failed: cos={cos} rel={rel}"
                print(f"  [on {i}] NOT_RUN cos={cos} rel={rel}")
                continue
            m = res["metrics"]
            walls.append(m["wall_ns"])
            time_ticks.append(m.get("time_ticks", 0))
            prof_totals.append(m.get("prof_total_ticks", 0))
            tbl = res.get("op_profile") or []
            if table is None:
                table = tbl
            for rec in tbl:
                per_op_ticks[rec["id"]] += rec.get("ticks", 0)
                per_op_hits[rec["id"]] = rec.get("hits", 0)
            gated += 1
            print(f"  [on {i}] wall_ns={m['wall_ns']} cos={cos:.7f} "
                  f"prof_cov={m.get('prof_total_ticks',0)/max(m.get('time_ticks',1),1):.3f} "
                  f"marks={m.get('prof_marks')}")
        except Exception as e:  # noqa: BLE001
            blocker = f"{type(e).__name__}: {str(e)[:300]}"
            print(f"  [on {i}] BLOCKED — {blocker}")
            break
    if table is None or gated == 0:
        return {"ok": False, "blocker": blocker or "no gated run", "n_gated": 0}
    for rec in table:
        rec["ticks_avg"] = per_op_ticks[rec["id"]] / gated
        rec["ms_avg"] = _ticks_to_ms(rec["ticks_avg"])
        rec["hits"] = per_op_hits[rec["id"]]
        rec["join_key"] = join_key(rec)
        rec["vectorized"] = (rec.get("family") in _VECTORIZED_FAMILIES)
    return {"ok": True, "n_gated": gated, "table": table,
            "walls": walls, "median_wall_ns": _median(walls),
            "median_time_ticks": _median(time_ticks),
            "median_prof_total_ticks": _median(prof_totals),
            "rvv_coverage": _rvv_coverage(work_root / "on_0" / "v" / "merlin_k1"),
            "blocker": None}


def run_unprofiled(model_dir: Path, pkg, golden: np.ndarray, n: int, work_root: Path,
                   kernel_backend: str | None = None, cos_th: float = 0.9999,
                   rel_th: float = 1e-2) -> dict:
    """Build+run the byte-identical UN-instrumented binary n times (the perturbation control)."""
    walls: list[int] = []
    gated = 0
    blocker = None
    for i in range(n):
        work = work_root / f"off_{i}"
        try:
            res = k1.run_on_k1(model_dir, work, pkg, timeout=1800, op_profile=False,
                               kernel_backend=kernel_backend)
            g = zm._gate(res["prefix"], {"fp32": golden})
            cos = g["fp32_cos"]
            rel = g.get("fp32_rel")
            if cos is None or cos < cos_th or (rel is not None and rel > rel_th):
                blocker = f"gate failed: cos={cos} rel={rel}"
                print(f"  [off {i}] NOT_RUN cos={cos} rel={rel}")
                continue
            walls.append(res["metrics"]["wall_ns"])
            gated += 1
            print(f"  [off {i}] wall_ns={res['metrics']['wall_ns']} cos={cos:.7f} rel={rel}")
        except Exception as e:  # noqa: BLE001
            blocker = f"{type(e).__name__}: {str(e)[:300]}"
            print(f"  [off {i}] BLOCKED — {blocker}")
            break
    return {"ok": gated > 0, "n_gated": gated, "walls": walls,
            "median_wall_ns": _median(walls), "blocker": blocker}


def _rollup(table: list[dict], keyfn, label: str) -> list[dict]:
    agg: dict[str, dict] = defaultdict(lambda: {"ms": 0.0, "n_ops": 0, "hits": 0, "vectorized": None})
    for rec in table:
        k = keyfn(rec)
        a = agg[k]
        a["ms"] += rec["ms_avg"]
        a["n_ops"] += 1
        a["hits"] += rec["hits"]
        v = rec.get("vectorized")
        a["vectorized"] = v if a["vectorized"] is None else (a["vectorized"] and v)
    rows = [{label: k, **v} for k, v in agg.items()]
    rows.sort(key=lambda r: r["ms"], reverse=True)
    return rows


def build_breakdown(model_dir: Path, prof: dict, unprof: dict, noise_pct: float) -> dict:
    table = prof["table"]
    total_ms = sum(r["ms_avg"] for r in table)

    by_family = _rollup(table, lambda r: r.get("family") or "(none)", "family")
    by_op = _rollup(table, lambda r: r.get("mlir_op") or "(none)", "mlir_op")
    by_key = _rollup(table, lambda r: r["join_key"], "join_key")
    by_aten = _rollup(table, lambda r: r.get("aten") or "(none)", "aten")

    # scalar (non-contraction) vs vectorized (contraction) split — the headline "left on table".
    scalar_ms = sum(r["ms_avg"] for r in table if not r["vectorized"])
    vector_ms = sum(r["ms_avg"] for r in table if r["vectorized"])

    # perturbation check: profiled vs unprofiled median wall.
    pon = prof.get("median_wall_ns")
    poff = unprof.get("median_wall_ns")
    pert = None
    if pon and poff:
        delta_pct = 100.0 * (pon - poff) / poff
        # ONE-SIDED test: instrumentation can only ADD cost, so a perturbation problem is the
        # profiled wall being SLOWER than the control by MORE than the noise floor. A delta at or
        # below the floor (incl. negative — profiled faster) is noise, not a measurable perturbation.
        pert = {"profiled_median_wall_ns": pon, "unprofiled_median_wall_ns": poff,
                "delta_pct": round(delta_pct, 3), "noise_floor_pct": noise_pct,
                "perturbation_ok": delta_pct <= noise_pct,
                "note": ("delta = profiled/unprofiled-1. Instrumentation can only add cost, so the "
                         "one-sided guard fails only when profiled is SLOWER than the un-instrumented "
                         "control by more than the board noise floor. |delta|<=floor (or negative) "
                         "means the profiler did not measurably move the wall.")}

    # top ops by ms (individual dispatches).
    top_ops = sorted(table, key=lambda r: r["ms_avg"], reverse=True)[:40]
    top_ops = [{k: r.get(k) for k in ("id", "mlir_op", "family", "op", "aten", "region_id",
                                      "fqn", "join_key", "result_type", "elems", "hits",
                                      "ms_avg", "vectorized")} for r in top_ops]

    return {
        "total_attributed_ms": round(total_ms, 3),
        "profiler_coverage": (round(prof["median_prof_total_ticks"] / prof["median_time_ticks"], 4)
                              if prof.get("median_time_ticks") else None),
        "vectorized_flag_meaning": ("`vectorized`/`vectorized_ms` reflect the pipeline's INTENT: the "
                                    "K1 RVV schedule vectorizes only the contraction family; every "
                                    "other family lowers to scalar loops. It is NOT proof the "
                                    "emitted asm is vector — see `rvv_coverage` for the measured "
                                    "static coverage of `forward`."),
        "rvv_coverage": prof.get("rvv_coverage"),
        "contraction_intended_ms": round(vector_ms, 3), "scalar_ms": round(scalar_ms, 3),
        "scalar_frac": round(scalar_ms / total_ms, 4) if total_ms else None,
        "perturbation": pert,
        "by_family": by_family, "by_aten": by_aten, "by_mlir_op": by_op,
        "top_ops": top_ops,
        "by_join_key_top": by_key[:40],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="out/artifacts/recaptures/bitvla_fp32_consistent")
    ap.add_argument("--baseline", default="out/artifacts/targets/rvv/hand_v0")
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--cos", type=float, default=0.9999)
    ap.add_argument("--rel", type=float, default=1e-2,
                    help="fail-closed per-ELEMENT gate: max |pred-ref| / max|ref| must be below this "
                         "(cos alone once accepted a kernel 1209%% wrong per-element). Default 1e-2 "
                         "passes legit fp32/quant baselines and rejects a mis-fused/wrong kernel.")
    ap.add_argument("--noise", type=float, default=1.9, help="board noise floor %% for perturbation")
    ap.add_argument("--kernel-backend", default=None,
                    help="route matmuls to a fast expert kernel (xnnpack/openblas/ours) so the "
                         "per-op breakdown isolates the NON-matmul 94%% when the GEMM is fast")
    ap.add_argument("--features", default=None,
                    help="comma-separated default-off compiler features to enable on BOTH the "
                         "profiled and control builds (e.g. fuse_transpose_b). OMITTED -> the "
                         "package's own compiler_features, i.e. the lowering it was certified with. "
                         "Pass an empty string for the frozen baseline lowering. Run once with and "
                         "once without a feature to get its before/after whole-model profile.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    md = Path(a.model)
    base = load_rvv_package(a.baseline)
    # The reference has to match the DATAPATH, not just the bundle. An int8 bundle's golden.npy is
    # weight-only (int8 weights dequantized into an f32 contraction); a W8A8 build also quantizes the
    # activations and is graded against golden_w8a8.npy. Gating a W8A8 run on the weight-only golden
    # fails cos for a correct build, which reads as "the model regressed" rather than "the wrong
    # reference was loaded". Fail closed when the package is int8 and no W8A8 reference exists.
    golden_path = md / "golden.npy"
    if base.is_int8:
        w8a8 = md / "golden_w8a8.npy"
        if not w8a8.is_file():
            raise SystemExit(f"{a.baseline} is an int8 package but {md.name} ships no "
                             f"golden_w8a8.npy — recapture it rather than grading W8A8 output "
                             f"against the weight-only golden")
        golden_path = w8a8
    print(f"[golden] {golden_path.name} (package is_int8={base.is_int8})")
    golden = np.load(golden_path)
    # A package's compiler_features ARE part of what it is: the champion int8 package carries
    # `accumulator_resident_wholemodel_vf`, and replacing that with [] compiles the BASELINE lowering
    # while still calling itself that package. Measured on deepjscc W8A8, the discarded feature is the
    # difference between gating and cos=0.9176 -- which reads as a model regression rather than as the
    # profiler having built something else. So an omitted --features means "whatever this package was
    # certified with"; an explicitly empty one still means the frozen baseline.
    feats = ([f.strip() for f in a.features.split(",") if f.strip()] if a.features is not None
             else list(base.compiler_features or []))
    print(f"[features] {feats or '<baseline>'} "
          f"({'from the package' if a.features is None else 'from --features'})")
    # distinct run_ids so the two builds get distinct /tmp deploy paths on the shared board
    # (tr_ prefix per the shared-board tag convention; backend + feature-tag suffix keeps native vs
    # routed and baseline vs feature-on builds apart so before/after never collide on /tmp).
    bk = (a.kernel_backend or "native")
    ftag = "base" if not feats else "_".join(sorted(feats))[:40]
    pkg_on = replace(base, run_id=f"tr_opon_{bk}_{ftag}", compiler_features=feats)
    pkg_off = replace(base, run_id=f"tr_opoff_{bk}_{ftag}", compiler_features=feats)

    work_root = Path("/tmp") / f"tr_opprofile_{md.name}_{bk}_{ftag}"
    work_root.mkdir(parents=True, exist_ok=True)

    kb = a.kernel_backend
    print(f"=== {md.name}: profiled (instrumented, kernel_backend={kb}) x{a.n} ===")
    prof = run_profiled(md, pkg_on, golden, a.n, work_root, kernel_backend=kb,
                        cos_th=a.cos, rel_th=a.rel)
    print(f"=== {md.name}: unprofiled control x{a.n} ===")
    unprof = run_unprofiled(md, pkg_off, golden, a.n, work_root, kernel_backend=kb,
                            cos_th=a.cos, rel_th=a.rel)

    summary: dict = {
        "model": str(md), "board": "k1_spacemit", "vlen": k1.VLEN, "kernel_backend": kb,
        "compiler_features": feats,
        "timer": "rdtime per-op marks + CLOCK_MONOTONIC wall_ns; cycle_accurate=false",
        "timebase_hz": TIMEBASE_HZ, "n": a.n, "cos_threshold": a.cos, "rel_threshold": a.rel,
        "method": ("Per-op rdtime marks interleaved between the top-level ops of @forward "
                   "(default-off; un-instrumented build byte-identical). Ticks credited to the "
                   "previous op; keyed on prov.fqn||region_id (the cross-compiler join key). "
                   "Perturbation-gated vs an un-instrumented control within the board noise floor."),
        "profiled": {k: v for k, v in prof.items() if k != "table"},
        "unprofiled": unprof,
    }
    if prof.get("ok"):
        summary["breakdown"] = build_breakdown(md, prof, unprof, a.noise)
        summary["op_table"] = prof["table"]
    else:
        summary["breakdown"] = None
        print(f"!! profiled run did not gate: {prof.get('blocker')}")

    out = Path(a.out) if a.out else (artifacts_dir() / "measurements" / "k1_spacemit"
                                     / md.name / f"op_profile_{ftag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote -> {out}")

    b = summary.get("breakdown")
    if b:
        print("\n=== TOP FAMILIES (measured ms) ===")
        for row in b["by_family"][:12]:
            print(f"  {row['family']:<16} {row['ms']:8.2f} ms  n_ops={row['n_ops']:<5} "
                  f"vec={row['vectorized']}")
        p = b["perturbation"]
        if p:
            print(f"\nperturbation: profiled={p['profiled_median_wall_ns']/1e6:.2f}ms "
                  f"unprofiled={p['unprofiled_median_wall_ns']/1e6:.2f}ms "
                  f"delta={p['delta_pct']}% ok={p['perturbation_ok']}")
        cov = b.get("rvv_coverage")
        covs = f"{cov['static_vector_coverage']:.3f}" if cov and cov.get("static_vector_coverage") else "n/a"
        print(f"scalar={b['scalar_ms']:.1f}ms ({b['scalar_frac']*100:.1f}%) "
              f"contraction(intended-vec)={b['contraction_intended_ms']:.1f}ms  "
              f"profiler_cov={b['profiler_coverage']}  forward_static_rvv_cov={covs}")


if __name__ == "__main__":
    main()
