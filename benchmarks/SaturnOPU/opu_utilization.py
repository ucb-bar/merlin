#!/usr/bin/env python3
"""OPU cycle-utilization analyzer — offline, no FireSim needed.

Computes, per model, the fraction of total run cycles spent executing
Saturn OPU custom instructions (VOPACC, OPMVINBCAST, VMV_RV, VMV_VR).

Inputs (already on disk, no re-runs required):
  * /tmp/verify_all/<profile>/*.s
      Per-dispatch assembly dumps from iree-compile
      --iree-hal-dump-executable-intermediates-to=. These carry the
      inlined ukernel bodies with .insn r 87 opcodes.
  * /tmp/sweep_cycles.csv
      Per-dispatch measured cycles from the profile-phase FireSim runs
      (columns: model, variant, ordinal, symbol, total_cycles, wg_count).
  * /tmp/sweep_iters.csv
      Per-model avg iteration cycles (the denominator for the final
      per-model util%). avg is already a 5-iter average.

Method:
  1. Parse each dispatch function body from the .s dump; count OPU
     opcodes (VOPACC=funct3=2/funct7=81, BCAST=6/89, FETCH=6/85, VMV_VR=6/93).
  2. Extract M, N, K shape from the dispatch symbol.
  3. Estimate K-reduction trip count from shape and the (heuristic) inner-
     tile K = 1 (OPU does elementwise K reduction): trip_factor = K.
     For batch_matmul_BxMxNxK add a factor of B.
  4. Join with sweep_cycles.csv by (model, symbol) to get wg_count and
     total_cycles. If the symbol is stale (redesigned model), fall back
     to shape-derived wg_count (ceil(M/32) * ceil(N/32)).
  5. dynamic_OPU_ops = static × trip_factor × wg_count.
  6. Assume 1 cycle / OPU op (single-issue, back-to-back VOPACCs).
     opu_cyc_est = dynamic_OPU_ops.
  7. util_% = 100 * Σ opu_cyc_est / avg_total_cycles_per_iter.

Caveat: the 1-cyc assumption is an upper-bound approximation; real
issue rate can stall on register dependencies. Report this clearly.

Usage:
  uv run benchmarks/SaturnOPU/opu_utilization.py [--dump-dir /tmp/verify_all]
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

DUMP_DIR_DEFAULT = Path("/tmp/verify_all")
SWEEP_CYCLES = Path("/tmp/sweep_cycles.csv")
SWEEP_ITERS = Path("/tmp/sweep_iters.csv")
OUT_DISPATCH = Path(__file__).parent / "opu_utilization_per_dispatch.csv"
OUT_MODEL = Path(__file__).parent / "opu_utilization_per_model.csv"

# Saturn OPU instruction patterns. Assembler emits decimal 87 (not 0x57).
OPU_OPCODES = {
    "vopacc": re.compile(r"\.insn\s+r\s+87\s*,\s*2\s*,\s*81\b"),
    "bcast": re.compile(r"\.insn\s+r\s+87\s*,\s*6\s*,\s*89\b"),
    "fetch": re.compile(r"\.insn\s+r\s+87\s*,\s*6\s*,\s*85\b"),
    "vmv_vr": re.compile(r"\.insn\s+r\s+87\s*,\s*6\s*,\s*93\b"),
}
FUNC_BOUND = re.compile(r"^\s*\.type\s+([^,\s]+),@function", re.M)
MATMUL_RE = re.compile(r"matmul(?:_like)?_(\d+)x(\d+)x(\d+)")
BATCH_MATMUL_RE = re.compile(r"(?:batch_)?matmul(?:_like)?_(\d+)x(\d+)x(\d+)x(\d+)")
MATVEC_RE = re.compile(r"matvec(?:_like)?_(\d+)x(\d+)")

# Dump directory -> (model_key, variant) as it appears in sweep_cycles.csv.
# variant is always "opu" because the sweep ran clean-phase OPU binaries.
DUMP_TO_MODEL: dict[str, tuple[str, str]] = {
    "mlp_wide_OPU": ("mlp_wide", "opu"),
    "mlp_OPU": ("mlp", "opu"),
    "large_mlp_OPU": ("opu_bench_large_mlp", "opu"),
    "vit_small_OPU": ("opu_bench_vit_small", "opu"),
    "vit_OPU_LLM": ("opu_bench_vit", "opu"),
    "hybrid_OPU_LLM": ("opu_bench_hybrid", "opu"),
    "convnet_OPU_IM2COL": ("opu_bench_convnet", "opu"),
    "dronet_OPU_IM2COL": ("dronet", "opu"),
    "yolov8_OPU_IM2COL": ("yolov8_nano", "opu"),
}


def parse_shape(symbol: str) -> dict[str, int]:
    """Return {M,N,K,B} extracted from a dispatch symbol name."""
    m = BATCH_MATMUL_RE.search(symbol)
    if m:
        b, mm, nn, kk = map(int, m.groups())
        return {"B": b, "M": mm, "N": nn, "K": kk}
    m = MATMUL_RE.search(symbol)
    if m:
        mm, nn, kk = map(int, m.groups())
        return {"B": 1, "M": mm, "N": nn, "K": kk}
    m = MATVEC_RE.search(symbol)
    if m:
        mm, kk = map(int, m.groups())
        return {"B": 1, "M": mm, "N": 1, "K": kk}
    return {"B": 0, "M": 0, "N": 0, "K": 0}


def count_opu_in_function(body: str) -> dict[str, int]:
    return {k: len(r.findall(body)) for k, r in OPU_OPCODES.items()}


def parse_dispatch_dump(asm_path: Path) -> dict[str, dict[str, int]]:
    """Return {symbol: {vopacc,bcast,fetch,vmv_vr}} for each function in a .s."""
    text = asm_path.read_text()
    bounds = [(m.start(), m.group(1)) for m in FUNC_BOUND.finditer(text)]
    bounds.append((len(text), None))
    out: dict[str, dict[str, int]] = {}
    for i, (start, name) in enumerate(bounds[:-1]):
        body = text[start : bounds[i + 1][0]]
        counts = count_opu_in_function(body)
        if sum(counts.values()) > 0:
            out[name] = counts
    return out


def load_sweep_cycles() -> dict[tuple[str, str], dict]:
    """(model_key, symbol) -> {total_cycles, wg_count} from the sweep CSV."""
    out = {}
    if not SWEEP_CYCLES.exists():
        return out
    for r in csv.DictReader(SWEEP_CYCLES.open()):
        if r.get("variant") != "opu":
            continue
        try:
            tc = int(r["total_cycles"])
            wg = int(r["wg_count"])
        except (KeyError, ValueError):
            continue
        out[(r["model"], r["symbol"])] = {"total_cycles": tc, "wg_count": wg}
    return out


def load_sweep_iters() -> dict[tuple[str, str], int]:
    """(model_key, variant) -> avg cycles/iter (int)."""
    out = {}
    if not SWEEP_ITERS.exists():
        return out
    # Keep the LAST numeric row per (model, variant) — fresh runs at the end.
    for r in csv.DictReader(SWEEP_ITERS.open()):
        avg = (r.get("avg") or "").strip()
        if not avg.isdigit():
            continue
        out[(r["model"], r["variant"])] = int(avg)
    return out


def infer_wg_count(shape: dict[str, int], wg_tile_m: int = 32, wg_tile_n: int = 32) -> int:
    """Fallback wg estimator when sweep_cycles lacks this symbol (stale data)."""
    M, N, B = shape["M"], shape["N"], shape["B"]
    if M == 0 or N == 0:
        return 1
    return B * math.ceil(M / wg_tile_m) * math.ceil(N / wg_tile_n)


def analyze(dump_dir: Path):
    sweep_cycles = load_sweep_cycles()
    sweep_iters = load_sweep_iters()

    per_dispatch_rows = []
    per_model: dict[str, dict] = {}

    for dump_name, (model_key, variant) in DUMP_TO_MODEL.items():
        asm_files = list((dump_dir / dump_name).glob("*.s"))
        if not asm_files:
            continue
        funcs = parse_dispatch_dump(asm_files[0])
        # Aggregate per model
        mstats = per_model.setdefault(
            model_key,
            {"opu_cyc_est": 0, "dispatches": 0, "matched": 0, "stale": 0},
        )

        for symbol, opc in funcs.items():
            shape = parse_shape(symbol)
            static_opu = sum(opc.values())
            key = (model_key, symbol)
            measured = sweep_cycles.get(key)
            if measured is not None:
                total_cyc_dispatch = measured["total_cycles"]
                source = "measured"
                mstats["matched"] += 1
            else:
                total_cyc_dispatch = None
                source = "inferred"
                mstats["stale"] += 1

            # Dynamic OPU instruction count comes from first principles:
            # every matmul MxNxKxB requires floor(M*N*K*B / 256) VOPACCs
            # (each VOPACC executes a 16x16 outer product = 256 MACs).
            # Pre/epilogue BCAST+VMV_VR+FETCH scale with output-tile count
            # (M*N / 256) but are 1-2 orders of magnitude smaller than
            # VOPACCs for typical matmul shapes, so we fold them in as
            # rough overhead via the static_outer ratio.
            MNKB = shape["M"] * shape["N"] * shape["K"] * shape["B"]
            if MNKB > 0:
                # VOPACCs: one per 16x16 MAC block. Sub-tile register width=16.
                dynamic_vopacc = MNKB // 256
                # Outer-ops (BCAST+VMV_VR+FETCH) scale with output tiles,
                # not with K. Assume one set per 16x16 output tile.
                output_tiles = max(1, (shape["M"] * shape["N"] * shape["B"]) // 256)
                ratio_outer = (opc["bcast"] + opc["vmv_vr"] + opc["fetch"]) / max(1, opc["vopacc"])
                dynamic_outer = int(dynamic_vopacc * ratio_outer) if opc["vopacc"] else output_tiles
                dynamic_opu = dynamic_vopacc + dynamic_outer
            else:
                dynamic_vopacc = 0
                dynamic_opu = static_opu  # non-matmul: take static as dynamic
            opu_cyc_est = dynamic_opu  # 1 cyc/op assumption

            mstats["opu_cyc_est"] += opu_cyc_est
            mstats["dispatches"] += 1

            per_dispatch_rows.append(
                {
                    "model": model_key,
                    "variant": variant,
                    "symbol": symbol,
                    "M": shape["M"],
                    "N": shape["N"],
                    "K": shape["K"],
                    "B": shape["B"],
                    "static_vopacc": opc["vopacc"],
                    "static_bcast": opc["bcast"],
                    "static_fetch": opc["fetch"],
                    "static_vmv_vr": opc["vmv_vr"],
                    "static_opu_total": static_opu,
                    "dynamic_vopacc": dynamic_vopacc,
                    "dynamic_opu_total": dynamic_opu,
                    "cycle_source": source,
                    "opu_cyc_est_1cyc": opu_cyc_est,
                    "total_cycles_dispatch": total_cyc_dispatch or "",
                }
            )

    # Emit per-dispatch CSV
    with OUT_DISPATCH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_dispatch_rows[0].keys()))
        w.writeheader()
        w.writerows(per_dispatch_rows)

    # Per-model CSV + console table
    print(
        f"{'model':28s}  {'iter_cycles':>14s}  {'opu_cyc_est':>14s}  " f"{'util%':>7s}  {'matched':>7s}  {'stale':>7s}"
    )
    print("-" * 90)
    model_rows = []
    for model_key, s in sorted(per_model.items()):
        iter_cyc = sweep_iters.get((model_key, "opu"))
        if iter_cyc:
            util = 100.0 * s["opu_cyc_est"] / iter_cyc
            util_s = f"{util:6.2f}%"
        else:
            util = None
            util_s = f"{'n/a':>7s}"
        print(
            f"{model_key:28s}  {iter_cyc or 0:>14,}  {s['opu_cyc_est']:>14,}  "
            f"{util_s:>7s}  {s['matched']:>7d}  {s['stale']:>7d}"
        )
        model_rows.append(
            {
                "model": model_key,
                "iter_cycles": iter_cyc or "",
                "opu_cyc_est_1cyc": s["opu_cyc_est"],
                "opu_util_pct": f"{util:.4f}" if util is not None else "",
                "n_dispatches": s["dispatches"],
                "n_matched": s["matched"],
                "n_stale": s["stale"],
            }
        )
    print("-" * 90)
    print(
        "(util% assumes 1 cyc per OPU insn — single-issue upper bound; "
        "memory-bound dispatches will show a low number.)"
    )

    with OUT_MODEL.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(model_rows[0].keys()))
        w.writeheader()
        w.writerows(model_rows)

    print(f"\nPer-dispatch CSV: {OUT_DISPATCH}")
    print(f"Per-model CSV:    {OUT_MODEL}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--dump-dir",
        default=str(DUMP_DIR_DEFAULT),
        help="Directory containing <profile>/*.s dumps (default: /tmp/verify_all)",
    )
    args = p.parse_args()
    analyze(Path(args.dump_dir))


if __name__ == "__main__":
    main()
