#!/usr/bin/env python3
"""Per-dispatch performance breakdown from a FireSim uartlog.

Parses `[dc] o=N wg=X,Y,Z cyc=C` workgroup events AND the final
`CYC, <ord>, <sym>, <total_cycles>, <wg_count>` summary emitted by
`iree_merlin_dump_cycles()` to produce:

  1. Per-dispatch cycle table (sym, total_cycles, wg_count, % of total).
  2. Bucket totals by dispatch KIND (matmul / conv / elementwise /
     encoding / initializer / memcpy / unknown).
  3. Top-K hot dispatches by total cycles.

Usage:
    ./merlin perf-decompose <uartlog>
    ./merlin perf-decompose <uartlog> --topk 20
    ./merlin perf-decompose <uartlog> --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

# Output line format from iree_merlin_dump_cycles (final summary):
#   CYC, <ord>, <sym>, <cycles>, <wg_count>
_CYC_RE = re.compile(r"^CYC,\s*(\d+),\s*([^,]+?),\s*(\d+),\s*(\d+)\s*$")

# Workgroup-level per-call profiling line (only when MERLIN_PROFILE_CYCLES=1):
#   [dc] o=N wg=X,Y,Z cyc=C ret=R
_DC_RE = re.compile(r"^\[dc\] o=(\d+) wg=(\d+),(\d+),(\d+) cyc=(\d+) ret=(-?\d+)")

# Dispatch-launch line carries the symbol per ordinal:
#   [dn] o=N sym=<sym> wg_count=X,Y,Z
_DN_RE = re.compile(r"^\[dn\] o=(\d+) sym=(\S+) wg_count=")


def _kind(sym: str) -> str:
    s = sym.lower()
    if "matmul" in s:
        return "matmul"
    if "_encoding_" in s or "encode_" in s:
        return "encoding"
    if "_initializer_" in s:
        return "initializer"
    if "slow_memcpy" in s or "memcpy" in s:
        return "memcpy"
    if "elementwise" in s or "generic_" in s:
        return "elementwise"
    if "conv" in s:
        return "conv"
    return "other"


def parse_uartlog(path: Path) -> dict[int, dict]:
    """Returns dict keyed by ordinal: {sym, cycles, wg_count}."""
    rows: dict[int, dict] = {}
    text = path.read_text(errors="ignore")
    for line in text.splitlines():
        m = _CYC_RE.match(line)
        if m:
            ord_ = int(m.group(1))
            rows[ord_] = {
                "sym": m.group(2).strip(),
                "cycles": int(m.group(3)),
                "wg_count": int(m.group(4)),
            }
            continue
        m = _DN_RE.match(line)
        if m:
            ord_ = int(m.group(1))
            # Capture symbol if CYC summary didn't run / partial run.
            rows.setdefault(ord_, {"sym": m.group(2), "cycles": 0, "wg_count": 0})
            if "sym" not in rows[ord_] or not rows[ord_]["sym"]:
                rows[ord_]["sym"] = m.group(2)
            continue
        m = _DC_RE.match(line)
        if m:
            ord_ = int(m.group(1))
            cyc = int(m.group(5))
            rows.setdefault(ord_, {"sym": f"(ord={ord_})", "cycles": 0, "wg_count": 0})
            rows[ord_]["cycles"] += cyc
            rows[ord_]["wg_count"] += 1
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("uartlog", type=Path, help="FireSim uartlog file")
    p.add_argument("--topk", type=int, default=20, help="Print top-K hot dispatches (default 20)")
    p.add_argument("--csv", type=Path, default=None, help="Also write a CSV summary")
    args = p.parse_args()

    if not args.uartlog.exists():
        raise SystemExit(f"not found: {args.uartlog}")

    rows = parse_uartlog(args.uartlog)
    if not rows:
        raise SystemExit("no [dc]/CYC/[dn] data found. Re-run with MERLIN_PROFILE_CYCLES=1 " "and the final CYC dump.")

    total = sum(r["cycles"] for r in rows.values())
    if total == 0:
        print(
            "WARNING: all cyc=0. The build doesn't have MERLIN_PROFILE_CYCLES "
            "enabled. Rebuild with -DMERLIN_PROFILE_CYCLES=1 on the merlin_iree "
            "target. Continuing with workgroup counts only."
        )

    # === per-dispatch table ===
    print(f"# Per-dispatch breakdown (total={total:,} cycles)")
    print(f"{'ord':>4}  {'wg':>6}  {'cycles':>15}  {'pct':>6}  kind          sym")
    print("-" * 100)
    sorted_rows = sorted(rows.items(), key=lambda kv: -kv[1]["cycles"])
    for ord_, r in sorted_rows[: args.topk]:
        pct = (100.0 * r["cycles"] / total) if total else 0.0
        print(f"{ord_:>4}  {r['wg_count']:>6}  {r['cycles']:>15,}  " f"{pct:>5.2f}%  {_kind(r['sym']):<12}  {r['sym']}")

    # === by kind ===
    by_kind: dict[str, dict] = {}
    for r in rows.values():
        k = _kind(r["sym"])
        b = by_kind.setdefault(k, {"cycles": 0, "wg_count": 0, "n_dispatches": 0})
        b["cycles"] += r["cycles"]
        b["wg_count"] += r["wg_count"]
        b["n_dispatches"] += 1

    print()
    print("# Aggregated by dispatch kind")
    print(f"{'kind':<14} {'#disp':>6} {'#wg':>8} {'cycles':>15} {'pct':>6}")
    print("-" * 60)
    for kind in sorted(by_kind, key=lambda k: -by_kind[k]["cycles"]):
        b = by_kind[kind]
        pct = (100.0 * b["cycles"] / total) if total else 0.0
        print(f"{kind:<14} {b['n_dispatches']:>6} {b['wg_count']:>8} " f"{b['cycles']:>15,} {pct:>5.2f}%")

    if args.csv:
        with args.csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ord", "wg_count", "cycles", "pct", "kind", "sym"])
            for ord_, r in sorted_rows:
                pct = (100.0 * r["cycles"] / total) if total else 0.0
                w.writerow([ord_, r["wg_count"], r["cycles"], f"{pct:.4f}", _kind(r["sym"]), r["sym"]])
        print(f"\nCSV written: {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
