"""`./merlin perf-decompose` — per-dispatch performance decomposition from uartlog."""

from pathlib import Path


def setup_parser(parser):
    parser.add_argument("uartlog", type=Path, help="FireSim uartlog file")
    parser.add_argument("--topk", type=int, default=20, help="Print top-K hot dispatches (default 20)")
    parser.add_argument("--csv", type=Path, default=None, help="Also write a CSV summary")


def main(args) -> int:
    from perf import decompose as perf_decompose

    if not args.uartlog.exists():
        print(f"not found: {args.uartlog}")
        return 1

    rows = perf_decompose.parse_uartlog(args.uartlog)
    if not rows:
        print("no [dc]/CYC/[dn] data found. Re-run with MERLIN_PROFILE_CYCLES=1.")
        return 1

    total = sum(r["cycles"] for r in rows.values())
    if total == 0:
        print(
            "WARNING: all cyc=0. Rebuild Zephyr with the CMakeLists change "
            "adding -DMERLIN_PROFILE_CYCLES=1. Showing wg counts only."
        )

    print(f"# Per-dispatch breakdown (total={total:,} cycles)")
    print(f"{'ord':>4}  {'wg':>6}  {'cycles':>15}  {'pct':>6}  kind          sym")
    print("-" * 100)
    sorted_rows = sorted(rows.items(), key=lambda kv: -kv[1]["cycles"])
    for ord_, r in sorted_rows[: args.topk]:
        pct = (100.0 * r["cycles"] / total) if total else 0.0
        print(
            f"{ord_:>4}  {r['wg_count']:>6}  {r['cycles']:>15,}  "
            f"{pct:>5.2f}%  {perf_decompose._kind(r['sym']):<12}  {r['sym']}"
        )

    by_kind = {}
    for r in rows.values():
        k = perf_decompose._kind(r["sym"])
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
        import csv

        with args.csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ord", "wg_count", "cycles", "pct", "kind", "sym"])
            for ord_, r in sorted_rows:
                pct = (100.0 * r["cycles"] / total) if total else 0.0
                w.writerow([ord_, r["wg_count"], r["cycles"], f"{pct:.4f}", perf_decompose._kind(r["sym"]), r["sym"]])
        print(f"\nCSV written: {args.csv}")
    return 0
