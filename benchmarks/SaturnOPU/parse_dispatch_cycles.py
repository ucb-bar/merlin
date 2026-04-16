#!/usr/bin/env python3
"""Parse uartlogs from the FireSim matrix sweep into per-dispatch cycle
totals, optionally joined with the offline
benchmarks/SaturnOPU/model_dispatch_decomposition.csv so we can see which
symbols dominate runtime and how much of that runtime is on the OPU.

Inputs
------
  - one or more uartlog files (typically /tmp/firesim_matrix_logs/*.uartlog)
  - benchmarks/SaturnOPU/model_dispatch_decomposition.csv (optional)

Output
------
  - /tmp/firesim_dispatch_cycles.csv
    columns: model,variant,ordinal,symbol,workgroups,total_cycles,
             is_opu_path,op_kind,compute_pct
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

DN_RE = re.compile(r"\[dn\] o=(?P<ord>\d+) sym=(?P<sym>\S+) wg_count=(?P<wgc>\d+,\d+,\d+)")
DC_RE = re.compile(r"\[dc\] o=(?P<ord>\d+) wg=\d+,\d+,\d+ cyc=(?P<cyc>\d+) ret=-?\d+")
BIN_RE = re.compile(r"bench_model_(?P<model_var>.+?)\.uartlog$")


def parse_uartlog(path: Path):
    """Return (names_by_ord, cycles_by_ord, wg_seen_by_ord)."""
    names = {}
    cycles = defaultdict(int)
    wg_seen = defaultdict(int)
    with path.open() as f:
        for line in f:
            m = DN_RE.search(line)
            if m:
                o = int(m["ord"])
                names[o] = (m["sym"], m["wgc"])
                continue
            m = DC_RE.search(line)
            if m:
                o = int(m["ord"])
                cycles[o] += int(m["cyc"])
                wg_seen[o] += 1
    return names, cycles, wg_seen


def split_model_variant(key: str) -> tuple[str, str]:
    """bench_model_foo_opu -> (foo, opu); also handles mlp_wide_rvv etc."""
    for var in ("opu", "rvv", "rvvtest"):
        if key.endswith("_" + var):
            return key[: -len(var) - 1], var
    return key, ""


def load_decomposition(csv_path: Path) -> dict:
    """Map (model_key, symbol) → row."""
    if not csv_path.exists():
        return {}
    out = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[(row["model_key"], row["symbol"])] = row
    return out


def guess_model_key(model: str) -> str:
    """Collapse variants used in model_dispatch_decomposition.csv.
    mlp_wide → mlp_wide; opu_bench_large_mlp → opu_bench_large_mlp; etc."""
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "logs",
        nargs="+",
        type=Path,
        help="uartlog files to parse (e.g. /tmp/firesim_matrix_logs/*.uartlog)",
    )
    ap.add_argument(
        "-o",
        "--out",
        default=Path("/tmp/firesim_dispatch_cycles.csv"),
        type=Path,
    )
    ap.add_argument(
        "--decomposition",
        default=Path(__file__).resolve().parent / "model_dispatch_decomposition.csv",
        type=Path,
    )
    args = ap.parse_args()

    decomp = load_decomposition(args.decomposition)
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "variant",
                "ordinal",
                "symbol",
                "workgroups_seen",
                "total_cycles",
                "is_opu_path",
                "op_kind",
                "compute_pct",
            ]
        )
        for log in args.logs:
            bn = BIN_RE.search(log.name)
            if bn:
                model_var = bn.group("model_var")
            else:
                model_var = log.stem
            model, variant = split_model_variant(model_var)
            mkey = guess_model_key(model)
            names, cycles, wg_seen = parse_uartlog(log)
            for o in sorted(cycles):
                sym = names.get(o, ("(unknown)", ""))[0]
                row = decomp.get((mkey, sym)) or decomp.get((model, sym))
                is_opu = row.get("opu_path") if row else ""
                op_kind = row.get("op_kind") if row else ""
                comp = row.get("compute_pct") if row else ""
                w.writerow(
                    [
                        model,
                        variant,
                        o,
                        sym,
                        wg_seen[o],
                        cycles[o],
                        is_opu,
                        op_kind,
                        comp,
                    ]
                )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
