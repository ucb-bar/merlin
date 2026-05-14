#!/usr/bin/env python3
"""Aggregate benchmark results from multiple runs into a unified report."""

import json
import sys
from pathlib import Path


def load_results(dirs: list[Path]) -> list[dict]:
    all_results = []
    for d in dirs:
        json_path = d / "benchmark_results.json"
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            all_results.extend(data.get("results", []))
    return all_results


def print_table(results: list[dict]):
    backends = sorted({r["backend"] for r in results})
    workloads = []
    seen = set()
    for r in results:
        key = (r["workload"], r["size"])
        if key not in seen:
            workloads.append(key)
            seen.add(key)

    header = f"{'Workload':<20} {'Size':<12}"
    for b in backends:
        header += f" {b:<12}"
    header += " | compile(s)"
    print(header)
    print("-" * len(header))

    for wl, sz in workloads:
        row = f"{wl:<20} {sz:<12}"
        for b in backends:
            match = [
                r
                for r in results
                if r["workload"] == wl and r["size"] == sz and r["backend"] == b
            ]
            if match and match[0].get("subsequent_p50_ms", 0) > 0:
                row += f" {match[0]['subsequent_p50_ms']:<12.3f}"
            elif match and match[0].get("error"):
                row += f" {'FAIL':<12}"
            else:
                row += f" {'-':<12}"
        # Add compile time for first two backends
        for b in backends[:2]:
            match = [
                r
                for r in results
                if r["workload"] == wl and r["size"] == sz and r["backend"] == b
            ]
            if match and match[0].get("compile_time_s", 0) > 0:
                row += f" {match[0]['compile_time_s']:.1f}"
        print(row)

    print()
    print("Summary (avg p50 ms):")
    for b in backends:
        vals = [
            r["subsequent_p50_ms"]
            for r in results
            if r["backend"] == b and r.get("subsequent_p50_ms", 0) > 0
        ]
        if vals:
            print(f"  {b}: {sum(vals)/len(vals):.3f}ms ({len(vals)} benchmarks)")


if __name__ == "__main__":
    dirs = [Path(d) for d in sys.argv[1:]]
    if not dirs:
        print("Usage: python aggregate_results.py <dir1> <dir2> ...")
        sys.exit(1)
    results = load_results(dirs)
    print_table(results)
