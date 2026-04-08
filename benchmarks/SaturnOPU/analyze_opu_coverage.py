#!/usr/bin/env python3
"""Analyze OPU VOPACC coverage per model from compiled assembly files.

Parses IREE-compiled RISC-V assembly to determine:
- Dispatch types (conv, matmul, elementwise, reduction, etc.)
- OPU acceleration status per dispatch (VOPACC, fused QDQ, mmt4d, none)
- Coverage percentages and compute weight estimates

Usage:
    python analyze_opu_coverage.py <assembly_file> [--model-name NAME]
    python analyze_opu_coverage.py --all  # analyze all pre-built models
"""

import argparse
import csv
import os
import re
import sys


def classify_dispatch(name):
    """Classify a dispatch by its operator type based on the function name."""
    name_lower = name.lower()
    if "conv_" in name_lower:
        return "conv"
    if "matmul_like" in name_lower or "matmul" in name_lower:
        return "matmul"
    if "matvec_like" in name_lower or "matvec" in name_lower:
        return "matvec"
    if "reduction" in name_lower:
        return "reduction"
    if "elementwise" in name_lower:
        return "elementwise"
    if "slow_memcpy" in name_lower or "memcpy" in name_lower:
        return "memcpy"
    if "encoding" in name_lower or "encode" in name_lower:
        return "encoding"
    if "softmax" in name_lower:
        return "softmax"
    if "transpose" in name_lower:
        return "transpose"
    return "other"


def classify_opu_status(func_asm):
    """Determine OPU acceleration status from assembly text."""
    if "call\tiree_uk_opu_matmul_qdq" in func_asm:
        return "fused_qdq"
    if ".insn r 87, 2, 81" in func_asm:
        return "vopacc"
    if "call\tiree_uk_mmt4d" in func_asm:
        return "mmt4d_ukernel"
    return "none"


def analyze_assembly(asm_path, model_name="unknown"):
    """Analyze an assembly file and return dispatch-level metrics."""
    with open(asm_path) as f:
        asm_text = f.read()

    # Find all .type...@function lines
    type_lines = re.findall(r"\.type\s+(\S+),@function", asm_text)

    # Filter to dispatch functions (exclude internal helpers)
    dispatch_funcs = [f for f in type_lines if "dispatch" in f or "encoding" in f or "initializer" in f]

    results = []
    for func_name in dispatch_funcs:
        # Extract function body (between .type and .size)
        pattern = re.compile(
            rf"\.type\s+{re.escape(func_name)},@function.*?" rf"\.size\s+{re.escape(func_name)}",
            re.DOTALL,
        )
        match = pattern.search(asm_text)
        if not match:
            continue

        func_asm = match.group(0)
        line_count = func_asm.count("\n")

        # Classify
        short_name = re.sub(r".*dispatch_\d+_", "", func_name)
        short_name = re.sub(r".*encoding_\d+_", "encoding_", short_name)
        op_type = classify_dispatch(func_name)
        opu_status = classify_opu_status(func_asm)

        # Count OPU instructions
        vopacc_count = func_asm.count(".insn r 87, 2, 81")
        opmvinbcast_count = func_asm.count(".insn r 87, 6, 89")
        vmv_vr_count = func_asm.count(".insn r 87, 6, 93")

        results.append(
            {
                "model": model_name,
                "dispatch": short_name,
                "op_type": op_type,
                "opu_status": opu_status,
                "lines": line_count,
                "vopacc": vopacc_count,
                "opmvinbcast": opmvinbcast_count,
                "vmv_vr": vmv_vr_count,
            }
        )

    return results


def print_summary(results, model_name):
    """Print a formatted summary of OPU coverage."""
    compute = [r for r in results if r["op_type"] != "encoding"]
    opu = [r for r in compute if r["opu_status"] != "none"]

    total_compute = len(compute)
    total_opu = len(opu)
    pct = (total_opu * 100 // total_compute) if total_compute > 0 else 0

    # Weighted by lines (proxy for compute)
    total_lines = sum(r["lines"] for r in compute)
    opu_lines = sum(r["lines"] for r in opu)
    wpct = (opu_lines * 100 // total_lines) if total_lines > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"  {model_name}")
    print(f"{'=' * 60}")
    print(f"  Compute dispatches: {total_compute}")
    print(f"  OPU dispatches:     {total_opu} ({pct}%)")
    print(f"  Weighted OPU:       {wpct}% (by assembly size)")
    print()

    # Breakdown by op type
    types = {}
    for r in compute:
        t = r["op_type"]
        if t not in types:
            types[t] = {"total": 0, "opu": 0, "lines": 0, "opu_lines": 0}
        types[t]["total"] += 1
        types[t]["lines"] += r["lines"]
        if r["opu_status"] != "none":
            types[t]["opu"] += 1
            types[t]["opu_lines"] += r["lines"]

    print(f"  {'Type':<15} {'Total':>6} {'OPU':>6} {'%':>6} {'Lines':>8} {'OPU Lines':>10}")
    print(f"  {'-' * 55}")
    for t in sorted(types, key=lambda x: -types[x]["lines"]):
        d = types[t]
        p = (d["opu"] * 100 // d["total"]) if d["total"] > 0 else 0
        print(f"  {t:<15} {d['total']:>6} {d['opu']:>6} {p:>5}% {d['lines']:>8} {d['opu_lines']:>10}")

    # Show non-OPU dispatches
    non_opu = [r for r in compute if r["opu_status"] == "none"]
    if non_opu:
        print(f"\n  Non-OPU dispatches ({len(non_opu)}):")
        for r in sorted(non_opu, key=lambda x: -x["lines"])[:10]:
            print(f"    {r['dispatch']:<50} {r['lines']:>5} lines")

    return {
        "model": model_name,
        "total_dispatches": total_compute,
        "opu_dispatches": total_opu,
        "opu_pct": pct,
        "weighted_opu_pct": wpct,
        "types": types,
    }


def save_csv(all_results, output_path):
    """Save per-dispatch results to CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "dispatch",
                "op_type",
                "opu_status",
                "lines",
                "vopacc",
                "opmvinbcast",
                "vmv_vr",
            ],
        )
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)


def find_assembly(directory):
    """Find the linked assembly file in a dump directory."""
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith(".s") and "linked" in f:
                return os.path.join(root, f)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze OPU coverage")
    parser.add_argument("assembly", nargs="?", help="Path to assembly .s file")
    parser.add_argument("--model-name", default="unknown", help="Model name")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyze all pre-built models",
    )
    parser.add_argument(
        "--csv",
        default="benchmarks/SaturnOPU/opu_coverage_results.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    all_results = []
    summaries = []

    if args.all:
        models = {
            "MLP": "/tmp/final_verify_mlp/",
            "DroNet": "/tmp/final_all_dronet/",
            "YOLOv8": "/tmp/yolo_opu_verify/",
        }
        for name, directory in models.items():
            asm = find_assembly(directory)
            if asm:
                results = analyze_assembly(asm, name)
                all_results.extend(results)
                summaries.append(print_summary(results, name))
            else:
                print(f"  {name}: No assembly found in {directory}")
    elif args.assembly:
        results = analyze_assembly(args.assembly, args.model_name)
        all_results.extend(results)
        summaries.append(print_summary(results, args.model_name))
    else:
        parser.print_help()
        sys.exit(1)

    if all_results:
        save_csv(all_results, args.csv)
        print(f"\nResults saved to {args.csv}")
