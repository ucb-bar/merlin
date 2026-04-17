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


def classify_opu_kernel(func_asm, opu_status):
    """Classify the specific OPU kernel path used by this dispatch.

    Maps to the optimization journey categories:
    - "opu_encoding_resolver" — iree_uk_opu_matmul via +xopu encoding (best)
    - "opu_fused_qdq"        — iree_uk_opu_matmul_qdq (fused dequant+bias+requant)
    - "opu_runtime_mmt4d"    — iree_uk_mmt4d with runtime OPU detection (OPU_IM2COL)
    - "opu_vectorcontract"   — inline VOPACC via VectorContractCustomKernels
    - "rvv_baseline"         — no OPU, standard RVV
    - "non_compute"          — elementwise/reduction/softmax (not a matmul)
    """
    if opu_status == "fused_qdq":
        return "opu_fused_qdq"
    if opu_status == "vopacc":
        # Distinguish: encoding resolver vs vector contract custom kernels.
        # Encoding resolver calls iree_uk_opu_matmul which gets inlined
        # and shows OPMVINBCAST (insn r 87, 6, 89) + VFETCH (insn r 87, 6, 93)
        # + VOPACC (insn r 87, 2, 81).
        # VectorContractCustomKernels also produce VOPACC but the pattern is
        # different — they use llvm.riscv.opu.* intrinsics which produce
        # the same opcodes. Distinguish by checking for OPMVINBCAST (the
        # encoding resolver's ukernel always initializes matrix registers).
        if ".insn r 87, 6, 89" in func_asm:  # OPMVINBCAST
            return "opu_encoding_resolver"
        return "opu_vectorcontract"
    if opu_status == "mmt4d_ukernel":
        return "opu_runtime_mmt4d"
    return "rvv_baseline"


def extract_matmul_dims(dispatch_name):
    """Extract M, N, K dimensions from dispatch name like 'matmul_like_64x512x128'.

    Returns (M, N, K) or None if not a matmul dispatch.
    Also handles batch matmuls like 'batch_matmul_8x64x64x128' → (64, 64, 128).
    """
    # batch_matmul_BxMxNxK
    m = re.search(r"batch_matmul_(\d+)x(\d+)x(\d+)x(\d+)", dispatch_name)
    if m:
        return int(m.group(2)), int(m.group(3)), int(m.group(4))
    # matmul_like_MxNxK or matmul_MxNxK
    m = re.search(r"matm(?:ul|vec)(?:_like)?_(\d+)x(\d+)x(\d+)", dispatch_name)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    # Batch: BxMxNxK pattern (e.g., 3x64x128x128)
    m = re.search(r"(\d+)x(\d+)x(\d+)x(\d+)_i8", dispatch_name)
    if m:
        return int(m.group(2)), int(m.group(3)), int(m.group(4))
    return None


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

        opu_kernel = classify_opu_kernel(func_asm, opu_status)
        dims = extract_matmul_dims(func_name)
        ops = 2 * dims[0] * dims[1] * dims[2] if dims else 0

        # Detect OPU tile configuration from assembly patterns.
        # OPMVINBCAST count reveals sub-tiling: 1=single 16×16, 4=2×2 sub-tiling.
        # Packed tensor shape from the ukernel args reveals M0/N0.
        if opmvinbcast_count == 4:
            tile_config = "32x32_2x2sub"  # Uses m0+m1+m2+m3
        elif opmvinbcast_count == 1:
            tile_config = "16x16_single"  # Uses m0 only
        elif opmvinbcast_count == 0 and vopacc_count > 0:
            tile_config = "vectorcontract"  # Inline VOPACC (no ukernel)
        elif vopacc_count == 0 and opu_status != "none":
            tile_config = "mmt4d_runtime"  # Runtime-detected, tile unknown
        else:
            tile_config = "none"

        results.append(
            {
                "model": model_name,
                "dispatch": short_name,
                "op_type": op_type,
                "opu_status": opu_status,
                "opu_kernel": opu_kernel,
                "tile_config": tile_config,
                "M": dims[0] if dims else 0,
                "N": dims[1] if dims else 0,
                "K": dims[2] if dims else 0,
                "ops": ops,
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

    # OPU kernel decomposition (maps to optimization journey categories)
    kernels = {}
    for r in compute:
        k = r.get("opu_kernel", "rvv_baseline")
        if k not in kernels:
            kernels[k] = {"count": 0, "lines": 0, "ops": 0}
        kernels[k]["count"] += 1
        kernels[k]["lines"] += r["lines"]
        kernels[k]["ops"] += r.get("ops", 0)

    print("\n  OPU Kernel Decomposition:")
    print(f"  {'Kernel Type':<25} {'Count':>6} {'Lines':>8} {'Ops (M)':>10}")
    print(f"  {'-' * 55}")
    for k in sorted(kernels, key=lambda x: -kernels[x]["ops"]):
        d = kernels[k]
        mops = d["ops"] / 1e6 if d["ops"] > 0 else 0
        print(f"  {k:<25} {d['count']:>6} {d['lines']:>8} {mops:>9.1f}M")

    # OPU matmul dispatches with M, N, K
    matmul_dispatches = [r for r in compute if r.get("M", 0) > 0]
    if matmul_dispatches:
        print("\n  Matmul Dispatch Detail:")
        print(f"  {'Dispatch':<40} {'M':>5} {'N':>5} {'K':>6} {'Ops(M)':>8} {'Tile':>18} {'Kernel':<25}")
        print(f"  {'-' * 110}")
        for r in sorted(matmul_dispatches, key=lambda x: -x.get("ops", 0)):
            mops = r["ops"] / 1e6 if r["ops"] > 0 else 0
            print(
                f"  {r['dispatch'][:40]:<40} {r['M']:>5} {r['N']:>5} {r['K']:>6}"
                f" {mops:>7.1f}M {r.get('tile_config', 'none'):>18}"
                f" {r.get('opu_kernel', 'unknown'):<25}"
            )

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
                "opu_kernel",
                "tile_config",
                "M",
                "N",
                "K",
                "ops",
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
