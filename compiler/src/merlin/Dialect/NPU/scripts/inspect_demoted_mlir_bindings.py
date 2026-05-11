#!/usr/bin/env python3
"""Inspect demoted SmolVLA MLIR with IREE's MLIR Python bindings.

The demoted SmolVLA dump embeds hundreds of megabytes of exact model weights in
the trailing dense-resource block. By default this script strips only that
payload block before parsing, so the binding parser can walk the actual op
structure while `extract_mlir_dense_resources.py` remains the byte-accurate
weight payload checker.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from iree.compiler import ir

DENSE_RESOURCE_RE = re.compile(r"dense_resource<([^>]+)>\s*:\s*tensor<([^>]+)>")


def read_parseable_mlir(path: Path, strip_resource_payloads: bool) -> str:
    if not strip_resource_payloads:
        return path.read_text(errors="ignore")

    lines: list[str] = []
    with path.open("r", errors="ignore") as handle:
        for line in handle:
            if line.lstrip().startswith("{-#"):
                break
            lines.append(line)
    return "".join(lines)


def raw_operation(op: Any) -> ir.Operation:
    return getattr(op, "operation", op)


def walk_operations(op: Any) -> Iterable[ir.Operation]:
    raw = raw_operation(op)
    yield raw
    for region in raw.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from walk_operations(child)


def type_strings(op: ir.Operation) -> list[str]:
    types: list[str] = []
    for operand in op.operands:
        types.append(str(operand.type))
    for result in op.results:
        types.append(str(result.type))
    return types


def classify_family(op: ir.Operation) -> str | None:
    name = op.name
    if name == "iree_linalg_ext.attention":
        return "attention"
    if name == "linalg.softmax":
        return "softmax"
    if name in {"linalg.matmul", "linalg.batch_matmul"}:
        return "matmul"
    if name == "math.rsqrt":
        return "rms_norm"
    if name in {"math.tanh", "math.fpowi"}:
        return "gelu_tanh"
    if name == "math.powf":
        return "rope_frequency"
    if name.startswith("npu_kernel."):
        return name.removeprefix("npu_kernel.")
    if name.startswith("npu_schedule."):
        return name.removeprefix("npu_schedule.")

    if name == "arith.truncf":
        rendered = str(op)
        if "f8E4M3FN" in rendered:
            return "requant"
    if name == "arith.extf":
        rendered = str(op)
        if "f8E4M3FN" in rendered and "bf16" in rendered:
            return "matmul"
    return None


def inspect_mlir(path: Path, strip_resource_payloads: bool) -> dict[str, Any]:
    text = read_parseable_mlir(path, strip_resource_payloads)
    context = ir.Context()
    context.allow_unregistered_dialects = True
    module = ir.Module.parse(text, context)

    op_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    f32_typed_ops: Counter[str] = Counter()

    for op in walk_operations(module.operation):
        op_counts[op.name] += 1
        family = classify_family(op)
        if family is not None:
            family_counts[family] += 1

        if any("f32" in type_text for type_text in type_strings(op)):
            f32_typed_ops[op.name] += 1

    resource_refs = DENSE_RESOURCE_RE.findall(text)
    return {
        "mlir": str(path),
        "strip_resource_payloads": strip_resource_payloads,
        "operation_count": sum(op_counts.values()),
        "op_counts": dict(op_counts.most_common()),
        "family_counts": dict(family_counts.most_common()),
        "f32_typed_ops": dict(f32_typed_ops.most_common()),
        "dense_resource_refs_in_parse_text": len(resource_refs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect demoted MLIR structure using IREE MLIR bindings")
    parser.add_argument("mlir", type=Path)
    parser.add_argument(
        "--keep-resource-payloads",
        action="store_true",
        help="Parse the full file, including the large trailing payload block",
    )
    parser.add_argument("--json-out", type=Path, help="Optional JSON output")
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of operation names to print in the text summary",
    )
    args = parser.parse_args()

    report = inspect_mlir(args.mlir, strip_resource_payloads=not args.keep_resource_payloads)

    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("MLIR binding inspection PASSED")
    print(f"mlir: {report['mlir']}")
    print(f"strip_resource_payloads: {report['strip_resource_payloads']}")
    print(f"operation_count: {report['operation_count']}")
    print("dense_resource_refs_in_parse_text: " f"{report['dense_resource_refs_in_parse_text']}")

    print("family_counts:")
    for name, count in report["family_counts"].items():
        print(f"  {count:4d} {name}")

    print("f32_typed_ops:")
    if report["f32_typed_ops"]:
        for name, count in report["f32_typed_ops"].items():
            print(f"  {count:4d} {name}")
    else:
        print("     0")

    print("top_ops:")
    for index, (name, count) in enumerate(report["op_counts"].items()):
        if index >= args.top:
            break
        print(f"  {count:4d} {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
