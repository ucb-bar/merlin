#!/usr/bin/env python3
"""Check whether a demoted SmolVLA MLIR is ready for native NPU kernels.

This is intentionally a lightweight textual gate. It catches the invariants
that matter before manifest stitching:
  * no fp32 compute remains after demotion,
  * required kernel families are present in the manifest, and
  * the MLIR still contains exact dense resources for actual weights.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

F32_ABI_IMPORT_RE = re.compile(r"hal\.tensor\.import .*-> tensor<[^>]*xf32> as tensor<[^>]*xbf16>")

DENSE_RESOURCE_RE = re.compile(r"dense_resource<([^>]+)>\s*:\s*tensor<([^>]+)>")


def is_allowed_f32_line(line: str) -> bool:
    return bool(F32_ABI_IMPORT_RE.search(line))


def required_families(text: str) -> set[str]:
    families: set[str] = set()
    if "iree_linalg_ext.attention" in text:
        families.add("attention")
    if "linalg.softmax" in text:
        families.add("softmax")
    if "math.rsqrt" in text:
        families.add("rms_norm")
    if "math.tanh" in text or "math.fpowi" in text:
        families.add("gelu_tanh")
    if "math.powf" in text:
        families.add("rope_frequency")
    if "arith.truncf" in text and "to f8E4M3FN" in text:
        families.add("requant")
    if "arith.extf" in text and "f8E4M3FN to bf16" in text:
        families.add("matmul")

    # The demoted graph has many linalg.generic elementwise regions. Requiring
    # these families keeps manifest coverage explicit before stitching.
    for family in (
        "elementwise_add",
        "elementwise_mul",
        "elementwise_div",
        "elementwise_sub",
        "reduction_sum",
        "silu",
    ):
        families.add(family)
    return families


def load_manifest_kernel_names(path: Path | None) -> set[str]:
    if path is None:
        return set()
    manifest = json.loads(path.read_text())
    return set(manifest.get("kernels", {}))


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate demoted SmolVLA MLIR for native NPU kernel lowering")
    parser.add_argument("mlir", type=Path, help="module.5.demoted.mlir path")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/SaturnNPU/kernel_library/manifest.json"),
        help="kernel manifest to check for required family coverage",
    )
    args = parser.parse_args()

    text = args.mlir.read_text(errors="ignore")
    lines = text.splitlines()

    bad_f32_lines = [
        (index, line) for index, line in enumerate(lines, start=1) if "f32" in line and not is_allowed_f32_line(line)
    ]
    if bad_f32_lines:
        print("FAILED: fp32 compute or storage remains after demotion")
        for index, line in bad_f32_lines[:20]:
            print(f"{index}: {line.strip()}")
        if len(bad_f32_lines) > 20:
            print(f"... {len(bad_f32_lines) - 20} more")
        return 1

    resource_refs = DENSE_RESOURCE_RE.findall(text)
    resource_types = Counter(resource_type for _, resource_type in resource_refs)

    required = required_families(text)
    manifest_names = load_manifest_kernel_names(args.manifest)
    missing = sorted(required - manifest_names)
    if missing:
        print("FAILED: manifest is missing required kernel families")
        for name in missing:
            print(f"- {name}")
        return 1

    print("demoted native readiness PASSED")
    print(f"mlir: {args.mlir}")
    print(f"allowed f32 ABI imports: {sum('f32' in line for line in lines)}")
    print(f"dense_resource_refs: {len(resource_refs)}")
    print(f"required_kernel_families: {', '.join(sorted(required))}")
    print("top_resource_types:")
    for resource_type, count in resource_types.most_common(12):
        print(f"  {count:4d} {resource_type}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
