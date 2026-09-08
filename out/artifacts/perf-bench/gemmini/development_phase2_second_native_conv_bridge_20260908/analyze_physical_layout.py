#!/usr/bin/env python3
"""Emit a payload-free physical-layout graph and plan for a captured model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
sys.path.insert(0, str(ROOT / "compiler"))
sys.path.insert(0, str(REPO / "merlin/python"))

from mlir_oot.frontend.capture_layout import RegionLayoutPolicy, extract_region_layout_graph
from mlir_oot.frontend.parse import parse_module
from mlir_oot.lowering.physical_layout import solve


def policy() -> RegionLayoutPolicy:
    """Capability policy for the public convolution compiler snapshot."""
    return RegionLayoutPolicy(
        layouts=("NCHW", "NHWC"),
        canonical_layout="NCHW",
        preferred_accelerator_layout="NHWC",
        accelerator_ops=frozenset({"convolution_im2col_matmul"}),
        preserving_ops=frozenset({
            "quantize_per_tensor", "dequantize_per_tensor", "minmax",
        }),
        residual_ops=frozenset({"add"}),
        view_ops=frozenset({"view", "reshape"}),
        fixed_layout_ops=frozenset({"max_pool2d", "adaptive_avg_pool2d"}),
        weight_ops=frozenset({"dequantize_per_channel"}),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path, help="provenance-annotated linalg MLIR")
    parser.add_argument("--graph-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()

    capture_bytes = args.capture.read_bytes()
    module = parse_module(capture_bytes.decode())
    graph, census = extract_region_layout_graph(module, policy())
    plan = solve(graph)
    graph_doc = graph.to_dict()
    graph_doc["source"] = {
        "kind": "provenance_annotated_linalg",
        "sha256": hashlib.sha256(capture_bytes).hexdigest(),
        "bytes": len(capture_bytes),
        "payload_included": False,
    }
    report = {
        "schema": "physical_layout_census_v1",
        "status": "structurally_planned_not_runtime_lowered",
        "source": graph_doc["source"],
        "census": census,
        "plan": plan.to_dict(),
        "limitations": [
            "the current backend has not implemented NHWC allocation/index rewrites",
            "pooling remains canonical-layout-only in this bounded policy",
            "rank-changing views fail closed without an affine physical-axis proof",
            "conversion byte counts describe source IR storage, not measured runtime traffic",
        ],
    }
    args.graph_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.graph_out.write_text(json.dumps(graph_doc, indent=2, sort_keys=True) + "\n")
    args.report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["plan"]["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
