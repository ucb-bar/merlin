#!/usr/bin/env python3
"""Prepare and place a full model without paying LLVM emission cost."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
BUNDLE = Path(os.environ.get("MERLIN_CENSUS_BUNDLE", HERE.parent)).resolve()
REPOSITORY = BUNDLE.parents[4]
for path in (REPOSITORY / "merlin/python", BUNDLE / "support", BUNDLE / "compiler"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mlir_oot.frontend import linalg_reader, parse as mlir_parse  # noqa: E402
from mlir_oot.frontend.integer_prepare import prepare_int8_text  # noqa: E402
from mlir_oot.lowering.source_conv_model_lane import (           # noqa: E402
    build,
    place_source_convolutions,
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def census(input_path: Path) -> dict:
    source = input_path.read_text()
    started = time.monotonic()
    normalized, preparation = prepare_int8_text(source)
    prepared_s = time.monotonic() - started
    started = time.monotonic()
    module = mlir_parse.parse_module(normalized)
    workload = linalg_reader.read(module)
    place_source_convolutions(module, workload)
    plan = build(module, workload, normalized_source_sha256=sha256_text(normalized))
    planned_s = time.monotonic() - started

    cb = plan.command_buffer
    params = cb["params"]
    program = params["global_program_plan"]
    maximal = params["maximal_accelerator_regions"]
    tasks = program["tasks"]
    spills = program.get("host_tensor_spills", [])
    host_rows = maximal["residual_host_operations"]
    epilogues = params["target_neutral_quantized_epilogues"]
    residuals = params["target_neutral_residual_epilogues"]
    return {
        "schema": "gemmini_friendly_resnet50_structural_census_v1",
        "input": {
            "path": str(input_path),
            "source_sha256": sha256_text(source),
            "normalized_sha256": sha256_text(normalized),
            "source_bytes": len(source.encode()),
            "normalized_bytes": len(normalized.encode()),
        },
        "bundle": str(BUNDLE),
        "timing_seconds": {"prepare": prepared_s, "parse_and_plan": planned_s},
        "preparation": preparation,
        "command_buffer": {
            "command_count": len(cb["commands"]),
            "opcode_histogram": dict(sorted(Counter(
                row["opcode"] for row in cb["commands"]).items())),
            "task_count": len(tasks),
            "task_kind_histogram": dict(sorted(Counter(
                row["kind"] for row in tasks).items())),
            "accelerator_task_count": maximal["accelerator_task_count"],
            "host_task_count": sum(row["kind"] == "host" for row in tasks),
            "host_segment_count": len(params["host_lane_segments"]),
            "maximal_accelerator_region_count": maximal["maximal_region_count"],
        },
        "host_residual": {
            "source_operation_count": maximal["residual_host_operation_count"],
            "operation_histogram": dict(sorted(Counter(
                row["op"] for row in host_rows).items())),
            "spill_tensor_count": len(spills),
            "spill_bytes": sum(row["bytes"] for row in spills),
        },
        "fusion": {
            "native_narrow_epilogue_count": epilogues["selected_count"],
            "native_narrow_source_ops_absorbed": epilogues["source_ops_absorbed"],
            "residual_epilogue_formed_count": residuals["formed_site_count"],
            "residual_epilogue_selected_count": residuals["selected_site_count"],
        },
        "interpretation": {
            "conv_capture_form": "model2MLIR im2col plus rank-2 matmul",
            "direct_loop_conv_tasks": sum(row["kind"] == "convolution" for row in tasks),
            "accelerator_contraction_tasks": sum(row["kind"] == "contraction" for row in tasks),
            "full_llvm_emission_performed": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    result = census(args.input.resolve())
    if args.baseline:
        baseline = json.loads(args.baseline.read_text())
        old = baseline["host_residual"]["source_operation_count"]
        new = result["host_residual"]["source_operation_count"]
        result["delta_vs_baseline"] = {
            "baseline": str(args.baseline),
            "host_source_operations": new - old,
            "host_source_operations_percent": 100.0 * (new - old) / old,
            "normalized_bytes": (
                result["input"]["normalized_bytes"] - baseline["input"]["normalized_bytes"]),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "command_buffer": result["command_buffer"],
        "host_residual": result["host_residual"],
        "fusion": result["fusion"],
        "delta_vs_baseline": result.get("delta_vs_baseline"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
