#!/usr/bin/env python3
"""Produce inspectable full-model fusion IR and equivalence receipts without execution.

This is a structural compiler stress test, not an accelerator benchmark. Contiguous region sizes
are chosen by the operator to exercise the emitter; profitability selection belongs to Phase 2.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from merlin.common.paths import artifacts_dir
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.xdsl_dialects.lowering import global_plan, global_plan_emission, outlined_plan_emission
from merlin.xdsl_dialects.lowering.dispatch_program import lower_model_to_dispatch_program


def run(inputs: list[Path], destination: Path, region_nodes: int) -> dict:
    if region_nodes < 2:
        raise ValueError("region_nodes must be at least two")
    if not inputs or len({path.resolve() for path in inputs}) != len(inputs):
        raise ValueError("provide a nonempty list of distinct captured model paths")
    destination = destination.resolve()
    destination.relative_to(artifacts_dir().resolve())
    destination.mkdir(parents=True, exist_ok=False)
    implementations = {
        module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for module in (global_plan, global_plan_emission, outlined_plan_emission)
    }
    rows = []
    for index, path in enumerate(inputs):
        started = time.monotonic()
        source = path.read_bytes()
        module = parse_mlir_text(source.decode())
        outlined, graph = lower_model_to_dispatch_program(module, prune=False)
        groups = [tuple(range(i, min(i + region_nodes, len(graph.nodes))))
                  for i in range(0, len(graph.nodes), region_nodes) if len(graph.nodes) - i > 1]
        plan = outlined_plan_emission.plan_dispatch_fusion(
            graph, groups, placement="compiler_function",
            representation=lambda name: global_plan.ValueRepresentation(
                "tensor_ssa", "logical", graph.buffers[name].dtype))
        emitter = outlined_plan_emission.OutlinedGlobalPlanEmitter(outlined)
        emission = global_plan_emission.emit_global_plan(graph, plan, emitter)
        output = destination / f"model_{index:03d}"
        output.mkdir()
        (output / "before.mlir").write_text(str(outlined.module))
        (output / "after.mlir").write_text(str(emitter.module))
        documents = {
            "logical_graph.json": graph.to_dict(), "plan.json": plan.to_dict(),
            "emitted_graph.json": emission.dispatch.to_dict(),
            "emission_receipt.json": emission.receipt(), "proof.json": emitter.proof,
        }
        for name, document in documents.items():
            (output / name).write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")
        row = dict(emitter.proof)
        row.update({"source": str(path.resolve()), "source_sha256": hashlib.sha256(source).hexdigest(),
                    "artifact_directory": str(output), "compile_and_verify_seconds": time.monotonic() - started})
        rows.append(row)
    report = {
        "schema": "full_model_fusion_compilation_proof_v1", "models": rows,
        "compiler_sources": implementations, "xdsl_version": importlib.metadata.version("xdsl"),
        "region_nodes": region_nodes,
        "selection": "contiguous grouping stress test; no profitability claim",
        "simulator_invocations": 0, "full_model_executions": 0,
        "remaining": ["candidate target-codegen adoption", "target materialization accounting",
                      "mechanism-equivalent probe calibration", "global profitability selection"],
    }
    (destination / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-mlir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--region-nodes", type=int, required=True)
    options = parser.parse_args()
    report = run(options.input_mlir, options.output_dir, options.region_nodes)
    print(json.dumps({"models": [{key: row[key] for key in (
        "source", "logical_nodes", "emitted_nodes", "logical_dispatches", "emitted_dispatches",
        "computation", "compile_and_verify_seconds")} for row in report["models"]],
        "simulator_invocations": report["simulator_invocations"]}, indent=2))
