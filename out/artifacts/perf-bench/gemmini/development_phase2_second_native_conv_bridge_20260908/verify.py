#!/usr/bin/env python3
"""Payload-free verifier for the q545 cost guard artifact."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise SystemExit(message)


def census(path: Path) -> tuple[int, int, list[str]]:
    data = json.loads(path.read_text())["params"]["convolution_lowering"]
    return (int(data["native_loop_conv_count"]), int(data["fallback_count"]),
            [row["reason"] for row in data["selections"]])


def verify_physical_layout() -> None:
    sys.path.insert(0, str(ROOT / "compiler"))
    from mlir_oot.lowering.physical_layout import LayoutGraph, solve

    graph_path = ROOT / "validation/canonical_resnet50/physical_layout_graph.json"
    report_path = ROOT / "validation/canonical_resnet50/physical_layout_census.json"
    graph = LayoutGraph.from_dict(json.loads(graph_path.read_text()))
    expected = json.loads(report_path.read_text())
    actual = solve(graph).to_dict()
    require(actual == expected["plan"], "physical-layout plan is not reproducible")
    summary = actual["summary"]
    require(summary["op_counts"]["accelerator"] == 53,
            "physical-layout graph does not cover all 53 convolutions")
    require(summary["op_counts"]["residual"] == 16,
            "physical-layout graph does not enforce all residual merges")
    require(summary["op_counts"]["layout_preserving"] == 151,
            "rank-4 quant/dequant/ReLU propagation census changed")
    require(summary["conversions"] == 4,
            "canonical conversion-boundary census changed")
    require(expected["status"] == "structurally_planned_not_runtime_lowered",
            "structural layout plan was mislabeled as a runtime result")


def main() -> int:
    status = json.loads((ROOT / "STATUS.json").read_text())
    require(status["status"] == "hardware_tested_rejected_perf_guarded",
            "unexpected artifact status")
    prod_target = ROOT / "validation/canonical_resnet50/production_target.mlir"
    diag_target = ROOT / "validation/canonical_resnet50/diagnostic_target.mlir"
    require(sha(prod_target) == status["target_identity"]["q535_target_sha256"],
            "production output is not byte-identical to q535")
    require(sha(diag_target) == status["target_identity"]["q545_target_sha256"],
            "diagnostic output is not byte-identical to q545")
    pn, pf, pr = census(
        ROOT / "validation/canonical_resnet50/production_command_buffer.json")
    dn, df, dr = census(
        ROOT / "validation/canonical_resnet50/diagnostic_command_buffer.json")
    require((pn, pf) == (0, 53), "production convolution census changed")
    require(pr.count("hardware_cost_guard_transposed_nchw_underfills_systolic_rows") == 1,
            "production hardware guard is absent")
    require((dn, df) == (1, 52), "diagnostic convolution census changed")
    require(dr.count("selected_compute_only_full_width_mvout") == 1,
            "diagnostic opt-in no longer selects the exact mechanism")
    verify_physical_layout()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "compiler"), str(REPO / "merlin/python")]
    )
    subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT,
                   env=env, check=True)
    print("q545 cost guard and 53-conv structural layout plan verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
