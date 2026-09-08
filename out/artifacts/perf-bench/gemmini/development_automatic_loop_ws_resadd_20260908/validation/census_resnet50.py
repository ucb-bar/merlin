#!/usr/bin/env python3
"""Compile one prepared model and emit a payload-free Gemmini resadd census."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "compiler"))

from mlir_oot.gemmini_opt import Pipeline, _print  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.input.read_text()
    pipe = Pipeline(source, enable_source_conv=True).run()
    if pipe.declined is not None or pipe.mixed_declined is not None:
        raise SystemExit(f"lowering declined: {pipe.declined or pipe.mixed_declined}")
    placement = pipe.plan.command_buffer["params"]["gemmini_resadd_placement"]
    global_plan = pipe.plan.command_buffer["params"]["global_program_plan"]
    receipt = {
        "schema": "canonical_pt2e_resnet50_gemmini_resadd_census_v1",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "source_op_count": global_plan["source_op_count"],
        "target_sha256": hashlib.sha256(_print(pipe.artifact).encode()).hexdigest(),
        "scheduled_instruction_count": len(pipe.instrs),
        "candidate_count": placement["candidate_count"],
        "selected_count": placement["selected_count"],
        "integer_refused": placement["refused"],
        "float_residual_refused": placement["float_residual_refused"],
        "hardware_semantics": placement["hardware_semantics"],
        "approximation": placement["approximation"],
        "lowering_status": (
            "no_exact_loop_ws_resadd_site_in_canonical_capture"
            if placement["selected_count"] == 0 else "exact_sites_selected"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
