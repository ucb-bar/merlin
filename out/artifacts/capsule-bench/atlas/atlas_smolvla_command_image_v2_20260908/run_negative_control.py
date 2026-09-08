#!/usr/bin/env python3
"""Prove GSIM output is RTL readback, not an expected-value substitution."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
sys.path.insert(0, str(REPO / "merlin/python"))
sys.path.insert(0, str(ROOT))

from merlin.targetgen.program_oracle import run_program_verilator_oracle  # noqa: E402
from run_integration import fixture  # noqa: E402


def main() -> int:
    case_dir = ROOT / "cases" / "bf16_movements"
    cb = json.loads((case_dir / "command_buffer.json").read_text())
    inputs, expected = fixture("bf16_movements", cb)
    halt = case_dir / "halt_control.S"
    # ECALL is the Atlas harness halt.  This program deliberately performs no
    # movement or store before the oracle reads the declared output addresses.
    halt.write_text(".text\n.globl atlas_kernel\natlas_kernel:\n  .word 0x00000073\n", encoding="utf-8")
    run_dir = case_dir / "halt_control_run"
    run_dir.mkdir(exist_ok=True)
    result = run_program_verilator_oracle(
        "atlas",
        model_ext="npu_model",
        vsim_dir=Path("/scratch/agustin/tmp/gsim-atlas-core"),
        engine="gsim",
        cb=cb,
        kernel_s=halt,
        inputs=inputs,
        workdir=run_dir,
        timeout=120,
        max_cycles=100,
    )
    comparisons = {}
    for name, reference in expected.items():
        actual = np.asarray(result["outputs"][name], dtype=np.float32)
        comparisons[name] = {
            "nonzero_expected": int(np.count_nonzero(reference)),
            "nonzero_readback": int(np.count_nonzero(actual)),
            "mismatches": int(np.count_nonzero(actual != reference)),
        }
    record = {
        "schema": "atlas_gsim_negative_control_v1",
        "purpose": "An ECALL-only program must not reproduce externally computed expected outputs.",
        "cycles": result["cycles"],
        "oracle": result["oracle"],
        "comparisons": comparisons,
        "control_failed_as_expected": all(v["mismatches"] > 0 for v in comparisons.values()),
    }
    (case_dir / "halt_control_result.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["control_failed_as_expected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
