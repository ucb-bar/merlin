#!/usr/bin/env python3
"""Verify the pitched-NCHW legality census and the isolated compiler regression tests."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SECOND_NATIVE = ROOT.parent / "development_phase2_second_native_conv_bridge_20260908"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    receipt = json.loads((ROOT / "legality_receipt.json").read_text())
    require(receipt["status"] == "compiler_legality_proved_performance_disabled",
            "legality disposition changed")
    scope = receipt["scope"]
    require(scope == {"model": "torchvision_resnet50_imagenet1k_v2_w8a8",
                      "source_convolutions": 53, "semantically_representable": 53,
                      "automatically_selected": 0, "hardware_promoted": 0},
            "legality/profitability scope changed")
    require(sum(row["convolutions"] for row in receipt["pitch_census"]) == 53,
            "pitch census is incomplete")
    guard = receipt["profitability_guard"]
    require(guard["q545_cycles"] - guard["q535_cycles"] == guard["delta_cycles"],
            "hardware performance delta changed")
    production = json.loads(
        (SECOND_NATIVE / "validation/canonical_resnet50/production_command_buffer.json").read_text())
    diagnostic = json.loads(
        (SECOND_NATIVE / "validation/canonical_resnet50/diagnostic_command_buffer.json").read_text())
    prod_census = production["params"]["convolution_lowering"]
    diag_census = diagnostic["params"]["convolution_lowering"]
    require((prod_census["native_loop_conv_count"], prod_census["fallback_count"]) == (0, 53),
            "production cost guard changed")
    require((diag_census["native_loop_conv_count"], diag_census["fallback_count"]) == (1, 52),
            "diagnostic selection changed")

    # Check the exact RTL authority when Jack's clean Gemmini checkout is available. The compiler
    # tests remain self-contained when the external checkout is not mounted.
    rtl = Path("/scratch/jack/chipyard/generators/gemmini/src/main/scala/gemmini/LoopConv.scala")
    if rtl.exists():
        digest = hashlib.sha256(rtl.read_bytes()).hexdigest()
        require(digest == receipt["rtl_authority"]["loop_conv_scala_sha256"],
                "LoopConv RTL authority changed")
        text = rtl.read_text()
        require("ich * in_col_dim * in_row_dim +& irow*in_col_dim +& icol" in text,
                "transposed input address formula changed")
        require(text.count("in_col_dim") == 6,
                "in_col_dim gained an unreviewed RTL use")

    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_pitched_nchw_legality.py"],
        cwd=ROOT, check=True)
    print("Pitched-NCHW legality gate passed: 53/53 representable; auto-selection remains disabled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
