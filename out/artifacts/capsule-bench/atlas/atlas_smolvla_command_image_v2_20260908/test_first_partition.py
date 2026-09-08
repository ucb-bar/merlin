"""Compile-time checks for the first concrete SmolVLA device partition."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PARTITION = ROOT / "partitions/first_addmm_matmul_0"


def test_first_addmm_partition_contains_bias_and_fits_imem() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "build_first_partition.py")],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    receipt = json.loads((PARTITION / "compile_receipt.json").read_text())
    assert receipt["capture_regions"] == ["matmul_0", "add_3"]
    assert receipt["instruction_words"] == 8037
    assert receipt["fits_imem"] is True
    assert receipt["command_opcodes"] == [
        "RES_PACK",
        "MATMUL_RESIDENT",
        "COMMIT",
        "EVICT",
    ]
    command_buffer = json.loads((PARTITION / "command_buffer.json").read_text())
    commit = command_buffer["commands"][2]
    assert commit["attributes"]["epilogue"] == ["bias_add"]
    assert commit["attributes"]["bias"] == "B"
    assert command_buffer["kernel_abi"]["outputs"] == ["Y0"]
    manifest = json.loads((PARTITION / "partition_manifest.json").read_text())
    assert manifest["selection"]["fused_capture_regions"] == ["matmul_0", "add_3"]
    assert [row["capture_value"] for row in manifest["capture_boundary"]["inputs"]] == [
        "%929",
        "%931",
        "%8",
    ]
    assert manifest["capture_boundary"]["outputs"][0]["capture_value"] == "%937"
    assert manifest["image"]["instruction_words"] == 8037
