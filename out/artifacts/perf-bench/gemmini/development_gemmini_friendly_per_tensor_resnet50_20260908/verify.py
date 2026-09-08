#!/usr/bin/env python3
"""Offline integrity and claim-boundary verifier for this artifact."""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def uncompressed_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with gzip.open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def main() -> int:
    receipt = json.loads((ROOT / "validation/full_compile_receipt.json").read_text())
    cb_path = ROOT / "validation" / receipt["command_buffer"]["path"]
    target_path = ROOT / "validation" / receipt["target_artifact"]["compressed_path"]
    cb = json.loads(cb_path.read_text())
    program = cb["params"]["global_program_plan"]
    maximal = cb["params"]["maximal_accelerator_regions"]
    target_hash, target_size = uncompressed_sha256(target_path)
    assert sha256(cb_path) == receipt["command_buffer"]["sha256"]
    assert sha256(target_path) == receipt["target_artifact"]["compressed_sha256"]
    assert target_hash == receipt["target_artifact"]["uncompressed_sha256"]
    assert target_size == receipt["target_artifact"]["uncompressed_bytes"]
    assert len(cb["commands"]) == 162
    assert Counter(row["opcode"] for row in cb["commands"]) == {
        "RES_PACK": 54, "MATMUL_RESIDENT": 54, "COMMIT": 54}
    assert Counter(row["kind"] for row in program["tasks"]) == {
        "host": 55, "contraction": 54}
    assert maximal["maximal_region_count"] == 54
    assert maximal["residual_host_operation_count"] == 2850
    assessment = json.loads((ROOT / "validation/deployment_gate_assessment.json").read_text())
    assert assessment["promotion_eligible"] is False
    assert "cosine_vs_fp32" in assessment["failed_checks"]
    print("verified: complete compiler artifact; deployment promotion rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
