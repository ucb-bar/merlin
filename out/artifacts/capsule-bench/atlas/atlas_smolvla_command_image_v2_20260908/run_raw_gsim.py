#!/usr/bin/env python3
"""Replay the real SmolVLA-shape kernel while retaining the raw GSIM transaction."""
from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from run_integration import fixture  # noqa: E402

CASE = ROOT / "cases" / "smolvla_tail_50_720_32"
GSIM = Path("/scratch/agustin/tmp/gsim-atlas-core/atlas_gsim_sim")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    cb = json.loads((CASE / "command_buffer.json").read_text())
    kernel_bytes = (CASE / "gsim_run" / "kernel.bin").read_bytes()
    words = np.frombuffer(kernel_bytes, dtype="<u4").astype(np.uint32).tolist()
    laid = json.loads((CASE / "gsim_run" / "npu_inputs.json").read_text())["inputs"]
    preload = [[int(item["base"]), base64.b64decode(item["b64"]).hex()] for item in laid]
    reads = []
    for name in cb["kernel_abi"]["outputs"]:
        spec = cb["tensors"][name]
        elements = int(np.prod(spec["shape"]))
        reads.append([int(spec["base"]), elements * 2])
    spec = {"words": words, "preload": preload, "reads": reads, "max_cycles": 50_000_000}
    spec_path = CASE / "raw_gsim_spec.json"
    spec_path.write_text(json.dumps(spec, separators=(",", ":")) + "\n")
    proc = subprocess.run([str(GSIM), str(spec_path)], capture_output=True, text=True, timeout=300)
    (CASE / "raw_gsim_stdout.txt").write_text(proc.stdout)
    (CASE / "raw_gsim_stderr.txt").write_text(proc.stderr)
    if proc.returncode:
        raise RuntimeError(f"GSIM rc={proc.returncode}: {proc.stderr[-500:]}")
    line = next(line for line in reversed(proc.stdout.splitlines()) if line.startswith("{"))
    raw = json.loads(line)
    raw_outputs = [bytes.fromhex(value) for value in raw["outputs"]]
    _, expected = fixture("smolvla_tail_50_720_32", cb)
    comparisons = {}
    for output_raw, (name, reference) in zip(raw_outputs, expected.items()):
        (CASE / f"raw_{name}.bf16.bin").write_bytes(output_raw)
        actual = (np.frombuffer(output_raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        actual = actual.reshape(reference.shape)
        comparisons[name] = {
            "base": cb["tensors"][name]["base"],
            "bytes": len(output_raw),
            "raw_sha256": sha(output_raw),
            "mismatches": int(np.count_nonzero(actual != reference)),
            "max_abs_error": float(np.max(np.abs(actual - reference))),
        }
    record = {
        "schema": "atlas_raw_gsim_readback_v1",
        "engine_binary": str(GSIM),
        "engine_sha256": sha(GSIM.read_bytes()),
        "kernel_binary_sha256": sha(kernel_bytes),
        "spec_sha256": sha(spec_path.read_bytes()),
        "returncode": proc.returncode,
        "halted": bool(raw["halted"]),
        "cycles": int(raw["cycles"]),
        "reads": raw.get("reads"),
        "writes": raw.get("writes"),
        "comparisons": comparisons,
        "all_outputs_bit_exact": all(v["mismatches"] == 0 for v in comparisons.values()),
        "provenance_note": "Expected values are computed after raw GSIM readback and are absent from raw_gsim_spec.json.",
    }
    (CASE / "raw_readback.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["halted"] and record["all_outputs_bit_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
