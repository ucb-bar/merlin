#!/usr/bin/env python3
"""Bounded, durable probe of the unchanged full SmolVLA capture."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
CAPTURE = REPO / "out/artifacts/recaptures/smolvla_fp32_consistent/model.mlir"
TOOL = ROOT / "submission/mlir_oot/atlas-opt"
CB = ROOT / "full_capture_command_buffer.json"

argv = [str(TOOL), f"--emit-command-buffer={CB}", "--emit-target-artifact", str(CAPTURE)]
started = time.monotonic()
try:
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    timed_out = False
    returncode = proc.returncode
    stdout, stderr = proc.stdout, proc.stderr
except subprocess.TimeoutExpired as exc:
    timed_out = True
    returncode = None
    stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
    stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")

(ROOT / "full_capture.stdout").write_text(stdout)
(ROOT / "full_capture.stderr").write_text(stderr)
record = {
    "schema": "atlas_full_capture_probe_v1",
    "argv": argv,
    "capture": str(CAPTURE),
    "capture_sha256": hashlib.sha256(CAPTURE.read_bytes()).hexdigest(),
    "capture_bytes": CAPTURE.stat().st_size,
    "timeout_seconds": 60,
    "elapsed_seconds": time.monotonic() - started,
    "timed_out": timed_out,
    "returncode": returncode,
    "command_buffer_produced": CB.is_file(),
    "stdout_bytes": len(stdout.encode()),
    "stderr_bytes": len(stderr.encode()),
    "claim": "bounded integration probe only",
}
(ROOT / "full_capture_probe.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(json.dumps(record, indent=2, sort_keys=True))
