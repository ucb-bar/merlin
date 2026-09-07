#!/usr/bin/env python3
"""Isolated worker for the generic self-hosted-ISA Arc program oracle.

The parent supplies already assembled words, byte preloads, and explicit memory ranges to capture.
This process is deliberately disposable: an Arc backend may need an uncancellable large-stack Python
thread, while terminating this worker gives the grading harness a real wall-clock bound.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    request = json.loads(Path(args.request).read_text())
    from merlin.targetgen.program_oracle import _timing_block, run_raw_program

    result = run_raw_program(
        str(request["target"]),
        words=[int(word) for word in request.get("words") or []],
        preload=[(int(item["base"]), base64.b64decode(item["b64"]))
                 for item in request.get("preload") or []],
        max_cycles=int(request["max_cycles"]),
    )
    observations, capability = _timing_block(result)
    captured = {}
    if result.halted:
        for read in request.get("reads") or []:
            captured[str(read["name"])] = base64.b64encode(bytes(
                result.slave.captured(int(read["base"]), int(read["nbytes"])))).decode()
    payload = {
        "halted": bool(result.halted),
        "cycles": int(result.cycles),
        "captured": captured,
    }
    if observations:
        payload["timing_observations"] = observations
    if capability:
        payload["timing_capability"] = capability
    Path(args.out).write_text(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
