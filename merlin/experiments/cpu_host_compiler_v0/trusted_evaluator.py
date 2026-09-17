#!/usr/bin/env python3
"""In-workspace shim for the driver-side trusted CPU-host search broker.

The shim has no board, grader, or held-out access.  It forwards the fixed evaluator protocol used by
``beam_search.py`` through a workspace-local file channel.  The broker validates every request against
the frozen public split before compiling and timing it on K1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("screen", "confirm"), required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parent-policy", type=Path, required=True)
    parser.add_argument("--capsules", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    workspace = Path(__file__).resolve().parents[1]
    channel = workspace / ".trusted_search_channel"
    channel.mkdir(parents=True, exist_ok=True)
    request_id = f"{os.getpid()}_{time.monotonic_ns()}"
    request = {
        "version": 1,
        "phase": args.phase,
        "policy": str(args.policy.resolve()),
        "parent_policy": str(args.parent_policy.resolve()),
        "capsules": str(args.capsules.resolve()),
        "split": args.split,
        "repeats": args.repeats,
    }
    request_path = channel / f"req_{request_id}.json"
    request_temporary = channel / f".req_{request_id}.json.tmp"
    with request_temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(request, sort_keys=True))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(request_temporary, request_path)
    response = channel / f"resp_{request_id}.jsonl"
    receipt_path = channel / f"receipt_{request_id}.json"
    deadline = time.monotonic() + 7200
    while time.monotonic() < deadline:
        if receipt_path.is_file():
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise SystemExit(f"trusted search broker receipt is invalid: {exc}")
            if receipt.get("request_id") != request_id:
                raise SystemExit("trusted search broker receipt identity mismatch")
            if receipt.get("status") != "pass":
                raise SystemExit(str(receipt.get("error", "trusted search request failed"))[:4000])
            if not response.is_file():
                raise SystemExit("trusted search broker produced no observation artifact")
            if hashlib.sha256(response.read_bytes()).hexdigest() != receipt.get("response_sha256"):
                raise SystemExit("trusted search broker response differs from its terminal receipt")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(response.read_bytes())
            return 0
        time.sleep(0.2)
    client_failure = {
        "version": 1, "authority": "workspace_client", "request_id": request_id,
        "status": "timeout", "reason": "no driver terminal receipt before client deadline",
    }
    (channel / f"client_failure_{request_id}.json").write_text(
        json.dumps(client_failure, sort_keys=True) + "\n", encoding="utf-8")
    raise SystemExit("trusted search broker timed out without a terminal receipt")


if __name__ == "__main__":
    raise SystemExit(main())
