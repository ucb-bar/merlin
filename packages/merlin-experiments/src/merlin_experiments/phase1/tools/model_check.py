"""In-sandbox client for the host-owned visible model-check channel."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import time
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Visible full-model and kernel checks")
    parser.add_argument("action", choices=("check", "replay-kernel", "wait"))
    parser.add_argument("--submission", default="submission")
    parser.add_argument("--model")
    parser.add_argument("--group")
    parser.add_argument("--split", default="visible", choices=("visible",))
    parser.add_argument("--engine", choices=("spike", "gsim"))
    parser.add_argument("--out")
    parser.add_argument("--job-id")
    parser.add_argument("--wait-budget", type=int, default=540)
    args = parser.parse_args(argv)
    if args.submission != "submission" or not 0 <= args.wait_budget <= 540:
        parser.error("submission must be the workspace submission; wait budget must be 0..540 seconds")
    if args.action != "wait" and not args.model:
        parser.error("--model is required")
    if args.action == "replay-kernel" and not args.group:
        parser.error("replay-kernel requires --group")
    if args.action == "wait" and not args.job_id:
        parser.error("wait requires --job-id")
    ws = Path.cwd()
    channel = ws / ".model_check_channel"
    if not channel.is_dir():
        parser.error("model-check broker is unavailable")
    job_id = args.job_id if args.action == "wait" else f"{os.getpid()}_{time.time_ns()}_{secrets.token_hex(8)}"
    if not all(c.isascii() and (c.isalnum() or c == "_") for c in job_id):
        parser.error("invalid job id")
    if args.action != "wait":
        request = {"action": args.action, "model": args.model, "group": args.group,
                   "engine": args.engine, "split": "visible"}
        target = channel / f"req_{job_id}.json"
        with target.open("x", encoding="utf-8") as stream:
            json.dump(request, stream)
    response = channel / f"resp_{job_id}.json"
    deadline = time.monotonic() + args.wait_budget
    while not response.is_file() and time.monotonic() < deadline:
        time.sleep(0.25)
    if not response.is_file():
        print(json.dumps({"status": "pending", "job_id": job_id, "wait_command":
                          f"python agent_model_check.py wait --job-id {job_id}"}))
        return 2
    result = json.loads(response.read_text(encoding="utf-8"))
    if args.out:
        output = Path(args.out)
        if output.is_absolute() or ".." in output.parts or output.is_symlink():
            parser.error("--out must be a relative workspace file")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    return 0 if result.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
