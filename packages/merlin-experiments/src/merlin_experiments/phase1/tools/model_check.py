"""In-sandbox client for the host-owned visible model-check channel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import time
from pathlib import Path


def compact_result(result, *, artifact):
    """Summarize an already-public report; retain full detail in its original file."""
    report = result.get("report") or {}
    failures = []

    def failure(value):
        if isinstance(value, dict):
            value = next((value[key] for key in ("detail", "reason", "error", "message")
                          if isinstance(value.get(key), str)), None)
        if isinstance(value, str) and value.strip():
            failures.append(value)

    failure(result.get("error"))
    for value in report.get("failures") or []:
        failure(value)
    programs = []
    for program in report.get("programs") or []:
        refused = program.get("refused") or {}
        for value in refused.values() if isinstance(refused, dict) else refused:
            failure(value)
        build = program.get("build") or {}
        census = program.get("census") or {}
        images = program.get("images") or []
        programs.append({
            "program": program.get("program"),
            "build": {key: build[key] for key in ("status", "built", "refused", "seconds") if key in build},
            "census": {key: census[key] for key in ("denominator", "device", "refused", "device_mac_fraction") if key in census},
            "replay_images": len(images),
            "replay_cases": sum(image["cases"] for image in images if type(image.get("cases")) is int),
            "replay_engines": sorted({image["engine"] for image in images if isinstance(image.get("engine"), str)}),
        })
    if not failures:
        for gate in (report.get("gates") or {}).values():
            if isinstance(gate, dict) and gate.get("status") not in {"pass", "passed"}:
                failure(gate)
    distinct = list(dict.fromkeys(failures))
    gates = {name: {key: value[:512] if isinstance(value, str) else value
                    for key, value in gate.items() if key in {"status", "partial", "detail"}}
             for name, gate in (report.get("gates") or {}).items() if isinstance(gate, dict)}
    return {
        "schema": "merlin.model_check.compact_feedback.v1",
        **{key: result.get(key) for key in ("status", "model", "action", "source_sha256", "returncode")},
        "scope": {"report_schema": report.get("schema"), "command": report.get("command"),
                  "split": report.get("split"), "mode": result.get("mode", report.get("mode", "unspecified")),
                  "interpretation": "Recorded gates and kernel replay counts only; no inferred whole-chain qualification."},
        "gates": gates, "programs": programs[:16], "programs_omitted": max(0, len(programs) - 16),
        "failure_count": len(failures), "distinct_failure_count": len(distinct),
        "first_distinct_failure": distinct[0][:2000] if distinct else None,
        "first_failure_truncated": bool(distinct and len(distinct[0]) > 2000),
        "full_artifact": artifact,
        "detail_hint": "Read the full artifact or repeat wait with --full-json; --out always keeps full JSON.",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Visible full-model and kernel checks")
    parser.add_argument("action", choices=("check", "replay-kernel", "wait"))
    parser.add_argument("--submission", default="submission")
    parser.add_argument("--model")
    parser.add_argument("--program")
    parser.add_argument("--group")
    parser.add_argument("--split", default="visible", choices=("visible",))
    parser.add_argument("--engine", choices=("spike",))
    parser.add_argument("--out")
    parser.add_argument("--full-json", action="store_true", help="print the full report instead of its compact summary")
    parser.add_argument("--job-id")
    parser.add_argument("--wait-budget", type=int, default=540)
    args = parser.parse_args(argv)
    if args.submission != "submission" or not 0 <= args.wait_budget <= 540:
        parser.error("submission must be the workspace submission; wait budget must be 0..540 seconds")
    if args.action != "wait" and not args.model:
        parser.error("--model is required")
    if args.action == "replay-kernel" and (not args.group or not args.program):
        parser.error("replay-kernel requires --program and --group")
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
        request = {"action": args.action, "model": args.model, "program": args.program, "group": args.group,
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
    raw = response.read_bytes()
    result = json.loads(raw)
    artifact_path = response.relative_to(ws)
    if args.out:
        output = Path(args.out)
        if output.is_absolute() or ".." in output.parts or output.is_symlink():
            parser.error("--out must be a relative workspace file")
        if not output.resolve().is_relative_to(ws.resolve()):
            parser.error("--out must stay inside the workspace")
        output.parent.mkdir(parents=True, exist_ok=True)
        raw = (json.dumps(result, indent=2) + "\n").encode("utf-8")
        output.write_bytes(raw)
        artifact_path = output
    artifact = {"path": str(artifact_path), "sha256": hashlib.sha256(raw).hexdigest()}
    print(json.dumps(result if args.full_json else compact_result(result, artifact=artifact)))
    return 0 if result.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
