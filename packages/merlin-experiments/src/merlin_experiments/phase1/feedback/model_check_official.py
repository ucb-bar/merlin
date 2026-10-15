"""Host-only sealed model checks and bounded RTL replays for a frozen submission."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from merlin.common.tree_hash import hash_tree
from merlin.model_check.models import default_models_path

REPORT_SCHEMA = "merlin.model_check.report.v1"
DEFAULT_BUNDLE = Path("/work/merlin/private/gemmini/numerics/v1")


def _invoke(command: str, *, submission: Path, model: str, output: Path, bundle: Path,
            timeout_s: int, program: str | None = None, group: str | None = None) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    argv = [sys.executable, "-m", "merlin.model_check", command, "--submission", str(submission),
            "--model", model, "--split", "sealed", "--bundle", str(bundle),
            "--models", str(default_models_path("gemmini")),
            "--work", str(output.with_suffix(".work")), "--out", str(output), "--timeout-s", str(timeout_s)]
    if command == "check":
        argv += ["--mode", "full"]
    else:
        argv += ["--program", str(program), "--group", str(group), "--engine", "gsim"]
    log = output.with_suffix(".stderr.log")
    with log.open("wb") as stream:
        process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            process.wait(timeout=timeout_s + 30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return {"status": "timeout", "report": str(output), "log": str(log)}
    try:
        report = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        report = None
    if not isinstance(report, dict) or any((
        report.get("schema") != REPORT_SCHEMA, report.get("command") != command,
        report.get("model") != model, report.get("split") != "sealed",
        report.get("exit_code") != process.returncode,
    )):
        return {"status": "unavailable", "report": str(output), "log": str(log)}
    if report.get("status") not in {"pass", "fail", "error"}:
        return {"status": "unavailable", "report": str(output), "log": str(log)}
    return {"status": report["status"], "report": str(output), "log": str(log), "body": report}


def run_gate(submission: Path, run_dir: Path, required_programs: dict[str, tuple[str, ...]], *,
             bundle: Path = DEFAULT_BUNDLE, model_timeout_s: int = 7200,
             rtl_budget_s: int = 7200, kernel_timeout_s: int = 300) -> dict:
    """Require sealed full-chain Spike and every observed unique device kernel on gSIM."""
    root = run_dir / "grading_model_check"
    root.mkdir(parents=True, exist_ok=True)
    source_sha256 = hash_tree(submission)["sha256"]
    summary = {"schema": "merlin.phase1.model_check_official.v1", "passed": False,
               "split": "sealed", "bundle": str(bundle), "submission_sha256": source_sha256,
               "models": {}, "rtl": [],
               "unique_kernel_count": 0, "rtl_complete": False}
    selected: dict[tuple, tuple[str, str, str]] = {}
    for model, programs in required_programs.items():
        result = _invoke("check", submission=submission, model=model,
                         output=root / "models" / f"{model}.json", bundle=bundle,
                         timeout_s=model_timeout_s)
        body = result.pop("body", None)
        observed = {row.get("program"): row for row in (body or {}).get("programs", [])
                    if isinstance(row, dict) and isinstance(row.get("program"), str)}
        if set(observed) != set(programs):
            result["status"] = "unavailable" if result["status"] == "pass" else result["status"]
        for program in programs:
            item = observed.get(program, {})
            census = item.get("census") or {}
            kernels = item.get("kernels") or []
            if (not isinstance(census, dict) or not isinstance(census.get("device"), int)
                    or census["device"] > 0 and not any(
                        isinstance(row, dict) and isinstance(row.get("source_region_id"), str)
                        for row in kernels)):
                result["status"] = "unavailable" if result["status"] == "pass" else result["status"]
        summary["models"][model] = {**result, "programs": {
            program: {"build": observed.get(program, {}).get("build"),
                      "census": observed.get(program, {}).get("census")}
            for program in programs}}
        for program in programs:
            for kernel in observed.get(program, {}).get("kernels", []):
                if not isinstance(kernel, dict) or not isinstance(kernel.get("source_region_id"), str):
                    continue
                group = kernel["source_region_id"]
                # Cross-program reuse needs an explicit interface and shape identity. With either
                # absent, replay the group again; missing evidence can never reduce RTL coverage.
                interface = kernel.get("interface_sha256")
                shape = kernel.get("shape_sha256")
                key = (interface, shape) if interface and shape else (model, program, group)
                selected.setdefault(key, (model, program, group))
    summary["unique_kernel_count"] = len(selected)
    deadline = time.monotonic() + rtl_budget_s
    for index, (model, program, group) in enumerate(selected.values()):
        remaining = int(deadline - time.monotonic())
        if remaining < 1:
            summary["rtl"].append({"model": model, "program": program, "group": group,
                                   "status": "pending", "reason": "RTL budget exhausted"})
            continue
        result = _invoke("replay-kernel", submission=submission, model=model, program=program,
                         group=group, output=root / "rtl" / f"kernel_{index:05d}.json", bundle=bundle,
                         timeout_s=min(kernel_timeout_s, remaining))
        result.pop("body", None)
        summary["rtl"].append({"model": model, "program": program, "group": group, **result})
    summary["rtl_complete"] = len(summary["rtl"]) == len(selected) and all(
        row["status"] == "pass" for row in summary["rtl"])
    summary["passed"] = bool(required_programs) and all(
        row["status"] == "pass" for row in summary["models"].values()) and summary["rtl_complete"]
    if hash_tree(submission)["sha256"] != source_sha256:
        summary["passed"] = False
        summary["source_changed_during_grade"] = True
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
