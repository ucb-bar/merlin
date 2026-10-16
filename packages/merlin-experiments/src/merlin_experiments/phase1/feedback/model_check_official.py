"""Host-only sealed model checks, with optional bounded model-kernel RTL replays."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from merlin.common.tree_hash import hash_tree
from merlin.model_check.bundle import Bundle
from merlin.model_check.models import default_models_path, target_from_contract

REPORT_SCHEMA = "merlin.model_check.report.v1"


def _invoke(command: str, *, submission: Path, model: str, output: Path, bundle: Path,
            timeout_s: int, program: str | None = None, group: str | None = None) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    target = target_from_contract()
    argv = [sys.executable, "-m", "merlin.model_check", command, "--submission", str(submission),
            "--model", model, "--split", "sealed", "--bundle", str(bundle),
            "--models", str(default_models_path(target)), "--no-cache",
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
             bundle: Path | None = None, model_timeout_s: int = 7200,
             rtl_budget_s: int = 7200, kernel_timeout_s: int = 300,
             rtl_workers: int = 4, replay_model_kernels: bool = False) -> dict:
    """Require sealed full-chain Spike; model-kernel RTL is an explicit opt-in.

    Capsule L3 qualification and timing collection are separate and stay enabled.
    An unrequested model RTL check is never reported as completed.
    """
    if bundle is None:
        bundle = Bundle.default(target_from_contract()).root
    root = run_dir / "grading_model_check"
    root.mkdir(parents=True, exist_ok=True)
    source_sha256 = hash_tree(submission)["sha256"]
    configured_slots = os.environ.get("MERLIN_GSIM_MAX_SLOTS", "").strip()
    try:
        slot_cap = max(1, int(configured_slots)) if configured_slots else 4
    except ValueError:
        slot_cap = 1
    summary = {"schema": "merlin.phase1.model_check_official.v1", "passed": False,
               "split": "sealed", "bundle": str(bundle), "submission_sha256": source_sha256,
               "models": {}, "rtl": [],
               "unique_kernel_count": 0, "rtl_complete": False,
               "rtl_required": replay_model_kernels,
               "rtl_status": "pending" if replay_model_kernels else "not_requested",
               "rtl_workers": max(1, min(int(rtl_workers), 4, slot_cap))}
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
        metrics = (body or {}).get("model_metrics")
        if not isinstance(metrics, dict) or not isinstance(metrics.get("worst"), dict):
            result["status"] = "unavailable" if result["status"] == "pass" else result["status"]
        for program in programs:
            item = observed.get(program, {})
            build = item.get("build")
            census = item.get("census") or {}
            kernels = item.get("kernels") or []
            runs = item.get("runs") or []
            if (not isinstance(build, dict) or build.get("status") != "linked"
                    or not isinstance(census, dict)
                    or not isinstance(census.get("denominator"), int)
                    or not isinstance(census.get("device"), int)
                    or not 0 <= census["device"] <= census["denominator"]
                    or not any(isinstance(run, dict) and run.get("engine") == "spike"
                               and run.get("status") == "completed" for run in runs)
                    or census["device"] > 0 and not any(
                        isinstance(row, dict) and isinstance(row.get("source_region_id"), str)
                        for row in kernels)):
                result["status"] = "unavailable" if result["status"] == "pass" else result["status"]
        summary["models"][model] = {**result, "programs": {
            program: {"build": observed.get(program, {}).get("build"),
                      "census": observed.get(program, {}).get("census")}
            for program in programs}}
        if result["status"] != "pass":
            continue  # no RTL cost can turn an incomplete full-chain check into an official pass
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

    def replay(entry: tuple[int, tuple[str, str, str]]) -> dict:
        index, (model, program, group) = entry
        # The host wait may exceed the CLI timeout by 30 seconds, followed by termination grace.
        # Reserve that time inside the global budget so unfinished kernels become pending.
        remaining = int(deadline - time.monotonic()) - 40
        if remaining < 1:
            return {"model": model, "program": program, "group": group,
                    "status": "pending", "reason": "RTL budget exhausted"}
        try:
            result = _invoke("replay-kernel", submission=submission, model=model, program=program,
                             group=group, output=root / "rtl" / f"kernel_{index:05d}.json", bundle=bundle,
                             timeout_s=min(kernel_timeout_s, remaining))
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            result = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
        result.pop("body", None)
        return {"model": model, "program": program, "group": group, **result}

    if replay_model_kernels:
        with ThreadPoolExecutor(max_workers=summary["rtl_workers"]) as executor:
            summary["rtl"] = list(executor.map(replay, enumerate(selected.values())))
    models_complete = bool(required_programs) and all(
        row["status"] == "pass" for row in summary["models"].values())
    summary["rtl_complete"] = replay_model_kernels and models_complete and len(summary["rtl"]) == len(selected) and all(
        row["status"] == "pass" for row in summary["rtl"])
    if replay_model_kernels:
        summary["rtl_status"] = "pass" if summary["rtl_complete"] else "incomplete"
    summary["passed"] = models_complete and (not replay_model_kernels or summary["rtl_complete"])
    if hash_tree(submission)["sha256"] != source_sha256:
        summary["passed"] = False
        summary["source_changed_during_grade"] = True
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
