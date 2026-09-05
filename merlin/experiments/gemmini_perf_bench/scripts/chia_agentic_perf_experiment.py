#!/usr/bin/env python3
"""CHIA envelope for the unchanged sequential agentic performance coordinator.

This wrapper creates one CHIA/Ray task holding ``codex_slots`` and ``gsim_slots`` for the complete
resume-safe campaign.  The task shells out to ``run_agentic_perf_experiment.py`` under Merlin's main
venv; CHIA never enters the agent/simulator process tree and does not reorder holdout sealing,
authoring, regrade, reveal, or paired measurement.  It emits CHIA profiler/scalar artifacts in an
AET-managed orchestration run.

This is a Codex-only experiment, not a Claude-vs-Codex arm comparison.  Driver parity would require a
separately predeclared multi-driver design; CHIA is only the resource/profiling envelope here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "merlin/python"))

try:
    from chia.base.ChiaFunction import ChiaFunction
    _HAVE_CHIA = True
except Exception:  # noqa: BLE001 - planning/dry-run must import in the main venv
    _HAVE_CHIA = False

    def ChiaFunction(**_kwargs):  # type: ignore[no-redef]
        def decorate(function):
            return function
        return decorate


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n").encode()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _command_artifacts(command: list[str]) -> list[dict]:
    """Hash the executable and any file-valued entrypoint before dispatch."""
    artifacts = []
    for index in range(min(len(command), 2)):
        path = Path(command[index])
        if path.is_file():
            artifacts.append({"index": index, "path": str(path.resolve()),
                              "sha256": _sha_file(path)})
    return artifacts


def _content_addressed_receipt(root: Path, prefix: str, document: dict) -> tuple[Path, str]:
    payload = _canonical(document)
    digest = hashlib.sha256(payload).hexdigest()
    path = root / f"{prefix}.{digest}.json"
    root.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o444)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise RuntimeError(f"content-addressed CHIA receipt is inconsistent: {path}")
    else:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
    path.chmod(0o444)
    return path.resolve(), digest


@ChiaFunction(resources={"codex_slots": 1, "gsim_slots": 1}, num_cpus=1, max_retries=0)
def run_coordinator(command: list[str], cwd: str, plan: dict, receipt_root: str) -> dict:
    """One non-retried task; the child coordinator alone owns resume/retry semantics."""
    import chia.trace
    import ray
    assigned = validate_assigned_resources(ray.get_runtime_context().get_assigned_resources())
    plan_sha256 = str(plan.get("sha256") or "")
    unhashed_plan = {key: value for key, value in plan.items() if key != "sha256"}
    if hashlib.sha256(_canonical(unhashed_plan)).hexdigest() != plan_sha256:
        raise RuntimeError("CHIA launch plan digest is invalid")
    wrapper = Path(__file__).resolve()
    chia_trace = Path(chia.trace.__file__).resolve()
    expected_wrapper = {"path": str(wrapper), "sha256": _sha_file(wrapper)}
    expected_chia = {"path": str(chia_trace), "sha256": _sha_file(chia_trace)}
    command_artifacts = _command_artifacts(command)
    if (plan.get("wrapper") != expected_wrapper or plan.get("chia_trace") != expected_chia
            or plan.get("command_artifacts") != command_artifacts):
        raise RuntimeError("CHIA launch plan changed after its orchestration sources were pinned")
    launch = {
        "schema": "merlin.chia-agentic-perf-launch.v1",
        "status": "assigned_before_coordinator",
        "plan": plan,
        "plan_sha256": plan_sha256,
        "command": command,
        "command_artifacts": command_artifacts,
        "required_resources": {"codex_slots": 1, "gsim_slots": 1},
        "assigned_resources": assigned,
        "wrapper": expected_wrapper,
        "chia_trace": expected_chia,
    }
    launch_path, launch_sha256 = _content_addressed_receipt(
        Path(receipt_root), "launch_receipt", launch)
    started = time.monotonic()
    environment = {
        **os.environ,
        "MERLIN_CHIA_ENVELOPE_PLAN_SHA256": plan_sha256,
        "MERLIN_CHIA_LAUNCH_RECEIPT": str(launch_path),
        "MERLIN_CHIA_LAUNCH_RECEIPT_SHA256": launch_sha256,
    }
    completed = subprocess.run(command, cwd=cwd, env=environment)
    result = {"returncode": completed.returncode,
              "wall_s": round(time.monotonic() - started, 3),
              "assigned_resources": assigned,
              "launch_receipt": {"path": str(launch_path), "sha256": launch_sha256}}
    result["assigned_resources_sha256"] = hashlib.sha256(
        (json.dumps(assigned, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()
    completion = {
        "schema": "merlin.chia-agentic-perf-completion.v1",
        "status": "complete" if completed.returncode == 0 else "failed",
        "plan_sha256": plan_sha256,
        "launch_receipt": result["launch_receipt"],
        "returncode": completed.returncode,
        "wall_s": result["wall_s"],
        "assigned_resources": assigned,
        "assigned_resources_sha256": result["assigned_resources_sha256"],
    }
    completion_path, completion_sha256 = _content_addressed_receipt(
        Path(receipt_root), "completion_receipt", completion)
    result["completion_receipt"] = {
        "path": str(completion_path), "sha256": completion_sha256}
    return result


def validate_assigned_resources(resources: dict) -> dict[str, float]:
    """Fail closed on Ray's runtime truth, independent of CHIA profiler option metadata."""
    normalized = {str(key): float(value) for key, value in resources.items()}
    missing = [name for name in ("codex_slots", "gsim_slots")
               if normalized.get(name, 0.0) < 1.0]
    if missing:
        raise RuntimeError(f"CHIA task lacks assigned logical resources: {', '.join(missing)}")
    return normalized


def plan_command(coordinator_args: list[str], *, stub_seconds: float = 0.0) -> list[str]:
    """Pure command plan used by offline tests and ``--dry-run``."""
    from merlin.benchharness.chia_bridge import driver_python
    if stub_seconds:
        return [driver_python(), "-c", f"import time; time.sleep({float(stub_seconds)!r})"]
    if not coordinator_args:
        raise ValueError("coordinator arguments are required after --")
    if "--dry-run" in coordinator_args:
        raise ValueError("the actual CHIA campaign cannot wrap a coordinator --dry-run")
    return [driver_python(), str(HERE / "run_agentic_perf_experiment.py"), *coordinator_args]


def _argument_value(arguments: list[str], option: str) -> str | None:
    try:
        return arguments[arguments.index(option) + 1]
    except (ValueError, IndexError):
        return None


def _target(arguments: list[str]) -> str:
    descriptor = _argument_value(arguments, "--descriptor")
    if not descriptor:
        return "gemmini"
    import yaml
    document = yaml.safe_load(Path(descriptor).read_text(encoding="utf-8")) or {}
    return str(document.get("target") or "gemmini")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orchestration-run-id", required=True)
    parser.add_argument("--codex-slots", type=int, default=1)
    parser.add_argument("--gsim-slots", type=int, default=1)
    parser.add_argument("--stub-seconds", type=float, default=0.0,
                        help="token-free CHIA/Ray envelope smoke; does not run the coordinator")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("coordinator_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    coordinator_args = list(args.coordinator_args)
    if coordinator_args[:1] == ["--"]:
        coordinator_args = coordinator_args[1:]
    if args.codex_slots < 1 or args.gsim_slots < 1 or args.stub_seconds < 0:
        parser.error("slot counts must be positive and stub seconds non-negative")
    try:
        command = plan_command(coordinator_args, stub_seconds=args.stub_seconds)
    except ValueError as exc:
        parser.error(str(exc))
    plan = {
        "schema_version": 1, "driver": "codex", "driver_parity_claim": False,
        "protocol": "unchanged_sequential_resume_safe_coordinator",
        "resources": {"codex_slots": 1, "gsim_slots": 1},
        "cluster_capacity": {"codex_slots": args.codex_slots, "gsim_slots": args.gsim_slots},
        "command": command, "stub": bool(args.stub_seconds),
    }
    plan["sha256"] = hashlib.sha256(
        (json.dumps(plan, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if not _HAVE_CHIA:
        from merlin.benchharness.chia_bridge import require_chia
        require_chia()
    from merlin.benchharness.chia_bridge import chia_get, chia_run
    import chia.trace
    wrapper = Path(__file__).resolve()
    trace_path = Path(chia.trace.__file__).resolve()
    plan["wrapper"] = {"path": str(wrapper), "sha256": _sha_file(wrapper)}
    plan["chia_trace"] = {"path": str(trace_path), "sha256": _sha_file(trace_path)}
    plan["command_artifacts"] = _command_artifacts(command)
    plan.pop("sha256", None)
    plan["sha256"] = hashlib.sha256(_canonical(plan)).hexdigest()
    print(json.dumps(plan, indent=2))
    with chia_run(
            suite="gemmini-perf-bench", method="chia_agentic_perf_experiment",
            target=_target(coordinator_args), run_id=args.orchestration_run_id,
            extra={"driver": "codex", "driver_parity_claim": False,
                   "protocol": plan["protocol"], "plan_sha256": plan["sha256"]},
            ray_resources={"codex_slots": args.codex_slots, "gsim_slots": args.gsim_slots}) as run:
        result = chia_get(run_coordinator.chia_remote(
            command, str(REPO), plan, str(run.run_dir / "chia")))
        run.metrics.log_scalar("coordinator/wall_s", result["wall_s"], 0)
        run.metrics.log_scalar("coordinator/returncode", result["returncode"], 0)
        run.summary = {**plan, "result": result}
        plan_path = run.run_dir / "chia" / "agentic_perf_plan.json"
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        print(f"CHIA/AET orchestration run: {run.run_dir}")
        print(f"CHIA profile: {run.profile_path}")
    return int(result["returncode"])


if __name__ == "__main__":
    raise SystemExit(main())
