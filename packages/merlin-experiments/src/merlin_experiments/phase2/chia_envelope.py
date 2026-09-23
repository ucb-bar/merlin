"""Installed managed envelope for measured-claims coordination.

Explicit execution context; existing Chia supervisors own every child process.
Planning is not resource assignment, simulator qualification or compiler certification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

from merlin_experiments.phase2 import chia_launch


@dataclass(frozen=True)
class EnvelopeContext:
    """Explicit launcher identity, command selection and accounting ownership."""

    python: Path
    cwd: Path
    wrapper_source: Path
    coordinator_prefix: tuple[str, ...]
    suite: str
    target: str

    def __post_init__(self):
        for name in ("python", "cwd", "wrapper_source"):
            value = getattr(self, name)
            if not isinstance(value, Path) or not value.is_absolute():
                raise ValueError(f"envelope {name} must be an absolute Path")
        if not self.coordinator_prefix or any(not isinstance(x, str) or not x for x in self.coordinator_prefix):
            raise ValueError("envelope requires a nonempty coordinator prefix")
        if any(not isinstance(value, str) or not value.strip() for value in (self.suite, self.target)):
            raise ValueError("envelope requires explicit suite and target")


def _owner_identity() -> dict[str, str]:
    from merlin.common.paths import module_source_path

    owner = module_source_path(__name__)
    return {"path": str(owner), "sha256": _sha_file(owner)}


def _python_selection(command: list[str]) -> tuple[dict, dict]:
    """Capture only Python source selection, never arbitrary environment or credentials."""
    keys = chia_launch.PYTHON_SOURCE_ENVIRONMENT_KEYS
    if command[1:3] != ["-m", "merlin_experiments.phase2.checkpoint_cli"]:
        return {}, {key: os.environ.get(key) for key in keys}
    from merlin_experiments.measured_launch import source_inputs

    values = {}
    for name in ("core_package_root", "experiments_package_root", "experiments_namespace_root"):
        flag = "--" + name.replace("_", "-")
        if command.count(flag) != 1 or command.index(flag) + 1 >= len(command):
            raise RuntimeError(f"installed coordinator requires one explicit {flag}")
        value = command[command.index(flag) + 1]
        if not Path(value).is_absolute():
            raise RuntimeError(f"installed coordinator requires an absolute {flag}")
        values[name] = value
    paths = source_inputs({"measured_launch": {"values": values}, "argv": command})
    sources = {name: {"path": path, "sha256": _sha_file(Path(path))} for name, path in paths.items()}
    roots = [
        str(Path(values[name]).parent)
        for name in ("experiments_package_root", "experiments_namespace_root", "core_package_root")
    ]
    environment = {key: None for key in keys}
    environment.update(PYTHONPATH=os.pathsep.join(dict.fromkeys(roots)), PYTHONSAFEPATH="1", PYTHONNOUSERSITE="1")
    return sources, environment


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
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
def run_coordinator(command: list[str], cwd: str, plan: dict, receipt_root: str, wrapper_source: str) -> dict:
    """One non-retried task; the child coordinator alone owns resume/retry semantics."""
    return execute_coordinator(command, cwd, plan, receipt_root, wrapper_source)


def execute_coordinator(command: list[str], cwd: str, plan: dict, receipt_root: str, wrapper_source: str) -> dict:
    """Admit the assigned task and supervise its coordinator, retaining exact receipts."""
    import chia.trace
    import ray

    from merlin_experiments.execution.chia_native import run

    assigned = validate_assigned_resources(ray.get_runtime_context().get_assigned_resources())
    plan_sha256 = str(plan.get("sha256") or "")
    unhashed_plan = {key: value for key, value in plan.items() if key != "sha256"}
    if hashlib.sha256(_canonical(unhashed_plan)).hexdigest() != plan_sha256:
        raise RuntimeError("CHIA launch plan digest is invalid")
    if plan.get("command") != command or plan.get("cwd") != cwd:
        raise RuntimeError("CHIA worker command or cwd differs from its launch plan")
    sources, selected_environment = _python_selection(command)
    recorded_environment = plan.get("python_source_environment")
    if (
        plan.get("python_sources") != sources
        or not isinstance(recorded_environment, dict)
        or set(recorded_environment) != set(chia_launch.PYTHON_SOURCE_ENVIRONMENT_KEYS)
        or any(value is not None and not isinstance(value, str) for value in recorded_environment.values())
        or (sources and recorded_environment != selected_environment)
    ):
        raise RuntimeError("CHIA worker Python source selection differs from its launch plan")
    wrapper = Path(wrapper_source).resolve()
    chia_trace = Path(chia.trace.__file__).resolve()
    expected_wrapper = {"path": str(wrapper), "sha256": _sha_file(wrapper)}
    expected_chia = {"path": str(chia_trace), "sha256": _sha_file(chia_trace)}
    command_artifacts = chia_launch.command_artifacts(command)
    launch_policy = chia_launch.policy_identity()
    if (
        plan.get("envelope_owner") != _owner_identity()
        or plan.get("wrapper") != expected_wrapper
        or plan.get("chia_trace") != expected_chia
        or plan.get("command_artifacts") != command_artifacts
        or plan.get("launch_policy") != launch_policy
    ):
        raise RuntimeError("CHIA launch plan changed after its orchestration sources were pinned")
    launch = {
        "schema": "merlin.chia-agentic-perf-launch.v2",
        "status": "assigned_before_coordinator",
        "plan": plan,
        "plan_sha256": plan_sha256,
        "command": command,
        "command_artifacts": command_artifacts,
        "launch_policy": launch_policy,
        "required_resources": {"codex_slots": 1, "gsim_slots": 1},
        "assigned_resources": assigned,
        "wrapper": expected_wrapper,
        "chia_trace": expected_chia,
    }
    launch_path, launch_sha256 = _content_addressed_receipt(Path(receipt_root), "launch_receipt", launch)
    started = time.monotonic()
    environment = {
        **os.environ,
        "MERLIN_CHIA_ENVELOPE_PLAN_SHA256": plan_sha256,
        "MERLIN_CHIA_LAUNCH_RECEIPT": str(launch_path),
        "MERLIN_CHIA_LAUNCH_RECEIPT_SHA256": launch_sha256,
    }
    for key, value in recorded_environment.items():
        if value is None:
            environment.pop(key, None)
        else:
            environment[key] = value
    # Transport is prepared in the guarded parent and sealed into the plan. Ray workers do
    # not inherit a process-local import finder (or necessarily the parent's environment).
    returncode = run(plan.get("transport_command", command), cwd=cwd, env=environment)
    result = {
        "returncode": returncode,
        "wall_s": round(time.monotonic() - started, 3),
        "assigned_resources": assigned,
        "launch_receipt": {"path": str(launch_path), "sha256": launch_sha256},
    }
    result["assigned_resources_sha256"] = hashlib.sha256(
        (json.dumps(assigned, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    completion = {
        "schema": "merlin.chia-agentic-perf-completion.v1",
        "status": "complete" if returncode == 0 else "failed",
        "plan_sha256": plan_sha256,
        "launch_receipt": result["launch_receipt"],
        "returncode": returncode,
        "wall_s": result["wall_s"],
        "assigned_resources": assigned,
        "assigned_resources_sha256": result["assigned_resources_sha256"],
    }
    completion_path, completion_sha256 = _content_addressed_receipt(
        Path(receipt_root), "completion_receipt", completion
    )
    result["completion_receipt"] = {"path": str(completion_path), "sha256": completion_sha256}
    return result


def validate_assigned_resources(resources: dict) -> dict[str, float]:
    """Fail closed on Ray's runtime truth, independent of CHIA profiler option metadata."""
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in resources.values()
    ):
        raise RuntimeError("CHIA assigned logical resources must be finite numbers")
    normalized = {str(key): float(value) for key, value in resources.items()}
    missing = [name for name in ("codex_slots", "gsim_slots") if normalized.get(name, 0.0) < 1.0]
    if missing:
        raise RuntimeError(f"CHIA task lacks assigned logical resources: {', '.join(missing)}")
    return normalized


def plan_command(coordinator_args: list[str], *, context: EnvelopeContext, stub_seconds: float = 0.0) -> list[str]:
    """Pure command plan used by offline tests and ``--dry-run``."""
    if not math.isfinite(stub_seconds) or stub_seconds < 0:
        raise ValueError("stub seconds must be finite and non-negative")
    if stub_seconds:
        return [str(context.python), "-c", f"import time; time.sleep({float(stub_seconds)!r})"]
    if not coordinator_args:
        raise ValueError("coordinator arguments are required after --")
    if "--dry-run" in coordinator_args:
        raise ValueError("the actual CHIA campaign cannot wrap a coordinator --dry-run")
    return [str(context.python), *context.coordinator_prefix, *coordinator_args]


def main(argv: list[str] | None = None, *, context: EnvelopeContext) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orchestration-run-id", required=True)
    parser.add_argument("--codex-slots", type=int, default=1)
    parser.add_argument("--gsim-slots", type=int, default=1)
    parser.add_argument(
        "--stub-seconds",
        type=float,
        default=0.0,
        help="token-free CHIA/Ray envelope smoke; does not run the coordinator",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--managed-native-endpoint",
        help="required for execution: provisioned supervisor socket on the driver's host; single-node workers only",
    )
    parser.add_argument("coordinator_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    coordinator_args = list(args.coordinator_args)
    if coordinator_args[:1] == ["--"]:
        coordinator_args = coordinator_args[1:]
    if args.codex_slots < 1 or args.gsim_slots < 1 or args.stub_seconds < 0:
        parser.error("slot counts must be positive and stub seconds non-negative")
    try:
        command = plan_command(coordinator_args, context=context, stub_seconds=args.stub_seconds)
    except ValueError as exc:
        parser.error(str(exc))
    plan = {
        "schema_version": 1,
        "driver": "codex",
        "driver_parity_claim": False,
        "protocol": "unchanged_sequential_resume_safe_coordinator",
        "resources": {"codex_slots": 1, "gsim_slots": 1},
        "cluster_capacity": {"codex_slots": args.codex_slots, "gsim_slots": args.gsim_slots},
        "command": command,
        "cwd": str(context.cwd),
        "stub": bool(args.stub_seconds),
    }
    plan["sha256"] = hashlib.sha256(
        (json.dumps(plan, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.managed_native_endpoint:
        parser.error("execution requires --managed-native-endpoint; provision a supervisor on the driver's host")

    from merlin_experiments.execution.chia_native import Session

    with Session(args.managed_native_endpoint) as native_session:
        return _execute(args, command, plan, native_session, context=context)


def _execute(args, command, plan, native_session, *, context: EnvelopeContext):
    from merlin_experiments.execution.chia_group import NativeTaskGroup, local_options

    if not _HAVE_CHIA:
        from merlin.benchharness.chia_bridge import require_chia

        require_chia()
    import chia.trace

    from merlin.benchharness.chia_bridge import chia_run
    from merlin.benchharness.chia_tasks import chia_tasks

    wrapper = context.wrapper_source.resolve()
    trace_path = Path(chia.trace.__file__).resolve()
    plan["envelope_owner"] = _owner_identity()
    plan["wrapper"] = {"path": str(wrapper), "sha256": _sha_file(wrapper)}
    plan["chia_trace"] = {"path": str(trace_path), "sha256": _sha_file(trace_path)}
    plan["command_artifacts"] = chia_launch.command_artifacts(command)
    plan["launch_policy"] = chia_launch.policy_identity()
    plan["python_sources"], plan["python_source_environment"] = _python_selection(command)
    from merlin_experiments.frozen_python import inherited_python_command

    transport = inherited_python_command(command)
    if transport != command:
        plan["transport_command"] = transport
    plan.pop("sha256", None)
    plan["sha256"] = hashlib.sha256(_canonical(plan)).hexdigest()
    print(json.dumps(plan, indent=2))
    with (
        chia_run(
            accounting="child-ledgers",
            suite=context.suite,
            method="chia_agentic_perf_experiment",
            target=context.target,
            run_id=args.orchestration_run_id,
            extra={
                "driver": "codex",
                "driver_parity_claim": False,
                "protocol": plan["protocol"],
                "plan_sha256": plan["sha256"],
            },
            ray_resources={"codex_slots": args.codex_slots, "gsim_slots": args.gsim_slots},
        ) as run,
        chia_tasks(run) as tasks,
        NativeTaskGroup(run, native_session) as native_owner,
    ):
        launcher = run_coordinator.options(**local_options({"codex_slots": 1, "gsim_slots": 1}))
        ref = native_owner.submit(
            tasks,
            launcher.chia_remote,
            command,
            str(context.cwd),
            plan,
            str(run.run_dir / "chia"),
            str(wrapper),
            run_id=args.orchestration_run_id,
        )
        result = tasks.get(ref)
        native_owner.returned(ref, result)
        if result["returncode"] != 0:
            run.mark_failed()
        run.metrics.log_scalar("coordinator/wall_s", result["wall_s"], 0)
        run.metrics.log_scalar("coordinator/returncode", result["returncode"], 0)
        run.summary = {**plan, "result": result}
        plan_path = run.run_dir / "chia" / "agentic_perf_plan.json"
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        print(f"CHIA/AET orchestration run: {run.run_dir}")
        print(f"CHIA profile: {run.profile_path}")
    return int(result["returncode"])
