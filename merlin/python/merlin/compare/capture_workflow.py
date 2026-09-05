"""All-or-nothing materialization of full paper-model capture bundles.

This workflow is deliberately downstream of the generic CPU-host campaign. It will not import or
execute a paper holdout until that campaign is marked complete and its compiler/runtime outputs are
sealed. The default CLI operation is a preflight plan; ``--execute`` is the explicit heavy action.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import time
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from merlin.baselines import bundle as capture_bundle
from merlin.common.artifacts import ProductDir, new_product
from merlin.common.paths import bench_dir, repo_root
from merlin.common.yaml import write_yaml

from .freeze import sha256_paths
from .host_experiment import HostExperimentSpec
from .paper import ModelSpec, PaperStudySpec
from .session import validate_capture_session, validate_paper_input_binding


_ENV_PREFIXES = ("M2M_", "VITFLY_", "HF_", "TRANSFORMERS_")
_FORMATS = {"w8a8": "int8", "fp32": "fp32"}


class CaptureWorkflowNotReady(RuntimeError):
    """The requested paper capture set was not authorized or not fully materialized."""

    def __init__(self, reasons: list[str], output_dir: Path):
        super().__init__("; ".join(reasons))
        self.reasons = tuple(reasons)
        self.output_dir = output_dir


@dataclass(frozen=True)
class CaptureTask:
    model: ModelSpec
    precision: str
    variant: str
    fmt: str
    workload: str
    python: Path
    output: Path
    command: tuple[str, ...]
    environment: dict[str, str]
    loader_sha256: str

    def to_dict(self) -> dict[str, Any]:
        command = list(self.command)
        return {
            "model": self.model.name,
            "capture": self.model.capture,
            "checkpoint": self.model.checkpoint,
            "precision": self.precision,
            "variant": self.variant,
            "format": self.fmt,
            "workload": self.workload,
            "python": str(self.python),
            "output": str(self.output),
            "command": command,
            "command_sha256": _json_sha256(command),
            "environment": dict(sorted(self.environment.items())),
            "environment_sha256": _json_sha256(self.environment),
            "loader_sha256": self.loader_sha256,
            "expected_provenance": dict(self.model.expected_provenance),
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_provenance(root: Path) -> dict[str, Any]:
    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, timeout=30)

    revision = run("rev-parse", "HEAD")
    status = run("status", "--porcelain=v1")
    return {
        "git_sha": revision.stdout.strip() if revision.returncode == 0 else None,
        "git_sha_error": revision.stderr.strip() if revision.returncode else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "dirty_paths": status.stdout.splitlines() if status.returncode == 0 else [],
        "status_error": status.stderr.strip() if status.returncode else None,
    }


def _resolve_paper_inputs(study: PaperStudySpec) -> Path:
    path = Path(str(study.paper_inputs.get("path", "")))
    return path.resolve() if path.is_absolute() else (repo_root() / path).resolve()


def _capture_python(root: Path, workload: str) -> Path:
    model_dir = root / "workloads" / workload
    config_path = model_dir / "capture.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    value = Path(str(config.get("venv", ".venv")))
    venv = value if value.is_absolute() else (model_dir / value).resolve()
    return venv / "bin" / "python"


def _render_environment(record: dict[str, Any], bundle: Path) -> dict[str, str]:
    raw = record.get("environment")
    if not isinstance(raw, dict):
        raise ValueError("paper input model record has no environment mapping")
    rendered = {str(key): str(value).replace("{bundle}", str(bundle))
                for key, value in raw.items()}
    unresolved = sorted(key for key, value in rendered.items() if "{bundle}" in value)
    if unresolved:
        raise ValueError(f"paper input environment has unresolved bundle paths: {unresolved}")
    if not any(key.endswith("_PAPER_READY") and value == "1"
               for key, value in rendered.items()):
        raise ValueError("paper input environment does not explicitly require paper readiness")
    return rendered


def _sanitized_environment(exact: dict[str, str]) -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(_ENV_PREFIXES)}
    environment.update(exact)
    return environment


def _source_paths(root: Path, study: PaperStudySpec) -> list[Path]:
    paths = [root / "workloads" / "capture_consistent.py", root / "workloads" / "capture.py",
             root / "m2m"]
    for model in study.models:
        workload = root / "workloads" / model.capture
        paths.extend([workload / "loader.py", workload / "capture.toml"])
    return paths


def _preflight(study: PaperStudySpec, host: HostExperimentSpec, model2mlir: Path,
               paper_inputs: Path) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    errors: list[str] = []
    if study.status != "draft":
        errors.append("paper capture requires a draft study; recapture after study freeze is forbidden")
    for model in study.models:
        for precision, artifact in model.artifacts.items():
            if str(artifact.get("sha256", "")).lower() != "unresolved":
                errors.append(
                    f"{model.name}/{precision}: capture is already registered; post-registration "
                    "recapture is forbidden")
    if host.status != "campaign_complete":
        errors.append(
            "CPU-host campaign is not campaign_complete; paper holdouts remain sealed until the "
            "generic compiler/runtime outputs are final")
    if set(host.paper_holdouts) != set(study.holdout_models):
        errors.append("CPU-host campaign and paper study have different holdout rosters")
    if host.freeze.get("require_all_four_arms_complete") is not True:
        errors.append("CPU-host campaign does not require all four arms to complete")
    host_preflight = host.preflight(check_environment=False, probe_board=False)
    errors.extend(host_preflight.errors)
    errors.extend(host_preflight.blockers)

    expected_input_digest = str(study.paper_inputs.get("sha256", ""))
    actual_input_digest = None
    if not paper_inputs.is_dir():
        errors.append(f"paper input bundle is absent: {paper_inputs}")
    else:
        actual_input_digest = sha256_paths([paper_inputs])
        if actual_input_digest != expected_input_digest:
            errors.append(
                "paper input bundle digest differs: "
                f"study={expected_input_digest} actual={actual_input_digest}")
        else:
            errors.extend(validate_paper_input_binding(paper_inputs, study.models))

    record: dict[str, Any] = {}
    record_path = paper_inputs / "paper_inputs.json"
    if record_path.is_file():
        try:
            loaded = json.loads(record_path.read_text(encoding="utf-8"))
            record = loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"paper input record cannot be loaded: {exc}")
    if set((record.get("models") or {}).keys()) != set(study.holdout_models):
        errors.append("paper input record does not contain exactly the frozen holdout roster")

    source_paths = _source_paths(model2mlir, study)
    missing_sources = [str(path) for path in source_paths if not path.exists()]
    if missing_sources:
        errors.append(f"model2MLIR capture sources are absent: {missing_sources}")
    source_digest = sha256_paths(source_paths) if not missing_sources else None
    evidence = {
        "paper_inputs": {"path": str(paper_inputs), "expected_sha256": expected_input_digest,
                         "actual_sha256": actual_input_digest},
        "host_campaign": {"path": str(host.source_path) if host.source_path else None,
                          "status": host.status, "freeze": dict(host.freeze),
                          "preflight": host_preflight.to_dict()},
        "model2mlir": {"path": str(model2mlir), "source_sha256": source_digest,
                       **_git_provenance(model2mlir)},
    }
    return list(dict.fromkeys(errors)), evidence, record


def _tasks(study: PaperStudySpec, record: dict[str, Any], paper_inputs: Path,
           model2mlir: Path, product: ProductDir) -> tuple[list[CaptureTask], list[str]]:
    tasks: list[CaptureTask] = []
    errors: list[str] = []
    script = (model2mlir / "workloads" / "capture_consistent.py").resolve()
    records = record.get("models") if isinstance(record.get("models"), dict) else {}
    for model in study.models:
        try:
            exact_environment = _render_environment(records.get(model.name, {}), paper_inputs)
        except ValueError as exc:
            errors.append(f"{model.name}: {exc}")
            continue
        python = _capture_python(model2mlir, model.capture)
        if not python.is_file():
            errors.append(f"{model.name}: capture interpreter is absent: {python}")
        loader = model2mlir / "workloads" / model.capture / "loader.py"
        loader_digest = _file_sha256(loader) if loader.is_file() else "unresolved"
        for precision in model.precisions:
            artifact = model.artifacts[precision]
            variant = str(artifact["variant"])
            fmt = _FORMATS.get(precision)
            if fmt is None or variant != fmt:
                errors.append(
                    f"{model.name}/{precision}: paper precision must map exactly to "
                    f"capture format {fmt!r}, got variant {variant!r}")
                continue
            output = product.path / "bundles" / f"{model.capture}_{variant}_full"
            command = (str(python), str(script), model.capture, fmt, str(output))
            tasks.append(CaptureTask(
                model=model, precision=precision, variant=variant, fmt=fmt,
                workload=model.capture, python=python, output=output, command=command,
                environment=exact_environment, loader_sha256=loader_digest))
    expected = sum(len(model.precisions) for model in study.models)
    if len(tasks) != expected:
        errors.append(f"capture plan has {len(tasks)} tasks, expected exactly {expected}")
    if len({str(task.output) for task in tasks}) != len(tasks):
        errors.append("capture plan contains duplicate output paths")
    return tasks, list(dict.fromkeys(errors))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_output(task: CaptureTask, source: dict[str, Any], elapsed_ns: int) -> dict[str, Any]:
    sidecar = {
        "version": 1, "model": task.model.name, "capture": task.model.capture,
        "precision": task.precision, "variant": task.variant,
        "command": list(task.command), "command_sha256": _json_sha256(list(task.command)),
        "environment": dict(sorted(task.environment.items())),
        "environment_sha256": _json_sha256(task.environment),
        "expected_provenance": dict(task.model.expected_provenance),
        "model2mlir": source, "elapsed_ns": elapsed_ns,
    }
    _write_json(task.output / "paper_capture_provenance.json", sidecar)
    capture_bundle.CaptureBundle(
        model=task.model.capture, variant=task.variant, root=task.output).require()
    session, session_errors = validate_capture_session(
        task.output, task.model.session,
        expected_provenance=task.model.expected_provenance)
    provenance = session.get("provenance", {}) if isinstance(session, dict) else {}
    if isinstance(provenance, dict) and provenance.get("checkpoint") != task.model.checkpoint:
        session_errors.append(
            f"session checkpoint differs: study={task.model.checkpoint!r} "
            f"capture={provenance.get('checkpoint')!r}")
    if session_errors:
        raise ValueError("; ".join(session_errors))
    from .paper_measurement_freeze import write_capture_measurement_source_receipt
    source_receipt = write_capture_measurement_source_receipt(
        task.output, model=task.model.name, precision=task.precision,
        observations=task.model.session.observations)
    return {"path": str(task.output.resolve()), "sha256": sha256_paths([task.output]),
            "session_kind": session.get("kind"), "paper_ready": session.get("paper_ready"),
            "measurement_source_receipt": {"path": str(source_receipt),
                                            "sha256": _file_sha256(source_receipt)}}


def _default_runner(task: CaptureTask, environment: dict[str, str],
                    stdout_path: Path, stderr_path: Path) -> int:
    with stdout_path.open("w", encoding="utf-8") as stdout, \
            stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(
            list(task.command), cwd=Path(task.command[1]).parent.parent,
            env=environment, stdout=stdout, stderr=stderr)
    return int(proc.returncode)


def materialize(study_path: str | Path, host_experiment_path: str | Path,
                model2mlir_root: str | Path, *, execute: bool = False,
                runner: Callable[[CaptureTask, dict[str, str], Path, Path], int] | None = None,
                product: ProductDir | None = None) -> Path:
    """Plan or execute the complete five-model/two-precision paper capture set.

    A plan always writes a timestamped evidence product. Execution writes ``staged-study.yaml`` only
    after every task returns successfully and passes the semantic session/provenance validator.
    """
    study_path = Path(study_path).resolve()
    host_experiment_path = Path(host_experiment_path).resolve()
    model2mlir = Path(model2mlir_root).resolve()
    study = PaperStudySpec.from_yaml(study_path)
    host = HostExperimentSpec.from_yaml(host_experiment_path)
    product = product or new_product(
        "paper-captures", version=1, target=study.target,
        sources=[str(study_path), str(host_experiment_path), str(model2mlir)])
    plan_path = product.add_artifact("capture-plan.json")
    paper_inputs = _resolve_paper_inputs(study)
    errors, evidence, record = _preflight(study, host, model2mlir, paper_inputs)
    tasks, task_errors = _tasks(study, record, paper_inputs, model2mlir, product)
    errors.extend(task_errors)
    started_at = _utc_now()
    plan: dict[str, Any] = {
        "version": 1, "mode": "execute" if execute else "preflight",
        "status": "blocked" if errors else "ready", "started_at": started_at,
        "study": {"path": str(study_path), "sha256": _file_sha256(study_path),
                  "status": study.status, "holdout_models": list(study.holdout_models)},
        "host_experiment": {"path": str(host_experiment_path),
                            "sha256": _file_sha256(host_experiment_path)},
        "evidence": evidence, "environment_policy": {
            "inherited_prefixes_removed": list(_ENV_PREFIXES),
            "exact_model_environment_source": str(paper_inputs / "paper_inputs.json")},
        "tasks": [task.to_dict() for task in tasks], "errors": errors,
    }
    _write_json(plan_path, plan)
    product.notes = f"paper capture {plan['status']}; execute={execute}; tasks={len(tasks)}"
    product.write_manifest()
    if errors:
        raise CaptureWorkflowNotReady(errors, product.path)
    if not execute:
        return product.path

    run_task = runner or _default_runner
    source = dict(evidence["model2mlir"])
    results: list[dict[str, Any]] = []
    total_start_ns = time.monotonic_ns()
    failure: str | None = None
    for task in tasks:
        log_base = f"logs/{task.model.name}/{task.variant}"
        stdout_path = product.add_artifact(log_base + ".stdout.log")
        stderr_path = product.add_artifact(log_base + ".stderr.log")
        task.output.parent.mkdir(parents=True, exist_ok=True)
        if task.output.exists():
            failure = f"refusing to overwrite capture output: {task.output}"
            break
        task_start_wall = _utc_now()
        task_start_ns = time.monotonic_ns()
        returncode = run_task(
            task, _sanitized_environment(task.environment), stdout_path, stderr_path)
        elapsed_ns = time.monotonic_ns() - task_start_ns
        result = {**task.to_dict(), "started_at": task_start_wall,
                  "finished_at": _utc_now(), "elapsed_ns": elapsed_ns,
                  "returncode": returncode, "stdout": str(stdout_path),
                  "stderr": str(stderr_path)}
        if returncode:
            result["status"] = "failed"
            failure = f"{task.model.name}/{task.precision}: capture command returned {returncode}"
        else:
            try:
                result.update(_validate_output(task, source, elapsed_ns))
                result["status"] = "validated"
            except (OSError, ValueError) as exc:
                result["status"] = "rejected"
                result["validation_error"] = str(exc)
                failure = f"{task.model.name}/{task.precision}: {exc}"
        results.append(result)
        if failure:
            break

    plan["results"] = results
    plan["finished_at"] = _utc_now()
    plan["elapsed_ns"] = time.monotonic_ns() - total_start_ns
    if failure or len(results) != len(tasks):
        plan["status"] = "failed"
        plan["errors"] = [failure or "capture set ended before every task completed"]
        _write_json(plan_path, plan)
        product.notes = f"paper capture failed after {len(results)}/{len(tasks)} tasks"
        product.write_manifest()
        raise CaptureWorkflowNotReady(plan["errors"], product.path)

    registered = {(result["model"], result["precision"]): result for result in results}
    staged = copy.deepcopy(study.canonical_dict())
    for model in staged["models"]:
        for precision, artifact in model["artifacts"].items():
            result = registered[(model["name"], precision)]
            artifact["path"] = result["path"]
            artifact["sha256"] = result["sha256"]
    staged_path = product.add_artifact("staged-study.yaml")
    write_yaml(staged_path, staged, header=(
        "Capture-complete draft; pass this file to merlin-compare --freeze. Do not tune on captures"))
    registration_path = product.add_artifact("capture-registration.json")
    _write_json(registration_path, {
        "version": 1, "complete": True, "study": str(staged_path),
        "study_sha256": _file_sha256(staged_path),
        "paper_inputs_sha256": study.paper_inputs["sha256"],
        "host_campaign_freeze": dict(host.freeze), "captures": results,
    })
    plan["status"] = "complete"
    plan["staged_study"] = str(staged_path)
    plan["capture_registration"] = str(registration_path)
    _write_json(plan_path, plan)
    product.notes = f"complete paper capture set; validated={len(results)}/{len(tasks)}"
    product.write_manifest()
    return product.path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="merlin-paper-capture",
        description="Preflight or materialize the complete frozen-compiler paper capture set.")
    parser.add_argument(
        "--study", type=Path, default=bench_dir() / "rvv_paper" / "study_v2.yaml")
    parser.add_argument(
        "--host-experiment", type=Path, required=True,
        help="completed, output-sealed CPU-host campaign YAML")
    parser.add_argument("--model2mlir", type=Path, default=capture_bundle.model2mlir_root())
    parser.add_argument(
        "--execute", action="store_true",
        help="run all ten heavy captures; without this flag only a timestamped preflight is written")
    args = parser.parse_args(argv)
    try:
        output = materialize(
            args.study, args.host_experiment, args.model2mlir, execute=args.execute)
    except CaptureWorkflowNotReady as exc:
        print(f"merlin-paper-capture: BLOCKED — {exc}")
        print(f"  {exc.output_dir / 'capture-plan.json'}")
        return 2
    print(f"merlin-paper-capture: wrote {output}")
    print(f"  {output / 'capture-plan.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
