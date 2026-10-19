"""Operator-only fresh launch from one config; reject incomplete input preparation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from merlin_experiments.phase1 import launch_config as L

from merlin.model_check import bundle as B
from merlin.model_check import seal as S


def save(path: Path, value: object) -> None:
    staging = path.with_suffix(path.suffix + ".next")
    staging.write_text(json.dumps(value, indent=2) + "\n")
    staging.replace(path)


def operator_module(name: str):
    """Load the selected sibling bytes even under Python's safe-path startup."""
    source = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ValueError(f"operator helper is unavailable: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def memory(records: dict[int, tuple[int, int]]) -> tuple[int, int]:
    charged = 0
    for pid in records:
        try:
            rows = (Path("/proc") / str(pid) / "smaps_rollup").read_text().splitlines()
            charged += next(int(row.split()[1]) * 1024 for row in rows if row.startswith("Pss:"))
        except (OSError, ValueError, StopIteration):
            # Match the campaign guard: inaccessible PSS is charged at full RSS.
            try:
                charged += int((Path("/proc") / str(pid) / "statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
            except (OSError, ValueError, IndexError):
                pass
    available = next(
        int(row.split()[1]) * 1024
        for row in Path("/proc/meminfo").read_text().splitlines()
        if row.startswith("MemAvailable:")
    )
    return charged, available


@contextmanager
def configured_environment(launch: L.Launch):
    """Use the selected child runtime while inspecting its readiness evidence."""
    before = dict(os.environ)
    selected = {key: value for key, value in before.items() if not key.startswith("MERLIN_")}
    for name, value in {**launch.environment, **launch.launch_environment}.items():
        selected[name] = (
            value.prefix + (":" + selected[name] if selected.get(name) else "")
            if isinstance(value, L.Prepend)
            else value
        )
    selected["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        os.environ.clear()
        os.environ.update(selected)
        yield
    finally:
        os.environ.clear()
        os.environ.update(before)


def prepare(launch: L.Launch, prep: Path) -> dict:
    with configured_environment(launch):
        return _prepare(launch, prep)


def _source_preflight_inputs(launch):
    """Reopen the gate's selected source inputs and compiler-owned index premise."""
    import yaml
    from merlin_experiments.phase1.feedback import private_control_support as control
    from merlin_experiments.phase1.feedback import private_full_model_execution as execution
    from merlin_experiments.phase1.feedback import private_full_models as models

    from merlin.compile.model_execution_inputs import file_sha256, strict_tree_sha256

    spec = launch.path(launch.document["inputs"]["private_full_model_spec"])
    descriptor = launch.path(launch.document["inputs"]["descriptor"])
    target = launch.document["identity"]["target"]
    required = models.program_requirements_for(descriptor)
    if not required or tuple(required) != models.requirements_for(descriptor):
        raise ValueError("source preparation requires the complete descriptor model/program roster")
    document = yaml.safe_load(spec.read_text())
    gate = execution.gate_for(descriptor, required_programs=required)
    options = execution.build_options(gate) if gate is not None else None
    pins, observations = {}, {}

    def path(value):
        selected = Path(value)
        selected = selected if selected.is_absolute() else spec.parent / selected
        if selected.resolve() != selected.absolute():
            raise ValueError("source preparation input has an indirect path")
        return selected

    for row in document["models"]:
        name = row["id"]
        for key, value in row.items():
            if key.endswith("_tree_sha256"):
                selected = path(row[key.removesuffix("_tree_sha256")])
                digest = strict_tree_sha256(selected)["sha256"]
            elif key.endswith("_sha256") and key.removesuffix("_sha256") in row:
                selected = path(row[key.removesuffix("_sha256")])
                if selected.is_symlink() or not selected.is_file():
                    raise ValueError("source preparation input is absent or indirect")
                digest = file_sha256(selected)
            else:
                continue
            if digest != value:
                raise ValueError(f"source preparation input differs: {name}/{key}")
            pins[str(selected.absolute())] = digest
        archive = path(row["recipe_derivation_root"])
        pins[str(archive.absolute())] = strict_tree_sha256(archive)["sha256"]
        observations[name] = {
            program: control.selected_build_observation(
                path(row["host_package"]),
                path(row["board_catalog"]),
                ((options or {}).get(name, {}).get(program) or {}).get("board", row["board"]),
                target,
            )
            for program in required.get(name, ())
        }
    argv = list(launch.argv)
    flag = "--native-result-format-selection"
    if argv.count(flag) > 1 or (flag in argv and argv.index(flag) + 1 == len(argv)):
        raise ValueError("source preparation has an ambiguous original result-format selection")
    native = Path(argv[argv.index(flag) + 1]) if flag in argv else None
    if native is not None:
        if native.is_symlink() or not native.is_file() or native.resolve() != native:
            raise ValueError("original result-format selection is absent or indirect")
        pins[str(native.absolute())] = file_sha256(native)
    identity = {
        "selected": operator_module("full_model_verification").selected_identity(launch),
        "operator_source_sha256": file_sha256(Path(__file__)),
        "descriptor_sha256": file_sha256(descriptor),
        "spec_sha256": file_sha256(spec),
        "target": target,
        "programs": {name: list(programs) for name, programs in required.items()},
        "input_pins": pins,
        "build_options": options,
        "index_observations": observations,
        "native_result_format_selection": str(native.absolute()) if native is not None else None,
    }
    return identity, native


def source_preparation(launch, prep: Path) -> dict:
    """Reuse only this preparation's exact source-only check, never candidate credit."""
    import hashlib

    from merlin_experiments.phase1.feedback import private_full_models as models
    from merlin_experiments.phase1.feedback import private_source_freeze as freeze
    from merlin_experiments.phase1.feedback import private_source_preflight as preflight

    from merlin.compile.model_execution_inputs import file_sha256, strict_tree_sha256

    identity, native = _source_preflight_inputs(launch)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    root = prep / "source-preflight"
    if root.is_symlink():
        raise ValueError("source preparation directory is indirect")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory, cached = root / key, root / (key + ".json")
    reused = cached.exists()
    if reused:
        if cached.is_symlink():
            raise ValueError("source preparation cache is indirect")
        record = json.loads(cached.read_text())
        if record.get("identity") != identity or record.get("output_tree") != strict_tree_sha256(directory):
            raise ValueError("prepared source receipt or frozen source bytes changed")
    else:
        preflight.preflight(
            launch.path(launch.document["inputs"]["private_full_model_spec"]),
            descriptor=launch.path(launch.document["inputs"]["descriptor"]),
            out=directory,
            target=identity["target"],
            native_result_format_selection=native,
        )
        if _source_preflight_inputs(launch)[0] != identity:
            raise ValueError("source preparation inputs changed during preflight")
        record = {"identity": identity, "output_tree": strict_tree_sha256(directory)}
        save(cached, record)
        cached.chmod(0o600)
    report_path = directory / preflight.REPORT
    report = json.loads(report_path.read_text())
    rows = report.get("models") or []
    expected = identity["programs"]
    if (
        report.get("mode") != "source_preflight"
        or report.get("source_preflight_passed") is not True
        or report.get("passed") is not False
        or report.get("candidate_tree_sha256") is not None
        or report.get("target") != identity["target"]
        or report.get("scope") != models.SOURCE_PREFLIGHT_SCOPE
        or report.get("private_spec_sha256") != identity["spec_sha256"]
        or report.get("required_models") != list(expected)
        or report.get("required_programs") != expected
        or len(rows) != len(expected)
        or {row.get("model") for row in rows} != set(expected)
        or any(
            row.get("status") != "source_inputs_verified"
            or set((row.get("checks") or {}).get("source") or {}) != set(expected[row["model"]])
            for row in rows
        )
        or not freeze.valid_binding(report.get("authored_source_freeze"), identity["spec_sha256"])
    ):
        raise ValueError(f"compiler-independent private source preflight refused; see {report_path}")
    return {"source_preflight_passed": True, "candidate_qualified": False, "reused": reused,
            "report": {"path": str(report_path), "sha256": file_sha256(report_path)}, "identity_sha256": key}


def _prepare(launch: L.Launch, prep: Path) -> dict:
    doc = launch.document
    origin = doc["origin"]
    if (
        origin["seed_submission"] is not None
        or origin["resume"]
        or not origin["initial_submission"].startswith("empty")
    ):
        raise ValueError("this launcher requires an empty submission with no seed or resume")
    repo = launch.path(doc["inputs"]["repo"])
    baseline = launch.variables["SOURCE_REVISION"]
    # Operator metadata can be committed without changing the selected compiler
    # harness. Comparing the tree avoids an impossible self-referential HEAD pin
    # in the tracked config itself; the launcher bytes have their own hash pin.
    scope = ["--", ".", ":(exclude)examples/gemmini/phase1/launch"]
    ancestry = subprocess.run(["git", "merge-base", "--is-ancestor", baseline, "HEAD"], cwd=repo)
    changed = subprocess.run(["git", "diff", "--quiet", baseline, *scope], cwd=repo)
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", *scope], cwd=repo, text=True
    )
    if ancestry.returncode or changed.returncode or untracked.strip():
        raise ValueError(f"source tree differs from selected revision {baseline}")
    required = {"private_bundle_manifest", "private_bundle_sums"}
    if not required <= set(doc.get("runtime_pins") or {}):
        raise ValueError(
            "private model inputs are not ready: config lacks pinned private_bundle_manifest/private_bundle_sums; "
            "finish and verify input preparation first"
        )
    selected = Path(str(launch.environment["MERLIN_MODEL_CHECK_BUNDLE"]))
    for key, name in [("private_bundle_manifest", "MANIFEST.json"), ("private_bundle_sums", "SHA256SUMS")]:
        if launch.path(doc["runtime_pins"][key]["path"]).resolve() != (selected / name).resolve():
            raise ValueError(f"{key} does not select the configured private bundle")
    private_spec = launch.path(doc["inputs"]["private_full_model_spec"])
    import yaml

    spec = yaml.safe_load(private_spec.read_text())
    if Path(spec["numerics_bundle"]["path"]).resolve() != selected.resolve():
        raise ValueError("full-model selection and configured private input bundle differ")
    wanted = {row["id"] for row in spec["models"]}
    manifest = B.Bundle(selected).manifest()
    if wanted != set(manifest.get("models") or {}):
        raise ValueError("private bundle does not contain exactly the selected full models")
    out = launch.path(doc["launcher"]["environment_file"]).parent
    if launch.path(doc["launcher"]["plan"]).parent != out:
        raise ValueError("rendered environment and plan must use one directory")
    errata = doc.get("errata")
    if errata:
        content = (launch.config.path.parent / errata["source"]).read_bytes()
        target = launch.path(errata["path"])
        if target.exists() and target.read_bytes() != content:
            raise ValueError("existing served errata differs; use a fresh preparation directory")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    L.render(launch, out)
    plan = launch.path(doc["launcher"]["plan"])
    desired = L.launch_plan(launch, {})
    if plan.exists() and json.loads(plan.read_text()) != desired:
        raise ValueError("existing operator plan differs; use a fresh preparation directory")
    save(plan, desired)
    report = L.verify(launch)
    record = L.report_record(launch, report)
    save(out / "config-verification.json", record)
    if report.drifts:
        raise ValueError(f"launch config drift: see {out / 'config-verification.json'}")
    # This checks every selected byte, not just that a manifest exists. Never seal
    # or repair here: incomplete data must stop before any author is started.
    verified = S.verify(B.Bundle(selected), jobs=min(8, len(launch.manager_cpus)))
    save(out / "private-input-verification.json", verified)
    if verified["status"] != "pass":
        raise ValueError("private full-model input seal failed verification")
    if verified["manifest_sha256"] != doc["runtime_pins"]["private_bundle_manifest"]["sha256"]:
        raise ValueError("private input manifest changed during preparation")
    references = {int(cpu) for cpu in launch.variables.get("REFERENCE_CPUS", "").split(",") if cpu.strip()}
    author = set(doc["cpu"]["author"]["cpus"])
    compute = set(doc["cpu"]["compute_pool"]["cpus"])
    if (references & author) or (references & compute) or (author & compute):
        raise ValueError("reference, author and compute CPU assignments overlap")
    assigned = references | author | compute
    if len(assigned) != 18 or not assigned <= os.sched_getaffinity(0):
        raise ValueError("the full selected 18-CPU allocation is unavailable")
    source_readiness = source_preparation(launch, prep)
    readiness = operator_module("full_model_verification").validate_readiness(launch)
    return {
        "status": "ready",
        "assigned_cpus": sorted(assigned),
        "config": record["config"],
        "private_inputs": verified,
        "private_source_preparation": source_readiness,
        "full_model_readiness": readiness,
        "readiness_scope": "infrastructure_preflight"
        if readiness and readiness.get("schema") == "merlin.phase1.infrastructure_readiness.v2"
        else "retained_full_model_qualification" if readiness else "launch_configuration",
        "phase1_complete": False,
    }


def run(launch: L.Launch, prep: Path, started: float) -> int:
    with configured_environment(launch):
        operator_module("full_model_verification")
        return operator_module("launch_supervisor").run(launch, prep, started, memory=memory)


def main() -> int:
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("phase1_el4_config.yaml"))
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--var", action="append", default=[])
    parser.add_argument(
        "--run",
        action="store_true",
        help="start a fresh native controller after every readiness check passes; default checks only",
    )
    args = parser.parse_args()
    try:
        overrides = L._overrides(args.var)
        launch = L.build(
            L.load(args.config), task_root=args.task_root, prep=args.prep, run_id=args.run_id, overrides=overrides
        )
        repo = launch.path(launch.document["inputs"]["repo"])
        if args.run and (repo / "out/runs" / launch.document["identity"]["target"] / "phase1" / args.run_id).exists():
            raise ValueError("run ID already exists; fresh launches require a new ID")
        args.prep.mkdir(parents=True, exist_ok=True)
        ready = prepare(launch, args.prep)
        save(args.prep / "launch-readiness.json", ready)
        print(json.dumps({"status": "ready", "readiness_scope": ready["readiness_scope"], "phase1_complete": False,
                          "config": str(launch.config.path), "run_id": args.run_id}), flush=True)
        return run(launch, args.prep, started) if args.run else 0
    except (ValueError, OSError, B.BundleError, RuntimeError) as error:
        print(json.dumps({"status": "refused", "error": str(error)}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
