"""Freeze and execute process plans while leaving experiment evidence with its engine."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import signal
import subprocess
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from yaml import YAMLError

from .adapters import ADAPTERS, PHASE0_MODULE, PHASE1_MODULE, phase0_sources, phase0_startup_inputs, phase1_entrypoint
from .phase1.source_inputs import fingerprint
from .spec import ExperimentSpec, SpecError, read_yaml


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def default_run_dir(spec: ExperimentSpec) -> Path:
    from merlin.common.paths import runs_dir

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return runs_dir() / spec.target / spec.id / f"{stamp}_{uuid4().hex[:8]}"


def _corpus_closure(command: dict) -> dict:
    """Discover native grading roots without importing or executing a grader.

    Pin category membership as well as bytes: hashing only today's directories
    would miss a newly added sibling category or the appearance of hidden inputs.
    Records disclose category roots, never hidden capsule names or their contents.
    Native bundle snapshots and cohort admission remain the grading authority.
    """
    from merlin.common.paths import repo_root
    from merlin.targetgen.target_experiment import load_target_experiment

    expected_root = Path(command["env"]["MERLIN_REPO_ROOT"]).resolve()
    if repo_root().resolve() != expected_root:
        raise SpecError("corpus repository root changed; restore the frozen MERLIN_REPO_ROOT to resume")
    descriptor = command["inputs"].get("descriptor")
    if not descriptor:
        raise SpecError("capsule_bench requires a descriptor-owned corpus closure")
    try:
        target = load_target_experiment(descriptor)
        if target.capsule_corpus is None:
            raise ValueError("descriptor does not declare capsule_corpus")
        return {
            "graded": [str(path.resolve()) for path in target.graded_roots()],
            "hidden": [str(path.resolve()) for path in target.hidden_roots()],
        }
    except (OSError, ValueError, TypeError, AttributeError, YAMLError) as exc:
        raise SpecError(f"cannot discover descriptor-owned corpus closure: {exc}") from exc


def _verify_corpus_closures(plan: dict) -> None:
    for number, command in plan["phases"].items():
        if command["adapter"] != "capsule_bench":
            continue
        frozen = plan.get("corpus_closures", {}).get(number)
        if frozen is None:
            raise SpecError("capsule_bench plan has no frozen corpus closure; create a new experiment run")
        if _corpus_closure(command) != frozen:
            raise SpecError("frozen corpus category membership changed; create a new experiment run")
        seal = command["inputs"].get("corpus_seal")
        if command.get("requires_reviewed_corpus") and not seal:
            raise SpecError(
                "this functional experiment requires a reviewed Phase 0 release; "
                "select its seal and release descriptor before starting Phase 1"
            )
        if seal:
            from .corpus.release import verify

            if command["env"].get("MERLIN_CORPUS_SEAL") != seal:
                raise SpecError("native corpus seal differs from the explicitly selected input")
            verify(Path(seal), Path(command["inputs"]["descriptor"]))
            if command.get("requires_reviewed_corpus"):
                from merlin.targetgen.sandbox.bwrap import resolve_grant

                source_root = Path(command["env"]["MERLIN_REPO_ROOT"]).resolve()
                legacy = source_root / "merlin" / "contract" / "capsules"
                bundle = read_yaml(Path(command["inputs"]["bundle_manifest"]))
                for kind in ("allowed", "host_inputs"):
                    for row in bundle.get(kind) or []:
                        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
                            raise SpecError(f"reviewed bundle has an invalid {kind} grant")
                        grant = resolve_grant(row["path"], source_root).absolute()
                        if grant == legacy or grant.is_relative_to(legacy) or legacy.is_relative_to(grant):
                            raise SpecError(
                                "reviewed Phase 1 bundle still grants the historical in-tree capsule corpus; "
                                "generate a bundle for the sealed release"
                            )


def _phase0_source_inputs() -> tuple[Path, dict[str, str]]:
    entrypoint, sources = phase0_sources()
    inputs = {f"phase0:source:{path.relative_to(entrypoint.parent)}": str(path) for path in sources}
    inputs.update({f"phase0:startup:{name}": str(path) for name, path in phase0_startup_inputs().items()})
    return entrypoint, inputs


def _verify_phase0_sources(plan: dict) -> None:
    """Bind module resolution and complete source membership before execution/resume."""
    for command in plan["phases"].values():
        if command["adapter"] != "capsule_derivation" or not command.get("module"):
            continue  # Historical script receipts keep their original validation.
        if "recipe" in command["inputs"] or "phase0_operator_inputs" in plan:
            observed = _phase0_operator_inputs(command)
            if plan.get("phase0_operator_inputs") != observed:
                raise SpecError("phase-0 operator input membership changed or is missing")
            expected_paths = _phase0_input_paths(observed)
            actual = {name: path for name, path in plan["input_paths"].items() if name.startswith("phase0:operator:")}
            if actual != expected_paths:
                raise SpecError("phase-0 operator inputs lack their frozen closure")
            if "inputs" in plan and any(
                plan["inputs"].get(name, {}).get("path") != path for name, path in expected_paths.items()
            ):
                raise SpecError("phase-0 operator inputs lack their frozen fingerprints")
        if command["module"] != PHASE0_MODULE or command["argv"][1:3] != ["-m", PHASE0_MODULE]:
            raise SpecError("phase-0 module binding is not the supported derivation implementation")
        entrypoint, expected = _phase0_source_inputs()
        if command["env"].get("PYTHONSAFEPATH") != "1" or command["env"].get("PYTHONPATH", "").split(os.pathsep)[
            0
        ] != str(entrypoint.parent.parent.parent):
            raise SpecError("phase-0 process import path differs from its pinned implementation")
        actual = {
            name: path
            for name, path in plan["input_paths"].items()
            if name.startswith(("phase0:source:", "phase0:startup:"))
        }
        if actual != expected or command["entrypoint"] != str(entrypoint):
            raise SpecError("phase-0 implementation source closure changed; create a new experiment run")
        if "inputs" in plan:
            for name, path in expected.items():
                if plan["inputs"].get(name, {}).get("path") != path:
                    raise SpecError("phase-0 implementation lacks its frozen source closure")


_PHASE0_OPTIONAL_INPUTS = frozenset({"synth_profile", "smt_profile", "hidden_profile"})


def _phase0_input_paths(membership: dict) -> dict[str, str]:
    return {
        name: record["path"]
        for name, record in membership.items()
        if record is not None
        and (record["present"] or name in ("phase0:operator:recipe", "phase0:operator:performance_template"))
    }


def _phase0_operator_inputs(command: dict) -> dict[str, dict | None]:
    """Bind explicit file membership, including deliberately absent sidecars."""
    inputs = command["inputs"]
    argv = command["argv"]
    if "profiles_root" in inputs or any(arg.split("=", 1)[0] == "--profiles-root" for arg in argv):
        raise SpecError("explicit phase-0 inputs cannot use profiles_root")
    observed = {}
    for name in ("recipe", "performance_template", *sorted(_PHASE0_OPTIONAL_INPUTS)):
        flag = "--" + name.replace("_", "-")
        if any(arg.startswith(flag + "=") for arg in argv):
            raise SpecError(f"phase-0 command {flag} must use its frozen explicit argument")
        path = inputs.get(name)
        if path is None:
            if name not in _PHASE0_OPTIONAL_INPUTS or flag in command["argv"]:
                raise SpecError(f"phase-0 command {flag} lacks its declared input")
            observed["phase0:operator:" + name] = None
            continue
        if _command_value(command, flag) != path:
            raise SpecError(f"phase-0 command {flag} differs from its frozen input")
        source = Path(path)
        if source.exists() or source.is_symlink():
            if not source.is_file():
                raise SpecError(f"phase-0 input is not a file: {path}")
        observed["phase0:operator:" + name] = {
            "path": str(source.resolve()),
            "present": source.exists() or source.is_symlink(),
        }
    return observed


def _phase1_source_inputs(command: dict) -> dict[str, str]:
    from .phase1.source_inputs import paths

    return paths(
        repo=Path(command["env"]["MERLIN_REPO_ROOT"]),
        entrypoint=Path(command["entrypoint"]),
        descriptor=Path(command["inputs"]["descriptor"]) if command["inputs"].get("descriptor") else None,
    )


def _verify_phase1_sources(plan: dict) -> None:
    from .phase1.source_inputs import PREFIXES

    commands = [command for command in plan["phases"].values() if command["adapter"] == "capsule_bench"]
    if not commands:
        return
    for command in commands:
        if command.get("module") is None:
            if command["argv"][1:2] != [command["entrypoint"]]:
                raise SpecError("historical phase-1 command differs from its recorded script entrypoint")
            continue  # Stored native plans retain their original executable and source closure.
        entrypoint = phase1_entrypoint()
        if (
            command["module"] != PHASE1_MODULE
            or command["argv"][1:3] != ["-m", PHASE1_MODULE]
            or command["entrypoint"] != str(entrypoint)
            or command["env"].get("PYTHONSAFEPATH") != "1"
            or command["env"].get("PYTHONPATH", "").split(os.pathsep)[0] != str(entrypoint.parent.parent.parent)
        ):
            raise SpecError("phase-1 installed command differs from its pinned implementation")
        for flag, value in {
            "--repo": command["env"]["MERLIN_REPO_ROOT"],
            "--descriptor": command["inputs"].get("descriptor"),
            "--bundle-manifest": command["inputs"].get("bundle_manifest"),
            "--oracle-timing": command["inputs"].get("oracle_timing"),
        }.items():
            if _command_value(command, flag) != value:
                raise SpecError(f"phase-1 command {flag} differs from its frozen input")
        config = plan["spec"]["phases"]["1"]["config"]
        for name in ("bundle", "arm", "treatment"):
            flag = "--" + name
            # Older installed baseline plans omitted the default treatment flag.
            if name == "treatment" and flag not in command["argv"]:
                selected = "baseline"
            else:
                selected = _command_value(command, flag)
            if selected != config.get(name, "baseline" if name == "treatment" else None):
                raise SpecError(f"phase-1 command {flag} differs from its frozen definition")
        observed = _phase1_operator_inputs(command)
        if plan.get("phase1_operator_inputs") != observed:
            raise SpecError("phase-1 operator input membership changed or is missing")
        expected_paths = {name: path for name, path in observed.items() if path is not None}
        actual_paths = {name: path for name, path in plan["input_paths"].items() if name.startswith("phase1:operator:")}
        if expected_paths != actual_paths:
            raise SpecError("phase-1 operator inputs lack their frozen closure")
        if "inputs" in plan and any(
            plan["inputs"].get(name, {}).get("path") != path for name, path in expected_paths.items()
        ):
            raise SpecError("phase-1 operator inputs lack their frozen fingerprints")
    expected = _phase1_source_inputs(commands[0])
    actual = {name: path for name, path in plan["input_paths"].items() if name.startswith(PREFIXES)}
    if actual != expected:
        raise SpecError(
            "phase-1 implementation or public client source closure changed or is missing; create a new experiment run"
        )
    if "inputs" in plan and any(plan["inputs"].get(name, {}).get("path") != path for name, path in expected.items()):
        raise SpecError("phase-1 implementation lacks its frozen source closure; create a new experiment run")


def _command_value(command: dict, flag: str) -> str:
    argv = command["argv"]
    if argv.count(flag) != 1 or argv.index(flag) + 1 >= len(argv):
        raise SpecError(f"installed Phase 1 requires one explicit {flag}")
    return argv[argv.index(flag) + 1]


def _phase1_operator_inputs(command: dict) -> dict[str, str | None]:
    """Rediscover authored resources and optional timing membership in the existing plan."""
    from merlin.targetgen.sandbox.bwrap import resolve_grant
    from merlin.targetgen.target_experiment import (
        declared_contracts_root,
        declared_resources_root,
        declared_task_root,
        resolve_resource_path,
    )

    root = Path(command["env"]["MERLIN_REPO_ROOT"])
    descriptor = Path(command["inputs"]["descriptor"])
    document = read_yaml(descriptor) if descriptor.is_file() else {}
    resources = declared_resources_root(document, root=root) or descriptor.parent
    manifest = Path(command["inputs"]["bundle_manifest"])
    if manifest.name != "input_bundle_manifest.yaml":
        raise SpecError("installed Phase 1 requires the canonical input_bundle_manifest.yaml")
    bundle = read_yaml(manifest) if manifest.is_file() else {}
    if manifest.is_file() and bundle.get("bundle_id") != _command_value(command, "--bundle"):
        raise SpecError("explicit bundle identity differs from its manifest")
    paths = {
        "bundle": manifest.parent,
        "contract": root / "merlin/contract",
        "schemas": root / "merlin/schemas",
        "oracle_timing": Path(command["inputs"]["oracle_timing"]),
    }
    hardware = document.get("hardware_spec") or {}
    harness = hardware.get("curated_harness")
    if harness:
        paths["curated_harness"] = resolve_resource_path(
            harness,
            resources=resources,
            task=declared_task_root(document, root=root),
            contracts=declared_contracts_root(document, root=root),
        )
    for kind in ("allowed", "host_inputs"):
        rows = bundle.get(kind, [])
        if not isinstance(rows, list) or any(
            not isinstance(row, dict) or not isinstance(row.get("path"), str) or not row["path"].strip() for row in rows
        ):
            raise SpecError(f"bundle {kind} must contain explicit path records")
        for index, row in enumerate(rows):
            paths[f"{kind}:{index}"] = resolve_grant(row["path"], root)
    result = {"phase1:operator:" + name: str(path.resolve()) for name, path in paths.items()}
    target = document.get("target", "")
    for name, path in {
        "task": declared_task_root(document, root=root) or resources / "task",
        "timing:target": resources / "scripts" / f".oracle_timing.{target}.json",
        "timing:legacy": resources / "scripts/.oracle_timing.json",
        "environment": resources / "experiment.env",
    }.items():
        # Record absence, not an invented empty file: later appearance changes the plan.
        result["phase1:operator:" + name] = str(path.resolve()) if path.exists() or path.is_symlink() else None
    return result


def resolve_plan(
    spec: ExperimentSpec,
    *,
    phase: str = "all",
    run_dir: Path | None = None,
    corpus_seal: Path | None = None,
    bundle_manifest: Path | None = None,
) -> dict:
    from merlin.common.paths import out_dir, repo_root

    root = repo_root().resolve()
    destination = (run_dir or default_run_dir(spec)).expanduser().resolve()
    phases = spec.document["phases"]
    selected = sorted(phases) if phase == "all" else [phase]
    if any(number not in phases for number in selected):
        raise SpecError(f"definition does not declare phase {phase}")
    if (corpus_seal is not None or bundle_manifest is not None) and selected != ["1"]:
        raise SpecError("a reviewed corpus and replacement bundle can only select Phase 1")
    if (corpus_seal is None) != (bundle_manifest is None):
        raise SpecError("select both --corpus-seal and --bundle-manifest for a new reviewed Phase 1 run")
    commands = {}
    corpus_closures = {}
    phase1_operator_inputs = None
    phase0_operator_inputs = None
    inputs = {f"declared:{name}": str(spec.resolve(value)) for name, value in spec.document.get("inputs", {}).items()}
    inputs["definition"] = str(spec.path)
    for number in selected:
        entry = phases[number]
        adapter = ADAPTERS[entry["adapter"]]
        config = dict(entry["config"])
        if number == "1" and corpus_seal is not None:
            selected_seal = corpus_seal.expanduser().absolute()
            if selected_seal.name != "seal.json" or selected_seal.parent.name != "private":
                raise SpecError("--corpus-seal must name a reviewed release's private/seal.json")
            config["corpus_seal"] = str(selected_seal)
            config["descriptor"] = str(
                selected_seal.parent.parent / "payload" / "experiment" / "target_experiment.yaml"
            )
            selected_bundle = bundle_manifest.expanduser().absolute()
            if selected_bundle.name != "input_bundle_manifest.yaml":
                raise SpecError("--bundle-manifest must name input_bundle_manifest.yaml")
            original_bundle = spec.resolve(entry["config"]["bundle_manifest"])
            if selected_bundle.resolve() == original_bundle:
                raise SpecError("a reviewed corpus needs a newly generated bundle, not the retained example manifest")
            config["bundle_manifest"] = str(selected_bundle)
            if selected_bundle.is_file():
                config["bundle"] = read_yaml(selected_bundle).get("bundle_id")
            adapter.validate(config)
        command = adapter.resolve(spec, config, root, destination)
        commands[number] = command
        inputs[f"phase{number}:entrypoint"] = command["entrypoint"]
        for name, value in command["inputs"].items():
            if adapter.name == "capsule_derivation" and "recipe" in command["inputs"]:
                if name in _PHASE0_OPTIONAL_INPUTS:
                    continue  # Membership and present bytes are pinned below, not absent paths.
            inputs[f"phase{number}:{name}"] = value
        if adapter.name == "capsule_derivation":
            if "recipe" in command["inputs"]:
                phase0_operator_inputs = _phase0_operator_inputs(command)
                inputs.update(_phase0_input_paths(phase0_operator_inputs))
            else:
                profiles = Path(command["inputs"].get("profiles_root", root / "merlin/contract/capsules/profiles"))
                inputs["phase0:profiles"] = str(profiles)
                profile = entry["config"].get("profile", spec.target)
                target_profile = profiles / f"{profile}.yaml"
                inputs["phase0:target_profile"] = str(target_profile)
            if command.get("module"):
                _, source_inputs = _phase0_source_inputs()
                inputs.update(source_inputs)
        if adapter.name == "capsule_bench":
            inputs.update(_phase1_source_inputs(command))
            if command.get("module") == PHASE1_MODULE:
                phase1_operator_inputs = _phase1_operator_inputs(command)
                inputs.update({name: path for name, path in phase1_operator_inputs.items() if path is not None})
        if adapter.name == "measured_claims" and command.get("module") and spec.document.get("kind") != "template":
            from .measured_launch import source_inputs

            inputs.update(source_inputs(command))
        if adapter.name == "model_portfolio" and command.get("module"):
            from .portfolio_catalog import source_inputs

            inputs.update(source_inputs(command))
        descriptor = command["inputs"].get("descriptor")
        if descriptor and Path(descriptor).is_file():
            actual = read_yaml(Path(descriptor)).get("target")
            if actual != spec.target:
                raise SpecError(f"descriptor target {actual!r} differs from definition target {spec.target!r}")
            if adapter.name == "capsule_bench":
                closure = _corpus_closure(command)
                corpus_closures[number] = closure
                for visibility, roots in closure.items():
                    for index, path in enumerate(roots):
                        inputs[f"phase{number}:corpus:{visibility}:{index}"] = path
    # No engine-owned output or editable workspace may overlap the immutable closure,
    # even when the orchestration directory itself lives elsewhere.
    mutable = [destination]
    for command in commands.values():
        if command.get("engine_output"):
            mutable.append(Path(command["engine_output"]).resolve())
        mutable.extend(Path(workspace).resolve() for workspace in command.get("workspaces", []))
        mutable.extend(Path(path).resolve() for path in command.get("mutable_outputs", []))
        if command.get("managed_endpoint"):
            mutable.append(Path(command["managed_endpoint"]).resolve())
    declared_optional_paths = [
        record["path"] for record in (phase0_operator_inputs or {}).values() if record is not None
    ]
    input_owner_roots = [path for command in commands.values() for path in command.get("input_owner_roots", [])]
    for value in [*inputs.values(), *declared_optional_paths, *input_owner_roots]:
        frozen_input = Path(value).resolve()
        for writable in mutable:
            if (
                writable == frozen_input
                or writable.is_relative_to(frozen_input)
                or frozen_input.is_relative_to(writable)
            ):
                raise SpecError(f"mutable output/workspace {writable} overlaps frozen input {frozen_input}")
    return {
        "schema_version": 1,
        **({"phase0_operator_inputs": phase0_operator_inputs} if phase0_operator_inputs is not None else {}),
        **({"phase1_operator_inputs": phase1_operator_inputs} if phase1_operator_inputs is not None else {}),
        "experiment": spec.id,
        "target": spec.target,
        "definition": str(spec.path),
        "spec": spec.document,
        "run_dir": str(destination),
        "storage_root": str(out_dir().resolve()),
        "phases": commands,
        "input_paths": inputs,
        "corpus_closures": corpus_closures,
        "evidence_authority": "legacy phase engines and AET; process completion is not a scientific verdict",
    }


def preflight(plan: dict) -> dict:
    """Check only definition, files, and process transport, without starting engines.

    Oracle qualification, sandbox admission, credentials, and hardware readiness
    remain the native engines' live gates. This must never report a hardware GO.
    """
    errors = []
    pins = {}
    try:
        _verify_corpus_closures(plan)
        _verify_phase0_sources(plan)
        _verify_phase1_sources(plan)
        from .measured_launch import verify_plan

        verify_plan(plan)
        from .portfolio_catalog import verify_plan as verify_portfolio_plan

        verify_portfolio_plan(plan)
    except SpecError as exc:
        errors.append(str(exc))
    adapters = {command["adapter"] for command in plan["phases"].values()}
    if {"capsule_derivation", "capsule_bench"} <= adapters:
        errors.append(
            "phase0 to phase1 handoff is not sealed: generated capsules are not the descriptor's "
            "frozen grading corpus; select --phase 0 or --phase 1 separately and explicitly review/seal "
            "the corpus before promoting it into a functional experiment"
        )
    for name, path in plan["input_paths"].items():
        try:
            pins[name] = {"path": path, "sha256": fingerprint(path)}
        except (OSError, SpecError) as exc:
            errors.append(str(exc))
    for command in plan["phases"].values():
        if not Path(command["argv"][0]).is_file():
            errors.append(f"Python interpreter missing: {command['argv'][0]}")
        if not Path(command["cwd"]).is_dir():
            errors.append(f"engine checkout missing: {command['cwd']}")
        for workspace in command.get("workspaces", []):
            if not Path(workspace).is_dir():
                errors.append(f"candidate workspace missing: {workspace}")
    return {"configuration_ready": not errors, "engine_readiness": "not_executed", "errors": errors, "inputs": pins}


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SpecError(f"invalid run record {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise SpecError(f"invalid run record {path}: expected an object")
    return document


@contextmanager
def _exclusive(run_dir: Path):
    with (run_dir / ".orchestration.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SpecError(f"an orchestrator already owns {run_dir}") from exc
        yield


def status(run_dir: Path) -> dict:
    root = run_dir.expanduser().resolve()
    plan = _read_json(root / "resolved-plan.json")
    record = _read_json(root / "orchestration.json")
    if _sha(root / "resolved-plan.json") != record.get("plan_sha256"):
        raise SpecError("frozen resolved plan changed")
    phases = {}
    for number, command in plan.get("phases", {}).items():
        attempts = [entry for entry in record["attempts"] if entry.get("phase") == number]
        latest = attempts[-1] if attempts else {}
        phases[number] = {
            "adapter": command["adapter"],
            "state": latest.get("state", "not_started"),
            "attempt_count": len(attempts),
            "engine_output": latest.get("engine_output", command.get("engine_output")),
            "latest_log": latest.get("log"),
            "resume_policy": command.get("resume_policy"),
        }
    return {
        "experiment": plan["experiment"],
        "target": plan["target"],
        "run_dir": str(root),
        "state": record["state"],
        "attempts": record["attempts"],
        "phases": phases,
        "evidence_authority": plan["evidence_authority"],
    }


def run(plan: dict) -> int:
    if plan["spec"].get("kind") == "template":
        raise SpecError(
            "template cannot execute: copy it, supply the required operator inputs, and set kind: experiment"
        )
    check = preflight(plan)
    if not check["configuration_ready"]:
        raise SpecError("preflight failed: " + "; ".join(check["errors"]))
    for command in plan["phases"].values():
        native = command.get("engine_output")
        if command["resume_policy"] == "native_flag" and native and Path(native).exists():
            raise SpecError(f"native engine run already exists: {native}; select a different run directory name")
    root = Path(plan["run_dir"])
    try:
        root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise SpecError(f"run directory already exists; use resume: {root}") from exc
    frozen = dict(plan, frozen_at=_now(), inputs=check["inputs"])
    _write_json(root / "resolved-plan.json", frozen)
    record = {"schema_version": 1, "state": "pending", "attempts": [], "plan_sha256": _sha(root / "resolved-plan.json")}
    _write_json(root / "orchestration.json", record)
    return _execute(root, frozen, record)


def resume(run_dir: Path, *, checkpoint: Path | None = None) -> int:
    root = run_dir.expanduser().resolve()
    status(root)  # Validate frozen plan binding before using any stored argv.
    plan = _read_json(root / "resolved-plan.json")
    record = _read_json(root / "orchestration.json")
    return _execute(root, plan, record, checkpoint=checkpoint)


def _verify_inputs(plan: dict) -> None:
    _verify_corpus_closures(plan)
    _verify_phase0_sources(plan)
    _verify_phase1_sources(plan)
    from .measured_launch import verify_plan

    verify_plan(plan)
    from .portfolio_catalog import verify_plan as verify_portfolio_plan

    verify_portfolio_plan(plan)
    for name, pin in plan["inputs"].items():
        if fingerprint(pin["path"]) != pin["sha256"]:
            raise SpecError(f"frozen input changed: {name} ({pin['path']}); create a new experiment definition/run")


def _process_active(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A reaped-by-another-supervisor zombie no longer owns any working process.
    stat = Path(f"/proc/{pid}/stat")
    try:
        return stat.read_text().rpartition(") ")[2].split()[0] != "Z"
    except (OSError, IndexError):
        return True


def _process_group_active(group: int) -> bool:
    """A driver exit is not proof that its workers exited. Unknown stays active."""
    processes = Path("/proc")
    if not processes.is_dir():
        return True
    try:
        for path in processes.iterdir():
            if not path.name.isdigit():
                continue
            try:
                fields = (path / "stat").read_text().rpartition(") ")[2].split()
                if int(fields[2]) == group and fields[0] != "Z":
                    return True
            except FileNotFoundError:
                continue  # a process exited during inspection
            except (OSError, IndexError, ValueError):
                return True
    except OSError:
        return True
    return False


def _stop_process_group(process: subprocess.Popen) -> None:
    """Best-effort shutdown, never proof that an abnormal run left no descendants."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # Workers may outlive their driver; always signal the whole private group.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass


@contextmanager
def _engine_ownership(root: Path, plan: dict):
    """Retain ownership after ANY ambiguous post-spawn exit, even after killpg.

    Process groups cannot prove that a descendant did not detach. Only explicit
    operator acknowledgement may release that abnormal ownership. Never let a
    generic exception-closing context manager declare this output abandoned.
    """
    from merlin.common.paths import out_dir
    from merlin.common.storage_lifecycle import acquire, inventory

    frozen_root = plan.get("storage_root")
    if not frozen_root or Path(frozen_root).resolve() != out_dir().resolve():
        raise SpecError("frozen storage root differs from MERLIN_OUT_ROOT; restore the original output root to resume")
    for command in plan["phases"].values():
        child_root = command.get("env", {}).get("MERLIN_OUT_ROOT")
        if not child_root or Path(child_root).resolve() != Path(frozen_root):
            raise SpecError("engine output root differs from the frozen storage ownership root")
    state = inventory()
    if state["status"] != "known":
        raise SpecError("storage ownership cannot be established")
    for row in state["paths"]:
        owned = Path(row["path"])
        if row["leases"] and (owned == root or owned in root.parents or root in owned.parents):
            raise SpecError(
                "unresolved engine ownership; verify all descendants stopped before acknowledging abandonment"
            )
    held = acquire(root, owner="merlin-experiments")
    ownership = {"unresolved": False, "failed": False}
    try:
        yield ownership
    except BaseException:
        if not ownership["unresolved"]:
            held.close("failed")
        raise
    else:
        if not ownership["unresolved"]:
            held.close("failed" if ownership["failed"] else "completed")


def _execute(root: Path, plan: dict, record: dict, *, checkpoint: Path | None = None) -> int:
    with _exclusive(root), _engine_ownership(root, plan) as ownership:
        # Re-read after acquiring the lock: a preceding orchestrator may have finished.
        record = _read_json(root / "orchestration.json")
        for attempt in record["attempts"]:
            if attempt["state"] == "running" and _process_active(attempt.get("pid")):
                raise SpecError(f"previous engine process {attempt['pid']} is still running")
        _verify_inputs(plan)
        for number, command in plan["phases"].items():
            previous = [entry for entry in record["attempts"] if entry["phase"] == number]
            if previous and previous[-1]["state"] == "execution_succeeded":
                continue
            argv = list(command["argv"])
            checkpoint_pin = None
            if previous and command["resume_policy"] == "native_flag":
                argv.append("--resume")
            if previous and command["resume_policy"] == "checkpoint_segment":
                if checkpoint is None:
                    raise SpecError("model_portfolio resume requires --checkpoint for an explicit new policy segment")
                resolved = checkpoint.expanduser().resolve()
                checkpoint_pin = {"path": str(resolved), "sha256": fingerprint(resolved)}
                argv += ["--resume-checkpoint", str(resolved)]
                argv[argv.index("--output") + 1] = str(root / "phase2" / f"segment-{len(previous) + 1:04d}")
            engine_output = command["engine_output"]
            if command["resume_policy"] == "checkpoint_segment":
                engine_output = argv[argv.index("--output") + 1]
            _verify_inputs(plan)
            log = root / f"phase{number}-attempt-{len(previous) + 1:04d}.log"
            entry = {
                "phase": number,
                "adapter": command["adapter"],
                "state": "running",
                "started_at": _now(),
                "argv": argv,
                "log": str(log),
                "engine_output": engine_output,
                "resume_checkpoint": checkpoint_pin,
            }
            record["attempts"].append(entry)
            record["state"] = "running"
            _write_json(root / "orchestration.json", record)
            process = None
            try:
                with log.open("w", encoding="utf-8") as output:
                    from .measured_launch import execution_environment

                    process = subprocess.Popen(
                        argv,
                        cwd=command["cwd"],
                        env=execution_environment(command),
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    ownership["unresolved"] = True
                    entry["pid"] = process.pid
                    _write_json(root / "orchestration.json", record)
                    try:
                        returncode = process.wait()
                    except KeyboardInterrupt:
                        _stop_process_group(process)
                        entry["ownership"] = "retained_until_descendants_verified"
                        returncode = 130
                    else:
                        if _process_group_active(process.pid):
                            raise SpecError("engine driver exited while workers remain; storage lease retained")
                        ownership["unresolved"] = False
            except OSError as exc:
                if process is not None:
                    _stop_process_group(process)
                    raise SpecError(
                        "post-spawn bookkeeping failed; storage lease retained for descendant verification"
                    ) from exc
                entry["error"] = str(exc)
                returncode = 127
            except BaseException:
                if process is not None and ownership["unresolved"]:
                    _stop_process_group(process)
                raise
            entry.update(
                returncode=returncode,
                ended_at=_now(),
                state="execution_succeeded" if returncode == 0 else "execution_failed",
            )
            if returncode == 0 and command["adapter"] == "capsule_derivation":
                try:
                    _verify_inputs(plan)
                    entry["output_sha256"] = fingerprint(engine_output)
                except (OSError, SpecError):
                    entry.update(returncode=1, state="execution_failed", error="phase0 output identity not established")
                    returncode = 1
            elif returncode == 0 and command["adapter"] == "capsule_bench":
                try:
                    _verify_inputs(plan)
                except (OSError, SpecError) as exc:
                    entry.update(
                        returncode=1,
                        engine_returncode=0,
                        state="execution_failed",
                        error=f"phase1 input identity changed during execution; engine evidence unchanged: {exc}",
                    )
                    returncode = 1
            record["state"] = entry["state"]
            _write_json(root / "orchestration.json", record)
            if returncode != 0:
                ownership["failed"] = True
                return returncode if returncode > 0 else 1
        record["state"] = "execution_succeeded"
        _write_json(root / "orchestration.json", record)
    return 0
