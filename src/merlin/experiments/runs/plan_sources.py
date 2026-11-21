"""Freeze and execute process plans while leaving experiment evidence with its engine."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from yaml import YAMLError

from merlin.common.digest import sha256_file as _sha
from merlin.development.backend.session.inputs.implementation_inventory import fingerprint
from merlin.experiments.definition import (
    ExperimentSpec,
    SpecError,
    backend_launch_inputs,
    engine_definitions,
    read_yaml,
    require_plan_selection,
    selected_engines,
)
from merlin.experiments.definition import default_run_dir as default_run_dir
from merlin.experiments.runs.adapters import (
    ADAPTERS,
    PHASE0_MODULE,
    PHASE1_MODULE,
    phase0_sources,
    phase0_startup_inputs,
    phase1_entrypoint,
)
from merlin.experiments.runs.plan_execution import (
    _process_active,
    _process_group_active,
    _read_json,
    _stop_process_group,
    _write_json,
)

# ---- runner ------------------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _corpus_closure(command: dict) -> dict:
    """Discover native grading roots without importing or executing a grader.

    Pin category membership as well as bytes: hashing only today's directories
    would miss a newly added sibling category or the appearance of hidden inputs.
    Records disclose category roots, never hidden capsule names or their contents.
    Native bundle snapshots and cohort admission remain the grading authority.
    """
    from merlin.common.paths import repo_root
    from merlin.target.registry.descriptor_loading import load_target_experiment

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
            from merlin.development.backend.session.inputs.corpus_views import require_reviewed_bundle
            from merlin.experiments.corpus.release import verify
            from merlin.target.registry.descriptor_loading import load_target_experiment

            if command["env"].get("MERLIN_CORPUS_SEAL") != seal:
                raise SpecError("native corpus seal differs from the explicitly selected input")
            descriptor = Path(command["inputs"]["descriptor"])
            verify(Path(seal), descriptor)
            manifest = Path(command["inputs"]["bundle_manifest"])
            bundle = read_yaml(manifest)
            try:
                require_reviewed_bundle(load_target_experiment(descriptor), manifest, bundle)
            except ValueError as exc:
                raise SpecError(str(exc)) from exc
            if command.get("requires_reviewed_corpus"):
                from merlin.common.paths import resolve_grant as source_grant
                from merlin.development.targetgen.sandbox.grants import resolve_grant

                source_root = Path(command["env"]["MERLIN_REPO_ROOT"]).resolve()
                legacy = source_grant("merlin/contract/capsules", root=source_root).absolute()
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
        if command["adapter"] != "capsule_derivation":
            continue
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
        if not command.get("module"):
            continue  # Historical script receipts keep their original source validation.
        if command["module"] != PHASE0_MODULE or command["argv"][1:3] != ["-m", PHASE0_MODULE]:
            raise SpecError("phase-0 module binding is not the supported derivation implementation")
        if plan.get("phase0_source_snapshot"):
            from merlin.development.testgen.generation.freeze import verify

            try:
                verify(plan)
            except (OSError, ValueError) as exc:
                raise SpecError(f"frozen Phase 0 sources changed: {exc}") from exc
            continue
        selection = command.get("phase0_m2m_selection")
        if selection is not None:
            from merlin.development.testgen.sealed_capture.generation import observe

            try:
                profile = command["inputs"].get("synth_profile")
                current = observe(
                    Path(selection["root"]),
                    Path(selection["python"]),
                    synth_profile=Path(profile) if profile else None,
                    require_source_origin="source_origin" in selection,
                )
            except (OSError, ValueError) as exc:
                raise SpecError(f"selected Model2MLIR runtime is unavailable: {exc}") from exc
            if current != selection:
                raise SpecError("selected Model2MLIR runtime changed before freezing")
        entrypoint, expected = _phase0_source_inputs()
        if command["env"].get("PYTHONSAFEPATH") != "1" or command["env"].get("PYTHONPATH", "").split(os.pathsep)[
            0
        ] != str(entrypoint.parents[len(PHASE0_MODULE.split("."))]):
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


_PHASE0_OPTIONAL_INPUTS = frozenset(
    {
        "conformance_spec",
        "synth_profile",
        "smt_profile",
        "chain_profile",
        "hidden_profile",
        "software_spec",
        "capability_contract",
        "hardware_spec",
        "rtl_facts",
        "component_coverage",
    }
)


def _phase0_input_paths(membership: dict) -> dict[str, str]:
    return {
        name: record["path"]
        for name, record in membership.items()
        if record is not None
        and (record["present"] or name in ("phase0:operator:recipe", "phase0:operator:performance_template"))
    }


def _phase0_operator_inputs(command: dict) -> dict[str, dict | None]:
    """Bind explicit file membership, including deliberately absent sidecars."""
    from merlin.development.testgen.profiles.synthesis_selection import application_inventory_path

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
    conformance_spec = inputs.get("conformance_spec")
    if conformance_spec and Path(conformance_spec).is_file():
        try:
            sidecar = application_inventory_path(conformance_spec)
        except (OSError, ValueError, YAMLError) as exc:
            raise SpecError(f"invalid selected application-demand sidecar: {exc}") from exc
        observed["phase0:operator:application_demands_sidecar"] = (
            {"path": str(sidecar.resolve()), "present": sidecar.is_file()} if sidecar is not None else None
        )
    return observed


def _phase0_synthesis_status(plan: dict) -> dict:
    """Evaluate selected synthesis from the plan's explicit inputs, never a checkout default."""
    from merlin.development.testgen.profiles.synthesis_selection import verify_selected_synthesis

    results = {}
    for number, command in plan["phases"].items():
        if command["adapter"] != "capsule_derivation":
            continue
        selected = command["inputs"]
        diagnostic = "--evidence-mode" in command["argv"] and _command_value(command, "--evidence-mode") == "diagnostic"
        try:
            results[number] = verify_selected_synthesis(
                selected.get("synth_profile"),
                conformance_spec=selected.get("conformance_spec"),
                recipe=selected.get("recipe"),
                descriptor=selected.get("descriptor"),
                **({"software_spec": selected["software_spec"]} if selected.get("software_spec") else {}),
                diagnostic=diagnostic,
            )
        except (OSError, ValueError, YAMLError) as exc:
            raise SpecError(f"phase-0 selected synthesis is invalid: {exc}") from exc
    return results


@contextmanager
def _phase1_environment(command: dict):
    """Observe the same frozen selectors that native launch/resume receives."""
    from merlin.development.backend.context import PreparedEnvironment, applied_environment
    from merlin.experiments.runs.execution import execution_environment

    with applied_environment(PreparedEnvironment(execution_environment(command), {})):
        yield


def _phase1_source_inputs(command: dict) -> dict[str, str]:
    from merlin.development.backend.session.inputs.implementation_inventory import paths

    with _phase1_environment(command):
        return paths(
            repo=Path(command["env"]["MERLIN_REPO_ROOT"]),
            entrypoint=Path(command["entrypoint"]),
            descriptor=Path(command["inputs"]["descriptor"]) if command["inputs"].get("descriptor") else None,
        )


def _verify_phase1_sources(plan: dict) -> None:
    from merlin.development.backend.session.inputs.implementation_inventory import PREFIXES

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
            or command["env"].get("PYTHONPATH", "").split(os.pathsep)[0]
            != str(entrypoint.parents[len(PHASE1_MODULE.split("."))])
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
    with _phase1_environment(command):
        return _phase1_operator_paths(command)


def _phase1_operator_paths(command: dict) -> dict[str, str | None]:
    from merlin.development.targetgen.sandbox.grants import resolve_grant
    from merlin.target.registry.experiment_descriptor import (
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
        # Corpus staging reads contract schemas, not the legacy contract tree.
        # The latter contains compatibility symlinks and is not an input to
        # this operator; freezing it would reject otherwise ordinary runs.
        "contract_schemas": root / "src/merlin/resources/contracts/schemas",
        "schemas": root / "src/merlin/resources/schemas",
        "oracle_timing": Path(command["inputs"]["oracle_timing"]),
    }
    selected_clang = command["env"].get("MERLIN_CLANG")
    if selected_clang:
        paths["clang"] = Path(selected_clang)
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
    chipyard_timing = False
    if descriptor.is_file():
        from merlin.target.registry.descriptor_loading import (
            declared_vs_resolved_contract,
            load_target_experiment,
        )

        selected = load_target_experiment(descriptor)
        if selected.sim_via == "chipyard":
            chipyard_timing = True
            _, contract_path, agreement = declared_vs_resolved_contract(selected)
            if agreement != "agree" or contract_path is None:
                raise SpecError(f"selected Phase 1 target contract is not agreed and resolvable: {agreement}")
            paths["target_contract"] = contract_path
    result = {"phase1:operator:" + name: str(path.resolve()) for name, path in paths.items()}
    target = document.get("target", "")
    for name, path in {
        "task": declared_task_root(document, root=root) or resources / "task",
        "timing:target": (resources if chipyard_timing else resources / "scripts") / f".oracle_timing.{target}.json",
        "timing:legacy": resources / "scripts/.oracle_timing.json",
        "environment": resources / "experiment.env",
    }.items():
        # Record absence, not an invented empty file: later appearance changes the plan.
        result["phase1:operator:" + name] = str(path.resolve()) if path.exists() or path.is_symlink() else None
    return result
