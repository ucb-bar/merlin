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


def default_run_dir(spec: ExperimentSpec, phase: str = "all") -> Path:
    """Where a new orchestration lives when the operator names no ``--run-dir``.

    A single-phase run is a phase run: ``out/runs/<target>/phase<N>/<TS>_<experiment>_<sha7>/``, so
    a Phase 0 run owns its evidence, capsules and coverage at the phase address every later phase
    and the target index cite. A multi-phase orchestration keeps
    ``out/runs/<target>/<experiment>/<TS>_<uuid8>/``.
    """
    from merlin.common.artifacts import phase_run_id
    from merlin.common.paths import phase_runs_root, runs_dir

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if phase in ("0", "1", "2"):
        base = phase_runs_root(spec.target, phase)
        candidate = base / phase_run_id(spec.id, timestamp=stamp)
        return candidate if not candidate.exists() else base / f"{candidate.name}_{uuid4().hex[:6]}"
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
            from merlin.targetgen.target_experiment import load_target_experiment

            from .corpus.release import verify
            from .phase1.corpus_inputs import require_reviewed_bundle

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
                from merlin.targetgen.sandbox.bwrap import resolve_grant

                source_root = Path(command["env"]["MERLIN_REPO_ROOT"]).resolve()
                legacy = source_root / "merlin" / "contract" / "capsules"
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
            from .phase0.freeze import verify

            try:
                verify(plan)
            except (OSError, ValueError) as exc:
                raise SpecError(f"frozen Phase 0 sources changed: {exc}") from exc
            continue
        selection = command.get("phase0_m2m_selection")
        if selection is not None:
            from .phase0.m2m_runtime import observe

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


_PHASE0_OPTIONAL_INPUTS = frozenset(
    {
        "conformance_spec",
        "synth_profile",
        "smt_profile",
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
    from .phase0.profiles import application_inventory_path

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
    from .phase0.profiles import verify_selected_synthesis

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
    from .measured_launch import execution_environment
    from .phase1.runtime_environment import PreparedEnvironment, applied_environment

    with applied_environment(PreparedEnvironment(execution_environment(command), {})):
        yield


def _phase1_source_inputs(command: dict) -> dict[str, str]:
    from .phase1.source_inputs import paths

    with _phase1_environment(command):
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
    with _phase1_environment(command):
        return _phase1_operator_paths(command)


def _phase1_operator_paths(command: dict) -> dict[str, str | None]:
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
        # Corpus staging reads contract schemas, not the legacy contract tree.
        # The latter contains compatibility symlinks and is not an input to
        # this operator; freezing it would reject otherwise ordinary runs.
        "contract_schemas": root / "merlin/contract/schemas",
        "schemas": root / "merlin/schemas",
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
        from merlin.targetgen.target_experiment import declared_vs_resolved_contract, load_target_experiment

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


def resolve_plan(
    spec: ExperimentSpec,
    *,
    phase: str = "all",
    run_dir: Path | None = None,
    corpus_seal: Path | None = None,
    bundle_manifest: Path | None = None,
    phase1_driver: str | None = None,
    phase1_model: str | None = None,
    phase1_effort: str | None = None,
    phase1_provider: str | None = None,
    phase0_conformance_spec: Path | None = None,
    phase0_capability_contract: Path | None = None,
    phase0_synth_profile: Path | None = None,
    phase0_hidden_profile: Path | None = None,
    phase0_component_coverage: Path | None = None,
    phase0_rtl_facts: Path | None = None,
    phase0_evidence_mode: str | None = None,
    phase0_m2m_root: Path | None = None,
    phase0_m2m_python: Path | None = None,
    phase0_capture_timeout_seconds: int | None = None,
    phase0_bwrap: Path | None = None,
) -> dict:
    from merlin.common.paths import out_dir, repo_root

    root = repo_root().resolve()
    destination = (run_dir or default_run_dir(spec, phase)).expanduser().resolve()
    phases = spec.document["phases"]
    selected = sorted(phases) if phase == "all" else [phase]
    if any(number not in phases for number in selected):
        raise SpecError(f"definition does not declare phase {phase}")
    if (corpus_seal is not None or bundle_manifest is not None) and selected != ["1"]:
        raise SpecError("a reviewed corpus and replacement bundle can only select Phase 1")
    phase1_selection = {
        name: value
        for name, value in (
            ("driver", phase1_driver),
            ("model", phase1_model),
            ("effort", phase1_effort),
            ("provider", phase1_provider),
        )
        if value is not None
    }
    if phase1_selection and selected != ["1"]:
        raise SpecError("Phase 1 launch overrides require a Phase 1-only plan")
    if (corpus_seal is None) != (bundle_manifest is None):
        raise SpecError("select both --corpus-seal and --bundle-manifest for a new reviewed Phase 1 run")
    if (phase0_conformance_spec is None) != (phase0_synth_profile is None):
        raise SpecError("select both --phase0-conformance-spec and --phase0-synth-profile")
    if (phase0_m2m_root is None) != (phase0_m2m_python is None):
        raise SpecError("select both --phase0-m2m-root and --phase0-m2m-python")
    if (
        any(
            path is not None
            for path in (
                phase0_conformance_spec,
                phase0_capability_contract,
                phase0_synth_profile,
                phase0_hidden_profile,
                phase0_component_coverage,
                phase0_rtl_facts,
                phase0_evidence_mode,
                phase0_m2m_root,
                phase0_m2m_python,
                phase0_capture_timeout_seconds,
                phase0_bwrap,
            )
        )
        and phase != "0"
    ):
        raise SpecError("Phase 0 artifact selection requires --phase 0")
    phase0_selection = {}
    for name, path in (
        ("conformance_spec", phase0_conformance_spec),
        ("capability_contract", phase0_capability_contract),
        ("synth_profile", phase0_synth_profile),
        ("hidden_profile", phase0_hidden_profile),
        ("component_coverage", phase0_component_coverage),
        ("rtl_facts", phase0_rtl_facts),
    ):
        if path is None:
            continue
        candidate = path.expanduser().absolute()
        if candidate.is_symlink() or not candidate.is_file():
            raise SpecError(f"selected Phase 0 {name} is not an existing ordinary file: {candidate}")
        phase0_selection[name] = str(candidate)
    if phase0_evidence_mode is not None:
        if phase0_evidence_mode not in ("diagnostic", "verified"):
            raise SpecError("Phase 0 evidence mode must be diagnostic or verified")
        phase0_selection["evidence_mode"] = phase0_evidence_mode
    for name, path in (("m2m_root", phase0_m2m_root), ("m2m_python", phase0_m2m_python)):
        if path is not None:
            phase0_selection[name] = str(path.expanduser().absolute())
    if phase0_capture_timeout_seconds is not None:
        from .phase0.m2m_runtime import capture_timeout

        try:
            capture_timeout(phase0_capture_timeout_seconds)
        except ValueError as exc:
            raise SpecError(str(exc)) from exc
    if phase0_bwrap is not None:
        from .phase0.m2m_runtime import capture_bwrap

        try:
            phase0_bwrap = Path(capture_bwrap(phase0_bwrap))
        except ValueError as exc:
            raise SpecError(str(exc)) from exc
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
        if number == "1" and phase1_selection:
            if adapter.name != "capsule_bench":
                raise SpecError("Phase 1 launch overrides require the capsule_bench adapter")
            config.update(phase1_selection)
            adapter.validate(config)
        if number == "0" and phase0_selection:
            config.update(phase0_selection)
            adapter.validate(config)
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
            # Preflight and native admission require this path to be the exact
            # generated bundle inside the selected release. Path inequality
            # against the example default is not that proof: an example may
            # already point at this release, while an unrelated copied bundle
            # can have a different path and the same unsafe grants.
            config["bundle_manifest"] = str(selected_bundle)
            if selected_bundle.is_file():
                config["bundle"] = read_yaml(selected_bundle).get("bundle_id")
            adapter.validate(config)
        command = adapter.resolve(spec, config, root, destination)
        if adapter.name == "capsule_bench" and command.get("module") == PHASE1_MODULE:
            # The native process inherits the caller's environment. Freeze selections
            # that otherwise could differ between preflight and launch/resume.
            for key in ("MERLIN_TARGET_PATH", "MERLIN_TARGET_CONTRACT"):
                if key in os.environ:
                    command["env"][key] = os.environ[key]
            from merlin.targetgen.target_registry import effective_target_path

            # An unset selection is the checkout's in-repo support default. Spell it out, so a
            # resume after the vendored support changed cannot pick up different support code.
            command["env"]["MERLIN_TARGET_PATH"] = effective_target_path()
            from merlin.targetgen.target_experiment import load_target_experiment, selected_experiment_contract

            descriptor_path = Path(command["inputs"]["descriptor"])
            if descriptor_path.is_file():
                try:
                    experiment = load_target_experiment(descriptor_path)
                    contract = selected_experiment_contract(experiment, environment=command["env"])
                except ValueError as error:
                    raise SpecError(f"invalid Phase 1 capability selection: {error}") from error
                if contract is not None:
                    command["env"]["MERLIN_TARGET_CONTRACT"] = str(contract)
            from merlin.llvmlower.toolchain import clang_for

            selected_clang = clang_for(root, dict(os.environ, **command["env"]))
            if selected_clang.is_absolute() and selected_clang.is_file():
                command["env"]["MERLIN_CLANG"] = str(selected_clang.resolve())
        if adapter.name == "capsule_derivation" and command.get("phase0_m2m_selection"):
            if not command["inputs"].get("software_spec"):
                raise SpecError("selected Model2MLIR capture requires explicit Phase 0 software evidence")
            from .phase0.m2m_runtime import observe, sealed_capture_config

            choice = command["phase0_m2m_selection"]
            try:
                profile = command["inputs"].get("synth_profile")
                command["phase0_m2m_selection"] = observe(
                    Path(choice["m2m_root"]),
                    Path(choice["m2m_python"]),
                    synth_profile=Path(profile) if profile else None,
                    require_source_origin=config.get("component_coverage") is not None
                    or config.get("evidence_mode") != "diagnostic",
                )
                command["input_owner_roots"].append(command["phase0_m2m_selection"]["base"])
            except (OSError, ValueError) as exc:
                raise SpecError(f"invalid selected Model2MLIR runtime: {exc}") from exc
            if config.get("component_coverage") is not None or config.get("evidence_mode") != "diagnostic":
                # Verified Phase 0 admits a generation-time capture only from the sealed runner
                # (operator policy, 2026-10-01): every PyTorch capture this run makes is preselected,
                # sandboxed, replayed and attested against the runtime selected here.
                from merlin.common.artifacts import cache_dir

                from .capture_execution.runtime_store import STORE_ENV
                from .phase0.sealed_generation import CONFIG_ENV

                selection = command["phase0_m2m_selection"]
                if phase0_capture_timeout_seconds is not None:
                    # Frozen with the plan; the freeze rebinds and re-verifies the capture config.
                    command["phase0_capture_timeout_seconds"] = phase0_capture_timeout_seconds
                if phase0_bwrap is not None:
                    command["phase0_bwrap"] = str(phase0_bwrap)
                command["env"][CONFIG_ENV] = json.dumps(
                    sealed_capture_config(
                        selection,
                        destination / "phase0",
                        execution_timeout_seconds=phase0_capture_timeout_seconds,
                        bwrap=phase0_bwrap,
                    ),
                    sort_keys=True,
                )
                # Keep the operator's shared store selection in the frozen
                # command. Substituting a checkout-local cache prevents reuse
                # and can copy the entire runtime onto another filesystem.
                command["env"][STORE_ENV] = os.environ.get(STORE_ENV) or str(cache_dir("sealed-m2m-runtime"))
        commands[number] = command
        inputs[f"phase{number}:entrypoint"] = command["entrypoint"]
        for name, value in command["inputs"].items():
            if adapter.name == "capsule_derivation" and "recipe" in command["inputs"]:
                if name in _PHASE0_OPTIONAL_INPUTS:
                    continue  # Membership and present bytes are pinned below, not absent paths.
            inputs[f"phase{number}:{name}"] = value
        if adapter.name == "capsule_derivation":
            if command["inputs"].get("software_spec"):
                from .phase0.freeze import selected_inputs

                try:
                    evidence_inputs, command["phase0_evidence"] = selected_inputs(command, spec.target)
                except (OSError, ValueError) as exc:
                    raise SpecError(f"invalid Phase 0 evidence selection: {exc}") from exc
                inputs.update(evidence_inputs)
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
    if phase0_bwrap is not None and "phase0_bwrap" not in commands.get("0", {}):
        raise SpecError(
            "--phase0-bwrap applies only to sealed generation captures: select "
            "--phase0-m2m-root/--phase0-m2m-python with verified (or component-coverage) Phase 0 evidence"
        )
    if phase0_capture_timeout_seconds is not None and "phase0_capture_timeout_seconds" not in commands.get("0", {}):
        raise SpecError(
            "--phase0-capture-timeout-seconds applies only to sealed generation captures: select "
            "--phase0-m2m-root/--phase0-m2m-python with verified (or component-coverage) Phase 0 evidence"
        )
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
        **({"phase0_selected_artifacts": phase0_selection} if phase0_selection else {}),
        **({"phase1_launch_overrides": phase1_selection} if phase1_selection else {}),
        "run_dir": str(destination),
        "storage_root": str(out_dir().resolve()),
        "phases": commands,
        "input_paths": inputs,
        "corpus_closures": corpus_closures,
        "evidence_authority": "legacy phase engines and AET; process completion is not a scientific verdict",
    }


def _verify_toolchain(plan: dict) -> list[str]:
    """Every grading phase's compiler must resolve to a real executable before anything is graded.

    ``merlin.llvmlower.toolchain.clang`` resolves from the process environment, the checkout's
    ``.env`` and its ``third_party`` install, and its fallback chain ends in a bare name, so an
    absent install otherwise surfaces capsule by capsule as compile failures of the candidate
    rather than once, here, as the missing toolchain it is. Each command is resolved as its
    engine will see it: the engine's checkout and launch environment, not this process's.
    """
    import shutil

    from merlin.llvmlower.toolchain import clang_for

    from .measured_launch import execution_environment

    errors = []
    for number, command in sorted(plan["phases"].items()):
        adapter = ADAPTERS.get(command["adapter"])
        if adapter is None or adapter.phase == "0":
            continue
        environment = execution_environment(command)
        root = Path(environment.get("MERLIN_REPO_ROOT") or command["cwd"])
        value = clang_for(root, environment)
        resolved = value
        if not value.is_absolute():
            found = shutil.which(str(value), path=environment.get("PATH")) if len(value.parts) == 1 else None
            resolved = Path(found) if found else Path(command["cwd"]) / value
        if not (resolved.is_file() and os.access(resolved, os.X_OK)):
            errors.append(
                f"phase {number} toolchain clang does not resolve to an executable file: {str(value)!r} "
                f"(checkout {root}); set MERLIN_CLANG or install the checkout's LLVM toolchain"
            )
    return errors


def _verify_rtlcheck_support(plan: dict) -> list[str]:
    """Reject an EL4 selection with no selected, statically resolvable check provider.

    RTL checks resolve from the target's host backend.  Importing an arbitrary OOT
    backend at preflight would execute user code, so this checks only the selected
    support contract and its declared file.  Native startup still validates that
    the module loads and implements the complete check capability.
    """
    from merlin.targetgen.plugins import core_module_path, resolve_support, validate

    for command in plan["phases"].values():
        if command.get("module") != PHASE1_MODULE or "--treatment" not in command["argv"]:
            continue
        try:
            selected_treatment = _command_value(command, "--treatment")
        except SpecError:
            continue  # _verify_phase1_sources reports malformed or duplicated selections.
        if selected_treatment != "rtlchecks":
            continue
        target = plan["target"]
        try:
            with _phase1_environment(command):
                support = resolve_support(target)
                plugin = support.plugin()
            if not plugin.get("backend"):
                return [
                    f"phase 1 EL4 target {target!r} has no selected support plugin.backend; "
                    "RTL checks cannot run from the metadata-only example. Select a reviewed OOT support provider."
                ]
            problems = validate(plugin, root=support.base, where=f"{target} plugin")
            # A data-only provider served by a GENERIC core backend has RTL checks only through the
            # modules its plugin block selects; say so here rather than at native startup.
            if core_module_path(str(plugin["backend"])) is not None and not (
                plugin.get("rocc_semantics") and plugin.get("rtl_checks")
            ):
                declared = sorted(key for key in ("rocc_semantics", "rtl_checks") if plugin.get(key))
                problems.append(
                    f"{target} plugin: the generic backend {plugin['backend']!r} serves RTL checks only from "
                    f"plugin.rocc_semantics and plugin.rtl_checks; the selected provider declares {declared or 'neither'}"
                )
            return [f"phase 1 EL4 support: {problem}" for problem in problems]
        except (OSError, ValueError) as exc:
            return [f"phase 1 EL4 support for {target!r} cannot be resolved: {exc}"]
    return []


def preflight(plan: dict) -> dict:
    """Check only definition, files, and process transport, without starting engines.

    The grading phases' compiler is one of those files, resolved as each engine will resolve it.
    Oracle qualification, sandbox admission, credentials, and hardware readiness
    remain the native engines' live gates. This must never report a hardware GO.
    """
    errors = []
    pins = {}
    synthesis = {}
    try:
        _verify_corpus_closures(plan)
        _verify_phase0_sources(plan)
        _verify_phase1_sources(plan)
        from .phase1.timing import read_verified_timing, requires_chipyard_timing

        for command in plan["phases"].values():
            if (
                command.get("module") == PHASE1_MODULE
                and "--no-oracle" not in command["argv"]
                and requires_chipyard_timing(Path(command["inputs"]["descriptor"]))
            ):
                try:
                    with _phase1_environment(command):
                        read_verified_timing(
                            Path(command["inputs"]["oracle_timing"]),
                            descriptor=Path(command["inputs"]["descriptor"]),
                            target=plan["target"],
                        )
                except ValueError as exc:
                    raise SpecError(str(exc)) from exc
        from .measured_launch import verify_plan

        verify_plan(plan)
        from .portfolio_catalog import verify_plan as verify_portfolio_plan

        verify_portfolio_plan(plan)
        synthesis = _phase0_synthesis_status(plan)
        for command in plan["phases"].values():
            if command.get("phase0_evidence") and command["phase0_evidence"]["status"] != "verified":
                diagnostic = (
                    "--evidence-mode" in command["argv"] and _command_value(command, "--evidence-mode") == "diagnostic"
                )
                if not diagnostic:
                    errors.append("Phase 0 evidence is not qualified; select explicit diagnostic mode for inspection")
        for number, result in synthesis.items():
            if result["status"] in {"unverified_legacy", "incomplete_diagnostic"} and not (
                "--evidence-mode" in plan["phases"][number]["argv"]
                and _command_value(plan["phases"][number], "--evidence-mode") == "diagnostic"
            ):
                errors.append(
                    f"phase {number} selected synthesis is {result['status']}; "
                    "regenerate and review a digest-bound profile, then freeze a new run"
                )
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
    errors.extend(_verify_toolchain(plan))
    errors.extend(_verify_rtlcheck_support(plan))
    from .phase1.levels import level_for_phase1

    phase1 = plan.get("spec", {}).get("phases", {}).get("1") if "1" in plan["phases"] else None
    return {
        "configuration_ready": not errors,
        "engine_readiness": "not_executed",
        "phase1_level": level_for_phase1(phase1["config"]) if phase1 else None,
        "errors": errors,
        "inputs": pins,
        "phase0_synthesis": synthesis,
    }


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
    from .phase1.levels import level_for_phase1

    for number, command in plan.get("phases", {}).items():
        attempts = [entry for entry in record["attempts"] if entry.get("phase") == number]
        latest = attempts[-1] if attempts else {}
        phases[number] = {
            "adapter": command["adapter"],
            **(
                {"level": level_for_phase1(plan["spec"]["phases"][number]["config"])}
                if command["adapter"] == "capsule_bench"
                else {}
            ),
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
    if any(command.get("phase0_evidence") for command in frozen["phases"].values()):
        from .phase0.freeze import stage

        try:
            frozen = stage(frozen)
        except (OSError, ValueError) as exc:
            raise SpecError(f"Phase 0 freezing failed: {exc}; retained partial run at {root}") from exc
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


_PHASE1_COMPLETION_MEMBERS = (
    "submission",
    "run_manifest.yaml",
    "qa_loop_summary.yaml",
    "timing_detailed.json",
)


def _phase1_completion_inputs(command: dict) -> dict[str, dict[str, str]]:
    """Bind the installed functional compiler and its final formal evidence.

    The native run may also contain mutable workspaces and caches. Those are not
    the compiler handed to Phase 2; pin the four final handoff members instead.
    """
    root = Path(command["engine_output"])
    if root.resolve() != root or root.is_symlink():
        raise SpecError("installed Phase 1 handoff output changed location")
    result = {}
    for name in _PHASE1_COMPLETION_MEMBERS:
        path = root / name
        if path.is_symlink() or not path.exists():
            raise SpecError(f"installed Phase 1 completion member is missing or linked: {name}")
        if name == "submission" and not path.is_dir():
            raise SpecError("installed Phase 1 compiler submission is not a directory")
        if name == "submission" and any(member.is_symlink() for member in path.rglob("*")):
            raise SpecError("installed Phase 1 compiler submission contains linked members")
        if name != "submission" and not path.is_file():
            raise SpecError(f"installed Phase 1 completion member is not a file: {name}")
        result[name] = {"path": str(path), "sha256": fingerprint(path)}
    return result


def _verify_completed_phase1(command: dict, attempt: dict) -> None:
    expected = attempt.get("completion_inputs")
    if not isinstance(expected, dict) or set(expected) != set(_PHASE1_COMPLETION_MEMBERS):
        raise SpecError("completed installed Phase 1 lacks bound handoff outputs; create a new experiment run")
    try:
        observed = _phase1_completion_inputs(command)
    except (OSError, SpecError) as exc:
        raise SpecError(f"completed installed Phase 1 handoff changed: {exc}") from exc
    if observed != expected:
        raise SpecError("completed installed Phase 1 handoff bytes changed; create a new experiment run")


def _installed_phase2_output(command: dict, attempt: dict) -> dict[str, dict[str, str]]:
    """Bind terminal evidence without hashing mutable Phase 2 work areas."""
    from merlin.benchharness import hash_tree

    output = Path(attempt["engine_output"])
    if output.is_symlink() or output.resolve() != output or not output.is_dir():
        raise SpecError("completed installed Phase 2 output is absent, linked or moved")
    expected = Path(command["engine_output"])
    if command["resume_policy"] == "checkpoint_segment":
        if output.parent != expected or not output.name.startswith("segment-"):
            raise SpecError("completed portfolio segment is outside its frozen output root")
    elif output != expected:
        raise SpecError("completed installed Phase 2 output differs from its frozen selection")

    def pin(name: str, path: Path, *, directory: bool = False) -> dict[str, str]:
        if (
            path.is_symlink()
            or path.resolve() != path
            or not path.is_relative_to(output)
            or not (path.is_dir() if directory else path.is_file())
        ):
            raise SpecError(f"completed installed Phase 2 terminal member is absent or linked: {name}")
        if directory and any(member.is_symlink() for member in path.rglob("*")):
            raise SpecError(f"completed installed Phase 2 terminal tree contains a link: {name}")
        return {"path": str(path), "sha256": fingerprint(path)}

    if command["adapter"] == "measured_claims":
        from .phase2.checkpoint_admission import SCHEMA

        finals = sorted(output.glob("experiment_manifest.*.json"))
        if len(finals) != 1:
            raise SpecError("completed measured Phase 2 run needs one final experiment manifest")
        manifest = finals[0]
        final = pin("experiment_manifest", manifest)
        if (
            manifest.name != f"experiment_manifest.{final['sha256']}.json"
            or (document := _read_json(manifest)).get("schema") != SCHEMA
            or document.get("status") != "GO"
        ):
            raise SpecError("completed measured Phase 2 final manifest is not a content-addressed GO record")
        return {"experiment_manifest": final}

    if command["adapter"] != "model_portfolio":
        raise SpecError("unsupported installed Phase 2 terminal output")
    records = output / "global_iterations"
    result = {
        "host_resource_telemetry": pin("host_resource_telemetry", output / "host_resource_telemetry.json"),
        "launch": pin("launch", output / "launch.json"),
        "agent_sequence": pin("agent_sequence", records / "agent_sequence.json"),
    }
    telemetry = _read_json(Path(result["host_resource_telemetry"]["path"]))
    if telemetry.get("status") != "completed" or telemetry.get("worker_returncode") != 0:
        raise SpecError("completed portfolio segment lacks a successful worker telemetry receipt")
    sequence = _read_json(Path(result["agent_sequence"]["path"]))
    selected = sequence.get("last_good_checkpoint") or {}
    if (
        sequence.get("schema") != "global_agent_sequence_v1"
        or sequence.get("status") != "budget_complete"
        or not isinstance(selected, dict)
        or not isinstance(selected.get("path"), str)
        or not isinstance(selected.get("candidate_sha256"), str)
        or type(sequence.get("promotion_ready")) is not bool
    ):
        raise SpecError("completed portfolio segment lacks a selected terminal checkpoint")
    checkpoint = Path(selected["path"])
    if not checkpoint.is_relative_to(records):
        raise SpecError("completed portfolio checkpoint escapes its segment")
    result["selected_checkpoint"] = pin("selected_checkpoint", checkpoint)
    if result["selected_checkpoint"]["sha256"] != selected.get("sha256"):
        raise SpecError("completed portfolio checkpoint differs from the sequence receipt")
    final_checkpoint = _read_json(checkpoint)
    candidate = Path(str(final_checkpoint.get("candidate_path") or ""))
    if not candidate.is_relative_to(records):
        raise SpecError("completed portfolio candidate escapes its segment")
    result["selected_candidate"] = pin("selected_candidate", candidate, directory=True)
    if (
        final_checkpoint.get("candidate_sha256") != selected["candidate_sha256"]
        or hash_tree(candidate)["sha256"] != selected["candidate_sha256"]
    ):
        raise SpecError("completed portfolio candidate differs from its selected checkpoint")
    # A recovered round can leave a ready checkpoint without a final review
    # seal. The worker writes global_candidate only for failure-free sequences.
    if sequence["promotion_ready"] and not sequence.get("failures"):
        ready = records / "global_candidate.json"
        result["global_candidate"] = pin("global_candidate", ready)
        ready_record = _read_json(ready)
        ready_candidate = Path(str(ready_record.get("candidate_path") or ""))
        if not ready_candidate.is_relative_to(records):
            raise SpecError("completed portfolio review candidate escapes its segment")
        result["global_candidate_snapshot"] = pin("global_candidate_snapshot", ready_candidate, directory=True)
        if (
            ready_record.get("schema") != "global_perf_candidate_v1"
            or ready_record.get("candidate_sha256") != selected["candidate_sha256"]
            or hash_tree(ready_candidate)["sha256"] != selected["candidate_sha256"]
        ):
            raise SpecError("completed portfolio review candidate differs from its selected checkpoint")
    return result


def _verify_completed_phase2(command: dict, attempt: dict) -> None:
    expected = attempt.get("terminal_outputs")
    if not isinstance(expected, dict) or not expected:
        raise SpecError("completed installed Phase 2 lacks output identity; create a new experiment run")
    try:
        observed = _installed_phase2_output(command, attempt)
    except (OSError, SpecError) as exc:
        raise SpecError(f"completed installed Phase 2 output changed: {exc}") from exc
    if observed != expected:
        raise SpecError("completed installed Phase 2 terminal bytes changed; create a new experiment run")


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
        if checkpoint is not None and not any(
            command["resume_policy"] == "checkpoint_segment" for command in plan["phases"].values()
        ):
            raise SpecError("--checkpoint requires a checkpoint-segment experiment")
        if checkpoint is not None and not any(
            plan["phases"][attempt["phase"]]["resume_policy"] == "checkpoint_segment" for attempt in record["attempts"]
        ):
            raise SpecError("--checkpoint requires a prior portfolio segment")
        for attempt in record["attempts"]:
            pin = attempt.get("resume_checkpoint")
            if pin is not None:
                if (
                    not isinstance(pin, dict)
                    or not isinstance(pin.get("path"), str)
                    or not isinstance(pin.get("sha256"), str)
                ):
                    raise SpecError("previous portfolio resume checkpoint identity is malformed")
                try:
                    observed = fingerprint(pin["path"])
                except (OSError, SpecError) as exc:
                    raise SpecError("previous portfolio resume checkpoint is unavailable or changed") from exc
                if observed != pin["sha256"]:
                    raise SpecError("previous portfolio resume checkpoint bytes changed")
            command = plan["phases"][attempt["phase"]]
            if (
                attempt["state"] == "execution_succeeded"
                and command["adapter"] in {"measured_claims", "model_portfolio"}
                and command.get("module")
            ):
                _verify_completed_phase2(command, attempt)
        for number, command in plan["phases"].items():
            previous = [entry for entry in record["attempts"] if entry["phase"] == number]
            continue_segment = bool(
                previous and command["resume_policy"] == "checkpoint_segment" and checkpoint is not None
            )
            if previous and previous[-1]["state"] == "execution_succeeded" and not continue_segment:
                if command["adapter"] == "capsule_derivation":
                    expected = previous[-1].get("output_sha256")
                    try:
                        observed = fingerprint(command["engine_output"])
                    except (OSError, SpecError) as exc:
                        raise SpecError("phase-0 output identity changed; create a new experiment run") from exc
                    if not expected or observed != expected:
                        raise SpecError("phase-0 output identity changed; create a new experiment run")
                if command["adapter"] == "capsule_bench" and command.get("module") == PHASE1_MODULE:
                    _verify_completed_phase1(command, previous[-1])
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
                        command.get("frozen_launch", argv),
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
                    if command.get("module") == PHASE1_MODULE:
                        entry["completion_inputs"] = _phase1_completion_inputs(command)
                except (OSError, SpecError) as exc:
                    entry.update(
                        returncode=1,
                        engine_returncode=0,
                        state="execution_failed",
                        error=f"phase1 handoff identity not established; engine evidence unchanged: {exc}",
                    )
                    returncode = 1
            elif (
                returncode == 0
                and command["adapter"] in {"measured_claims", "model_portfolio"}
                and command.get("module")
            ):
                try:
                    _verify_inputs(plan)
                    entry["terminal_outputs"] = _installed_phase2_output(command, entry)
                except (OSError, SpecError) as exc:
                    entry.update(
                        returncode=1,
                        engine_returncode=0,
                        state="execution_failed",
                        error=f"phase2 output identity not established; engine evidence unchanged: {exc}",
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
