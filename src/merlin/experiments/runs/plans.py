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


def resolve_plan(
    spec: ExperimentSpec,
    *,
    workflow: str | None = None,
    phase: str | None = None,
    run_dir: Path | None = None,
    corpus_seal: Path | None = None,
    bundle_manifest: Path | None = None,
    backend_joint_campaign: Path | None = None,
    backend_oracle_timing: Path | None = None,
    phase1_driver: str | None = None,
    phase1_model: str | None = None,
    phase1_effort: str | None = None,
    phase1_provider: str | None = None,
    phase0_conformance_spec: Path | None = None,
    phase0_capability_contract: Path | None = None,
    phase0_synth_profile: Path | None = None,
    phase0_hidden_profile: Path | None = None,
    phase0_generation_scope: str | None = None,
    phase0_component_coverage: Path | None = None,
    phase0_rtl_facts: Path | None = None,
    phase0_evidence_mode: str | None = None,
    phase0_m2m_root: Path | None = None,
    phase0_m2m_python: Path | None = None,
    phase0_capture_timeout_seconds: int | None = None,
    phase0_capture_runtime_reuse: Path | None = None,
    phase0_rtl_source_relocations: Path | None = None,
    phase0_bwrap: Path | None = None,
) -> dict:
    """Resolve named work and explicit inputs into a frozen execution plan."""
    from merlin.common.paths import out_dir, repo_root
    from merlin.experiments.runs.plan_sources import (
        _PHASE0_OPTIONAL_INPUTS,
        _corpus_closure,
        _phase0_input_paths,
        _phase0_operator_inputs,
        _phase0_source_inputs,
        _phase1_operator_inputs,
        _phase1_source_inputs,
    )

    phase, selected = selected_engines(spec.document, workflow=workflow, phase=phase)
    root = repo_root().resolve()
    destination = (run_dir or default_run_dir(spec, phase)).expanduser().resolve()
    phases = engine_definitions(spec.document)
    if (corpus_seal is not None or bundle_manifest is not None) and selected != ["1"]:
        raise SpecError("a reviewed corpus and replacement bundle can only select Phase 1")
    phase1_selection = backend_launch_inputs(
        spec,
        selected,
        corpus_seal=corpus_seal,
        joint_campaign=backend_joint_campaign,
        oracle_timing=backend_oracle_timing,
        driver=phase1_driver,
        model=phase1_model,
        effort=phase1_effort,
        provider=phase1_provider,
    )
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
                phase0_generation_scope,
                phase0_component_coverage,
                phase0_rtl_facts,
                phase0_evidence_mode,
                phase0_m2m_root,
                phase0_m2m_python,
                phase0_capture_timeout_seconds,
                phase0_capture_runtime_reuse,
                phase0_rtl_source_relocations,
                phase0_bwrap,
            )
        )
        and phase != "0"
    ):
        raise SpecError("Phase 0 artifact selection requires --phase 0")
    phase0_selection = {}
    if phase0_generation_scope is not None:
        if phase0_generation_scope not in ("full", "phase1-functional"):
            raise SpecError("Phase 0 generation scope must be full or phase1-functional")
        phase0_selection["generation_scope"] = phase0_generation_scope
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
        from merlin.development.testgen.sealed_capture.generation import capture_timeout

        try:
            capture_timeout(phase0_capture_timeout_seconds)
        except ValueError as exc:
            raise SpecError(str(exc)) from exc
    if phase0_bwrap is not None:
        from merlin.development.testgen.sealed_capture.generation import capture_bwrap

        try:
            phase0_bwrap = Path(capture_bwrap(phase0_bwrap))
        except ValueError as exc:
            raise SpecError(str(exc)) from exc
    capture_runtime_reuse = None
    if phase0_capture_runtime_reuse is not None:
        from merlin.development.testgen.sealed_capture.generation import read_runtime_reuse

        phase0_capture_runtime_reuse = phase0_capture_runtime_reuse.expanduser().absolute()
        try:
            capture_runtime_reuse = read_runtime_reuse(phase0_capture_runtime_reuse)
        except (OSError, ValueError) as exc:
            raise SpecError(f"invalid Phase 0 capture runtime reuse selection: {exc}") from exc
    rtl_source_relocations = None
    if phase0_rtl_source_relocations is not None:
        from merlin.development.testgen.generation.staging import read_mapping

        phase0_rtl_source_relocations = phase0_rtl_source_relocations.expanduser().absolute()
        try:
            rtl_source_relocations = read_mapping(phase0_rtl_source_relocations, subject="RTL source relocations")
        except (OSError, ValueError) as exc:
            raise SpecError(f"invalid Phase 0 RTL source relocations: {exc}") from exc
    commands = {}
    corpus_closures = {}
    phase1_operator_inputs = None
    phase0_operator_inputs = None
    inputs = {f"declared:{name}": str(spec.resolve(value)) for name, value in spec.document.get("inputs", {}).items()}
    inputs["definition"] = str(spec.path)
    if phase0_capture_runtime_reuse is not None:
        inputs["phase0:capture-runtime-reuse-selection"] = str(phase0_capture_runtime_reuse)
    if phase0_rtl_source_relocations is not None:
        inputs["phase0:rtl-source-relocations-selection"] = str(phase0_rtl_source_relocations)
    for number in selected:
        entry = phases[number]
        adapter = ADAPTERS[entry["adapter"]]
        config = dict(entry["config"])
        if rtl_source_relocations is not None and adapter.name != "capsule_derivation":
            raise SpecError("--phase0-rtl-source-relocations requires the capsule derivation adapter")
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
            for key in (
                "MERLIN_TARGET_PATH",
                "MERLIN_TARGET_CONTRACT",
                "MERLIN_CODEX_CPUS",
                "MERLIN_COMPUTE_CPUS",
                "MERLIN_COMPUTE_CPU_SLOTS",
                "MERLIN_DEVICE_BUILD_JOBS",
                "MERLIN_CAPSULE_ORDER",
                "MERLIN_BUILD_CPUS",
                "MERLIN_BUILD_SLOT_ROOT",
                "MERLIN_GRADE_WORKERS",
                "MERLIN_GRADE_CPUS",
                "MERLIN_GSIM_CPUS",
                "MERLIN_GSIM_MAX_SLOTS",
                "MERLIN_GSIM_STARTUP_PRELOAD",
                "MERLIN_SPIKE_MAX_SLOTS",
                "MERLIN_OPERATOR_GRADE_PROCESSES",
                "MERLIN_PUBLIC_SCALE_BATCH_WORKERS",
                "MERLIN_SELFCHECK_BROKER_JOBS",
                "MERLIN_SNAPSHOT_LEASE_CACHE",
                "MERLIN_SIMULATION_CERT_CACHE",
                "MERLIN_COMMAND_RECORDER",
                "MERLIN_GSIM_SLOT_ROOT",
            ):
                if key in os.environ:
                    command["env"][key] = os.environ[key]
            from merlin.target.registry.identity import effective_target_path

            # An unset selection is the checkout's in-repo support default. Spell it out, so a
            # resume after the vendored support changed cannot pick up different support code.
            command["env"]["MERLIN_TARGET_PATH"] = effective_target_path()
            from merlin.target.registry.descriptor_loading import (
                load_target_experiment,
                selected_experiment_contract,
            )

            descriptor_path = Path(command["inputs"]["descriptor"])
            if descriptor_path.is_file():
                try:
                    experiment = load_target_experiment(descriptor_path)
                    contract = selected_experiment_contract(experiment, environment=command["env"])
                except ValueError as error:
                    raise SpecError(f"invalid Phase 1 capability selection: {error}") from error
                if contract is not None:
                    command["env"]["MERLIN_TARGET_CONTRACT"] = str(contract)
            from merlin.compiler.lower.model.toolchain import clang_for

            selected_clang = clang_for(root, dict(os.environ, **command["env"]))
            if selected_clang.is_absolute() and selected_clang.is_file():
                command["env"]["MERLIN_CLANG"] = str(selected_clang.resolve())
        if adapter.name == "capsule_derivation" and command.get("phase0_m2m_selection"):
            if not command["inputs"].get("software_spec"):
                raise SpecError("selected Model2MLIR capture requires explicit Phase 0 software evidence")
            from merlin.development.testgen.sealed_capture.generation import observe, sealed_capture_config

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
                from merlin.development.testgen.sealed_capture.generation import CONFIG_ENV
                from merlin.experiments.capture_execution.runtime_store import STORE_ENV

                selection = command["phase0_m2m_selection"]
                if phase0_capture_timeout_seconds is not None:
                    # Frozen with the plan; the freeze rebinds and re-verifies the capture config.
                    command["phase0_capture_timeout_seconds"] = phase0_capture_timeout_seconds
                if phase0_bwrap is not None:
                    command["phase0_bwrap"] = str(phase0_bwrap)
                if capture_runtime_reuse is not None:
                    command["phase0_capture_runtime_reuse"] = capture_runtime_reuse
                try:
                    capture_config = sealed_capture_config(
                        selection,
                        destination / "phase0",
                        execution_timeout_seconds=phase0_capture_timeout_seconds,
                        bwrap=phase0_bwrap,
                        **({"runtime_reuse": capture_runtime_reuse} if capture_runtime_reuse is not None else {}),
                    )
                except (OSError, ValueError) as exc:
                    raise SpecError(f"invalid selected sealed capture runtime: {exc}") from exc
                if capture_runtime_reuse is not None:
                    command["input_owner_roots"].append(capture_config["runtime_reuse"]["entry"])
                command["env"][CONFIG_ENV] = json.dumps(
                    capture_config,
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
            if rtl_source_relocations is not None:
                command["phase0_rtl_source_relocations"] = rtl_source_relocations
            if command["inputs"].get("software_spec"):
                from merlin.development.testgen.generation.freeze import selected_inputs

                try:
                    evidence_inputs, command["phase0_evidence"] = selected_inputs(command, spec.target)
                except (OSError, ValueError) as exc:
                    raise SpecError(f"invalid Phase 0 evidence selection: {exc}") from exc
                inputs.update(evidence_inputs)
            if "recipe" in command["inputs"]:
                phase0_operator_inputs = _phase0_operator_inputs(command)
                inputs.update(_phase0_input_paths(phase0_operator_inputs))
            else:
                profiles = Path(command["inputs"].get("profiles_root", root / "experiments/corpora/profiles"))
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
    if phase0_capture_runtime_reuse is not None and "phase0_capture_runtime_reuse" not in commands.get("0", {}):
        raise SpecError("--phase0-capture-runtime-reuse requires a selected sealed generation runtime")
    if phase0_rtl_source_relocations is not None and not commands.get("0", {}).get("phase0_evidence"):
        raise SpecError("--phase0-rtl-source-relocations requires selected Phase 0 RTL evidence")
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
    """Resolve each grading command's compiler before starting any grading work.

    Use the selected checkout and launch environment. A missing compiler is an
    input error, rather than a compilation failure attributed to every candidate.
    """
    import shutil

    from merlin.compiler.lower.model.toolchain import clang_for
    from merlin.experiments.runs.execution import execution_environment

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
    from merlin.experiments.runs.plan_sources import _command_value, _phase1_environment
    from merlin.target.registry.plugins import core_module_path, resolve_support, validate

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
                    f"plugin.rocc_semantics and plugin.rtl_checks; "
                    f"the selected provider declares {declared or 'neither'}"
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
    from merlin.experiments.runs.plan_sources import (
        _command_value,
        _phase0_synthesis_status,
        _phase1_environment,
        _verify_corpus_closures,
        _verify_phase0_sources,
        _verify_phase1_sources,
    )

    errors = []
    pins = {}
    synthesis = {}
    try:
        _verify_corpus_closures(plan)
        _verify_phase0_sources(plan)
        _verify_phase1_sources(plan)
        from merlin.development.backend.session.inputs.oracle_timing import (
            read_verified_timing,
            requires_chipyard_timing,
        )

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

        synthesis = _phase0_synthesis_status(plan)
        from merlin.development.testgen.performance.performance_readiness import check_performance_oracles

        for command in plan["phases"].values():
            if (
                command["adapter"] == "capsule_derivation"
                and command.get("phase0_evidence")
                and command["inputs"].get("recipe")
                and command["inputs"].get("capability_contract")
            ):
                try:
                    check_performance_oracles(
                        command,
                        target=plan["target"],
                        profile_target=_command_value(command, "--target"),
                        diagnostic=(
                            "--evidence-mode" in command["argv"]
                            and _command_value(command, "--evidence-mode") == "diagnostic"
                        ),
                    )
                except (OSError, ValueError, YAMLError) as exc:
                    raise SpecError(f"invalid Phase 0 performance selection: {exc}") from exc
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
    from merlin.development.backend.session.treatments import level_for_phase1

    phase1 = engine_definitions(plan.get("spec", {})).get("1") if "1" in plan["phases"] else None
    return {
        "configuration_ready": not errors,
        "engine_readiness": "not_executed",
        "phase1_level": level_for_phase1(phase1["config"]) if phase1 else None,
        "errors": errors,
        "inputs": pins,
        "phase0_synthesis": synthesis,
    }


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
    from merlin.development.backend.session.treatments import level_for_phase1

    for number, command in plan.get("phases", {}).items():
        attempts = [entry for entry in record["attempts"] if entry.get("phase") == number]
        latest = attempts[-1] if attempts else {}
        phases[number] = {
            "adapter": command["adapter"],
            **(
                {"level": level_for_phase1(engine_definitions(plan["spec"])[number]["config"])}
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


def run(plan: dict, *, workflow: str | None = None, phase: str | None = None) -> int:
    from merlin.experiments.runs.plan_execution import _execute
    from merlin.experiments.runs.plan_sources import _now

    require_plan_selection(plan, workflow=workflow, phase=phase)
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
        from merlin.development.testgen.generation.freeze import stage

        try:
            frozen = stage(frozen)
        except (OSError, ValueError) as exc:
            raise SpecError(f"Phase 0 freezing failed: {exc}; retained partial run at {root}") from exc
    _write_json(root / "resolved-plan.json", frozen)
    record = {"schema_version": 1, "state": "pending", "attempts": [], "plan_sha256": _sha(root / "resolved-plan.json")}
    _write_json(root / "orchestration.json", record)
    return _execute(root, frozen, record)
