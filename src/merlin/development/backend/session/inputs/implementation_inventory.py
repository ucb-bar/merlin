"""One host-owned implementation inventory shared by native and outer Phase 1 execution.

The interim scope includes the complete retained native harness Python tree (including
maintenance tools), extracted Phase 1 and cross-phase corpus packages, declared public
clients and exact startup owners. This is not the transitive core/upstream dependency graph or a Python sandbox.
Records embed in existing run receipts; this module creates no second seal.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from merlin.common.digest import SourceMembershipError, python_members
from merlin.common.digest import sha256_file as _sha
from merlin.experiments import definition as access_policy
from merlin.experiments.definition import SpecError
from merlin.experiments.runs.adapters import _relative_entrypoint, phase0_startup_inputs

# ---- source_inputs -----------------------------------------------------------------------------

PREFIXES = ("phase1:client:", "phase1:source:", "phase1:startup:", "phase1:native:")

SCOPE = "phase1-package-public-clients-startup-and-retained-native-harness-access-policy-v2"


def fingerprint(path: str | Path, *, leased: bool = False) -> str:
    """Bind files and directory trees without copying models or hardware checkouts.

    Directory symlinks are refused to avoid an incomplete closure or cycles. Files
    reached through a symlink bind both the link destination and its current bytes.
    """
    if leased:
        from merlin.common.storage import read_leases as read_lease_digest

        # Only immutable, operator-owned callers opt in. Membership and link
        # spelling are still enumerated on every call; no tree identity cache.
        with read_lease_digest.verification():

            def digest(member):
                value = None
                if member.is_absolute() and member.resolve() == member:
                    value = read_lease_digest.digest(member)
                return _sha(member) if value is None else value

            return _fingerprint(Path(path), digest)
    return _fingerprint(Path(path), _sha)


def _fingerprint(path: Path, file_digest) -> str:
    if path.is_file():
        return file_digest(path)
    if not path.is_dir():
        raise SpecError(f"input is absent or is not a regular file/directory: {path}")
    digest = hashlib.sha256()
    for member in sorted(path.rglob("*")):
        if member.is_symlink() and member.is_dir():
            raise SpecError(f"input directory contains a directory symlink: {member}; name its closure explicitly")
        if member.is_file():
            row = [str(member.relative_to(path)), file_digest(member)]
            if member.is_symlink():
                row.append(os.readlink(member))
            digest.update(json.dumps(row).encode())
        elif member.is_symlink():
            raise SpecError(f"input contains a broken symlink: {member}")
        elif member.is_dir():
            digest.update(json.dumps([str(member.relative_to(path)), "directory"]).encode())
        else:
            raise SpecError(f"input contains a non-regular filesystem entry: {member}")
    return digest.hexdigest()


def _source(module: str) -> Path:
    from merlin.common.paths import module_source_path

    try:
        path = module_source_path(module)
    except (ImportError, OSError) as exc:
        raise SpecError(f"phase-1 implementation source is unavailable: {module}") from exc
    if path.is_symlink() or not path.is_file():
        raise SpecError(f"phase-1 implementation source is absent or symlinked: {module}")
    return path


def _python_members(directory: Path, prefix: str) -> dict[str, str]:
    try:
        return {prefix + name: str(path) for name, path in python_members(directory, label="phase-1").items()}
    except SourceMembershipError as exc:
        raise SpecError(str(exc)) from exc


def _selected_provider_inputs(target: object) -> dict[str, str]:
    """Bind explicit support selection without importing its executable implementation.

    Paths bind the selected root, identity documents bind its declared identity,
    and rediscovered Python membership binds additions as well as changed bytes.
    This is live-source admission, not a copied-provider replay mechanism.
    """
    from merlin.runtime.backends.registry import _assert_oot_plugin_ownership
    from merlin.target.registry.identity import explicit_targets
    from merlin.target.registry.plugins import resolve_support

    if not isinstance(target, str) or not target:
        return {}
    try:
        _assert_oot_plugin_ownership()
        if target not in explicit_targets():
            return {}
        selected = resolve_support(target)
        root = selected.base.resolve()
        inputs = _python_members(root, "phase1:startup:provider:python:")
        for name, path in (
            ("contract", selected.contract_path),
            ("identity", root / "provider.yaml"),
        ):
            if name == "identity" and not path.exists() and not path.is_symlink():
                continue
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
                raise SpecError(f"phase-1 selected provider identity is absent, linked or foreign: {path}")
            inputs[f"phase1:startup:provider:{name}"] = str(path.resolve())
        return inputs
    except (ValueError, OSError, RuntimeError) as exc:
        raise SpecError(f"phase-1 selected provider source admission failed: {exc}") from exc


def paths(
    *, repo: Path, entrypoint: Path, descriptor: Path | None = None, require_native: bool = False
) -> dict[str, str]:
    """Rediscover supported source ownership, including native executable aliases."""
    from merlin.development.targetgen.authoring.tool_registry import public_client_modules

    access_policy.require_current_policy()
    inputs = {
        f"phase1:client:{module}": str(_source(module).resolve())
        for module in (*public_client_modules(), "merlin.development.targetgen.authoring.tool_registry")
    }
    package = _source("merlin.development.backend").parent
    members = _python_members(package, "phase1:source:")
    for module in (
        "merlin.development.backend",
        "merlin.development.backend.context",
        "merlin.development.backend.session.inputs.run_record",
        "merlin.development.backend.session.inputs.implementation_inventory",
    ):
        if str(_source(module).resolve()) not in members.values():
            raise SpecError("phase-1 implementation has a missing or foreign source owner")
    inputs.update(members)
    startup = phase0_startup_inputs()
    inputs.update({f"phase1:startup:{name}": str(path) for name, path in startup.items()})
    inputs["phase1:startup:legacy_target_access"] = str(access_policy.resource_path())
    corpus_members = _python_members(_source("merlin.experiments.corpus").parent, "phase1:startup:corpus:")
    for module in ("release", "preparation", "admission"):
        if str(_source(f"merlin.experiments.corpus.{module}").resolve()) not in corpus_members.values():
            raise SpecError("corpus implementation has a missing or foreign source owner")
    inputs.update(corpus_members)
    execution_members = _python_members(_source("merlin.experiments.execution").parent, "phase1:startup:execution:")
    for module in ("_protocol", "chia_native", "native_guardian", "native_supervisor"):
        if str(_source(f"merlin.experiments.execution.{module}").resolve()) not in execution_members.values():
            raise SpecError("managed execution implementation has a missing or foreign source owner")
    inputs.update(execution_members)
    evaluation_members = _python_members(_source("merlin.evaluation.capsule").parent, "phase1:startup:evaluation:")
    for module in ("runner", "operation", "model", "bundle"):
        if str(_source(f"merlin.evaluation.capsule.{module}").resolve()) not in evaluation_members.values():
            raise SpecError("capsule evaluation has a missing or foreign source owner")
    inputs.update(evaluation_members)
    for key, module in (
        ("environment_parser", "merlin.target.registry.descriptor_loading"),
        ("source_digest", "merlin.common.digest"),
        ("snapshot_copy", "merlin.common.storage.content_store"),
        ("frozen_host_transport", "merlin.experiments.frozen_python"),
        ("frozen_import_resolver", "merlin.common.frozen_imports"),
        ("arrival_stamp", "merlin.development.backend.providers.execution"),
        ("sandbox_bwrap", "merlin.development.targetgen.sandbox.grants"),
        ("sandbox_mount_visibility", "merlin.development.targetgen.sandbox.mounts"),
        ("capsule_diagnostics", "merlin.evaluation.capsule.operations.numerics"),
        ("capsule_bundle", "merlin.evaluation.capsule.models.bundle"),
        ("sandbox_host_surfaces", "merlin.development.targetgen.sandbox.host_surfaces"),
        ("sandbox_toolchain", "merlin.development.targetgen.sandbox.toolchain"),
        ("sandbox_claude_home", "merlin.development.targetgen.sandbox.cleanroom"),
        ("source_discovery_helper", "merlin.common.digest"),
        ("provider_selection", "merlin.target.registry.identity"),
        ("provider_identity", "merlin.compiler.backend.providers"),
        ("provider_plugins", "merlin.target.registry.plugins"),
        ("provider_loaded_ownership", "merlin.runtime.backends.registry"),
        ("python_regex_scanner", "merlin.common.record_formats"),
        ("shared_access_policy", "merlin.common.access"),
        ("historical_target_access_policy", "merlin.experiments.definition"),
        ("answer_surface_policy", "merlin.development.targetgen.sandbox.answer_surfaces"),
        ("read_audit", "merlin.development.targetgen.sandbox.read_audit"),
        ("source_discovery", "merlin.experiments.runs.adapters"),
        ("corpus_workflow", "merlin.experiments.corpus.admission"),
        ("corpus_materialization", "merlin.development.targetgen.corpus.materialization"),
        ("corpus_description", "merlin.development.targetgen.corpus.materialization"),
        ("corpus_copy", "merlin.experiments.corpus.source_staging"),
        ("benchharness", "merlin.evaluation.harness"),
        ("tree_hash", "merlin.common.digest"),
        ("formal_grader", "merlin.development.backend.feedback.grading.formal"),
        ("compiler_library_contract", "merlin.development.targetgen.authoring.library_admission"),
        ("compiler_library_profile", "merlin.development.targetgen.authoring.profile_admission"),
        ("compiler_library_snapshot", "merlin.development.backend.optimization.campaign.compiler_library_state"),
        ("compiler_library_overlay", "merlin.development.backend.optimization.campaign.compiler_libraries"),
        ("compiler_library_inventory", "merlin.development.backend.optimization.campaign.compiler_libraries"),
        ("compiler_library_records", "merlin.development.backend.optimization.contracts"),
        ("rtl_check_runner", "merlin.development.targetgen.checks.rtl_check_runner"),
        ("rtl_check_compiler", "merlin.development.targetgen.checks.rtl_check_compiler"),
        ("rtl_checks", "merlin.target.isa.rocc.rtl_checks"),
        ("circt_gate", "merlin.development.targetgen.checks.circt_gate"),
        ("software_spec", "merlin.target.spec.software"),
        ("saved_model_model_execution_inputs", "merlin.runtime.execution.model_inputs"),
        ("saved_model_materialized_model_inputs", "merlin.compiler.frontend.inputs"),
        ("saved_model_route_before_build", "merlin.compiler.planning.model"),
        ("saved_model_complete_group_routing", "merlin.compiler.planning.complete"),
        ("saved_model_model_coverage", "merlin.compiler.planning.source_coverage"),
        ("saved_model_model_preflight", "merlin.compiler.frontend.staged"),
        ("saved_model_model_interface", "merlin.compiler.partition.regions"),
        ("saved_model_owned_host_model", "merlin.compiler.build.owned_sources"),
        ("saved_model_owned_shared_source", "merlin.compiler.build.owned_sources"),
        ("saved_model_owned_spatial_source", "merlin.compiler.build.spatial_sources"),
        ("saved_model_completed_model_link", "merlin.verification.references.saved_model"),
        ("saved_model_scalar_host_qualification", "merlin.verification.scalar_host"),
        ("mesh_selection", "merlin.compiler.selection.kernels"),
        ("mesh_execution", "merlin.runtime.dispatch.mesh"),
        ("mesh_backend", "merlin.runtime.dispatch.mesh_driver"),
        ("mesh_model", "merlin.runtime.dispatch.mesh_driver"),
        ("mesh_certification", "merlin.verification.references.mesh_certification"),
        ("mesh_reference", "merlin.verification.references.mesh_arithmetic"),
        ("quantization_recipe", "merlin.compiler.quantization.capabilities"),
        ("quantization_recipe_records", "merlin.compiler.quantization.recipe"),
        ("quantization_recipe_static", "merlin.compiler.quantization.recipe"),
        ("quantization_recipe_dynamic", "merlin.compiler.quantization.recipe"),
        ("quantization_recipe_cli", "merlin.compiler.quantization.capabilities"),
        ("quantization_admission", "merlin.compiler.quantization.admission"),
        ("quantization_capture_projection", "merlin.compiler.quantization.torchao.recipes"),
        ("quantization_methodology", "merlin.compiler.quantization.methodology"),
        ("numerical_recipe_policy", "merlin.development.targetgen.policies.precision"),
        ("quantization_output_path", "merlin.target.capabilities.output_path"),
        ("quantization_formats", "merlin.common.target_data"),
        ("task_prompt", "merlin.development.targetgen.authoring.prompt_slots"),
        ("shared_pass_requirement", "merlin.development.targetgen.authoring.shared_pass_contract"),
        ("shared_pass_context", "merlin.development.targetgen.authoring.pass_selection"),
        ("target_schedule", "merlin.compiler.lower.device.target"),
        ("device_target_passes", "merlin.compiler.lower.device.target"),
        ("general_compiler_contract", "merlin.development.targetgen.authoring.compiler_contract"),
        ("compiler_prompt_renderer", "merlin.development.targetgen.authoring.prompt"),
        ("semantic_search", "merlin.development.targetgen.authoring.semantic_search"),
        ("linalg_inventory", "merlin.target.contract.interfaces.linalg"),
        ("public_compile_sources", "merlin.development.testgen.public_sources.public_compile_sources"),
        ("public_scale_sources", "merlin.development.testgen.public_sources.public_scale_sources"),
        ("public_interface_emitter", "merlin.target.contract.interfaces.grammar"),
        ("instruction_model", "merlin.target.spec.instructions"),
        ("scalar_computation", "merlin.target.spec.semantic_sources"),
        ("instruction_signature", "merlin.target.spec.instructions"),
        ("readback_policy", "merlin.target.contract.harness.readback_policy"),
        ("readback_codec", "merlin.runtime.harness.output_decoding"),
        ("binary_readback_codec", "merlin.runtime.harness.output_decoding"),
        ("simulation_certificate", "merlin.development.targetgen.oracles.certificates"),
        ("capsule_source_identity", "merlin.evaluation.capsule.evidence.source_custody"),
    ):
        inputs[f"phase1:startup:{key}"] = str(_source(module).resolve())
    instructions = _source("merlin.target.spec.instructions")
    if instructions.is_symlink() or not instructions.is_file():
        raise SpecError("phase-1 instruction semantics source is absent or symlinked")
    inputs["phase1:startup:instruction_semantics"] = str(instructions.resolve())
    try:
        table = json.loads(startup["provenance_binding"].read_text())["entrypoints"]["capsule_bench"]
        if not isinstance(table, dict) or "default" not in table:
            raise ValueError("missing default capsule-bench binding")
        relatives = [_relative_entrypoint(value) for value in table.values()]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SpecError("phase-1 native entrypoint binding is invalid") from exc

    root = Path(repo).resolve()
    if descriptor is not None and Path(descriptor).is_file():
        import yaml

        from merlin.experiments.corpus.numeric_policy import numeric_profile_path
        from merlin.target.registry.descriptor_loading import (
            declared_vs_resolved_contract,
            load_target_experiment,
            selected_experiment_contract,
        )
        from merlin.target.spec.software import software_spec_path_for_recipe

        document = yaml.safe_load(Path(descriptor).read_bytes())
        if not isinstance(document, dict):
            raise SpecError("phase-1 descriptor must be a mapping")
        inputs.update(_selected_provider_inputs(document.get("target")))
        selected_target = load_target_experiment(descriptor)
        capability_path = selected_experiment_contract(selected_target, environment=os.environ)
        if capability_path is not None:
            inputs["phase1:startup:target_contract"] = str(capability_path)
        if selected_target.sim_via == "chipyard":
            _, contract_path, agreement = declared_vs_resolved_contract(selected_target)
            if agreement != "agree" or contract_path is None:
                raise SpecError(f"phase-1 selected target contract is not agreed and resolvable: {agreement}")
            inputs["phase1:startup:target_contract"] = str(contract_path.resolve())
        profile = numeric_profile_path(document.get("numeric_profile"), repo=root)
        if profile is not None:
            # Membership binds absence of this declaration too: introducing one
            # on resume changes the input set rather than borrowing live policy.
            inputs["phase1:startup:numeric_profile"] = str(profile)
            if profile.is_file():
                selected_spec = software_spec_path_for_recipe(profile)
                if selected_spec is not None:
                    inputs["phase1:startup:numeric_software_spec"] = str(selected_spec)
    native_paths = {(root / relative).resolve() for relative in relatives}
    if any(not path.is_relative_to(root) for path in native_paths):
        raise SpecError("phase-1 native entrypoint binding escapes its checkout")
    selected = Path(entrypoint).resolve()
    if require_native and selected not in native_paths:
        raise SpecError("phase-1 native entrypoint is outside its declared repository binding")
    if selected in native_paths:
        harness = selected.parent
        inputs.update(_python_members(harness, "phase1:native:"))
        for name in ("_common.py", "sandbox_toolchain.py"):
            path = harness / name
            if path.is_symlink() or not path.is_file():
                raise SpecError(f"phase-1 native startup input is absent or symlinked: {path}")
            inputs[f"phase1:startup:native:{name}"] = str(path.resolve())
        if descriptor is None:
            raise SpecError("phase-1 native implementation binding requires its explicit descriptor")
    return inputs


def record(*, repo: Path, entrypoint: Path, descriptor: Path | None = None, require_native: bool = False) -> dict:
    """Create the source-binding value stored inside the existing native environment."""
    inputs = paths(repo=repo, entrypoint=entrypoint, descriptor=descriptor, require_native=require_native)
    return {
        "version": 1,
        "scope": SCOPE,
        "inputs": {name: {"path": path, "sha256": fingerprint(path)} for name, path in inputs.items()},
    }


def verify(
    expected: object, *, repo: Path, entrypoint: Path, descriptor: Path | None = None, require_native: bool = False
) -> None:
    """Refuse missing historical attribution or any current path/member/byte drift."""
    if not isinstance(expected, dict) or expected.get("version") != 1 or expected.get("scope") != SCOPE:
        raise SpecError("phase-1 implementation source record is missing or unsupported; create a new qualified run")
    observed = record(repo=repo, entrypoint=entrypoint, descriptor=descriptor, require_native=require_native)
    if expected != observed:
        raise SpecError("phase-1 implementation source identity changed; create a new qualified run")
