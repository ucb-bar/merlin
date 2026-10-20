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

from merlin.common.digest import sha256_file as _sha
from merlin.common.source_membership import SourceMembershipError, python_members

from .. import access_policy
from ..adapters import _relative_entrypoint, phase0_startup_inputs
from ..spec import SpecError

PREFIXES = ("phase1:client:", "phase1:source:", "phase1:startup:", "phase1:native:")
SCOPE = "phase1-package-public-clients-startup-and-retained-native-harness-access-policy-v2"


def fingerprint(path: str | Path) -> str:
    """Bind files and directory trees without copying models or hardware checkouts.

    Directory symlinks are refused to avoid an incomplete closure or cycles. Files
    reached through a symlink bind both the link destination and its current bytes.
    """
    path = Path(path)
    if path.is_file():
        return _sha(path)
    if not path.is_dir():
        raise SpecError(f"input is absent or is not a regular file/directory: {path}")
    digest = hashlib.sha256()
    for member in sorted(path.rglob("*")):
        if member.is_symlink() and member.is_dir():
            raise SpecError(f"input directory contains a directory symlink: {member}; name its closure explicitly")
        if member.is_file():
            row = [str(member.relative_to(path)), _sha(member)]
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
    from merlin.runtime.backends.base import _assert_oot_plugin_ownership
    from merlin.targetgen.plugins import resolve_support
    from merlin.targetgen.target_registry import explicit_targets

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
    from merlin.targetgen.tool_registry import public_client_modules

    access_policy.require_current_policy()
    inputs = {
        f"phase1:client:{module}": str(_source(module).resolve())
        for module in (*public_client_modules(), "merlin.targetgen.tool_registry")
    }
    package = _source("merlin_experiments.phase1").parent
    members = _python_members(package, "phase1:source:")
    for module in (
        "merlin_experiments.phase1",
        "merlin_experiments.phase1.context",
        "merlin_experiments.phase1.run_inputs",
        "merlin_experiments.phase1.source_inputs",
    ):
        if str(_source(module).resolve()) not in members.values():
            raise SpecError("phase-1 implementation has a missing or foreign source owner")
    inputs.update(members)
    startup = phase0_startup_inputs()
    inputs.update({f"phase1:startup:{name}": str(path) for name, path in startup.items()})
    inputs["phase1:startup:legacy_target_access"] = str(access_policy.resource_path())
    corpus_members = _python_members(_source("merlin_experiments.corpus").parent, "phase1:startup:corpus:")
    for module in ("release", "preparation", "admission"):
        if str(_source(f"merlin_experiments.corpus.{module}").resolve()) not in corpus_members.values():
            raise SpecError("corpus implementation has a missing or foreign source owner")
    inputs.update(corpus_members)
    execution_members = _python_members(_source("merlin_experiments.execution").parent, "phase1:startup:execution:")
    for module in ("_protocol", "chia_native", "native_guardian", "native_supervisor"):
        if str(_source(f"merlin_experiments.execution.{module}").resolve()) not in execution_members.values():
            raise SpecError("managed execution implementation has a missing or foreign source owner")
    inputs.update(execution_members)
    for key, module in (
        ("environment_parser", "merlin.targetgen.corpora"),
        ("source_digest", "merlin.common.digest"),
        ("snapshot_copy", "merlin.common.content_store"),
        ("frozen_host_transport", "merlin_experiments.frozen_python"),
        ("frozen_import_resolver", "merlin.common.frozen_imports"),
        ("arrival_stamp", "merlin.common.arrival_stamp"),
        ("sandbox_bwrap", "merlin.targetgen.sandbox.bwrap"),
        ("sandbox_host_surfaces", "merlin.targetgen.sandbox.host_surfaces"),
        ("sandbox_toolchain", "merlin.targetgen.sandbox.toolchain"),
        ("sandbox_claude_home", "merlin.targetgen.sandbox.claude_home"),
        ("source_discovery_helper", "merlin.common.source_membership"),
        ("provider_selection", "merlin.targetgen.target_registry"),
        ("provider_identity", "merlin.targetgen.providers"),
        ("provider_plugins", "merlin.targetgen.plugins"),
        ("provider_loaded_ownership", "merlin.runtime.backends.base"),
        ("python_regex_scanner", "merlin.common.regex_scan"),
        ("shared_access_policy", "merlin.common.access"),
        ("historical_target_access_policy", "merlin_experiments.access_policy"),
        ("answer_surface_policy", "merlin.targetgen.sandbox.answer_surfaces"),
        ("read_audit", "merlin.targetgen.sandbox.read_audit"),
        ("source_discovery", "merlin_experiments.adapters"),
        ("corpus_workflow", "merlin_experiments.corpus.admission"),
        ("corpus_materialization", "merlin.targetgen.contract.materialize"),
        ("corpus_copy", "merlin_experiments.corpus.preparation"),
        ("benchharness", "merlin.benchharness"),
        ("tree_hash", "merlin.common.tree_hash"),
        ("formal_grader", "merlin_experiments.phase1.feedback.formal"),
        ("compiler_library_contract", "merlin.targetgen.compiler_library"),
        ("compiler_library_profile", "merlin.targetgen.compiler_profile_admission"),
        ("compiler_library_snapshot", "merlin_experiments.phase2.compiler_library_state"),
        ("compiler_library_overlay", "merlin_experiments.phase2.compiler_libraries"),
        ("compiler_library_inventory", "merlin_experiments.phase2.compiler_library_guidance"),
        ("compiler_library_records", "merlin_experiments.phase2.contracts"),
        ("rtl_check_runner", "merlin.targetgen.rtl_check_runner"),
        ("rtl_check_compiler", "merlin.targetgen.rtl_check_compiler"),
        ("rtl_checks", "merlin.targetgen.rtl_checks"),
        ("circt_gate", "merlin.targetgen.circt_gate"),
        ("software_spec", "merlin.targetgen.software_spec"),
        ("task_prompt", "merlin.targetgen.generate_prompt"),
        ("shared_pass_requirement", "merlin.targetgen.shared_pass_contract"),
        ("shared_pass_context", "merlin.targetgen.shared_pass_context"),
        ("general_compiler_contract", "merlin.targetgen.generalization_prompt"),
        ("semantic_search", "merlin.targetgen.semantic_search.search"),
        ("linalg_inventory", "merlin.targetgen.contract.linalg_iface"),
        ("public_compile_sources", "merlin_experiments.phase0.public_compile_sources"),
        ("public_scale_sources", "merlin_experiments.phase0.public_scale_sources"),
        ("public_interface_emitter", "merlin.targetgen.contract.interface_emit"),
        ("instruction_semantics", "merlin.targetgen.instruction_semantics"),
        ("readback_policy", "merlin.targetgen.contract.readback_policy"),
        ("readback_codec", "merlin.runtime.out_b64"),
        ("binary_readback_codec", "merlin.runtime.out_bin"),
        ("simulation_certificate", "merlin.targetgen.simulation_certificate"),
        ("capsule_source_identity", "merlin.targetgen.capsule_source_identity"),
    ):
        inputs[f"phase1:startup:{key}"] = str(_source(module).resolve())
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

        from merlin.targetgen.software_spec import software_spec_path_for_recipe
        from merlin.targetgen.target_experiment import (
            declared_vs_resolved_contract,
            load_target_experiment,
            selected_experiment_contract,
        )
        from merlin_experiments.corpus.numeric_policy import numeric_profile_path

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
