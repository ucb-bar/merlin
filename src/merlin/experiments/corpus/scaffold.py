"""Stage selected compiler resources, task inputs and explicit runtime contracts."""

from __future__ import annotations

import copy
import io
import shutil
from contextlib import redirect_stdout
from pathlib import Path

import yaml

from merlin.experiments.corpus.source_staging import (
    ordinary_tree,
)
from merlin.experiments.definition import SpecError, read_yaml
from merlin.experiments.corpus.preparation import _stage_capability_contract

# ---- preparation -------------------------------------------------------------------------------


def scaffold(
    te,
    corpus: Path,
    experiment: Path,
    *,
    private: Path,
    selected_facts: Path | None = None,
    selected_contract: tuple[Path, str] | None = None,
    phase1_gates: dict | None = None,
    optional_levels: tuple[str, ...] = (),
    selected_levels: tuple[str, ...] | None = None,
    selected_variants: tuple[str, ...] | None = None,
    lean_toolchain_manifest: Path | None = None,
    common_substrate: str | None = None,
    python_source_root: Path | None = None,
    host_index_bits: int | None = None,
    native_timeout_s: int | None = None,
    model_spike_timeout_s: int | None = None,
    gsim_readback_transport: str | None = None,
    operand_protocol_selection: tuple[Path, str] | None = None,
    operand_protocol_revision: tuple[Path, str] | None = None,
    kernel_timing_selection: tuple[Path, str] | None = None,
    logical_storage_budget_bytes: int | None = None,
    operator_public_members: tuple[str, ...] = (),
    public_scalability: dict | None = None,
    public_program_suite: dict | None = None,
    host_capability_selection: tuple[Path, str] | None = None,
    public_reference_manifest: Path | None = None,
    author_profile=None,
    author_profile_sources=None,
    author_profile_common_inputs=None,
    cca_bank: tuple[Path, str] | None = None,
) -> dict:
    """Stage only declared task/selfcheck/harness resources, never prior runs or bundles."""
    from merlin.common.digest import sha256_file
    from merlin.development.targetgen.sandbox.grants import resolve_grant
    from merlin.experiments.corpus.source_staging import copy_curated_harness, copy_input, copy_public_hardware
    from merlin.target.registry.descriptor_loading import load_target_experiment, observed_experiment_contract

    from merlin.experiments.corpus.admission import (
        DERIVED_RESOURCE_BOUND,
        derive_resource_bound,
        derive_release_admission,
    )

    substrate, compiler_root = compiler_source_selection(common_substrate, python_source_root)
    arms, variants = _bundle_selection(optional_levels, selected_levels, selected_variants)
    generated_sdk = None
    if python_source_root is None:
        from merlin.development.targetgen.authoring import tool_registry
        from merlin.development.targetgen.authoring.shared_sources import prepare_shared_source_release

        tools = tuple(dict.fromkeys(tool for arm in arms for tool in tool_registry.ARM_TOOLS[arm]))
        selected_root = private / "compiler-sdk"
        generated_sdk = prepare_shared_source_release(selected_root, python_source_root=compiler_root, tools=tools)
        compiler_root = selected_root.resolve(strict=True)
    experiment.mkdir(parents=True)
    inputs = {}
    for relative in ("task",):
        source = te.resource_path(relative)
        try:
            source.lstat()
        except FileNotFoundError:
            # Prompt rendering is shared code, not a target-authored placeholder file.
            # Keep a release-local directory for its generated bundle grant and bind
            # the source's absence explicitly in the private preparation record.
            (experiment / relative).mkdir()
            inputs[relative] = {"path": str(source), "present": False, "sha256": None}
        else:
            if not source.is_dir():
                raise SpecError("authored task resource must be a directory")
            inputs[relative] = {
                "path": str(source),
                "present": True,
                "sha256": copy_input(source, experiment / relative),
            }
    # Sealed admission requires bwrap. Its native broker replaces this entry with
    # the same public shim at launch; staging the shim now also makes the prepared
    # resource independently relocatable without copying grader dependencies.
    from merlin.common.paths import module_source_path

    shim = module_source_path("merlin.development.backend.tools.selfcheck")
    inputs["scripts/agent_selfcheck.py"] = {
        "path": str(shim),
        "sha256": copy_input(shim, experiment / "scripts/agent_selfcheck.py"),
    }
    document = copy.deepcopy(read_yaml(te.path))
    if host_capability_selection is not None:
        inputs["host_capability_selection"] = _select_host_capability(
            document, host_capability_selection, descriptor=te.path
        )
    if public_reference_manifest is not None:
        from merlin.common.paths import repo_root
        from merlin.experiments.corpus import public_references

        reference_root = experiment / "reference-kernels"
        references = public_references.stage(public_reference_manifest, reference_root)
        inputs["public_reference_sources"] = references
        public_references.declare(
            document,
            destination=reference_root,
            variants=variants,
            source_sha256=references["sha256"],
            repo=repo_root(),
        )
    if cca_bank is not None:
        # One coverage bank arm's pinned precomputed bank, beside its unchanged raw references.
        from merlin.common.paths import repo_root
        from merlin.development.backend.authoring.profiles import resolve as resolve_author_profile
        from merlin.experiments.corpus import public_references

        selected = resolve_author_profile(author_profile) if author_profile else None
        if (
            public_reference_manifest is None
            or type(cca_bank) is not tuple
            or len(cca_bank) != 2
            or selected is None
            or selected.bank_sha256 != cca_bank[1]
        ):
            raise SpecError("a CCA bank requires its bank-arm author profile, the same pin and public references")
        bank_root = experiment / public_references.BANK_GRANT
        inputs["cca_bank"] = public_references.stage_bank(Path(cca_bank[0]), cca_bank[1], bank_root)
        public_references.declare_bank(
            document, destination=bank_root, variants=variants, sha256=cca_bank[1], repo=repo_root()
        )
    if public_scalability is not None:
        from merlin.experiments.corpus.public_scalability import KEY, verify

        verify(Path(public_scalability["path"]), target=te.target)
        document.setdefault("workload_spec", {})[KEY] = copy.deepcopy(public_scalability)
    program_view = _stage_public_program_view(document, public_program_suite, experiment, target=te.target)
    if program_view is not None:
        inputs["public_program_sources"] = program_view
    if operator_public_members and not te.graded_release_admission:
        grading = document.get("grading") or {}
        cohort = grading.get("expected_cohort")
        if cohort:
            for field in ("source_capsules", "admitted_capsules"):
                cohort[field] += len(operator_public_members)
        if te.graded_include:
            grading["search_cohort"]["include_capsules"].extend(Path(member).name for member in operator_public_members)
    if phase1_gates is not None:
        document["phase1_gates"] = copy.deepcopy(phase1_gates)
    # A release owns the resources copied below; never retain a pointer to live authored inputs.
    document.pop("resources_root", None)
    document.pop("task_root", None)
    document.pop("contracts_root", None)
    document["capsule_corpus"] = str(corpus / te.capsule_corpus.name)
    hardware = document.get("hardware_spec") or {}
    from merlin.experiments.corpus.operand_relocation import existing_selection

    existing_observer = existing_selection(selected_contract, selected_facts)
    facts_source = selected_facts
    if existing_observer is not None:
        facts_source = existing_observer["facts"]
        inputs["rtl_facts_phase0_export"] = existing_observer["phase0_export"]
        if operand_protocol_selection is None:
            operand_protocol_selection = existing_observer["sidecar"]
    facts_root = None
    if selected_facts is not None:
        facts_root = experiment / "rtl_facts"
        digest = copy_input(facts_source, facts_root / "facts.json")
        inputs["rtl_facts"] = {"path": str(facts_source), "sha256": digest}
    operand_selection = None
    operand_revision = None
    if operand_protocol_revision is not None:
        if (
            operand_protocol_selection is None
            or type(operand_protocol_revision) is not tuple
            or len(operand_protocol_revision) != 2
        ):
            raise SpecError("observer revision requires an explicit selection and pinned review")
        review_source, review_sha = operand_protocol_revision
        staged_review = experiment / "contracts/operand-revision.json"
        digest = copy_input(review_source, staged_review, expected_sha256=review_sha, private=True)
        operand_revision = {"path": str(staged_review), "sha256": digest}
        inputs["operand_revision"] = {"path": str(review_source), "sha256": digest}
    checker_inputs = ()
    if operand_protocol_selection is not None:
        from merlin.target.isa.rocc.operand_protocol import selected_inputs

        if (
            type(operand_protocol_selection) is not tuple
            or len(operand_protocol_selection) != 2
            or not isinstance(operand_protocol_selection[0], Path)
            or facts_root is None
        ):
            raise SpecError("operand protocol selection requires a pinned sidecar and unchanged selected facts")
        sidecar, sidecar_sha256 = operand_protocol_selection
        staged_sidecar = experiment / "contracts/operand-protocol.json"
        digest = copy_input(sidecar, staged_sidecar, expected_sha256=sidecar_sha256)
        inputs["operand_protocol"] = {"path": str(sidecar), "sha256": digest}
        staged_facts = experiment / "contracts/operand-facts.json"
        facts_digest = copy_input(
            facts_source, staged_facts, expected_sha256=inputs["rtl_facts"]["sha256"], private=True
        )
        operand_selection = {
            "original_facts": {"path": str(staged_facts), "sha256": facts_digest},
            "sidecar": {"path": str(staged_sidecar), "sha256": digest},
        }
        checker_inputs = selected_inputs(operand_selection)
    if selected_contract is not None:
        contract_source, contract_sha256 = selected_contract
        staged_contract = experiment / "contracts/target_contract.yaml"
        contract_record = _stage_capability_contract(
            contract_source,
            contract_sha256,
            staged_contract,
            native_timeout_s,
            gsim_readback_transport,
            model_spike_timeout_s=model_spike_timeout_s,
            operand_selection=operand_selection,
            operand_revision=operand_revision,
            kernel_timing_selection=kernel_timing_selection,
            logical_storage_budget_bytes=logical_storage_budget_bytes,
        )
        hardware["target_contract"] = str(staged_contract)
        inputs["capability_contract"] = contract_record
    elif te.declared_contract:
        raise SpecError("descriptor declares a capability contract without a selected Phase 0 export; freeze a new run")
    elif (
        native_timeout_s is not None
        or model_spike_timeout_s is not None
        or gsim_readback_transport is not None
        or operand_selection is not None
        or kernel_timing_selection is not None
        or logical_storage_budget_bytes is not None
    ):
        raise SpecError("execution selection requires an original Phase 0 contract")
    hardware_root = None
    if te.hwbringup_set:
        source = resolve_grant(te.hwbringup_set)
        hardware_root = experiment / "hardware_spec/hwbringup"
        digest, links = copy_public_hardware(source, hardware_root)
        hardware["hwbringup_set"] = str(hardware_root)
        inputs["hardware_spec/hwbringup"] = {"path": str(source), "sha256": digest, "materialized_links": links}
    if te.isa_headers:
        staged_headers = []
        source_root = resolve_grant(te.hwbringup_set) if te.hwbringup_set else None
        for index, declared in enumerate(te.isa_headers):
            source = resolve_grant(declared)
            if source_root is not None and source.is_relative_to(source_root):
                staged = hardware_root / source.relative_to(source_root)
            else:
                staged = experiment / "hardware_spec/isa_headers" / f"{index:02d}-{source.name}"
                if not source.is_file():
                    raise SpecError(f"declared ISA header is not a regular file: {source}")
                before = sha256_file(source)
                staged.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, staged, follow_symlinks=True)
                if sha256_file(source) != before or sha256_file(staged) != before:
                    raise SpecError("ISA header changed while copying; prepare a new release")
            if not staged.is_file() or staged.is_symlink():
                raise SpecError(f"staged ISA header is not an ordinary file: {staged}")
            staged_headers.append(str(staged))
        hardware["isa_headers"] = staged_headers
    if hardware:
        document["hardware_spec"] = hardware
    if te.curated_harness:
        relative = Path(te.curated_harness)
        if relative.is_absolute() or ".." in relative.parts:
            raise SpecError("curated harness must be a declared contained experiment resource")
        source = te.resource_path(relative)
        digest, links = copy_curated_harness(source, experiment / relative)
        inputs[relative.as_posix()] = {"path": str(source), "sha256": digest, "materialized_file_links": links}
    resource_bound = (document.get("grading") or {}).get("resource_bound") or {}
    if resource_bound.get("derive") is not None:
        if resource_bound.get("derive") != DERIVED_RESOURCE_BOUND:
            raise SpecError(f"unknown resource_bound derivation {resource_bound.get('derive')!r}")
        if resource_bound.get("required_admitted_models") or resource_bound.get("exclude_capsules"):
            raise SpecError("a derived resource policy cannot also carry hand-maintained model names")
        derived_bound = derive_resource_bound(corpus)
        document["grading"]["resource_bound"] = {
            key: value for key, value in {**resource_bound, **derived_bound}.items() if key != "derive" and value != []
        }
        inputs["resource_bound"] = {"policy": DERIVED_RESOURCE_BOUND, **derived_bound}
    descriptor = experiment / "target_experiment.yaml"
    descriptor.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    if te.graded_release_admission:
        staged = load_target_experiment(descriptor)
        with observed_experiment_contract(staged):
            derived = derive_release_admission(staged)
        grading = document["grading"]
        grading.pop("release_admission")
        grading.update({key: value for key, value in derived.items() if key != "resource_decisions"})
        descriptor.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        inputs["release_admission"] = {
            "policy": "derive_from_corpus_v1",
            "public_source": derived["expected_cohort"]["source_capsules"],
            "public_admitted": derived["expected_cohort"]["admitted_capsules"],
            "hidden_source": derived["hidden_capability_admission"]["source_capsules"],
            "hidden_admitted": derived["hidden_capability_admission"]["admitted_capsules"],
            "resource_decisions": derived["resource_decisions"],
        }
    # Generate fresh declarations, not copies that might still point at the old corpus.
    from merlin.development.targetgen.authoring.experiment_bundles import materialize_bundles
    from merlin.development.targetgen.sandbox.toolchain import ToolchainPaths

    promoted = load_target_experiment(descriptor)
    toolchain = ToolchainPaths.from_checkout()
    llvm_root = Path(toolchain.llvm)
    if not llvm_root.is_dir():
        raise SpecError(
            f"selected LLVM/MLIR toolchain is absent: {llvm_root}; "
            "select an installed clang-23 with MERLIN_CLANG before preparing a release"
        )
    checkout_llvm = toolchain.repo / "third_party/llvm-install"
    # Native generation can explain unavailable contracts. Keep those diagnostics
    # in the private preparation record, never the public inspect response.
    diagnostics = io.StringIO()
    with redirect_stdout(diagnostics), observed_experiment_contract(promoted):
        materialize_bundles(
            promoted,
            experiment / "input_bundles",
            variants=variants,
            host_inputs=(
                str(private),
                *checker_inputs,
                *([str(corpus / "_phase0")] if (corpus / "_phase0").is_dir() else []),
            ),
            python_source_root=compiler_root,
            llvm_toolchain_root=llvm_root if llvm_root != checkout_llvm else None,
            rtl_facts_root=facts_root,
            common_substrate=substrate,
            lean_toolchain_manifest=lean_toolchain_manifest,
            host_index_bits=host_index_bits,
            arms=arms,
            **(
                {
                    "public_inputs": tuple(
                        [
                            *(
                                [str(Path(public_scalability["path"]) / "public_sources")]
                                if public_scalability is not None
                                else []
                            ),
                            *([program_view["path"]] if program_view is not None else []),
                        ]
                    )
                }
                if public_scalability is not None or program_view is not None
                else {}
            ),
        )
    inputs["bundle_generation"] = {"diagnostics": diagnostics.getvalue()}
    if author_profile:
        from merlin.development.backend.authoring.profiles import stage_release_bundles

        inputs["author_profile"] = stage_release_bundles(
            experiment,
            promoted,
            profile=author_profile,
            source_selection=author_profile_sources,
            common_inputs=author_profile_common_inputs,
        )
    inputs["bundle_generation"]["compiler_source"] = {
        "common_substrate": substrate,
        "python_source_root": str(compiler_root),
        **(
            {"preparation": "owned_sdk26_snapshot", "library_sha256": generated_sdk["library_sha256"]}
            if generated_sdk is not None
            else {}
        ),
    }
    if selected_levels is not None or selected_variants is not None:
        inputs["bundle_generation"]["selection"] = {"arms": list(arms), "variants": list(variants)}
    return inputs


def _select_host_capability(document, selection, *, descriptor, root=None):
    """Pin a reviewed support file while preserving the existing host package."""
    from dataclasses import replace

    from merlin.common.digest import sha256_file
    from merlin.common.paths import repo_root
    from merlin.target.registry.experiment_descriptor import HostLaneMatrix

    if not isinstance(selection, tuple) or len(selection) != 2:
        raise SpecError("host capability selection requires an exact source file and digest")
    source, digest = Path(selection[0]), selection[1]
    ordinary_tree(source)
    if (
        not source.is_absolute()
        or source != source.resolve()
        or not source.is_file()
        or not isinstance(digest, str)
        or sha256_file(source) != digest
    ):
        raise SpecError("selected host capability source changed or is indirect")
    owner = (root or repo_root()).resolve()
    matrix = HostLaneMatrix.from_mapping(document.get("host_lane"), descriptor=descriptor)
    if matrix is None:
        raise SpecError("host capability selection requires an existing declared host lane")
    lane = matrix.profiles[matrix.default]
    if lane.capability_spec is None or source != owner / lane.capability_spec:
        raise SpecError("host capability selection must retain the declared support file ownership")
    _, before = lane.resolve(root=owner, descriptor=descriptor)
    selected = replace(lane, capability_spec_sha256=digest)
    path, view, after = selected.resolve_capabilities(root=owner, descriptor=descriptor)
    if path != source or view["status"] != "reviewed" or after["package_sha256"] != before["package_sha256"]:
        raise SpecError("host capability selection changes package or lacks reviewed support")
    if sha256_file(source) != digest:
        raise SpecError("selected host capability source changed during selection")
    host = document["host_lane"]
    profile = host["profiles"][matrix.default] if "profiles" in host else host
    profile["capability_spec_sha256"] = digest
    return {
        "path": str(source),
        "sha256": digest,
        "profile": matrix.default,
        "capability_spec": lane.capability_spec,
        "package_sha256": before["package_sha256"],
        "scope": "reviewed source support declaration only; no numerical or runtime qualification",
    }


def verify_host_capability(document, record, *, descriptor, root=None):
    """Reopen the selected support/package before inspect or seal, without rewriting."""
    selected = copy.deepcopy(document)
    actual = _select_host_capability(
        selected, (Path(record["path"]), record["sha256"]), descriptor=descriptor, root=root
    )
    if actual != record or selected != document:
        raise SpecError("prepared host capability source, profile or package changed")


def _stage_public_program_view(document, selection, experiment, *, target):
    """Bind selected complete programs before admission; grant only public sources."""
    from merlin.experiments.corpus import public_program_suite as programs

    if selection is None:
        if (document.get("workload_spec") or {}).get(programs.KEY) is not None:
            raise SpecError("public program source refresh requires an explicit selected roster")
        return None
    # Validate the exact pin as well as the complete source and exclusion tree.
    path = programs._pin(selection)
    programs.read(path, target=target)
    view = programs.public_view(path, experiment / "public-program-sources", target=target)
    document.setdefault("workload_spec", {})[programs.KEY] = copy.deepcopy(selection)
    return view


def _bundle_selection(optional_levels, selected_levels, selected_variants):
    """Select declarations only; corpus and qualification policy stay unchanged."""
    from merlin.development.backend.session.treatments import LEVELS
    from merlin.development.targetgen.authoring.experiment_bundles import _ARMS

    historical = tuple(_ARMS) + tuple({"EL5": "merlin_isa", "EL6": "merlin_lean"}[level] for level in optional_levels)
    arms = historical
    if selected_levels is not None:
        available = {row["id"]: row["bundle_arm"] for row in LEVELS if row["bundle_arm"] in historical}
        if (
            not isinstance(selected_levels, tuple)
            or not selected_levels
            or len(set(selected_levels)) != len(selected_levels)
            or any(level not in available for level in selected_levels)
        ):
            raise SpecError("selected corpus levels must be a nonempty unique tuple of enabled levels")
        arms = tuple(available[level] for level in selected_levels)
    variants = ("public_v0", "realistic_v0", "hwbringup_v0")
    if selected_variants is not None:
        if (
            not isinstance(selected_variants, tuple)
            or not selected_variants
            or len(set(selected_variants)) != len(selected_variants)
            or any(variant not in variants for variant in selected_variants)
        ):
            raise SpecError("selected corpus variants must be a nonempty unique tuple of supported variants")
        variants = selected_variants
    return arms, variants


def compiler_source_selection(common_substrate=None, python_source_root=None):
    """Select current SDK sources or validate an explicitly selected reviewed current release."""
    from merlin.common.paths import python_source_dir
    from merlin.development.targetgen.authoring.tool_profile import (
        CURRENT_COMMON_SUBSTRATE,
        require_current_substrate,
    )

    version = CURRENT_COMMON_SUBSTRATE if common_substrate is None else common_substrate
    require_current_substrate(version)  # Retired or unknown versions refuse before output creation.
    root = python_source_dir() if python_source_root is None else Path(python_source_root)
    if not root.is_absolute() or root.resolve() != root or not root.is_dir():
        raise SpecError("compiler source selection requires an existing canonical absolute directory")
    if python_source_root is None:
        return version, root
    from merlin.development.targetgen.authoring.shared_sources import reviewed_shared_manifest

    reviewed_shared_manifest(root)
    return version, root
