"""Host-owned Phase 0 generation implementation."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from contextlib import closing
from pathlib import Path

import yaml

from merlin.development.targetgen.corpus import binding as CS  # noqa: E402
from merlin.development.targetgen.corpus import model_builders as _merlin_model_builders_1
from merlin.development.testgen.evidence.program_admission import (
    accelerator_placement_refusal,
    entry_refusal_is_final,
    screen_written,
)
from merlin.development.testgen.generation.capsule_writer import (
    SYNTH_ROLE,
    UnprovableForbid,
    _finish_capsule,
    _write_capsule,
    _write_capsule_source,
)
from merlin.development.testgen.generation.claim_boundary import assert_no_claim_capsules, held_out_models
from merlin.development.testgen.generation.manifest import _require_distinct_corpus_destinations
from merlin.development.testgen.generation.provenance import (
    _capture_failure_reason,
    _scrub_capsule_dir,
    update_provenance_manifest,
)
from merlin.development.testgen.generation.staging import PreparedWrite, prepare_staged, write_staged
from merlin.development.testgen.generation.writing import write_generation
from merlin.development.testgen.performance.scope import enforcement_problems, validate_roles
from merlin.development.testgen.profiles.loading import load_profile, validate_profile_inputs
from merlin.development.testgen.profiles.screens import diagnostic_entry, screen_entry
from merlin.development.testgen.profiles.sweep_axes import _performance_facts, _resolve_flat_extents
from merlin.development.testgen.profiles.sweep_expansion import expand_sweeps
from merlin.development.testgen.sealed_capture.generation import verified_capture_failure
from merlin.development.testgen.sealed_capture.selection import (
    _prepare_model_capture_entry,
    _with_selected_model_recipe,
)
from merlin.target.registry.descriptor_loading import load_target_experiment  # noqa: E402

# ---- generation --------------------------------------------------------------------------------


def _select_generation_scope(entries: list[dict], scope: str) -> tuple[list[dict], list[dict]]:
    """Defer only the existing nonfunctional performance category, retaining full records."""
    if scope not in {"full", "phase1-functional"}:
        raise ValueError("generation_scope must be full or phase1-functional")
    if scope == "full":
        return entries, []
    functional, deferred = [], []
    for entry in entries:
        if entry.get("cat") == "_perf":
            if not isinstance(entry.get("performance"), dict) or not entry["performance"]:
                raise ValueError("deferred performance member lacks its explicit claim")
            deferred.append(copy.deepcopy(entry))
        else:
            functional.append(entry)
    return functional, deferred


def _screen_selected_entry(
    spec: dict,
    entry: dict,
    *,
    binding: CS.CorpusBinding,
    host_capabilities: dict | None = None,
    capsule: dict | None = None,
) -> dict:
    """Screen the numeric binding that the writer selects for this entry."""
    _, selected = CS.entry_binding(entry, binding)
    return screen_entry(
        spec,
        entry,
        defaults={"operand_dtype": selected.operand_dtype, "accumulator_dtype": selected.accum_dtype},
        host_capabilities=host_capabilities,
        capsule=capsule,
    )


def _descriptor_for(target: str) -> Path:
    from merlin.target.registry.descriptor_loading import standard_descriptor_path

    return standard_descriptor_path(target)


def _ensure_contract_on_path(descriptor: Path) -> None:
    """If the descriptor names an out-of-tree ``target_contract`` (e.g. a contract that lives under the
    target's own package), prepend its package root to ``MERLIN_TARGET_PATH`` so the registry
    resolves the manifest. Read from the descriptor, so it stays target-agnostic."""
    from merlin.common.paths import repo_root

    raw = yaml.safe_load(descriptor.read_text())
    tc = (raw.get("hardware_spec") or {}).get("target_contract")
    if not tc:
        return
    from merlin.target.registry.identity import effective_target_path

    pkg = (repo_root() / tc).resolve().parent.parent  # .../contracts/target_contract.yaml -> package root
    # Prepend to the EFFECTIVE selection: an unset variable means the in-repo support default, and
    # writing only this package would silently deselect it.
    cur = effective_target_path()
    if str(pkg) not in cur.split(os.pathsep):
        os.environ["MERLIN_TARGET_PATH"] = os.pathsep.join([str(pkg), cur]) if cur else str(pkg)


def _verify_derivation_evidence(conformance_spec: str | Path, evidence) -> None:
    """Refuse a derived corpus when its selected provider/facts changed."""
    requirement = yaml.safe_load(Path(conformance_spec).read_text(encoding="utf-8")) or {}
    if not isinstance(requirement, dict):
        raise ValueError("derived conformance requirement must be a mapping")
    execution = (requirement.get("derivation") or {}).get("phase0_execution") or {}
    expected = evidence.derivation_identity
    missing = sorted(key for key in expected if key not in execution)
    if missing:
        raise ValueError(
            "derived requirement lacks selected capability identity "
            f"({', '.join(missing)}); rerun corpus derive with the selected provider"
        )
    changed = sorted(key for key, value in expected.items() if execution[key] != value)
    if changed:
        raise ValueError(
            "stale capability evidence: "
            f"{', '.join(changed)} changed since corpus derivation; "
            "rerun corpus derive with the selected provider and facts"
        )


def _frozen_application_captures(root: Path, manifest: dict) -> dict[str, Path]:
    """Resolve saved capture bytes from this run's exported evidence only."""
    captures = {}
    for source in manifest.get("sources") or []:
        role = source.get("role")
        if not isinstance(role, str) or not role.startswith("application-capture:"):
            continue
        label = role.partition(":")[2]
        member = Path(source["path"])
        if not label or label in captures or member.is_absolute() or ".." in member.parts:
            raise ValueError("invalid or duplicate frozen application capture")
        path = root / member
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError(f"frozen application capture differs from exported evidence: {label}")
        captures[label] = path
    return captures


def _frozen_performance_basis(root: Path, manifest: dict, target: str) -> tuple[dict | None, str | None]:
    """Load the exported capture basis by its manifest-bound bytes, not an ambient path.

    Evidence exported before the basis existed names no such member: that is reported as no
    basis (the capture-shape sweeps then record why they were skipped), never as a basis.
    A member the manifest does name must be present, direct and byte-identical.
    """
    member = "coverage/performance-basis.json"
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("frozen evidence manifest has no artifact roster")
    if member not in artifacts:
        return None, None
    identity = artifacts[member]
    path = root / member
    if not isinstance(identity, dict) or path.is_symlink() or not path.is_file():
        raise ValueError("frozen performance basis is absent or indirect")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != identity.get("sha256") or len(raw) != identity.get("size_bytes"):
        raise ValueError("frozen performance basis differs from exported evidence")
    basis = json.loads(raw)
    if basis.get("schema") != "merlin.phase0.performance_basis.v1" or basis.get("target") != target:
        raise ValueError("frozen performance basis has wrong schema or target")
    return basis, digest


def generate_target(
    target: str,
    *,
    descriptor: str | Path | None = None,
    output_root: str | Path | None = None,
    profiles_root: str | Path | None = None,
    recipe: str | Path | None = None,
    software_spec: str | Path | None = None,
    capability_contract: str | Path | None = None,
    hardware_spec: str | Path | None = None,
    rtl_facts: str | Path | None = None,
    evidence_root: str | Path | None = None,
    evidence_input: str | Path | None = None,
    evidence_mode: str | None = None,
    performance_template: str | Path | None = None,
    generation_scope: str = "full",
    conformance_spec: str | Path | None = None,
    synth_profile: str | Path | None = None,
    smt_profile: str | Path | None = None,
    hidden_profile: str | Path | None = None,
    chain_profile: str | Path | None = None,
    prohibited_instruction_roles: list[str] | None = None,
    component_only: bool = False,
    component_coverage: str | Path | None = None,
    hardware_intake=None,
    software_intake=None,
    operator_schema_intake=None,
    arithmetic_intake=None,
    packing_intake=None,
    source_preparation: str | None = None,
    operator_exclusion=None,
) -> list[Path]:
    from merlin.common.paths import checkout_root
    from merlin.development.testgen.generation.manifest import (
        _form_perf_coverage,
        _instruction_policy,
        _is_roster_capsule,
        _member_path,
        _prune_superseded_synth,
        _qualifying_group_capsules,
        _with_candidate_policy,
        _with_reference_gate,
        require_enforceable_policy,
        withhold_unaccelerated,
    )

    if output_root is None:
        raise ValueError("Phase 0 requires an explicit output_root; generated capsules must not default to source data")
    from merlin.development.testgen.public_sources.operator_exclusion import PublicationRefusal
    from merlin.development.testgen.public_sources.operator_exclusion import (
        require_selection as require_operator_exclusion,
    )

    require_operator_exclusion(operator_exclusion, output_root, evidence_root)
    if generation_scope not in {"full", "phase1-functional"}:
        raise ValueError("generation_scope must be full or phase1-functional")
    if component_only and generation_scope != "full":
        raise ValueError("functional generation scope is separate from component-only generation")
    if type(component_only) is not bool:
        raise TypeError("component_only must be an explicit Boolean")
    if component_coverage is not None and not component_only:
        raise ValueError("component_coverage requires the explicit independent component_only mode")
    if hardware_intake is not None and not component_only:
        raise ValueError("independent hardware authority requires component_only generation")
    if software_intake is not None and (not component_only or hardware_intake is None):
        raise ValueError("independent minimal software requires component_only and independently issued hardware")
    if software_intake is not None:
        from merlin.development.testgen.intake.software import IndependentSoftwareIntake

        if type(software_intake) is not IndependentSoftwareIntake or software_intake.hardware is not hardware_intake:
            raise ValueError("component hardware and minimal software must share exact independently issued inputs")
        software_intake.verify()
    source_components = (
        component_only and hardware_intake is not None and software_intake is not None and capability_contract is None
    )
    from merlin.development.testgen.components.source_performance import require_selection

    require_selection(source_preparation, source_components=source_components)
    if source_components and evidence_input is not None:
        raise ValueError("fresh source-only components require live source replay, not exported backend evidence")
    if source_components and component_coverage is None:
        raise ValueError(
            "fresh source-only components require automatic mandatory coverage, not authored teaching samples"
        )
    if operator_schema_intake is not None and (
        not component_only or software_intake is None or component_coverage is None
    ):
        raise ValueError("independent operator schemas require explicit minimal software and component coverage")
    if arithmetic_intake is not None and (not component_only or hardware_intake is None or component_coverage is None):
        raise ValueError("local arithmetic requires independent hardware and explicit component coverage")
    if packing_intake is not None and (not component_only or hardware_intake is None or component_coverage is None):
        raise ValueError("local packing requires independent hardware and explicit component coverage")
    if component_only:
        from merlin.development.testgen.components.component_generation import require_inputs

        if Path(output_root).exists() or Path(output_root).is_symlink():
            raise ValueError(
                "component generation requires a fresh output_root; previous corpus/history is not an input"
            )
        require_inputs(
            descriptor=descriptor,
            recipe=recipe,
            performance_template=performance_template,
            profiles_root=profiles_root,
            evidence_mode=evidence_mode,
            evidence_input=evidence_input,
            conformance_spec=conformance_spec,
            synth_profile=synth_profile,
            smt_profile=smt_profile,
            hidden_profile=hidden_profile,
        )
    profile_inputs = dict(
        profiles_root=profiles_root,
        recipe=recipe,
        performance_template=performance_template,
        conformance_spec=conformance_spec,
        synth_profile=synth_profile,
        smt_profile=smt_profile,
        hidden_profile=hidden_profile,
        chain_profile=chain_profile,
    )
    if software_spec is not None:
        profile_inputs["software_spec"] = software_spec
    validate_profile_inputs(**profile_inputs)
    if profiles_root is None and recipe is None:
        raise ValueError("Phase 0 requires explicit recipe inputs or profiles_root before descriptor setup")
    if checkout_root() is None and (descriptor is None or (profiles_root is None and recipe is None)):
        raise ValueError("installed Phase 0 requires explicit profiles_root or recipe, descriptor and output_root")
    # The profile selector and the hardware descriptor are separate identities:
    # a profile can be reused with an explicitly supplied out-of-tree target.
    explicit_descriptor = descriptor is not None
    descriptor = Path(descriptor).expanduser().resolve() if descriptor is not None else _descriptor_for(target)
    if evidence_input is None and not source_components:
        _ensure_contract_on_path(descriptor)
    te = load_target_experiment(descriptor)
    _require_distinct_corpus_destinations(te, output_root=output_root, evidence_root=evidence_root)
    hardware_target = te.target if explicit_descriptor else target
    profile = load_profile(
        target,
        descriptor=descriptor,
        diagnostic=evidence_mode == "diagnostic",
        **({"include_holdouts": False} if component_only else {}),
        **{key: value for key, value in profile_inputs.items() if value is not None},
    )
    from merlin.development.testgen.components.source_performance import SCHEMA_V2, verify_curriculum_template

    if source_preparation == SCHEMA_V2:
        verify_curriculum_template(performance_template, software=software_intake, hardware=hardware_intake)
    if profile.get("capsule_policy") == "derived_only":
        # Never substitute the old authored/reference corpus for a missing
        # derivation. Functional membership must come from new byte-bound inputs.
        accepted_synthesis = {"verified"}
        if evidence_mode == "diagnostic":
            accepted_synthesis.add("incomplete_diagnostic")
        if conformance_spec is None or profile.get("_synth_verification", {}).get("status") not in accepted_synthesis:
            raise ValueError(
                "derived-only Phase 0 requires fresh digest-bound conformance and synthesis inputs; "
                "run merlin experiment corpus derive first"
            )
        authored = [
            e.get("name")
            for e in profile.get("capsules", [])
            if e.get("source_role")
            not in {
                "derived_sweep",
                "model_derived",
                "solver_derived",
                "materialized_iteration_capture",
                "materialized_performance_capture",
            }
        ]
        if authored:
            raise ValueError(f"derived-only Phase 0 refuses authored capsule membership: {authored}")
    if evidence_mode not in (None, "diagnostic", "verified"):
        raise ValueError("evidence_mode must be diagnostic or verified")
    if (
        evidence_mode == "verified"
        and evidence_input is None
        and software_spec is None
        and not profile.get("_software_spec_path")
    ):
        raise ValueError("verified Phase 0 requires selected software and hardware evidence")
    evidence = None
    evidence_manifest = None
    if evidence_input is not None or software_spec is not None or profile.get("_software_spec_path"):
        from merlin.development.testgen.evidence.evidence import export_evidence, load_exported_evidence
        from merlin.development.testgen.evidence.selection import select_evidence

        if evidence_input is not None:
            evidence = load_exported_evidence(Path(evidence_input))
            if evidence.target != hardware_target:
                raise ValueError("frozen hardware evidence target differs from the descriptor")
            captured_specs = {source.sha256 for source in evidence.source_snapshots if source.role == "software-spec"}
            if profile.get("_software_spec_identity", {}).get("sha256") not in captured_specs:
                raise ValueError("selected software spec differs from the exact captured evidence")
        else:
            evidence = select_evidence(
                hardware_target,
                descriptor=descriptor,
                capability_contract_path=capability_contract,
                facts_path=rtl_facts,
                software_spec=software_spec or profile.get("_software_spec_path"),
                hardware_spec=hardware_spec,
                conformance_spec=conformance_spec,
                prohibited_roles=tuple(prohibited_instruction_roles or ()),
                hardware_intake=hardware_intake,
                software_intake=software_intake,
                source_components=source_components,
            )
        if evidence_mode != "diagnostic" and evidence.status != "verified":
            raise ValueError(
                "verified Phase 0 requires reviewed, coherent evidence: "
                + "; ".join(
                    str(row.get("reason", row)) if isinstance(row, dict) else str(row) for row in evidence.diagnostics
                )
            )
        if profile.get("capsule_policy") == "derived_only":
            _verify_derivation_evidence(conformance_spec, evidence)
        artifact_root = Path(evidence_root) if evidence_root is not None else Path(output_root) / "_evidence"
        evidence_manifest = export_evidence(evidence, artifact_root)
    declared_claims = [str(model) for model in (getattr(te, "workload_spec", None) or {}).get("models") or ()]
    claim_plan = profile.get("_claim_model_evaluation")
    if profile.get("_synth_verification", {}).get("status") == "verified":
        if (
            not isinstance(claim_plan, dict)
            or claim_plan.get("schema") != "claim_model_evaluation_v1"
            or claim_plan.get("model_count") != len(declared_claims)
            or claim_plan.get("visibility") != "owner_only_after_phase1_freeze"
            or claim_plan.get("public_capsules_emitted") != 0
        ):
            raise ValueError("verified synthesis lacks an owner-only claim-model evaluation obligation")
    if (
        profile.get("_synth_verification", {"status": "absent"})["status"] == "unverified_legacy"
        and evidence_mode != "diagnostic"
    ):
        raise ValueError(
            "selected synthesized profile has no digest-bound conformance/recipe/workload inputs; "
            "regenerate, review, and select a new sidecar before verified Phase 0 execution"
        )
    selected = (
        {"contract": evidence.contract, "facts": evidence.loaded_facts, "taxonomy": evidence.isa_taxonomy}
        if evidence is not None
        else {}
    )
    if source_components:
        from merlin.development.testgen.components.component_source_binding import derive

        binding = derive(software_intake, hardware=hardware_intake, datapath=profile.get("datapath", {}))
    else:
        binding = CS.derive_binding(te, profile.get("datapath", {}), **selected)
    declared_roles = validate_roles(prohibited_instruction_roles)
    instruction_policy = _instruction_policy(hardware_target, declared_roles, evidence, rtl_facts)
    if declared_roles:
        require_enforceable_policy(evidence_mode, declared_roles, instruction_policy)
    out_root = Path(output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    # `sweeps:` (if any) expand into the same flat entries `capsules:` holds, so
    # everything downstream — builders, goldens, coverage — is unchanged.
    facts = (
        _performance_facts(hardware_target, evidence=evidence)
        if evidence is not None
        else _performance_facts(hardware_target)
    )
    _sweep_skips: list = []
    _runtime_blocked: list = []
    _performance_errors: list = []
    requirement_bytes = Path(conformance_spec).read_bytes() if conformance_spec is not None else None
    selected_requirement = yaml.safe_load(requirement_bytes) if requirement_bytes is not None else None
    if selected_requirement is not None and not isinstance(selected_requirement, dict):
        raise ValueError("selected conformance requirement must be a mapping")
    selected_performance_basis, performance_basis_sha256 = (
        _frozen_performance_basis(artifact_root, evidence_manifest, hardware_target)
        if evidence is not None and selected_requirement is not None
        else (None, None)
    )
    entries = expand_sweeps(
        profile,
        binding,
        trait_facts=facts,
        skipped=_sweep_skips,
        blocked_unimplemented=_runtime_blocked,
        errors=_performance_errors,
        selected_requirement=selected_requirement,
        requirement_sha256=hashlib.sha256(requirement_bytes).hexdigest() if requirement_bytes is not None else None,
        selected_performance_basis=selected_performance_basis,
        performance_basis_sha256=performance_basis_sha256,
        source_preparation=source_preparation,
        **({"evidence": evidence} if evidence is not None else {}),
    )
    assert_no_claim_capsules(entries, held_out_models(te))
    entries = [_resolve_flat_extents(e, binding) for e in entries]
    entries = [_with_candidate_policy(e, instruction_policy) for e in entries]
    entries = [_with_reference_gate(e, te, descriptor) for e in entries]
    entries, deferred_performance = _select_generation_scope(entries, generation_scope)
    component_identity = None
    component_plan = None
    component_coverage_report = None
    execution_budget = None
    execution_admission = None
    preallocation_denials = {}
    semantic_basis = None
    automatic_derivation = None
    if component_only:
        from merlin.development.testgen.components.component_generation import bind_entries
        from merlin.development.testgen.components.coverage_plan import ComponentSemanticBasis

        if component_coverage is not None:
            from merlin.development.testgen.components.automatic_selection import require_basis_selection

            require_basis_selection(component_coverage, recipe=recipe, software_intake=software_intake)
        semantic_basis = ComponentSemanticBasis.from_recipe(recipe)

        if component_coverage is not None:
            from merlin.development.testgen.components.automatic_selection import resolve
            from merlin.development.testgen.components.component_coverage_inputs import expand

            component_plan, automatic_derivation = resolve(
                component_coverage,
                evidence=evidence,
                semantic_basis=semantic_basis,
                hardware_intake=hardware_intake,
                software_intake=software_intake,
                output_root=artifact_root,
                operator_schema_intake=operator_schema_intake,
                arithmetic_intake=arithmetic_intake,
                packing_intake=packing_intake,
            )
            if source_components and automatic_derivation is None:
                raise ValueError(
                    "source-only admission requires automatic original-source obligations with unknown domains"
                )
            execution_budget = component_plan.to_dict().get("execution_budget")
            coverage_entries, component_coverage_report = expand(component_plan, binding=binding, evidence=evidence)
            if automatic_derivation is not None:
                from merlin.development.testgen.components.automatic_plan import required_unknown_rows

                component_coverage_report["automatic_derivation"] = automatic_derivation
                component_coverage_report["obligations"].extend(
                    required_unknown_rows(automatic_derivation["required_unknowns"])
                )
            coverage_entries = [_with_candidate_policy(e, instruction_policy) for e in coverage_entries]
            coverage_entries = [_with_reference_gate(e, te, descriptor) for e in coverage_entries]
            entries.extend(coverage_entries)
        if execution_budget is not None:
            from merlin.development.testgen.components.component_execution_budget import admit

            execution_admission = admit(entries, binding=binding, policy=execution_budget)
            component_coverage_report["execution_admission"] = execution_admission
            preallocation_denials = {
                row["name"]: row["reason"] for row in execution_admission["decisions"] if row["state"] != "admitted"
            }
        entries, component_identity = bind_entries(
            entries,
            evidence=evidence,
            profile=profile,
            recipe=recipe,
            performance_template=performance_template,
            coverage_plan_sha256=component_plan.source_sha256 if component_plan is not None else None,
            execution_budget=execution_budget,
            execution_admission=execution_admission,
            semantic_basis=semantic_basis,
            hardware_intake=hardware_intake,
            software_intake=software_intake,
            automatic_derivation=automatic_derivation,
            source_components=source_components,
            source_preparation=source_preparation,
        )
    semantics = (profile.get("datapath") or {}).get("numerical_semantics")
    if semantics is not None:
        semantics = copy.deepcopy(semantics)
        if evidence is not None:
            model_sources = [
                (str(source.path), source.sha256)
                for source in evidence.source_snapshots
                if source.role == "software-reference:numerical_model"
            ]
            semantics["model"]["source_bundle_sha256"] = hashlib.sha256(
                json.dumps(sorted(model_sources), separators=(",", ":")).encode()
            ).hexdigest()
        entries = [{**entry, "numerical_semantics": copy.deepcopy(semantics)} for entry in entries]
    from merlin.development.testgen.components.component_integer_bounds import preflight_entry

    for entry in entries:
        if (
            _merlin_model_builders_1.entry_component_program(entry, binding) is not None
            and entry["name"] not in preallocation_denials
        ):
            try:
                preflight_entry(entry, binding=binding)
            except ValueError as exc:
                preallocation_denials[entry["name"]] = str(exc)
    if evidence is not None and evidence.software_spec:
        screened = []
        for entry in entries:
            if entry["name"] in preallocation_denials:
                screened.append(entry)
                continue
            decision = _screen_selected_entry(
                evidence.software_spec,
                entry,
                binding=binding,
                host_capabilities=evidence.host_capabilities,
            )
            source_development = (
                source_preparation is not None
                and entry.get("op") == "component_program"
                and entry.get("cat") == "_perf"
            )
            screened.append(
                diagnostic_entry(entry, decision)
                if entry_refusal_is_final(entry, decision) and not source_development
                else entry
            )
        entries = screened
    from merlin.development.testgen.sealed_capture.generation import bind_source

    capture_selection_failure = None
    try:
        selected_capture = bind_source(
            [entry for entry in entries if entry["name"] not in preallocation_denials],
            verified=component_plan is not None or (evidence is not None and evidence_mode != "diagnostic"),
        )
    except (ValueError, RuntimeError) as exc:
        if component_plan is None:
            raise
        selected_capture = None
        capture_selection_failure = str(exc)
    writer_options = {"capture": selected_capture} if selected_capture is not None else {}
    if component_only:
        writer_options["component_only"] = True
    entries = [
        entry
        if entry["name"] in preallocation_denials
        else _prepare_model_capture_entry(
            entry,
            evidence_root=artifact_root if evidence is not None else None,
            target=hardware_target,
            binding=binding,
            capture=selected_capture,
        )
        for entry in entries
    ]
    from merlin.development.testgen.public_sources.functional_sources import (
        require as require_public_source_eligibility,
    )

    public_source_eligibility = require_public_source_eligibility(entries, profile, synth_profile, operator_exclusion)
    for _s in _sweep_skips:
        _why = _s.get("reason") or f"gate {(_s.get('gate') or {}).get('outcome')}"
        _what = f"class {_s['label']} of family {_s['family']}" if _s.get("label") else f"family {_s['family']}"
        print(f"  [skip] performance {_what}: {_why}")
    template = copy.deepcopy(profile.get("_performance_template") or {})
    declared_families = [dict(row) for row in (template.get("families") or [])]
    # A requirement-selected pattern may derive several exact claim cohorts.
    # Keep both identities: the shared template's digest-bound pattern and each
    # concrete family with the frozen requirement row that produced it.
    derived_families: dict[str, dict] = {}
    for entry in entries:
        performance = entry.get("performance") or {}
        basis = performance.get("requirement_basis") or {}
        family = performance.get("family")
        if basis.get("axis") in {"scope.performance.required", "coverage.performance-basis.capture_shape"} and family:
            record = {
                "family": family,
                "claim": performance.get("claim"),
                "derived_from_pattern": basis.get("pattern_family"),
                "requirement_basis": copy.deepcopy(basis),
                "fit_axes": ["K"],
                "comparison_roles": ["prediction", "measurement"],
            }
            if family in derived_families and derived_families[family] != record:
                raise ValueError(f"derived performance family {family!r} has divergent requirement provenance")
            derived_families[family] = record
    template["derived_families"] = [derived_families[name] for name in sorted(derived_families)]
    declared_families.extend(template["derived_families"])
    family_counts = {row["family"]: {"admitted_members": 0, "written_members": 0} for row in declared_families}
    for entry in entries:
        family = (entry.get("performance") or {}).get("family")
        if family and entry.get("cat") == "_perf":
            family_counts.setdefault(family, {"admitted_members": 0, "written_members": 0})
            family_counts[family]["admitted_members"] += 1
    # SCRUB EACH CAPSULE AS IT IS WRITTEN, not after the whole corpus succeeds. Scrubbing at the end
    # means one unrelated failure -- a capture that needs an external exporter, say -- aborts the run
    # with every capsule written so far still carrying its absolute `prov.weights_file` path. Measured:
    # a run that died on the last entry left `/scratch/.../capsule_m2m_<rand>/weights.safetensors` in
    # tracked MLIR across six capsules, in a repo that is published. Hygiene that only holds on the
    # happy path is not hygiene.
    #
    # ONE FAILING CAPSULE MUST NOT DESTROY THE WHOLE CORPUS. Letting the exception propagate meant a
    # single entry that needs an external exporter took every entry after it down with it: measured, a
    # capture that torch.export refuses (an LSTM whose `_flat_weights` are assigned rather than
    # registered) aborted the run before any of the tail-path sweep capsules were written, so a coverage
    # gap stayed open for a reason that had nothing to do with it. Failures are COLLECTED, reported by
    # name, and re-raised at the end -- the run still fails, it just fails after doing the work it could.
    return write_generation(
        writer_options=writer_options,
        operator_exclusion=operator_exclusion,
        PublicationRefusal=PublicationRefusal,
        _form_perf_coverage=_form_perf_coverage,
        _is_roster_capsule=_is_roster_capsule,
        _performance_errors=_performance_errors,
        _prune_superseded_synth=_prune_superseded_synth,
        _qualifying_group_capsules=_qualifying_group_capsules,
        _runtime_blocked=_runtime_blocked,
        _sweep_skips=_sweep_skips,
        artifact_root=artifact_root,
        binding=binding,
        capture_selection_failure=capture_selection_failure,
        claim_plan=claim_plan,
        component_coverage_report=component_coverage_report,
        component_identity=component_identity,
        component_only=component_only,
        declared_families=declared_families,
        declared_roles=declared_roles,
        deferred_performance=deferred_performance,
        entries=entries,
        evidence=evidence,
        evidence_manifest=evidence_manifest,
        evidence_mode=evidence_mode,
        facts=facts,
        family_counts=family_counts,
        generation_scope=generation_scope,
        hardware_intake=hardware_intake,
        hardware_target=hardware_target,
        instruction_policy=instruction_policy,
        out_root=out_root,
        preallocation_denials=preallocation_denials,
        profile=profile,
        public_source_eligibility=public_source_eligibility,
        requirement_bytes=requirement_bytes,
        selected_capture=selected_capture,
        selected_requirement=selected_requirement,
        semantic_basis=semantic_basis,
        software_intake=software_intake,
        source_components=source_components,
        source_preparation=source_preparation,
        target=target,
        template=template,
        withhold_unaccelerated=withhold_unaccelerated,
    )
