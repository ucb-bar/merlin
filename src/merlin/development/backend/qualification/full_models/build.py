"""Build-only qualification against operator-private frozen model inputs.

Full-model execution and representative-capsule L3 grades remain independent.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

from merlin.development.backend.qualification.full_models import audits as program_checks
from merlin.development.backend.qualification.full_models import (
    private_compilation_inputs as compilation_support,
)
from merlin.development.backend.qualification.full_models import (
    private_source_optimizations as source_optimizations,
)
from merlin.development.backend.qualification.full_models import private_static_products as static_products
from merlin.development.backend.qualification.full_models.audits import (
    audit_built_device_host_compute as _audit_built_device_host_compute,
)
from merlin.development.backend.qualification.full_models.audits import load_diagnostic_receipt
from merlin.development.backend.qualification.full_models.audits import (
    require_linked_symbols as _linked_symbols,
)
from merlin.development.backend.qualification.full_models.audits import (
    require_static_build_inputs as _require_static_build_inputs,
)
from merlin.development.backend.qualification.full_models.completion import (
    BUILD_BOARD_SCOPE,
    RESULT_SCHEMA,
    _complete_capture_identity,
    compile_saved_model,
    complete,
)
from merlin.development.backend.qualification.full_models.private_capture_roster import (
    captured_input_provenance as _captured_input_provenance,
)
from merlin.development.backend.qualification.full_models.private_capture_roster import (
    captured_programs as _captured_programs,
)
from merlin.development.backend.qualification.full_models.private_model_inputs import (
    _capture_tree_bindings as _verify_capture_bindings,
)
from merlin.development.backend.qualification.full_models.private_model_inputs import (
    _expected_input_provenance,
    _require_selected_input_bindings,
    _selected_admission_views,
)
from merlin.development.backend.qualification.full_models.private_model_inputs import (
    _recipe_derivation as _derive_recipe_binding,
)
from merlin.development.backend.qualification.full_models.private_model_inputs import (
    _tree as _verify_input_tree,
)
from merlin.development.backend.qualification.full_models.private_source_preflight import (
    _notify as _source_progress,
)
from merlin.development.backend.qualification.full_models.private_source_preflight import (
    source_result as _source_result,
)
from merlin.development.backend.qualification.provenance import private_model_math_policy as math_policy
from merlin.development.backend.qualification.provenance import private_source_demand as source_demand
from merlin.development.backend.qualification.provenance import private_source_freeze as source_freeze_api
from merlin.development.backend.qualification.provenance.groups import (
    convolution_input_preparations,
    source_group_roster,
    verify_routed_source_coverage,
)
from merlin.development.backend.qualification.provenance.groups import (
    eligible_source_identity as _eligible_source_identity,
)
from merlin.development.backend.qualification.provenance.groups import (
    join_routed_source_groups as _join_routed_source_groups,
)
from merlin.development.backend.qualification.provenance.groups import (
    routed_kernel_symbols as _routed_kernel_symbols,
)
from merlin.development.backend.qualification.provenance.private_source_demand import (
    exact_source_groups as _exact_source_groups,
)
from merlin.development.backend.qualification.source_witnesses import bucketize as bucketize_support
from merlin.development.backend.qualification.source_witnesses import control_guards as control_support
from merlin.development.backend.qualification.source_witnesses import data_movement
from merlin.development.backend.qualification.source_witnesses import integer_reductions as integer_support
from merlin.development.backend.qualification.source_witnesses import linalg_bodies as linalg_support
from merlin.development.backend.qualification.source_witnesses import ordered_scan as ordered_scan_support
from merlin.development.backend.qualification.source_witnesses import (
    private_f32_maximum_support as maximum_support,
)
from merlin.development.backend.qualification.source_witnesses import (
    private_f32_reduction_support as f32_sum_support,
)
from merlin.development.backend.qualification.source_witnesses import (
    private_host_source_dispatch as host_source_dispatch,
)
from merlin.development.backend.qualification.source_witnesses import (
    private_index_host_support as index_support,
)
from merlin.development.backend.qualification.source_witnesses import (
    private_linkage_support as linkage_support,
)
from merlin.development.backend.qualification.source_witnesses import (
    private_linked_elf_selection as linked_policy,
)
from merlin.development.backend.qualification.source_witnesses import (
    private_literal_arange_admission as arange_support,
)
from merlin.development.backend.qualification.source_witnesses import pure_stages as pure_stage
from merlin.development.backend.qualification.source_witnesses.integer_reductions import (
    contraction_zero_init_fill as _contraction_zero_init_fill,
)
from merlin.development.backend.qualification.source_witnesses.private_source_support_join import (
    linked_source_support_complete as _linked_source_support_complete,
)
from merlin.runtime.execution.model_inputs import file_sha256, strict_tree_sha256

# ---- private_full_models -----------------------------------------------------------------------


def run(
    submission: str | Path,
    private_spec: str | Path,
    *,
    target: str,
    required_models: Sequence[str],
    required_programs: Mapping[str, Sequence[str]],
    loader_env_requirements: Mapping[str, Mapping[str, Any]],
    out: str | Path,
    selected_model: str | None = None,
    diagnostic_model: str | None = None,
    prebuilt_receipts: Mapping[str, str | Path] | None = None,
    source_freeze: Mapping[str, Any] | None = None,
    source_freeze_root: str | Path | None = None,
    build_options: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
    linked_elf_admission=None,
    native_result_format_selection=None,
    source_only: bool = False,
    progress=None,
    retained_products=None,
) -> dict[str, Any]:
    """Build the frozen roster or all programs of one canonical/diagnostic partition.
    Build options select per-program full readback/profiling. Partitions retain
    the full roster/source freeze and cannot pass the full gate; source-only takes no compiler.
    Typed semantic refusals retain failed rows; identity/route/unexpected errors abort.
    """
    from merlin.compiler.lower.device.offload import BY_GROUP
    from merlin.compiler.planning.model import plan_before_build
    from merlin.development.backend.qualification.full_models.build_obligations import _verify_compiled_program
    from merlin.development.backend.qualification.full_models.private_full_models import (
        SCHEMA,
        SOURCE_PREFLIGHT_SCOPE,
        SourceAdmissionError,
        _authored_file,
        _file,
        _public_failure_reason,
        _source_obligations,
    )
    from merlin.development.backend.qualification.full_models.private_model_inputs import (
        _capture_tree_bindings,
        _recipe_derivation,
        _tree,
    )
    from merlin.runtime.execution.model_inputs import selected_firrtl
    from merlin.target.registry import identity as target_registry

    instruction_selection = linked_policy.freeze(linked_elf_admission, target=target)
    if native_result_format_selection is not None:
        from merlin.development.backend.qualification.provenance.private_result_format_selection import (
            FrozenSelection,
        )

        if (
            type(native_result_format_selection) is not FrozenSelection
            or native_result_format_selection.target != target
        ):
            raise ValueError("native result format selection has no matching original run owner")
        native_result_format_selection.verify()
    spec = Path(private_spec)
    if spec.is_symlink() or not spec.is_file():
        raise ValueError("operator-private full-model specification is absent or indirect")
    spec_bytes = spec.read_bytes()
    spec_digest = sha256(spec_bytes).hexdigest()
    document = yaml.safe_load(spec_bytes)
    frozen_authored = source_freeze_api.resolve_optional(
        spec.absolute(), source_freeze, source_freeze_root, target=target
    )
    rows = document.get("models") if isinstance(document, Mapping) else None
    names = (
        [row.get("id") for row in rows]
        if isinstance(rows, list) and all(isinstance(row, Mapping) for row in rows)
        else []
    )
    if (
        not isinstance(document, Mapping)
        or document.get("schema") != SCHEMA
        or document.get("target") != target
        or len(names) != len(set(names))
        or set(names) != set(required_models)
        or len(names) != len(required_models)
        or set(required_programs) != set(required_models)
        or set(loader_env_requirements) != set(required_models)
    ):
        raise ValueError("private full-model specification differs from required target roster")
    diagnostic = diagnostic_model is not None
    static_products.require_selection(
        retained_products, selected_model=selected_model, diagnostic=diagnostic, source_only=source_only
    )
    if selected_model is not None and (
        not isinstance(selected_model, str)
        or selected_model not in required_models
        or diagnostic
        or prebuilt_receipts is not None
    ):
        raise ValueError("canonical model partition requires one declared model and no diagnostic or prebuilt inputs")
    if not diagnostic and frozen_authored is None:
        raise ValueError("canonical private full-model builds require a run-owned authored-source freeze")
    if diagnostic:
        if diagnostic_model not in required_models or not isinstance(prebuilt_receipts, Mapping):
            raise ValueError("prebuilt diagnostic must select one required model and its receipts")
        if set(prebuilt_receipts) != set(required_programs[diagnostic_model]):
            raise ValueError("prebuilt diagnostic receipts differ from the complete selected model")
    elif prebuilt_receipts is not None:
        raise ValueError("prebuilt receipts cannot enter the canonical full-model build")
    if source_only and (diagnostic or submission is not None):
        raise ValueError("source preflight takes no candidate compiler or prebuilt diagnostic")
    package = None if source_only else Path(submission)
    package_digest = None if source_only else strict_tree_sha256(package)
    base = Path(out)
    base.mkdir(mode=0o700, parents=True, exist_ok=True)
    results = []
    native_result_ceilings = {}
    for row in rows:
        name = row["id"]
        if (diagnostic and name != diagnostic_model) or (selected_model is not None and name != selected_model):
            continue
        item: dict[str, Any] = {"model": name, "status": "fail", "checks": {}}
        results.append(item)
        _source_progress(progress, "model_inputs", model=name)
        try:
            identity = row.get("source_identity") or {}
            if identity != {"model_id": name, "scope": "complete_network", "role": "private_validation"}:
                raise ValueError("model has no reviewed complete-network validation identity")
            from merlin.development.testgen.sealed_capture import selection as capture_selection_api

            selection_path = _file(spec, row.get("capture_selection"), row.get("capture_selection_sha256"))
            selection = capture_selection_api.load(selection_path, expected_sha256=row["capture_selection_sha256"])
            selected_env = (selection.get("plan") or {}).get("loader_env") or {}
            env_requirements = loader_env_requirements[name]
            selected_plan = selection.get("plan") or {}
            _require_selected_input_bindings(row, selected_plan)
            selected_roles = {
                entry.get("role") for entry in selected_plan.get("selected_inputs") or [] if isinstance(entry, Mapping)
            }
            if (
                selection.get("schema") != capture_selection_api.SCHEMA_V2
                or Path(str(selected_plan.get("workload_root") or "")).name != env_requirements["workload_dir"]
                or not set(env_requirements["selected_roles"]).issubset(selected_roles)
                or row.get("deployment_dtype") != env_requirements["deployment_dtype"]
            ):
                raise ValueError("preselected source, checkpoint or deployment differs from the full model")
            if (
                not isinstance(selected_env, Mapping)
                or any(selected_env.get(key) != value for key, value in env_requirements["required"].items())
                or any(selected_env.get(key) is not None for key in env_requirements["forbidden"])
            ):
                raise ValueError("preselected loader environment permits a truncated or diagnostic model")
            capture = Path(str(row.get("capture") or ""))
            capture = capture if capture.is_absolute() else spec.parent / capture
            if capture.is_symlink() or capture.resolve() != Path(selection["run_dir"]) / "capture":
                raise ValueError("full-model capture is not the preselected sealed run output")
            from merlin.development.testgen.sealed_capture.attestation import require_verified_execution

            attestation_path = Path(str(row.get("capture_execution_attestation") or ""))
            attestation_path = attestation_path if attestation_path.is_absolute() else spec.parent / attestation_path
            if attestation_path.is_symlink() or not attestation_path.is_file():
                raise ValueError("preselected capture has no safe post-execution attestation")
            attestation_sha = file_sha256(attestation_path)
            if row.get("capture_execution_attestation_sha256") not in (None, attestation_sha):
                raise ValueError("capture execution attestation differs from an explicitly pinned digest")
            attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
            require_verified_execution(attestation)
            attested_selection = attestation.get("selection") or {}
            if (
                Path(str(attested_selection.get("path") or "")).resolve() != selection_path
                or attested_selection.get("sha256") != row["capture_selection_sha256"]
            ):
                raise ValueError("capture execution attestation names another preselection")
            attested_capture = attestation.get("capture") or {}
            capture_identity = _capture_tree_bindings(
                capture, row.get("capture_tree_sha256"), attested_capture.get("capture_tree_sha256")
            )
            item["checks"]["capture_identity"] = capture_identity
            expected_stages = tuple(required_programs[name])
            if retained_products is not None:
                retained_products.require(name, expected_stages, package_digest["sha256"], spec_digest)
            captured_kind = "session" if (capture / "session-receipt.json").exists() else "single"
            if (
                attestation.get("issuer") != "merlin.sealed_m2m_cpu.v3"
                or attested_capture.get("kind") != captured_kind
                or Path(str(attested_capture.get("capture_path") or "")).resolve() != capture
            ):
                raise ValueError("verified source-execution attestation names another complete capture")
            if captured_kind == "single":
                if (
                    expected_stages != ("model",)
                    or attested_capture.get("model_sha256") != file_sha256(capture / "model.mlir")
                    or attested_capture.get("receipt_sha256") != file_sha256(capture / "capture_receipt.json")
                ):
                    raise ValueError("attested single-network bytes differ from the full model")
            elif (
                [stage.get("name") for stage in (attested_capture.get("programs") or [])] != list(expected_stages)
                or attested_capture.get("session_contract_sha256") != file_sha256(capture / "session_contract.yaml")
                or attested_capture.get("session_receipt_sha256") != file_sha256(capture / "session-receipt.json")
            ):
                raise ValueError("attested session omits a required full-model program")
            if env_requirements["require_integer_contractions"] and (
                type(attested_capture.get("integer_contractions")) is not int
                or attested_capture["integer_contractions"] < 1
            ):
                raise ValueError("capture has no independently attested integerized accelerator computation")
            programs = _captured_programs(capture, expected_stages)
            item["checks"]["input_provenance"] = _captured_input_provenance(programs, _expected_input_provenance(row))
            facts_path = _file(spec, row.get("rtl_facts"), row.get("rtl_facts_sha256"))
            facts = selected_firrtl(facts_path, target=target, config=str(row.get("rtl_config") or ""))
            ambient_facts = os.environ.get("MERLIN_RTL_FACTS", "").strip()
            if not ambient_facts or file_sha256(Path(ambient_facts)) != facts["sha256"]:
                raise ValueError("formal source and placement readers are not bound to selected RTL facts")
            host_package = _tree(spec, row.get("host_package"), row.get("host_package_tree_sha256"))
            software_path = _authored_file(spec, row, "software_spec", frozen_authored)
            capability_path = _file(spec, row.get("capability_contract"), row.get("capability_contract_sha256"))
            host_path = _authored_file(spec, row, "host_capabilities", frozen_authored)
            catalog = _file(spec, row.get("board_catalog"), row.get("board_catalog_sha256"))
            dts = _file(spec, row.get("host_dts"), row.get("host_dts_sha256"))
            board_catalog_sha256 = file_sha256(catalog)
            board_dts_sha256 = file_sha256(dts)
            recipe_proof, exported_software, exported_host_profiles = _recipe_derivation(
                spec,
                row,
                selected_plan,
                target=target,
                software_sha256=file_sha256(software_path),
                capability_sha256=file_sha256(capability_path),
                facts_sha256=facts["sha256"],
                host_capabilities_sha256=file_sha256(host_path),
            )
            item["checks"]["recipe_derivation"] = recipe_proof
            software, host = _selected_admission_views(
                exported_software,
                exported_host_profiles,
                target=target,
                host_package=host_package,
                host_package_sha256=row["host_package_tree_sha256"],
                host_capabilities_path=host_path,
                host_capabilities_sha256=file_sha256(host_path),
            )
            capability = yaml.safe_load(capability_path.read_bytes())
            if not isinstance(capability, Mapping):
                raise ValueError("selected software, capability or host declaration is malformed")
            if software.get("target") != target or capability.get("name") != target:
                raise ValueError("selected software or capability declaration names another target")
            program_options = {
                program: dict(((build_options or {}).get(name) or {}).get(program) or {}) for program, _ in programs
            }
            selected_math = math_policy.bind(document, spec, row, target, program_options, item["checks"])
            from merlin.compiler.lower.device.selection import DeviceIRBudget

            program_ir_budgets = {}
            for program, options in program_options.items():
                selected_budget = options.pop("device_ir_budget", None)
                if selected_budget is not None:
                    if type(selected_budget) is not dict:
                        raise ValueError("device_ir_budget must be explicit original build policy data")
                    program_ir_budgets[program] = DeviceIRBudget(**selected_budget)
            program_boards = {program: opts.pop("board", row["board"]) for program, opts in program_options.items()}
            if any(not isinstance(board, str) or not board for board in program_boards.values()):
                raise ValueError("every program requires an explicit valid selected build board")
            selected_index_observations = {
                program: control_support.selected_build_observation(host_package, catalog, board, target)
                for program, board in program_boards.items()
            }
            from contextlib import nullcontext

            from merlin.target.capabilities.eligibility import original_result_format_scope

            original_contract_bytes, original_facts_bytes = capability_path.read_bytes(), facts_path.read_bytes()
            native_result_ceiling = None
            if native_result_format_selection is not None:
                binding = (sha256(original_contract_bytes).hexdigest(), sha256(original_facts_bytes).hexdigest())
                if binding not in native_result_ceilings:
                    native_result_ceilings[binding] = native_result_format_selection.issue(
                        original_contract_bytes=original_contract_bytes,
                        original_facts_bytes=original_facts_bytes,
                    )
                native_result_ceiling = native_result_ceilings[binding]
            native_scope = (
                original_result_format_scope(
                    native_result_ceiling,
                    target=target,
                    original_contract_bytes=original_contract_bytes,
                    original_facts_bytes=original_facts_bytes,
                )
                if native_result_ceiling is not None
                else nullcontext()
            )
            with (
                target_registry.observed_contract(target, dict(capability), source_path=capability_path),
                native_scope,
            ):
                sources, unresolved_demands = {}, {}
                for program, stage in programs:
                    try:
                        _source_progress(progress, "program_source", model=name, program=program)

                        def observe(event):
                            _source_progress(progress, model=name, program=program, **event)

                        sources[program] = _source_obligations(
                            stage,
                            target,
                            software,
                            capability,
                            host,
                            selected_index_observations[program],
                            facts_sha256=facts["sha256"],
                            numeric_policy_sha256=recipe_proof["recipe_semantic_sha256"],
                            **({"progress": observe} if progress is not None else {}),
                        )
                        if native_result_ceiling is not None:
                            sources[program]["accelerator_source_demand"]["native_result_format_ceiling"] = (
                                native_result_ceiling.record()
                            )
                    except source_demand.SourceDemandIncomplete as error:
                        unresolved_demands[program] = error.receipt
                if unresolved_demands:
                    item["checks"]["source_demand_refusals"] = unresolved_demands
                    raise ValueError(
                        f"original accelerator source demand is incomplete for {len(unresolved_demands)} program(s)"
                    )
            if not any(source["eligible_groups"] for source in sources.values()):
                raise ValueError("full model has no independently eligible accelerator computation")
            item["checks"]["source"] = sources
            math_policy.require_sources(selected_math, sources)
            if source_only:
                item["status"] = "source_inputs_verified"
                _source_progress(progress, "model_complete", model=name, status=item["status"])
                continue
            compiled = []
            diagnostic_receipts = {}

            def revalidate_program_inputs():
                current_capture = _capture_tree_bindings(
                    capture, row.get("capture_tree_sha256"), attested_capture.get("capture_tree_sha256")
                )
                if current_capture != capture_identity:
                    raise ValueError("private model changed during candidate build")
                if selected_firrtl(facts_path, target=target, config=facts["config"]) != facts:
                    raise ValueError("selected RTL facts changed during candidate build")
                if strict_tree_sha256(package) != package_digest:
                    raise ValueError("candidate compiler changed during model build")
                if diagnostic and any(
                    file_sha256(Path(entry["path"])) != entry["sha256"] for entry in diagnostic_receipts.values()
                ):
                    raise ValueError("prebuilt diagnostic receipt changed during static verification")
                pinned_files = (
                    (spec, spec_digest),
                    (selection_path, row["capture_selection_sha256"]),
                    (attestation_path, attestation_sha),
                    (facts_path, row["rtl_facts_sha256"]),
                    (software_path, row["software_spec_sha256"]),
                    (capability_path, row["capability_contract_sha256"]),
                    (host_path, row["host_capabilities_sha256"]),
                    (catalog, row["board_catalog_sha256"]),
                    (dts, row["host_dts_sha256"]),
                )
                if any(file_sha256(path) != digest for path, digest in pinned_files):
                    raise ValueError("reviewed model input changed during linked-build verification")
                if strict_tree_sha256(host_package)["sha256"] != row["host_package_tree_sha256"]:
                    raise ValueError("selected host package changed during linked-build verification")
                current_recipe = _recipe_derivation(
                    spec,
                    row,
                    selected_plan,
                    target=target,
                    software_sha256=file_sha256(software_path),
                    capability_sha256=file_sha256(capability_path),
                    facts_sha256=facts["sha256"],
                    host_capabilities_sha256=file_sha256(host_path),
                )[0]
                if current_recipe != recipe_proof:
                    raise ValueError("selected recipe derivation changed during linked-build verification")
                linked_policy.unchanged(instruction_selection)
                if native_result_ceiling is not None:
                    native_result_ceiling.revalidate_sources()
                math_policy.recheck(selected_math)

            with target_registry.observed_contract(target, dict(capability), source_path=capability_path):
                for program, stage in programs:
                    source = sources[program]
                    stage_tree = strict_tree_sha256(stage)
                    device = None
                    if source["eligible_groups"]:
                        routed = plan_before_build(
                            target,
                            (stage / "model.mlir").read_text(encoding="utf-8"),
                            datapath=str(row["deployment_dtype"]),
                            device_package=str(package),
                            model=f"{name}:{program}",
                            capture=stage / "model.mlir",
                            granularity=BY_GROUP,
                            linked_elf_admission=linked_elf_admission,
                            require_all_accelerator_groups=True,
                        )
                        device = routed.get("device_routing")
                        if device is None or routed.get("offload") is None:
                            raise ValueError(
                                f"{program} has no buildable accelerator route: {routed.get('device_routing_why')}"
                            )
                        if program in program_ir_budgets:
                            from dataclasses import replace

                            device = replace(device, device_ir_budget=program_ir_budgets[program])
                    source_demand.request(source, device)
                    linked_policy.require_route(device, instruction_selection)
                    if diagnostic or retained_products is not None:
                        receipt, diagnostic_receipts[program] = static_products.acquire(
                            retained_products,
                            program,
                            diagnostic_path=Path(prebuilt_receipts[program]) if diagnostic else None,
                            directory=base / name / program / "linked-policy",
                            capture=stage,
                            host_package=host_package,
                            candidate=package,
                            package_digest=package_digest,
                            arena_mb=int(row["arena_mb"]),
                            target=target,
                            device=device,
                            facts=facts,
                            options=program_options[program],
                            math_policy=selected_math["policy"],
                        )
                    else:
                        _source_progress(progress, "program_build", model=name, program=program)
                        receipt = source_optimizations.compile_model(
                            compile_saved_model,
                            compiler_package=package,
                            original_inputs=lambda: {
                                "facts": facts_path,
                                "software": software_path,
                                "capability": capability_path,
                                "host": host_path,
                                "numerical_policy": recipe_proof["recipe_semantic_sha256"],
                            },
                            capture=stage,
                            package=host_package,
                            board_catalog=catalog,
                            board=program_boards[program],
                            dts=dts,
                            output=base / name / program,
                            target=target,
                            run="none",
                            arena_mb=int(row["arena_mb"]),
                            device=device,
                            respect_captured_quantization_scope=True,
                            completed_build_reuse=document.get("completed_build_reuse"),
                            build_context={"capability": dict(capability), "facts": facts},
                            **linkage_support.build_options(source, selected_math["policy"]),
                            **program_options[program],
                        )
                    linked_policy.require_route(device, instruction_selection)
                    compiled.append(
                        program_checks.verify(
                            _verify_compiled_program,
                            receipt,
                            revalidate=revalidate_program_inputs,
                            program=program,
                            capture_path=stage,
                            source=source,
                            stage_tree=stage_tree,
                            package_digest=package_digest,
                            target=target,
                            board=program_boards[program],
                            catalog=catalog,
                            dts=dts,
                            device_selected=device is not None,
                            host_package=host_package,
                            host_package_tree_sha256=row["host_package_tree_sha256"],
                            require_compilation_recipe=not diagnostic,
                        )
                    )
                    # A later sibling may fail to build. Keep earlier typed
                    # observations visible without granting partial builds credit.
                    item["checks"]["build"] = {
                        "programs": list(compiled),
                        "linked_device_groups": sum(entry["linked_device_groups"] for entry in compiled),
                        "candidate_tree_sha256": package_digest["sha256"],
                        "required_programs": [name for name, _stage in programs],
                        "partial": True,
                        "status": "fail",
                    }
                    check_stage = program_checks.progress_stage(compiled[-1])
                    _source_progress(progress, check_stage, model=name, program=program)
            revalidate_program_inputs()
            item["checks"]["build"] = {
                "programs": compiled,
                "linked_device_groups": sum(entry["linked_device_groups"] for entry in compiled),
                "candidate_tree_sha256": package_digest["sha256"],
                "static_build_board": (
                    next(iter(program_boards.values()))
                    if len(set(program_boards.values())) == 1
                    else "per_program_selection"
                ),
                "static_build_program_boards": program_boards,
                "static_build_board_catalog_source": str(catalog),
                "static_build_board_catalog_sha256": board_catalog_sha256,
                "static_build_board_dts_source": str(dts),
                "static_build_board_dts_sha256": board_dts_sha256,
                "static_build_board_scope": BUILD_BOARD_SCOPE,
                "accelerator_rtl_facts_sha256": facts["sha256"],
                "accelerator_rtl_config": facts["config"],
                "status": program_checks.status(compiled),
            }
            if diagnostic:
                item["checks"]["prebuilt_receipts"] = diagnostic_receipts
            program_checks.require_complete(compiled)
            item["status"] = "diagnostic_static_checks_passed" if diagnostic else "pass"
        except Exception as exc:  # noqa: BLE001 -- one model's refusal must not hide the others
            item["reason"] = _public_failure_reason(exc)
            _source_progress(progress, "model_complete", model=name, status="fail")
            if source_only and isinstance(exc, SourceAdmissionError):
                item["operator_unresolved"] = json.loads(json.dumps(exc._private_unresolved, default=str))
    result = {
        "schema": RESULT_SCHEMA,
        "target": target,
        "private_spec_sha256": spec_digest,
        "candidate_tree_sha256": package_digest["sha256"] if package_digest else None,
        "required_models": list(required_models),
        "required_programs": {name: list(required_programs[name]) for name in required_models},
        "models": results,
        "passed": (
            selected_model is None
            and not diagnostic
            and not source_only
            and bool(results)
            and all(item["status"] == "pass" for item in results)
        ),
        "scope": (
            "diagnostic selected prebuilt model only; producer and toolchain unbound; not a full-model gate"
            if diagnostic
            else "complete captured networks, static source and linked-ELF evidence; full-model execution deferred"
        ),
        "full_model_numerical_equivalence": "not_run",
        "paper_accuracy": "not_claimed",
    }
    if native_result_format_selection is not None:
        native_result_format_selection.verify()
        for ceiling in native_result_ceilings.values():
            ceiling.revalidate_sources()
        result["native_result_format_selection"] = {
            "selection_sha256": native_result_format_selection.sha256,
            "scope": "original source-bound negative native output exclusions only; no positive support",
            "proofs": [ceiling.record() for ceiling in native_result_ceilings.values()],
        }
    if file_sha256(spec) != spec_digest:
        raise ValueError("operator-private full-model specification changed during verification")
    result.update(source_freeze_api.claim_binding(spec.absolute(), source_freeze, source_freeze_root, target=target))
    if diagnostic:
        result["mode"] = "prebuilt_diagnostic"
        result["diagnostic_model"] = diagnostic_model
        result["diagnostic_passed"] = len(results) == 1 and results[0]["status"] == "diagnostic_static_checks_passed"
        result["producer_binding"] = "unavailable_in_prebuilt_receipt"
    if selected_model is not None:
        result.update(
            mode="canonical_model_partition",
            selected_model=selected_model,
            partition_passed=len(results) == 1 and results[0]["status"] == "pass",
            scope="one complete declared model, canonical source and linked-ELF evidence; full roster gate incomplete",
        )
    if retained_products is not None:
        result["retained_product_revalidation"] = retained_products.record()
    if source_only:
        verified = bool(results) and all(item["status"] == "source_inputs_verified" for item in results)
        _source_result(result, verified, selected_model, SOURCE_PREFLIGHT_SCOPE)
    report = (
        base / "private_full_model_source_preflight.json"
        if source_only
        else base / "private_full_model_gate.json"
        if selected_model is None
        else base / f"private_full_model_partition-{sha256(selected_model.encode()).hexdigest()}.json"
    )
    with os.fdopen(os.open(report, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    return result
