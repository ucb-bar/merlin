"""Deterministic, installed requirement derivation from explicit iteration captures.

No agent, candidate compiler or headline evaluation is involved. Authored inputs
are declarations; generated inventories and synthesis plans are not certificates.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

from merlin.targetgen import application_inventory, conformance, corpus_synth, target_registry
from merlin.targetgen.rtl.facts import observed_facts
from merlin.targetgen.target_experiment import load_target_experiment
from merlin_experiments.spec import load_spec

from .certification_floor import require_direct_tier, selected_floor
from .declarations import from_definition
from .evidence import _materialize_evidence, export_evidence, select_evidence
from .performance_scope import derive_performance_scope
from .profiles import selected_software_spec_path, synthesis_input_identity
from .program_admission import entry_refusal_is_final
from .software_screen import diagnostic_entry, intersect_requirement, screen_entry
from .typed_scope import typed_required_instances


def _json(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def _recipe_tier_declarations(path: Path) -> tuple[dict, list[str]]:
    recipe_doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    planned_tiers = (
        (recipe_doc.get("datapath") or {}).get("required_oracle_tiers") if isinstance(recipe_doc, dict) else None
    )
    if planned_tiers is not None and (
        not isinstance(planned_tiers, list)
        or any(
            not isinstance(tier, str) or not tier.startswith("L") or not tier[1:].isdigit() for tier in planned_tiers
        )
    ):
        raise ValueError("selected recipe required_oracle_tiers must be a list of fidelity tiers")
    return recipe_doc, list(planned_tiers or [])


def _materialized_iteration_capsules(
    full: dict,
    digest: str,
    *,
    loader_snapshots: dict[str, tuple[Path, str]] | None = None,
) -> tuple[list[dict], dict[str, bytes]]:
    """Select full iteration programs, not a family representative or a held-out model.

    Save all producer-receipt members before exposing entries. A source loader
    becomes gradeable only when a verified preselected capture binds its exact
    snapshot bytes; an older receipt alone remains diagnostic.
    """
    from merlin.targetgen.capsule_source import materialized_model_artifacts

    entries, outputs = [], {}
    loader_snapshots = loader_snapshots or {}
    for label, application in sorted(full["applications"].items()):
        if not label or Path(label).name != label or label in {".", ".."}:
            raise ValueError("application identity must be a single safe path component")
        source = Path(application["capture_source_path"])
        selection = {
            "path": str(source),
            "capture_sha256": application["capture_sha256"],
            "receipt_sha256": application["capture_receipt"]["receipt_sha256"],
            "workload_id": label,
            "workload_role": "iteration",
            "coverage_scope": "full_capture",
            "full_inventory_sha256": digest,
            "operation_count": application["n_operations"],
        }
        artifact = materialized_model_artifacts(selection)
        receipt_raw = (source.parent / "capture_receipt.json").read_bytes()
        if hashlib.sha256(receipt_raw).hexdigest() != selection["receipt_sha256"]:
            raise ValueError(f"capture receipt changed while copying {label}")
        receipt = json.loads(receipt_raw)
        members = set(receipt["artifacts"]) | {"capture_receipt.json"}
        if artifact.meta.get("framework_catalog"):
            members.add("pytorch-opset.json")
        for member in sorted(members):
            path = source.parent / member
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"materialized member missing or symlinked: {path}")
            outputs[f"materialized/{label}/{member}"] = path.read_bytes()
        # Independent recheck closes a producer mutation during the copy.
        copied = outputs[f"materialized/{label}/model.mlir"]
        if hashlib.sha256(copied).hexdigest() != selection["capture_sha256"]:
            raise ValueError(f"capture bytes changed while copying {label}")
        for member, identity in receipt["artifacts"].items():
            raw = outputs[f"materialized/{label}/{member}"]
            if len(raw) != identity["bytes"] or hashlib.sha256(raw).hexdigest() != identity["sha256"]:
                raise ValueError(f"receipt-bound bytes changed while copying {label}/{member}")
        if (
            artifact.meta.get("framework_catalog")
            and hashlib.sha256(outputs[f"materialized/{label}/pytorch-opset.json"]).hexdigest()
            != artifact.meta["framework_catalog"]["sha256"]
        ):
            raise ValueError(f"framework catalog changed while copying {label}")
        loader_binding = {}
        if label in loader_snapshots:
            loader_path, loader_sha256 = loader_snapshots[label]
            if loader_path.is_symlink() or not loader_path.is_file():
                raise ValueError(f"selected workload loader is absent or symlinked: {label}")
            loader_bytes = loader_path.read_bytes()
            if hashlib.sha256(loader_bytes).hexdigest() != loader_sha256:
                raise ValueError(f"selected workload loader changed while copying {label}")
            outputs[f"materialized/{label}/loader.py"] = loader_bytes
            loader_binding = {"loader_path": f"materialized/{label}/loader.py", "loader_sha256": loader_sha256}
        entries.append(
            {
                "name": f"SY_source_{label}",
                "cat": "model",
                "kind": "model",
                "op": "model",
                "model": label,
                "label": "public",
                "operand_dtype": artifact.dtype,
                "source_role": "materialized_iteration_capture",
                "source_reference": "full saved iteration capture; target compile and execution unverified",
                "materialized_capture": {
                    **selection,
                    "path": f"materialized/{label}/model.mlir",
                    **loader_binding,
                },
                "generalization": {"generalization_axis": "composition"},
            }
        )
    return entries, outputs


def capture_selections(selections: list[str]) -> dict[str, Path]:
    result = {}
    for item in selections:
        label, separator, location = item.partition("=")
        if not separator or not label or not location or label in result:
            raise ValueError(f"invalid/duplicate capture selection {item!r}; use LABEL=PATH")
        path = Path(location).expanduser().absolute()
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
            raise ValueError(f"capture selection traverses a symlink: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        result[label] = path
    return result


def _selected_file_specs(selections: list[str], role: str) -> dict[str, tuple[Path, str]]:
    result = {}
    for item in selections:
        label, separator, location = item.partition("=")
        name, digest_separator, digest = location.rpartition("@")
        if (
            not separator
            or not digest_separator
            or not label
            or not name
            or label in result
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"invalid/duplicate {role} {item!r}; use LABEL=PATH@SHA256")
        path = Path(name).expanduser().absolute()
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents) or not path.is_file():
            raise ValueError(f"{role} is absent or indirect: {path}")
        result[label] = (path, digest)
    return result


def capture_selection_specs(selections: list[str]) -> dict[str, tuple[Path, str]]:
    """Parse independent pre-execution capture selections and byte digests."""
    return _selected_file_specs(selections, "capture preselection")


def quantization_policy_specs(selections: list[str]) -> dict[str, tuple[Path, str]]:
    """Parse operator-selected external quantization policies and byte digests."""
    return _selected_file_specs(selections, "quantization policy")


def _validate_capture_recipes(
    captures: dict[str, Path],
    selected_recipe_hashes: set[str],
    *,
    software_spec_sha256: str | None = None,
    policy_selections: dict[str, tuple[Path, str]] | None = None,
) -> dict[str, str]:
    """A realized quantized graph must use a recipe this provider actually derived."""
    selected_policies = {}
    policy_selections = policy_selections or {}
    for label, path in sorted(captures.items()):
        meta_path = path.with_name("meta.json")
        if meta_path.is_symlink():
            raise ValueError(f"{label}: capture metadata may not be a symlink")
        meta = json.loads(meta_path.read_bytes())
        if not isinstance(meta, dict):
            raise ValueError(f"{label}: capture metadata must be a mapping")
        stats = meta.get("quantization_stats") or {}
        if not isinstance(stats, dict):
            raise ValueError(f"{label}: quantization statistics must be a mapping")
        actual = stats.get("recipe_sha256")
        if actual is not None and (
            not isinstance(actual, str)
            or len(actual) != 64
            or any(char not in "0123456789abcdef" for char in actual)
            or actual not in selected_recipe_hashes
        ):
            raise ValueError(
                f"{label}: capture used a different quantization recipe from the selected provider; "
                "regenerate the capture from its selected Phase 0 recipe"
            )
        manifest_path = path.with_name("quantization-manifest.json")
        if (
            manifest_path.exists()
            or meta.get("quantization_manifest") is not None
            or (b"prov.quantization_manifest_sha256" in path.read_bytes())
        ):
            verified = application_inventory.verify_capture_receipt(path)
            if verified["status"] != "verified_materialized":
                raise ValueError(f"{label}: external quantization manifest is not byte-bound to the capture")
            manifest = json.loads(manifest_path.read_bytes())
            if software_spec_sha256 is None or manifest.get("contract_sha256") != software_spec_sha256:
                raise ValueError(f"{label}: external quantization contract differs from selected software spec")
            selected_policy = policy_selections.get(label)
            if selected_policy is None:
                raise ValueError(f"{label}: external quantization requires an independent policy selection")
            policy_path, policy_sha256 = selected_policy
            if (
                policy_path.is_symlink()
                or any(parent.is_symlink() for parent in policy_path.parents)
                or not policy_path.is_file()
                or hashlib.sha256(policy_path.read_bytes()).hexdigest() != policy_sha256
                or manifest.get("policy_sha256") != policy_sha256
            ):
                raise ValueError(f"{label}: selected quantization policy differs from capture manifest")
            selected_policies[label] = policy_sha256
    if set(policy_selections) != set(selected_policies):
        raise ValueError("quantization policy selections must name exactly the external captures")
    return selected_policies


def performance_scale_selection(
    workload_spec: dict | None, captures: dict[str, Path], performance_captures: dict[str, Path] | None
) -> dict[str, Path]:
    """The selected Phase-2-only captures, checked against the declared performance roster.

    ``workload_spec.performance_applications`` names independent workloads whose forms enter only
    ``scope.performance.forms``: the Phase 2 cohort's sources, never Phase 1 forms or source capsules.
    The roster is declared, like the iteration roster, so a selection can be neither silently partial
    nor silently extra; the labels and capture paths must be disjoint from the iteration selection.
    Held-out models are refused by the form-scope derivation like any other source.
    """
    declared = (workload_spec or {}).get("performance_applications")
    selected = dict(performance_captures or {})
    if declared is None:
        if selected:
            raise ValueError("performance-scale captures need a declared workload_spec.performance_applications")
        return {}
    if not isinstance(declared, (list, tuple)) or not declared or len(declared) != len(set(declared)):
        raise ValueError("workload_spec.performance_applications must be a nonempty list of unique labels")
    declared = [str(label) for label in declared]
    if overlap := sorted(set(declared) & set((workload_spec or {}).get("applications") or ())):
        raise ValueError(f"performance applications {overlap} are also iteration applications")
    if set(selected) != set(declared):
        raise ValueError(
            f"performance roster mismatch: missing={sorted(set(declared) - set(selected))}, "
            f"extra={sorted(set(selected) - set(declared))}"
        )
    iteration_paths = {str(Path(path).resolve()) for path in captures.values()}
    resolved = [str(Path(path).resolve()) for path in selected.values()]
    if len(set(resolved)) != len(resolved) or set(resolved) & iteration_paths:
        raise ValueError("performance-scale captures must be distinct files, disjoint from the iteration captures")
    return {label: Path(selected[label]) for label in sorted(selected)}


def _screen_defaults(entry: dict, semantics: dict) -> dict:
    """The numeric defaults a candidate is screened with before it is written.

    The accelerator's numerical semantics describe accelerator work. A host-only probe (a program that
    must NOT be accelerated) is written through the float host path in its own operand format, so it is
    screened in that format and with no accelerator accumulator -- the same binding generation's
    writer selects (``generation._screen_selected_entry``); otherwise a float host contraction is
    refused before it exists for not accumulating in the accelerator's integer format."""
    probe = entry.get("generalization") or {}
    if probe.get("must_accelerate") is False and probe.get("eligible") is False:
        return {"operand_dtype": entry.get("operand_dtype") or semantics.get("operand_dtype")}
    return semantics


def incomplete_inventory_blocker(full: dict, host_capabilities: dict | None) -> str | None:
    """The reason a detailed inventory is incomplete, by application and operation, or ``None``.

    An unclassified operation is usually one whose accelerator admission is undetermined and which only
    a reviewed host declaration can place. When the host lane's capability declaration was not selected
    (its package is absent, for instance), no such operation can be placed, and the useful statement is
    that one -- not a bare count."""
    if full.get("status") == "inventoried":
        return None
    rows = []
    for label, app in sorted((full.get("applications") or {}).items()):
        if not isinstance(app, dict) or app.get("status") == "inventoried":
            continue
        unclassified = Counter(
            str(row.get("operation"))
            for row in app.get("signatures") or ()
            for _ in range(int(row.get("count") or 1))
            if row.get("disposition") == "unclassified"
        )
        rows.append(f"{label}: {', '.join(f'{op} x{n}' for op, n in sorted(unclassified.items())) or 'unreadable'}")
    unselected = [
        f"{name}: {profile.get('reason')}"
        for name, profile in sorted((host_capabilities or {}).items())
        if isinstance(profile, dict) and profile.get("capability_spec") is None
    ]
    why = (
        f"; the host lane's capability declaration was not selected ({'; '.join(unselected)}), so no "
        "reviewed host declaration could place them"
        if unselected
        else ""
    )
    return f"declared iteration captures are not fully inventoried: unclassified {'; '.join(rows)}{why}"


def unresolved_epilogue_blocker(requirement: dict) -> str | None:
    """A blocker naming the epilogue stages the requirement could not decide, or ``None``.

    An undetermined stage is neither required nor refused, so the derived corpus asks nothing about
    it. That is acceptable for a diagnostic census and never for a verified derivation: a backend that
    cannot fuse the stage would then fail nothing. The usual cause is a derivation run without the
    target's support provider, whose readout declaration is the stage-granular evidence.
    """
    rows = ((requirement.get("epilogue") or {}).get("unresolved")) or []
    stages = sorted({str(row.get("stage")) for row in rows if isinstance(row, dict) and row.get("stage")})
    if not stages:
        return None
    return (
        f"epilogue stages {stages} are undetermined: no readout declaration and no derived instruction "
        "taxonomy were readable, so the requirement can neither demand nor refuse them; select the "
        "target's support provider and derive again"
    )


def grouping_oracle(target: str, prohibited_roles):
    """The grouping's target oracle under the experiment's prohibited instruction roles.

    ``None`` (the grouping's own default oracle) when the experiment prohibits nothing. Otherwise a
    standalone form whose every evidence path needs a prohibited role -- a residual add licensed only
    by a hardware loop descriptor, say -- is refused at grouping, so no form the policy forbids reaches
    the requirement (``merlin.targetgen.capability_roles``).
    """
    roles = tuple(prohibited_roles or ())
    if not roles:
        return None
    from merlin.xdsl_dialects.lowering import compute_groups as CG

    return CG.TargetOracle(target, prohibited_roles=roles)


def derive(
    definition: str | Path,
    captures: dict[str, Path],
    *,
    rtl_facts: str | Path,
    output_root: str | Path,
    native_qualifications: dict[str, Path] | None = None,
    capture_preselections: dict[str, tuple[Path, str]] | None = None,
    quantization_policies: dict[str, tuple[Path, str]] | None = None,
    performance_captures: dict[str, Path] | None = None,
    performance_preselections: dict[str, tuple[Path, str]] | None = None,
    heldout_layers: str | Path | None = None,
) -> dict:
    """Write a byte-bound requirement, complete census and diagnostic candidate plan.

    All declared iteration applications must be supplied. Old synthesis/private
    profiles and historical corpus members are deliberately not inputs. Repeating
    the same selection produces identical bytes; changed inputs need a new root.

    ``performance_captures`` are the descriptor's declared ``performance_applications``:
    independent captures that feed only the Phase 2 form scope (see
    :func:`performance_scale_selection`).
    """
    declaration = from_definition(definition)
    te = load_target_experiment(declaration.descriptor)
    performance_captures = performance_scale_selection(te.workload_spec, captures, performance_captures)
    layer_guard = None
    if heldout_layers is not None:
        # Operator-private: the held-out networks' exact layer shapes, read before any derivation work.
        from merlin.common.paths import repo_root

        from . import heldout_layers as HL

        try:
            layer_guard = HL.load(heldout_layers, repository=repo_root())
        except (OSError, json.JSONDecodeError) as exc:
            raise HL.HeldoutLayerError(f"held-out layer file is unreadable: {exc}") from exc
    performance_preselections = performance_preselections or {}
    if performance_preselections and set(performance_preselections) != set(performance_captures):
        raise ValueError("performance capture preselection must cover the entire declared performance roster")
    declared = (te.workload_spec or {}).get("applications")
    if not isinstance(declared, (list, tuple)) or not declared or len(declared) != len(set(declared)):
        raise ValueError("deterministic derivation needs an explicit nonempty, unique application roster")
    if set(captures) != set(declared):
        raise ValueError(
            f"iteration roster mismatch: missing={sorted(set(declared) - set(captures))}, "
            f"extra={sorted(set(captures) - set(declared))}"
        )
    if len({str(path.resolve()) for path in captures.values()}) != len(captures):
        raise ValueError("distinct application labels cannot select the same capture path")
    capture_preselections = capture_preselections or {}
    if capture_preselections and set(capture_preselections) != set(captures):
        raise ValueError("capture preselection must cover the entire declared iteration roster")
    selected_capture_evidence, capture_attestations, loader_snapshots = {}, {}, {}
    capture_python = None
    if capture_preselections:
        from .capture_execution_attestation import attest_sealed_m2m
        from .capture_selection import load, verify

        for label, (selection_path, selected_sha256) in sorted(capture_preselections.items()):
            selected_capture_evidence[label] = verify(
                selection_path, expected_sha256=selected_sha256, model_path=captures[label]
            )
            capture_attestations[label] = attest_sealed_m2m(
                selected_capture_evidence[label], selection_path=selection_path, model_path=captures[label]
            )
            selected = load(selection_path, expected_sha256=selected_sha256)
            loader_snapshots[label] = (
                Path(selected["run_dir"]) / "snapshots/source/workload/loader.py",
                selected["plan"]["loader_sha256"],
            )
            interpreter = Path(selected["plan"]["venv"]) / "bin" / "python"
            if capture_python is not None and capture_python != interpreter:
                raise ValueError("iteration captures select different PyTorch interpreters")
            capture_python = interpreter
    performance_capture_evidence, performance_attestations = {}, {}
    if performance_preselections:
        # The same pre-execution selection and sealed-replay attestation as an iteration capture; a
        # performance-scale capture feeds only the Phase 2 form scope, never the Phase 1 sources.
        from .capture_execution_attestation import attest_sealed_m2m
        from .capture_selection import verify

        for label, (selection_path, selected_sha256) in sorted(performance_preselections.items()):
            performance_capture_evidence[label] = verify(
                selection_path, expected_sha256=selected_sha256, model_path=performance_captures[label]
            )
            performance_attestations[label] = attest_sealed_m2m(
                performance_capture_evidence[label],
                selection_path=selection_path,
                model_path=performance_captures[label],
            )
    spec = load_spec(definition)
    config = spec.document["phases"]["0"]["config"]
    software = selected_software_spec_path(
        declaration.recipe, spec.resolve(config["software_spec"]) if config.get("software_spec") else None
    )
    capability_contract_path = (
        spec.resolve(config["capability_contract"]) if config.get("capability_contract") else None
    )
    hardware = spec.resolve(config["hardware_spec"]) if config.get("hardware_spec") else None
    if software is None:
        raise ValueError("deterministic derivation requires an explicit software spec in the recipe")
    recipe_doc = None
    planned_tiers = None
    certification_floor = None
    if (te.workload_spec or {}).get("certification_floor") is not None:
        recipe_doc, planned_tiers = _recipe_tier_declarations(declaration.recipe)
        certification_floor = selected_floor(te.workload_spec, planned_tiers)
    selected = select_evidence(
        te.target,
        descriptor=declaration.descriptor,
        capture_python=capture_python,
        capability_contract_path=capability_contract_path,
        software_spec=software,
        hardware_spec=hardware,
        facts_path=rtl_facts,
        prohibited_roles=spec.prohibited_instruction_roles,
    )
    options = {
        "capability_contract": selected.contract,
        "software_spec": selected.software_spec,
        "host_capabilities": selected.host_capabilities,
        "include_graph": True,
        "application_metadata": {
            label: {"workload_id": label, "workload_role": "iteration", "coverage_scope": "full_capture"}
            for label in captures
        },
    }
    # Legacy readers use target names; these scopes prevent a second live
    # contract/facts selection or silent extraction from an ambient cache.
    with (
        target_registry.observed_contract(te.target, selected.contract),
        observed_facts(te.target, selected.refreshed_facts, Path(rtl_facts)),
    ):
        full = application_inventory.application_demand_inventory(captures, te.target, detailed=True, **options)
        requirement = conformance.derive_spec(
            te.target,
            captures,
            applications=captures,
            oracle_tiers=[],
            corpus_roots=[],
            cert_budget_s=(te.workload_spec or {}).get("cert_budget_s"),
            certification_floor=certification_floor,
            application_inventory_options=options,
        )
    inventory_blocker = incomplete_inventory_blocker(full, selected.host_capabilities)
    if inventory_blocker is not None and config.get("evidence_mode") != "diagnostic":
        raise ValueError(inventory_blocker)
    epilogue_blocker = unresolved_epilogue_blocker(requirement)
    if epilogue_blocker is not None and config.get("evidence_mode") != "diagnostic":
        raise ValueError(epilogue_blocker)
    digest = hashlib.sha256(json.dumps(full, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != requirement["application_demands"]["full_inventory_sha256"]:
        raise ValueError("capture bytes changed while deriving requirements")
    # Retain the exact operation IDs and SSA types behind the family-only raw
    # scope census. This is source evidence, not an accelerator-eligible Phase 2
    # requirement; placement and compiler correspondence remain unresolved.
    requirement["scope"]["typed_required_instances"] = typed_required_instances(requirement["scope"], full)
    # Each declared capture's per-operation admission, judged now with the selected declarations, so
    # generation can hold the whole-program capsule built from that capture to the same placements.
    from .program_admission import SCHEMA as WHOLE_PROGRAM_SCHEMA
    from .program_admission import derive_inventory

    whole_program = derive_inventory(full, target=te.target, evidence=selected)
    requirement["whole_program_admission"] = {
        "schema": WHOLE_PROGRAM_SCHEMA,
        "sidecar": "whole-program-admission.json",
        "sha256": whole_program["sha256"],
    }
    from merlin.targetgen.quantization_spec import build_quantization_contract, capture_recipe_candidates

    quantization = build_quantization_contract(
        selected.software_spec,
        {
            "contract": selected.contract,
            "quantization_candidates": selected.quantization_snapshot.get("quantization_candidates", []),
            "readout_facets": selected.readout_facets,
            "readout_numerics": selected.readout_numerics,
        },
    )
    selected_recipes = capture_recipe_candidates(selected.software_spec, quantization)
    if performance_captures:
        # A performance-scale capture is held to the same provider-derived recipe as an iteration
        # capture; external quantization policies are an iteration-roster selection only.
        _validate_capture_recipes(
            performance_captures,
            {row["recipe"]["recipe_sha256"] for row in selected_recipes},
            software_spec_sha256=hashlib.sha256(software.read_bytes()).hexdigest(),
        )
    selected_policies = _validate_capture_recipes(
        captures,
        {row["recipe"]["recipe_sha256"] for row in selected_recipes},
        software_spec_sha256=hashlib.sha256(software.read_bytes()).hexdigest(),
        policy_selections=quantization_policies,
    )
    policy_outputs = {}
    if selected_policies:
        policy_members = {}
        for label, expected in sorted(selected_policies.items()):
            source = quantization_policies[label][0]
            raw = source.read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError(f"{label}: selected quantization policy changed during derivation")
            member = f"selected-quantization-policies/{label}.input"
            policy_outputs[member] = raw
            policy_members[label] = {"artifact": member, "sha256": expected}
        requirement["quantization_policy_selections"] = {
            "schema": "merlin.phase0.quantization_policy_selections.v1",
            "status": "byte_selected_not_numerically_reviewed",
            "applications": policy_members,
        }
    requirement["application_demands"]["sidecar"] = "application-demands.json"
    if selected_capture_evidence:
        requirement["capture_execution_preselections"] = {
            "schema": "merlin.phase0.capture_preselections.v1",
            "status": "replay_verified_nonadmissible",
            "applications": selected_capture_evidence,
            "phase0_admission": "not_granted",
        }
        # The replay records above stay nonadmissible; the separately issued
        # attestations are what the coverage gate re-verifies from disk.
        requirement["capture_execution_attestations"] = capture_attestations
    if performance_capture_evidence:
        requirement["performance_capture_preselections"] = {
            "schema": "merlin.phase0.capture_preselections.v1",
            "status": "replay_verified_nonadmissible",
            "applications": performance_capture_evidence,
            "attestations": performance_attestations,
            "phase0_admission": "not_granted",
        }
    requirement = intersect_requirement(requirement, selected.software_spec, selected.contract)
    # The recipe's tier ladder is an authored PLAN, not evidence that an oracle
    # was constructed. Keep it separate from ``oracle_tiers`` (which remains
    # observed-only) so synthesis can cap unaffordable members to a declared
    # functional screen without claiming that screen executed at derivation.
    if recipe_doc is None:
        # Preserve the historical no-floor observation point: reading the recipe
        # earlier changes which selected-input error wins during legacy derivation.
        recipe_doc, planned_tiers = _recipe_tier_declarations(declaration.recipe)
    requirement["oracle_tiers_declared"] = planned_tiers
    if certification_floor is not None:
        if (requirement.get("certification_floor") or {}).get("tier") != certification_floor:
            raise ValueError("core requirement omitted the selected certification floor")
        requirement["certification_floor"]["source"] = "descriptor.workload_spec.certification_floor"
    requirement["scope"]["performance"] = derive_performance_scope(requirement["scope"], selected.software_spec)
    # The two capture-derived stages share ONE grouping and ONE binding (the corpus binding under the
    # selected recipe, contract and facts), read only the declared ITERATION captures, and refuse any
    # held-out model. Their outputs are byte-bound here with every other derived member.
    from merlin.targetgen.group_capsule_entries import group_binding

    from .claim_boundary import held_out_models
    from .form_perf import derive_form_scope
    from .model_forms import derive_model_forms

    held_out = held_out_models(te)
    stage_binding = group_binding(
        te,
        (recipe_doc.get("datapath") or {}) if isinstance(recipe_doc, dict) else {},
        contract=selected.contract,
        facts=selected.refreshed_facts,
        taxonomy=selected.isa_taxonomy,
    )
    with (
        target_registry.observed_contract(te.target, selected.contract),
        observed_facts(te.target, selected.refreshed_facts, Path(rtl_facts)),
    ):
        oracle = grouping_oracle(te.target, spec.prohibited_instruction_roles)
        requirement["scope"]["performance"]["forms"] = derive_form_scope(
            te.target,
            captures,
            stage_binding,
            iteration_roster=list(declared),
            held_out=held_out,
            facts=selected.refreshed_facts,
            oracle=oracle,
            performance_scale=performance_captures,
            performance_scale_roster=sorted(performance_captures),
        )
        if layer_guard is not None:
            from . import heldout_layers as HL

            forms = requirement["scope"]["performance"]["forms"]
            HL.refuse(HL.form_scope_collisions(forms, layer_guard), what="performance form")
            requirement["heldout_layer_guard"] = {
                **layer_guard.summary(),
                "status": "no_member_reproduces_a_heldout_layer",
                "members_checked": sum(len(row.get("members") or []) for row in forms.get("classes") or []),
            }
        model_forms = derive_model_forms(
            te.target,
            captures,
            stage_binding,
            iteration_roster=list(declared),
            held_out=held_out,
            facts=selected.refreshed_facts,
            ceiling=(requirement.get("cert_affordability") or {}).get("max_elements"),
            numerical_semantics=(selected.software_spec or {}).get("numerical_semantics"),
            oracle=oracle,
        )
    requirement["derivation"]["phase0_execution"] = {
        "agentic": False,
        "policy": "deterministic from selected inputs",
        "definition_sha256": hashlib.sha256(spec.path.read_bytes()).hexdigest(),
        "oracle_tiers": "not constructed during derivation; establish in execution qualification",
        "historical_corpus": "not selected",
        "headline_workloads": "held out",
        **selected.derivation_identity,
    }
    root = Path(output_root).absolute()
    outputs = {
        "requirements.yaml": yaml.safe_dump(requirement, sort_keys=False).encode(),
        "application-demands.json": _json(full),
        "whole-program-admission.json": _json(whole_program),
        **policy_outputs,
    }
    if selected_capture_evidence:
        outputs["capture-preselections.json"] = _json(requirement["capture_execution_preselections"])
    # Save the census even when an exact writer cannot express every signature.
    # Such a plan is diagnostic and must not become a selectable verified corpus.
    try:
        plan = corpus_synth.synthesize(
            requirement,
            workload_spec=te.workload_spec,
            application_inventory=full,
            capability_contract=selected.contract,
        )
    except corpus_synth.SynthesisError as exc:
        plan = {"status": "blocked", "reason": str(exc), "capsules": [], "provenance": {}}
    source_entries, source_outputs = _materialized_iteration_capsules(full, digest, loader_snapshots=loader_snapshots)
    outputs.update(source_outputs)
    gradeable_sources = [entry for entry in source_entries if entry["materialized_capture"].get("loader_sha256")]
    if plan.get("status") != "blocked":
        plan["capsules"] = [*plan.get("capsules", []), *model_forms["entries"], *gradeable_sources]
        require_direct_tier(
            plan["capsules"], floor=certification_floor, declared_tiers=requirement["oracle_tiers_declared"]
        )
    plan.setdefault("provenance", {})["model_forms"] = model_forms["provenance"]
    screens = []
    for index, entry in enumerate(plan.get("capsules") or []):
        decision = screen_entry(
            selected.software_spec,
            entry,
            defaults=_screen_defaults(entry, selected.software_spec["numerical_semantics"]),
            host_capabilities=selected.host_capabilities,
        )
        screens.append({"capsule": entry.get("name"), **decision})
        if entry_refusal_is_final(entry, decision):
            plan["capsules"][index] = diagnostic_entry(entry, decision)
    plan.setdefault("provenance", {})["software_intersection"] = {
        **requirement["software_intersection"],
        "candidate_screens": screens,
    }
    # A preselected, replay-verified workload snapshot supplies exact loader bytes.
    # Older receipt-only captures remain diagnostic and cannot become graded models.
    plan.setdefault("provenance", {})["materialized_iteration_captures"] = {
        "status": "byte_verified",
        "applications": sorted(captures),
        "full_inventory_sha256": digest,
        "scope": "full capture, not headline validation",
        "qualification": "host reference and source coverage only; target support and execution unverified",
        "gradeable_model_capsules": len(gradeable_sources),
        "diagnostic_model_names": [entry["name"] for entry in source_entries if entry not in gradeable_sources],
    }
    outputs["synthesis-plan.json"] = _json(plan)
    _materialize_evidence(root, outputs)
    selected = select_evidence(
        te.target,
        descriptor=declaration.descriptor,
        capture_python=capture_python,
        capability_contract_path=capability_contract_path,
        software_spec=software,
        hardware_spec=hardware,
        facts_path=rtl_facts,
        conformance_spec=root / "requirements.yaml",
        native_qualifications=native_qualifications,
        prohibited_roles=spec.prohibited_instruction_roles,
    )
    manifest = export_evidence(selected, root / "evidence")
    identity = synthesis_input_identity(
        conformance_spec=root / "requirements.yaml",
        recipe=declaration.recipe,
        descriptor=declaration.descriptor,
        software_spec=software,
    )
    if plan.get("status") != "blocked":
        profile = {
            "provenance": {
                **plan.get("provenance", {}),
                "selected_inputs": identity,
                "qualification": "diagnostic candidate generation, not a reviewed corpus",
            },
            "capsules": plan.get("capsules", []),
        }
        _materialize_evidence(root, {"synthesis.yaml": yaml.safe_dump(profile, sort_keys=False).encode()})
    accounting = json.loads((root / "evidence/coverage/operation-accounting.json").read_bytes())
    operation_plan = plan.get("provenance", {}).get("application_operation_plan") or {}
    capsules = plan.get("capsules") or []
    report = {
        "schema": "merlin.phase0_derivation.v1",
        "target": te.target,
        "status": "diagnostic",
        "agentic": False,
        "selected_inputs": identity,
        "raw_facts_sha256": selected.raw_facts_sha256,
        "evidence_artifacts": len(manifest["artifacts"]),
        "applications": sorted(captures),
        "performance_scale_applications": sorted(performance_captures),
        "native_baseline_observations": {
            label: {
                key: value
                for key, value in observation.items()
                if key
                in {"status", "executor", "capture_sha256", "receipt_sha256", "max_absolute_error", "target_executed"}
            }
            for label, observation in selected.native_baseline_observations.items()
        },
        "mlir_operations": full["n_operations"],
        "source_trace_statuses": {
            label: app.get("pytorch_provenance", {}).get("source_trace_status", "unknown")
            for label, app in accounting.get("applications", {}).items()
        },
        "admitted_cells": len(requirement.get("cells") or []),
        "declared_compute_units": len(selected.contract.get("compute_units") or []),
        "candidate_capsules": len(capsules),
        "candidate_capsules_by_source_role": dict(
            sorted(Counter(str(entry.get("source_role") or "unspecified") for entry in capsules).items())
        ),
        "candidate_screen_statuses": dict(
            sorted(Counter(str(screen.get("status") or "unknown") for screen in screens).items())
        ),
        "synthesis_profile": "synthesis.yaml" if plan.get("status") != "blocked" else None,
        "application_operation_plan": {
            key: value for key, value in operation_plan.items() if key not in {"obligations", "missing_mapping"}
        },
        "blockers": [
            *selected.qualification_blockers,
            *([epilogue_blocker] if epilogue_blocker is not None else []),
            *([inventory_blocker] if inventory_blocker is not None else []),
            *([plan["reason"]] if plan.get("status") == "blocked" else []),
        ],
        "qualification": "derivation only; no compiler execution, oracle or release approval",
    }
    _materialize_evidence(root, {"derivation.json": _json(report)})
    return report
