"""Selected direct-certification minimum for newly derived and staged corpora.

This is an input check, not a tier promotion. A cost-capped member remains
cost-capped until a new producer honestly derives a directly certified member.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from math import prod
from pathlib import Path

import yaml

from merlin.development.targetgen.applications import inventory as application_inventory
from merlin.development.targetgen.conformance import (
    requirements as conformance,
)
from merlin.development.targetgen.conformance import (
    synthesis_entries as corpus_synth,
)
from merlin.development.testgen.evidence.evidence import export_evidence
from merlin.development.testgen.evidence.observations import _materialize_evidence
from merlin.development.testgen.evidence.program_admission import entry_refusal_is_final
from merlin.development.testgen.evidence.selection import select_evidence
from merlin.development.testgen.performance.scope import derive_performance_scope, typed_required_instances
from merlin.development.testgen.profiles.loading import from_definition
from merlin.development.testgen.profiles.screens import diagnostic_entry, intersect_requirement, screen_entry
from merlin.development.testgen.profiles.synthesis_selection import (
    selected_software_spec_path,
    synthesis_input_identity,
)
from merlin.experiments.definition import engine_definitions, load_spec
from merlin.target.registry import identity as target_registry
from merlin.target.registry.descriptor_loading import load_target_experiment
from merlin.target.rtl.facts import observed_facts

# ---- certification_floor -----------------------------------------------------------------------


def _tier_index(value: object) -> int | None:
    if not isinstance(value, str) or not value.startswith("L") or not value[1:].isdigit():
        return None
    return int(value[1:])


def selected_floor(workload_spec: Mapping | None, declared_tiers: Sequence[str]) -> str | None:
    """Return the explicitly selected floor, refusing absent or invalid tier vocabulary."""
    raw = (workload_spec or {}).get("certification_floor")
    if raw is None:
        return None
    if _tier_index(raw) is None or raw not in declared_tiers:
        raise ValueError("certification_floor must name a tier in the selected recipe")
    return raw


def require_direct_tier(
    capsules: Sequence[Mapping], *, floor: str | None, declared_tiers: Sequence[str] | None = None
) -> dict:
    """Check every member against a selected floor without changing membership or tiers.

    ``declared_tiers`` supplies the recipe default before materialization.
    Materialized capsules must state their own required tiers instead.
    """
    if floor is None:
        return {"floor": None, "checked": len(capsules), "below_floor": []}
    floor_index = _tier_index(floor)
    if floor_index is None:
        raise ValueError("certification_floor is not a fidelity tier")
    refused: list[str] = []
    for capsule in capsules:
        name = capsule.get("name") if isinstance(capsule, Mapping) else None
        if not isinstance(name, str) or not name:
            raise ValueError("certification-floor preflight requires named capsules")
        tiers = capsule.get("required_oracle_tiers")
        if tiers is None:
            tiers = declared_tiers
        if not isinstance(tiers, (list, tuple)) or floor not in tiers:
            refused.append(name)
            continue
        cap = capsule.get("max_oracle_tier")
        if cap is not None and (_tier_index(cap) is None or _tier_index(cap) < floor_index):
            refused.append(name)
    if refused:
        raise ValueError(
            f"certification_floor {floor} requires direct certification for every capsule; "
            f"{len(refused)} of {len(capsules)} fall below it: {', '.join(sorted(refused))}"
        )
    return {"floor": floor, "checked": len(capsules), "below_floor": []}


# ---- requirements ------------------------------------------------------------------------------
# Deterministic, installed requirement derivation from explicit iteration captures.
#
# No agent, candidate compiler or headline evaluation is involved. Authored inputs
# are declarations; generated inventories and synthesis plans are not certificates.


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


def _independent_public_sources(path: Path | None) -> bool:
    """Select the prospective source-only policy without reading private models.

    Leave legacy recipe validation and its original error ordering at their
    existing observation points. A malformed recipe never selects this policy.
    """
    from merlin.development.testgen.public_sources.boundaries import SCHEMA_V3, validate

    if path is None:
        return False
    try:
        recipe = yaml.safe_load(path.read_bytes())
    except (OSError, yaml.YAMLError):
        return False
    boundary = recipe.get("public_joint_boundaries") if isinstance(recipe, dict) else None
    if not isinstance(boundary, dict) or boundary.get("schema") != SCHEMA_V3:
        return False
    validate(boundary)
    return True


def _materialized_iteration_capsules(
    full: dict,
    digest: str,
    *,
    loader_snapshots: dict[str, tuple[Path, str]] | None = None,
    performance_selection: dict | None = None,
    performance_forms: dict | None = None,
    facts_sha256: str | None = None,
    capture_attestations: dict | None = None,
    source_role_selection=None,
    source_role_members: dict[str, str] | None = None,
) -> tuple[list[dict], dict[str, bytes]]:
    """Select full iteration programs, not a family representative or a held-out model.

    Save all producer-receipt members before exposing entries. A source loader
    becomes gradeable only when a verified preselected capture binds its exact
    snapshot bytes; an older receipt alone remains diagnostic.
    """
    from merlin.capture.artifacts import materialized_model_artifacts
    from merlin.development.testgen.public_sources import programs as PP

    entries, outputs = [], {}
    loader_snapshots = loader_snapshots or {}
    role = "iteration"
    prefix = "materialized"
    if source_role_selection is not None or source_role_members is not None:
        from merlin.development.testgen.public_sources.public_source_roles import OperatorSourceRoles

        if (
            type(source_role_selection) is not OperatorSourceRoles
            or performance_selection is None
            or not isinstance(source_role_members, dict)
            or set(source_role_members) != set(full["applications"])
        ):
            raise ValueError("prospective source roles need one live operator and every selected capture identity")
        source_role_selection.verify()
    if performance_selection is not None:
        PP.validate_selection(performance_selection)
        if (
            set(loader_snapshots) != set(full["applications"])
            or len(loader_snapshots) > performance_selection["max_members"]
        ):
            raise ValueError(
                "connected performance sources require the complete preselected public roster within budget"
            )
        role, prefix = "performance_scale", "materialized-performance"
    for label, application in sorted(full["applications"].items()):
        if not label or Path(label).name != label or label in {".", ".."}:
            raise ValueError("application identity must be a single safe path component")
        source = Path(application["capture_source_path"])
        if performance_selection is not None:
            PP.check_budget(source.parent, performance_selection)
        selection = {
            "path": str(source),
            "capture_sha256": application["capture_sha256"],
            "receipt_sha256": application["capture_receipt"]["receipt_sha256"],
            "workload_id": label,
            "workload_role": role,
            "coverage_scope": "full_capture",
            "full_inventory_sha256": digest,
            "operation_count": application["n_operations"],
        }
        if performance_selection is not None:
            selection["schema"] = PP.capture_schema(performance_selection)
            if performance_selection.get("source_mode"):
                selection["source_mode"] = performance_selection["source_mode"]
        # Read the common public bundle payload before installing its selected
        # loader/profile beside it. The final performance selection validates
        # that complete closure again through the ordinary writer and freeze.
        artifact = materialized_model_artifacts({**selection, "workload_role": "iteration"})
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
            outputs[f"{prefix}/{label}/{member}"] = path.read_bytes()
        # Independent recheck closes a producer mutation during the copy.
        copied = outputs[f"{prefix}/{label}/model.mlir"]
        if hashlib.sha256(copied).hexdigest() != selection["capture_sha256"]:
            raise ValueError(f"capture bytes changed while copying {label}")
        for member, identity in receipt["artifacts"].items():
            raw = outputs[f"{prefix}/{label}/{member}"]
            if len(raw) != identity["bytes"] or hashlib.sha256(raw).hexdigest() != identity["sha256"]:
                raise ValueError(f"receipt-bound bytes changed while copying {label}/{member}")
        if (
            artifact.meta.get("framework_catalog")
            and hashlib.sha256(outputs[f"{prefix}/{label}/pytorch-opset.json"]).hexdigest()
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
            outputs[f"{prefix}/{label}/loader.py"] = loader_bytes
            loader_binding = {"loader_path": f"{prefix}/{label}/loader.py", "loader_sha256": loader_sha256}
            profile = loader_path.with_name("profile.json")
            declared_profile = (artifact.meta.get("loader_provenance") or {}).get("profile_sha256")
            if profile.is_file() or declared_profile is not None:
                if profile.is_symlink() or not profile.is_file():
                    raise ValueError("materialized source lacks its selected profile")
                profile_raw = profile.read_bytes()
                if hashlib.sha256(profile_raw).hexdigest() != declared_profile:
                    raise ValueError("materialized loader profile differs from its actual capture")
                outputs[f"{prefix}/{label}/profile.json"] = profile_raw
                loader_binding.update(profile_path=f"{prefix}/{label}/profile.json", profile_sha256=declared_profile)
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
                    "path": f"{prefix}/{label}/model.mlir",
                    **loader_binding,
                },
                "generalization": {"generalization_axis": "composition"},
            }
        )
        if performance_selection is not None:
            entry = entries[-1]
            intent = None
            if source_role_selection is not None:
                intent = source_role_selection.bind(
                    source_role_members[label], {**entry["materialized_capture"], "path": str(source)}
                )
                entry["public_source_role"] = intent
            guard = intent is not None and intent["selected_member"]["role"] == "functional_guard"
            entry.update(
                name=f"FG_source_{label}" if guard else f"PG_source_{label}",
                cat="model" if guard else "_perf",
                label="public" if guard else "dev",
                source_role="materialized_performance_capture",
            )
            entry["source_reference"] = (
                "complete unchanged independent public performance capture; target execution and timing unverified"
            )
            if not guard:
                entry["performance"] = PP.performance(
                    entry,
                    trace=json.loads(outputs[f"{prefix}/{label}/frontend-trace.json"]),
                    forms=performance_forms or {},
                    facts_sha256=facts_sha256,
                )
            if performance_selection.get("source_mode"):
                from merlin.development.testgen.sealed_capture.attestation import require_verified_execution
                from merlin.development.testgen.sealed_capture.connected_integer_qualification import (
                    SELECTION,
                )

                if set(capture_attestations or {}) != set(full["applications"]):
                    raise ValueError("integer public sources require every original sealed capture attestation")
                attestation = capture_attestations[label]
                require_verified_execution(attestation)
                entry.update(
                    capture_quantization="already_materialized",
                    output_numeric_policy="captured_per_output_v1",
                    source_integer_proof=performance_selection.get("source_integer_proof", SELECTION),
                    capture_execution_attestations=[attestation],
                )
    if source_role_selection is not None:
        source_role_selection.verify()
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
    """Independent source captures checked against the declared performance roster.

    ``workload_spec.performance_applications`` names independent workloads whose forms enter only
    ``scope.performance.forms``. Explicit connected-source policies may also
    derive source capsules; functional v3 derives fresh bounded Phase 1 probes.
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
    the requirement (``merlin.target.capabilities.routing``).
    """
    roles = tuple(prohibited_roles or ())
    if not roles:
        return None
    from merlin.compiler.partition.groups import compute_groups as CG

    return CG.TargetOracle(target, prohibited_roles=roles)
