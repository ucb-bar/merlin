"""Host-owned Phase 0 profiles implementation."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

from merlin.analysis.perf.contract.profile import TRAITS  # noqa: E402
from merlin.common.digest import is_sha256
from merlin.common.paths import repo_root  # noqa: E402
from merlin.development.testgen.generation.provenance import _document_digest
from merlin.runtime.backends.registry import EXECUTION_CAPABILITIES  # noqa: E402
from merlin.target.spec.software import (
    load_software_spec,
    numerical_datapath,
    software_spec_identity,
    software_spec_path_for_recipe,
)

# ---- profiles ----------------------------------------------------------------------------------

_SYNTHESIS_INPUT_KEYS = frozenset({"conformance_spec_sha256", "recipe_sha256", "workload_spec_sha256"})


def _render_variant(variant: dict, combo: dict) -> dict:
    """A variant override with its string fields resolved against the shape point it is paired with.

    Only ``{extent}`` substitution, one level into a nested mapping -- enough for a comparison group to
    name the shape its members share (``fmb_{M}x{K}x{N}``) without any entry writing that shape down.
    A field with no placeholder is copied through untouched.
    """
    out: dict = {}
    for key, value in variant.items():
        if isinstance(value, str):
            out[key] = value.format(**combo)
        elif isinstance(value, dict):
            out[key] = {k: (v.format(**combo) if isinstance(v, str) else v) for k, v in value.items()}
        else:
            out[key] = value
    return out


def _cross_product(axes: dict) -> list[dict]:
    """Cross-product of ``{axis: [values]}`` preserving declaration order."""
    combos: list[dict] = [{}]
    for axis, values in axes.items():
        combos = [{**combo, axis: value} for combo in combos for value in values]
    return combos


def application_inventory_path(conformance_spec: str | Path, *, document: dict | None = None) -> Path | None:
    """Resolve only the requirement's adjacent generated sidecar, never an arbitrary path."""
    spec = Path(conformance_spec)
    document = document if document is not None else yaml.safe_load(spec.read_text(encoding="utf-8")) or {}
    if not isinstance(document, dict):
        raise ValueError(f"{spec}: conformance spec must be a mapping")
    demands = document.get("application_demands") or {}
    if not isinstance(demands, dict):
        raise ValueError(f"{spec}: application_demands must be a mapping")
    name = demands.get("sidecar")
    if name is None:
        return None
    if (
        not isinstance(name, str)
        or not name
        or name in (".", "..")
        or Path(name).name != name
        or "\\" in name
        or any(ord(char) < 32 for char in name)
    ):
        raise ValueError(f"{spec}: application-demand sidecar must be one adjacent basename")
    path = spec.parent / name
    if path.is_symlink():
        raise ValueError(f"{spec}: application-demand sidecar may not be a symlink")
    return path


def selected_software_spec_path(recipe: str | Path, override: str | Path | None = None) -> Path | None:
    """Resolve the recipe's explicit declaration or a frozen-input path override."""
    selected = software_spec_path_for_recipe(recipe)
    if override is not None:
        return Path(override).expanduser().absolute()
    return selected


def synthesis_input_identity(
    *,
    conformance_spec: str | Path,
    recipe: str | Path,
    descriptor: str | Path,
    software_spec: str | Path | None = None,
) -> dict:
    """The exact requirement and authored software inputs a synthesis used.

    The requirement and recipe are byte identities. ``workload_spec`` is a parsed
    declaration inside the descriptor, so its canonical document digest avoids
    invalidating synthesis for unrelated descriptor comments or hardware setup.
    """
    source = yaml.safe_load(Path(descriptor).read_text(encoding="utf-8")) or {}
    if not isinstance(source, dict) or not isinstance(source.get("workload_spec") or {}, dict):
        raise ValueError(f"{descriptor}: workload_spec must be a mapping")
    requirement_path = Path(conformance_spec)
    requirement_raw = requirement_path.read_bytes()
    requirement = yaml.safe_load(requirement_raw) or {}
    if not isinstance(requirement, dict):
        raise ValueError(f"{requirement_path}: conformance spec must be a mapping")
    selected_policies = requirement.get("quantization_policy_selections")
    if selected_policies is not None:
        if (
            not isinstance(selected_policies, dict)
            or selected_policies.get("schema") != "merlin.phase0.quantization_policy_selections.v1"
            or selected_policies.get("status") != "byte_selected_not_numerically_reviewed"
            or not isinstance(selected_policies.get("applications"), dict)
        ):
            raise ValueError("selected quantization policy sidecars are malformed")
        for label, record in sorted(selected_policies["applications"].items()):
            member = f"selected-quantization-policies/{label}.input"
            if (
                not isinstance(label, str)
                or not label
                or Path(label).name != label
                or not isinstance(record, dict)
                or record.get("artifact") != member
                or not is_sha256(record.get("sha256"))
            ):
                raise ValueError("selected quantization policy identity is malformed")
            path = requirement_path.parent / member
            if (
                path.is_symlink()
                or any(parent.is_symlink() for parent in path.parents)
                or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]
            ):
                raise ValueError(f"{label}: selected quantization policy sidecar differs")
    identity = {
        "conformance_spec_sha256": hashlib.sha256(requirement_raw).hexdigest(),
        "recipe_sha256": hashlib.sha256(Path(recipe).read_bytes()).hexdigest(),
        "workload_spec_sha256": _document_digest(source.get("workload_spec") or {}),
    }
    selected = selected_software_spec_path(recipe, software_spec)
    if selected is not None:
        identity["software_spec_sha256"] = software_spec_identity(selected)["sha256"]
    return identity


def verify_selected_synthesis(
    synth_profile: str | Path | None,
    *,
    conformance_spec: str | Path | None = None,
    recipe: str | Path | None = None,
    descriptor: str | Path | None = None,
    software_spec: str | Path | None = None,
    document: dict | None = None,
    diagnostic: bool = False,
) -> dict:
    """Verify a reviewed synth sidecar against its selected, frozen inputs.

    Historical profiles predate the identity block. They remain inspectable as
    diagnostics, but are explicitly *unverified*, never upgraded by an ambient
    checkout reference. Verified execution needs newly frozen selected inputs.
    """
    if synth_profile is None or not Path(synth_profile).is_file():
        return {"status": "absent"}
    selected = document if document is not None else yaml.safe_load(Path(synth_profile).read_text(encoding="utf-8"))
    if not isinstance(selected, dict):
        raise ValueError(f"{synth_profile}: synthesized profile must be a mapping")
    profile_document = selected
    provenance = selected.get("provenance") or {}
    if not isinstance(provenance, dict):
        raise ValueError(f"{synth_profile}: provenance must be a mapping")
    identity = provenance.get("selected_inputs")
    if identity is None:
        return {
            "status": "unverified_legacy",
            "reason": "selected synthesis predates conformance/recipe/workload input digests; regenerate and review it",
        }
    if (
        not isinstance(identity, dict)
        or set(identity) not in (_SYNTHESIS_INPUT_KEYS, _SYNTHESIS_INPUT_KEYS | {"software_spec_sha256"})
        or any(
            not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value)
            for value in identity.values()
        )
    ):
        raise ValueError(f"{synth_profile}: selected_inputs must contain selected-input SHA-256 digests")
    if conformance_spec is None or recipe is None or descriptor is None:
        raise ValueError(
            f"{synth_profile}: verified synthesis requires explicit conformance_spec, recipe, and descriptor inputs"
        )
    expected = synthesis_input_identity(
        conformance_spec=conformance_spec,
        recipe=recipe,
        descriptor=descriptor,
        software_spec=software_spec,
    )
    changed = sorted(key for key in set(identity) | set(expected) if identity.get(key) != expected.get(key))
    if changed:
        raise ValueError(
            f"{synth_profile}: stale selected synthesis ({', '.join(changed)} changed); "
            "regenerate a versioned sidecar and review/select it explicitly"
        )
    descriptor_doc = yaml.safe_load(Path(descriptor).read_text(encoding="utf-8")) or {}
    if not isinstance(descriptor_doc, dict):
        raise ValueError(f"{descriptor}: descriptor must be a mapping")
    applications = (descriptor_doc.get("workload_spec") or {}).get("applications") or {}
    spec_doc = yaml.safe_load(Path(conformance_spec).read_text(encoding="utf-8")) or {}
    if not isinstance(spec_doc, dict):
        raise ValueError(f"{conformance_spec}: conformance spec must be a mapping")
    demands = spec_doc.get("application_demands") or {}
    if not isinstance(demands, dict):
        raise ValueError(f"{conformance_spec}: application_demands must be a mapping")
    sidecar = application_inventory_path(conformance_spec)
    if applications and sidecar is None:
        raise ValueError(f"{conformance_spec}: declared applications require a selected detailed-demand sidecar")
    if sidecar is not None:
        if not sidecar.is_file():
            raise ValueError(f"{sidecar}: selected detailed-demand sidecar is missing")
        try:
            detailed = json.loads(sidecar.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{sidecar}: invalid detailed-demand JSON: {exc}") from exc
        if not isinstance(detailed, dict):
            raise ValueError(f"{sidecar}: detailed-demand inventory must be a mapping")
        actual = hashlib.sha256(json.dumps(detailed, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if actual != demands.get("full_inventory_sha256"):
            raise ValueError(f"{sidecar}: detailed-demand content differs from selected conformance spec")
        permitted_statuses = {"inventoried", "incomplete"} if diagnostic else {"inventoried"}
        if applications and (
            detailed.get("status") not in permitted_statuses or detailed.get("coverage_status") != "unverified"
        ):
            raise ValueError(
                f"{sidecar}: detailed-demand inventory must be "
                f"{'inventoried or incomplete' if diagnostic else 'inventoried'} with coverage unverified"
            )
        if applications:
            labels = {str(label) for label in applications}
            selected = detailed.get("applications") or {}
            if not isinstance(selected, dict) or set(selected) != labels:
                raise ValueError(f"{sidecar}: selected capture roster differs from declared applications")
            invalid = sorted(
                label
                for label, row in selected.items()
                if not isinstance(row, dict)
                or not isinstance(row.get("capture_receipt"), dict)
                or row["capture_receipt"].get("status") != "verified_materialized"
            )
            if invalid:
                raise ValueError(f"{sidecar}: declared applications lack verified materialization receipts: {invalid}")
        source_entries = [entry for entry in profile_document.get("capsules", []) if entry.get("materialized_capture")]
        performance_detailed = {}
        selected_programs = spec_doc.get("public_program_sources")
        if selected_programs is not None:
            from merlin.development.testgen.public_sources.programs import validate_selection

            validate_selection(selected_programs["selection"])
            recipe_document = yaml.safe_load(Path(recipe).read_bytes()) or {}
            if recipe_document.get("performance_source_programs") != selected_programs["selection"]:
                raise ValueError("public program selection differs from the ordinary recipe")
            if selected_programs.get("sidecar") != "performance-application-demands.json":
                raise ValueError("public program inventory must remain beside its selected requirement")
            if (Path(conformance_spec).parent / selected_programs["sidecar"]).is_symlink():
                raise ValueError("public program inventory may not be a symlink")
            performance_detailed = json.loads(
                (Path(conformance_spec).parent / selected_programs["sidecar"]).read_bytes()
            )
            performance_digest = hashlib.sha256(
                json.dumps(performance_detailed, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if performance_digest != selected_programs.get("full_inventory_sha256") or set(
                performance_detailed.get("applications") or {}
            ) != set((descriptor_doc.get("workload_spec") or {}).get("performance_applications") or []):
                raise ValueError("public program inventory differs from the declared selected performance roster")
        seen_sources = set()
        for entry in source_entries:
            binding = entry["materialized_capture"]
            label = binding.get("workload_id") if isinstance(binding, dict) else None
            performance_source = binding.get("workload_role") == "performance_scale"
            source_inventory = performance_detailed if performance_source else detailed
            source_row = (source_inventory.get("applications") or {}).get(label)
            if not isinstance(source_row, dict) or label in seen_sources:
                raise ValueError(f"{synth_profile}: duplicate or unknown materialized iteration identity")
            seen_sources.add(label)
            expected_source = {
                "capture_sha256": source_row["capture_sha256"],
                "receipt_sha256": source_row["capture_receipt"]["receipt_sha256"],
                "full_inventory_sha256": performance_digest if performance_source else actual,
                "operation_count": source_row["n_operations"],
                "workload_role": "performance_scale" if performance_source else "iteration",
                "coverage_scope": "full_capture",
            }
            expected_role = (
                "materialized_performance_capture" if performance_source else "materialized_iteration_capture"
            )
            if entry.get("source_role") != expected_role or any(
                binding.get(key) != value for key, value in expected_source.items()
            ):
                raise ValueError(
                    f"{synth_profile}: materialized source does not bind its exact full iteration inventory"
                )
            if performance_source:
                from merlin.development.testgen.public_sources.programs import capture_schema

                if (
                    binding.get("schema") != capture_schema(selected_programs["selection"])
                    or entry.get("cat") != "_perf"
                    or entry.get("label") != "dev"
                ):
                    raise ValueError("public performance program must be an explicitly selected dev member")
                source_mode = selected_programs["selection"].get("source_mode")
                if binding.get("source_mode") != source_mode:
                    raise ValueError("public capture source mode differs from its original selection")
                if source_mode:
                    from merlin.development.testgen.sealed_capture.connected_integer_qualification import (
                        SELECTION,
                    )

                    expected_attestation = (
                        (spec_doc.get("performance_capture_preselections") or {}).get("attestations", {}).get(label)
                    )
                    if (
                        expected_attestation is None
                        or entry.get("capture_execution_attestations") != [expected_attestation]
                        or entry.get("capture_quantization") != "already_materialized"
                        or entry.get("output_numeric_policy") != "captured_per_output_v1"
                        or entry.get("source_integer_proof")
                        != selected_programs["selection"].get("source_integer_proof", SELECTION)
                    ):
                        raise ValueError(
                            "integer public profile changed its sealed source or exact arithmetic selection"
                        )
    permitted_statuses = {"inventoried", "incomplete"} if diagnostic else {"inventoried"}
    from merlin.development.testgen.public_sources import boundaries as PB

    selected_boundaries = spec_doc.get("public_joint_boundaries")
    boundary_entries = [e for e in profile_document.get("capsules") or [] if e.get("public_boundary")]
    if selected_boundaries is not None:
        recipe_document = yaml.safe_load(Path(recipe).read_bytes()) or {}
        if PB.validate(recipe_document.get("public_joint_boundaries")) != selected_boundaries.get(
            "selection"
        ) or PB.digest(boundary_entries) != selected_boundaries.get("members_sha256"):
            raise ValueError("public boundary synthesis differs from its selected ordinary derivation")
        for entry in boundary_entries:
            member = entry["public_boundary"]
            if selected_boundaries["selection"]["schema"] == PB.SCHEMA_V3:
                selected_software = selected_software_spec_path(recipe, software_spec)
                if selected_software is None:
                    raise ValueError("functional sources require a selected software numerical contract")
                numerical = (yaml.safe_load(selected_software.read_bytes()) or {}).get("numerical_semantics")
                if (
                    not numerical
                    or selected_boundaries["numerical_semantics_sha256"] != PB.digest(numerical)
                    or member["numerical_semantics_sha256"] != selected_boundaries["numerical_semantics_sha256"]
                    or member["raw_facts_sha256"] != selected_boundaries["raw_facts_sha256"]
                    or member["selection"] != selected_boundaries["selection"]
                    or member["numerical_selection"] != selected_boundaries["numerical_selection"]
                    or member["numerical_selection"]["policy"] != recipe_document.get("public_numerical_policy")
                ):
                    raise ValueError("functional source changed its selected numerical/facts/policy identity")
                original = member["origin"]
                source = (performance_detailed.get("applications") or {}).get(original["workload_id"])
                if (
                    not source
                    or original["workload_role"] != "performance_scale"
                    or original["full_inventory_sha256"] != selected_programs["full_inventory_sha256"]
                    or original["capture_sha256"] != source["capture_sha256"]
                    or original["receipt_sha256"] != source["capture_receipt"]["receipt_sha256"]
                ):
                    raise ValueError("functional source is not part of the selected complete public source roster")
            PB.selected_loader(entry, Path(synth_profile).parent)
    elif boundary_entries:
        raise ValueError("public boundary members require an explicit ordinary recipe selection")
    if applications and demands.get("status") not in permitted_statuses:
        raise ValueError(
            f"{conformance_spec}: declared applications require an inventoried operation-demand requirement"
            f"{' or an explicitly diagnostic incomplete one' if diagnostic else ''}; "
            "regenerate and review the conformance spec and selected synthesis"
        )
    if applications and sidecar is not None and detailed.get("status") != demands.get("status"):
        raise ValueError(f"{conformance_spec}: detailed-demand and requirement statuses differ")
    coverage_status = demands.get("coverage_status", "unverified")
    if applications and coverage_status != "unverified":
        raise ValueError(f"{conformance_spec}: application-demand coverage must remain explicitly unverified")
    status = "incomplete_diagnostic" if applications and demands.get("status") == "incomplete" else "verified"
    return {"status": status, "selected_inputs": dict(identity), "application_coverage_status": coverage_status}


_PERFORMANCE_FIELDS = frozenset(
    {
        "level",
        "family",
        "lever",
        "member_class",
        "comparand",
        "falsifier",
        "gate",
        "regime",
        "emitter",
        "cost",
    }
)


_PERFORMANCE_CLAIMS = frozenset({"RECOVERS", "PREDICTS", "DIFFERENTIAL", "EMITS"})


_STREAM_CLAIMS = frozenset({"EMITS"})


_MEMBER_CLASSES = frozenset({"LAW", "OBJECTIVE", "REFERENCE"})


_PERFORMANCE_LEVELS = frozenset(
    {
        "L1_tile",
        "L1_separation_floor",
        "L2_intra_layer",
        "L3_inter_layer",
        "L4_boundary",
        "L5_fusion",
        "L6_global",
    }
)


_PERFORMANCE_NESTED_FIELDS = {
    "comparand": frozenset({"kind", "against", "cancels", "demand_equal"}),
    "falsifier": frozenset({"observation", "fires_when", "negative_control"}),
    "gate": frozenset({"traits", "instrument", "capacity", "on_missing"}),
    "regime": frozenset({"separation", "layout"}),
    "emitter": frozenset({"status", "entry", "knobs"}),
    "cost": frozenset({"tier", "runs", "projected_cycles", "basis"}),
}


HOST_SOURCE_BASIS = "merlin.phase0.host_source_execution_basis.v1"
