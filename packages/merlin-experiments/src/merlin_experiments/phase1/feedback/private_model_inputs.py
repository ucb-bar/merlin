"""Validate the private source-input, capture-tree and recipe bindings of a build."""
from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any


def _expected_input_provenance(row: Mapping[str, Any]) -> dict[str, bool | None]:
    """The frozen operator's claim about selected inputs, not a target-name rule."""
    value = row.get("input_provenance")
    if (
        not isinstance(value, Mapping)
        or set(value) != {"paper_ready", "synthetic_inputs"}
        or any(item is not None and type(item) is not bool for item in value.values())
    ):
        raise ValueError("operator-private model needs explicit input-provenance expectations")
    return dict(value)


def _require_selected_input_bindings(row: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    """Freeze the entire selected source-input roster, including transitive models.

    This exact operator-private declaration is not an authoring or public
    descriptor field. It binds guest identity and content without embedding
    validation tensors, model shapes, or source paths in the agent surface.
    """
    selected = plan.get("selected_inputs")
    expected = row.get("selected_input_bindings")
    if not isinstance(selected, list) or not selected or not isinstance(expected, list):
        raise ValueError("complete model needs an exact private selected-input roster")
    observed = []
    for entry in selected:
        if not isinstance(entry, Mapping):
            raise ValueError("selected source input is malformed")
        kind = entry.get("kind")
        digest = entry.get("sha256") if kind == "file" else (entry.get("tree") or {}).get("sha256")
        guest = entry.get("guest_member")
        role = entry.get("role")
        if (
            kind not in {"file", "tree"}
            or not isinstance(role, str)
            or not role
            or not isinstance(guest, str)
            or not guest
            or Path(guest).is_absolute()
            or ".." in Path(guest).parts
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise ValueError("selected source input has no safe content identity")
        observed.append({"role": role, "kind": kind, "guest_member": guest, "sha256": digest})
    if len({entry["guest_member"] for entry in observed}) != len(observed) or expected != observed:
        raise ValueError("private selected-input roster omits or changes a source dependency")


def _tree(spec: Path, value: Any, digest: Any, *, tree_digest) -> Path:
    if not isinstance(value, str) or not value or not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("private model tree needs a path and SHA256")
    path = Path(value)
    path = path if path.is_absolute() else spec.parent / path
    if tree_digest(path)["sha256"] != digest:
        raise ValueError(f"private model tree changed: {path}")
    return path.resolve(strict=True)


def _capture_tree_bindings(capture: Path, pinned_strict: Any, attested_sealed: Any, *, tree_digest) -> dict[str, str]:
    """Check independent compiler and sealed-issuer identities of one capture."""
    from merlin_experiments.phase0.capture_execution_attestation import sealed_m2m_tree_snapshot

    strict = tree_digest(capture)["sha256"]
    if not isinstance(pinned_strict, str) or pinned_strict != strict:
        raise ValueError("full-model capture differs from an explicitly pinned compiler tree")
    sealed = sealed_m2m_tree_snapshot(capture)["sha256"]
    if not isinstance(attested_sealed, str) or attested_sealed != sealed:
        raise ValueError("verified source-execution attestation names another complete capture")
    return {"compiler_strict_tree_sha256": strict, "issuer_sealed_tree_sha256": sealed}


def _recipe_derivation(
    spec: Path,
    row: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    target: str,
    software_sha256: str,
    capability_sha256: str,
    facts_sha256: str,
    host_capabilities_sha256: str,
    pinned_file,
    file_digest,
) -> tuple[dict[str, Any], dict, dict]:
    """Bind the selected capture recipe to canonical current-spec derivation.

    The Phase 0 export is a diagnostic transformation input, not an admission
    or a model-validation result.  Its independently checked source snapshots
    must match the very contracts used by this build, and its derived recipe
    must match the sealed capture preselection's actual recipe bytes. Return
    its resolved software and selected host views for independent source
    admission screening; never reinterpret the raw authored YAML here.
    """
    from merlin.targetgen.quant_recipe import digest as recipe_digest
    from merlin_experiments.phase0.evidence import load_exported_evidence

    root_value = row.get("recipe_derivation_root")
    if not isinstance(root_value, str) or not Path(root_value).is_absolute() or ".." in Path(root_value).parts:
        raise ValueError("private model needs an absolute recipe derivation root")
    root = Path(root_value)
    manifest = pinned_file(spec, str(root / "evidence-manifest.json"), row.get("recipe_derivation_manifest_sha256"))
    evidence = load_exported_evidence(root)
    if evidence.target != target:
        raise ValueError("recipe derivation names another target")
    expected_sources = {
        "software-spec": software_sha256,
        "target-contract": capability_sha256,
        "rtl-facts": facts_sha256,
    }
    for role, digest in expected_sources.items():
        snapshots = [source for source in evidence.source_snapshots if source.role == role]
        if len(snapshots) != 1 or snapshots[0].sha256 != digest:
            raise ValueError(f"recipe derivation does not bind selected {role} bytes")
    host_profiles = evidence.host_capabilities
    if not isinstance(host_profiles, Mapping):
        raise ValueError("recipe derivation has no selected host profiles")
    host_sources = [
        source
        for source in evidence.source_snapshots
        if source.role.startswith("host-capability-spec:") and source.sha256 == host_capabilities_sha256
    ]
    if len(host_sources) != 1 or host_sources[0].role.removeprefix("host-capability-spec:") not in host_profiles:
        raise ValueError("recipe derivation does not bind selected host-capability bytes")
    selected_host_name = host_sources[0].role.removeprefix("host-capability-spec:")
    artifacts = dict(evidence.archived_artifacts)
    index_raw = artifacts.get("software/quantization-recipes.json")
    if index_raw is None:
        raise ValueError("recipe derivation has no verified recipe index")
    index = json.loads(index_raw)
    if (
        not isinstance(index, Mapping)
        or index.get("schema") != "merlin.phase0.capture_recipes.v1"
        or index.get("target") != target
        or index.get("software_review") != "reviewed"
        or not isinstance(index.get("recipes"), list)
    ):
        raise ValueError("recipe derivation has no reviewed target recipe index")
    selected = plan.get("recipe")
    if not isinstance(selected, Mapping):
        raise ValueError("complete model has no selected capture recipe")
    selected_file = pinned_file(spec, selected.get("path"), selected.get("sha256"))
    variant = row.get("capture_recipe_variant")
    wanted = (
        {"sha256": variant.get("base_file_sha256"), "recipe_sha256": variant.get("base_recipe_sha256")}
        if isinstance(variant, Mapping) else selected
    )
    candidates = [
        entry
        for entry in index["recipes"]
        if isinstance(entry, Mapping)
        and entry.get("sha256") == wanted.get("sha256")
        and entry.get("recipe_sha256") == wanted.get("recipe_sha256")
    ]
    if len(candidates) != 1:
        raise ValueError("selected capture recipe is absent from current-spec derivation")
    member = candidates[0].get("path")
    if not isinstance(member, str) or not member.startswith("software/quantization-recipes/"):
        raise ValueError("current-spec recipe has no safe indexed member")
    raw = artifacts.get(member)
    if raw is None:
        raise ValueError("selected capture recipe differs from current-spec derived bytes")
    derived = json.loads(raw)
    if (
        not isinstance(derived, Mapping)
        or derived.get("target") != target
        or derived.get("recipe_sha256") != candidates[0]["recipe_sha256"]
        or recipe_digest(derived) != candidates[0]["recipe_sha256"]
        or sha256(raw).hexdigest() != candidates[0]["sha256"]
    ):
        raise ValueError("selected capture recipe has no verified current-spec content identity")
    variant_proof = None
    if variant is not None:
        from .private_recipe_variant import verify

        variant_proof = verify(variant, parent=raw, selected=selected_file.read_bytes(), target=target)
        if variant.get("selected_recipe_sha256") != selected.get("recipe_sha256"):
            raise ValueError("capture recipe variant names another selected semantic digest")
    elif raw != selected_file.read_bytes():
        raise ValueError("selected capture recipe has no verified current-spec content identity")
    proof = {
        "evidence_manifest_sha256": file_digest(manifest),
        "recipe_sha256": file_digest(selected_file),
        "recipe_semantic_sha256": selected["recipe_sha256"],
        "scope": "current-spec diagnostic recipe derivation only; no model or hardware admission",
    }
    if variant_proof is not None:
        proof["variant"] = variant_proof
    return proof, evidence.software_spec, {selected_host_name: host_profiles[selected_host_name]}
