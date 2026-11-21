"""Private source assembly for reviewed corpus releases; never a grading implementation."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import yaml

from merlin.experiments.corpus.phase_selection import validate_phase_selections
from merlin.experiments.corpus.source_staging import (
    _members,
    ordinary_tree,
)
from merlin.experiments.definition import SpecError, read_yaml

# ---- preparation -------------------------------------------------------------------------------


def assemble(
    te,
    generated: Path,
    destination: Path,
    *,
    private_baseline: Path | None = None,
    retirements: Path | None = None,
    generated_only: bool = True,
    capsule_scope: str = "all",
    loader_profile_sources: tuple[Path, ...] | None = None,
) -> dict:
    """Assemble generated public members, with an explicit safe historical overlay."""
    from merlin.development.targetgen.sandbox.grants import _validate_host_sources
    from merlin.experiments.corpus.coverage import select_members, validate_scope
    from merlin.experiments.corpus.source_staging import (
        _reviewed_retirements,
        copy_input,
        publish_loader_profile,
        retain_deferred_performance,
    )
    from merlin.experiments.runs.plan_sources import fingerprint

    validate_scope(capsule_scope, generated_only=generated_only)
    if capsule_scope != "all" and private_baseline is not None:
        raise SpecError("accelerator selection cannot mix a historical private baseline")

    source_parent = te.capsule_corpus.parent
    if generated_only and retirements is not None:
        raise SpecError("generated-only release has no historical members to retire")
    private_roots = [] if generated_only else te.hidden_roots()
    if private_baseline is not None:
        if private_roots:
            raise SpecError("private baseline is already present in the descriptor corpus; do not replace it")
        ordinary_tree(private_baseline)
        if not private_baseline.is_dir():
            raise SpecError("private baseline must be a directory")
        private_roots = [private_baseline]
    roots = (
        list(private_roots)
        if generated_only
        else list(dict.fromkeys([*te.graded_roots(), *private_roots, *te.perf_roots(), *te.model_layer_roots()]))
    )
    # Copying private files into independent inodes must not launder an existing
    # public alias. Reuse the native snapshot's authoritative privacy admission.
    generated_private = generated / "hidden"
    # Check all source surfaces together. A private external baseline aliased to
    # a generated public file is just as unsafe as an alias inside either tree.
    _validate_host_sources(
        [(str(root), root) for root in roots if root not in private_roots]
        + [(str(root), root) for root in generated.iterdir() if root != generated_private],
        [(str(root), root) for root in private_roots]
        + ([(str(generated_private), generated_private)] if generated_private.exists() else []),
    )
    from merlin.experiments.corpus.publication import require_public_capsules

    require_public_capsules(te, [root for root in roots if root not in private_roots])
    require_public_capsules(te, [root for root in generated.iterdir() if root != generated_private])
    baseline = {}
    for root in roots:
        private_external = root == private_baseline
        if root.parent != source_parent and not private_external:
            raise SpecError("descriptor corpus categories do not share one source root")
        category = "hidden" if private_external else root.name
        baseline[category] = {
            "path": str(root),
            "sha256": copy_input(root, destination / category, private=root in private_roots),
        }
    before = _members(destination) if roots else {}
    emitted = _members(generated)
    provenance = read_yaml(generated / "MANIFEST.yaml")
    declared = set(provenance.get("generated") or [])
    public_emitted = {key for key in emitted if not key.startswith("hidden/")}
    hidden_emitted = len(emitted) - len(public_emitted)
    if declared != public_emitted or (provenance.get("held_out") or {}).get("n_generated", 0) != hidden_emitted:
        raise SpecError("phase-0 provenance does not account for every generated capsule")
    phase_corpora = provenance.get("phase_corpora")
    if phase_corpora is not None:
        record = (provenance.get("performance_generation") or {}).get(te.target) or {}
        category = (record.get("phase") or {}).get("category") or "_perf"
        try:
            validate_phase_selections(
                phase_corpora.get(te.target) if isinstance(phase_corpora, dict) else None,
                performance_category=category,
                generated_members=declared,
                complete=True,
            )
        except ValueError as exc:
            raise SpecError(f"phase-0 corpus selections do not match emitted capsules: {exc}") from exc
    scope_record = None
    if capsule_scope == "accelerator":
        emitted, scope_record = select_members(emitted, fingerprint=fingerprint)
        declared = {key for key in emitted if not key.startswith("hidden/")}
    original = (
        {
            "generated": [],
            "hand_authored": [],
            "held_out": {"n_generated": 0, "n_hand_authored": len(before)},
        }
        if generated_only
        else read_yaml(source_parent / "MANIFEST.yaml")
    )
    prior_generated = set(original.get("generated") or [])
    classified = (
        prior_generated | set(original.get("hand_authored") or []) | set(original.get("operator_requested") or [])
    )
    if {key for key in before if not key.startswith("hidden/")} - classified:
        raise SpecError("baseline provenance does not classify every retained public corpus member")
    hidden_counts = original.get("held_out") or {}
    if sum(hidden_counts.get(key, 0) for key in ("n_generated", "n_hand_authored")) != sum(
        key.startswith("hidden/") for key in before
    ):
        raise SpecError("baseline provenance does not account for its private corpus")
    missing = (prior_generated & set(before)) - set(emitted)
    retired, retirement_digest = _reviewed_retirements(retirements)
    # Historical hand-authored public members are normally preserved.  An
    # operator may explicitly retire one (for example, a validation model that
    # must move behind the post-freeze boundary), but the review file must not
    # name a new, private, or freshly generated member.
    hand_authored = set(original.get("hand_authored") or []) & set(before)
    reviewed_hand_authored = set(retired) - missing
    if not missing.issubset(retired) or not reviewed_hand_authored.issubset(hand_authored - set(emitted)):
        raise SpecError(
            "retirement review must account exactly for absent baseline generated members "
            "and name only existing, unreplaced hand-authored public members "
            f"(missing={len(missing)}, declared={len(retired)})"
        )
    retired_receipts = []
    for key in sorted(retired):
        target = destination / key
        retired_receipts.append({"member": key, "previous_sha256": fingerprint(target), "reason": retired[key]})
        # Only the release-local copy is removed; source and frozen run stay untouched.
        shutil.rmtree(target)
    existing_names = {value[1]["name"]: key for key, value in before.items()}
    replacements = []
    profile_dependencies = []
    for key, (source, document) in emitted.items():
        previous = before.get(key)
        if document["name"] in existing_names and existing_names[document["name"]] != key:
            raise SpecError("generated capsule collides with a different source category")
        if previous and any(previous[1].get(field) != document.get(field) for field in ("name", "label", "kind")):
            raise SpecError("generated replacement changes capsule identity, kind, or visibility")
        target = destination / key
        previous_sha = fingerprint(target) if target.exists() else None
        if target.exists():
            # This is only the fresh release's copy, never its source or historical evidence.
            shutil.rmtree(target)
        digest = copy_input(source, target, private=key.startswith("hidden/"))
        if loader_profile_sources is not None:
            dependency = publish_loader_profile(source, target, document, profile_sources=loader_profile_sources)
            if dependency is not None:
                profile_dependencies.append({"member": key, **dependency})
                digest = fingerprint(target)
        replacements.append({"member": key, "previous_sha256": previous_sha, "sha256": digest})
    final_members = _members(destination)
    from merlin.experiments.corpus.publication import require_accelerator_placement

    # Every graded public member, generated or retained, must place work on the accelerator.
    require_accelerator_placement(
        sorted({destination / key.split("/", 1)[0] for key in final_members if not key.startswith(("hidden/", "_"))})
    )
    # Only this derivation may supply new completeness inputs. A historical
    # baseline sidecar cannot fill an absent source trace in the selected run.
    from merlin.development.testgen.evidence.coverage_obligations_coverage_commitment import (
        INPUT_PATH,
        read_inputs,
    )

    coverage_inputs = read_inputs(generated, provenance)
    if coverage_inputs is not None and scope_record is None:
        copy_input(generated / INPUT_PATH, destination / INPUT_PATH, private=True)
    retain_deferred_performance(generated, destination, provenance, target=te.target)
    # The promoted descriptor points at this release, so Phase 2 must see the
    # same generated provenance as Phase 0, not a live checkout's MANIFEST.
    # Functional grading still discovers only non-underscore categories.
    merged = copy.deepcopy(provenance)
    if scope_record is not None:
        from merlin.development.testgen.evidence.coverage_obligations_coverage_commitment import write_inputs
        from merlin.experiments.corpus.phase_selection import generate_phase_selections

        if coverage_inputs is None:
            raise SpecError("accelerator selection requires original manifest-bound coverage inputs")
        # Source application obligations stay unchanged; only standalone test scope is selected.
        public_scope = {key: value for key, value in scope_record.items() if key != "members"}
        public_scope["source_coverage_inputs_sha256"] = provenance["coverage_inputs"]["sha256"]
        merged["coverage_inputs"] = write_inputs(destination, {**coverage_inputs, "capsule_scope": public_scope})
        merged["capsule_scope"] = public_scope
        merged["generated"] = sorted(declared)
        if phase_corpora is not None:
            merged["phase_corpora"][te.target] = generate_phase_selections(
                sorted(declared), performance_category=category
            )
    if retired_receipts:
        # The public manifest needs an audit commitment, not a list of removed
        # claim-model identities or reviewer prose. Detailed decisions stay in
        # the release-private preparation record.
        for key, members in (("retired_generated", missing), ("retired_hand_authored", reviewed_hand_authored)):
            if members:
                selected = [row for row in retired_receipts if row["member"] in members]
                encoded = json.dumps(selected, sort_keys=True, separators=(",", ":")).encode("utf-8")
                merged[key] = {"count": len(selected), "sha256": hashlib.sha256(encoded).hexdigest()}
    merged["hand_authored"] = sorted((set(original.get("hand_authored") or []) - declared) & set(final_members))
    if original.get("operator_requested"):
        merged["operator_requested"] = sorted((set(original["operator_requested"]) - declared) & set(final_members))
        merged["public_supplements"] = copy.deepcopy(original.get("public_supplements", []))
    previous_hidden = sum((original.get("held_out") or {}).get(key, 0) for key in ("n_generated", "n_hand_authored"))
    new_hidden = {key for key in emitted if key.startswith("hidden/")} - set(before)
    merged["held_out"] = {
        "n_generated": (original.get("held_out") or {}).get("n_generated", 0) + len(new_hidden),
        "n_hand_authored": (original.get("held_out") or {}).get("n_hand_authored", 0),
    }
    if previous_hidden + len(new_hidden) != sum(key.startswith("hidden/") for key in final_members):
        raise SpecError("promoted hidden corpus count disagrees with provenance")
    (destination / "MANIFEST.yaml").write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")
    return {
        "mode": "generated_only" if generated_only else "historical_overlay",
        "baseline": baseline,
        "retirements": {
            "source": str(retirements) if retirements else None,
            "sha256": retirement_digest,
            "members": retired_receipts,
        },
        "generated_manifest_sha256": fingerprint(generated / "MANIFEST.yaml"),
        "replacements": replacements,
        **({"capsule_scope": scope_record} if scope_record is not None else {}),
        **({"loader_profile_dependencies": profile_dependencies} if loader_profile_sources is not None else {}),
    }


def _stage_capability_contract(
    source,
    expected_sha256,
    destination,
    native_timeout_s,
    gsim_readback_transport=None,
    *,
    model_spike_timeout_s=None,
    operand_selection=None,
    operand_revision=None,
    kernel_timing_selection=None,
    logical_storage_budget_bytes=None,
):
    """Record closed execution selections separately from original Phase 0 bytes."""
    from merlin.common.digest import sha256_file
    from merlin.experiments.corpus.source_staging import copy_input
    from merlin.runtime.backends.rocc import tool_selection as rocc_selection
    from merlin.runtime.backends.rocc import tool_selection as runtime

    if (
        native_timeout_s is None
        and model_spike_timeout_s is None
        and gsim_readback_transport is None
        and operand_selection is None
        and operand_revision is None
        and kernel_timing_selection is None
        and logical_storage_budget_bytes is None
    ):
        return {"path": str(source), "sha256": copy_input(source, destination, expected_sha256=expected_sha256)}
    if operand_revision is not None and operand_selection is None:
        raise SpecError("observer revision requires an explicit selected observer")
    ordinary_tree(source)
    if sha256_file(source) != expected_sha256:
        raise SpecError("selected corpus input differs from verified evidence; prepare a new release")
    try:
        selected = read_yaml(source)
        timing_record = None
        storage_record = None
        if kernel_timing_selection is not None:
            from merlin.experiments.corpus.kernel_timing_selection import select

            selected, timing_record = select(selected, kernel_timing_selection)
        if native_timeout_s is not None:
            selected = rocc_selection.select_native_timeout(selected, native_timeout_s)
        if model_spike_timeout_s is not None:
            selected = rocc_selection.select_model_spike_timeout(selected, model_spike_timeout_s)
        if gsim_readback_transport is not None:
            selected = rocc_selection.select_gsim_readback(selected, gsim_readback_transport)
        if logical_storage_budget_bytes is not None:
            from merlin.experiments.corpus.storage_budget_selection import select as select_storage

            selected, storage_record = select_storage(selected, logical_storage_budget_bytes)
        if operand_selection is not None:
            select = (
                rocc_selection.relocate_operand_protocol
                if "rocc_operand_selection" in selected
                else rocc_selection.select_operand_protocol
            )
            if operand_revision is not None:
                from merlin.experiments.corpus.operand_revision import select as revise

                selected = revise(selected, operand_selection, operand_revision)
            else:
                selected = select(selected, operand_selection)
    except (runtime.RoCCExecutionError, ValueError) as error:
        raise SpecError(str(error)) from error
    native_engines = set(selected["runner"]["chipyard_rocc"]["engines"]) & {"gsim", "verilator"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as output:
        yaml.safe_dump(selected, output, sort_keys=False)
    if sha256_file(source) != expected_sha256:
        raise SpecError("selected original contract changed during execution selection")
    record = {"path": str(source), "sha256": expected_sha256}
    selected_sha256 = sha256_file(destination)
    if storage_record is not None:
        record["logical_storage_selection"] = {**storage_record, "selected_contract_sha256": selected_sha256}
    if timing_record is not None:
        record["kernel_timing_selection"] = {**timing_record, "selected_contract_sha256": selected_sha256}
    if native_timeout_s is not None:
        record["execution_resource_selection"] = {
            "max_native_timeout_s": native_timeout_s,
            "engines": sorted(native_engines),
            "selected_contract_sha256": selected_sha256,
            "scope": "host process resources only; original semantic and numerical declarations retained",
        }
    if model_spike_timeout_s is not None:
        record["model_spike_resource_selection"] = {
            "max_model_spike_timeout_s": model_spike_timeout_s,
            "engine": "spike",
            "selected_contract_sha256": selected_sha256,
            "scope": "whole-model host process budget only; original semantic and numerical declarations retained",
        }
    if gsim_readback_transport is not None:
        record["execution_transport_selection"] = {
            "engine": "gsim",
            "readback_transport": gsim_readback_transport,
            "selected_contract_sha256": selected_sha256,
            "scope": "complete output values only; original ABI, instruction and numerical declarations retained",
        }
    if operand_selection is not None:
        record["operand_protocol_selection"] = {
            **copy.deepcopy(operand_selection),
            "selected_contract_sha256": selected_sha256,
            "scope": (
                "explicit reviewed additive operand revision; original facts and existing fields retained"
                if operand_revision is not None
                else "source-derived operand observations only; original facts and semantic declarations retained"
            ),
        }
    if operand_revision is not None:
        record["operand_protocol_revision"] = copy.deepcopy(operand_revision)
    return record
