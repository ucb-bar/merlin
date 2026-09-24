"""Private source assembly for reviewed corpus releases; never a grading implementation."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import shutil
from contextlib import redirect_stdout
from pathlib import Path

import yaml

from ..spec import SpecError, read_yaml
from .phase_selection import validate_phase_selections


def private_json(path: Path, document: dict) -> None:
    """Publish a host-only record without an intermediate world-readable file."""
    temporary = path.with_name(path.name + ".pending")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(document, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def ordinary_tree(path: Path) -> None:
    """A release has no live indirection or unclassified filesystem entries."""
    if path.is_symlink() or not path.exists():
        raise SpecError("corpus release source is absent or symlinked")
    for member in [path, *path.rglob("*")] if path.is_dir() else [path]:
        if member.is_symlink() or not (member.is_file() or member.is_dir()):
            raise SpecError("corpus release source contains symlinked or nonregular entries")


def copy_input(source: Path, destination: Path, *, private: bool = False) -> str:
    from merlin.common import content_store

    from ..runner import fingerprint

    ordinary_tree(source)
    before = fingerprint(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Private and public inputs may independently contain identical bytes. Do
    # not merge their storage identities: an inode alias is an answer surface,
    # while byte equality alone does not make a declared public input private.
    store = None if private else content_store.store_root()
    if source.is_dir():
        content_store.place_tree(source, destination, store)
    else:
        content_store.place_file(source, destination, store)
    if fingerprint(source) != before or fingerprint(destination) != before:
        raise SpecError("corpus release source changed while copying; prepare a new release")
    return before


def source_run(run_dir: Path) -> tuple[dict, dict, Path]:
    from .. import runner

    record = runner.status(run_dir)
    plan = runner._read_json(run_dir / "resolved-plan.json")
    runner._verify_inputs(plan)
    phase = plan["phases"].get("0")
    attempts = [row for row in record["attempts"] if row["phase"] == "0"]
    if not phase or phase["adapter"] != "capsule_derivation" or not attempts:
        raise SpecError("release preparation requires a completed capsule-derivation phase")
    latest = attempts[-1]
    if latest["state"] != "execution_succeeded" or not latest.get("output_sha256"):
        raise SpecError("phase 0 has no successful immutable output receipt; derive a new corpus")
    output = Path(phase["engine_output"]).resolve()
    if latest.get("engine_output") != str(output) or not output.is_relative_to(run_dir):
        raise SpecError("phase-0 output receipt does not bind its run-owned corpus")
    ordinary_tree(output)
    if runner.fingerprint(output) != latest["output_sha256"]:
        raise SpecError("phase-0 output changed after successful derivation")
    provenance = read_yaml(output / "MANIFEST.yaml")
    if phase.get("module"):
        from ..adapters import PHASE0_MODULE, _legacy_script

        # _verify_inputs above binds actual installed module resolution, complete
        # source membership and every frozen source hash. Only that implementation
        # may retain the historical generator citation; an arbitrary string is not
        # a substitute for an execution/source receipt.
        if phase["module"] != PHASE0_MODULE or provenance.get("generated_by") != _legacy_script("capsule_derivation"):
            raise SpecError("phase-0 provenance does not identify the frozen derivation implementation")
        if latest.get("argv") != phase["argv"]:
            raise SpecError("phase-0 execution receipt differs from its frozen module command")
    else:
        source = Path(phase["env"]["MERLIN_REPO_ROOT"]) / str(provenance.get("generated_by", ""))
        if source.resolve() != Path(phase["entrypoint"]).resolve() or phase["argv"][1:2] != [phase["entrypoint"]]:
            raise SpecError("phase-0 provenance does not identify the frozen derivation entrypoint")
    return plan, latest, output


def _members(root: Path) -> dict[str, tuple[Path, dict]]:
    result = {}
    names = set()
    for path in sorted(root.glob("*/*/capsule.yaml")):
        document = read_yaml(path)
        name = document.get("name")
        relative = path.parent.relative_to(root).as_posix()
        if not isinstance(name, str) or name != path.parent.name or name in names:
            raise SpecError("corpus contains duplicate or inconsistent capsule identities")
        names.add(name)
        result[relative] = (path.parent, document)
    if not result:
        raise SpecError("corpus contains no capsule members")
    return result


def _reviewed_retirements(path: Path | None) -> tuple[dict[str, str], str | None]:
    """Read an explicit public-only retirement decision, bound by its source bytes."""
    from ..runner import fingerprint

    if path is None:
        return {}, None
    ordinary_tree(path)
    if not path.is_file():
        raise SpecError("retirement review input must be an ordinary file")
    document = read_yaml(path)
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != 1
        or not isinstance(document.get("retired"), dict)
    ):
        raise SpecError("retirement review input requires schema_version 1 and retired mapping")
    retired = document["retired"]
    for member, reason in retired.items():
        if (
            not isinstance(member, str)
            or len(Path(member).parts) != 2
            or Path(member).is_absolute()
            or ".." in Path(member).parts
            or member.startswith("hidden/")
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            raise SpecError("retirement decisions require public category/member paths and nonempty reasons")
    return retired, fingerprint(path)


def assemble(
    te,
    generated: Path,
    destination: Path,
    *,
    private_baseline: Path | None = None,
    retirements: Path | None = None,
) -> dict:
    """Copy one target's baseline and overlay only receipt-declared generated members."""
    from merlin.targetgen.sandbox.bwrap import _validate_host_sources

    from ..runner import fingerprint

    source_parent = te.capsule_corpus.parent
    private_roots = te.hidden_roots()
    if private_baseline is not None:
        if private_roots:
            raise SpecError("private baseline is already present in the descriptor corpus; do not replace it")
        ordinary_tree(private_baseline)
        if not private_baseline.is_dir():
            raise SpecError("private baseline must be a directory")
        private_roots = [private_baseline]
    roots = list(dict.fromkeys([*te.graded_roots(), *private_roots, *te.perf_roots(), *te.model_layer_roots()]))
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
    before = _members(destination)
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
    original_manifest = source_parent / "MANIFEST.yaml"
    original = read_yaml(original_manifest)
    prior_generated = set(original.get("generated") or [])
    classified = prior_generated | set(original.get("hand_authored") or [])
    if {key for key in before if not key.startswith("hidden/")} - classified:
        raise SpecError("baseline provenance does not classify every retained public corpus member")
    hidden_counts = original.get("held_out") or {}
    if sum(hidden_counts.get(key, 0) for key in ("n_generated", "n_hand_authored")) != sum(
        key.startswith("hidden/") for key in before
    ):
        raise SpecError("baseline provenance does not account for its private corpus")
    missing = (prior_generated & set(before)) - set(emitted)
    retired, retirement_digest = _reviewed_retirements(retirements)
    if set(retired) != missing:
        raise SpecError(
            "retirement review must account exactly for absent baseline generated members "
            f"(missing={len(missing)}, declared={len(retired)})"
        )
    retired_receipts = []
    for key in sorted(missing):
        target = destination / key
        retired_receipts.append({"member": key, "previous_sha256": fingerprint(target), "reason": retired[key]})
        # Only the release-local copy is removed; source and frozen run stay untouched.
        shutil.rmtree(target)
    existing_names = {value[1]["name"]: key for key, value in before.items()}
    replacements = []
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
        replacements.append({"member": key, "previous_sha256": previous_sha, "sha256": digest})
    final_members = _members(destination)
    # The promoted descriptor points at this release, so Phase 2 must see the
    # same generated provenance as Phase 0, not a live checkout's MANIFEST.
    # Functional grading still discovers only non-underscore categories.
    merged = copy.deepcopy(provenance)
    if retired_receipts:
        # The public manifest needs an audit commitment, not a list of removed
        # claim-model identities or reviewer prose. Detailed decisions stay in
        # the release-private preparation record.
        encoded = json.dumps(retired_receipts, sort_keys=True, separators=(",", ":")).encode("utf-8")
        merged["retired_generated"] = {
            "count": len(retired_receipts),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
    merged["hand_authored"] = sorted((set(original.get("hand_authored") or []) - declared) & set(final_members))
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
        "baseline": baseline,
        "retirements": {
            "source": str(retirements) if retirements else None,
            "sha256": retirement_digest,
            "members": retired_receipts,
        },
        "generated_manifest_sha256": fingerprint(generated / "MANIFEST.yaml"),
        "replacements": replacements,
    }


def derive_release_admission(staged) -> dict:
    """Seal a staged corpus's capability and cardinality decisions for review.

    Model resource decisions are authored policy until comparable cost evidence
    exists. Require one explicit decision per model so corpus growth cannot
    silently expand the expensive formal denominator.
    """
    from merlin.targetgen import capsule_runner, eligibility

    if not eligibility.capability_map_for_target(staged.target):
        raise SpecError("release admission requires a resolvable nonempty hardware capability contract")
    public = capsule_runner.discover_capsules(staged.graded_roots(), labels={"public", "dev"})
    public_names = [str(cap.get("name")) for cap in public]
    if len(public_names) != len(set(public_names)):
        raise SpecError("release public corpus has duplicate capsule names")
    models = {str(cap["name"]) for cap in public if cap.get("kind") == "model"}
    resource = set(staged.graded_resource_exclude)
    required = set(staged.graded_required_models)
    if resource & required or resource | required != models:
        raise SpecError(
            "release resource policy must classify every staged public model exactly once "
            "as excluded or required admitted"
        )
    operations = [cap for cap in public if cap.get("kind") != "model"]
    _, withheld = capsule_runner._split_ineligible(operations, staged.target)
    capability = sorted({str(row["capsule"]) for row in withheld})
    if set(capability) & models:
        raise SpecError("model capability admission cannot be inferred from operation admission")
    hidden = capsule_runner.discover_capsules(staged.hidden_roots(), labels={"hidden"})
    hidden_ops = [cap for cap in hidden if cap.get("kind") != "model"]
    _, hidden_withheld = capsule_runner._split_ineligible(hidden_ops, staged.target)
    hidden_excluded = {str(row["capsule"]) for row in hidden_withheld}
    return {
        "capability_exclude_capsules": capability,
        "expected_cohort": {
            "source_capsules": len(public),
            "admitted_capsules": len(public) - len(capability) - len(resource),
        },
        "hidden_capability_admission": {
            "source_capsules": len(hidden),
            "admitted_capsules": len(hidden) - len(hidden_excluded),
        },
        "resource_decisions": len(resource) + len(required),
    }


def scaffold(te, corpus: Path, experiment: Path, *, private: Path) -> dict:
    """Stage only declared task/selfcheck/harness resources, never prior runs or bundles."""
    from merlin.targetgen.target_experiment import load_target_experiment

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

    shim = module_source_path("merlin_experiments.phase1.tools.selfcheck")
    inputs["scripts/agent_selfcheck.py"] = {
        "path": str(shim),
        "sha256": copy_input(shim, experiment / "scripts/agent_selfcheck.py"),
    }
    document = copy.deepcopy(read_yaml(te.path))
    # A release owns the resources copied below; never retain a pointer to live authored inputs.
    document.pop("resources_root", None)
    document.pop("task_root", None)
    document.pop("contracts_root", None)
    document["capsule_corpus"] = str(corpus / te.capsule_corpus.name)
    if te.curated_harness:
        relative = Path(te.curated_harness)
        if relative.is_absolute() or ".." in relative.parts:
            raise SpecError("curated harness must be a declared contained experiment resource")
        source = te.resource_path(relative)
        inputs[relative.as_posix()] = {"path": str(source), "sha256": copy_input(source, experiment / relative)}
    descriptor = experiment / "target_experiment.yaml"
    descriptor.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    if te.graded_release_admission:
        staged = load_target_experiment(descriptor)
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
    from merlin.common.paths import python_source_dir
    from merlin.targetgen.generate_bundles import materialize_bundles

    promoted = load_target_experiment(descriptor)
    # Native generation can explain unavailable contracts. Keep those diagnostics
    # in the private preparation record, never the public inspect response.
    diagnostics = io.StringIO()
    with redirect_stdout(diagnostics):
        materialize_bundles(
            promoted,
            experiment / "input_bundles",
            variants=("public_v0", "realistic_v0", "hwbringup_v0"),
            host_inputs=(str(private),),
            python_source_root=python_source_dir(),
        )
    inputs["bundle_generation"] = {"diagnostics": diagnostics.getvalue()}
    return inputs


def admission(descriptor: Path) -> dict:
    """Reuse native discovery/admission without launching an oracle or changing policy."""
    from merlin.targetgen import capsule_runner
    from merlin.targetgen.contract.materialize import (
        _TIER_ORDER,
        materialize_public_cohort,
        validate_materialized_cohort,
    )
    from merlin.targetgen.target_experiment import load_target_experiment

    te = load_target_experiment(descriptor)
    capsule_runner.discover_capsules(te.graded_roots(), labels={"public", "dev"})
    materialized = materialize_public_cohort(te, tier_ceiling=_TIER_ORDER[-1])
    public = validate_materialized_cohort(materialized, te)
    hidden = capsule_runner.discover_capsules(te.hidden_roots(), labels={"hidden"})
    _, withheld = capsule_runner._split_ineligible([cap for cap in hidden if cap.get("kind") != "model"], te.target)
    excluded = {row["capsule"] for row in withheld}
    count = len(hidden)
    admitted = sum(cap["name"] not in excluded for cap in hidden)
    if public["n_admitted_capsules"] <= 0 or count <= 0 or admitted <= 0:
        raise SpecError("reviewable corpus must have nonempty public and hidden grading cohorts")
    if (te.hidden_expected_source_capsules is not None and count != te.hidden_expected_source_capsules) or (
        te.hidden_expected_admitted_capsules is not None and admitted != te.hidden_expected_admitted_capsules
    ):
        raise SpecError("hidden corpus differs from descriptor-frozen admission counts")
    return {
        "public_source": public["n_source_capsules"],
        "public_admitted": public["n_admitted_capsules"],
        "hidden_source": count,
        "hidden_admitted": admitted,
        "public_commitment": public["admitted_name_set_sha256"],
        "scope": "native cohort admission only; numerical and hardware readiness not executed",
    }
