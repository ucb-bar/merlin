"""Private source assembly for reviewed corpus releases; never a grading implementation."""

from __future__ import annotations

import copy
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


def assemble(te, generated: Path, destination: Path) -> dict:
    """Copy one target's baseline and overlay only receipt-declared generated members."""
    from merlin.targetgen.sandbox.bwrap import _validate_host_sources

    from ..runner import fingerprint

    source_parent = te.capsule_corpus.parent
    roots = list(dict.fromkeys([*te.graded_roots(), *te.hidden_roots(), *te.perf_roots(), *te.model_layer_roots()]))
    private_roots = te.hidden_roots()
    # Copying private files into independent inodes must not launder an existing
    # public alias. Reuse the native snapshot's authoritative privacy admission.
    _validate_host_sources(
        [(str(root), root) for root in roots if root not in private_roots],
        [(str(root), root) for root in private_roots],
    )
    generated_private = generated / "hidden"
    if generated_private.exists():
        _validate_host_sources(
            [(str(root), root) for root in generated.iterdir() if root != generated_private],
            [(str(generated_private), generated_private)],
        )
    baseline = {}
    for root in roots:
        if root.parent != source_parent:
            raise SpecError("descriptor corpus categories do not share one source root")
        baseline[root.name] = {
            "path": str(root),
            "sha256": copy_input(root, destination / root.name, private=root in private_roots),
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
    if (prior_generated & set(before)) - set(emitted):
        raise SpecError("baseline generated members are absent from derivation; removals require explicit provenance")
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
        "generated_manifest_sha256": fingerprint(generated / "MANIFEST.yaml"),
        "replacements": replacements,
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
