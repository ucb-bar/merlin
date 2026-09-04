"""Fail-closed boundary between the completed functional Arm-4 run and performance.

The performance campaign is allowed to consume exactly one content-addressed functional
submission.  It copies that submission into a run-private workspace, makes the copy read-only,
records the functional fork, and executes every untrusted package entrypoint in a credential-free,
mount-scoped bwrap sandbox. Network is available and is not part of the isolation claim. Pure helpers
in this module make the refusal paths testable without starting a simulator.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import stat
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import yaml

from merlin.benchharness import hash_tree
from merlin.common.paths import repo_root, resolve_grant
from merlin.perf.campaign import ReplicaIdentity
from merlin.perf.fork import ForkPoint, candidate_states, check_invariants, fork_from
from merlin.perf.profile import TRAITS
from merlin.targetgen import oot_runner
from merlin.targetgen.oracle_schedule import CapsuleState, Verdict
from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.sandbox import toolchain as TC
from merlin.targetgen.sandbox.answer_surfaces import answer_surfaces
from merlin.targetgen.target_experiment import TargetExperiment


class CampaignGateError(RuntimeError):
    """A phase-boundary or completion condition was not proved."""


@dataclass(frozen=True)
class FunctionalRun:
    run_dir: Path
    submission_dir: Path
    run_id: str
    digest: str
    public_capsules: int
    hidden_capsules: int
    public_score: dict
    hidden_score: dict
    frozen_at: str
    bundle_input_snapshot: dict
    model_host_lane_snapshot: dict
    model_host_package: Path


@dataclass(frozen=True)
class PackageSandboxPolicy:
    argv: tuple[str, ...]
    coverage_gap: tuple[str, ...]
    required_tools: tuple[TC.ToolProbe, ...]
    workspace: Path
    package: Path
    execution_package: Path
    package_sha256: str
    target_experiment: TargetExperiment


@dataclass(frozen=True)
class PerformanceCapsule:
    """One generator-owned capsule admitted from the descriptor-derived performance phase."""

    family: str
    capsule: str
    source_dir: Path
    source_relative_path: str
    descriptor: dict
    source_sha256: str
    n_files: int
    n_bytes: int


@dataclass(frozen=True)
class PerformanceCorpus:
    """The exact generated corpus selected before a campaign allocates simulator work."""

    target: str
    corpus_root: Path
    phase_root: Path
    provenance_manifest: Path
    provenance_sha256: str
    performance_generation: dict
    capsules: tuple[PerformanceCapsule, ...]


@dataclass(frozen=True)
class FrozenPerformanceCorpus:
    """Read-only private copy of the selected performance inputs and its byte manifest."""

    root: Path
    capsules_root: Path
    manifest_path: Path
    manifest_sha256: str
    capsules_sha256: str
    capsules: tuple[PerformanceCapsule, ...]


@dataclass(frozen=True)
class FullModelSentinel:
    """Smallest frozen public whole-model capsule spanning accelerator and scalar/vector lanes."""

    capsule: str
    source_dir: Path
    descriptor: dict
    source_sha256: str
    n_files: int
    n_bytes: int


def candidate_handoff_record(handoff: object) -> dict[str, Any]:
    """Serialize only the stage's verified, normalized handoff into campaign attribution.

    This deliberately accepts the narrow accessor result by protocol instead of importing the
    authoring module (which already imports this campaign module).  Both the runner and reporter call
    this one serializer, so neither mirrors the candidate-record schema.
    """
    def path(name: str) -> str:
        value = Path(getattr(handoff, name))
        return str(value.resolve(strict=True))

    record = {
        "record_path": path("record_path"),
        "record_sha256": getattr(handoff, "record_sha256"),
        "candidate_path": path("candidate_path"),
        "candidate_sha256": getattr(handoff, "candidate_sha256"),
        "candidate_initial_sha256": getattr(handoff, "candidate_initial_sha256"),
        "functional_run_id": getattr(handoff, "functional_run_id"),
        "functional_submission_sha256": getattr(handoff, "functional_submission_sha256"),
        "functional_base_path": path("functional_base_path"),
        "functional_bundle_snapshot_sha256": getattr(
            handoff, "functional_bundle_snapshot_sha256"),
        "functional_bundle_manifest": path("functional_bundle_manifest"),
        "functional_bundle_manifest_sha256": getattr(
            handoff, "functional_bundle_manifest_sha256"),
        "target_descriptor": path("target_descriptor"),
        "target_descriptor_sha256": getattr(handoff, "target_descriptor_sha256"),
        "corpus_root": path("corpus_root"),
        "corpus_manifest": path("corpus_manifest"),
        "corpus_manifest_sha256": getattr(handoff, "corpus_manifest_sha256"),
        "corpus_sha256": getattr(handoff, "corpus_sha256"),
        "replicates": getattr(handoff, "replicates"),
        "formal_replicate_identities": list(getattr(handoff, "formal_replicate_identities")),
        "formal_claim": dict(getattr(handoff, "formal_claim")),
        "smoke_replicates": getattr(handoff, "smoke_replicates"),
        "expected_cells": [dict(row) for row in getattr(handoff, "expected_cells")],
        "families": [dict(row) for row in getattr(handoff, "families")],
        "host_lane": dict(getattr(handoff, "host_lane")),
        "e2e_sentinel": dict(getattr(handoff, "e2e_sentinel")),
        "prompt_path": path("prompt_path"),
        "prompt_sha256": getattr(handoff, "prompt_sha256"),
        "prompt_facts_sha256": getattr(handoff, "prompt_facts_sha256"),
        "transcript_path": path("transcript_path"),
        "transcript_sha256": getattr(handoff, "transcript_sha256"),
        "transcript_audit": dict(getattr(handoff, "transcript_audit")),
        "receipt_path": path("receipt_path"),
        "receipt_sha256": getattr(handoff, "receipt_sha256"),
        "required_actions": list(getattr(handoff, "required_actions")),
        "tool_evidence": dict(getattr(handoff, "tool_evidence")),
        "sandbox_evidence": dict(getattr(handoff, "sandbox_evidence")),
    }
    return record


@dataclass(frozen=True, order=True)
class PerfCellIdentity:
    """The non-inferable identity of one simulator observation."""

    family: str
    capsule: str
    simulator: str
    replicate: str

    def __post_init__(self) -> None:
        for field, value in (("family", self.family), ("capsule", self.capsule),
                             ("simulator", self.simulator), ("replicate", self.replicate)):
            if not isinstance(value, str) or not value.strip():
                raise CampaignGateError(
                    f"performance cell identity {field} must be a non-empty string")

    def as_dict(self) -> dict[str, str]:
        return {
            "family": self.family,
            "capsule": self.capsule,
            "simulator": self.simulator,
            "replicate": self.replicate,
        }

    def decision_identity(self) -> ReplicaIdentity:
        """Narrow bridge to the target-agnostic promotion engine.

        The runner can provide exact measurements through this bridge.  It cannot itself decide
        promotion because a capsule grade has no run-authored falsifier or differential evidence.
        """
        tier = "screen" if self.simulator == "spike" else "certify"
        return ReplicaIdentity(self.family, self.capsule, tier, self.replicate)


#: `acceptance` is DELIBERATELY NOT HERE; it is required below, and only of a PREDICTS claim.
#:
#: An acceptance block freezes the decision rule for a FITTED prediction -- the fit form, the cohort,
#: the replicate contract and the thresholds a residual is judged against. A DIFFERENTIAL claim fits
#: nothing: its verdict is its falsifier over an A/B on identical work, and there is no coefficient to
#: bound. Requiring one of every family made the whole corpus refuse on the first DIFFERENTIAL capsule,
#: and the only way to satisfy it would have been to invent six thresholds nobody measured -- the exact
#: failure the frozen-contract discipline exists to prevent.
#:
#: This restores agreement with the two ends of the chain that were already conditional. The WRITER
#: (`generate_corpus._PERFORMANCE_FIELDS`) has never required it, so the generator emits DIFFERENTIAL
#: capsules without one; and `perf_prompt.PerfFamily.__post_init__`, the type THIS module's own
#: consumer builds, requires it for PREDICTS and explicitly admits `None` otherwise. The gate here was
#: the only unconditional demand, and it contradicted both.
_PERFORMANCE_FIELDS = frozenset({
    "level", "family", "lever", "claim", "comparand", "falsifier", "gate", "regime",
    "emitter", "cost",
})
_PERFORMANCE_CLAIMS = frozenset({"RECOVERS", "PREDICTS", "DIFFERENTIAL"})
_PERFORMANCE_NESTED_FIELDS = {
    "comparand": frozenset({"kind", "against", "cancels", "demand_equal"}),
    "falsifier": frozenset({"observation", "fires_when", "negative_control"}),
    "gate": frozenset({"traits", "instrument", "capacity", "on_missing"}),
    "regime": frozenset({"separation", "layout"}),
    "emitter": frozenset({"status", "entry", "knobs"}),
    "cost": frozenset({"tier", "runs", "projected_cycles", "basis"}),
}
_SIMULATOR_POLICY = {
    "spike": {"tier": "L2", "purpose": "correctness_screen", "citable": False},
    "verilator": {"tier": "L3", "purpose": "performance_certification", "citable": True},
}
SMOKE_CLAIM_NONCLAIM = "measurement-smoke mode does not request a performance-claim decision"


def _mapping_file(path: Path, *, yaml_file: bool = False) -> dict:
    if not path.is_file():
        raise CampaignGateError(f"required campaign evidence is absent: {path}")
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) if yaml_file else json.loads(
            path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise CampaignGateError(f"campaign evidence is unreadable at {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise CampaignGateError(f"campaign evidence must be a mapping: {path}")
    return doc


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(document: object) -> bytes:
    return (json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            + "\n").encode("utf-8")


def _document_digest(document: object) -> str:
    return _sha256_bytes(_canonical_json_bytes(document).rstrip(b"\n"))


def _is_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64 and value.lower() == value
            and all(character in "0123456789abcdef" for character in value))


def _simple_name(value: object, *, label: str) -> str:
    name = str(value or "")
    if not name or Path(name).name != name or name in (".", ".."):
        raise CampaignGateError(f"{label} must be a simple non-empty path component, got {name!r}")
    return name


def _exact_tree_record(root: Path) -> dict[str, Any]:
    """Hash every file and relative path; reject anything the historical tree hash would omit."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise CampaignGateError(f"exact performance input is not a real directory: {root}")
    digest = hashlib.sha256()
    n_files = n_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise CampaignGateError(f"exact performance input contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise CampaignGateError(f"exact performance input contains a special file: {path}")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(relative)
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
        n_files += 1
        n_bytes += len(payload)
    if n_files <= 0 or n_bytes <= 0:
        raise CampaignGateError(f"exact performance input is empty: {root}")
    return {"sha256": digest.hexdigest(), "n_files": n_files, "n_bytes": n_bytes}


def _validate_performance_block(block: object, *, owner: str) -> dict:
    """Validate the canonical claim block at the consumption boundary, without importing its writer."""
    if not isinstance(block, dict):
        raise CampaignGateError(f"{owner} is not a generated performance capsule")
    missing = sorted(_PERFORMANCE_FIELDS - block.keys())
    if missing:
        raise CampaignGateError(f"{owner} performance block is non-canonical; missing {missing}")
    for field in ("level", "family", "lever"):
        if not isinstance(block.get(field), str) or not str(block[field]).strip():
            raise CampaignGateError(f"{owner} performance.{field} must be a non-empty string")
    if block.get("claim") not in _PERFORMANCE_CLAIMS:
        raise CampaignGateError(f"{owner} performance.claim is not canonical")
    # Same rule as `perf_prompt.PerfFamily.__post_init__`, stated once at each boundary: a PREDICTS
    # claim MUST carry its frozen acceptance contract, and any claim that carries one must carry a
    # mapping. Absent on a DIFFERENTIAL family is canonical, not a hole -- there is no fitted
    # coefficient for an acceptance block to bound.
    acceptance = block.get("acceptance")
    if block["claim"] == "PREDICTS" and not isinstance(acceptance, dict):
        raise CampaignGateError(
            f"{owner} declares a fitted PREDICTS claim and must carry its frozen acceptance mapping")
    if acceptance is not None and not isinstance(acceptance, dict):
        raise CampaignGateError(f"{owner} performance.acceptance must be a frozen mapping")
    for field, required in _PERFORMANCE_NESTED_FIELDS.items():
        nested = block.get(field)
        if not isinstance(nested, dict):
            raise CampaignGateError(f"{owner} performance.{field} must be a mapping")
        absent = sorted(required - nested.keys())
        if absent:
            raise CampaignGateError(
                f"{owner} performance.{field} is non-canonical; missing {absent}")
    if "fired" in block["falsifier"]:
        raise CampaignGateError(
            f"{owner} authors performance.falsifier.fired; only measured evidence may write it")
    traits = block["gate"].get("traits")
    if (not isinstance(traits, list) or not traits
            or any(not isinstance(name, str) or not name for name in traits)
            or len(set(traits)) != len(traits)):
        raise CampaignGateError(f"{owner} performance.gate.traits is not a unique non-empty list")
    unknown = sorted(set(traits) - set(TRAITS))
    if unknown:
        raise CampaignGateError(f"{owner} performance block names unknown traits {unknown}")
    if block["gate"].get("on_missing") != "skip_with_evidence":
        raise CampaignGateError(f"{owner} performance.gate.on_missing is not fail-closed")
    if not isinstance(block["emitter"].get("knobs"), dict):
        raise CampaignGateError(f"{owner} performance.emitter.knobs must be a mapping")
    return block


def _selected_names(value: str | Sequence[str] | None, *, label: str) -> tuple[str, ...]:
    if value is None or value == "all":
        return ()
    values = ([item.strip() for item in value.split(",")] if isinstance(value, str)
              else [str(item).strip() for item in value])
    names = tuple(_simple_name(item, label=label) for item in values if item)
    if not names or len(names) != len(set(names)):
        raise CampaignGateError(f"{label} selection must contain unique names or 'all'")
    return names


def discover_performance_corpus(
        target_experiment: TargetExperiment, *, families: str | Sequence[str] | None = None,
        capsules: str | Sequence[str] | None = None) -> PerformanceCorpus:
    """Discover only generator-proven performance capsules from the descriptor's phase root.

    ``_perf`` is intentionally *not* a public graded root.  The generator records that category in
    the corpus provenance manifest; this consumer derives its absolute path from the descriptor's
    corpus parent and verifies that the ordinary functional root discovery excludes it.
    """
    target = str(target_experiment.target or "").strip()
    if not target:
        raise CampaignGateError("target experiment has no target identity")
    corpus_root = Path(target_experiment.capsule_corpus).resolve().parent
    provenance = corpus_root / "MANIFEST.yaml"
    manifest = _mapping_file(provenance, yaml_file=True)
    records = manifest.get("performance_generation")
    record = records.get(target) if isinstance(records, dict) else None
    if not isinstance(record, dict) or not record:
        raise CampaignGateError(
            f"corpus provenance has no generated performance record for target {target!r}")
    if record.get("errors") != []:
        raise CampaignGateError("generated performance corpus records unresolved generation errors")

    facts = record.get("facts")
    if (not isinstance(facts, dict) or facts.get("target") != target
            or not _is_sha256(facts.get("sha256"))):
        raise CampaignGateError("performance provenance belongs to a foreign or unverified target")
    phase = record.get("phase")
    if (not isinstance(phase, dict) or phase.get("included_in_functional_grade") is not False
            or phase.get("label") != "dev"):
        raise CampaignGateError("performance provenance does not prove a dev-only phase boundary")
    category = _simple_name(phase.get("category"), label="performance phase category")
    if not category.startswith("_"):
        raise CampaignGateError("performance phase category is not excluded from functional discovery")
    phase_root = (corpus_root / category).resolve()
    if corpus_root not in phase_root.parents:
        raise CampaignGateError("performance phase root escapes the descriptor-derived corpus")
    graded = {Path(path).resolve() for path in target_experiment.graded_roots()}
    if phase_root in graded:
        raise CampaignGateError("performance phase root leaks into the functional graded roots")
    if not phase_root.is_dir() or phase_root.is_symlink():
        raise CampaignGateError(f"generated performance phase root is absent: {phase_root}")

    template = record.get("shared_template")
    if (not isinstance(template, dict) or not _is_sha256(template.get("sha256"))
            or not isinstance(template.get("path"), str) or not template["path"]):
        raise CampaignGateError("performance provenance lacks the canonical template digest")
    template_path = Path(template["path"])
    if not template_path.is_absolute():
        # Recorded repo-root-relative. `resolve_grant` additionally honors the repo's documented
        # in-`merlin/` shorthand, so a record written before the tree moved still names a real file.
        # It resolves against two declared roots and nothing else -- no searching -- and returns the
        # repo-root spelling when neither exists, which `_mapping_file` then refuses below. A record
        # whose path does not name a file is a stale corpus, and the gate must say so rather than go
        # looking for a template that happens to be nearby.
        template_path = resolve_grant(template["path"])
    template_doc = _mapping_file(template_path, yaml_file=True)
    if _document_digest(template_doc) != template["sha256"]:
        raise CampaignGateError("generated performance corpus is stale relative to its shared template")

    generated = manifest.get("generated")
    hand_authored = manifest.get("hand_authored")
    if not isinstance(generated, list) or not isinstance(hand_authored, list):
        raise CampaignGateError("corpus provenance lacks generated/hand-authored classification")
    generated_paths = {str(value) for value in generated}
    hand_paths = {str(value) for value in hand_authored}
    if generated_paths & hand_paths:
        raise CampaignGateError("corpus provenance classifies a capsule as both generated and manual")
    phase_generated = {
        value for value in generated_paths
        if Path(value).parts and Path(value).parts[0] == category
    }
    phase_manual = {
        value for value in hand_paths
        if Path(value).parts and Path(value).parts[0] == category
    }
    if phase_manual:
        raise CampaignGateError(
            f"performance phase contains manually classified capsules: {sorted(phase_manual)}")

    found: list[PerformanceCapsule] = []
    for descriptor_path in sorted(phase_root.rglob("capsule.yaml")):
        source_dir = descriptor_path.parent
        relative = source_dir.relative_to(corpus_root)
        if len(relative.parts) != 2 or relative.parts[0] != category:
            raise CampaignGateError(
                f"performance capsule is not a direct member of the phase root: {descriptor_path}")
        relative_text = relative.as_posix()
        if relative_text in hand_paths or relative_text not in generated_paths:
            raise CampaignGateError(
                f"performance capsule is manual or lacks generator provenance: {relative_text}")
        descriptor = _mapping_file(descriptor_path, yaml_file=True)
        name = _simple_name(descriptor.get("name"), label="performance capsule name")
        if source_dir.name != name:
            raise CampaignGateError(
                f"performance capsule directory/name mismatch: {source_dir.name!r} != {name!r}")
        if descriptor.get("label") != phase.get("label"):
            raise CampaignGateError(f"performance capsule {name!r} is not phase-labelled dev")
        if descriptor.get("source_role") != "derived_sweep":
            raise CampaignGateError(f"performance capsule {name!r} is not a generated sweep member")
        performance = _validate_performance_block(
            descriptor.get("performance"), owner=f"performance capsule {name!r}")
        family = _simple_name(performance["family"], label="performance family")
        tree = _exact_tree_record(source_dir)
        found.append(PerformanceCapsule(
            family, name, source_dir.resolve(), relative_text, descriptor,
            str(tree["sha256"]), int(tree["n_files"]), int(tree["n_bytes"])))
    if not found:
        raise CampaignGateError("generated performance corpus contains zero canonical capsules")
    observed_paths = {row.source_relative_path for row in found}
    if observed_paths != phase_generated:
        raise CampaignGateError(
            "generated performance phase paths are stale relative to corpus provenance: "
            f"on_disk={sorted(observed_paths)}, manifest={sorted(phase_generated)}")
    names = [row.capsule for row in found]
    if len(names) != len(set(names)):
        raise CampaignGateError("generated performance corpus repeats a capsule identity")

    counts = record.get("counts")
    by_family = counts.get("by_family") if isinstance(counts, dict) else None
    if not isinstance(by_family, dict):
        raise CampaignGateError("performance provenance lacks structured family counts")
    observed_counts: dict[str, int] = {}
    for row in found:
        observed_counts[row.family] = observed_counts.get(row.family, 0) + 1
    if (counts.get("generated_members") != len(found)
            or counts.get("generated_families") != len(observed_counts)):
        raise CampaignGateError("generated performance corpus is stale relative to its manifest counts")
    for family, count in observed_counts.items():
        family_count = by_family.get(family)
        if (not isinstance(family_count, dict) or family_count.get("written_members") != count
                or family_count.get("admitted_members") != count):
            raise CampaignGateError(
                f"generated performance family {family!r} is partial or stale in provenance")
    claimed_nonempty = {
        str(family) for family, value in by_family.items()
        if isinstance(value, dict) and value.get("written_members")
    }
    if claimed_nonempty != set(observed_counts):
        raise CampaignGateError("performance provenance names generated families absent from the phase root")

    wanted_families = set(_selected_names(families, label="performance family"))
    wanted_capsules = set(_selected_names(capsules, label="performance capsule"))
    known_families, known_capsules = set(observed_counts), set(names)
    if wanted_families - known_families:
        raise CampaignGateError(
            f"unknown generated performance families: {sorted(wanted_families - known_families)}")
    if wanted_capsules - known_capsules:
        raise CampaignGateError(
            f"unknown generated performance capsules: {sorted(wanted_capsules - known_capsules)}")
    selected = tuple(row for row in found
                     if (not wanted_families or row.family in wanted_families)
                     and (not wanted_capsules or row.capsule in wanted_capsules))
    if not selected:
        raise CampaignGateError("performance selection contains zero generated capsules")
    return PerformanceCorpus(
        target, corpus_root, phase_root, provenance, _sha256_bytes(provenance.read_bytes()),
        record, selected)


def _full_ratio(value: object, *, label: str) -> int:
    parts = str(value or "").split("/")
    if len(parts) != 2:
        raise CampaignGateError(f"{label} must state an explicit passed/total ratio")
    try:
        passed, total = (int(x) for x in parts)
    except ValueError as exc:
        raise CampaignGateError(f"{label} has a malformed passed/total ratio: {value!r}") from exc
    if total <= 0:
        raise CampaignGateError(f"{label} must be non-vacuous, not {passed}/{total}")
    if passed != total:
        raise CampaignGateError(f"{label} is incomplete: {passed}/{total}")
    return total


def _validate_score(score: dict, *, label: str, expected: int) -> None:
    n_capsules = score.get("n_capsules")
    n_passed = score.get("n_passed")
    rows = score.get("per_capsule")
    if (not isinstance(n_capsules, int) or isinstance(n_capsules, bool)
            or not isinstance(n_passed, int) or isinstance(n_passed, bool)
            or n_capsules <= 0 or n_passed != n_capsules or n_capsules != expected):
        raise CampaignGateError(
            f"{label} score is not a full non-vacuous {expected}/{expected} grade")
    if score.get("functional_pass") != 1 or score.get("gradeable") is not True:
        raise CampaignGateError(f"{label} score is not gradeable and functionally complete")
    if score.get("integrity_status") != "clean" or score.get("integrity_exempt") is not False:
        raise CampaignGateError(f"{label} score did not pass the integrity gate")
    if not isinstance(rows, list) or len(rows) != expected:
        raise CampaignGateError(f"{label} score has no complete per-capsule evidence")
    names: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not row.get("capsule") or row.get("status") != "pass":
            raise CampaignGateError(f"{label} contains a capsule without a passing verdict")
        name = str(row["capsule"])
        if name in names:
            raise CampaignGateError(f"{label} repeats capsule {name!r}")
        names.add(name)
        tiers = row.get("tiers") or {}
        if tiers.get("L2") != "pass" or tiers.get("L3") != "pass":
            raise CampaignGateError(f"{label} capsule {name!r} did not earn both L2 and L3")


def _validate_clean_run(environment: dict, summary: dict) -> None:
    if environment.get("sandbox") != "bwrap":
        raise CampaignGateError("functional run was not executed in the required bwrap sandbox")
    if environment.get("isolation_violations") != []:
        raise CampaignGateError("functional run lacks a clean isolation-violation audit")
    inputs = environment.get("bundle_input_snapshot")
    snapshot_digest = inputs.get("content_sha256") if isinstance(inputs, dict) else None
    if (not isinstance(inputs, dict) or inputs.get("version") != 2
            or not isinstance(snapshot_digest, str) or len(snapshot_digest) != 64
            or any(c not in "0123456789abcdef" for c in snapshot_digest)
            or not isinstance(inputs.get("n_files"), int) or isinstance(inputs.get("n_files"), bool)
            or inputs["n_files"] <= 0
            or not isinstance(inputs.get("n_bytes"), int) or isinstance(inputs.get("n_bytes"), bool)
            or inputs["n_bytes"] <= 0):
        raise CampaignGateError(
            "functional run lacks a complete immutable bundle-input snapshot v2 record")
    mask = environment.get("golden_mask_selftest")
    if (not isinstance(mask, dict) or mask.get("leaked_answer_files") != []
            or int(mask.get("n_answer_files_masked") or 0) <= 0):
        raise CampaignGateError("functional run did not prove a non-vacuous clean answer mask")
    if (summary.get("converged") is not True
            or summary.get("numeric_all_pass") is not True
            or summary.get("workflow_conformant") is not True):
        raise CampaignGateError(
            "functional QA loop did not converge with numeric and workflow conformance")
    rounds = summary.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise CampaignGateError("functional QA evidence has no completed round")
    for row in rounds:
        if (not isinstance(row, dict) or row.get("answer_access_clean") is not True
                or row.get("audit_hits") != []):
            raise CampaignGateError("functional QA round failed the answer-access audit")
    finalize = summary.get("finalize")
    if (not isinstance(finalize, dict)
            or finalize.get("answer_access_clean") is not True
            or finalize.get("audit_hits") != []
            or finalize.get("regrade_all_pass") is not True):
        raise CampaignGateError("functional finalization did not pass clean regrade and audit gates")


def _has_symlink(root: Path) -> Path | None:
    if root.is_symlink():
        return root
    for path in root.rglob("*"):
        if path.is_symlink():
            return path
    return None


def _digest_excluded_path(root: Path) -> Path | None:
    """Return executable state omitted by the functional harness's historical tree hash."""
    excluded = {"build", "__pycache__", ".git"}
    for path in root.rglob("*"):
        if excluded & set(path.relative_to(root).parts):
            return path
    return None


def _require_read_only_tree(root: Path, *, label: str) -> None:
    if root.is_symlink() or not root.is_dir():
        raise CampaignGateError(f"{label} is absent or linked: {root}")
    for path in (root, *root.rglob("*")):
        if path.is_symlink():
            raise CampaignGateError(f"{label} contains a symlink: {path}")
        if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise CampaignGateError(f"{label} is writable: {path}")


def _bundle_snapshot_content(root: Path) -> dict[str, Any]:
    """Recompute the v2 bundle digest using the functional harness's exact row encoding."""
    rows: list[tuple[str, str, int]] = []
    n_bytes = 0
    marker = root / "snapshot.json"
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise CampaignGateError(f"functional bundle snapshot contains a symlink: {path}")
        if not path.is_file() or path == marker:
            continue
        payload = path.read_bytes()
        rows.append((path.relative_to(root).as_posix(), _sha256_bytes(payload), len(payload)))
        n_bytes += len(payload)
    digest = hashlib.sha256()
    for relative, file_sha, size in rows:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_sha.encode("ascii"))
        digest.update(b"\n")
    return {"content_sha256": digest.hexdigest(), "n_files": len(rows), "n_bytes": n_bytes}


def _safe_relative_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str):
        raise CampaignGateError(f"{label} must be a repository-relative path")
    path = Path(value)
    if path.is_absolute() or not path.parts or any(part == ".." for part in path.parts):
        raise CampaignGateError(f"{label} must be a safe repository-relative path")
    return path


def _validate_functional_input_snapshots(environment: Mapping) -> tuple[dict, dict, Path]:
    """Join the functional input aggregate to its exact frozen host-compiler package."""
    snapshot = environment.get("bundle_input_snapshot")
    if not isinstance(snapshot, Mapping) or snapshot.get("version") != 2:
        raise CampaignGateError(
            "functional run lacks a complete immutable bundle-input snapshot v2 record")
    expected = {key: snapshot.get(key) for key in ("content_sha256", "n_files", "n_bytes")}
    if (not _is_sha256(expected["content_sha256"])
            or isinstance(expected["n_files"], bool) or not isinstance(expected["n_files"], int)
            or expected["n_files"] <= 0
            or isinstance(expected["n_bytes"], bool) or not isinstance(expected["n_bytes"], int)
            or expected["n_bytes"] <= 0):
        raise CampaignGateError(
            "functional run lacks a complete immutable bundle-input snapshot v2 record")
    raw_root = snapshot.get("path")
    if not isinstance(raw_root, str) or not Path(raw_root).is_absolute():
        raise CampaignGateError("functional bundle-input snapshot path is not absolute")
    root = Path(raw_root)
    _require_read_only_tree(root, label="functional bundle-input snapshot")
    marker = _mapping_file(root / "snapshot.json")
    marker_identity = {key: marker.get(key) for key in ("content_sha256", "n_files", "n_bytes")}
    observed = _bundle_snapshot_content(root)
    if marker.get("version") != 2 or marker_identity != expected or observed != expected:
        raise CampaignGateError(
            "functional bundle-input snapshot bytes do not match the recorded v2 identity")

    host = environment.get("model_host_lane_snapshot")
    if not isinstance(host, Mapping):
        raise CampaignGateError("functional run has no frozen model host-lane snapshot")
    if host.get("run_snapshot") != dict(snapshot):
        raise CampaignGateError(
            "functional model host lane does not name the exact run input snapshot")
    package_rel = _safe_relative_path(host.get("package"), label="model host-lane package")
    repo = root / "repo"
    if repo.is_symlink() or not repo.is_dir():
        raise CampaignGateError("functional bundle snapshot has no safe repository payload")
    try:
        resolved_root = root.resolve(strict=True)
        resolved_repo = repo.resolve(strict=True)
        package = (resolved_repo / package_rel).resolve(strict=True)
        package.relative_to(resolved_repo)
    except (OSError, ValueError) as exc:
        raise CampaignGateError(
            "functional model host-lane package escapes or is absent from the run snapshot") from exc
    if resolved_root not in resolved_repo.parents or package.is_symlink() or not package.is_dir():
        raise CampaignGateError(
            "functional model host-lane package is not a real directory under snapshot/repo")
    expected_package = resolved_repo / package_rel
    if package != expected_package or host.get("resolved_package") != str(package):
        raise CampaignGateError(
            "functional model host-lane resolved path does not equal snapshot/repo/package")
    _require_read_only_tree(package, label="functional model host-lane package")
    if _digest_excluded_path(package) is not None:
        raise CampaignGateError("functional model host-lane package contains unhashed state")

    required = host.get("required_paths")
    if not isinstance(required, list) or "manifest.yaml" not in required:
        raise CampaignGateError("functional model host lane does not require its manifest")
    required_paths = [_safe_relative_path(value, label="model host-lane required path")
                      for value in required]
    if any(not (package / relative).is_file() or (package / relative).is_symlink()
           for relative in required_paths):
        raise CampaignGateError("functional model host-lane required file is absent or linked")

    from merlin.mining.registry import load_rvv_package
    try:
        loaded = load_rvv_package(package)
    except Exception as exc:
        raise CampaignGateError(f"functional model host-lane package is invalid: {exc}") from exc
    schedule = _safe_relative_path(
        str(loaded.knobs.get("schedule_file", "schedule.mlir")),
        label="model host-lane schedule")
    package_hash = hash_tree(package)
    comparisons = {
        "package_sha256": package_hash.get("sha256"),
        "n_files": package_hash.get("n_files"),
        "target": loaded.name,
        "run_id": loaded.run_id,
        "dtype_strategy": loaded.dtype_strategy,
        "schedule_file": schedule.as_posix(),
    }
    if any(host.get(key) != value for key, value in comparisons.items()):
        raise CampaignGateError(
            "functional model host-lane identity does not match its snapshotted package bytes")
    return dict(snapshot), dict(host), package


def verify_functional_host_lane_snapshot(host: Mapping) -> tuple[dict, dict, Path]:
    """Re-verify a recorded host lane and its owning v2 bundle snapshot for consumers/reports."""
    if not isinstance(host, Mapping) or not isinstance(host.get("run_snapshot"), Mapping):
        raise CampaignGateError("functional model host-lane record omits its run snapshot")
    return _validate_functional_input_snapshots({
        "bundle_input_snapshot": host["run_snapshot"],
        "model_host_lane_snapshot": host,
    })


def frozen_bundle_grant_path(
        manifest_path: Path, manifest_sha256: str, destination: Path, *, label: str) -> Path:
    """Resolve one live path spelling to the immutable bytes its v2 bundle grant mounted there."""
    manifest_path = Path(manifest_path)
    if (manifest_path.is_symlink() or not manifest_path.is_file()
            or not _is_sha256(manifest_sha256)
            or _sha256_bytes(manifest_path.read_bytes()) != manifest_sha256):
        raise CampaignGateError(f"{label} bundle snapshot manifest bytes do not match the handoff")
    marker = _mapping_file(manifest_path)
    if marker.get("version") != 2 or not isinstance(marker.get("grants"), list):
        raise CampaignGateError(f"{label} bundle snapshot has no v2 grant manifest")
    root = manifest_path.parent.resolve(strict=True)
    destination = Path(destination).absolute()
    matches: list[Path] = []
    for row in marker["grants"]:
        if not isinstance(row, Mapping):
            raise CampaignGateError(f"{label} bundle snapshot has a malformed grant row")
        mounted = Path(str(row.get("destination") or ""))
        snapshot_rel = Path(str(row.get("snapshot") or ""))
        if (not mounted.is_absolute() or snapshot_rel.is_absolute()
                or ".." in snapshot_rel.parts):
            raise CampaignGateError(f"{label} bundle snapshot has an unsafe grant row")
        try:
            suffix = destination.relative_to(mounted)
        except ValueError:
            continue
        source = (root / snapshot_rel / suffix).resolve(strict=True)
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise CampaignGateError(f"{label} frozen grant escapes its snapshot") from exc
        matches.append(source)
    if not matches or any(path != matches[0] for path in matches[1:]):
        raise CampaignGateError(
            f"{label} is absent from, or ambiguously mapped by, the frozen bundle grants")
    _require_read_only_tree(matches[0], label=f"frozen {label}")
    return matches[0]


def select_full_model_sentinel(
        record: FunctionalRun, target_experiment: TargetExperiment) -> FullModelSentinel:
    """Derive the smallest declared-L3 public A/H-lane model from the functional input snapshot.

    SELECTION IS BY DECLARED METADATA AND SIZE, NOT BY A PASSED VERDICT, and that is deliberate --
    the sibling spelling of this idea claimed "the smallest ALREADY-PASSED public whole model", which
    it never was and, measured, cannot be. Every real gemmini public functional grade is a 20/20 over
    the A*/B*/C* kernel capsules; not one of them contains a `kind: model` capsule. Intersecting this
    scan with ``record.public_score["per_capsule"]`` would therefore leave zero candidates and turn
    the whole performance campaign into an unconditional refusal -- the same shape of defect as the
    tier predicate that demanded a tier the model path could not emit.

    So the sentinel is not inherited as passed; it is GRADED, here, by
    ``run_perf_bench.run_full_model_admission`` before any perf cell runs. What this function owes the
    reader is the honest description of what it does: pick the cheapest frozen public capsule that
    DECLARES the mixed-lane whole-model shape, so the admission gate has a well-defined subject.
    """
    snapshot_repo = Path(record.bundle_input_snapshot["path"]) / "repo"
    try:
        corpus_relative = Path(target_experiment.capsule_corpus).resolve().relative_to(repo_root())
    except ValueError as exc:
        raise CampaignGateError(
            "target corpus path cannot be mapped into the functional repository snapshot") from exc
    snapshot_parent = snapshot_repo / corpus_relative.parent
    if snapshot_parent.is_symlink() or not snapshot_parent.is_dir():
        raise CampaignGateError("functional input snapshot has no frozen public capsule corpus")
    candidates: list[FullModelSentinel] = []
    for category in sorted(snapshot_parent.iterdir()):
        if (category.is_symlink() or not category.is_dir() or category.name == "hidden"
                or category.name.startswith(("_", "."))):
            continue
        for capsule_dir in sorted(category.iterdir()):
            descriptor_path = capsule_dir / "capsule.yaml"
            if capsule_dir.is_symlink() or not capsule_dir.is_dir() or not descriptor_path.is_file():
                continue
            descriptor = _mapping_file(descriptor_path, yaml_file=True)
            lanes = descriptor.get("lanes")
            required_lanes = lanes.get("require") if isinstance(lanes, Mapping) else None
            tiers = descriptor.get("required_oracle_tiers")
            if (descriptor.get("kind") != "model" or descriptor.get("label") != "public"
                    or not isinstance(required_lanes, list)
                    or not {"on_mesh", "scalar_rvv_lane"}.issubset(required_lanes)
                    or not isinstance(tiers, list) or not {"L2", "L3"}.issubset(tiers)):
                continue
            name = _simple_name(descriptor.get("name"), label="full-model sentinel")
            if name != capsule_dir.name:
                raise CampaignGateError("frozen full-model sentinel directory/name mismatch")
            tree = _exact_tree_record(capsule_dir)
            candidates.append(FullModelSentinel(
                name, capsule_dir.resolve(), descriptor, str(tree["sha256"]),
                int(tree["n_files"]), int(tree["n_bytes"])))
    if not candidates:
        raise CampaignGateError(
            "functional input snapshot has no public L2/L3 model spanning mesh and RVV lanes")
    return min(candidates, key=lambda item: (item.n_bytes, item.capsule))


def inspect_functional_run(run_root: Path, run_id: str, expected_digest: str) -> FunctionalRun:
    """Validate one explicitly named, fully graded Arm-4 functional run.

    No directory search is performed: the caller supplies both the run ID and the whole-submission
    SHA-256, and all independently recorded digests plus the bytes on disk must agree with it.
    """
    if not run_id:
        raise CampaignGateError("an explicit functional run id is required")
    if Path(run_id).name != run_id or run_id in (".", ".."):
        raise CampaignGateError("functional run id must be a simple directory name")
    if (len(expected_digest) != 64 or expected_digest.lower() != expected_digest
            or any(c not in "0123456789abcdef" for c in expected_digest)):
        raise CampaignGateError("functional submission digest must be an explicit lowercase SHA-256")

    arm_root = (Path(run_root) / "merlin_assisted").resolve()
    run_dir = arm_root / run_id
    if run_dir.is_symlink() or not run_dir.is_dir() or arm_root not in run_dir.resolve().parents:
        raise CampaignGateError(f"explicit functional run does not resolve safely: {run_dir}")
    run_dir = run_dir.resolve()
    submission = run_dir / "submission"
    if not submission.is_dir():
        raise CampaignGateError(f"functional submission is absent: {submission}")
    linked = _has_symlink(submission)
    if linked is not None:
        raise CampaignGateError(f"functional submission contains a live symlink: {linked}")
    excluded = _digest_excluded_path(submission)
    if excluded is not None:
        raise CampaignGateError(
            f"functional submission contains a digest-excluded path: {excluded}")

    environment = _mapping_file(run_dir / "environment.yaml", yaml_file=True)
    summary = _mapping_file(run_dir / "qa_loop_summary.yaml", yaml_file=True)
    freeze = _mapping_file(run_dir / "freeze.json")
    manifest = _mapping_file(run_dir / "run_manifest.yaml", yaml_file=True)
    public_score = _mapping_file(run_dir / "grading_public" / "score_capsule.json")
    hidden_score = _mapping_file(run_dir / "grading_hidden" / "score_capsule.json")

    if environment.get("run_id") != run_id or manifest.get("run_id") != run_id:
        raise CampaignGateError("functional evidence names a different run id")
    if not str(environment.get("bundle_id") or "").startswith("merlin_assisted_rtlchecks_"):
        raise CampaignGateError("functional run is not from the Arm-4 RTL-checks bundle")
    _validate_clean_run(environment, summary)
    bundle_snapshot, host_snapshot, host_package = _validate_functional_input_snapshots(environment)
    if (manifest.get("integrity_status") != "clean" or manifest.get("integrity_exempt") is not False
            or manifest.get("gradeable") is not True):
        raise CampaignGateError("functional run did not pass integrity and gradeability gates")
    public = manifest.get("public_dev") or {}
    hidden = manifest.get("hidden") or {}
    if public.get("functional_pass") != 1 or hidden.get("functional_pass") != 1:
        raise CampaignGateError("functional run did not pass both public and hidden grades")
    if public.get("highest_tier") != "L3":
        raise CampaignGateError("functional public run did not reach the required L3 tier")
    n_public = _full_ratio(public.get("passed"), label="public functional grade")
    n_hidden = _full_ratio(hidden.get("passed"), label="hidden functional grade")
    _validate_score(public_score, label="public functional grade", expected=n_public)
    _validate_score(hidden_score, label="hidden functional grade", expected=n_hidden)
    if {str(r["capsule"]) for r in public_score["per_capsule"]} & {
            str(r["capsule"]) for r in hidden_score["per_capsule"]}:
        raise CampaignGateError("public and hidden grades reuse a capsule identity")
    if freeze.get("workspace_mutable_after_freeze") is not False or not freeze.get("frozen_at"):
        raise CampaignGateError("functional freeze did not record an immutable workspace and timestamp")

    observed = hash_tree(submission)["sha256"]
    recorded = {
        "requested": expected_digest,
        "run_manifest": manifest.get("submission_sha256"),
        "freeze": freeze.get("submission_sha256"),
        "freeze_recheck": freeze.get("submission_sha256_recheck"),
        "submission_tree": observed,
    }
    if any(value != expected_digest for value in recorded.values()):
        raise CampaignGateError(
            f"explicit functional submission digest does not match every record: {recorded}")
    return FunctionalRun(
        run_dir=run_dir,
        submission_dir=submission.resolve(),
        run_id=run_id,
        digest=expected_digest,
        public_capsules=n_public,
        hidden_capsules=n_hidden,
        public_score=public_score,
        hidden_score=hidden_score,
        frozen_at=str(freeze.get("frozen_at") or manifest.get("frozen_at") or ""),
        bundle_input_snapshot=bundle_snapshot,
        model_host_lane_snapshot=host_snapshot,
        model_host_package=host_package,
    )


def materialize_perf_workspace(record: FunctionalRun, perf_root: Path) -> Path:
    """Copy the exact functional package into a new performance-only workspace."""
    perf_root = Path(perf_root).resolve()
    workspace = perf_root / "workspace"
    snapshot = workspace / "submission"
    observed = materialize_readonly_tree(record.submission_dir, snapshot)
    if observed != record.digest:
        raise CampaignGateError(
            f"copied performance submission digest {observed} does not match {record.digest}")
    return snapshot


@contextlib.contextmanager
def functional_host_lane(record: FunctionalRun) -> Iterator[None]:
    """Force model grading to resolve the host compiler from the functional run snapshot."""
    root_key = "MERLIN_MODEL_HOST_LANE_SNAPSHOT_ROOT"
    required_key = "MERLIN_MODEL_HOST_LANE_SNAPSHOT_REQUIRED"
    previous = {root_key: os.environ.get(root_key), required_key: os.environ.get(required_key)}
    os.environ[root_key] = str(record.bundle_input_snapshot["path"])
    os.environ[required_key] = "1"
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def materialize_readonly_tree(source: Path, snapshot: Path) -> str:
    """Copy a symlink-free, fully hashed tree once and make the destination read-only."""
    source = Path(source).resolve()
    snapshot = Path(snapshot).resolve()
    if not source.is_dir():
        raise CampaignGateError(f"immutable campaign source is not a directory: {source}")
    if snapshot.exists() or snapshot.is_symlink():
        raise CampaignGateError(f"immutable campaign snapshot already exists: {snapshot}")
    linked = _has_symlink(source)
    if linked is not None:
        raise CampaignGateError(f"immutable campaign source contains a live symlink: {linked}")
    excluded = _digest_excluded_path(source)
    if excluded is not None:
        raise CampaignGateError(f"immutable campaign source contains a digest-excluded path: {excluded}")
    before = hash_tree(source)["sha256"]
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, snapshot, symlinks=False)
    linked = _has_symlink(snapshot)
    if linked is not None:
        raise CampaignGateError(f"immutable campaign snapshot contains a symlink: {linked}")
    observed = hash_tree(snapshot)["sha256"]
    if observed != before:
        raise CampaignGateError(
            f"immutable campaign copy changed during materialization: {before} != {observed}")
    for path in sorted(snapshot.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir():
            path.chmod(0o555)
        elif path.is_file():
            path.chmod(0o555 if path.stat().st_mode & 0o111 else 0o444)
    snapshot.chmod(0o555)
    return str(observed)


def freeze_performance_corpus(corpus: PerformanceCorpus, snapshot_root: Path) -> FrozenPerformanceCorpus:
    """Copy only the selected capsule bytes and seal a structured, content-addressed manifest."""
    snapshot_root = Path(snapshot_root).resolve()
    if snapshot_root.exists() or snapshot_root.is_symlink():
        raise CampaignGateError(f"frozen performance snapshot already exists: {snapshot_root}")
    if _sha256_bytes(corpus.provenance_manifest.read_bytes()) != corpus.provenance_sha256:
        raise CampaignGateError("performance provenance changed between discovery and freeze")
    capsules_root = snapshot_root / "capsules"
    capsules_root.mkdir(parents=True)
    frozen: list[PerformanceCapsule] = []
    manifest_rows: list[dict[str, Any]] = []
    for capsule in corpus.capsules:
        before = _exact_tree_record(capsule.source_dir)
        if before["sha256"] != capsule.source_sha256:
            raise CampaignGateError(
                f"performance capsule changed between discovery and freeze: {capsule.capsule}")
        destination = capsules_root / capsule.source_relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(capsule.source_dir, destination, symlinks=False)
        copied = _exact_tree_record(destination)
        after = _exact_tree_record(capsule.source_dir)
        if copied != before or after != before:
            raise CampaignGateError(
                f"performance capsule bytes changed while freezing: {capsule.capsule}")
        frozen_capsule = PerformanceCapsule(
            capsule.family, capsule.capsule, destination, capsule.source_relative_path,
            capsule.descriptor, str(copied["sha256"]), int(copied["n_files"]),
            int(copied["n_bytes"]))
        frozen.append(frozen_capsule)
        manifest_rows.append({
            "family": capsule.family,
            "capsule": capsule.capsule,
            "source_relative_path": capsule.source_relative_path,
            "snapshot_relative_path": str(destination.relative_to(snapshot_root)),
            "source_sha256": capsule.source_sha256,
            "snapshot_sha256": copied["sha256"],
            "n_files": copied["n_files"],
            "n_bytes": copied["n_bytes"],
            "performance": capsule.descriptor["performance"],
            "performance_sha256": _document_digest(capsule.descriptor["performance"]),
        })
    capsules_record = _exact_tree_record(capsules_root)
    if _sha256_bytes(corpus.provenance_manifest.read_bytes()) != corpus.provenance_sha256:
        raise CampaignGateError("performance provenance changed while freezing capsule bytes")
    manifest = {
        "schema_version": 1,
        "target": corpus.target,
        "source": {
            "corpus_root": str(corpus.corpus_root),
            "phase_root": str(corpus.phase_root),
            "provenance_manifest": str(corpus.provenance_manifest),
            "provenance_sha256": corpus.provenance_sha256,
            "performance_generation_sha256": _document_digest(corpus.performance_generation),
            "shared_template": corpus.performance_generation["shared_template"],
            "facts": corpus.performance_generation["facts"],
            "phase": corpus.performance_generation["phase"],
        },
        "counts": {
            "capsules": len(manifest_rows),
            "families": len({row["family"] for row in manifest_rows}),
            "files": capsules_record["n_files"],
            "bytes": capsules_record["n_bytes"],
        },
        "capsules_sha256": capsules_record["sha256"],
        "capsules": manifest_rows,
    }
    manifest_path = snapshot_root / "performance_corpus_manifest.json"
    manifest_payload = _canonical_json_bytes(manifest)
    with manifest_path.open("xb") as stream:
        stream.write(manifest_payload)
        stream.flush()
        os.fsync(stream.fileno())
    for path in sorted(snapshot_root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_dir():
            path.chmod(0o555)
        elif path.is_file():
            path.chmod(0o555 if path.stat().st_mode & 0o111 else 0o444)
    snapshot_root.chmod(0o555)
    record = FrozenPerformanceCorpus(
        snapshot_root, capsules_root, manifest_path, _sha256_bytes(manifest_payload),
        str(capsules_record["sha256"]), tuple(frozen))
    verify_frozen_performance_corpus(record)
    return record


def load_frozen_performance_corpus(
        root: Path, *, manifest_sha256: str, capsules_sha256: str,
        expected_target: str | None = None) -> FrozenPerformanceCorpus:
    """Reconstruct a stage-frozen corpus without consulting its live source or descriptor siblings."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise CampaignGateError(f"frozen performance corpus is absent or linked: {root}")
    root = root.resolve(strict=True)
    manifest_path = root / "performance_corpus_manifest.json"
    if (manifest_path.is_symlink() or not manifest_path.is_file()
            or not _is_sha256(manifest_sha256)
            or _sha256_bytes(manifest_path.read_bytes()) != manifest_sha256):
        raise CampaignGateError("frozen performance manifest bytes do not match their record")
    if not _is_sha256(capsules_sha256):
        raise CampaignGateError("frozen performance corpus digest is not a lowercase SHA-256")
    manifest = _mapping_file(manifest_path)
    if manifest.get("schema_version") != 1:
        raise CampaignGateError("frozen performance manifest schema is not supported")
    if expected_target is not None and manifest.get("target") != expected_target:
        raise CampaignGateError("frozen performance corpus names a different target")
    if manifest.get("capsules_sha256") != capsules_sha256:
        raise CampaignGateError("frozen performance manifest names a different capsule-tree digest")
    rows = manifest.get("capsules")
    if not isinstance(rows, list) or not rows:
        raise CampaignGateError("frozen performance manifest has an empty capsule set")
    capsules_root = root / "capsules"
    capsules: list[PerformanceCapsule] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise CampaignGateError("frozen performance manifest contains a malformed capsule row")
        family = _simple_name(row.get("family"), label="frozen performance family")
        capsule = _simple_name(row.get("capsule"), label="frozen performance capsule")
        identity = (family, capsule)
        if identity in seen:
            raise CampaignGateError(f"frozen performance manifest repeats capsule {identity}")
        seen.add(identity)
        source_relative = str(row.get("source_relative_path") or "")
        snapshot_relative = Path(str(row.get("snapshot_relative_path") or ""))
        if (not source_relative or snapshot_relative.is_absolute()
                or ".." in snapshot_relative.parts):
            raise CampaignGateError(f"frozen performance capsule {identity} has an unsafe path")
        source_dir = (root / snapshot_relative).resolve(strict=True)
        try:
            source_dir.relative_to(capsules_root.resolve(strict=True))
        except ValueError as exc:
            raise CampaignGateError(
                f"frozen performance capsule {identity} escapes its corpus") from exc
        descriptor = _mapping_file(source_dir / "capsule.yaml", yaml_file=True)
        if descriptor.get("name") != capsule:
            raise CampaignGateError(
                f"frozen performance capsule directory/name mismatch for {identity}")
        performance = _validate_performance_block(
            descriptor.get("performance"), owner=f"frozen performance capsule {capsule!r}")
        if performance.get("family") != family or row.get("performance") != performance:
            raise CampaignGateError(
                f"frozen performance capsule {identity} disagrees with its manifest claim")
        capsules.append(PerformanceCapsule(
            family, capsule, source_dir, source_relative, descriptor,
            str(row.get("snapshot_sha256") or ""), int(row.get("n_files") or 0),
            int(row.get("n_bytes") or 0)))
    record = FrozenPerformanceCorpus(
        root, capsules_root.resolve(strict=True), manifest_path, manifest_sha256,
        capsules_sha256, tuple(capsules))
    verify_frozen_performance_corpus(record)
    return record


def verify_frozen_performance_corpus(corpus: FrozenPerformanceCorpus) -> dict:
    """Re-read every frozen byte and require it to match the structured freeze record."""
    if _sha256_bytes(corpus.manifest_path.read_bytes()) != corpus.manifest_sha256:
        raise CampaignGateError("frozen performance manifest bytes changed")
    manifest = _mapping_file(corpus.manifest_path)
    if manifest.get("schema_version") != 1:
        raise CampaignGateError("frozen performance manifest schema is not supported")
    aggregate = _exact_tree_record(corpus.capsules_root)
    if aggregate["sha256"] != corpus.capsules_sha256:
        raise CampaignGateError("frozen performance capsule bytes changed")
    rows = manifest.get("capsules")
    if not isinstance(rows, list) or len(rows) != len(corpus.capsules) or not rows:
        raise CampaignGateError("frozen performance manifest has an incomplete capsule set")
    by_identity = {(row.family, row.capsule): row for row in corpus.capsules}
    for item in rows:
        if not isinstance(item, dict):
            raise CampaignGateError("frozen performance manifest contains a malformed capsule row")
        identity = (str(item.get("family") or ""), str(item.get("capsule") or ""))
        capsule = by_identity.get(identity)
        if capsule is None:
            raise CampaignGateError(f"frozen performance manifest names an unexpected capsule {identity}")
        observed = _exact_tree_record(capsule.source_dir)
        if (observed["sha256"] != item.get("snapshot_sha256")
                or observed["n_files"] != item.get("n_files")
                or observed["n_bytes"] != item.get("n_bytes")):
            raise CampaignGateError(f"frozen performance capsule digest changed: {identity}")
    return {
        "manifest_sha256": corpus.manifest_sha256,
        "capsules_sha256": corpus.capsules_sha256,
        "capsules": len(corpus.capsules),
        "families": len({row.family for row in corpus.capsules}),
        "verified": True,
    }


def expected_perf_cells(capsules: Sequence[PerformanceCapsule], replicas: int) -> tuple[PerfCellIdentity, ...]:
    """Enumerate the exact L2-screen and L3-certification identities before execution."""
    if isinstance(replicas, bool) or not isinstance(replicas, int) or replicas <= 0:
        raise CampaignGateError("performance replicas must be a positive integer")
    if not capsules:
        raise CampaignGateError("performance campaign has zero selected capsules")
    identities = tuple(
        PerfCellIdentity(capsule.family, capsule.capsule, simulator, f"r{replica:03d}")
        for capsule in capsules
        for replica in range(replicas)
        for simulator in _SIMULATOR_POLICY
    )
    if len(identities) != len(set(identities)):
        raise CampaignGateError("performance campaign repeats an expected cell identity")
    return identities


def write_immutable_json(path: Path, document: object) -> dict[str, Any]:
    """Create one canonical JSON artifact, seal it read-only, and verify its exact bytes."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise CampaignGateError(f"immutable result artifact already exists: {path}")
    payload = _canonical_json_bytes(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)
    record = {"path": path.name, "sha256": _sha256_bytes(payload), "n_bytes": len(payload)}
    verify_immutable_json(path, record["sha256"])
    return record


def seal_campaign_manifest(path: Path, document: Mapping) -> dict[str, Any]:
    """Atomically replace the mutable progress manifest with sealed final bytes plus a sidecar."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise CampaignGateError(f"campaign progress manifest is absent or linked: {path}")
    sidecar = path.with_name("campaign_manifest.digest.json")
    temporary = path.with_name(".campaign_manifest.final.json")
    if sidecar.exists() or sidecar.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise CampaignGateError("campaign finalization paths are not fresh")
    payload = _canonical_json_bytes(document)
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    record = {
        "path": path.name, "sha256": _sha256_bytes(payload), "n_bytes": len(payload),
        "verified": True, "immutable": True,
    }
    if verify_immutable_json(path, record["sha256"]) != dict(document):
        raise CampaignGateError("sealed campaign manifest did not round-trip exactly")
    digest = write_immutable_json(sidecar, record)
    return {**record, "digest_record_sha256": digest["sha256"]}


def verify_immutable_json(path: Path, expected_sha256: str) -> object:
    """Verify both the byte digest and the read-only bit contract before an artifact is consumed."""
    if not _is_sha256(expected_sha256):
        raise CampaignGateError("immutable result digest is not a lowercase SHA-256")
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise CampaignGateError(f"immutable result artifact is absent or linked: {path}")
    if stat.S_IMODE(path.stat().st_mode) & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise CampaignGateError(f"immutable result artifact is writable: {path}")
    payload = path.read_bytes()
    observed = _sha256_bytes(payload)
    if observed != expected_sha256:
        raise CampaignGateError(
            f"immutable result digest mismatch at {path}: {observed} != {expected_sha256}")
    try:
        return json.loads(payload)
    except (UnicodeDecodeError, ValueError) as exc:
        raise CampaignGateError(f"immutable result artifact is not valid JSON: {path}") from exc


def functional_fork(record: FunctionalRun) -> ForkPoint:
    """Convert the public+hidden L2/L3 evidence into the immutable Phase-F fork record."""
    states: list[CapsuleState] = []
    tiers: set[str] = set()
    for score in (record.public_score, record.hidden_score):
        for row in score["per_capsule"]:
            passed = {str(tier): Verdict(str(status), record.digest)
                      for tier, status in (row.get("tiers") or {}).items() if status == "pass"}
            tiers.update(passed)
            states.append(CapsuleState(str(row["capsule"]), record.digest, verdicts=passed))
    if not states or not tiers:
        raise CampaignGateError("cannot fork from an empty functional verdict set")
    return fork_from(states, tier_order=sorted(tiers), digest=record.digest,
                     recorded_at=record.frozen_at or None)


def check_fork(fork: ForkPoint, snapshot: Path):
    """Check the copied compiler's current bytes against every Phase-F invariant."""
    digest = hash_tree(Path(snapshot))["sha256"]
    states = candidate_states(fork, digest=digest)
    return check_invariants(fork, states, provenance=fork.provenance)


def _remove_agent_home_mounts(argv: Sequence[str]) -> list[str]:
    """Remove the agent-launch credential bind from a package-only bwrap prefix.

    ``base_argv`` also serves paid agent launches and therefore binds ``~/.claude``.  A compiled
    submission never needs that state.  Strip both that bind and its nested projects mask, then clear
    the inherited environment so API/provider credentials cannot enter through environment variables.
    """
    home_claude = str(Path(os.path.expanduser("~/.claude")))
    out: list[str] = []

    def is_agent_home_path(value: str) -> bool:
        path = Path(value)
        home = Path(home_claude)
        return path == home or home in path.parents

    i = 0
    while i < len(argv):
        if argv[i] in ("--bind", "--ro-bind", "--bind-try", "--ro-bind-try") and i + 2 < len(argv):
            if is_agent_home_path(argv[i + 2]):
                i += 3
                continue
        if argv[i] == "--tmpfs" and i + 1 < len(argv) and is_agent_home_path(argv[i + 1]):
            i += 2
            continue
        out.append(argv[i])
        i += 1
    if any(is_agent_home_path(value) for value in out if value.startswith("/")):
        raise CampaignGateError("package sandbox still exposes the agent credential directory")
    return out


def package_sandbox_policy(te: TargetExperiment, workspace: Path,
                           package: Path) -> PackageSandboxPolicy:
    """Derive the sandbox and a fresh writable build copy from one sealed candidate."""
    workspace = Path(workspace).resolve()
    package = Path(package).resolve()
    if not workspace.is_dir() or not package.is_dir() or workspace == package:
        raise CampaignGateError("performance sandbox workspace and copied package must be distinct directories")
    linked = _has_symlink(package)
    excluded = _digest_excluded_path(package)
    if linked is not None or excluded is not None:
        raise CampaignGateError(
            f"sealed performance candidate has unsafe state: {linked or excluded}")
    _require_read_only_tree(package, label="sealed performance candidate")
    package_sha256 = str(hash_tree(package)["sha256"])
    execution_package = workspace / "_package_build" / "submission"
    if execution_package.exists() or execution_package.is_symlink():
        raise CampaignGateError("performance build copy already exists in the cell workspace")
    execution_package.parent.mkdir(parents=True)
    shutil.copytree(package, execution_package, symlinks=False)
    if hash_tree(execution_package)["sha256"] != package_sha256:
        raise CampaignGateError("performance build copy differs from the sealed candidate")
    for path in (execution_package, *execution_package.rglob("*")):
        if path.is_symlink():
            raise CampaignGateError(f"performance build copy contains a symlink: {path}")
        path.chmod(path.stat().st_mode | stat.S_IWUSR)
    # Empty bundle is intentional: the package sees its input files and the derived toolchain, not the
    # experiment repository.  The policy-only escape avoids requiring an agent-input snapshot when no
    # bundle grants exist and does not expose any live input itself.
    argv = BW.base_argv(workspace, {}, _policy_test_live_inputs=True)
    argv = _remove_agent_home_mounts(argv)
    argv += ["--clearenv", "--setenv", "HOME", "/tmp",
             "--setenv", "XDG_RUNTIME_DIR", "/tmp/.xdg"]
    argv += TC.toolchain_binds(te)
    argv = _remove_agent_home_mounts(argv)
    surfaces = answer_surfaces(te)
    argv = BW.apply_answer_masks(argv, surfaces)
    argv += ["--ro-bind", str(package), str(package)]
    gaps = tuple(str(surface.path) for surface in BW.coverage_gap(argv, surfaces))
    if gaps:
        raise CampaignGateError(f"package sandbox exposes derived answer surfaces: {gaps}")
    probes = tuple(TC.required_tool_probes(te))
    if not probes:
        raise CampaignGateError("package sandbox derives zero required tool probes")
    return PackageSandboxPolicy(
        tuple(argv), gaps, probes, workspace, package, execution_package,
        package_sha256, te)


def run_tool_probes(policy: PackageSandboxPolicy, *, timeout: int = 60) -> list[dict]:
    """Run every descriptor-derived tool probe in the exact package sandbox."""
    rows: list[dict] = []
    for probe in policy.required_tools:
        proc = subprocess.run(
            [*policy.argv, "bash", "-c", TC.sandbox_env(policy.target_experiment, policy.workspace)
             + probe.cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        row = {"label": probe.label, "returncode": proc.returncode,
               "stdout": (proc.stdout or "")[-400:], "stderr": (proc.stderr or "")[-400:]}
        rows.append(row)
        if proc.returncode != 0:
            # A refusing gate must carry the evidence it refused on; a bare rc points at nothing.
            raise CampaignGateError(
                f"required sandbox tool probe {probe.label!r} failed with rc={proc.returncode}"
                f"; command={probe.cmd!r}; stderr={row['stderr'].strip()!r}"
                f"; stdout={row['stdout'].strip()!r}")
    if len(rows) != len(policy.required_tools):
        raise CampaignGateError("not every required package-sandbox tool probe ran")
    return rows


def _inside(path: Path, root: Path) -> bool:
    path = path.resolve()
    root = root.resolve()
    return path == root or root in path.parents


@contextlib.contextmanager
def boxed_entrypoints(policy: PackageSandboxPolicy) -> Iterator[None]:
    """Route oot_runner's untrusted package execution through ``policy`` for this serial campaign."""
    original_entrypoint = oot_runner.run_entrypoint
    original_build = oot_runner.build_package
    original_subprocess = oot_runner.subprocess
    original_usable_cmake = oot_runner._usable_cmake
    host_run = subprocess.run

    def execution_package(pkg):
        if pkg.directory.resolve() != policy.package:
            raise CampaignGateError(
                "capsule runner attempted to build or execute a package outside the sealed candidate")
        if hash_tree(policy.package)["sha256"] != policy.package_sha256:
            raise CampaignGateError("sealed performance candidate changed before package execution")
        try:
            tool_relative = pkg.tool.resolve().relative_to(policy.package)
        except ValueError as exc:
            raise CampaignGateError("candidate tool escapes the sealed package") from exc
        return oot_runner.Package(
            policy.execution_package, pkg.manifest, policy.execution_package / tool_relative)

    class BoxedBuildSubprocess:
        @staticmethod
        def run(argv, *, cwd=None, env=None, capture_output=True, text=True, timeout=1800,
                **kwargs):
            if kwargs:
                raise CampaignGateError(
                    f"untrusted package build requested unsupported subprocess options: "
                    f"{sorted(kwargs)}")
            if Path(str(cwd or "")).resolve() != policy.execution_package:
                raise CampaignGateError("untrusted package build escaped its per-cell build copy")
            if not isinstance(argv, (list, tuple)) or not argv or any(
                    not isinstance(value, str) or not value for value in argv):
                raise CampaignGateError("untrusted package build command is not an explicit argv")
            shell = TC.sandbox_env(policy.target_experiment, policy.workspace) + 'exec "$@"'
            return host_run(
                [*policy.argv, "--chdir", str(policy.execution_package),
                 "bash", "-c", shell, "perf-build", *argv],
                capture_output=capture_output, text=text, timeout=timeout)

    def build_boxed(pkg, *, timeout: int = 1800) -> None:
        built = execution_package(pkg)
        from merlin.targetgen.contract import toolchain as mlir_toolchain
        trusted_build_env = {
            "CM": "cmake",
            "CMAKE": "cmake",
            "MLIR_DIR": str(mlir_toolchain.mlir_cmake_dir()),
            "LLVM_DIR": str(mlir_toolchain.mlir_install() / "lib" / "cmake" / "llvm"),
        }
        previous_env = {key: os.environ.get(key) for key in trusted_build_env}
        os.environ.update(trusted_build_env)
        oot_runner.subprocess = BoxedBuildSubprocess
        oot_runner._usable_cmake = lambda: "cmake"
        try:
            original_build(built, timeout=timeout)
        finally:
            oot_runner.subprocess = original_subprocess
            oot_runner._usable_cmake = original_usable_cmake
            for key, value in previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def run_boxed(pkg, name: str, input_mlir: Path, output_json: Path | None = None,
                  *, timeout: int = 600) -> subprocess.CompletedProcess:
        built = execution_package(pkg)
        input_mlir = Path(input_mlir).resolve()
        output_json = Path(output_json).resolve() if output_json is not None else None
        if not _inside(input_mlir, policy.workspace):
            raise CampaignGateError("untrusted package input is outside the performance workspace")
        if output_json is not None and not _inside(output_json, policy.workspace):
            raise CampaignGateError("untrusted package output is outside the performance workspace")
        argv = oot_runner._resolve_argv(built, name, input_mlir, output_json)
        if oot_runner._needs_interpreter(built, argv):
            argv = [sys.executable, *argv]
        shell = TC.sandbox_env(policy.target_experiment, policy.workspace) + 'exec "$@"'
        return subprocess.run(
            [*policy.argv, "--chdir", str(policy.execution_package), "bash", "-c", shell,
             "perf-package", *argv],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    oot_runner.build_package = build_boxed
    oot_runner.run_entrypoint = run_boxed
    try:
        yield
    finally:
        oot_runner.run_entrypoint = original_entrypoint
        oot_runner.build_package = original_build
        oot_runner.subprocess = original_subprocess
        oot_runner._usable_cmake = original_usable_cmake


def _result_identity(row: Mapping) -> PerfCellIdentity:
    raw = row.get("identity")
    if not isinstance(raw, Mapping):
        raise CampaignGateError("performance result omits its exact identity mapping")
    return PerfCellIdentity(
        str(raw.get("family") or ""), str(raw.get("capsule") or ""),
        str(raw.get("simulator") or ""), str(raw.get("replicate") or ""))


def completion_report(results: Sequence[Mapping],
                      expected: Sequence[PerfCellIdentity]) -> dict:
    """Count exact replicas while keeping an L2 screen distinct from citable L3 evidence."""
    expected_cells = tuple(expected)
    if not expected_cells:
        raise CampaignGateError("performance campaign has zero expected Arm-4 cells")
    if any(not isinstance(identity, PerfCellIdentity) for identity in expected_cells):
        raise CampaignGateError("every expected performance cell must have an exact identity")
    if len(expected_cells) != len(set(expected_cells)):
        raise CampaignGateError("performance campaign repeats an expected cell identity")
    expected_set = set(expected_cells)
    observed: dict[PerfCellIdentity, Mapping] = {}
    for row in results:
        if not isinstance(row, Mapping):
            raise CampaignGateError("performance result row is not a mapping")
        identity = _result_identity(row)
        if identity not in expected_set:
            raise CampaignGateError(f"performance result has unexpected identity {identity.as_dict()}")
        if identity in observed:
            raise CampaignGateError(f"performance result repeats identity {identity.as_dict()}")
        policy = _SIMULATOR_POLICY.get(identity.simulator)
        if policy is None:
            raise CampaignGateError(f"performance result names unsupported simulator {identity.simulator!r}")
        if any(row.get(key) != value for key, value in policy.items()):
            raise CampaignGateError(
                f"performance result misclassifies {identity.simulator!r} tier/purpose/citability")
        observed[identity] = row

    screen_expected = sum(identity.simulator == "spike" for identity in expected_cells)
    citable_expected = sum(identity.simulator == "verilator" for identity in expected_cells)
    screen_passed = citable_passed = citable_measured = correct = failed = 0
    for identity, row in observed.items():
        # A tier-local pass is not enough: run_capsule may report an overall or numeric failure while
        # retaining a passing tier record.  The completion seal and the independent report verifier use
        # this exact evidence predicate so a measurement-smoke cannot print GO for rows the reporter
        # later refuses.
        is_correct = (row.get("correct") is True and row.get("tier_status") == "pass"
                      and row.get("grade_status") == "pass"
                      and row.get("numeric_status") == "pass"
                      and not row.get("error") and not row.get("failure"))
        correct += int(is_correct)
        if identity.simulator == "spike":
            # Spike is only a correctness screen.  Carrying its cycles into a result makes them
            # accidentally citable, so the row is incomplete unless the field is explicitly null.
            screen_ok = is_correct and row.get("cycles") is None
            screen_passed += int(screen_ok)
            failed += int(not screen_ok)
            continue
        cycles = row.get("cycles")
        # Simulator cycles are a discrete counter, not an estimated duration.  Accept only a positive
        # integer; floats were previously sealed by the campaign and then produced an unreportable run.
        has_cycles = isinstance(cycles, int) and not isinstance(cycles, bool) and cycles > 0
        citable_measured += int(has_cycles)
        citable_passed += int(is_correct and has_cycles)
        failed += int(not (is_correct and has_cycles))
    missing = len(expected_cells) - len(observed)
    return {
        "expected": len(expected_cells),
        "reported": len(observed),
        "correct": correct,
        "failed": failed,
        "missing": missing,
        "screen_expected": screen_expected,
        "screen_passed": screen_passed,
        "citable_expected": citable_expected,
        "citable_measured": citable_measured,
        "citable_passed": citable_passed,
        "complete": (missing == 0 and failed == 0 and screen_expected > 0
                     and citable_expected > 0 and screen_passed == screen_expected
                     and citable_passed == citable_expected),
    }


def completion_counts(results: Sequence[Mapping],
                      expected: Sequence[PerfCellIdentity]) -> dict:
    """Require every exact L2 screen and correct positive-cycle L3 certification result."""
    counts = completion_report(results, expected)
    if not counts["complete"]:
        raise CampaignGateError(
            f"Arm-4 performance reported {counts['reported']} of {counts['expected']} expected cells; "
            f"{counts['failed']} reported cell(s) failed correctness or positive-cycle measurement")
    return counts
