"""Full candidate, functional baseline and frozen corpus admission for paired measurement."""

from __future__ import annotations

import json
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from merlin.benchharness import hash_tree

from . import campaign as PC
from . import candidate_verification as VERIFY
from . import corpus as P2_CORPUS
from . import gsim_gate as GATE
from . import paired_measurement as PM
from . import revealed_corpus as RC


@dataclass(frozen=True)
class LoadedMember:
    family: str
    capsule: str
    source_dir: Path
    descriptor: dict[str, Any]
    source_sha256: str


@dataclass(frozen=True)
class LoadedCorpus:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    capsules_sha256: str
    capsules: tuple[LoadedMember, ...]
    format: str


def _assert_immutable_tree(path: Path, digest: str, *, label: str) -> Path:
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise PC.CampaignGateError(f"{label} is absent or linked: {path}")
    path = path.resolve(strict=True)
    for member in (path, *path.rglob("*")):
        if member.is_symlink():
            raise PC.CampaignGateError(f"{label} contains a symlink: {member}")
        if member.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise PC.CampaignGateError(f"{label} is writable: {member}")
    observed = str(hash_tree(path)["sha256"])
    if observed != digest:
        raise PC.CampaignGateError(f"{label} digest changed: {digest} != {observed}")
    return path


def _coordinates(corpus: PM.FrozenCorpus) -> set[tuple[str, str, str]]:
    return {
        (str(member.family), str(member.capsule), replicate)
        for member in corpus.capsules
        for replicate in PM.REPLICATES
    }


def _handoff_coordinates(handoff: PM.Handoff) -> set[tuple[str, str, str]]:
    if any(not isinstance(cell, Mapping) for cell in handoff.expected_cells):
        raise PC.CampaignGateError("candidate handoff has malformed expected cells")
    return {
        (str(cell.get("family") or ""), str(cell.get("capsule") or ""), str(cell.get("replicate") or ""))
        for cell in handoff.expected_cells
    }


def _validate_handoff(
    functional: PC.FunctionalRun, handoff: PM.Handoff, corpus: PM.FrozenCorpus, phase: str, target_experiment: object
) -> None:
    if phase not in PM.PHASES:
        raise PC.CampaignGateError(f"phase must be one of {PM.PHASES}")
    if handoff.functional_run_id != functional.run_id or handoff.functional_submission_sha256 != functional.digest:
        raise PC.CampaignGateError("candidate handoff names a different functional run")
    descriptor = Path(getattr(target_experiment, "path")).resolve(strict=True)
    if (
        Path(handoff.target_descriptor).resolve(strict=True) != descriptor
        or PM.sha256_file(descriptor) != handoff.target_descriptor_sha256
    ):
        raise PC.CampaignGateError("candidate handoff target descriptor differs")
    # COUNTED FROM THE DECLARATION, NOT WRITTEN DOWN. This read `!= 3` while REPLICATES holds two
    # identities and the frozen acceptance declares exact_count 2, so every handoff refused and the
    # formal paired campaign could not start at all. The literal was left behind when the replicate
    # count was cut from three to two; the error message on the next line was updated and this was
    # not, which is precisely why it reads as correct.
    if handoff.replicates != len(PM.REPLICATES) or tuple(handoff.formal_replicate_identities) != PM.REPLICATES:
        raise PC.CampaignGateError(
            f"measurement requires exactly the {len(PM.REPLICATES)} identities {list(PM.REPLICATES)}"
        )
    if handoff.candidate_initial_sha256 != functional.digest:
        raise PC.CampaignGateError("candidate was not forked from the functional baseline")
    if Path(handoff.candidate_path).resolve() == Path(handoff.functional_base_path).resolve():
        raise PC.CampaignGateError("candidate and baseline are the same path")
    same = (
        corpus.root.resolve() == Path(handoff.corpus_root).resolve()
        and corpus.manifest_sha256 == handoff.corpus_manifest_sha256
        and corpus.capsules_sha256 == handoff.corpus_sha256
    )
    if phase == "tuning" and (not same or _coordinates(corpus) != _handoff_coordinates(handoff)):
        raise PC.CampaignGateError("tuning phase differs from the authoring corpus/cells")
    if phase == "held_out" and (same or corpus.capsules_sha256 == handoff.corpus_sha256):
        raise PC.CampaignGateError("held_out bytes must not be the candidate-visible tuning corpus")


def holdout_tree_record(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.name == "holdout_manifest.json":
            continue
        if path.is_symlink():
            raise PC.CampaignGateError(f"held-out corpus contains a symlink: {path}")
        if path.is_file():
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": PM.sha256_file(path),
                }
            )
    payload = (json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()
    return {"files": rows, "sha256": PM.sha256_bytes(payload)}


def load_holdout_corpus(
    root: Path, manifest_path: Path, *, manifest_sha256: str, capsules_sha256: str, expected_target: str
) -> LoadedCorpus:
    root = Path(root).resolve(strict=True)
    manifest_path = Path(manifest_path).resolve(strict=True)
    if manifest_path.parent != root or manifest_path.name != "holdout_manifest.json":
        raise PC.CampaignGateError("held-out manifest must be the explicit corpus-root manifest")
    try:
        verified = RC.load_revealed_members(
            manifest_path,
            expected_manifest_sha256=manifest_sha256,
            expected_corpus_sha256=capsules_sha256,
            expected_target=expected_target,
        )
    except RC.QualificationError as exc:
        raise PC.CampaignGateError(f"held-out manifest/corpus is invalid: {exc}") from exc
    members = []
    for row in verified:
        descriptor = yaml.safe_load(row.manifest.read_text(encoding="utf-8"))
        members.append(
            LoadedMember(row.family, row.name, row.source_dir, descriptor, str(hash_tree(row.source_dir)["sha256"]))
        )
    return LoadedCorpus(root, manifest_path, manifest_sha256, capsules_sha256, tuple(members), "holdout_reveal_v2")


def _verify_measurement_corpus(corpus: PM.FrozenCorpus, phase: str) -> None:
    if phase == "tuning":
        P2_CORPUS.verify_frozen_performance_corpus(corpus)
        return
    tree = holdout_tree_record(corpus.root)
    if PM.sha256_file(corpus.manifest_path) != corpus.manifest_sha256 or tree["sha256"] != corpus.capsules_sha256:
        raise PC.CampaignGateError("held-out corpus changed during measurement")
    for member in corpus.capsules:
        if str(hash_tree(member.source_dir)["sha256"]) != member.source_sha256:
            raise PC.CampaignGateError(f"held-out member changed: {member.family}/{member.capsule}")


def load_paired_inputs(
    record_path: Path,
    functional_run_id: str,
    functional_submission_sha256: str,
    target_experiment: object,
    *,
    corpus_root: Path,
    corpus_manifest_sha256: str,
    corpus_capsules_sha256: str,
    phase: str,
    corpus_manifest: Path,
    gsim_certificate: Path,
    gsim_certificate_sha256: str,
    waive_functional_gate: tuple[str, ...] = (),
    functional_runs_root: Path,
) -> PM.PairedInputs:
    """Load a sealed candidate plus an explicit tuning or host-only held-out corpus."""
    functional = PC.inspect_functional_run(
        functional_runs_root,
        functional_run_id,
        functional_submission_sha256,
        waive=frozenset(waive_functional_gate),
    )
    try:
        handoff = VERIFY.verify_candidate_handoff(
            record_path, verify_authoring_tools=False, target_experiment=target_experiment
        )
        if phase == "tuning":
            expected_manifest = Path(corpus_root) / "performance_corpus_manifest.json"
            if Path(corpus_manifest).resolve(strict=True) != expected_manifest.resolve(strict=True):
                raise PC.CampaignGateError("tuning manifest is not under the explicit corpus root")
            corpus = P2_CORPUS.load_frozen_performance_corpus(
                corpus_root,
                manifest_sha256=corpus_manifest_sha256,
                capsules_sha256=corpus_capsules_sha256,
                expected_target=getattr(target_experiment, "target"),
            )
        else:
            corpus = load_holdout_corpus(
                corpus_root,
                corpus_manifest,
                manifest_sha256=corpus_manifest_sha256,
                capsules_sha256=corpus_capsules_sha256,
                expected_target=getattr(target_experiment, "target"),
            )
        certificate = GATE.load_certificate(gsim_certificate, expected_sha256=gsim_certificate_sha256)
    except Exception as exc:
        raise PC.CampaignGateError(f"paired inputs are not consumable: {exc}") from exc
    _validate_handoff(functional, handoff, corpus, phase, target_experiment)
    if certificate.target != getattr(target_experiment, "target"):
        raise PC.CampaignGateError("GSIM certificate names a different target")
    baseline = _assert_immutable_tree(handoff.functional_base_path, functional.digest, label="functional baseline")
    candidate = _assert_immutable_tree(handoff.candidate_path, handoff.candidate_sha256, label="sealed candidate")
    return PM.PairedInputs(
        functional,
        handoff,
        corpus,
        phase,
        baseline,
        functional.digest,
        candidate,
        handoff.candidate_sha256,
        certificate,
    )


def identity_guard(inputs: PM.PairedInputs) -> dict[str, Any]:
    _verify_measurement_corpus(inputs.corpus, inputs.phase)
    GATE.load_certificate(inputs.gsim_certificate.path, expected_sha256=inputs.gsim_certificate.sha256)
    return {
        "baseline_sha256": str(hash_tree(inputs.baseline)["sha256"]),
        "candidate_sha256": str(hash_tree(inputs.candidate)["sha256"]),
        "corpus_manifest_sha256": PM.sha256_file(inputs.corpus.manifest_path),
        "corpus_capsules_sha256": inputs.corpus.capsules_sha256,
        "candidate_record_sha256": inputs.handoff.record_sha256,
        "gsim_certificate_sha256": inputs.gsim_certificate.sha256,
    }
