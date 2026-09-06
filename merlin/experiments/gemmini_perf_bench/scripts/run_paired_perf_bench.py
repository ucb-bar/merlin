#!/usr/bin/env python3
"""Candidate-aware paired Arm-4 performance measurement.

An explicit frozen corpus is labelled ``tuning`` or ``held_out``.  A pure preflight declares an
adjacent/interleaved baseline-candidate schedule before execution. Spike is a fast correctness-only
semantic screen and GSIM is the sole RTL execution and timing backend. Verilator is used only while
producing the prerequisite GSIM equivalence certificates, never in this campaign. Raw executions are
content addressed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import stat
import threading
import traceback
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

import _pbcommon as PB
import perf_agent_stage as PAS
import perf_campaign as PC
import perf_gsim_gate as GATE
import heldout_gsim_qualification as HQUAL
import produce_gsim_certificate as CERTPROD
import run_perf_bench as FIXED
from merlin.benchharness import hash_tree, runs_root as _runs_root
from merlin.common.artifacts import cache_dir
from merlin.targetgen.contract import compile as OOT
from merlin.targetgen.target_experiment import load_target_experiment

ARMS = ("baseline", "candidate")
#: Two, because the engine is deterministic and the third replicate re-derives a number the second
#: already agreed on -- verified over 392 repeated measurements of identical bytes with zero
#: disagreement. Two and not one: one leaves the replicate dispersion UNDETERMINABLE, and assuming
#: it zero on a deterministic simulator is the assumption the shipped contracts refuse.
REPLICATES = ("r000", "r001")
PRIMARY_SIMULATOR = "gsim"
CORRECTNESS_SIMULATOR = "spike"
SIMULATORS = (CORRECTNESS_SIMULATOR, PRIMARY_SIMULATOR)
PHASES = ("tuning", "held_out")
PHYSICAL_BYTE_UNIT = "BYTES"
_FUNCTIONAL_RUNS = _runs_root(PB.TARGET, "capsule-bench")
_DESCRIPTOR = (PB.REPO / "merlin/experiments/capsule_bench/targets" / PB.TARGET
               / "target_experiment.yaml")


class Handoff(Protocol):
    record_sha256: str
    candidate_path: Path
    candidate_sha256: str
    candidate_initial_sha256: str
    functional_run_id: str
    functional_submission_sha256: str
    functional_base_path: Path
    target_descriptor: Path
    target_descriptor_sha256: str
    corpus_root: Path
    corpus_manifest_sha256: str
    corpus_sha256: str
    replicates: int
    formal_replicate_identities: tuple[str, ...]
    expected_cells: tuple[dict[str, str], ...]


class FrozenMember(Protocol):
    family: str
    capsule: str
    source_dir: Path
    descriptor: dict[str, Any]
    source_sha256: str


class FrozenCorpus(Protocol):
    root: Path
    manifest_path: Path
    manifest_sha256: str
    capsules_sha256: str
    capsules: tuple[FrozenMember, ...]


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


@dataclass(frozen=True)
class PairedInputs:
    functional: PC.FunctionalRun
    handoff: Handoff
    corpus: FrozenCorpus
    phase: str
    baseline: Path
    baseline_sha256: str
    candidate: Path
    candidate_sha256: str
    gsim_certificate: GATE.CertificateRecord


@dataclass(frozen=True)
class ExecutionSpec:
    execution_index: int
    pair_index: int
    pair_id: str
    phase: str
    arm: str
    family: str
    capsule: str
    replicate: str
    package: Path
    package_sha256: str
    member: FrozenMember
    workload: dict[str, Any]
    gsim_decision: GATE.EvaluationDecision
    gsim_certificate: GATE.CertificateRecord

    @property
    def simulators(self) -> tuple[str, ...]:
        return SIMULATORS

    def as_dict(self) -> dict[str, Any]:
        return {
            "execution_index": self.execution_index, "pair_index": self.pair_index,
            "pair_id": self.pair_id, "phase": self.phase, "arm": self.arm,
            "family": self.family, "capsule": self.capsule, "replicate": self.replicate,
            "package_sha256": self.package_sha256, "simulators": list(self.simulators),
            "spike_role": "correctness_only",
            "gsim_role": "primary_cycle_accurate_elaborated_rtl",
            "rtl_execution_backends": ["gsim"],
            "timing_authority": "gsim",
            "workload_sha256": self.gsim_decision.workload_sha256,
            "gsim_decision": self.gsim_decision.to_dict(),
        }


@dataclass(frozen=True, order=True)
class ResultIdentity:
    phase: str
    arm: str
    family: str
    capsule: str
    simulator: str
    replicate: str

    @property
    def label(self) -> str:
        return "/".join((self.phase, self.arm, self.family, self.capsule,
                         self.simulator, self.replicate))


@dataclass(frozen=True)
class MeasurementPlan:
    phase: str
    schedule: tuple[ExecutionSpec, ...]
    expected: tuple[ResultIdentity, ...]
    declaration: dict[str, Any]
    declaration_sha256: str


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n").encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _is_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64 and value.lower() == value
            and all(char in "0123456789abcdef" for char in value))


def _simple_component(value: str, *, label: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    if (not value or Path(value).name != value or value in (".", "..")
            or any(char not in allowed for char in value)):
        raise PC.CampaignGateError(f"{label} is not a safe path component: {value!r}")
    return value


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


def _coordinates(corpus: FrozenCorpus) -> set[tuple[str, str, str]]:
    return {(str(member.family), str(member.capsule), replicate)
            for member in corpus.capsules for replicate in REPLICATES}


def _handoff_coordinates(handoff: Handoff) -> set[tuple[str, str, str]]:
    if any(not isinstance(cell, Mapping) for cell in handoff.expected_cells):
        raise PC.CampaignGateError("candidate handoff has malformed expected cells")
    return {(str(cell.get("family") or ""), str(cell.get("capsule") or ""),
             str(cell.get("replicate") or "")) for cell in handoff.expected_cells}


def _validate_handoff(functional: PC.FunctionalRun, handoff: Handoff, corpus: FrozenCorpus,
                      phase: str, target_experiment: object) -> None:
    if phase not in PHASES:
        raise PC.CampaignGateError(f"phase must be one of {PHASES}")
    if (handoff.functional_run_id != functional.run_id
            or handoff.functional_submission_sha256 != functional.digest):
        raise PC.CampaignGateError("candidate handoff names a different functional run")
    descriptor = Path(getattr(target_experiment, "path")).resolve(strict=True)
    if (Path(handoff.target_descriptor).resolve(strict=True) != descriptor
            or _sha256_file(descriptor) != handoff.target_descriptor_sha256):
        raise PC.CampaignGateError("candidate handoff target descriptor differs")
    # COUNTED FROM THE DECLARATION, NOT WRITTEN DOWN. This read `!= 3` while REPLICATES holds two
    # identities and the frozen acceptance declares exact_count 2, so every handoff refused and the
    # formal paired campaign could not start at all. The literal was left behind when the replicate
    # count was cut from three to two; the error message on the next line was updated and this was
    # not, which is precisely why it reads as correct.
    if (handoff.replicates != len(REPLICATES)
            or tuple(handoff.formal_replicate_identities) != REPLICATES):
        raise PC.CampaignGateError(
            f"measurement requires exactly the {len(REPLICATES)} identities {list(REPLICATES)}")
    if handoff.candidate_initial_sha256 != functional.digest:
        raise PC.CampaignGateError("candidate was not forked from the functional baseline")
    if Path(handoff.candidate_path).resolve() == Path(handoff.functional_base_path).resolve():
        raise PC.CampaignGateError("candidate and baseline are the same path")
    same = (corpus.root.resolve() == Path(handoff.corpus_root).resolve()
            and corpus.manifest_sha256 == handoff.corpus_manifest_sha256
            and corpus.capsules_sha256 == handoff.corpus_sha256)
    if phase == "tuning" and (not same or _coordinates(corpus) != _handoff_coordinates(handoff)):
        raise PC.CampaignGateError("tuning phase differs from the authoring corpus/cells")
    if phase == "held_out" and (same or corpus.capsules_sha256 == handoff.corpus_sha256):
        raise PC.CampaignGateError("held_out bytes must not be the candidate-visible tuning corpus")


def _holdout_tree_record(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.name == "holdout_manifest.json":
            continue
        if path.is_symlink():
            raise PC.CampaignGateError(f"held-out corpus contains a symlink: {path}")
        if path.is_file():
            rows.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                         "sha256": _sha256_file(path)})
    payload = (json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
               + "\n").encode()
    return {"files": rows, "sha256": _sha256_bytes(payload)}


def _load_holdout_corpus(root: Path, manifest_path: Path, *, manifest_sha256: str,
                         capsules_sha256: str, expected_target: str) -> LoadedCorpus:
    root = Path(root).resolve(strict=True)
    manifest_path = Path(manifest_path).resolve(strict=True)
    if manifest_path.parent != root or manifest_path.name != "holdout_manifest.json":
        raise PC.CampaignGateError("held-out manifest must be the explicit corpus-root manifest")
    try:
        verified = HQUAL.load_revealed_members(
            manifest_path, expected_manifest_sha256=manifest_sha256,
            expected_corpus_sha256=capsules_sha256, expected_target=expected_target)
    except HQUAL.QualificationError as exc:
        raise PC.CampaignGateError(f"held-out manifest/corpus is invalid: {exc}") from exc
    members = []
    for row in verified:
        descriptor = yaml.safe_load(row.manifest.read_text(encoding="utf-8"))
        members.append(LoadedMember(row.family, row.name, row.source_dir, descriptor,
                                    str(hash_tree(row.source_dir)["sha256"])))
    return LoadedCorpus(root, manifest_path, manifest_sha256, capsules_sha256,
                        tuple(members), "holdout_reveal_v2")


def _verify_measurement_corpus(corpus: FrozenCorpus, phase: str) -> None:
    if phase == "tuning":
        PAS.verify_frozen_performance_corpus(corpus)
        return
    tree = _holdout_tree_record(corpus.root)
    if (_sha256_file(corpus.manifest_path) != corpus.manifest_sha256
            or tree["sha256"] != corpus.capsules_sha256):
        raise PC.CampaignGateError("held-out corpus changed during measurement")
    for member in corpus.capsules:
        if str(hash_tree(member.source_dir)["sha256"]) != member.source_sha256:
            raise PC.CampaignGateError(f"held-out member changed: {member.family}/{member.capsule}")


def load_paired_inputs(record_path: Path, functional_run_id: str,
                       functional_submission_sha256: str, target_experiment: object, *,
                       corpus_root: Path, corpus_manifest_sha256: str,
                       corpus_capsules_sha256: str, phase: str,
                       corpus_manifest: Path,
                       gsim_certificate: Path,
                       gsim_certificate_sha256: str) -> PairedInputs:
    """Load a sealed candidate plus an explicit tuning or host-only held-out corpus."""
    functional = PC.inspect_functional_run(
        _FUNCTIONAL_RUNS, functional_run_id, functional_submission_sha256)
    try:
        handoff = PAS.verify_candidate_handoff(
            record_path, verify_authoring_tools=False, target_experiment=target_experiment)
        if phase == "tuning":
            expected_manifest = Path(corpus_root) / "performance_corpus_manifest.json"
            if Path(corpus_manifest).resolve(strict=True) != expected_manifest.resolve(strict=True):
                raise PC.CampaignGateError("tuning manifest is not under the explicit corpus root")
            corpus = PAS.load_frozen_performance_corpus(
                corpus_root, manifest_sha256=corpus_manifest_sha256,
                capsules_sha256=corpus_capsules_sha256,
                expected_target=getattr(target_experiment, "target"))
        else:
            corpus = _load_holdout_corpus(
                corpus_root, corpus_manifest, manifest_sha256=corpus_manifest_sha256,
                capsules_sha256=corpus_capsules_sha256,
                expected_target=getattr(target_experiment, "target"))
        certificate = GATE.load_certificate(
            gsim_certificate, expected_sha256=gsim_certificate_sha256)
    except Exception as exc:
        raise PC.CampaignGateError(f"paired inputs are not consumable: {exc}") from exc
    _validate_handoff(functional, handoff, corpus, phase, target_experiment)
    if certificate.target != getattr(target_experiment, "target"):
        raise PC.CampaignGateError("GSIM certificate names a different target")
    baseline = _assert_immutable_tree(
        handoff.functional_base_path, functional.digest, label="functional baseline")
    candidate = _assert_immutable_tree(
        handoff.candidate_path, handoff.candidate_sha256, label="sealed candidate")
    return PairedInputs(functional, handoff, corpus, phase, baseline, functional.digest,
                        candidate, handoff.candidate_sha256, certificate)


def _gsim_workload(member: FrozenMember) -> dict[str, Any]:
    """Use the certificate producer's one canonical workload derivation implementation."""
    try:
        return CERTPROD.derive_workload(Path(member.source_dir) / "capsule.yaml")
    except Exception as exc:
        raise PC.CampaignGateError(
            f"cannot derive GSIM workload {member.family}/{member.capsule}: {exc}") from exc


def build_measurement_plan(inputs: PairedInputs) -> MeasurementPlan:
    """Side-effect-free preflight: read frozen descriptors and launch no tools."""
    if inputs.phase not in PHASES:
        raise PC.CampaignGateError(f"phase must be one of {PHASES}")
    members = sorted(inputs.corpus.capsules,
                     key=lambda member: (str(member.family), str(member.capsule)))
    if not members:
        raise PC.CampaignGateError("measurement corpus contains zero capsules")
    member_by_key = {(str(member.family), str(member.capsule)): member for member in members}
    if len(member_by_key) != len(members):
        raise PC.CampaignGateError("measurement corpus has duplicate capsule identities")
    workloads = {key: _gsim_workload(member) for key, member in member_by_key.items()}
    unique_workloads = {GATE.workload_sha256(workload): workload for workload in workloads.values()}
    decisions = {identity: GATE.plan_evaluation(
        inputs.gsim_certificate, workload, phase="final_performance", gsim_available=True)
        for identity, workload in sorted(unique_workloads.items())}
    if any(not decision.admitted or decision.selected_engine != "gsim"
           for decision in decisions.values()):
        raise PC.CampaignGateError(
            "all paired primary workloads must be inside the pinned GSIM certificate envelope")
    pair_keys = [(family, capsule, replicate) for family, capsule in member_by_key
                 for replicate in REPLICATES]
    schedule = []
    for pair_index, (family, capsule, replicate) in enumerate(pair_keys):
        _simple_component(family, label="family")
        _simple_component(capsule, label="capsule")
        pair_id = f"{family}__{capsule}__{replicate}"
        order = ARMS if pair_index % 2 == 0 else tuple(reversed(ARMS))
        for arm in order:
            package = inputs.baseline if arm == "baseline" else inputs.candidate
            digest = inputs.baseline_sha256 if arm == "baseline" else inputs.candidate_sha256
            member = member_by_key[(family, capsule)]
            workload = workloads[(family, capsule)]
            workload_sha = GATE.workload_sha256(workload)
            schedule.append(ExecutionSpec(
                len(schedule), pair_index, pair_id, inputs.phase, arm, family, capsule,
                replicate, package, digest, member, workload, decisions[workload_sha],
                inputs.gsim_certificate))
    expected = tuple(ResultIdentity(spec.phase, spec.arm, spec.family, spec.capsule,
                                    simulator, spec.replicate)
                     for spec in schedule for simulator in spec.simulators)
    declaration = {
        "schema": "paired_arm4_measurement_plan_v3", "phase": inputs.phase,
        "primary_simulator": PRIMARY_SIMULATOR, "replicates": list(REPLICATES),
        "rtl_execution_backends": ["gsim"], "timing_authority": "gsim",
        "correctness_screen": "spike_no_timing",
        "equivalence_evidence": {
            "certificate_sha256": inputs.gsim_certificate.sha256,
            "phase": "prelaunch_and_post_reveal_qualification",
            "runtime_recorroboration": False,
        },
        "gsim_qualification": [decision.to_dict() for _, decision in sorted(decisions.items())],
        "order": "adjacent_pairs_alternating_baseline_candidate",
        "schedule": [spec.as_dict() for spec in schedule],
        "expected_results": [identity.__dict__ for identity in expected],
    }
    return MeasurementPlan(inputs.phase, tuple(schedule), expected, declaration,
                           _sha256_bytes(_canonical_bytes(declaration)))


class ContentAddressedRawStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        if self.root.exists() or self.root.is_symlink():
            raise PC.CampaignGateError(f"raw store must be fresh: {self.root}")
        (self.root / "sha256").mkdir(parents=True)

    def put(self, document: object) -> dict[str, Any]:
        payload = _canonical_bytes(document)
        digest = _sha256_bytes(payload)
        path = self.root / "sha256" / f"{digest}.json"
        if path.exists():
            if path.is_symlink() or path.read_bytes() != payload:
                raise PC.CampaignGateError("raw-result digest collision")
        else:
            path.write_bytes(payload)
            path.chmod(0o444)
        if _sha256_file(path) != digest:
            raise PC.CampaignGateError("raw-result verification failed")
        return {"sha256": digest, "path": str(path.resolve()), "n_bytes": len(payload)}


def _kernel_record(member: FrozenMember) -> dict[str, Any]:
    descriptor = dict(member.descriptor)
    descriptor["id"] = member.capsule
    descriptor.setdefault("source", "frozen_generated_performance_corpus")
    # FAIL CHEAP. This defaulted an unlabelled kernel to "L2+L3", so a descriptor that simply did
    # not mention a tier took the most expensive path available -- the wrong direction to fail in,
    # since one deep member can cost more than the whole rest of the corpus. An unlabelled kernel
    # now stays at the loop tier; `run_perf_bench.plan_cert_tier` is what promotes one, from a
    # measured price rather than from the absence of a string.
    descriptor.setdefault("sim_hint", "L2_only")
    operation = descriptor.get("operation")
    attrs = operation.get("attributes") if isinstance(operation, Mapping) else None
    if isinstance(attrs, Mapping) and isinstance(attrs.get("output_dtype"), str):
        descriptor.setdefault("output_dtype", attrs["output_dtype"])
    return descriptor


def _fresh_directory(path: Path) -> Path:
    if path.exists() or path.is_symlink():
        raise PC.CampaignGateError(f"workspace is not fresh: {path}")
    path.mkdir(parents=True)
    return path


#: L3 RESULTS ALREADY PAID FOR, keyed by the bytes that determine them.
#:
#: The search re-measures programs it has already measured, constantly. Across every campaign on
#: disk, consecutive candidates emitted BYTE-IDENTICAL code for every corpus member -- the agent
#: edits, the harness pays a full 38-member cycle-accurate sweep, and the emitted program is the one
#: it measured last time. Measured over the 380-execution sweep of 2026-09-05: 296 of its 380 L3
#: executions were answered from this table -- 11.15 of the 14.32 GSIM-hours it would otherwise
#: have spent.
#:
#: This is not a screen and it is not a prediction. Two runs of the same program on the same pinned
#: simulator return the same cycles -- verified here over 392 repeated measurements of identical
#: bytes with zero disagreement -- so a hit RETURNS the measurement rather than estimating it. The
#: key is the compiler's own emitted output, the command buffer AND the lowered module together,
#: because the command buffer alone is not the program: 28 members shared one and only 15 of them
#: shared a cycle count. Keyed on the lowered module the agreement is exact, 15 of 15.
#:
#: Scoped per pinned engine, so a different simulator build shares nothing with this table, and per
#: MEASUREMENT SCOPE (see :func:`_l3_memo_key`), so a pass that exists to observe what the previous
#: pass did not observe is never answered by that previous pass.
#:
#: This table is the FAST half and it dies with the process; :class:`L3MeasurementStore` is the half
#: that survives one. The distinction is not academic: the three trials of the 2026-09-05 launch ran
#: as three stage processes over 81 distinct programs, 71 of which more than one process measured
#: from scratch, so each trial paid its own 3.2-3.6 GSIM-hours for very largely the same work.
_L3_MEMO: dict[str, dict[str, Any]] = {}

#: Where the surviving half lives. PURGEABLE by construction: every entry is a measurement that can
#: be re-derived by running the same program on the same pinned engine again.
_L3_CACHE_NAMESPACE = "perf_l3_measurements"
_L3_CACHE_SCHEMA = "l3_measurement_reuse_v1"


def _l3_memo_key(cb: Mapping[str, Any], llvm_text: str, binary_sha256: str, scope: str) -> str:
    """What the cycles are a function of: the emitted program, the engine, and what is being observed.

    ``scope`` names the independent observation the caller is making -- its replicate identity and
    its counter pass. It belongs in the key because two observations can run THE SAME program on THE
    SAME engine and still not be the same measurement, and a table that cannot tell them apart hands
    one the other's number.

    The counter passes are the sharp case. :func:`run_execution` runs one program up to three ways
    (unprofiled, occupancy, physical bytes); the only difference between them lives in the process
    environment the ELF is built under, so nothing in the command buffer or the lowered module can
    see it. Without the scope the physical-byte pass would be served the occupancy pass's readings
    and ``_link_counter_passes`` would link a pass to itself. The replicate is the other case:
    ``REPLICATES`` exists so a campaign can MEASURE the replicate dispersion rather than assume it
    zero, and a second replicate served from the first's number assumes precisely that.
    """
    digest = hashlib.sha256()
    for part in (binary_sha256, scope,
                 json.dumps(cb, sort_keys=True, separators=(",", ":")), llvm_text):
        digest.update(part.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _engine_identity(certificate: GATE.CertificateRecord) -> dict[str, str]:
    """Which engine a cycle count belongs to: the binary that ran it, and the RTL it was built from.

    The binary sha alone decides the key -- it IS the engine that produced the number. All three
    pins travel with a stored record anyway, and a reader compares them before it believes one, so a
    table filled under one certificate can never answer for a run pinned to a different one.
    """
    return {name: str(certificate.pins[name]["sha256"])
            for name in ("gsim_binary", "gsim_firrtl", "gsim_model")}


class L3MeasurementStore:
    """Measurements already paid for, kept where the NEXT process can find them.

    Deliberately not :class:`ContentAddressedRawStore`, which sits a few lines up and does something
    that looks similar and is not. That store addresses a document by the digest of its OWN bytes,
    requires its root to be fresh, and treats a second write at one address as a collision to
    refuse. All three are right for the campaign's raw-result ledger and all three are wrong here:
    this store addresses a measurement by the digest of the INPUTS that determine it, and it has to
    OUTLIVE the run that filled it, because surviving the process boundary is the only thing it adds
    over the in-process table above.

    Every failure resolves to "measure it again". An absent, unreadable, half-written or
    differently-pinned entry is a MISS and never a guess -- the cost of a miss is a simulation the
    run was going to pay for anyway, and a cache that answers when it is not sure is worse than no
    cache at all.
    """

    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str, engine: Mapping[str, str]) -> dict[str, Any] | None:
        """The stored measurement for ``key`` on ``engine``, or None -- never a partial answer."""
        try:
            record = json.loads(self._path(key).read_bytes())
        except (OSError, ValueError):
            return None
        if not isinstance(record, Mapping):
            return None
        # RE-DERIVED, NOT TRUSTED. The file is named by the key, so a record whose own key disagrees
        # with the name is corruption; and the engine pins are compared as data rather than assumed
        # from the filename, because reusing a cycle count across simulator builds is a wrong number
        # rather than a slow one.
        if record.get("key") != key or dict(record.get("engine_pins") or {}) != dict(engine):
            return None
        evidence, result = record.get("evidence"), record.get("result")
        if not isinstance(evidence, Mapping) or not isinstance(result, Mapping):
            return None
        return {"evidence": dict(evidence), "result": dict(result)}

    def put(self, key: str, engine: Mapping[str, str], entry: Mapping[str, Any]) -> None:
        """Record a measurement for the next process. Never raises: a cache is not a gate."""
        record = {"schema": _L3_CACHE_SCHEMA, "key": key, "engine_pins": dict(engine),
                  "evidence": entry.get("evidence"), "result": entry.get("result")}
        try:
            payload = _canonical_bytes(record)
        except (TypeError, ValueError):
            # A measurement that will not serialize stays in the in-process table, which is where it
            # was already useful. Refusing to write it is a lost saving; writing something lossy
            # would be a wrong number later.
            return
        tmp = None
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = self.root / f".{key}.{os.getpid()}.{threading.get_ident()}.tmp"
            tmp.write_bytes(payload)
            # ATOMIC PUBLICATION, because concurrent sweeps and concurrent processes both write here
            # and a reader must never see half a record. A torn read would be a miss anyway (the
            # parse fails), but the rename costs nothing and removes the question.
            os.replace(tmp, self._path(key))
        except OSError:
            if tmp is not None:
                tmp.unlink(missing_ok=True)


def _l3_store(target: str) -> L3MeasurementStore:
    """The cross-process store for ``target``, grouped by target the way every other product is.

    ONE STORE, SHARED BY EVERY PROCESS ON THIS CHECKOUT -- including the parallel trials of one
    experiment, which are otherwise required to share nothing mutable. That is sound, and it is worth
    saying why rather than leaving a reader to wonder. An entry is a pure function of the emitted
    program and the pinned engine, and a hit requires a trial's OWN bytes to be identical to bytes
    already measured; what comes back is then that trial's own number, the one it was about to spend
    an hour deriving. Nothing about another trial's candidate can reach an agent through it. What the
    sharing must not be is INVISIBLE, which is why every reused row now says so in its provenance.
    """
    return L3MeasurementStore(
        cache_dir(f"{_L3_CACHE_NAMESPACE}/{_simple_component(target, label='target')}",
                  ensure=False))


def _gsim_l3_adapter(target: str, evidence: dict[str, Any],
                     certificate: GATE.CertificateRecord, *, reuse_scope: str,
                     store: L3MeasurementStore | None = None) -> Callable[..., dict[str, Any]]:
    """The L3 oracle, with the measurements it has already paid for in front of it.

    ``reuse_scope`` names the independent observation this adapter is making, so that two passes
    over one program stay two measurements (see :func:`_l3_memo_key`). ``store`` is the cross-process
    half; the caller may pass one for a test, and otherwise it is the target's own cache.
    """
    engine = _engine_identity(certificate)
    store = _l3_store(target) if store is None else store

    def run(cb: dict[str, Any], llvm_text: str, workdir: str | Path,
            timeout: int) -> dict[str, Any]:
        from merlin.runtime.backends import base as backends
        backend = backends.get_backend(target)
        resolver = getattr(backend, "gsim_path", None)
        if not callable(resolver):
            raise RuntimeError("backend cannot expose the selected GSIM binary identity")
        actual_binary = Path(resolver()).resolve(strict=True)
        if _sha256_file(actual_binary) != certificate.pins["gsim_binary"]["sha256"]:
            raise RuntimeError("runtime GSIM binary differs from the GSIM certificate pin")
        pinned = engine["gsim_binary"]
        key = _l3_memo_key(cb, llvm_text, pinned, reuse_scope)
        cached = _L3_MEMO.get(key)
        if cached is None:
            # NOT IN THIS PROCESS, so ask the one that outlives it. Promoted into the table on the
            # way through, because the next lookup in this process should not go back to disk.
            cached = store.get(key, engine)
            if cached is not None:
                _L3_MEMO[key] = cached
        if cached is not None:
            # ALREADY MEASURED, so return the measurement rather than repeating it. Everything the
            # cycles depend on -- the emitted program and the pinned engine -- is in the key, and
            # this engine is deterministic, so re-running is guaranteed to return this same number.
            evidence["gsim"] = copy.deepcopy(cached["evidence"])
            # THIS RUN BUILT NO ELF, so it must not name one. The digest stays -- it identifies the
            # program the cycles belong to -- but the path is dropped, because a record pointing at
            # a file this run did not produce reads as evidence it did.
            evidence["gsim"]["elf"] = None
            evidence["gsim"]["reused_measurement"] = {
                "basis": ("an identical emitted program was already measured on this pinned engine "
                          "in this stage; the cycle count is the one it returned, not an estimate"),
                "measured_program_sha256": key}
            reused = copy.deepcopy(cached["result"])
            reused["elf"] = None
            reused["reused_measurement"] = True
            # THE TIMING BLOCK IS THE ONLY PART OF THIS RETURN THE TIER RECORD KEEPS, so it is the
            # only place a reader of `capsule_result.json` can be told that `sim_active_s` is time an
            # EARLIER run spent. Without it the record shows 138 s of simulation beside an adapter
            # wall of 0.02 s and nothing says which of the two this run actually paid -- which is how
            # a reuse that was working the whole time read as one that had never fired: every run
            # tree on disk was grepped for the stamp and returned nothing, because nothing on the
            # loop path ever wrote it down.
            timing = dict(reused.get("timing") or {})
            timing["reused_measurement"] = True
            reused["timing"] = timing
            return reused
        primary = OOT.run_on_oracle(cb, llvm_text, simulator="gsim", target=target,
                                    workdir=workdir, timeout=timeout)
        elf = Path(str(primary["elf"])).resolve(strict=True)
        digest = _sha256_file(elf)
        output_sha, output_tensors = CERTPROD.encode_declared_outputs(
            primary.get("outputs"), cb)
        primary_oracle = primary.get("oracle")
        evidence["gsim"] = {
            "engine": "gsim", "status": "pass", "elf": str(elf), "elf_sha256": digest,
            "output_sha256": output_sha, "output_encoding": CERTPROD.OUTPUT_ENCODING,
            "output_tensors": output_tensors,
            "oracle": copy.deepcopy(primary.get("oracle")),
            "cycles": primary.get("cycles"),
            "derived_from_rtl": (primary_oracle.get("derived_from_rtl") is True
                                 if isinstance(primary_oracle, Mapping) else False),
            "cycle_accurate": True,
            "binary_sha256": certificate.pins["gsim_binary"]["sha256"],
            "firrtl_sha256": certificate.pins["gsim_firrtl"]["sha256"],
            "model_sha256": certificate.pins["gsim_model"]["sha256"],
            # SAID EXPLICITLY, so that SILENCE is not one of the answers. A hit writes a
            # ``reused_measurement`` block here and a fresh measurement wrote nothing, which left an
            # absent key meaning BOTH "this run measured it" and "nobody recorded which" -- and a
            # reader cannot audit either one. False is a claim this run makes; None is now only ever
            # a defect, and :func:`reuse_report` refuses a campaign that produces one.
            "reused_measurement": False}
        _L3_MEMO[key] = {"evidence": copy.deepcopy(evidence["gsim"]),
                         "result": copy.deepcopy(primary)}
        store.put(key, engine, _L3_MEMO[key])
        return primary
    return run


def _run_arm4_engines(package: Path, kernel: dict[str, Any], kernel_dir: Path,
                      runs: Path, timeout: int, target: str, *, measurement_pass: str,
                      expected_package_sha256: str, rtl_identity: Mapping[str, Any],
                      decision: GATE.EvaluationDecision,
                      certificate: GATE.CertificateRecord,
                      reuse_scope: str,
                      workers: int | None = None) -> dict[str, Any]:
    """Fixed Arm-4 semantics with GSIM as the only RTL execution/timing backend."""
    result: dict[str, Any] = {"approach": "arm4", "ok_build": True, "per_sim": {}}
    package_before, inputs_before = (str(hash_tree(package)["sha256"]),
                                     str(hash_tree(kernel_dir)["sha256"]))
    capsule = dict(FIXED.CR.load_capsule(kernel_dir, contract=FIXED._CONTRACT))
    capsule["required_oracle_tiers"] = ["L0", "L1", "L2", "L3"]
    evidence: dict[str, Any] = {}
    adapters = {
        "L2": FIXED.CR._spike_verilator_adapter("spike", target),
        "L3": _gsim_l3_adapter(target, evidence, certificate, reuse_scope=reuse_scope),
    }
    try:
        grade = FIXED.CR.run_capsule(
            capsule, str(package), runs_root=str(runs),
            run_id=f"arm4_{kernel['id']}_{measurement_pass}", contract=FIXED._CONTRACT,
            # The fan-out this measurement actually ran at, so the stamp on its own result is not
            # a lie. It is only bookkeeping for cycles -- which are concurrency-invariant -- but the
            # cheapest-first ordering prices members by WALL time, and that reader needs to know
            # which rows were measured beside others.
            oracle_adapters=adapters, timeout=timeout, target=target, workers=int(workers or 1))
    except Exception as exc:
        result.update({"ok_build": False, "status": "error",
                       "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                       "traceback": traceback.format_exc()[-1600:],
                       "gsim_execution": evidence})
        return result
    result["status"] = grade.get("status")
    numeric = grade.get("numeric")
    result["numeric"] = numeric.get("status") if isinstance(numeric, Mapping) else numeric
    work = grade.get("work_volume") if isinstance(grade.get("work_volume"), Mapping) else {}
    result["work_volume"] = dict(work)
    if isinstance(grade.get("command_buffer_artifact"), Mapping):
        result["command_buffer_artifact"] = dict(grade["command_buffer_artifact"])
    facts = rtl_identity.get("rtl_facts") if isinstance(rtl_identity, Mapping) else None
    rtl_sha = facts.get("sha256") if isinstance(facts, Mapping) else None
    identity, refusals = FIXED._measurement_identity(
        package_before=package_before, package_after=str(hash_tree(package)["sha256"]),
        inputs_before=inputs_before, inputs_after=str(hash_tree(kernel_dir)["sha256"]),
        work_volume=work, toolchain_shas=grade.get("toolchain_shas"), target=target,
        expected_package_sha256=expected_package_sha256, rtl_facts_sha256=rtl_sha)
    result["measurement_identity"], result["measurement_identity_refusals"] = identity, refusals
    tiers = grade.get("tiers") or {}
    for simulator, tier in (("spike", "L2"), ("gsim", "L3")):
        tr = tiers.get(tier) or {}
        status = tr.get("status") if isinstance(tr, Mapping) else tr
        cycles = tr.get("cycles") if isinstance(tr, Mapping) else None
        rtl = (simulator == "gsim" and isinstance(tr, Mapping)
               and tr.get("derived_from_rtl") is True and tr.get("cycle_accurate") is True)
        elf = evidence.get("gsim", {})
        result["per_sim"][simulator] = {
            "cycles": cycles if rtl else None,
            "correctness_cycles": cycles if simulator == "spike" else None,
            "correct": status == "pass", "tier_status": status,
            "provenance": ({"tier": tier, "simulator": simulator,
                            "oracle_kind": ((elf.get("oracle") or {}).get("kind")
                                            if simulator == "gsim" else None),
                            "derived_from_rtl": tr.get("derived_from_rtl") is True,
                            "cycle_accurate": tr.get("cycle_accurate") is True,
                            "evidence": tr.get("evidence"),
                            # WHETHER THIS ROW'S CYCLES WERE MEASURED HERE OR REUSED, said in the
                            # provenance rather than left to be inferred from a wall time. A reused
                            # row is the same number by determinism, but a reader who cannot tell
                            # the two apart cannot audit either one.
                            "reused_measurement": (elf.get("reused_measurement")
                                                   if simulator == "gsim" else None),
                            "elf_sha256": elf.get("elf_sha256") if simulator == "gsim" else None}
                           if isinstance(tr, Mapping) else None),
            "counters": tr.get("counters") if isinstance(tr, Mapping) else None,
            "measurement_conditions": (tr.get("measurement_conditions")
                                       if isinstance(tr, Mapping) else None),
            "timing_observations": (tr.get("timing_observations")
                                    if isinstance(tr, Mapping) else None),
            "timing_capability": tr.get("timing_capability") if isinstance(tr, Mapping) else None,
            "utilization": tr.get("utilization") if isinstance(tr, Mapping) else None,
        }
    qualification_error = None
    try:
        result["gsim_qualification"] = GATE.validate_execution(
            certificate, decision, evidence.get("gsim", {}))
    except Exception as exc:
        qualification_error = f"{type(exc).__name__}: {exc}"
        result["gsim_qualification"] = {"admitted": False, "reason": qualification_error}
        result["per_sim"]["gsim"]["correct"] = False
    result["gsim_execution"] = evidence
    if qualification_error:
        result["failure"] = {"plane": "gsim_qualification", "category": "infra_refusal",
                             "detail": qualification_error}
    if grade.get("failure"):
        # `tier` and `oracle_ceiling` travel with the failure so a consumer can tell a real defect
        # from a tier that was SKIPPED for being deeper than the capsule's declared oracle ceiling.
        # Without them every consumer sees an opaque failure and must treat a skip as a defect.
        result["failure"] = {key: grade["failure"].get(key)
                             for key in ("plane", "category", "detail", "tier", "oracle_ceiling")}
    return result


def run_execution(spec: ExecutionSpec, workspace: Path, timeout: int,
                  target_experiment: object, rtl_identity: Mapping[str, Any], *,
                  hardware_counters: bool, counter_binding: object = None,
                  physical_unit: str = PHYSICAL_BYTE_UNIT,
                  workers: int | None = None) -> dict[str, Any]:
    workspace = _fresh_directory(workspace)
    package_before = str(hash_tree(spec.package)["sha256"])
    inputs_before = str(hash_tree(spec.member.source_dir)["sha256"])
    if package_before != spec.package_sha256 or inputs_before != spec.member.source_sha256:
        raise PC.CampaignGateError(f"{spec.pair_id}/{spec.arm} bytes changed before execution")
    kernel = _kernel_record(spec.member)

    def run_one(name: str) -> dict[str, Any]:
        # THE PASS IS PART OF WHAT IS BEING MEASURED. `name` distinguishes the unprofiled pass from
        # the two counter passes, whose difference lives entirely in the process environment the ELF
        # is built under and so is invisible to the emitted program; the replicate distinguishes the
        # repeat this campaign runs in order to measure its own dispersion. Both have to reach the
        # reuse key or a pass gets handed a number that answers a different question.
        work = _fresh_directory(workspace / name)
        runs = _fresh_directory(work / "capsule_runs")
        policy = PC.package_sandbox_policy(target_experiment, work, spec.package)
        with PC.boxed_entrypoints(policy):
            return _run_arm4_engines(
                spec.package, kernel, spec.member.source_dir, runs, timeout,
                getattr(target_experiment, "target"),
                measurement_pass=f"{spec.arm}_{spec.replicate}_{name}",
                expected_package_sha256=spec.package_sha256, rtl_identity=rtl_identity,
                decision=spec.gsim_decision, certificate=spec.gsim_certificate,
                reuse_scope=f"{spec.replicate}/{name}", workers=workers)

    facts = rtl_identity.get("rtl_facts") if isinstance(rtl_identity, Mapping) else None
    rtl_sha = facts.get("sha256") if isinstance(facts, Mapping) else None
    if hardware_counters:
        with FIXED._counter_environment(enabled=True, unit=None):
            occupancy = run_one("occupancy")
        with FIXED._counter_environment(enabled=True, unit=physical_unit):
            physical = run_one("physical_bytes")
        linked = FIXED._link_counter_passes(
            occupancy, physical, physical_unit=physical_unit,
            counter_binding=counter_binding, rtl_facts_sha256=rtl_sha)
        measurement = dict(occupancy)
        measurement["counter_passes"] = {"occupancy": occupancy, "physical_bytes": physical}
        measurement["linked_counter_evidence"] = linked
    else:
        with FIXED._counter_environment(enabled=False):
            measurement = run_one("unprofiled")
    package_after = str(hash_tree(spec.package)["sha256"])
    inputs_after = str(hash_tree(spec.member.source_dir)["sha256"])
    if package_after != package_before or inputs_after != inputs_before:
        raise PC.CampaignGateError(f"{spec.pair_id}/{spec.arm} immutable bytes changed")
    return {"schema": "paired_arm4_raw_execution_v2", "execution": spec.as_dict(),
            "byte_guards": {"package_before": package_before, "package_after": package_after,
                            "capsule_before": inputs_before, "capsule_after": inputs_after},
            "measurement": measurement}


def result_rows(raw: Mapping[str, Any], record: Mapping[str, Any]) -> list[dict[str, Any]]:
    execution, measurement = raw.get("execution"), raw.get("measurement")
    if not isinstance(execution, Mapping) or not isinstance(measurement, Mapping):
        raise PC.CampaignGateError("raw execution is malformed")
    per_sim = measurement.get("per_sim") if isinstance(measurement.get("per_sim"), Mapping) else {}
    simulators = execution.get("simulators")
    if not isinstance(simulators, list) or any(sim not in SIMULATORS for sim in simulators):
        raise PC.CampaignGateError("raw simulator declaration is malformed")
    overall = (measurement.get("status") == "pass" and measurement.get("numeric") == "pass"
               and not measurement.get("failure"))
    rows = []
    for simulator in simulators:
        measured = per_sim.get(simulator) if isinstance(per_sim.get(simulator), Mapping) else {}
        rows.append({
            "phase": execution.get("phase"), "arm": execution.get("arm"),
            "family": execution.get("family"), "capsule": execution.get("capsule"),
            "simulator": simulator, "replicate": execution.get("replicate"),
            "correct": measured.get("correct") is True and overall,
            "cycles": None if simulator == "spike" else measured.get("cycles"),
            "citable": simulator != "spike",
            "purpose": ("correctness_screen_only" if simulator == "spike" else
                        "primary_cycle_accurate_performance"),
            "provenance": (dict(measured["provenance"])
                           if isinstance(measured.get("provenance"), Mapping) else None),
            "qualification": (measurement.get("gsim_qualification")
                              if simulator == "gsim" else None),
            "raw_result_sha256": record.get("sha256"), "raw_result_path": record.get("path")})
    return rows


def completion_report(results: Sequence[Mapping[str, Any]],
                      expected: Sequence[ResultIdentity]) -> dict[str, Any]:
    wanted = tuple(expected)
    if not wanted or len(set(wanted)) != len(wanted):
        raise PC.CampaignGateError("expected identities are empty or duplicated")
    observed: dict[ResultIdentity, Mapping[str, Any]] = {}
    for row in results:
        identity = ResultIdentity(*(str(row.get(key) or "") for key in
                                    ("phase", "arm", "family", "capsule",
                                     "simulator", "replicate")))
        if (identity.phase not in PHASES or identity.arm not in ARMS
                or identity.simulator not in SIMULATORS or identity.replicate not in REPLICATES):
            raise PC.CampaignGateError(f"invalid result identity: {identity.label}")
        if identity in observed:
            raise PC.CampaignGateError(f"duplicate result: {identity.label}")
        observed[identity] = row
    extras = sorted(set(observed) - set(wanted))
    if extras:
        raise PC.CampaignGateError(f"unexpected results: {[item.label for item in extras]}")
    passed = failed = 0
    for identity in wanted:
        row = observed.get(identity)
        if row is None:
            continue
        if identity.simulator == "spike":
            valid = (row.get("correct") is True and row.get("cycles") is None
                     and not row.get("citable"))
        else:
            provenance, cycles = row.get("provenance"), row.get("cycles")
            valid = (row.get("correct") is True and isinstance(cycles, int)
                     and not isinstance(cycles, bool) and cycles > 0 and row.get("citable") is True
                     and isinstance(provenance, Mapping) and provenance.get("tier") == "L3"
                     and provenance.get("simulator") == identity.simulator
                     and provenance.get("oracle_kind") == f"rtl_{identity.simulator}"
                     and provenance.get("derived_from_rtl") is True
                     and provenance.get("cycle_accurate") is True
                     and _is_sha256(provenance.get("elf_sha256")))
            valid = (valid and isinstance(row.get("qualification"), Mapping)
                     and row["qualification"].get("admitted") is True)
        passed += int(valid)
        failed += int(not valid)
    missing = len(wanted) - len(observed)
    return {"expected": len(wanted), "reported": len(observed), "passed": passed,
            "failed": failed, "missing": missing,
            "complete": missing == failed == 0 and passed == len(wanted)}


def _cell_label(row: Mapping[str, Any]) -> str:
    return "/".join(str(row.get(key) or "") for key in
                    ("phase", "arm", "family", "capsule", "replicate"))


def carried_stamp(row: Mapping[str, Any]) -> bool | None:
    """Did this cited row's cycles come from an earlier measurement? None means it does not say."""
    provenance = row.get("provenance")
    stamp = provenance.get("reused_measurement") if isinstance(provenance, Mapping) else None
    if stamp is False:
        return False
    if isinstance(stamp, Mapping) and _is_sha256(stamp.get("measured_program_sha256")):
        return True
    return None


def reuse_report(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How many cited cycle counts this campaign MEASURED, and how many it CARRIED.

    A carried number is exact -- it is the number the pinned engine returned for these very bytes,
    not an estimate of it -- but a campaign that presents it as freshly measured is lying about its
    evidence, which is worse than the hours the carry saved. The per-row provenance already says so
    one row at a time; nothing said it at the level a reader of the campaign actually reads, so the
    saving was invisible and therefore unauditable.

    FAIL CLOSED on silence. A cited row whose provenance does not state which of the two it is
    counts as ``unstated`` and makes the campaign not auditable -- an absent stamp is a defect in
    the recording, never a licence to assume the row was measured here.
    """
    fresh: list[str] = []
    carried: list[dict[str, str]] = []
    unstated: list[str] = []
    for row in results:
        if row.get("simulator") != PRIMARY_SIMULATOR or row.get("citable") is not True:
            continue
        stamp = carried_stamp(row)
        if stamp is False:
            fresh.append(_cell_label(row))
        elif stamp is True:
            carried.append({"cell": _cell_label(row),
                            "measured_program_sha256":
                                row["provenance"]["reused_measurement"]["measured_program_sha256"]})
        else:
            unstated.append(_cell_label(row))
    return {
        "schema": "paired_measurement_reuse_v1", "timing_authority": PRIMARY_SIMULATOR,
        "cited_cells": len(fresh) + len(carried) + len(unstated),
        "measured_here": len(fresh), "carried": len(carried), "unstated": len(unstated),
        "carried_cells": sorted(carried, key=lambda item: item["cell"]),
        "unstated_cells": sorted(unstated),
        "auditable": not unstated,
        "basis": ("a carried cell ran the byte-identical emitted program on the byte-identical "
                  "pinned engine in an earlier measurement; its cycles are that measurement's "
                  "return value, not an estimate of it"),
    }


def paired_cycle_rows(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    indexed = {(str(row.get("arm")), str(row.get("family")), str(row.get("capsule")),
                str(row.get("simulator")), str(row.get("replicate"))): row for row in results}
    out = []
    for family, capsule, replicate in sorted({(key[1], key[2], key[4]) for key in indexed
                                               if key[3] == "gsim"}):
        baseline = indexed.get(("baseline", family, capsule, "gsim", replicate))
        candidate = indexed.get(("candidate", family, capsule, "gsim", replicate))
        b = baseline.get("cycles") if isinstance(baseline, Mapping) else None
        c = candidate.get("cycles") if isinstance(candidate, Mapping) else None
        valid = (isinstance(b, int) and not isinstance(b, bool) and b > 0
                 and isinstance(c, int) and not isinstance(c, bool) and c > 0
                 and baseline.get("correct") is True and candidate.get("correct") is True)
        out.append({"family": family, "capsule": capsule, "replicate": replicate,
                    "simulator": "gsim", "baseline_cycles": b, "candidate_cycles": c,
                    "baseline_over_candidate": b / c if valid else None, "comparable": valid,
                    # THE HEADLINE FILE SAYS WHICH HALF OF THE RATIO WAS CARRIED. The frozen
                    # baseline is the same program in every trial of a campaign, so most of these
                    # are numbers an earlier trial paid for; a reader comparing wall costs across
                    # trials needs to see that here rather than reconstruct it from raw provenance.
                    "baseline_carried": (carried_stamp(baseline)
                                         if isinstance(baseline, Mapping) else None),
                    "candidate_carried": (carried_stamp(candidate)
                                          if isinstance(candidate, Mapping) else None)})
    return out


def _roofline_cell(spec: ExecutionSpec, measurement: Mapping[str, Any]) -> dict[str, Any]:
    kernel = _kernel_record(spec.member)
    return {"kernel": f"{spec.arm}__{spec.family}__{spec.capsule}__{spec.replicate}",
            "shape": kernel.get("shape"), "work_volume": measurement.get("work_volume"),
            "command_buffer_artifact": measurement.get("command_buffer_artifact"),
            "resource_bindings": FIXED._resource_bindings(measurement),
            "output_dtype": kernel.get("output_dtype", ""), "source": kernel.get("source"),
            "sim_hint": kernel.get("sim_hint"), "approaches": {"arm4": dict(measurement)}}


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def schedule_fanout(requested: int, plan: MeasurementPlan, *,
                    hardware_counters: bool) -> dict[str, Any]:
    """How many executions may be in flight at once, and the reason whenever that is one.

    Fanning this schedule out buys WALL time and nothing else: a cycle count does not move with the
    fan-out (measured identical serial and at 16 workers; only wall times move, by up to 6.3x), so
    the campaign that comes out of a wide run is the same campaign, cell for cell. That is why this
    is an exact saving and not a cheaper answer -- every execution still runs, on the same pinned
    engine, and returns its own measured number.

    Two conditions drop the width back to one, and both are refusals rather than adjustments:

    * **Hardware counters.** ``_counter_environment`` selects a counter pass by setting PROCESS
      environment variables, and two concurrent executions cannot hold different values of one. A
      counter campaign is therefore serial; nothing about it is silently reinterpreted.
    * **A relative input path.** Some capsule paths chdir the process (an external compiler resolves
      its artifacts relative to its own root) and restore it afterwards. That is safe serially and
      is why every path this schedule carries into a worker thread must already be absolute -- a
      relative one resolved inside another thread's chdir window names a different file. Measured on
      the functional grader: of 26 capsules run 8-wide, 18 wrote their entire run tree into a
      sibling checkout. So an un-absolute input makes this run serial rather than making it wrong.
    """
    requested = int(requested)
    if requested < 1:
        raise PC.CampaignGateError("execution concurrency must be at least one")
    executions = len(plan.schedule)
    if hardware_counters:
        return {"requested": requested, "effective": 1, "executions": executions,
                "reason": "hardware-counter passes are selected by process environment variables, "
                          "which concurrent executions cannot hold different values of"}
    relative = sorted({str(path) for spec in plan.schedule
                       for path in (spec.package, spec.member.source_dir)
                       if not Path(path).is_absolute()})
    if relative:
        return {"requested": requested, "effective": 1, "executions": executions,
                "reason": "an input path is relative and a worker thread cannot resolve it safely: "
                          + ", ".join(relative[:4])}
    effective = max(1, min(requested, executions))
    return {"requested": requested, "effective": effective, "executions": executions,
            "reason": ("cycle counts are invariant under fan-out; only wall time moves"
                       if effective > 1 else "the launch declared a serial campaign")}


def execute_schedule(plan: MeasurementPlan, out_dir: Path, *, timeout: int,
                     target_experiment: object, rtl_identity: Mapping[str, Any],
                     hardware_counters: bool, counter_binding: object = None,
                     executor: Callable[..., dict[str, Any]] = run_execution,
                     progress: Callable[[str], None] = print,
                     fanout: Mapping[str, Any] | None = None
                     ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Measure every execution and record them in SCHEDULE order, however wide the run was.

    ``fanout`` is :func:`schedule_fanout`'s decision. Only the measurement leaves this thread: the
    content-addressed put, the index append and the row projection all happen here, in
    ``plan.schedule`` order, so the recorded campaign is byte-identical to the serial one whichever
    execution finishes first.
    """
    out_dir = Path(out_dir).resolve(strict=True)
    workspaces = _fresh_directory(out_dir / "_execution_workspaces")
    store = ContentAddressedRawStore(out_dir / "raw_results")
    width = 1 if fanout is None else int(fanout["effective"])
    rows, roofline, index = [], [], []
    _write_json(out_dir / "raw_results.index.json", index)
    _write_json(out_dir / "paired_completion_cells.json", rows)

    def measure(spec: ExecutionSpec) -> dict[str, Any]:
        name = f"e{spec.execution_index:04d}__{spec.pair_id}__{spec.arm}"
        return executor(spec, workspaces / name, timeout, target_experiment, rtl_identity,
                        hardware_counters=hardware_counters, counter_binding=counter_binding,
                        physical_unit=PHYSICAL_BYTE_UNIT, workers=width)

    def record_one(spec: ExecutionSpec, raw: dict[str, Any]) -> None:
        record = store.put(raw)
        index.append({**spec.as_dict(), **record})
        rows.extend(result_rows(raw, record))
        if isinstance(raw.get("measurement"), Mapping):
            roofline.append(_roofline_cell(spec, raw["measurement"]))
        _write_json(out_dir / "raw_results.index.json", index)
        _write_json(out_dir / "paired_completion_cells.json", rows)

    if width <= 1:
        for spec in plan.schedule:
            progress(f"[{spec.execution_index + 1}/{len(plan.schedule)}] {spec.pair_id} {spec.arm}")
            record_one(spec, measure(spec))
        return rows, roofline
    with ThreadPoolExecutor(max_workers=width, thread_name_prefix="perf-execution") as pool:
        launched = [(spec, pool.submit(measure, spec)) for spec in plan.schedule]
        try:
            for spec, future in launched:
                progress(f"[{spec.execution_index + 1}/{len(plan.schedule)}] "
                         f"{spec.pair_id} {spec.arm}")
                record_one(spec, future.result())
        except BaseException:
            # STOP BUYING ORACLE TIME FOR A CAMPAIGN THAT HAS ALREADY REFUSED. Cancelling only
            # reaches executions that have not started; the ones in flight are joined by the pool's
            # exit, exactly as the serial path finishes the execution it is inside.
            for _, pending in launched:
                pending.cancel()
            raise
    return rows, roofline


def _identity_guard(inputs: PairedInputs) -> dict[str, Any]:
    _verify_measurement_corpus(inputs.corpus, inputs.phase)
    GATE.load_certificate(inputs.gsim_certificate.path,
                          expected_sha256=inputs.gsim_certificate.sha256)
    return {"baseline_sha256": str(hash_tree(inputs.baseline)["sha256"]),
            "candidate_sha256": str(hash_tree(inputs.candidate)["sha256"]),
            "corpus_manifest_sha256": _sha256_file(inputs.corpus.manifest_path),
            "corpus_capsules_sha256": inputs.corpus.capsules_sha256,
            "candidate_record_sha256": inputs.handoff.record_sha256,
            "gsim_certificate_sha256": inputs.gsim_certificate.sha256}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--functional-run-id", required=True)
    parser.add_argument("--functional-submission-sha256", required=True)
    parser.add_argument("--candidate-record", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--corpus-manifest-sha256", required=True)
    parser.add_argument("--corpus-capsules-sha256", required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--gsim-certificate", type=Path, required=True)
    parser.add_argument("--gsim-certificate-sha256", required=True)
    parser.add_argument("--rtl-facts", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--hardware-counters", action=argparse.BooleanOptionalAction, default=False)
    # DECLARED, NEVER INFERRED -- the same discipline the experiment's phase width uses. Unset means
    # one, the fully serial campaign, so a launch that says nothing behaves exactly as before. The
    # width buys wall time only: cycles do not move with it, and the fan-out actually used is
    # stamped on every measured tier so no row's timing block is comparable to one it should not be.
    parser.add_argument("--sim-workers", type=int, default=1, metavar="N",
                        help="how many executions may run at once (default 1 = serial)")
    args = parser.parse_args(argv)
    _simple_component(args.run_id, label="run id")
    if args.timeout <= 0:
        raise PC.CampaignGateError("timeout must be positive")
    if not all(_is_sha256(value) for value in (args.corpus_manifest_sha256,
                                               args.corpus_capsules_sha256,
                                               args.gsim_certificate_sha256)):
        raise PC.CampaignGateError("corpus/certificate identities must be lowercase SHA-256 values")
    out_dir = PB.RUNS / args.run_id
    if out_dir.exists() or out_dir.is_symlink():
        raise PC.CampaignGateError(f"run directory must be fresh: {out_dir}")
    target = load_target_experiment(_DESCRIPTOR)
    inputs = load_paired_inputs(
        args.candidate_record, args.functional_run_id, args.functional_submission_sha256, target,
        corpus_root=args.corpus_root, corpus_manifest_sha256=args.corpus_manifest_sha256,
        corpus_capsules_sha256=args.corpus_capsules_sha256, phase=args.phase,
        corpus_manifest=args.corpus_manifest,
        gsim_certificate=args.gsim_certificate,
        gsim_certificate_sha256=args.gsim_certificate_sha256)
    plan = build_measurement_plan(inputs)
    fanout = schedule_fanout(args.sim_workers, plan, hardware_counters=args.hardware_counters)
    rtl = FIXED._load_rtl_identity(args.rtl_facts, PB.TARGET)
    counter_binding = FIXED._probe_counter_byte_bindings(rtl) if args.hardware_counters else None
    out_dir.mkdir(parents=True)
    before = _identity_guard(inputs)
    fork = PC.functional_fork(inputs.functional)
    fork_before = PC.check_fork(fork, inputs.baseline).to_dict()
    if fork_before.get("ok") is not True:
        raise PC.CampaignGateError("functional fork does not hold")
    manifest: dict[str, Any] = {
        "schema": "paired_arm4_performance_campaign_v2", "status": "NO_GO",
        "refusal": "campaign has not completed", "phase": args.phase,
        "functional_run_id": inputs.functional.run_id,
        "functional_submission_sha256": inputs.baseline_sha256,
        "candidate_record_sha256": inputs.handoff.record_sha256,
        "candidate_sha256": inputs.candidate_sha256,
        "gsim_certificate": inputs.gsim_certificate.to_dict(),
        "frozen_corpus": {"path": str(inputs.corpus.root),
                          "manifest_sha256": inputs.corpus.manifest_sha256,
                          "capsules_sha256": inputs.corpus.capsules_sha256,
                          "visibility": args.phase},
        "measurement_plan": plan.declaration, "measurement_plan_sha256": plan.declaration_sha256,
        "simulators": {"spike": "correctness_only_no_timing",
                       "gsim": "sole_rtl_execution_and_timing_backend"},
        "engine_policy": {"rtl_execution_backends": ["gsim"],
                          "timing_authority": "gsim",
                          "verilator": "prelaunch_certificate_qualification_only"},
        "execution_fanout": dict(fanout),
        "rtl_identity": rtl, "identity_before": before, "identity_after": None,
        "fork_before": fork_before, "fork_after": None,
        "completion": completion_report([], plan.expected),
        "raw_results": None, "roofline_evidence": None}
    _write_json(out_dir / "campaign_manifest.json", manifest)
    rows: list[dict[str, Any]] = []
    refusal = None
    try:
        rows, roofline_cells = execute_schedule(
            plan, out_dir, timeout=args.timeout, target_experiment=target, rtl_identity=rtl,
            hardware_counters=args.hardware_counters, counter_binding=counter_binding,
            fanout=fanout)
        manifest["completion"] = completion_report(rows, plan.expected)
        if not manifest["completion"]["complete"]:
            raise PC.CampaignGateError(f"paired completion failed: {manifest['completion']}")
        _write_json(out_dir / "paired_cycles.json", paired_cycle_rows(rows))
        roofline = FIXED._roofline_auxiliary_requirements(roofline_cells, rtl)
        _write_json(out_dir / "roofline_auxiliary_evidence.json", roofline)
        manifest["roofline_evidence"] = roofline
        # HOW MANY MEMBERS' CYCLES HAVE NO COUNTED WORK BEHIND THEM, said on the manifest a reader
        # of the campaign reads. Reported, never gated: a member `work_volume` cannot price is not
        # necessarily defective, but an absence nobody is told about reads as zero, and a zero
        # denominator on a perf bench reads as infinitely fast.
        manifest["compute_axis_coverage"] = FIXED.compute_axis_coverage(roofline_cells)
        _write_json(out_dir / "compute_axis_coverage.json", manifest["compute_axis_coverage"])
    except Exception as exc:
        refusal = f"{type(exc).__name__}: {exc}"
    finally:
        payload = _canonical_bytes({"schema": "paired_arm4_result_cells_v2", "cells": rows})
        digest = _sha256_bytes(payload)
        result_path = out_dir / f"paired_results.{digest}.json"
        result_path.write_bytes(payload)
        result_path.chmod(0o444)
        manifest["raw_results"] = {"index": str(out_dir / "raw_results.index.json"),
                                   "paired_cells": str(result_path),
                                   "paired_cells_sha256": digest, "n_cells": len(rows)}
        # WHAT THIS RUN BOUGHT AND WHAT IT CARRIED, on the manifest a reader of the campaign reads.
        # A carried cell is the same number the engine returned for the same bytes, so the campaign
        # is no weaker for it -- but only if it SAYS so. A cell that states neither is a hole in the
        # record and refuses the campaign here rather than being counted as freshly measured.
        manifest["measurement_reuse"] = reuse_report(rows)
        _write_json(out_dir / "measurement_reuse.json", manifest["measurement_reuse"])
        if refusal is None and not manifest["measurement_reuse"]["auditable"]:
            refusal = (f"CampaignGateError: {manifest['measurement_reuse']['unstated']} cited "
                       "cell(s) do not state whether their cycles were measured here or carried "
                       "from an earlier measurement of the same bytes on the same pinned engine")
        try:
            after = _identity_guard(inputs)
            manifest["identity_after"] = after
            if after != before:
                raise PC.CampaignGateError("input identities changed")
            manifest["fork_after"] = PC.check_fork(fork, inputs.baseline).to_dict()
            if manifest["fork_after"].get("ok") is not True:
                raise PC.CampaignGateError("functional fork changed")
            if FIXED._load_rtl_identity(args.rtl_facts, PB.TARGET) != rtl:
                raise PC.CampaignGateError("RTL identity changed")
        except Exception as exc:
            refusal = f"{type(exc).__name__}: {exc}"
        manifest["refusal"], manifest["status"] = refusal, "GO" if refusal is None else "NO_GO"
        _write_json(out_dir / "campaign_manifest.json", manifest)
    _coverage = manifest.get("compute_axis_coverage")
    if isinstance(_coverage, Mapping):
        print(f"compute axis: {_coverage['headline']}")
        for _row in _coverage.get("unattributed", []):
            print(f"  [no compute axis] {_row['kernel']}: {'; '.join(_row['reasons'])}")
    if refusal:
        print(f"NO-GO: {refusal}")
        return 2
    print(f"GO: {manifest['completion']['expected']} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
