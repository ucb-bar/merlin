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
import stat
import traceback
from collections.abc import Callable, Mapping, Sequence
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
from merlin.targetgen.contract import compile as OOT
from merlin.targetgen.target_experiment import load_target_experiment

ARMS = ("baseline", "candidate")
REPLICATES = ("r000", "r001", "r002")
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
    if (handoff.replicates != 3 or tuple(handoff.formal_replicate_identities) != REPLICATES):
        raise PC.CampaignGateError("measurement requires exact r000-r002 identities")
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


def _gsim_l3_adapter(target: str, evidence: dict[str, Any],
                     certificate: GATE.CertificateRecord) -> Callable[..., dict[str, Any]]:
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
            "model_sha256": certificate.pins["gsim_model"]["sha256"]}
        return primary
    return run


def _run_arm4_engines(package: Path, kernel: dict[str, Any], kernel_dir: Path,
                      runs: Path, timeout: int, target: str, *, measurement_pass: str,
                      expected_package_sha256: str, rtl_identity: Mapping[str, Any],
                      decision: GATE.EvaluationDecision,
                      certificate: GATE.CertificateRecord) -> dict[str, Any]:
    """Fixed Arm-4 semantics with GSIM as the only RTL execution/timing backend."""
    result: dict[str, Any] = {"approach": "arm4", "ok_build": True, "per_sim": {}}
    package_before, inputs_before = (str(hash_tree(package)["sha256"]),
                                     str(hash_tree(kernel_dir)["sha256"]))
    capsule = dict(FIXED.CR.load_capsule(kernel_dir, contract=FIXED._CONTRACT))
    capsule["required_oracle_tiers"] = ["L0", "L1", "L2", "L3"]
    evidence: dict[str, Any] = {}
    adapters = {
        "L2": FIXED.CR._spike_verilator_adapter("spike", target),
        "L3": _gsim_l3_adapter(target, evidence, certificate),
    }
    try:
        grade = FIXED.CR.run_capsule(
            capsule, str(package), runs_root=str(runs),
            run_id=f"arm4_{kernel['id']}_{measurement_pass}", contract=FIXED._CONTRACT,
            oracle_adapters=adapters, timeout=timeout, target=target, workers=1)
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
                  physical_unit: str = PHYSICAL_BYTE_UNIT) -> dict[str, Any]:
    workspace = _fresh_directory(workspace)
    package_before = str(hash_tree(spec.package)["sha256"])
    inputs_before = str(hash_tree(spec.member.source_dir)["sha256"])
    if package_before != spec.package_sha256 or inputs_before != spec.member.source_sha256:
        raise PC.CampaignGateError(f"{spec.pair_id}/{spec.arm} bytes changed before execution")
    kernel = _kernel_record(spec.member)

    def run_one(name: str) -> dict[str, Any]:
        work = _fresh_directory(workspace / name)
        runs = _fresh_directory(work / "capsule_runs")
        policy = PC.package_sandbox_policy(target_experiment, work, spec.package)
        with PC.boxed_entrypoints(policy):
            return _run_arm4_engines(
                spec.package, kernel, spec.member.source_dir, runs, timeout,
                getattr(target_experiment, "target"),
                measurement_pass=f"{spec.arm}_{spec.replicate}_{name}",
                expected_package_sha256=spec.package_sha256, rtl_identity=rtl_identity,
                decision=spec.gsim_decision, certificate=spec.gsim_certificate)

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
                    "baseline_over_candidate": b / c if valid else None, "comparable": valid})
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


def execute_schedule(plan: MeasurementPlan, out_dir: Path, *, timeout: int,
                     target_experiment: object, rtl_identity: Mapping[str, Any],
                     hardware_counters: bool, counter_binding: object = None,
                     executor: Callable[..., dict[str, Any]] = run_execution,
                     progress: Callable[[str], None] = print
                     ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    workspaces = _fresh_directory(out_dir / "_execution_workspaces")
    store = ContentAddressedRawStore(out_dir / "raw_results")
    rows, roofline, index = [], [], []
    _write_json(out_dir / "raw_results.index.json", index)
    _write_json(out_dir / "paired_completion_cells.json", rows)
    for spec in plan.schedule:
        name = f"e{spec.execution_index:04d}__{spec.pair_id}__{spec.arm}"
        progress(f"[{spec.execution_index + 1}/{len(plan.schedule)}] {spec.pair_id} {spec.arm}")
        raw = executor(spec, workspaces / name, timeout, target_experiment, rtl_identity,
                       hardware_counters=hardware_counters, counter_binding=counter_binding,
                       physical_unit=PHYSICAL_BYTE_UNIT)
        record = store.put(raw)
        index.append({**spec.as_dict(), **record})
        rows.extend(result_rows(raw, record))
        if isinstance(raw.get("measurement"), Mapping):
            roofline.append(_roofline_cell(spec, raw["measurement"]))
        _write_json(out_dir / "raw_results.index.json", index)
        _write_json(out_dir / "paired_completion_cells.json", rows)
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
            hardware_counters=args.hardware_counters, counter_binding=counter_binding)
        manifest["completion"] = completion_report(rows, plan.expected)
        if not manifest["completion"]["complete"]:
            raise PC.CampaignGateError(f"paired completion failed: {manifest['completion']}")
        _write_json(out_dir / "paired_cycles.json", paired_cycle_rows(rows))
        roofline = FIXED._roofline_auxiliary_requirements(roofline_cells, rtl)
        _write_json(out_dir / "roofline_auxiliary_evidence.json", roofline)
        manifest["roofline_evidence"] = roofline
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
    if refusal:
        print(f"NO-GO: {refusal}")
        return 2
    print(f"GO: {manifest['completion']['expected']} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
