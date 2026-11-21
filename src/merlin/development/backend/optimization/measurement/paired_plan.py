"""Admitted paired measurements, exact execution evidence and ordered result storage.

Admission remains the caller's responsibility; the engine never discovers a candidate.
The contract resource root is supplied explicitly by the trusted host.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
import traceback
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from merlin.common.artifacts import cache_dir
from merlin.common.digest import hash_tree
from merlin.development.backend.optimization.campaign import functional_boundary as PC
from merlin.development.backend.optimization.measurement import gsim_gate as GATE
from merlin.development.backend.optimization.measurement import gsim_workload as CERTPROD
from merlin.development.backend.optimization.measurement import gsim_workload as DIGEST
from merlin.development.backend.optimization.measurement import identities as MS
from merlin.development.backend.optimization.measurement import paired_evidence as ME
from merlin.development.targetgen.corpus import capsule_io as capsule_capsule_common
from merlin.development.targetgen.execution.runner import compile as OOT  # noqa: F401 -- the oracle tests patch here
from merlin.evaluation.capsule import runner as CR
from merlin.evaluation.capsule.evidence import certificates as capsule_adapters

# ---- paired_measurement ------------------------------------------------------------------------

ARMS = ME.ARMS

REPLICATES = ME.REPLICATES

PRIMARY_SIMULATOR = "gsim"

CORRECTNESS_SIMULATOR = "spike"

SIMULATORS = ME.SIMULATORS

PHASES = ME.PHASES

PHYSICAL_BYTE_UNIT = "BYTES"

_L3_MEMO: dict[str, dict[str, Any]] = {}

_L3_CACHE_NAMESPACE = "perf_l3_measurements"

_L3_CACHE_SCHEMA = "l3_measurement_reuse_v1"


def development_executor(
    *,
    contract_root: Path,
    arm,
    package,
    package_sha256,
    member,
    decision,
    workspace,
    timeout_s,
    certificate,
    target_experiment,
    rtl_identity,
    hardware_counters=False,
    counter_binding=None,
):
    """Run the admitted development arm through the ordinary paired executor."""
    from merlin.development.backend.optimization.author_interface.prompt_inputs import sweep_workers
    from merlin.development.backend.optimization.measurement.paired_execution import run_execution

    spec = ExecutionSpec(
        execution_index=0,
        pair_index=0,
        pair_id=f"{member.family}__{member.capsule}__development",
        phase="tuning",
        arm=arm,
        family=member.family,
        capsule=member.capsule,
        replicate="r000",
        package=package,
        package_sha256=package_sha256,
        member=member,
        workload=gsim_workload(member),
        gsim_decision=decision,
        gsim_certificate=certificate,
    )
    return run_execution(
        spec,
        workspace,
        timeout_s,
        target_experiment,
        rtl_identity,
        contract_root=contract_root,
        hardware_counters=hardware_counters,
        **({"counter_binding": counter_binding} if hardware_counters else {}),
        workers=1 if hardware_counters else sweep_workers(),
    )


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
    #: Which gSIM observation this replicate is. ``None``: its own (the replicate id). A member over the
    #: declared single-observation threshold shares one observation across its replicates.
    observation_scope: str | None = None

    @property
    def simulators(self) -> tuple[str, ...]:
        return SIMULATORS

    def as_dict(self) -> dict[str, Any]:
        return {
            "execution_index": self.execution_index,
            "pair_index": self.pair_index,
            "pair_id": self.pair_id,
            "phase": self.phase,
            "arm": self.arm,
            "family": self.family,
            "capsule": self.capsule,
            "replicate": self.replicate,
            "package_sha256": self.package_sha256,
            "simulators": list(self.simulators),
            "spike_role": "correctness_only",
            "gsim_role": "primary_cycle_accurate_elaborated_rtl",
            "rtl_execution_backends": ["gsim"],
            "timing_authority": "gsim",
            "workload_sha256": self.gsim_decision.workload_sha256,
            "gsim_decision": self.gsim_decision.to_dict(),
            **({"gsim_observation": self.observation_scope} if self.observation_scope is not None else {}),
        }


@dataclass(frozen=True)
class MeasurementPlan:
    phase: str
    schedule: tuple[ExecutionSpec, ...]
    expected: tuple[ME.ResultIdentity, ...]
    declaration: dict[str, Any]
    declaration_sha256: str
    compiler_library_binding: dict[str, Any] | None = None
    compiler_library_selection: Any = None
    compiler_library_host_selection: Any = None
    compiler_pass_preparation: dict[str, Any] | None = None
    compiler_pass_selection: Any = None
    compiler_pass_preparation_selection: Any = None


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value.lower() == value
        and all(char in "0123456789abcdef" for char in value)
    )


def simple_component(value: str, *, label: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    if not value or Path(value).name != value or value in (".", "..") or any(char not in allowed for char in value):
        raise PC.CampaignGateError(f"{label} is not a safe path component: {value!r}")
    return value


def gsim_workload(member: FrozenMember) -> dict[str, Any]:
    """Use the certificate producer's one canonical workload derivation implementation."""
    try:
        return CERTPROD.derive_workload(Path(member.source_dir) / "capsule.yaml")
    except Exception as exc:
        raise PC.CampaignGateError(f"cannot derive GSIM workload {member.family}/{member.capsule}: {exc}") from exc


SINGLE_OBSERVATION_SCOPE = "single_gsim_observation"


@dataclass(frozen=True)
class ReplicatePolicy:
    """One gSIM observation for members whose roofline floor exceeds a declared threshold.

    gSIM is deterministic: the same ELF on the same pinned engine returns the same cycles, so for a
    member above the threshold (its predeclared roofline floor, a lower bound that needs no
    measurement) the second replicate identity is served from the first's observation through the L3
    memo and stamped as carried. Members at or below it keep one gSIM observation per replicate, which
    still measures replicate dispersion where it is cheap to.
    """

    threshold_roofline_cycles: int
    machine: Mapping[str, Any]

    def floor(self, member: Any) -> int | None:
        from merlin.development.backend.optimization.measurement import feedback_metrics as FM

        value = FM.capsule_roofline(getattr(member, "descriptor", None) or {}, self.machine).get("roofline_cycles")
        return value if type(value) is int else None

    def single(self, member: Any) -> bool:
        floor = self.floor(member)
        return floor is not None and floor > self.threshold_roofline_cycles

    def record(self, members: Mapping[tuple[str, str], Any]) -> dict[str, Any]:
        return {
            "rule": "one gSIM observation per member whose roofline floor exceeds the threshold; both "
            "replicate identities cite it (gSIM is deterministic); every other member is observed per replicate",
            "threshold_roofline_cycles": self.threshold_roofline_cycles,
            "estimate": "declared-work roofline floor on the RTL-derived machine bounds (a lower bound)",
            "single_observation": sorted(
                {"family": family, "capsule": capsule, "roofline_floor_cycles": self.floor(member)}
                for (family, capsule), member in members.items()
                if self.single(member)
            ),
        }


def replicate_policy(
    threshold_roofline_cycles: int | None, *, rtl_facts: Path, certificate: Any, target: str
) -> ReplicatePolicy | None:
    """The declared single-observation policy, or None (every replicate observed) when unset."""
    if threshold_roofline_cycles is None:
        return None
    if type(threshold_roofline_cycles) is not int or threshold_roofline_cycles <= 0:
        raise PC.CampaignGateError("the single-observation threshold must be a positive cycle count")
    from merlin.development.backend.optimization.measurement.development_feedback import derive_machine_bounds

    return ReplicatePolicy(threshold_roofline_cycles, derive_machine_bounds(rtl_facts, certificate, target))


def build_measurement_plan(inputs: PairedInputs, replicate_policy: ReplicatePolicy | None = None) -> MeasurementPlan:
    """Side-effect-free preflight: read frozen descriptors and launch no tools."""
    from merlin.development.backend.optimization.campaign.compiler_library_state import handoff_selection

    library = handoff_selection(getattr(inputs, "handoff", None))
    from merlin.development.backend.optimization.campaign import pass_preparation as PREPARATION

    passes = PREPARATION.handoff_selection(getattr(inputs, "handoff", None))
    prepared = getattr(getattr(inputs, "handoff", None), "compiler_pass_preparation", None)
    if passes is not None and (
        library is None
        or passes.binding.library != library.contract.sha256
        or str(inputs.baseline) != prepared["baseline"]["path"]
        or inputs.baseline_sha256 != prepared["baseline"]["sha256"]
    ):
        raise PC.CampaignGateError("prepared measurement changed its actual baseline or exact selected library")
    if inputs.phase not in PHASES:
        raise PC.CampaignGateError(f"phase must be one of {PHASES}")
    members = sorted(inputs.corpus.capsules, key=lambda member: (str(member.family), str(member.capsule)))
    if not members:
        raise PC.CampaignGateError("measurement corpus contains zero capsules")
    try:
        ME.require_supported_oracle_selection(members)
    except ME.MeasurementEvidenceError as exc:
        raise PC.CampaignGateError(str(exc)) from exc
    member_by_key = {(str(member.family), str(member.capsule)): member for member in members}
    if len(member_by_key) != len(members):
        raise PC.CampaignGateError("measurement corpus has duplicate capsule identities")
    workloads = {key: gsim_workload(member) for key, member in member_by_key.items()}
    unique_workloads = {GATE.workload_sha256(workload): workload for workload in workloads.values()}
    decisions = {
        identity: GATE.plan_evaluation(
            inputs.gsim_certificate, workload, phase="final_performance", gsim_available=True
        )
        for identity, workload in sorted(unique_workloads.items())
    }
    if any(not decision.admitted or decision.selected_engine != "gsim" for decision in decisions.values()):
        raise PC.CampaignGateError("all paired primary workloads must be inside the pinned GSIM certificate envelope")
    pair_keys = [(family, capsule, replicate) for family, capsule in member_by_key for replicate in REPLICATES]
    schedule = []
    for pair_index, (family, capsule, replicate) in enumerate(pair_keys):
        simple_component(family, label="family")
        simple_component(capsule, label="capsule")
        pair_id = f"{family}__{capsule}__{replicate}"
        order = ARMS if pair_index % 2 == 0 else tuple(reversed(ARMS))
        for arm in order:
            package = inputs.baseline if arm == "baseline" else inputs.candidate
            digest = inputs.baseline_sha256 if arm == "baseline" else inputs.candidate_sha256
            member = member_by_key[(family, capsule)]
            workload = workloads[(family, capsule)]
            workload_sha = GATE.workload_sha256(workload)
            schedule.append(
                ExecutionSpec(
                    len(schedule),
                    pair_index,
                    pair_id,
                    inputs.phase,
                    arm,
                    family,
                    capsule,
                    replicate,
                    package,
                    digest,
                    member,
                    workload,
                    decisions[workload_sha],
                    inputs.gsim_certificate,
                    SINGLE_OBSERVATION_SCOPE
                    if replicate_policy is not None and replicate_policy.single(member)
                    else None,
                )
            )
    expected = tuple(
        ME.ResultIdentity(spec.phase, spec.arm, spec.family, spec.capsule, simulator, spec.replicate)
        for spec in schedule
        for simulator in spec.simulators
    )
    declaration = {
        "schema": "paired_arm4_measurement_plan_v3",
        "phase": inputs.phase,
        "primary_simulator": PRIMARY_SIMULATOR,
        "replicates": list(REPLICATES),
        "rtl_execution_backends": ["gsim"],
        "timing_authority": "gsim",
        "correctness_screen": "spike_no_timing",
        "equivalence_evidence": {
            "certificate_sha256": inputs.gsim_certificate.sha256,
            "certification": GATE.certification_of(inputs.gsim_certificate),
            "phase": "prelaunch_and_post_reveal_qualification",
            "runtime_recorroboration": False,
        },
        "gsim_qualification": [decision.to_dict() for _, decision in sorted(decisions.items())],
        "order": "adjacent_pairs_alternating_baseline_candidate",
        "schedule": [spec.as_dict() for spec in schedule],
        "expected_results": [identity.__dict__ for identity in expected],
    }
    if replicate_policy is not None:
        declaration["replicate_policy"] = replicate_policy.record(member_by_key)
    binding = inputs.handoff.compiler_library_binding if library is not None else None
    if binding is not None:
        declaration.update(schema="paired_arm4_measurement_plan_v4", additional_compiler_library=binding)
    if prepared is not None:
        declaration.update(schema="paired_arm4_measurement_plan_v5", compiler_pass_preparation=prepared)
    return MeasurementPlan(
        inputs.phase,
        tuple(schedule),
        expected,
        declaration,
        sha256_bytes(canonical_bytes(declaration)),
        binding,
        library,
        inputs.handoff.compiler_library_host_selection if library is not None else None,
        prepared,
        passes,
        getattr(inputs.handoff, "compiler_pass_preparation_selection", None),
    )


class ContentAddressedRawStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        if self.root.exists() or self.root.is_symlink():
            raise PC.CampaignGateError(f"raw store must be fresh: {self.root}")
        (self.root / "sha256").mkdir(parents=True)

    def put(self, document: object) -> dict[str, Any]:
        payload = canonical_bytes(document)
        digest = sha256_bytes(payload)
        path = self.root / "sha256" / f"{digest}.json"
        if path.exists():
            if path.is_symlink() or path.read_bytes() != payload:
                raise PC.CampaignGateError("raw-result digest collision")
        else:
            path.write_bytes(payload)
            path.chmod(0o444)
        if sha256_file(path) != digest:
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
    for part in (binary_sha256, scope, json.dumps(cb, sort_keys=True, separators=(",", ":")), llvm_text):
        digest.update(part.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _engine_identity(certificate: GATE.CertificateRecord) -> dict[str, str]:
    """Which engine a cycle count belongs to: the binary that ran it, and the RTL it was built from.

    The binary sha alone decides the key -- it IS the engine that produced the number. All three
    pins travel with a stored record anyway, and a reader compares them before it believes one, so a
    table filled under one certificate can never answer for a run pinned to a different one.
    """
    return {name: str(certificate.pins[name]["sha256"]) for name in ("gsim_binary", "gsim_firrtl", "gsim_model")}


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
        record = {
            "schema": _L3_CACHE_SCHEMA,
            "key": key,
            "engine_pins": dict(engine),
            "evidence": entry.get("evidence"),
            "result": entry.get("result"),
        }
        try:
            payload = canonical_bytes(record)
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
        cache_dir(f"{_L3_CACHE_NAMESPACE}/{simple_component(target, label='target')}", ensure=False)
    )


def _observed_engine_binaries(oracle: Any) -> dict[str, Any]:
    """The digests of the engine build that actually produced a measurement, or an explicit UNKNOWN.

    `_engine_provenance` digests every executable beside the engine wrapper -- merlin does not know an
    engine's binary name and must not learn one, so the pinned digest is checked for MEMBERSHIP rather
    than against a named file. Returning UNKNOWN when no provenance was established keeps an
    unestablished fact distinguishable from a verified one; it is never reported as agreement.
    """
    provenance = oracle.get("provenance") if isinstance(oracle, Mapping) else None
    binaries = provenance.get("binaries") if isinstance(provenance, Mapping) else None
    if not isinstance(binaries, Mapping) or not binaries:
        return {"status": "UNKNOWN", "reason": "the engine build that produced this measurement was not established"}
    return {"status": "observed", "digests": sorted(str(v) for v in binaries.values())}


_KEY_LOCKS: dict[str, threading.Lock] = {}

_KEY_LOCKS_GUARD = threading.Lock()


def _key_lock(key: str) -> threading.Lock:
    with _KEY_LOCKS_GUARD:
        return _KEY_LOCKS.setdefault(key, threading.Lock())


def _gsim_l3_adapter(
    target: str,
    evidence: dict[str, Any],
    certificate: GATE.CertificateRecord,
    *,
    reuse_scope: str,
    store: L3MeasurementStore | None = None,
    reuse_measurement: bool = True,
) -> Callable[..., dict[str, Any]]:
    """The L3 oracle, with the measurements it has already paid for in front of it.

    ``reuse_scope`` names the independent observation this adapter is making, so that two passes
    over one program stay two measurements (see :func:`_l3_memo_key`). ``store`` is the cross-process
    half; the caller may pass one for a test, and otherwise it is the target's own cache.
    Normal queued artifact observation and final execution bypass this legacy
    LLVM-key store, rebuilding the complete harness before every execution.
    """
    engine = _engine_identity(certificate)
    from merlin.development.targetgen.execution.runner.artifact_retesting import active_artifact_retests

    if type(reuse_measurement) is not bool:
        raise ValueError("measurement reuse requires an explicit boolean")
    reuse_measurement = reuse_measurement and active_artifact_retests() is None
    store = (_l3_store(target) if store is None else store) if reuse_measurement else None

    def run(cb: dict[str, Any], llvm_text: str, workdir: str | Path, timeout: int) -> dict[str, Any]:
        from merlin.runtime.backends import registry as backends

        backend = backends.get_backend(target)
        resolver = getattr(backend, "gsim_path", None)
        if not callable(resolver):
            raise RuntimeError("backend cannot expose the selected GSIM binary identity")
        actual_binary = Path(resolver()).resolve(strict=True)
        if sha256_file(actual_binary) != certificate.pins["gsim_binary"]["sha256"]:
            raise RuntimeError("runtime GSIM binary differs from the GSIM certificate pin")
        pinned = engine["gsim_binary"]
        key = _l3_memo_key(cb, llvm_text, pinned, reuse_scope)
        # ONE RUN PER KEY AT A TIME: a second request for the same observation (a replicate sharing a
        # single observation, run by another worker) waits for the first and is served from the memo.
        with _key_lock(key):
            cached = _L3_MEMO.get(key) if reuse_measurement else None
            if cached is None and reuse_measurement:
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
                    "basis": (
                        "an identical emitted program was already measured on this pinned engine "
                        "in this stage; the cycle count is the one it returned, not an estimate"
                    ),
                    "measured_program_sha256": key,
                }
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
            # The outputs come back as Spike-verified digests of the same ELF unless that fails or is off.
            primary = DIGEST.run_gsim(cb, llvm_text, target=target, workdir=workdir, timeout=timeout)
            elf = Path(str(primary["elf"])).resolve(strict=True)
            digest = sha256_file(elf)
            output_sha, output_tensors = CERTPROD.encode_declared_outputs(primary.get("outputs"), cb)
            primary_oracle = primary.get("oracle")
            evidence["gsim"] = {
                "engine": "gsim",
                "status": "pass",
                "elf": str(elf),
                "elf_sha256": digest,
                "output_sha256": output_sha,
                "output_encoding": CERTPROD.OUTPUT_ENCODING,
                "output_tensors": output_tensors,
                "oracle": copy.deepcopy(primary.get("oracle")),
                "cycles": primary.get("cycles"),
                "derived_from_rtl": (
                    primary_oracle.get("derived_from_rtl") is True if isinstance(primary_oracle, Mapping) else False
                ),
                "cycle_accurate": True,
                # THESE THREE ARE COPIED FROM THE CERTIFICATE, and `GATE.validate_execution` then
                # compares them back to that same certificate -- three checks that read as verification
                # and cannot fail. The engine build that actually produced the number is recorded
                # separately below, so the comparison has something to be about.
                "binary_sha256": certificate.pins["gsim_binary"]["sha256"],
                "firrtl_sha256": certificate.pins["gsim_firrtl"]["sha256"],
                "model_sha256": certificate.pins["gsim_model"]["sha256"],
                # WHAT THE RUN ACTUALLY LOADED. `program_oracle._engine_provenance` digests every
                # executable in the engine home it was loaded from, so the pinned binary must appear
                # among them. Absent provenance is recorded as UNKNOWN rather than asserted away: an
                # engine whose home could not be established is a fact about this run, not a match.
                "observed_engine_binaries": _observed_engine_binaries(primary_oracle),
                # SAID EXPLICITLY, so that SILENCE is not one of the answers. A hit writes a
                # ``reused_measurement`` block here and a fresh measurement wrote nothing, which left an
                # absent key meaning BOTH "this run measured it" and "nobody recorded which" -- and a
                # reader cannot audit either one. False is a claim this run makes; None is now only ever
                # a defect, and :func:`reuse_report` refuses a campaign that produces one.
                "reused_measurement": False,
                "readback": copy.deepcopy(primary.get("readback")),
                "artifact_retesting": copy.deepcopy(primary.get("artifact_retesting")),
            }
            if reuse_measurement:
                _L3_MEMO[key] = {"evidence": copy.deepcopy(evidence["gsim"]), "result": copy.deepcopy(primary)}
                store.put(key, engine, _L3_MEMO[key])
            return primary

    return run
