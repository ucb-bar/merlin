"""Candidate record schema, normalized handoff and declared formal-claim admission."""

from __future__ import annotations

import copy
import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.common.digest import sha256_bytes as _sha256
from merlin.development.backend.optimization import contracts as CONTRACTS
from merlin.development.backend.optimization.author_interface import broker_policy as BP
from merlin.development.backend.optimization.author_interface.broker_service import BROKER_NAME
from merlin.development.backend.optimization.authoring import telemetry_accounting as TEL
from merlin.development.backend.optimization.claims import dispatch as CD
from merlin.development.backend.optimization.contracts import ROUND_DEADLINE_EXIT, StageGateError, _is_sha256
from merlin.development.backend.optimization.contracts import canonical_json as _canonical_json
from merlin.development.backend.optimization.measurement import gsim_gate as GATE
from merlin.development.backend.optimization.search import corpus as CORPUS

# ---- candidate_record --------------------------------------------------------------------------

SCHEMA_VERSION = 3

LIBRARY_SCHEMA_VERSION = 4

PREPARED_SCHEMA_VERSION = 5

MEASUREMENT_CONSUMER = "run_paired_perf_bench.py"

LEGACY_MEASUREMENT_CONSUMERS = frozenset({"run_perf_bench.py"})

AUDIT_REQUALIFICATION_KIND = "audit_policy_requalification"

AUDIT_REQUALIFICATION_REASON = "original refusal was caused only by answer-reconnaissance audit false positives"


@dataclass(frozen=True)
class VerifiedCandidateHandoff:
    """Narrow, stable measurement boundary produced only from a validated record."""

    record_path: Path
    record_sha256: str
    candidate_path: Path
    candidate_sha256: str
    candidate_initial_sha256: str
    functional_run_id: str
    functional_submission_sha256: str
    functional_base_path: Path
    functional_bundle_snapshot_sha256: str
    functional_bundle_manifest: Path
    functional_bundle_manifest_sha256: str
    target_descriptor: Path
    target_descriptor_sha256: str
    corpus_root: Path
    corpus_manifest: Path
    corpus_manifest_sha256: str
    corpus_sha256: str
    replicates: int
    formal_replicate_identities: tuple[str, ...]
    formal_claim: dict[str, Any]
    smoke_replicates: int
    expected_cells: tuple[dict[str, str], ...]
    families: tuple[dict[str, Any], ...]
    host_lane: dict[str, Any]
    e2e_sentinel: dict[str, Any]
    prompt_sha256: str
    prompt_facts_sha256: str
    prompt_path: Path
    transcript_path: Path
    transcript_sha256: str
    transcript_audit: dict[str, Any]
    receipt_path: Path
    receipt_sha256: str
    required_actions: tuple[str, ...]
    tool_evidence: dict[str, Any]
    sandbox_evidence: dict[str, Any]
    telemetry_evidence: dict[str, Any]
    codex_binary_sha256: str
    authoring_stage_sha256: str
    telemetry_preflight_sha256: str
    telemetry_source_sha256: dict[str, str]
    agent_contract: dict[str, Any]
    schema_version: int = 1
    compiler_library_binding: dict[str, Any] | None = None
    compiler_library_selection: Any = None
    compiler_library_host_selection: Any = None
    compiler_pass_preparation: dict[str, Any] | None = None
    compiler_pass_selection: Any = None
    compiler_pass_preparation_selection: Any = None


def analyzer_kwargs(
    entry: Callable[..., Any], providers: Mapping[str, Callable[[], Any]], *, label: str, positional: int = 1
) -> dict[str, Any]:
    """Supply exactly the run facts one analyzer entry point DECLARES, and refuse the rest.

    The signature is the interface. A family whose procedure needs the replicate schedule declares a
    ``replicates`` parameter and is handed it; one that needs nothing is handed nothing; one that
    declares a fact this stage cannot derive is REFUSED rather than called without it, because a
    missing run fact arriving as a default is the difference between "not measured" and "measured
    zero". ``positional`` names how many arguments the caller passes positionally (the descriptors,
    and for a decision procedure the rows), which are never supplied from here.
    """
    supplied: dict[str, Any] = {}
    for index, (name, parameter) in enumerate(inspect.signature(entry).parameters.items()):
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if index < positional and parameter.kind is not inspect.Parameter.KEYWORD_ONLY:
            continue
        provide = providers.get(name)
        if provide is None:
            if parameter.default is not inspect.Parameter.empty:
                continue
            raise StageGateError(f"{label} requires run facts this stage cannot supply: {name!r}")
        value = provide()
        if not value:
            raise StageGateError(f"{label} requires run facts this stage cannot supply: {name!r} is unavailable")
        supplied[name] = value
    return supplied


def declared_claim_analyzer(capsules: Sequence[CORPUS.PerformanceCapsule]):
    """Resolve the ONE decision procedure the frozen corpus's own declarations name.

    Dispatch is on the declaration, never on a family name, and the resolution itself lives in
    :mod:`perf_claim_dispatch` so the stage and the report reach the same procedure from the same
    frozen fact. A family the stage has never heard of is routed by what its contract froze; one
    that declares no analyzer is refused BY NAME rather than falling through to whichever analyzer
    happened to be imported here, which is how ``PM`` and ``PV`` came to ship a frozen ``PREDICTS``
    contract that nothing ever evaluated.
    """
    from merlin.development.backend.optimization.claims import dispatch as DISPATCH

    if not capsules:
        raise StageGateError("the frozen performance corpus has no capsules")
    try:
        resolved = DISPATCH.resolve([capsule.descriptor for capsule in capsules])
    except DISPATCH.DispatchError as exc:
        raise StageGateError(str(exc)) from exc
    families = "+".join(sorted({str(capsule.family) for capsule in capsules}))
    return resolved.identity, resolved.module, resolved.preflight, families


def replicate_schedule(performance: object, requested: int | None) -> tuple[str, ...]:
    """The replicate identities this family's own declaration admits, for the requested count.

    Two declared shapes, both honoured as written: a contract that FROZE its identities gets them
    verbatim and refuses any other count, and one that states a FLOOR lets the run author the
    schedule at or above it. A declaration that states neither is refused -- a count chosen here
    would be this stage's opinion about how many measurements the family's band needs.
    """
    from merlin.analysis.perf.claims import reach as claim_reach

    try:
        contract = claim_reach.replicate_contract(performance if isinstance(performance, Mapping) else {})
    except ValueError as exc:
        raise StageGateError(f"frozen acceptance has an invalid replicate cohort: {exc}") from exc
    if contract is None:
        raise StageGateError(
            "frozen acceptance declares neither an exact nor a minimum replicate count, so the "
            "run has no schedule it could author"
        )
    if requested is not None and (isinstance(requested, bool) or not isinstance(requested, int) or requested <= 0):
        raise StageGateError("a formal replicate override must be a positive integer")
    if contract.exact_count is not None:
        if requested is not None and requested != contract.exact_count:
            raise StageGateError(f"formal replicate override must equal the frozen exact_count={contract.exact_count}")
        identities = contract.identities or tuple(f"r{index:03d}" for index in range(contract.exact_count))
        if len(identities) != contract.exact_count:
            raise StageGateError("frozen acceptance has an invalid exact replicate cohort")
        return tuple(identities)
    count = contract.minimum_count if requested is None else requested
    if count < contract.minimum_count:
        raise StageGateError(
            f"formal replicate count {count} is below the declared "
            f"minimum_count={contract.minimum_count} ({contract.source})"
        )
    return tuple(f"r{index:03d}" for index in range(count))


def preflight_cohort(formal_claim: Mapping[str, Any]) -> tuple[str, ...]:
    """The replicate identities the sealed preflight says every member is measured at.

    Read from the frozen contract when it froze them and from the preflight's own authored schedule
    when the contract states only a floor, so one helper answers for every family and no caller has
    to know which shape its family declared.
    """
    declaration = formal_claim.get("declaration") if isinstance(formal_claim, Mapping) else None
    contract = declaration.get("replicates") if isinstance(declaration, Mapping) else None
    identities = contract.get("identities") if isinstance(contract, Mapping) else None
    if identities is None:
        identities = formal_claim.get("replicates") if isinstance(formal_claim, Mapping) else None
    if (
        not isinstance(identities, Sequence)
        or isinstance(identities, str)
        or not identities
        or any(not isinstance(value, str) or not value for value in identities)
        or len(set(identities)) != len(identities)
    ):
        raise StageGateError("the formal claim preflight names no replicate schedule")
    return tuple(str(value) for value in identities)


_MEASURED_CLAIMS = ("RECOVERS", "PREDICTS", "DIFFERENTIAL")


def prepare_formal_claim(
    capsules: Sequence[CORPUS.PerformanceCapsule], requested_replicates: int | None = None
) -> dict[str, Any]:
    """Admit the frozen formal claim declaration and derive its formal result cohort.

    The corpus is already scoped to the families a run selected; which procedure decides them comes
    from their OWN ``acceptance.analyzer``, so a newly declared family reaches a handoff with no
    edit here, and one that declares nothing is refused instead of being silently skipped.
    """
    identity, module, preflight_entry, family = declared_claim_analyzer(capsules)
    descriptors = [capsule.descriptor for capsule in capsules]
    claim = (descriptors[0].get("performance") or {}).get("claim")
    if claim not in _MEASURED_CLAIMS:
        # Refused BEFORE any work is authored: this coordinator measures a correctness lane and a
        # timing lane, and a family decided from the emitted stream declares neither as its evidence.
        # Its verdict comes from trace_check's declared mode and the analyzer dispatch, not from here.
        raise StageGateError(
            f"frozen {family} declares a {claim} claim, which is decided from the candidate's emitted "
            "stream by its own analyzer; the measured-claims coordinator observes correctness and "
            "timing lanes only and cannot seal it"
        )
    identities = replicate_schedule(descriptors[0].get("performance"), requested_replicates)
    label = f"{identity.module}.{preflight_entry.__name__}"
    kwargs = analyzer_kwargs(preflight_entry, {"replicates": lambda: list(identities)}, label=label)
    preflight = preflight_entry(descriptors, **kwargs)
    if not isinstance(preflight, Mapping):
        raise StageGateError(f"{label} returned something other than a preflight mapping")
    if preflight.get("status") != "READY":
        reasons = preflight.get("refusal_reasons")
        detail = (
            "; ".join(str(value) for value in reasons)
            if isinstance(reasons, Sequence) and not isinstance(reasons, str) and reasons
            else "unknown"
        )
        raise StageGateError(f"frozen {family} formal claim preflight refused: {detail}")
    declaration = preflight.get("declaration")
    if not isinstance(declaration, Mapping):
        raise StageGateError(f"frozen {family} acceptance is not a mapping")
    CD.verify_supported_acceptance(module, declaration, family)
    if preflight_cohort(preflight) != identities:
        raise StageGateError(f"frozen {family} preflight reports a replicate schedule the stage did not author")
    expected = preflight.get("expected_identities")
    if not isinstance(expected, list) or not expected or any(not isinstance(row, Mapping) for row in expected):
        raise StageGateError(f"frozen {family} preflight produced no expected measurement cells")
    if any(not str(row.get("simulator") or "") or not str(row.get("tier") or "") for row in expected):
        raise StageGateError(f"frozen {family} preflight has an unattributed measurement cell")
    if {str(row.get("capsule")) for row in expected} != {capsule.capsule for capsule in capsules}:
        raise StageGateError(f"frozen {family} preflight cells do not cover exactly the frozen corpus")
    if {str(row.get("replicate")) for row in expected} != set(identities):
        raise StageGateError(f"frozen {family} preflight cells do not use the declared replicate schedule")
    return copy.deepcopy(dict(preflight))


def _validate_formal_claim_facts(
    formal: object,
    replicates: object,
    formal_replicate_identities: object,
    smoke_replicates: object,
    cells: object,
    families: object,
) -> None:
    if (
        not isinstance(formal, Mapping)
        or formal.get("schema_version") != 1
        or not isinstance(formal.get("family"), str)
        or not formal.get("family")
        or formal.get("claim") not in _MEASURED_CLAIMS
        or formal.get("status") != "READY"
        or formal.get("refusal_reasons") != []
    ):
        raise StageGateError("performance candidate omits a READY frozen formal claim")
    family = str(formal["family"])
    declaration = formal.get("declaration")
    if not isinstance(declaration, Mapping):
        raise StageGateError(f"performance candidate {family} acceptance is not a mapping")
    evidence = declaration.get("evidence")
    timing_simulator = evidence.get("timing_simulator") if isinstance(evidence, Mapping) else None
    supported = CD.supported_acceptance(CD.declaration_module(declaration, family), declaration, family)
    if supported is not None and _canonical_json(declaration) != _canonical_json(supported):
        raise StageGateError(f"performance candidate {family} acceptance contract drifted")
    identities = list(preflight_cohort(formal))
    if formal_replicate_identities != identities or replicates != len(identities):
        raise StageGateError(f"performance candidate formal replicates drift from {family} acceptance")
    if (
        isinstance(smoke_replicates, bool)
        or not isinstance(smoke_replicates, int)
        or smoke_replicates <= 0
        or smoke_replicates >= len(identities)
    ):
        raise StageGateError("performance candidate smoke replicas could masquerade as formal evidence")
    if not isinstance(families, list):
        raise StageGateError("performance candidate formal families are malformed")
    claiming = [row for row in families if isinstance(row, Mapping) and row.get("family") == family]
    if len(claiming) != 1 or _canonical_json(claiming[0].get("acceptance")) != _canonical_json(declaration):
        raise StageGateError(f"performance candidate family omits its exact {family} acceptance")
    cohort = formal.get("cohort")
    expected = formal.get("expected_identities")
    if not isinstance(cohort, Mapping) or not isinstance(expected, list) or not expected:
        raise StageGateError(f"performance candidate {family} preflight omits its exact cohort")
    # A cohort record that ECHOES its schedule must echo the one the preflight authored; one that
    # does not is not thereby excused, because the schedule itself is read above from the sealed
    # preflight and is what every other check here compares against.
    if cohort.get("replicates") not in (None, identities):
        raise StageGateError(f"performance candidate {family} preflight omits its exact cohort")
    expected_cells: list[dict[str, str]] = []
    for row in expected:
        if not isinstance(row, Mapping):
            raise StageGateError(f"performance candidate {family} preflight has a malformed identity")
        simulator, tier = row.get("simulator"), row.get("tier")
        if (simulator, tier) not in (("spike", "L2"), (timing_simulator, "L3")) or row.get("family") != family:
            raise StageGateError(f"performance candidate {family} preflight changes L2/L3 semantics")
        # A family measuring two ARMS per member names each arm in its own identity; the cells are
        # the measurement schedule, so identities are compared on the part a cell carries, and the
        # arm axis rides along as the extra fact the analyzer needs from each cell.
        expected_cells.append({key: str(row.get(key)) for key in ("family", "capsule", "simulator", "replicate")})
    unique_cells = sorted({tuple(row.items()) for row in expected_cells})
    if not isinstance(cells, list):
        raise StageGateError("performance candidate formal cells are malformed")
    recorded = sorted(
        {
            tuple({key: str(row.get(key)) for key in ("family", "capsule", "simulator", "replicate")}.items())
            for row in cells
            if isinstance(row, Mapping) and row.get("family") == family
        }
    )
    if unique_cells != recorded:
        raise StageGateError(f"performance candidate {family} formal identities drift from expected cells")


def audit_requalification_invariant_sha256(document: Mapping[str, Any]) -> str:
    """Hash every candidate fact except the audit-policy admission decision.

    Requalification is deliberately narrower than a waiver: it may replace only the transcript
    audits and the state/admission fields those audits determined.  Normalizing those exact fields
    lets the verifier prove that compiler bytes, receipts, feedback, telemetry, prompt, corpus, and
    every other fact still come from the immutable refused record.
    """
    normalized = copy.deepcopy(dict(document))
    normalized.pop("audit_requalification", None)
    normalized["state"] = "<audit-policy-decision>"
    admission = normalized.get("admission")
    agent = normalized.get("agent")
    if not isinstance(admission, dict) or not isinstance(agent, dict):
        raise StageGateError("performance candidate cannot normalize its audit admission")
    admission["consumable"] = "<audit-policy-decision>"
    admission["refusal"] = "<audit-policy-decision>"
    agent["audit"] = "<audit-policy-decision>"
    rounds = agent.get("rounds")
    if not isinstance(rounds, list):
        raise StageGateError("performance candidate cannot normalize its round audits")
    for row in rounds:
        if not isinstance(row, dict):
            raise StageGateError("performance candidate has a malformed round audit")
        row["audit"] = "<audit-policy-decision>"
    return _sha256(_canonical_json(normalized))


def require_audit_only_refusal(document: Mapping[str, Any]) -> None:
    """Require a complete run whose sole refusal class is answer-reconnaissance audit hits."""
    admission = document.get("admission")
    agent = document.get("agent")
    candidate = document.get("candidate")
    broker = document.get("broker")
    guard = document.get("functional_guard")
    if (
        not isinstance(admission, Mapping)
        or not isinstance(agent, Mapping)
        or not isinstance(candidate, Mapping)
        or not isinstance(broker, Mapping)
        or not isinstance(guard, Mapping)
    ):
        raise StageGateError("audit requalification source omits complete stage evidence")
    if (
        document.get("audit_requalification") is not None
        or document.get("state") != "refused"
        or admission.get("consumable") is not False
        or admission.get("refusal") != "combined Codex transcript failed the answer/tool-access audit"
    ):
        raise StageGateError("source is not the exact audit-only refusal eligible for requalification")
    round_rows = agent.get("rounds")
    audits = [agent.get("audit")]
    if isinstance(round_rows, list):
        audits.extend(row.get("audit") if isinstance(row, Mapping) else None for row in round_rows)
    if (
        not isinstance(round_rows, list)
        or not round_rows
        or agent.get("rounds_requested") != len(round_rows)
        or candidate.get("rounds_completed") != len(round_rows)
        or any(
            not isinstance(audit, Mapping)
            or audit.get("clean") is not False
            or not isinstance(audit.get("hits"), list)
            or not audit["hits"]
            or any(not isinstance(hit, Mapping) or hit.get("kind") != "answer_reconnaissance" for hit in audit["hits"])
            or not isinstance(audit.get("commands_seen"), int)
            or isinstance(audit.get("commands_seen"), bool)
            or audit["commands_seen"] <= 0
            or audit.get("broker_required") != BROKER_NAME
            or not isinstance(audit.get("broker_invocations"), list)
            for audit in audits
        )
        or any(
            not isinstance(row, Mapping) or row.get("agent_exit_code") not in (0, ROUND_DEADLINE_EXIT)
            for row in round_rows
        )
        or ((candidate.get("delta") or {}).get("execution_relevant_changed_file_count", 0) <= 0)
        or broker.get("all_required_succeeded") is not True
        or guard.get("status") != "clean"
        or guard.get("offenders") != []
    ):
        raise StageGateError("source has a refusal or incomplete evidence beyond answer-reconnaissance audit hits")


def _validate_audit_requalification(document: Mapping[str, Any]) -> None:
    requalification = document.get("audit_requalification")
    if requalification is None:
        return
    if not isinstance(requalification, Mapping):
        raise StageGateError("audit requalification provenance is not a mapping")
    source = requalification.get("source_record")
    original = requalification.get("original_audits")
    corrected = requalification.get("corrected_audits")
    snapshots = requalification.get("policy_snapshots")
    token_set = requalification.get("audit_token_set")
    agent = document.get("agent")
    round_rows = agent.get("rounds") if isinstance(agent, Mapping) else None
    expected_roles = {"audit_implementation", "answer_surface_policy"}
    version = requalification.get("schema_version")
    if version in (2, 3, 4, 5):
        expected_roles.add("shared_access_policy")
    if version == 3:
        expected_roles.add("native_controller")
    if version in (4, 5):
        identity = requalification.get("phase2_source_identity")
        TEL._validate_package_source_record(identity)
        if not {"candidate_record.py", "candidate_verification.py", "telemetry.py"} <= set(identity["members"]):
            raise StageGateError("audit requalification lacks candidate policy source owners")
        expected_roles.add("python_source_membership")
        expected_roles.update(f"phase2:{name}" for name in identity["members"])
    if version == 5:
        expected_roles.update({"historical_target_access_policy", "historical_target_access_data"})
    if (
        type(version) is not int
        or version not in (1, 2, 3, 4, 5)
        or requalification.get("kind") != AUDIT_REQUALIFICATION_KIND
        or requalification.get("reason") != AUDIT_REQUALIFICATION_REASON
        or not isinstance(source, Mapping)
        or not isinstance(source.get("path"), str)
        or not source.get("path")
        or not _is_sha256(source.get("sha256"))
        or not _is_sha256(requalification.get("invariant_evidence_sha256"))
        or not isinstance(original, Mapping)
        or not _is_sha256(original.get("combined_sha256"))
        or not isinstance(original.get("round_sha256"), list)
        or not isinstance(corrected, Mapping)
        or not _is_sha256(corrected.get("combined_sha256"))
        or not isinstance(corrected.get("round_sha256"), list)
        or not isinstance(round_rows, list)
        or len(original["round_sha256"]) != len(round_rows)
        or len(corrected["round_sha256"]) != len(round_rows)
        or any(not _is_sha256(value) for value in (*original["round_sha256"], *corrected["round_sha256"]))
        or not isinstance(original.get("hits"), list)
        or not original["hits"]
        or not isinstance(snapshots, list)
        or len(snapshots) != len(expected_roles)
        or {row.get("role") for row in snapshots if isinstance(row, Mapping)} != expected_roles
        or any(
            not isinstance(row, Mapping)
            or not isinstance(row.get("source_path"), str)
            or not row.get("source_path")
            or not isinstance(row.get("frozen_path"), str)
            or not row.get("frozen_path")
            or not _is_sha256(row.get("sha256"))
            for row in snapshots
        )
        or not isinstance(token_set, Mapping)
        or set(token_set) != {"answer", "grader", "oracle_subpath"}
        or any(
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value for value in values)
            for values in token_set.values()
        )
        or not _is_sha256(requalification.get("audit_token_set_sha256"))
        or _sha256(_canonical_json(token_set)) != requalification.get("audit_token_set_sha256")
        or not _is_sha256(requalification.get("policy_set_sha256"))
    ):
        raise StageGateError("audit requalification provenance is incomplete")
    if version in (4, 5):
        members = requalification["phase2_source_identity"]["members"]
        if any(
            row["sha256"] != members[row["role"].removeprefix("phase2:")]
            for row in snapshots
            if row["role"].startswith("phase2:")
        ):
            raise StageGateError("audit requalification source closure differs from policy snapshots")
    if (
        _sha256(_canonical_json(document["agent"]["audit"])) != corrected["combined_sha256"]
        or [_sha256(_canonical_json(row["audit"])) for row in round_rows] != corrected["round_sha256"]
    ):
        raise StageGateError("corrected audit evidence disagrees with requalification provenance")


def verify_sandbox_containment_evidence(sandbox: Mapping[str, Any], *, require_present: bool = True) -> None:
    """Check recorded operability and deny-rule coverage without upgrading historical evidence."""
    from merlin.development.targetgen.sandbox import preflight as sandbox_preflight  # noqa: PLC0415

    probe = sandbox.get("preflight")
    if probe is None and not require_present and "dropped_declarations" not in sandbox:
        return
    if (
        not isinstance(probe, Mapping)
        or probe.get("usable") is not True
        or probe.get("status") != sandbox_preflight.SANDBOX_OK
    ):
        raise StageGateError("performance candidate record does not evidence a sandbox that could actually be built")
    if sandbox.get("dropped_declarations") != []:
        raise StageGateError(
            "performance candidate record was produced with declared answer-surface rules that masked nothing"
        )
