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

from . import broker_policy as BP
from . import contracts as CONTRACTS
from . import corpus as CORPUS
from . import gsim_gate as GATE
from . import telemetry as TEL
from .broker import BROKER_NAME
from .broker_evidence import _is_sha256
from .claims import dispatch as CD
from .contracts import ROUND_DEADLINE_EXIT, StageGateError
from .contracts import canonical_json as _canonical_json

SCHEMA_VERSION = 3


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
    from merlin_experiments.phase2.claims import dispatch as DISPATCH

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
    from merlin.perf import claim_reach

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
        or formal.get("claim") not in ("RECOVERS", "PREDICTS", "DIFFERENTIAL")
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
    if version in (2, 3, 4):
        expected_roles.add("shared_access_policy")
    if version == 3:
        expected_roles.add("native_controller")
    if version == 4:
        identity = requalification.get("phase2_source_identity")
        TEL._validate_package_source_record(identity)
        if not {"candidate_record.py", "candidate_verification.py", "telemetry.py"} <= set(identity["members"]):
            raise StageGateError("audit requalification lacks candidate policy source owners")
        expected_roles.add("python_source_membership")
        expected_roles.update(f"phase2:{name}" for name in identity["members"])
    if (
        type(version) is not int
        or version not in (1, 2, 3, 4)
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
    if version == 4:
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
    from merlin.targetgen.sandbox import preflight as sandbox_preflight  # noqa: PLC0415

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


def validate_candidate_record(document: Mapping[str, Any], *, require_consumable: bool = True) -> dict:
    """Pure schema/boundary validator for the paired measurement runner."""
    if not isinstance(document, Mapping) or document.get("schema_version") != SCHEMA_VERSION:
        raise StageGateError("performance candidate record has an unsupported schema")
    if document.get("kind") != "arm4_performance_candidate":
        raise StageGateError("performance candidate record has a foreign kind")
    target = document.get("target")
    base = document.get("base_functional")
    candidate = document.get("candidate")
    prompt = document.get("prompt")
    corpus = document.get("performance_corpus")
    sandbox = document.get("sandbox")
    broker = document.get("broker")
    development_feedback = document.get("development_feedback")
    agent = document.get("agent")
    telemetry = document.get("telemetry")
    admission = document.get("admission")
    if not all(
        isinstance(value, Mapping)
        for value in (
            target,
            base,
            candidate,
            prompt,
            corpus,
            sandbox,
            broker,
            development_feedback,
            agent,
            telemetry,
            admission,
        )
    ):
        raise StageGateError("performance candidate record omits a required evidence mapping")
    for label, value in (
        ("functional submission", base.get("submission_sha256")),
        ("candidate initial", candidate.get("initial_sha256")),
        ("candidate final", candidate.get("sha256")),
        ("prompt", prompt.get("sha256")),
        ("prompt facts", prompt.get("facts_sha256")),
        ("prompt renderer", prompt.get("renderer_sha256")),
        ("performance manifest", corpus.get("manifest_sha256")),
        ("performance corpus", corpus.get("capsules_sha256")),
        ("agent-input manifest", corpus.get("agent_input_manifest_sha256")),
        ("agent-input content", corpus.get("agent_input_sha256")),
        ("target descriptor", target.get("descriptor_sha256")),
        ("Codex binary", agent.get("codex_binary_sha256")),
        ("transcript", agent.get("transcript_sha256")),
        ("broker registry", broker.get("registry_sha256")),
        ("broker receipt manifest", broker.get("receipt_manifest_sha256")),
        ("development GSIM certificate", (development_feedback.get("certificate") or {}).get("sha256")),
    ):
        if not _is_sha256(value):
            raise StageGateError(f"performance candidate record has no valid {label} SHA-256")
    if candidate.get("initial_sha256") != base.get("submission_sha256"):
        raise StageGateError("performance candidate was not forked byte-for-byte from functional")
    if (
        not isinstance(target.get("name"), str)
        or not target.get("name")
        or not isinstance(target.get("descriptor"), str)
        or not target.get("descriptor")
    ):
        raise StageGateError("performance candidate record omits its target descriptor identity")
    CONTRACTS.safe_component(str(base.get("run_id") or ""), label="base functional run id")
    bundle_snapshot = base.get("bundle_input_snapshot")
    host_lane = base.get("model_host_lane")
    sentinel = base.get("e2e_sentinel")
    if (
        not isinstance(bundle_snapshot, Mapping)
        or not _is_sha256(bundle_snapshot.get("content_sha256"))
        or not _is_sha256(bundle_snapshot.get("manifest_sha256"))
        or not isinstance(bundle_snapshot.get("grants"), list)
        or not bundle_snapshot["grants"]
    ):
        raise StageGateError("performance candidate omits the frozen functional grant snapshot")
    if (
        not isinstance(host_lane, Mapping)
        or not _is_sha256(host_lane.get("package_sha256"))
        or not isinstance(host_lane.get("integration_seam"), str)
    ):
        raise StageGateError("performance candidate omits its exact functional host lane")
    if (
        not isinstance(sentinel, Mapping)
        or not _is_sha256(sentinel.get("capsule_sha256"))
        or set(sentinel.get("required_lanes") or []) != {"on_mesh", "scalar_rvv_lane"}
        or "L3" not in (sentinel.get("required_tiers") or [])
    ):
        raise StageGateError("performance candidate omits its frozen full-model E2E sentinel")
    for evidence, label in (
        (candidate.get("path"), "sealed candidate path"),
        (prompt.get("staged_path"), "staged prompt path"),
        (corpus.get("path"), "frozen performance corpus path"),
        (agent.get("codex_binary"), "Codex binary path"),
        (agent.get("transcript"), "combined transcript path"),
    ):
        if not isinstance(evidence, str) or not evidence:
            raise StageGateError(f"performance candidate record omits the {label}")
    for attribute in ("agent_input_files", "agent_input_bytes"):
        value = corpus.get(attribute)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise StageGateError(f"performance candidate record has no non-vacuous {attribute}")
    replicates = corpus.get("replicates")
    cells, families = corpus.get("expected_cells"), corpus.get("families")
    formal_claim = corpus.get("formal_claim")
    formal_replicate_identities = corpus.get("formal_replicate_identities")
    smoke_replicates = corpus.get("smoke_replicates")
    prompt_facts = prompt.get("facts")
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates <= 0
        or not isinstance(cells, list)
        or not cells
        or not isinstance(families, list)
        or not families
        or not isinstance(prompt_facts, Mapping)
        or prompt_facts.get("expected_cells") != cells
        or prompt_facts.get("families") != families
        or prompt_facts.get("replicates") != replicates
        or prompt_facts.get("formal_claim") != formal_claim
        or prompt_facts.get("formal_replicate_identities") != formal_replicate_identities
        or prompt_facts.get("smoke_replicates") != smoke_replicates
    ):
        raise StageGateError("performance candidate omits exact prompt cells/families/replicates")
    _validate_formal_claim_facts(
        formal_claim, replicates, formal_replicate_identities, smoke_replicates, cells, families
    )
    if candidate.get("read_only") is not True or candidate.get("base_submission_overwritten") is not False:
        raise StageGateError("performance candidate record does not prove a separate read-only snapshot")
    delta = candidate.get("delta")
    if (
        not isinstance(delta, Mapping)
        or not isinstance(delta.get("changed_files"), list)
        or not isinstance(delta.get("execution_relevant_changed_files"), list)
        or delta.get("changed_file_count") != len(delta["changed_files"])
        or delta.get("execution_relevant_changed_file_count") != len(delta["execution_relevant_changed_files"])
    ):
        raise StageGateError("performance candidate record has no structured candidate delta")
    outer = sandbox.get("outer_codex_control_plane")
    inner = sandbox.get("inner_execution_plane")
    if not isinstance(outer, Mapping) or not isinstance(inner, Mapping):
        raise StageGateError("performance candidate record collapses its two sandbox boundaries")
    verify_sandbox_containment_evidence(sandbox, require_present=False)
    network_record = "available_not_an_isolation_claim"
    mounted_manifest_sha256 = bundle_snapshot.get("public_manifest_sha256") or bundle_snapshot.get("manifest_sha256")
    if (
        outer.get("network") != network_record
        or outer.get("clear_environment") is not True
        or outer.get("auth_exception") != "isolated_codex_home_explicit_auth_mount"
        or outer.get("session_history_mounted") is not False
        or outer.get("live_target_toolchain_mounted") is not False
        or outer.get("frozen_functional_grants_mounted") is not True
        or outer.get("frozen_grant_manifest_sha256") != mounted_manifest_sha256
        or outer.get("answer_surface_gap") != []
    ):
        raise StageGateError("outer Codex control-plane evidence is incomplete")
    if not isinstance(outer.get("bwrap_binary"), str) or not outer.get("bwrap_binary"):
        raise StageGateError("outer Codex control-plane evidence omits the bwrap binary path")
    for evidence, label in (
        (outer.get("bwrap_binary_sha256"), "bwrap binary"),
        (outer.get("policy_sha256"), "outer bwrap policy"),
        (inner.get("policy_sha256"), "inner bwrap policy"),
    ):
        if not _is_sha256(evidence):
            raise StageGateError(f"performance candidate record omits the {label} digest")
    if (
        inner.get("network") != network_record
        or inner.get("clear_environment") is not True
        or inner.get("credentials") != "none"
        or inner.get("answer_surface_gap") != []
        or inner.get("candidate_writable") is not True
        or inner.get("corpus_read_only") is not True
        or inner.get("frozen_functional_grants_mounted") is not True
        or inner.get("frozen_grant_manifest_sha256") != mounted_manifest_sha256
    ):
        raise StageGateError("inner execution-plane evidence is not credential-free and answer-masked")
    probes = inner.get("tool_probe_results")
    probe_recheck = inner.get("tool_probe_recheck_results")
    if (
        not isinstance(probes, list)
        or not probes
        or any(
            not isinstance(row, Mapping)
            or row.get("returncode") != 0
            or not isinstance(row.get("label"), str)
            or not row.get("label")
            or not isinstance(row.get("command"), str)
            or not row.get("command")
            for row in probes
        )
    ):
        raise StageGateError("performance candidate record lacks passing inner tool probes")
    if probe_recheck != probes:
        raise StageGateError("performance candidate tool evidence changed between preflight and recheck")
    registry, receipt_rows = broker.get("registry"), broker.get("round_receipts")
    if (
        not isinstance(registry, list)
        or not registry
        or _sha256(_canonical_json(registry)) != broker.get("registry_sha256")
        or not isinstance(receipt_rows, list)
        or any(
            not isinstance(row, Mapping)
            or not isinstance(row.get("path"), str)
            or not _is_sha256(row.get("sha256"))
            or row.get("all_required_succeeded") is not True
            or not _is_sha256(row.get("candidate_sha256"))
            or row.get("final_candidate_feedback_verified") is not True
            or not isinstance(row.get("feedback_successes"), int)
            or row.get("feedback_successes") < 1
            or not isinstance(row.get("feedback_receipts"), list)
            or not row.get("feedback_receipts")
            for row in receipt_rows
        )
        or broker.get("control_owned_by_harness") is not True
        or broker.get("control_writable_by_agent") is not False
        or not isinstance(broker.get("receipt_manifest"), str)
        or not broker.get("receipt_manifest")
    ):
        raise StageGateError("performance candidate lacks immutable broker registry/receipts")
    required_actions = sorted(
        str(row.get("name")) for row in registry if isinstance(row, Mapping) and row.get("required") is True
    )
    if not required_actions or broker.get("required_actions") != required_actions:
        raise StageGateError("performance candidate broker required-action contract is incomplete")
    feedback_actions = [
        row for row in registry if isinstance(row, Mapping) and row.get("name") == BP.DEVELOPMENT_FEEDBACK_ACTION
    ]
    if (
        len(feedback_actions) != 1
        or feedback_actions[0].get("required") is not True
        or feedback_actions[0].get("placeholders") != []
        or feedback_actions[0].get("argv_template") != [BP._HOST_FEEDBACK_SENTINEL]
        or BP.DEVELOPMENT_FEEDBACK_ACTION not in required_actions
    ):
        raise StageGateError("mandatory tuning GSIM feedback action is absent or drifted")
    certificate = development_feedback.get("certificate")
    rtl_identity = development_feedback.get("rtl_identity")
    recorded_feedback_receipts = development_feedback.get("round_receipts")
    if (
        development_feedback.get("action") != BP.DEVELOPMENT_FEEDBACK_ACTION
        or development_feedback.get("required_per_round") is not True
        or development_feedback.get("scope") != "frozen_tuning_corpus_only"
        or development_feedback.get("engine") != "gsim"
        or development_feedback.get("redaction") != "correctness_gsim_cycles_and_paired_deltas_only"
        or not isinstance(certificate, Mapping)
        or certificate.get("target") != target.get("name")
        or certificate.get("fidelity") != GATE.FIDELITY
        or not isinstance(rtl_identity, Mapping)
        or not isinstance(recorded_feedback_receipts, list)
        or recorded_feedback_receipts != [row.get("feedback_receipts") for row in receipt_rows]
    ):
        raise StageGateError("development GSIM feedback evidence is incomplete or drifted")
    budget_facts = prompt_facts.get("budgets")
    expected_budget_facts = {
        "wall_budget_seconds": agent.get("wall_budget_seconds"),
        "rounds": agent.get("rounds_requested"),
        "round_timeout_seconds": agent.get("round_timeout_seconds"),
        "max_tool_calls": agent.get("max_tool_calls"),
        "tool_timeout_seconds": agent.get("tool_timeout_seconds"),
    }
    prompt_host = prompt_facts.get("host_lane")
    host_fact_keys = {"target", "package_id", "package_path", "package_sha256", "manifest_path", "integration_seam"}
    if (
        budget_facts != expected_budget_facts
        or prompt_facts.get("tools") != registry
        or prompt_facts.get("e2e_sentinel") != sentinel
        or not isinstance(prompt_host, Mapping)
        or set(prompt_host) != host_fact_keys
        or any(prompt_host.get(key) != host_lane.get(key) for key in host_fact_keys)
        or prompt_facts.get("mount_destinations") != outer.get("mount_destinations")
    ):
        raise StageGateError("canonical prompt facts drift from recorded launch enforcement")
    round_rows = agent.get("rounds")
    rounds_requested = agent.get("rounds_requested")
    bounded_agent_values = (
        agent.get("wall_budget_seconds"),
        agent.get("rounds_requested"),
        agent.get("round_timeout_seconds"),
        agent.get("max_tool_calls"),
        agent.get("tool_timeout_seconds"),
    )
    if (
        agent.get("driver") != "codex"
        or any(
            not isinstance(agent.get(field), str) or not agent.get(field)
            for field in ("model", "resolved_model", "effort")
        )
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in bounded_agent_values)
        or not isinstance(round_rows, list)
        or not round_rows
        or isinstance(rounds_requested, bool)
        or not isinstance(rounds_requested, int)
        or rounds_requested < len(round_rows)
        or any(
            not isinstance(row, Mapping)
            or isinstance(row.get("agent_exit_code"), bool)
            or not isinstance(row.get("agent_exit_code"), int)
            or not isinstance(row.get("transcript"), str)
            or not row.get("transcript")
            or not isinstance(row.get("audit"), Mapping)
            or not isinstance(row["audit"].get("clean"), bool)
            or not isinstance(row["audit"].get("hits"), list)
            or row["audit"].get("broker_required") != BROKER_NAME
            or not isinstance(row["audit"].get("broker_invocations"), list)
            or not isinstance(row.get("telemetry"), Mapping)
            or row["telemetry"].get("event_count", 0) <= 0
            or (row["telemetry"].get("summary") or {}).get("usage_complete") is not True
            or (row["telemetry"].get("accounting") or {}).get("available") is not True
            or (row["telemetry"].get("accounting") or {}).get("usage_complete") is not True
            or isinstance(row["audit"].get("commands_seen"), bool)
            or not isinstance(row["audit"].get("commands_seen"), int)
            or not _is_sha256(row.get("transcript_sha256"))
            for row in round_rows
        )
    ):
        raise StageGateError("performance candidate was not produced by the bounded Codex driver")
    telemetry_artifacts = telemetry.get("artifacts")
    telemetry_accounting = telemetry.get("accounting")
    telemetry_reconciliation = telemetry.get("aet_reconciliation")
    activity_share = telemetry.get("activity_share")
    activity_seconds = (activity_share or {}).get("seconds_by_category") or {}
    activity_shares = (activity_share or {}).get("share_by_category") or {}
    activity_wall = (activity_share or {}).get("trajectory_wall_seconds")
    activity_occupancy = (activity_share or {}).get("classified_span_occupancy_ratio")
    activity_semantics_valid = (
        isinstance(activity_seconds, Mapping)
        and bool(activity_seconds)
        and isinstance(activity_shares, Mapping)
        and bool(activity_shares)
        and set(activity_seconds) == set(activity_shares)
        and all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0
            for value in (*activity_seconds.values(), *activity_shares.values())
        )
        and math.isclose(sum(float(value) for value in activity_shares.values()), 1.0, rel_tol=1e-6, abs_tol=1e-6)
        and isinstance((activity_share or {}).get("classified_seconds"), (int, float))
        and not isinstance((activity_share or {}).get("classified_seconds"), bool)
        and float(activity_share["classified_seconds"]) > 0
        and math.isclose(
            sum(float(value) for value in activity_seconds.values()),
            float(activity_share["classified_seconds"]),
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
        and activity_share.get("schema_version") == 2
        and activity_share.get("denominator") == "sum_of_classified_tool_span_seconds_including_overlap"
        and activity_share.get("is_wall_time_partition") is False
        and activity_share.get("overlapping_tool_spans_allowed") is True
        and activity_share.get("occupancy_ratio_may_exceed_one") is True
        and activity_share.get("subagent_tool_calls_tracked") is False
        and isinstance(activity_wall, (int, float))
        and not isinstance(activity_wall, bool)
        and float(activity_wall) > 0
        and isinstance(activity_occupancy, (int, float))
        and not isinstance(activity_occupancy, bool)
        and float(activity_occupancy) > 0
        and math.isclose(
            float(activity_occupancy),
            float(activity_share["classified_seconds"]) / float(activity_wall),
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
    )
    required_telemetry_artifacts = TEL.artifact_names(telemetry)
    if (
        telemetry.get("required") is not True
        or telemetry.get("driver") != "codex"
        or telemetry.get("billing_mode") != "subscription_notional"
        or telemetry.get("rounds_with_complete_usage") != len(round_rows)
        or not isinstance(telemetry.get("raw_event_count"), int)
        or telemetry.get("raw_event_count", 0) <= 0
        or not isinstance(telemetry.get("tool_call_count"), int)
        or telemetry.get("tool_call_count", 0) <= 0
        or telemetry.get("subagent_tool_calls_tracked") is not False
        or not _is_sha256(telemetry.get("preflight_sha256"))
        or not isinstance(telemetry_accounting, Mapping)
        or telemetry_accounting.get("available") is not True
        or telemetry_accounting.get("usage_complete") is not True
        or telemetry_accounting.get("billing_mode") != "subscription_notional"
        or telemetry_accounting.get("estimated_cost_usd") is not None
        or not isinstance(telemetry_accounting.get("subscription_notional_usd"), (int, float))
        or telemetry_accounting.get("subscription_notional_usd", 0) <= 0
        or not isinstance(telemetry_accounting.get("tokens_total"), int)
        or telemetry_accounting.get("tokens_total", 0) <= 0
        or telemetry_accounting.get("tool_calls") != telemetry.get("tool_call_count")
        or not isinstance(telemetry_reconciliation, Mapping)
        or telemetry_reconciliation.get("ok") is not True
        or (telemetry_reconciliation.get("raw_events") or {}).get("reconciled") is not True
        or (telemetry_reconciliation.get("token_ledger") or {}).get("all_match") is not True
        or not isinstance(activity_share, Mapping)
        or activity_share.get("basis") != "aet_native_codex_structured_tool_spans"
        or not activity_semantics_valid
        or not isinstance(telemetry_artifacts, Mapping)
        or set(telemetry_artifacts) != required_telemetry_artifacts
        or any(
            not isinstance(value, Mapping)
            or not isinstance(value.get("path"), str)
            or not value.get("path")
            or not _is_sha256(value.get("sha256"))
            for value in telemetry_artifacts.values()
        )
    ):
        raise StageGateError("performance candidate lacks complete raw/AET/cost/activity telemetry")
    if len(receipt_rows) != len(round_rows) or any(
        receipt.get("candidate_sha256") != round_row.get("candidate_sha256")
        for receipt, round_row in zip(receipt_rows, round_rows)
    ):
        raise StageGateError("per-round tuning GSIM feedback is not bound to the authored candidate bytes")
    audit = agent.get("audit")
    if (
        not isinstance(audit, Mapping)
        or not isinstance(audit.get("clean"), bool)
        or not isinstance(audit.get("hits"), list)
        or audit.get("broker_required") != BROKER_NAME
        or not isinstance(audit.get("broker_invocations"), list)
        or isinstance(audit.get("commands_seen"), bool)
        or not isinstance(audit.get("commands_seen"), int)
    ):
        raise StageGateError("performance candidate lacks structured agent audit evidence")
    if (
        admission.get("evaluation_performed_by_stage") is not False
        or admission.get("development_feedback_performed_by_stage") is not True
        or admission.get("success_declared_by_stage") is not False
        or admission.get("consumer") not in ({MEASUREMENT_CONSUMER} | LEGACY_MEASUREMENT_CONSUMERS)
    ):
        raise StageGateError("performance authoring stage crossed the evaluation boundary")
    _validate_audit_requalification(document)
    if require_consumable:
        if (
            admission.get("consumable") is not True
            or document.get("state") != "sealed"
            or rounds_requested != len(round_rows)
            or audit.get("clean") is not True
            or audit.get("hits") != []
            or audit.get("commands_seen", 0) <= 0
            or any(
                row["audit"].get("commands_seen", 0) <= 0 or row["audit"].get("clean") is not True for row in round_rows
            )
            or delta.get("execution_relevant_changed_file_count", 0) <= 0
            or broker.get("all_required_succeeded") is not True
            or any(row.get("feedback_successes", 0) < 1 for row in receipt_rows)
            or len(receipt_rows) != rounds_requested
            or any(row.get("agent_exit_code") not in (0, ROUND_DEADLINE_EXIT) for row in round_rows)
        ):
            raise StageGateError(f"performance candidate is not consumable: {admission.get('refusal')}")
    return dict(document)
