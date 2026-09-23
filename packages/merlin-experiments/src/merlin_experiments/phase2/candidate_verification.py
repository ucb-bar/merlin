"""Full candidate evidence verification and immutable audit-policy requalification."""

from __future__ import annotations

import copy
import inspect
import json
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.benchharness import hash_tree
from merlin.common import source_membership
from merlin.common.access import audit_token_in
from merlin.common.digest import sha256_bytes as _sha256
from merlin.common.paths import module_source_path
from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.sandbox.answer_surfaces import answer_surfaces, audit_tokens
from merlin.targetgen.target_experiment import TargetExperiment

from . import agent_workspace as AW
from . import campaign as PC
from . import candidate_record as RECORD
from . import contracts as CONTRACTS
from . import corpus as CORPUS
from . import corpus_feedback as CF
from . import gsim_gate as GATE
from . import telemetry as TEL
from .broker import actions_from_registry_contract
from .broker_evidence import _is_sha256
from .contracts import StageGateError
from .contracts import canonical_json as _canonical_json
from .contracts import sha256_file as _sha256_file
from .functional_inputs import _verify_private_functional_provenance, verify_functional_host_lane_snapshot
from .transcript_audit import audit_codex_transcript


def _round_audit_candidate(row: Mapping[str, Any]) -> Path:
    """Recover and verify the exact candidate tree against which a round was audited."""
    audit = row.get("audit")
    path = Path(str(audit.get("candidate") or "")) if isinstance(audit, Mapping) else Path("")
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_dir()
        or not _is_sha256(row.get("candidate_sha256"))
        or hash_tree(path)["sha256"] != row["candidate_sha256"]
    ):
        raise StageGateError("audit requalification round candidate bytes are absent or changed")
    return path


def _recomputed_candidate_audits(
    document: Mapping[str, Any],
    target_experiment: TargetExperiment,
    *,
    audit_token_set: Mapping[str, Sequence[str]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Replay the current audit policy on immutable transcript/candidate evidence."""
    agent = document["agent"]
    round_rows = agent["rounds"]
    final_candidate = _round_audit_candidate(round_rows[-1])
    actions = actions_from_registry_contract(document["broker"]["registry"], final_candidate)
    combined = audit_codex_transcript(
        Path(agent["transcript"]), target_experiment, final_candidate, actions, audit_token_set=audit_token_set
    )
    rounds: list[dict[str, Any]] = []
    for row in round_rows:
        round_candidate = _round_audit_candidate(row)
        round_actions = actions_from_registry_contract(document["broker"]["registry"], round_candidate)
        rounds.append(
            audit_codex_transcript(
                Path(row["transcript"]),
                target_experiment,
                round_candidate,
                round_actions,
                audit_token_set=audit_token_set,
            )
        )
    return combined, rounds


def _require_policy_only_audit_change(before: Mapping[str, Any], after: Mapping[str, Any], *, label: str) -> None:
    """Prove a replay removed hits without changing command or broker evidence."""
    if (
        after.get("clean") is not True
        or after.get("hits") != []
        or after.get("commands_seen") != before.get("commands_seen")
        or after.get("broker_required") != before.get("broker_required")
        or after.get("broker_invocations") != before.get("broker_invocations")
    ):
        raise StageGateError(f"{label} did not become clean solely through the corrected audit policy")


def _verify_audit_requalification(
    record_path: Path, document: Mapping[str, Any], target_experiment: TargetExperiment | None
) -> None:
    """Verify the immutable refused source, policy snapshot, and allowed-field-only rewrite."""
    requalification = document.get("audit_requalification")
    if requalification is None:
        return
    source_fact = requalification["source_record"]
    source_path = Path(source_fact["path"])
    if (
        source_path.resolve() == record_path.resolve()
        or source_path.is_symlink()
        or not source_path.is_file()
        or _sha256_file(source_path) != source_fact["sha256"]
        or source_path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    ):
        raise StageGateError("audit requalification source record is absent, mutable, or changed")
    source = verify_candidate_record(
        source_path, require_consumable=False, verify_authoring_tools=False, target_experiment=None
    )
    RECORD.require_audit_only_refusal(source)
    invariant = RECORD.audit_requalification_invariant_sha256(source)
    if (
        invariant != requalification["invariant_evidence_sha256"]
        or RECORD.audit_requalification_invariant_sha256(document) != invariant
    ):
        raise StageGateError("audit requalification changed evidence outside the audit decision")
    original = requalification["original_audits"]
    if (
        _sha256(_canonical_json(source["agent"]["audit"])) != original["combined_sha256"]
        or [_sha256(_canonical_json(row["audit"])) for row in source["agent"]["rounds"]] != original["round_sha256"]
        or source["agent"]["audit"]["hits"] != original["hits"]
    ):
        raise StageGateError("original audit provenance disagrees with its refused source")
    snapshots = requalification["policy_snapshots"]
    for row in snapshots:
        frozen = Path(row["frozen_path"])
        try:
            frozen.resolve().relative_to(record_path.parent.resolve())
        except ValueError:
            raise StageGateError("audit policy snapshot escapes its requalification directory") from None
        if requalification["schema_version"] == 4:
            parent = frozen.parent
            while True:
                if parent.is_symlink() or not parent.is_dir() or parent.stat().st_mode & 0o222:
                    raise StageGateError("audit policy snapshot ancestor is absent, linked, or mutable")
                if parent == record_path.parent:
                    break
                if parent == parent.parent:
                    raise StageGateError("audit policy snapshot ancestry escapes its requalification directory")
                parent = parent.parent
        if (
            frozen.is_symlink()
            or not frozen.is_file()
            or _sha256_file(frozen) != row["sha256"]
            or frozen.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        ):
            raise StageGateError("audit policy snapshot is absent, mutable, or changed")
    if _sha256(_canonical_json(snapshots)) != requalification["policy_set_sha256"]:
        raise StageGateError("audit policy snapshot set changed")
    if target_experiment is None:
        return
    if requalification["schema_version"] in (1, 2, 3):
        raise StageGateError(
            "historical audit requalification requires its original native audit owner; requalify anew"
        )
    audit_source = next(row for row in snapshots if row["role"] == "audit_implementation")
    live_audit = Path(inspect.getsourcefile(audit_codex_transcript) or "").resolve()
    if _sha256_file(live_audit) != audit_source["sha256"]:
        raise StageGateError("live audit implementation differs from the requalification policy snapshot")
    identity = requalification["phase2_source_identity"]
    if TEL._package_source_record() != identity:
        raise StageGateError("live performance package differs from the requalification source closure")
    discovery = next(row for row in snapshots if row["role"] == "python_source_membership")
    if _sha256_file(module_source_path("merlin.common.source_membership")) != discovery["sha256"]:
        raise StageGateError("live source membership implementation differs from requalification")
    if requalification["schema_version"] == 4:
        access_source = next(row for row in snapshots if row["role"] == "shared_access_policy")
        live_access = Path(inspect.getsourcefile(audit_token_in) or "").resolve()
        if _sha256_file(live_access) != access_source["sha256"]:
            raise StageGateError("live shared access policy differs from the requalification policy snapshot")
    combined, rounds = _recomputed_candidate_audits(
        document, target_experiment, audit_token_set=requalification["audit_token_set"]
    )
    if _canonical_json(combined) != _canonical_json(document["agent"]["audit"]) or [
        _canonical_json(row) for row in rounds
    ] != [_canonical_json(row["audit"]) for row in document["agent"]["rounds"]]:
        raise StageGateError("live corrected transcript audit disagrees with requalification record")


def requalify_audit_only_candidate(
    source_record: Path, output_record: Path, target_experiment: TargetExperiment
) -> Path:
    """Seal a new consumable record when a corrected audit clears an audit-only refusal.

    The refused record is never modified.  The new record embeds an allowed-field-only invariant,
    exact source identity, and read-only copies of the policy sources used for the replay.
    """
    source_path = Path(source_record)
    output_path = Path(output_record)
    if (
        source_path.is_symlink()
        or not source_path.is_file()
        or source_path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    ):
        raise StageGateError("audit requalification requires an immutable refused source record")
    if output_path.exists() or output_path.is_symlink() or output_path.parent.exists():
        raise StageGateError("audit requalification output must be in a fresh directory")
    source = verify_candidate_record(
        source_path, require_consumable=False, verify_authoring_tools=False, target_experiment=None
    )
    RECORD.require_audit_only_refusal(source)
    derived_tokens = audit_tokens(target_experiment)
    token_set = {key: list(values) for key, values in sorted(derived_tokens.items())}
    combined, rounds = _recomputed_candidate_audits(source, target_experiment, audit_token_set=token_set)
    _require_policy_only_audit_change(source["agent"]["audit"], combined, label="combined audit")
    for index, (before, after) in enumerate(zip(source["agent"]["rounds"], rounds)):
        _require_policy_only_audit_change(before["audit"], after, label=f"round {index} audit")

    answer_policy_path = Path(inspect.getsourcefile(answer_surfaces) or "").resolve()
    phase2_identity = TEL._package_source_record()
    phase2_members = source_membership.python_members(Path(phase2_identity["path"]), label="phase-2")
    policy_sources = (
        ("audit_implementation", Path(inspect.getsourcefile(audit_codex_transcript) or "").resolve()),
        ("answer_surface_policy", answer_policy_path),
        ("shared_access_policy", Path(inspect.getsourcefile(audit_token_in) or "").resolve()),
        ("python_source_membership", module_source_path("merlin.common.source_membership")),
        *((f"phase2:{name}", path) for name, path in phase2_members.items()),
    )
    policy_payloads: list[tuple[str, Path, bytes, str]] = []
    for role, source_policy in policy_sources:
        if source_policy.is_symlink() or not source_policy.is_file():
            raise StageGateError(f"audit policy source is absent or linked: {source_policy}")
        payload = source_policy.read_bytes()
        if role.startswith("phase2:") and _sha256(payload) != phase2_identity["members"][role.removeprefix("phase2:")]:
            raise StageGateError("performance package changed during audit requalification")
        policy_payloads.append((role, source_policy, payload, _sha256(payload)))
    # The replay above must be attributed to the same source bytes copied below.  A concurrent edit
    # cannot silently turn the snapshot into a description of a different policy.
    if any(source_policy.read_bytes() != payload for _role, source_policy, payload, _digest in policy_payloads):
        raise StageGateError("audit policy source changed during requalification")

    output_path.parent.mkdir(parents=True, exist_ok=False)
    policy_dir = output_path.parent / "policy"
    policy_dir.mkdir()
    snapshots: list[dict[str, str]] = []
    for role, source_policy, payload, digest in policy_payloads:
        frozen = (
            policy_dir / "phase2" / role.removeprefix("phase2:")
            if role.startswith("phase2:")
            else policy_dir / f"{role}{source_policy.suffix}"
        )
        frozen.parent.mkdir(parents=True, exist_ok=True)
        with frozen.open("xb") as stream:
            stream.write(payload)
        frozen.chmod(0o444)
        snapshots.append(
            {
                "role": role,
                "source_path": str(source_policy),
                "frozen_path": str(frozen.resolve()),
                "sha256": digest,
            }
        )

    rewritten = copy.deepcopy(source)
    rewritten["state"] = "sealed"
    rewritten["admission"]["consumable"] = True
    rewritten["admission"]["refusal"] = None
    rewritten["agent"]["audit"] = combined
    for row, audit in zip(rewritten["agent"]["rounds"], rounds):
        row["audit"] = audit
    rewritten["audit_requalification"] = {
        "schema_version": 4,
        "phase2_source_identity": phase2_identity,
        "kind": RECORD.AUDIT_REQUALIFICATION_KIND,
        "reason": RECORD.AUDIT_REQUALIFICATION_REASON,
        "source_record": {
            "path": str(source_path.resolve()),
            "sha256": _sha256_file(source_path),
        },
        "invariant_evidence_sha256": RECORD.audit_requalification_invariant_sha256(source),
        "original_audits": {
            "combined_sha256": _sha256(_canonical_json(source["agent"]["audit"])),
            "round_sha256": [_sha256(_canonical_json(row["audit"])) for row in source["agent"]["rounds"]],
            "hits": copy.deepcopy(source["agent"]["audit"]["hits"]),
        },
        "corrected_audits": {
            "combined_sha256": _sha256(_canonical_json(combined)),
            "round_sha256": [_sha256(_canonical_json(audit)) for audit in rounds],
        },
        "audit_token_set": token_set,
        "audit_token_set_sha256": _sha256(_canonical_json(token_set)),
        "policy_snapshots": snapshots,
        "policy_set_sha256": _sha256(_canonical_json(snapshots)),
    }
    with output_path.open("xb") as stream:
        stream.write(_canonical_json(rewritten))
    output_path.chmod(0o444)
    for directory in sorted(
        (path for path in policy_dir.rglob("*") if path.is_dir()), key=lambda path: len(path.parts), reverse=True
    ):
        directory.chmod(0o555)
    policy_dir.chmod(0o555)
    output_path.parent.chmod(0o555)
    verify_candidate_record(
        output_path, require_consumable=True, verify_authoring_tools=False, target_experiment=target_experiment
    )
    return output_path


def verify_candidate_record(
    path: Path,
    *,
    require_consumable: bool = True,
    verify_authoring_tools: bool = False,
    target_experiment: TargetExperiment | None = None,
) -> dict:
    """Re-hash immutable run artifacts; optionally require the live authoring tools to remain pinned."""
    raw_path = Path(path)
    if raw_path.is_symlink() or not raw_path.is_file():
        raise StageGateError(f"performance candidate record is absent or linked: {raw_path}")
    path = raw_path.resolve()
    document = RECORD.validate_candidate_record(
        json.loads(path.read_text(encoding="utf-8")), require_consumable=require_consumable
    )
    if verify_authoring_tools:
        for binary_path, expected, label in (
            (document["target"]["descriptor"], document["target"]["descriptor_sha256"], "target descriptor"),
            (document["agent"]["codex_binary"], document["agent"]["codex_binary_sha256"], "Codex binary"),
            (
                document["sandbox"]["outer_codex_control_plane"]["bwrap_binary"],
                document["sandbox"]["outer_codex_control_plane"]["bwrap_binary_sha256"],
                "bwrap binary",
            ),
        ):
            binary = Path(binary_path)
            if binary.is_symlink() or not binary.is_file() or _sha256_file(binary) != expected:
                raise StageGateError(f"{label} bytes do not match the performance candidate record")
    candidate = Path(document["candidate"]["path"])
    if candidate.is_symlink() or not candidate.is_dir():
        raise StageGateError("sealed performance candidate path is absent or linked")
    if hash_tree(candidate)["sha256"] != document["candidate"]["sha256"]:
        raise StageGateError("sealed performance candidate bytes do not match their record")
    for member in (candidate, *candidate.rglob("*")):
        if member.is_symlink():
            raise StageGateError(f"sealed performance candidate contains a symlink: {member}")
        if member.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise StageGateError(f"sealed performance candidate is writable: {member}")
    transcript = Path(document["agent"]["transcript"])
    if transcript.is_symlink() or not transcript.is_file():
        raise StageGateError("performance candidate transcript is absent or linked")
    if _sha256(transcript.read_bytes()) != document["agent"]["transcript_sha256"]:
        raise StageGateError("performance candidate transcript bytes do not match their record")
    # Requalified records replay below with the exact frozen audit-token set.  The paired runner
    # deliberately imports target libraries from its frozen source snapshot; using that ambient
    # policy here would make the same record change verdict with Python import order.
    if target_experiment is not None and document.get("audit_requalification") is None:
        recorded_actions = actions_from_registry_contract(document["broker"]["registry"], candidate)
        observed_audit = audit_codex_transcript(transcript, target_experiment, candidate, recorded_actions)
        recorded_audit = document["agent"]["audit"]
        for field in ("clean", "hits", "commands_seen", "broker_required"):
            if observed_audit.get(field) != recorded_audit.get(field):
                raise StageGateError(f"combined Codex transcript audit field {field!r} does not match its record")
    for row in document["agent"]["rounds"]:
        round_transcript = Path(str(row.get("transcript") or ""))
        if (
            round_transcript.is_symlink()
            or not round_transcript.is_file()
            or _sha256(round_transcript.read_bytes()) != row["transcript_sha256"]
        ):
            raise StageGateError(f"performance candidate round {row.get('round')} transcript bytes changed")
        for label, artifact in row["telemetry"]["artifacts"].items():
            evidence_path = Path(str(artifact.get("path") or ""))
            if (
                evidence_path.is_symlink()
                or not evidence_path.is_file()
                or _sha256_file(evidence_path) != artifact.get("sha256")
            ):
                raise StageGateError(f"performance candidate round {row.get('round')} {label} telemetry changed")
    for label, artifact in document["telemetry"]["artifacts"].items():
        evidence_path = Path(str(artifact.get("path") or ""))
        if (
            evidence_path.is_symlink()
            or not evidence_path.is_file()
            or _sha256_file(evidence_path) != artifact.get("sha256")
        ):
            raise StageGateError(f"performance candidate {label} telemetry changed")
    preflight_doc = json.loads(
        Path(document["telemetry"]["artifacts"]["preflight"]["path"]).read_text(encoding="utf-8")
    )
    if _sha256(_canonical_json(preflight_doc)) != document["telemetry"]["preflight_sha256"]:
        raise StageGateError("performance candidate telemetry preflight declaration changed")
    TEL.verify_price_evidence(document["telemetry"], document["agent"]["rounds"], preflight_doc)
    staged_prompt = Path(document["prompt"]["staged_path"])
    if (
        staged_prompt.is_symlink()
        or not staged_prompt.is_file()
        or _sha256(staged_prompt.read_bytes()) != document["prompt"]["sha256"]
    ):
        raise StageGateError("staged performance prompt bytes do not match their record")
    if _sha256(_canonical_json(document["prompt"]["facts"])) != document["prompt"]["facts_sha256"]:
        raise StageGateError("canonical performance prompt facts changed")
    renderer = Path(document["prompt"]["renderer_path"])
    if (
        renderer.is_symlink()
        or not renderer.is_file()
        or _sha256_file(renderer) != document["prompt"]["renderer_sha256"]
    ):
        raise StageGateError("performance prompt renderer bytes changed")
    corpus = document["performance_corpus"]
    frozen_manifest = Path(corpus["manifest"])
    if (
        frozen_manifest.is_symlink()
        or not frozen_manifest.is_file()
        or _sha256(frozen_manifest.read_bytes()) != corpus["manifest_sha256"]
    ):
        raise StageGateError("frozen performance manifest bytes do not match their record")
    frozen_root = Path(corpus["path"])
    try:
        frozen_loaded = CORPUS.load_frozen_performance_corpus(
            frozen_root,
            manifest_sha256=str(corpus["manifest_sha256"]),
            capsules_sha256=str(corpus["capsules_sha256"]),
            expected_target=str(document["target"]["name"]),
        )
    except PC.CampaignGateError as exc:
        raise StageGateError(f"frozen performance corpus verification failed: {exc}") from exc
    observed_formal_claim = RECORD.prepare_formal_claim(frozen_loaded.capsules, int(corpus["replicates"]))
    if _canonical_json(observed_formal_claim) != _canonical_json(corpus["formal_claim"]):
        raise StageGateError("frozen performance descriptors changed their formal claim preflight")
    agent_manifest = Path(corpus["agent_input_manifest"])
    if (
        agent_manifest.is_symlink()
        or not agent_manifest.is_file()
        or _sha256(agent_manifest.read_bytes()) != corpus["agent_input_manifest_sha256"]
    ):
        raise StageGateError("answer-free agent input manifest bytes do not match their record")
    AW.verify_answer_free_agent_inputs(
        AW.AgentInputSnapshot(
            Path(corpus["agent_input_path"]),
            agent_manifest,
            str(corpus["agent_input_manifest_sha256"]),
            str(corpus["agent_input_sha256"]),
            int(corpus["agent_input_files"]),
            int(corpus["agent_input_bytes"]),
        )
    )
    base = Path(document["base_functional"]["snapshot"])
    if (
        base.is_symlink()
        or not base.is_dir()
        or hash_tree(base)["sha256"] != document["base_functional"]["submission_sha256"]
    ):
        raise StageGateError("frozen functional base bytes do not match their record")
    bundle = document["base_functional"]["bundle_input_snapshot"]
    bundle_marker = Path(bundle["manifest"])
    if (
        bundle_marker.is_symlink()
        or not bundle_marker.is_file()
        or _sha256_file(bundle_marker) != bundle["manifest_sha256"]
    ):
        raise StageGateError("frozen functional grant marker bytes changed")
    if CONTRACTS.mapping_file(bundle_marker).get("version") != 4:
        raise StageGateError("verified execution requires bundle snapshot V4 ownership; create a newly frozen run")
    if CONTRACTS.mapping_file(bundle_marker).get("version") == 4:
        provenance = bundle.get("host_provenance")
        if not isinstance(provenance, Mapping):
            raise StageGateError("v3 candidate lacks host-owned functional provenance")
        verified = _verify_private_functional_provenance(provenance)
        if (
            BW.bundle_snapshot_root(verified["workspace"]) != Path(bundle["path"])
            or verified["marker"]["content_sha256"] != bundle["content_sha256"]
        ):
            raise StageGateError("v3 candidate names a different functional input snapshot")
        public_marker = Path(str(bundle.get("public_manifest") or ""))
        projection = CONTRACTS.mapping_file(public_marker)
        if _sha256_file(public_marker) != bundle.get("public_manifest_sha256") or projection.get(
            "content_sha256"
        ) != bundle.get("public_content_sha256"):
            raise StageGateError("functional public manifest projection bytes changed")
    verify_functional_host_lane_snapshot(document["base_functional"]["model_host_lane"])
    sentinel = document["base_functional"]["e2e_sentinel"]
    sentinel_source = Path(sentinel["frozen_source_path"])
    if (
        sentinel_source.is_symlink()
        or not sentinel_source.is_dir()
        or CONTRACTS.exact_tree_record(sentinel_source)["sha256"] != sentinel["capsule_sha256"]
    ):
        raise StageGateError("frozen full-model E2E sentinel bytes changed")
    receipt_manifest = Path(document["broker"]["receipt_manifest"])
    if (
        receipt_manifest.is_symlink()
        or not receipt_manifest.is_file()
        or _sha256_file(receipt_manifest) != document["broker"]["receipt_manifest_sha256"]
    ):
        raise StageGateError("broker receipt manifest bytes changed")
    receipt_document = json.loads(receipt_manifest.read_text(encoding="utf-8"))
    if (
        receipt_document.get("schema_version") != 1
        or receipt_document.get("rounds") != document["broker"]["round_receipts"]
    ):
        raise StageGateError("broker receipt manifest disagrees with the candidate record")
    for round_index, row in enumerate(document["broker"]["round_receipts"]):
        receipt_path = Path(str(row.get("path") or ""))
        if (
            receipt_path.is_symlink()
            or not receipt_path.is_file()
            or not _is_sha256(row.get("sha256"))
            or _sha256_file(receipt_path) != row["sha256"]
            or row.get("all_required_succeeded") is not True
        ):
            raise StageGateError("host-owned per-round broker receipt bytes changed")
        feedback_candidates: list[str] = []
        for feedback in row.get("feedback_receipts") or []:
            feedback_path = Path(str(feedback.get("path") or ""))
            feedback_sha = feedback.get("sha256")
            if (
                feedback_path.is_symlink()
                or not feedback_path.is_file()
                or not _is_sha256(feedback_sha)
                or _sha256(feedback_path.read_bytes()) != feedback_sha
            ):
                raise StageGateError("host-owned tuning GSIM feedback receipt bytes changed")
            validated = CF.validate_redacted_feedback(json.loads(feedback_path.read_text(encoding="utf-8")))
            if _sha256(_canonical_json(validated)) != feedback_sha:
                raise StageGateError("host-owned tuning GSIM feedback receipt is not canonical")
            feedback_candidates.append(str(validated["candidate_sha256"]))
        expected_candidate = document["agent"]["rounds"][round_index]["candidate_sha256"]
        if expected_candidate not in feedback_candidates:
            raise StageGateError("host-owned tuning GSIM feedback did not evaluate the recorded round candidate")
    feedback_certificate = document["development_feedback"]["certificate"]
    try:
        GATE.load_certificate(feedback_certificate["path"], expected_sha256=feedback_certificate["sha256"])
    except GATE.GsimGateError as exc:
        raise StageGateError(f"development GSIM certificate bytes changed: {exc}") from exc
    _verify_audit_requalification(path, document, target_experiment)
    return document


def verify_candidate_handoff(
    path: Path, *, verify_authoring_tools: bool = False, target_experiment: TargetExperiment | None = None
) -> RECORD.VerifiedCandidateHandoff:
    """Return the only normalized stage-to-measurement API after full byte verification."""
    document = verify_candidate_record(
        path,
        require_consumable=True,
        verify_authoring_tools=verify_authoring_tools,
        target_experiment=target_experiment,
    )
    base, candidate = document["base_functional"], document["candidate"]
    corpus, prompt = document["performance_corpus"], document["prompt"]
    broker, agent = document["broker"], document["agent"]
    bundle = base["bundle_input_snapshot"]
    preflight_path = Path(document["telemetry"]["artifacts"]["preflight"]["path"])
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    treatment_identity = TEL.treatment_identity(
        preflight,
        authoring_stage_sha256=str(document["prompt"]["renderer_sha256"]),
        codex_binary_sha256=str(agent["codex_binary_sha256"]),
    )
    source_sha256 = treatment_identity["telemetry_source_sha256"]
    if treatment_identity["requested_model"] != agent.get("model") or treatment_identity["resolved_model"] != agent.get(
        "resolved_model"
    ):
        raise StageGateError("performance candidate executable/source identities differ from telemetry preflight")
    agent_contract = {
        "model": str(agent.get("model") or ""),
        "resolved_model": str(agent.get("resolved_model") or ""),
        "effort": str(agent.get("effort") or ""),
        "wall_budget_seconds": agent.get("wall_budget_seconds"),
        "rounds": agent.get("rounds_requested"),
        "round_timeout_seconds": agent.get("round_timeout_seconds"),
        "max_tool_calls": agent.get("max_tool_calls"),
        "tool_timeout_seconds": agent.get("tool_timeout_seconds"),
        "smoke_replicates": corpus.get("smoke_replicates"),
        "measurement_replicates": corpus.get("replicates"),
        "functional_run_id": str(base.get("run_id") or ""),
        "functional_submission_sha256": str(base.get("submission_sha256") or ""),
        "telemetry_required": True,
        "telemetry_preflight_sha256": str(document["telemetry"]["preflight_sha256"]),
        "treatment_identity": treatment_identity,
    }
    return RECORD.VerifiedCandidateHandoff(
        record_path=Path(path).resolve(),
        record_sha256=_sha256_file(Path(path).resolve()),
        candidate_path=Path(candidate["path"]),
        candidate_sha256=str(candidate["sha256"]),
        candidate_initial_sha256=str(candidate["initial_sha256"]),
        functional_run_id=str(base["run_id"]),
        functional_submission_sha256=str(base["submission_sha256"]),
        functional_base_path=Path(base["snapshot"]),
        functional_bundle_snapshot_sha256=str(bundle["content_sha256"]),
        functional_bundle_manifest=Path(bundle["manifest"]),
        functional_bundle_manifest_sha256=str(bundle["manifest_sha256"]),
        target_descriptor=Path(document["target"]["descriptor"]),
        target_descriptor_sha256=str(document["target"]["descriptor_sha256"]),
        corpus_root=Path(corpus["path"]),
        corpus_manifest=Path(corpus["manifest"]),
        corpus_manifest_sha256=str(corpus["manifest_sha256"]),
        corpus_sha256=str(corpus["capsules_sha256"]),
        replicates=int(corpus["replicates"]),
        formal_replicate_identities=tuple(str(value) for value in corpus["formal_replicate_identities"]),
        formal_claim=copy.deepcopy(dict(corpus["formal_claim"])),
        smoke_replicates=int(corpus["smoke_replicates"]),
        expected_cells=tuple(dict(row) for row in corpus["expected_cells"]),
        families=tuple(dict(row) for row in corpus["families"]),
        host_lane=dict(base["model_host_lane"]),
        e2e_sentinel=dict(base["e2e_sentinel"]),
        prompt_sha256=str(prompt["sha256"]),
        prompt_facts_sha256=str(prompt["facts_sha256"]),
        prompt_path=Path(prompt["staged_path"]),
        transcript_path=Path(agent["transcript"]),
        transcript_sha256=str(agent["transcript_sha256"]),
        transcript_audit=dict(agent["audit"]),
        receipt_path=Path(broker["receipt_manifest"]),
        receipt_sha256=str(broker["receipt_manifest_sha256"]),
        required_actions=tuple(str(value) for value in broker["required_actions"]),
        tool_evidence={
            "registry_sha256": broker["registry_sha256"],
            "round_receipts": broker["round_receipts"],
            "tool_probe_results": document["sandbox"]["inner_execution_plane"]["tool_probe_results"],
            "tool_probe_recheck_results": document["sandbox"]["inner_execution_plane"]["tool_probe_recheck_results"],
        },
        sandbox_evidence={
            plane: {
                "network": document["sandbox"][plane]["network"],
                "clear_environment": document["sandbox"][plane]["clear_environment"],
                "policy_sha256": document["sandbox"][plane]["policy_sha256"],
            }
            for plane in ("outer_codex_control_plane", "inner_execution_plane")
        },
        telemetry_evidence=copy.deepcopy(dict(document["telemetry"])),
        codex_binary_sha256=str(agent["codex_binary_sha256"]),
        authoring_stage_sha256=str(document["prompt"]["renderer_sha256"]),
        telemetry_preflight_sha256=str(document["telemetry"]["preflight_sha256"]),
        telemetry_source_sha256=source_sha256,
        agent_contract=agent_contract,
    )
