"""Fail-closed reporting boundary for the Arm4-only performance campaign."""
from __future__ import annotations

import hashlib
import json
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path


class ReportingGateError(RuntimeError):
    """A report would overstate, mix, or lose attribution for the measured campaign."""


Identity = tuple[str, str, str, str]

_SIMULATOR_POLICY = {
    "spike": {"tier": "L2", "purpose": "correctness_screen", "citable": False},
    "verilator": {"tier": "L3", "purpose": "performance_certification", "citable": True},
}
_COMPLETION_FIELDS = {
    "expected", "reported", "correct", "failed", "missing",
    "screen_expected", "screen_passed", "citable_expected", "citable_measured",
    "citable_passed", "complete",
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _read_json(path: Path, *, label: str) -> object:
    if path.is_symlink() or not path.is_file():
        raise ReportingGateError(f"required {label} is absent or linked: {path}")
    try:
        return json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ReportingGateError(f"{label} is unreadable at {path}: {exc}") from exc


def _mapping(path: Path, *, label: str = "Arm4 reporting artifact") -> dict:
    doc = _read_json(path, label=label)
    if not isinstance(doc, dict):
        raise ReportingGateError(f"{label} must be a mapping: {path}")
    return doc


def _load_sealed_campaign(run: Path) -> dict:
    digest_path = run / "campaign_manifest.digest.json"
    _require_read_only(digest_path, label="campaign-manifest digest record")
    record = _mapping(digest_path, label="campaign-manifest digest record")
    required = {"path", "sha256", "n_bytes", "verified", "immutable"}
    if (set(record) != required or record.get("path") != "campaign_manifest.json"
            or record.get("verified") is not True or record.get("immutable") is not True):
        raise ReportingGateError("campaign-manifest digest record is incomplete")
    document = _read_immutable_json(
        run / "campaign_manifest.json", record.get("sha256"),
        label="sealed campaign manifest", expected_n_bytes=record.get("n_bytes"))
    if not isinstance(document, dict):
        raise ReportingGateError("sealed campaign manifest must be a mapping")
    return document


def _require_read_only(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.exists():
        raise ReportingGateError(f"{label} is absent or linked: {path}")
    if stat.S_IMODE(path.stat().st_mode) & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise ReportingGateError(f"{label} is writable: {path}")


def _read_immutable_json(path: Path, expected_sha256: object, *, label: str,
                         expected_n_bytes: object | None = None) -> object:
    if not _is_sha256(expected_sha256):
        raise ReportingGateError(f"{label} has no valid lowercase SHA-256")
    _require_read_only(path, label=label)
    payload = path.read_bytes()
    if expected_n_bytes is not None:
        if (isinstance(expected_n_bytes, bool) or not isinstance(expected_n_bytes, int)
                or expected_n_bytes <= 0 or len(payload) != expected_n_bytes):
            raise ReportingGateError(f"{label} byte count does not match its digest record")
    observed = _sha256_bytes(payload)
    if observed != expected_sha256:
        raise ReportingGateError(
            f"{label} digest mismatch: observed {observed}, expected {expected_sha256}")
    try:
        return json.loads(payload)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ReportingGateError(f"{label} is not valid JSON: {path}") from exc


def _simple_component(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value or value in (".", ".."):
        raise ReportingGateError(f"{label} must be a simple non-empty path component")
    return value


def identity_tuple(raw: object, *, owner: str) -> Identity:
    """Read one exact family/capsule/simulator/replicate identity without inference."""
    if not isinstance(raw, Mapping):
        raise ReportingGateError(f"{owner} omits its exact identity mapping")
    values = tuple(
        _simple_component(raw.get(field), label=f"{owner} identity {field}")
        for field in ("family", "capsule", "simulator", "replicate")
    )
    if values[2] not in _SIMULATOR_POLICY:
        raise ReportingGateError(f"{owner} names unsupported simulator {values[2]!r}")
    return values  # type: ignore[return-value]


def expected_identities(campaign: Mapping) -> tuple[Identity, ...]:
    """Validate the exact, non-vacuous cell schedule recorded before measurement."""
    raw = campaign.get("expected_identities")
    if not isinstance(raw, list) or not raw:
        raise ReportingGateError("Arm4 campaign has no non-vacuous expected identity set")
    identities = tuple(identity_tuple(item, owner="expected cell") for item in raw)
    if len(identities) != len(set(identities)):
        raise ReportingGateError("Arm4 campaign repeats an expected cell identity")

    by_replica: dict[tuple[str, str, str], list[str]] = {}
    for family, capsule, simulator, replicate in identities:
        by_replica.setdefault((family, capsule, replicate), []).append(simulator)
    if any(simulators != ["spike", "verilator"] for simulators in by_replica.values()):
        raise ReportingGateError(
            "every Arm4 capsule replicate must schedule one Spike L2 screen followed by one "
            "Verilator L3 certification")
    return identities


def _candidate_handoff(campaign: Mapping):
    """Re-verify the sealed authoring record through its stable normalized API."""
    stage = campaign.get("candidate_stage")
    if (not isinstance(stage, Mapping) or not isinstance(stage.get("record_path"), str)
            or not stage["record_path"] or not _is_sha256(stage.get("record_sha256"))):
        raise ReportingGateError("campaign lacks the exact sealed candidate-stage attribution")
    record_path = Path(str(stage.get("record_path") or ""))
    _read_immutable_json(
        record_path, stage.get("record_sha256"), label="performance candidate record")
    try:
        import perf_agent_stage as agent_stage  # lazy: only Arm4 candidate reports need the stage
        import perf_campaign as campaign_gate
        handoff = agent_stage.verify_candidate_handoff(
            record_path, verify_authoring_tools=False)
        expected = campaign_gate.candidate_handoff_record(handoff)
    except Exception as exc:
        raise ReportingGateError(f"performance candidate evidence is not consumable: {exc}") from exc
    if dict(stage) != expected:
        raise ReportingGateError("campaign candidate-stage attribution disagrees with its record")
    tools = handoff.tool_evidence
    probes = tools["tool_probe_results"]
    if (not handoff.required_actions or not isinstance(probes, list) or not probes
            or tools["tool_probe_recheck_results"] != probes
            or any(not isinstance(row, Mapping) or row.get("returncode") != 0
                   for row in probes)):
        raise ReportingGateError("candidate-stage tool evidence is incomplete or changed")
    audit = handoff.transcript_audit
    if (audit.get("clean") is not True or audit.get("hits") != []
            or not isinstance(audit.get("commands_seen"), int)
            or audit["commands_seen"] <= 0):
        raise ReportingGateError("candidate-stage audit is not clean")
    if (handoff.target_descriptor.is_symlink() or not handoff.target_descriptor.is_file()
            or _sha256_bytes(handoff.target_descriptor.read_bytes()) !=
            handoff.target_descriptor_sha256):
        raise ReportingGateError("candidate target descriptor bytes changed")
    for plane in ("outer_codex_control_plane", "inner_execution_plane"):
        evidence = handoff.sandbox_evidence.get(plane)
        if (not isinstance(evidence, Mapping)
                or evidence.get("network") != "available_not_an_isolation_claim"
                or evidence.get("clear_environment") is not True
                or not _is_sha256(evidence.get("policy_sha256"))):
            raise ReportingGateError("candidate-stage sandbox evidence is incomplete")
    return handoff


def _validate_completion_shape(completion: object) -> dict:
    if not isinstance(completion, Mapping) or set(completion) != _COMPLETION_FIELDS:
        raise ReportingGateError(
            "Arm4 campaign completion metrics do not match the exact-cell schema")
    count_fields = _COMPLETION_FIELDS - {"complete"}
    if any(isinstance(completion[field], bool) or not isinstance(completion[field], int)
           or completion[field] < 0 for field in count_fields):
        raise ReportingGateError("Arm4 campaign completion counts are malformed")
    if completion.get("complete") is not True or completion.get("expected", 0) <= 0:
        raise ReportingGateError("Arm4 campaign completion is incomplete or vacuous")
    return dict(completion)


def _declared_analyzer_identity(preflight: object):
    """The analyzer the SEALED preflight's own declaration names.

    The reporting gate must not carry its own opinion about which family may make a claim: it checks
    that the campaign's boundary names the decision procedure that family's frozen acceptance block
    declares.  A declaration naming none, or naming one that is malformed, is refused here rather
    than compared against some other family's analyzer.
    """
    from merlin.perf import claim_reach

    declaration = preflight.get("declaration") if isinstance(preflight, Mapping) else None
    try:
        identity = claim_reach.analyzer_identity({"acceptance": declaration})
    except ValueError as exc:
        raise ReportingGateError(
            f"sealed claim preflight names an unusable analyzer: {exc}") from exc
    if identity is None:
        raise ReportingGateError(
            "sealed claim preflight declares no acceptance.analyzer, so its decision boundary "
            "names no decision procedure")
    return identity


def _full_model_cert_tier(sentinel: object, handoff: object) -> str:
    """Re-derive the ONE tier a whole-model admission could have been cited at, or refuse.

    A model capsule never enters the tier ladder: its tier block is synthesised from the model's own
    layer accounting and carries exactly one execution tier -- the LAST tier the capsule declared that
    the target's capability manifest counts as RTL. Demanding any other tier (this gate demanded
    ``L2`` for years) reads ``None`` on a flawless run.

    The arithmetic is redone from the campaign's own recorded inputs rather than from a manifest this
    host may not own for the measured target, and every input is cross-checked against the sealed
    staging handoff, so a runner that recorded a flattering answer is refused rather than believed.
    """
    if not isinstance(sentinel, Mapping):
        raise ReportingGateError("campaign has no full-model admission record")
    derivation = sentinel.get("tier_derivation")
    if not isinstance(derivation, Mapping):
        raise ReportingGateError(
            "full-model admission records no tier derivation, so the tier it was admitted at cannot "
            "be re-derived and must not be taken on faith")
    declared = [str(value) for value in (derivation.get("declared") or [])]
    rtl = {str(value) for value in (derivation.get("rtl_tiers") or [])}
    staged = [str(value) for value in (dict(getattr(handoff, "e2e_sentinel", {}) or {})
                                       .get("required_tiers") or [])]
    if not staged or declared != staged:
        raise ReportingGateError(
            f"full-model admission was graded against tiers {declared} but the sealed staging "
            f"handoff declares {staged}")
    citable = [tier for tier in declared if tier in rtl]
    if not citable:
        raise ReportingGateError(
            f"full-model admission declares no tier the measured target counts as RTL "
            f"(declared={declared}, rtl_tiers={sorted(rtl)}); no citable whole-model verdict exists")
    expected = citable[-1]
    if derivation.get("cert_tier") != expected:
        raise ReportingGateError(
            f"full-model admission cites tier {derivation.get('cert_tier')!r}, but its own recorded "
            f"declaration and RTL tier set derive {expected!r}")
    return expected


def validate_campaign(campaign: Mapping) -> tuple[dict, tuple[Identity, ...], object]:
    """Require completed measurements and one explicit smoke or formal claim boundary."""
    if (campaign.get("status") != "GO" or campaign.get("measurement_status") != "GO"
            or campaign.get("refusal") is not None):
        raise ReportingGateError("Arm4 measurement campaign is not GO")
    if campaign.get("approach") != "arm4":
        raise ReportingGateError("reporting accepts the Arm4-only campaign, not another approach")
    mode = campaign.get("experiment_mode")
    boundary = campaign.get("decision_boundary")
    if not isinstance(boundary, Mapping) or not isinstance(boundary.get("reason"), str):
        raise ReportingGateError("Arm4 campaign lacks an explicit claim-decision boundary")
    if mode == "measurement_smoke_only":
        if (campaign.get("claim_status") != "NOT_ESTABLISHED"
                or campaign.get("claim_launch_status") != "NOT_REQUESTED"
                or campaign.get("claim_launch_blocker") !=
                "measurement-smoke mode does not request a performance-claim decision"
                or campaign.get("claim_preflight") is not None
                or campaign.get("claim_decision") is not None
                # The module names the analyzer that was NOT invoked, and the runner resolves it
                # from the frozen corpus' own declaration, so it is a family-dependent name rather
                # than a literal.  What makes this a non-claim are the four exact fields around it.
                or not isinstance(boundary.get("module"), str)
                or not boundary.get("module")
                or boundary.get("identity_bridge") != "not_invoked_in_measurement_smoke"
                or boundary.get("promotion_integration") != "blocked"
                or boundary.get("promotion_status") != "BLOCKED"):
            raise ReportingGateError(
                "measurement smoke does not preserve its explicit non-claim boundary")
    elif mode == "formal_claim":
        preflight = campaign.get("claim_preflight")
        decision = campaign.get("claim_decision")
        status = campaign.get("claim_status")
        if (campaign.get("claim_launch_status") != "GO"
                or campaign.get("claim_launch_blocker") is not None
                or not isinstance(preflight, Mapping) or preflight.get("status") != "READY"
                or not isinstance(decision, Mapping)
                or status not in {"ESTABLISHED", "REFUTED"}
                or decision.get("status") != status
                or boundary.get("promotion_integration") != "integrated"
                or boundary.get("promotion_status") != (
                    "PROMOTED" if status == "ESTABLISHED" else "BLOCKED")):
            raise ReportingGateError(
                "formal claim campaign lacks an exact integrated claim decision")
        analyzer = _declared_analyzer_identity(preflight)
        if (boundary.get("module") != analyzer.module
                or not str(boundary.get("identity_bridge") or "").startswith(
                    analyzer.function + "(")):
            raise ReportingGateError(
                "formal claim boundary does not name the analyzer the frozen declaration declares")
    else:
        raise ReportingGateError("Arm4 campaign has an unsupported experiment mode")
    handoff = _candidate_handoff(campaign)

    run_id = campaign.get("functional_run_id")
    digest = campaign.get("functional_submission_sha256")
    if not isinstance(run_id, str) or not run_id or not _is_sha256(digest):
        raise ReportingGateError("Arm4 report lacks a functional run ID and submission SHA-256")
    candidate = campaign["candidate_stage"]
    if (campaign.get("snapshot_sha256") != candidate["candidate_sha256"]
            or Path(str(campaign.get("snapshot") or "")).resolve() !=
            Path(candidate["candidate_path"]).resolve()
            or candidate["candidate_initial_sha256"] != digest
            or handoff.functional_run_id != run_id
            or handoff.functional_submission_sha256 != digest):
        raise ReportingGateError(
            "measured candidate does not trace to the pinned functional submission")
    for phase in ("fork_before", "fork_after"):
        check = campaign.get(phase)
        if (not isinstance(check, Mapping) or check.get("ok") is not True
                or check.get("state") != "held"):
            raise ReportingGateError(f"functional invariants were not held at {phase}")
    for phase in ("candidate_before", "candidate_after"):
        check = campaign.get(phase)
        if (not isinstance(check, Mapping) or check.get("ok") is not True
                or check.get("state") != "held"
                or check.get("sha256") != candidate["candidate_sha256"]):
            raise ReportingGateError(f"sealed candidate invariants were not held at {phase}")

    host = campaign.get("model_host_lane_snapshot")
    try:
        import perf_campaign as campaign_gate
        _bundle, verified_host, _package = campaign_gate.verify_functional_host_lane_snapshot(host)
    except Exception as exc:
        raise ReportingGateError(f"functional host-lane evidence is invalid: {exc}") from exc
    if verified_host != host:
        raise ReportingGateError("functional host-lane evidence changed while being verified")
    if dict(host) != handoff.host_lane:
        raise ReportingGateError("campaign host lane differs from the verified candidate handoff")
    run_snapshot = host.get("run_snapshot")
    if (not isinstance(run_snapshot, Mapping)
            or run_snapshot.get("content_sha256") !=
            handoff.functional_bundle_snapshot_sha256):
        raise ReportingGateError("candidate host lane names a different functional bundle snapshot")
    sentinel = campaign.get("full_model_admission")
    # WHICH tier admits a whole model is RE-DERIVED here, never asserted as a literal. This gate used
    # to demand `required_tiers == ["L2", "L3"]` and `tier_status == {"L2": "pass", "L3": "pass"}`; a
    # whole-model capsule emits exactly ONE execution tier, so on any target whose RTL tiers begin
    # above L2 that pair was unsatisfiable and every campaign was an unconditional NO-GO.
    #
    # The runner records the INPUTS to its choice (the capsule's declaration and the tier set the
    # target's capability manifest counts as RTL) alongside the answer, so the arithmetic is redone
    # from recorded facts here rather than trusted -- and rather than re-reading a capability manifest
    # this host may not even own for the target the sealed campaign was measured on.
    cert_tier = _full_model_cert_tier(sentinel, handoff)
    if (not isinstance(sentinel, Mapping) or sentinel.get("kind") != "model"
            or sentinel.get("source") != "functional_bundle_input_snapshot_v2"
            or not _is_sha256(sentinel.get("source_sha256"))
            or isinstance(sentinel.get("n_files"), bool)
            or not isinstance(sentinel.get("n_files"), int) or sentinel["n_files"] <= 0
            or isinstance(sentinel.get("n_bytes"), bool)
            or not isinstance(sentinel.get("n_bytes"), int) or sentinel["n_bytes"] <= 0
            or set(sentinel.get("lanes_required") or []) != {"on_mesh", "scalar_rvv_lane"}
            or sentinel.get("required_tiers") != [cert_tier]
            or sentinel.get("tier_status") != {cert_tier: "pass"}
            or sentinel.get("grade_status") != "pass"
            or sentinel.get("numeric_status") != "pass"
            or sentinel.get("passed") is not True
            or sentinel.get("cycles_recorded") is not False
            or sentinel.get("role") != "correctness_admission_not_performance_claim"):
        raise ReportingGateError(
            "campaign lacks a passing frozen mixed-lane full-model admission sentinel at "
            f"{cert_tier}")
    staged_sentinel = handoff.e2e_sentinel
    if (sentinel.get("capsule") != staged_sentinel.get("capsule")
            or sentinel.get("source_sha256") != staged_sentinel.get("capsule_sha256")):
        raise ReportingGateError(
            "measured full-model admission differs from the verified candidate handoff")

    frozen_contract = campaign.get("frozen_contract")
    if (not isinstance(frozen_contract, Mapping)
            or frozen_contract.get("source") != "functional_bundle_input_snapshot_v2"
            or not _is_sha256(frozen_contract.get("sha256"))):
        raise ReportingGateError("campaign has no authenticated frozen measurement contract")
    try:
        contract_record = campaign_gate._exact_tree_record(
            Path(str(frozen_contract.get("path") or "")))
    except Exception as exc:
        raise ReportingGateError(f"frozen measurement contract is invalid: {exc}") from exc
    if contract_record["sha256"] != frozen_contract["sha256"]:
        raise ReportingGateError("frozen measurement contract digest changed")

    sandbox = campaign.get("sandbox")
    if (not isinstance(sandbox, Mapping) or sandbox.get("engine") != "bwrap"
            or sandbox.get("network") != "available"
            or sandbox.get("package_read_only") is not True
            or sandbox.get("answer_surface_coverage_gap") != []):
        raise ReportingGateError(
            "Arm4 campaign did not record an answer-closed, read-only package sandbox with "
            "network available")
    required = sandbox.get("required_tool_probes")
    probes = sandbox.get("tool_probe_results")
    if (not isinstance(required, list) or not required
            or any(not isinstance(label, str) or not label for label in required)
            or len(required) != len(set(required)) or not isinstance(probes, list)
            or len(probes) != len(required)
            or any(not isinstance(row, Mapping) or not isinstance(row.get("label"), str)
                   for row in probes)):
        raise ReportingGateError("Arm4 campaign has incomplete or vacuous tool-probe evidence")
    by_label = {row["label"]: row for row in probes}
    if (set(by_label) != set(required)
            or any(by_label[label].get("returncode") != 0 for label in required)):
        raise ReportingGateError("Arm4 campaign did not pass every enforced tool probe")

    identities = expected_identities(campaign)
    if mode == "formal_claim":
        expected_prompt_cells = handoff.expected_cells
        if campaign.get("claim_preflight") != handoff.formal_claim:
            raise ReportingGateError(
                "formal PK preflight differs from the verified candidate handoff")
        replicate_contract = handoff.formal_claim.get("declaration", {}).get("replicates", {})
        if (handoff.replicates != replicate_contract.get("exact_count")
                or list(handoff.formal_replicate_identities) !=
                replicate_contract.get("identities")):
            raise ReportingGateError(
                "formal replicate identities differ from the frozen acceptance contract")
    else:
        if (isinstance(handoff.smoke_replicates, bool)
                or not isinstance(handoff.smoke_replicates, int)
                or handoff.smoke_replicates <= 0
                or handoff.smoke_replicates >= handoff.replicates):
            raise ReportingGateError("candidate handoff has no bounded non-claim smoke replicate count")
        smoke_ids = {f"r{index:03d}" for index in range(handoff.smoke_replicates)}
        expected_prompt_cells = tuple(
            dict(row) for row in handoff.expected_cells if row.get("replicate") in smoke_ids)
    if tuple(dict(row) for row in campaign["expected_identities"]) != expected_prompt_cells:
        raise ReportingGateError("measurement cells differ from the verified candidate prompt")
    completion = _validate_completion_shape(campaign.get("completion"))
    if completion["expected"] != len(identities):
        raise ReportingGateError("completion expected count disagrees with expected identities")
    return completion, identities, handoff


def validate_claim_decision(campaign: Mapping, handoff, rows: Sequence[Mapping]) -> None:
    """Recompute the formal decision from verified frozen descriptors and sealed rows.

    Which procedure recomputes it is resolved from the sealed declaration's own
    ``acceptance.analyzer``, so this gate re-derives the same verdict for any family that declares
    one and refuses -- rather than silently recomputing a different family's arithmetic -- for one
    that does not.
    """
    if campaign.get("experiment_mode") == "measurement_smoke_only":
        return
    import importlib
    import inspect

    analyzer = _declared_analyzer_identity(campaign.get("claim_preflight"))

    def _replicates():
        return [str(value) for value in getattr(handoff, "formal_replicate_identities", ()) or ()]

    def _counters():
        from merlin.perf.hw_counters import counters_for_target
        from merlin.targetgen.target_experiment import load_target_experiment
        descriptor = getattr(handoff, "target_descriptor", None)
        if descriptor is None:
            return None
        derived = counters_for_target(load_target_experiment(descriptor).target)
        return derived.get("counters") if derived.get("status") == "derived" else None

    providers = {"replicates": _replicates, "counters": _counters}
    try:
        import perf_campaign as campaign_gate
        module = importlib.import_module(analyzer.module)
        corpus = campaign_gate.load_frozen_performance_corpus(
            handoff.corpus_root,
            manifest_sha256=handoff.corpus_manifest_sha256,
            capsules_sha256=handoff.corpus_sha256,
        )
        descriptors = [capsule.descriptor for capsule in corpus.capsules]
        names = sorted(name for name in dir(module)
                       if name.startswith("preflight_") and callable(getattr(module, name, None)))
        if len(names) != 1:
            raise ReportingGateError(
                f"analyzer module {analyzer.module!r} publishes {len(names)} preflight entry "
                "points; exactly one is required")
        preflight_entry = getattr(module, names[0])
        analyze_entry = getattr(module, analyzer.function)

        def _kwargs(entry):
            declared = inspect.signature(entry).parameters
            supplied = {}
            for name, provide in providers.items():
                if name not in declared:
                    continue
                value = provide()
                if not value:
                    raise ReportingGateError(
                        f"{analyzer.module}.{entry.__name__} declares a {name!r} parameter this "
                        "report cannot re-derive, so the sealed decision cannot be recomputed")
                supplied[name] = value
            return supplied

        preflight = preflight_entry(descriptors, **_kwargs(preflight_entry))
        decision = analyze_entry(descriptors, rows, **_kwargs(analyze_entry))
    except Exception as exc:
        raise ReportingGateError(f"formal claim evidence cannot be recomputed: {exc}") from exc
    if preflight != campaign.get("claim_preflight"):
        raise ReportingGateError("sealed preflight differs from frozen descriptor recomputation")
    if decision != campaign.get("claim_decision"):
        raise ReportingGateError("sealed claim decision differs from exact-row recomputation")
    status = decision.get("status")
    expected_promotion = "PROMOTED" if status == "ESTABLISHED" else "BLOCKED"
    if (status not in {"ESTABLISHED", "REFUTED"}
            or campaign.get("claim_status") != status
            or campaign["decision_boundary"].get("promotion_status") != expected_promotion):
        raise ReportingGateError("formal promotion does not match its recomputed decision")


def _exact_tree_record(root: Path) -> dict[str, int | str]:
    if root.is_symlink() or not root.is_dir():
        raise ReportingGateError(f"frozen workload capsule tree is absent or linked: {root}")
    _require_read_only(root, label="frozen workload root")
    digest = hashlib.sha256()
    n_files = n_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink() or (not path.is_dir() and not path.is_file()):
            raise ReportingGateError(f"frozen workload contains a link or special file: {path}")
        _require_read_only(path, label="frozen workload entry")
        if path.is_dir():
            continue
        payload = path.read_bytes()
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
        n_files += 1
        n_bytes += len(payload)
    if n_files <= 0 or n_bytes <= 0:
        raise ReportingGateError("frozen workload capsule tree is empty")
    return {"sha256": digest.hexdigest(), "n_files": n_files, "n_bytes": n_bytes}


def _pinned_path(run: Path, raw: object, expected: Path, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ReportingGateError(f"campaign does not pin its {label}")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = run / candidate
    if candidate.resolve() != expected.resolve():
        raise ReportingGateError(f"campaign {label} escapes its run-private canonical path")
    return expected


def validate_frozen_workload(run: Path, campaign: Mapping, identities: Sequence[Identity],
                             handoff) -> dict:
    """Verify the structured workload manifest and every frozen capsule byte it attributes."""
    candidate_root = handoff.corpus_root.resolve(strict=True)
    candidate_manifest = handoff.corpus_manifest.resolve(strict=True)
    if candidate_manifest != candidate_root / "performance_corpus_manifest.json":
        raise ReportingGateError("candidate-stage workload manifest escapes its frozen corpus")
    root = _pinned_path(run, campaign.get("workload_snapshot"), candidate_root,
                        label="frozen workload snapshot")
    manifest_path = _pinned_path(
        run, campaign.get("workload_manifest"), candidate_manifest,
        label="frozen workload manifest")
    if (campaign.get("workload_manifest_sha256") != handoff.corpus_manifest_sha256
            or campaign.get("workload_sha256") != handoff.corpus_sha256):
        raise ReportingGateError(
            "measurement workload digests disagree with the candidate-stage corpus")
    manifest = _read_immutable_json(
        manifest_path, campaign.get("workload_manifest_sha256"),
        label="frozen workload manifest")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ReportingGateError("frozen workload manifest schema is unsupported")
    workload_sha = campaign.get("workload_sha256")
    if not _is_sha256(workload_sha) or manifest.get("capsules_sha256") != workload_sha:
        raise ReportingGateError("frozen workload aggregate digest is absent or inconsistent")

    capsules = manifest.get("capsules")
    counts = manifest.get("counts")
    if not isinstance(capsules, list) or not capsules or not isinstance(counts, Mapping):
        raise ReportingGateError("frozen workload manifest has a vacuous capsule set")
    seen: set[tuple[str, str]] = set()
    seen_paths: set[Path] = set()
    for item in capsules:
        if not isinstance(item, Mapping):
            raise ReportingGateError("frozen workload manifest contains a malformed capsule row")
        family = _simple_component(item.get("family"), label="workload family")
        capsule = _simple_component(item.get("capsule"), label="workload capsule")
        key = (family, capsule)
        if key in seen:
            raise ReportingGateError(f"frozen workload repeats capsule identity {key}")
        seen.add(key)
        relative = item.get("snapshot_relative_path")
        if not isinstance(relative, str) or not relative:
            raise ReportingGateError(f"frozen workload capsule {key} has no snapshot path")
        capsule_path = root / relative
        try:
            resolved_capsule = capsule_path.resolve()
            resolved_capsule.relative_to((root / "capsules").resolve())
        except ValueError as exc:
            raise ReportingGateError(f"frozen workload capsule {key} escapes its snapshot") from exc
        if resolved_capsule.name != capsule or resolved_capsule in seen_paths:
            raise ReportingGateError(
                f"frozen workload capsule {key} does not own one unique named snapshot")
        seen_paths.add(resolved_capsule)
        performance = item.get("performance")
        if not isinstance(performance, Mapping) or performance.get("family") != family:
            raise ReportingGateError(
                f"frozen workload capsule {key} disagrees with its performance family")
        observed = _exact_tree_record(capsule_path)
        if (item.get("source_sha256") != item.get("snapshot_sha256")
                or observed["sha256"] != item.get("snapshot_sha256")
                or observed["n_files"] != item.get("n_files")
                or observed["n_bytes"] != item.get("n_bytes")):
            raise ReportingGateError(f"frozen workload capsule digest changed: {key}")

    expected_capsules = {(family, capsule) for family, capsule, _sim, _rep in identities}
    if seen != expected_capsules:
        raise ReportingGateError("frozen workload capsule set disagrees with expected identities")
    aggregate = _exact_tree_record(root / "capsules")
    if (aggregate["sha256"] != workload_sha or counts.get("capsules") != len(capsules)
            or counts.get("families") != len({family for family, _capsule in seen})
            or counts.get("files") != aggregate["n_files"]
            or counts.get("bytes") != aggregate["n_bytes"]):
        raise ReportingGateError("frozen workload aggregate bytes or counts changed")
    return manifest


def validate_rows(rows: Sequence[Mapping], expected: Sequence[Identity],
                  expected_counts: Mapping) -> dict:
    """Recompute every exact-cell metric and require the sealed row sequence to match."""
    if not isinstance(rows, list) or not rows:
        raise ReportingGateError("Arm4 perf_results must contain a non-vacuous row list")
    observed = tuple(identity_tuple(row.get("identity"), owner="result row")
                     if isinstance(row, Mapping) else identity_tuple(None, owner="result row")
                     for row in rows)
    if observed != tuple(expected):
        raise ReportingGateError("sealed result row sequence disagrees with expected identities")

    correct = failed = screen_passed = citable_measured = citable_passed = 0
    for identity, row in zip(observed, rows, strict=True):
        simulator = identity[2]
        policy = _SIMULATOR_POLICY[simulator]
        if row.get("approach") != "arm4" or any(row.get(key) != value
                                                for key, value in policy.items()):
            raise ReportingGateError(f"result {identity} violates its simulator policy")
        is_correct = (row.get("correct") is True and row.get("tier_status") == "pass"
                      and row.get("grade_status") == "pass"
                      and row.get("numeric_status") == "pass"
                      and not row.get("error") and not row.get("failure"))
        correct += int(is_correct)
        cycles = row.get("cycles")
        if simulator == "spike":
            if cycles is not None:
                raise ReportingGateError(f"Spike L2 result {identity} exposes non-citable cycles")
            screen_passed += int(is_correct)
            failed += int(not is_correct)
            continue
        has_cycles = isinstance(cycles, int) and not isinstance(cycles, bool) and cycles > 0
        citable_measured += int(has_cycles)
        citable_passed += int(is_correct and has_cycles)
        failed += int(not (is_correct and has_cycles))

    screen_expected = sum(identity[2] == "spike" for identity in expected)
    citable_expected = sum(identity[2] == "verilator" for identity in expected)
    counts = {
        "expected": len(expected),
        "reported": len(rows),
        "correct": correct,
        "failed": failed,
        "missing": len(expected) - len(rows),
        "screen_expected": screen_expected,
        "screen_passed": screen_passed,
        "citable_expected": citable_expected,
        "citable_measured": citable_measured,
        "citable_passed": citable_passed,
        "complete": (len(rows) == len(expected) and failed == 0 and screen_expected > 0
                     and citable_expected > 0 and screen_passed == screen_expected
                     and citable_passed == citable_expected),
    }
    if counts != dict(expected_counts):
        raise ReportingGateError(
            f"Arm4 perf_results completion {counts} disagrees with campaign "
            f"{dict(expected_counts)}")
    return counts


def _load_sealed_results(run: Path, campaign: Mapping) -> list[dict]:
    record = campaign.get("results")
    required = {"path", "sha256", "n_bytes", "records", "verified", "immutable",
                "digest_record_sha256"}
    if not isinstance(record, Mapping) or set(record) != required:
        raise ReportingGateError("campaign has no complete sealed-result digest record")
    if (record.get("path") != "perf_results.json" or record.get("verified") is not True
            or record.get("immutable") is not True):
        raise ReportingGateError("campaign does not identify verified immutable perf_results.json")
    rows = _read_immutable_json(
        run / "perf_results.json", record.get("sha256"), label="sealed perf_results.json",
        expected_n_bytes=record.get("n_bytes"))
    if not isinstance(rows, list) or record.get("records") != len(rows):
        raise ReportingGateError("sealed perf_results row count disagrees with its digest record")

    digest_record = _read_immutable_json(
        run / "perf_results.digest.json", record.get("digest_record_sha256"),
        label="perf_results digest record")
    expected_record = dict(record)
    expected_record.pop("digest_record_sha256")
    if digest_record != expected_record:
        raise ReportingGateError("perf_results digest-record contents disagree with the campaign")
    return rows


def load_reportable_run(run: Path) -> tuple[dict, list[dict], dict]:
    """Load only a sealed, attributable, exact-cell Arm4 measurement campaign."""
    run = Path(run)
    campaign = _load_sealed_campaign(run)
    counts, identities, handoff = validate_campaign(campaign)
    validate_frozen_workload(run, campaign, identities, handoff)
    rows = _load_sealed_results(run, campaign)
    validate_rows(rows, identities, counts)
    validate_claim_decision(campaign, handoff, rows)
    return dict(campaign), rows, counts


def refuse_legacy_cross_approach(run: Path, tool: str) -> None:
    """Prevent a legacy multi-arm renderer from fabricating claims from an Arm4 campaign."""
    manifest = Path(run) / "campaign_manifest.json"
    if not manifest.is_file():
        return
    campaign = _mapping(manifest, label="Arm4 campaign manifest")
    if campaign.get("approach") == "arm4":
        raise ReportingGateError(
            f"{tool} is a legacy cross-approach renderer and cannot consume this Arm4 campaign; "
            "use the Arm4-only report generated by gen_perf_report.py")
