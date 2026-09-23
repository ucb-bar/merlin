"""Audit requalification source roles are versioned, not inferred from live code."""

import copy
import hashlib
import inspect
import json
import subprocess
from pathlib import Path

import pytest
from merlin_experiments.phase2 import candidate_record as RECORD
from merlin_experiments.phase2 import candidate_verification as VERIFY
from merlin_experiments.phase2 import telemetry as TEL


def _digest(value):
    return hashlib.sha256((json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()


@pytest.fixture(autouse=True)
def no_native(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("native process forbidden"))


def _document(tmp_path, version=4):
    audit = {"clean": True, "hits": []}
    tokens = {name: ["private-fixture"] for name in ("answer", "grader", "oracle_subpath")}
    roles = {"audit_implementation", "answer_surface_policy"}
    if version >= 2:
        roles.add("shared_access_policy")
    if version == 3:
        roles.add("native_controller")
    identity = None
    if version == 4:
        source = tmp_path / "source"
        source.mkdir(exist_ok=True)
        for name in ("__init__.py", "candidate_record.py", "candidate_verification.py", "telemetry.py", "helper.py"):
            (source / name).write_text("# synthetic source\n")
        identity = TEL._package_source_record(source)
        roles.add("python_source_membership")
        roles.update("phase2:" + name for name in identity["members"])
    snapshots = [
        {
            "role": role,
            "source_path": str(tmp_path / "original" / role),
            "frozen_path": str(tmp_path / "policy" / role),
            "sha256": identity["members"][role.removeprefix("phase2:")] if role.startswith("phase2:") else "a" * 64,
        }
        for role in sorted(roles)
    ]
    evidence = {
        "schema_version": version,
        "kind": RECORD.AUDIT_REQUALIFICATION_KIND,
        "reason": RECORD.AUDIT_REQUALIFICATION_REASON,
        "source_record": {"path": str(tmp_path / "source.json"), "sha256": "a" * 64},
        "invariant_evidence_sha256": "b" * 64,
        "original_audits": {"combined_sha256": "c" * 64, "round_sha256": ["c" * 64], "hits": ["old"]},
        "corrected_audits": {"combined_sha256": _digest(audit), "round_sha256": [_digest(audit)]},
        "policy_snapshots": snapshots,
        "policy_set_sha256": _digest(snapshots),
        "audit_token_set": tokens,
        "audit_token_set_sha256": _digest(tokens),
    }
    if identity is not None:
        evidence["phase2_source_identity"] = identity
    return {"agent": {"audit": audit, "rounds": [{"audit": audit}]}, "audit_requalification": evidence}


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_historical_and_current_roles_decode_without_live_source(tmp_path, version):
    document = _document(tmp_path, version)
    before = copy.deepcopy(document)
    RECORD._validate_audit_requalification(document)
    assert document == before


@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate", "old_role", "unknown_version", "bool_version", "member_missing", "member_digest"],
)
def test_current_role_and_closure_corruption_refuses(tmp_path, mutation):
    document = _document(tmp_path)
    evidence = document["audit_requalification"]
    rows = evidence["policy_snapshots"]
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "old_role":
        rows[0]["role"] = "native_controller"
    elif mutation == "unknown_version":
        evidence["schema_version"] = 5
    elif mutation == "bool_version":
        evidence["schema_version"] = True
    elif mutation == "member_missing":
        evidence["phase2_source_identity"]["members"].pop("candidate_verification.py")
        evidence["phase2_source_identity"]["sha256"] = _digest(evidence["phase2_source_identity"]["members"])
    else:
        next(row for row in rows if row["role"] == "phase2:helper.py")["sha256"] = "f" * 64
    evidence["policy_set_sha256"] = _digest(rows)
    with pytest.raises(RECORD.StageGateError):
        RECORD._validate_audit_requalification(document)


@pytest.mark.parametrize("version", [1, 2, 3])
def test_historical_versions_do_not_acquire_current_owner_roles(tmp_path, version):
    document = _document(tmp_path, version)
    evidence = document["audit_requalification"]
    row = copy.deepcopy(evidence["policy_snapshots"][0])
    row["role"] = "phase2:candidate_verification.py"
    evidence["policy_snapshots"].append(row)
    evidence["policy_set_sha256"] = _digest(evidence["policy_snapshots"])
    with pytest.raises(RECORD.StageGateError):
        RECORD._validate_audit_requalification(document)


@pytest.mark.parametrize("change", ["added", "removed", "changed"])
def test_live_closure_checks_all_members_after_immutable_snapshot_verification(tmp_path, monkeypatch, change):
    document = _document(tmp_path)
    evidence = document["audit_requalification"]
    source = {"agent": {"audit": {"hits": ["old"]}, "rounds": [{"audit": {"hits": ["old"]}}]}}
    source_path = Path(evidence["source_record"]["path"])
    source_path.write_text(json.dumps(source))
    source_path.chmod(0o444)
    evidence["source_record"]["sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    evidence["original_audits"].update(
        combined_sha256=_digest(source["agent"]["audit"]), round_sha256=[_digest(source["agent"]["audit"])]
    )
    monkeypatch.setattr(VERIFY, "verify_candidate_record", lambda *a, **k: source)
    monkeypatch.setattr(RECORD, "require_audit_only_refusal", lambda source: None)
    monkeypatch.setattr(RECORD, "audit_requalification_invariant_sha256", lambda doc: "b" * 64)
    source_root = Path(evidence["phase2_source_identity"]["path"])
    inventory = TEL._package_source_record
    monkeypatch.setattr(TEL, "_package_source_record", lambda: inventory(source_root))
    owners = {
        "audit_implementation": Path(inspect.getsourcefile(VERIFY.audit_codex_transcript)),
        "shared_access_policy": Path(inspect.getsourcefile(VERIFY.audit_token_in)),
        "answer_surface_policy": Path(inspect.getsourcefile(VERIFY.answer_surfaces)),
        "python_source_membership": VERIFY.module_source_path("merlin.common.source_membership"),
    }
    for index, row in enumerate(evidence["policy_snapshots"]):
        owner = (
            source_root / row["role"].removeprefix("phase2:")
            if row["role"].startswith("phase2:")
            else owners[row["role"]]
        )
        payload = owner.read_bytes()
        frozen = tmp_path / "policy" / str(index)
        frozen.parent.mkdir(exist_ok=True)
        frozen.write_bytes(payload)
        frozen.chmod(0o444)
        row.update(source_path=str(owner), frozen_path=str(frozen), sha256=hashlib.sha256(payload).hexdigest())
    evidence["policy_set_sha256"] = _digest(evidence["policy_snapshots"])
    (tmp_path / "policy").chmod(0o555)
    tmp_path.chmod(0o555)
    observed = []

    def replay(doc, target, *, audit_token_set):
        observed.append(audit_token_set)
        return doc["agent"]["audit"], [doc["agent"]["audit"]]

    monkeypatch.setattr(VERIFY, "_recomputed_candidate_audits", replay)
    RECORD._validate_audit_requalification(document)
    VERIFY._verify_audit_requalification(tmp_path / "qualified.json", document, object())
    assert observed == [evidence["audit_token_set"]]
    if change == "removed":
        (source_root / "helper.py").unlink()
    else:
        (source_root / ("new.py" if change == "added" else "helper.py")).write_text("# changed\n")
    # Historical inspection uses the frozen copies, never upgrades from current source.
    VERIFY._verify_audit_requalification(tmp_path / "qualified.json", document, None)
    with pytest.raises(RECORD.StageGateError, match="live performance package differs"):
        VERIFY._verify_audit_requalification(tmp_path / "qualified.json", document, object())
    assert len(observed) == 1
