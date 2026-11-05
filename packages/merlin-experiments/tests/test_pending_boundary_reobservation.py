"""A newer identical native observation cannot erase an in-flight proof's source."""

from copy import deepcopy

import pytest
from merlin_experiments.phase1.feedback import promotion as P
from merlin_experiments.phase1.feedback import tier_boundary_join as J
from test_tier_boundary_join import ledger, observations


def pending_with_reobservation(tmp_path):
    left, right = observations()
    ws = ledger(tmp_path, left)
    state = P._tier_state(ws)
    P._slots(state, "A", "L3")[left["execution_digest"]]["loop_instrument"] = "b" * 64
    current = deepcopy(left)
    current["common"].update(compiler_sha256="9" * 64, submission_digest="new-source")
    current["result_sha256"] = "a" * 64
    loop = dict(status="pass", digest="new-source", instrument="b" * 64,
                execution_digest=left["execution_digest"], tier_boundary=current)
    P._record(state, "A", "L2", loop)
    P._save_tier_state(ws, state)
    row = dict(capsule="A", **{"pass": True}, barrier_tier="L3", barrier_status="pass",
               execution_digest=right["execution_digest"], tier_boundary=right)
    return ws, state, left, right, loop, row


def test_original_completion_survives_new_identical_loop_without_source_reattribution(tmp_path):
    ws, before, left, right, loop, row = pending_with_reobservation(tmp_path)
    original_row = deepcopy(row)
    assert P.record_cert(ws, {"per_capsule": [row]}, "L3", identity=left["execution_digest"],
                         source_identity_verified=True) == ["A=pass"]
    state = P._tier_state(ws)
    assert P._slots_ro(state, "A", "L2") == P._slots_ro(before, "A", "L2")
    certs = P._slots_ro(state, "A", "L3")
    assert certs[left["execution_digest"]]["status"] == "joined"
    cert = certs[right["execution_digest"]]
    assert cert["status"] == "pass" and cert["digest"] == left["common"]["submission_digest"]
    assert cert["boundary_join"] == J.join(left, right)
    assert cert["completion_loop_reobservation"]["current_submission_digest"] == "new-source"
    assert J.reusable_certificate(cert, loop, name="A", source="new-source", instrument="a" * 64)
    assert row == original_row
    assert P.record_cert(ws, {"per_capsule": [row]}, "L3", identity=left["execution_digest"],
                         source_identity_verified=True) == []


@pytest.mark.parametrize("damage", ["not_verified", "loop_failed", "missing_loop_judge", "loop_judge",
    "elf", "harness", "input", "object", "current_source", "pending_source", "no_boundary", "physical_failed"])
def test_changed_or_unproved_loop_does_not_resolve_original_job(tmp_path, damage):
    ws, state, left, _, _, row = pending_with_reobservation(tmp_path)
    loop = P._slots(state, "A", "L2")[left["execution_digest"]]
    pending = P._slots(state, "A", "L3")[left["execution_digest"]]
    if damage == "loop_failed":
        loop["status"] = "fail"
    elif damage == "missing_loop_judge":
        pending.pop("loop_instrument")
    elif damage == "loop_judge":
        loop["instrument"] = "c" * 64
    elif damage in {"elf", "harness"}:
        loop["tier_boundary"][damage + "_sha256"] = "f" * 64
    elif damage in {"input", "object"}:
        field = "effective_command_buffer_sha256" if damage == "input" else "kernel_object_sha256"
        loop["tier_boundary"]["common"][field] = "0" * 64
    elif damage == "current_source":
        loop["tier_boundary"]["common"]["submission_digest"] = "unbound-source"
    elif damage == "pending_source":
        pending["digest"] = "unbound-source"
    elif damage == "no_boundary":
        loop.pop("tier_boundary")
    elif damage == "physical_failed":
        P._record(state, "A", "L3", dict(status="fail", instrument="a" * 64,
                  execution_digest=row["execution_digest"], digest="new-source"))
    P._save_tier_state(ws, state)
    before = P._tier_state(ws)
    assert P.record_cert(ws, {"per_capsule": [row]}, "L3", identity=left["execution_digest"],
                         source_identity_verified=damage != "not_verified") == []
    assert P._tier_state(ws) == before
