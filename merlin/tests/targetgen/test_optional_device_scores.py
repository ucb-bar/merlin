"""Host correctness, device evidence and trusted spec credit remain distinct."""

import copy

import pytest

from merlin.targetgen.capsule_common import host_guards_pass, lane_entry_fields, score_lanes


def test_optional_host_correctness_and_device_evidence_remain_separate():
    rows = [
        {"status": "pass", "lane": "host", "scored": True},
        {"status": "pass", "lane": "device", "executed_instructions": 7, "scored": True},
        {"status": "fail", "lane": "device", "executed_instructions": 9, "scored": True},
        {"status": "pass", "lane": "device", "executed_instructions": 0, "scored": True},
        {"status": "pass", "lane": "host", "scored": False},
    ]
    score = {}
    scored = score_lanes(rows, score)
    assert len(scored) == 4
    assert score == {
        "host_guard_total": 1,
        "host_guard_pass": 1,
        "device_lane_scored_pass": 1,
        "device_evidence_count": 2,
        "spec_qualified_device_pass": 0,
        "device_unqualified_pass": 1,
        "host_lane_scored_pass": 1,
    }
    assert host_guards_pass(score)
    rows[-1]["lane"] = "device"
    score_lanes(rows, score)
    assert not host_guards_pass(score)


BINDINGS = "case source software_spec host_capabilities catalog target_ir object elf trace".split()
CHECKS = "private_answers selected_policy admission source schedule partial_sums raw_i32 ordered_f32 full_call"


def _qualified_row():
    # Synthetic trusted evaluator output, not an authenticated proof fixture.
    bindings = dict.fromkeys((f"{name}_sha256" for name in BINDINGS), "a" * 64)
    identity = {name: bindings[name] for name in ("case_sha256", "source_sha256", "elf_sha256", "trace_sha256")}
    receipt = dict(schema="merlin.spec_qualified_device.v1", status="qualified", policy="spec_int8")
    receipt.update(scope="selected_software_spec", reasons=[], bindings=bindings)
    receipt["checks"] = dict.fromkeys(CHECKS.split() + ["device_binding"], True)
    row = {"status": "pass", "lane": "device", "executed_instructions": 7, "scored": True}
    return {**row, "full_call_execution_identity": identity, "spec_qualified_device": receipt}


def test_qualified_subset_survives_rollup_without_changing_correctness_or_guards():
    row = _qualified_row()
    other = copy.deepcopy(row)
    other["spec_qualified_device"]["checks"]["raw_i32"] = False
    host = {**copy.deepcopy(row), "lane": "host"}
    guard = {**copy.deepcopy(row), "scored": False}
    score = {}
    scored = score_lanes([row, other, host, guard], score)
    assert len(scored) == 3
    assert score["device_lane_scored_pass"] == 2
    assert score["spec_qualified_device_pass"] == 1
    assert score["device_unqualified_pass"] == 1
    assert score["host_lane_scored_pass"] == 1
    assert not host_guards_pass(score)  # An unscored device row is not a host guard pass.
    projected = {"status": row["status"], **lane_entry_fields(row)}
    next_score = {}
    score_lanes([projected], next_score)
    assert next_score["spec_qualified_device_pass"] == 1


@pytest.mark.parametrize("field", ["case_sha256", "source_sha256", "elf_sha256", "trace_sha256"])
def test_qualification_cannot_move_to_another_execution(field):
    row = _qualified_row()
    row["full_call_execution_identity"][field] = "b" * 64
    score = {}
    score_lanes([row], score)
    assert score["device_lane_scored_pass"] == 1
    assert score["spec_qualified_device_pass"] == 0
    assert score["device_unqualified_pass"] == 1


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r["spec_qualified_device"]["checks"].update(raw_i32=1),
        lambda r: r["spec_qualified_device"]["checks"].pop("partial_sums"),
        lambda r: r["spec_qualified_device"]["bindings"].update(object_sha256="A" * 64),
        lambda r: r["spec_qualified_device"].update(policy="int8_w8a8"),
        lambda r: r["spec_qualified_device"].update(reasons=["host_int8_narrowing_undeclared"]),
        lambda r: r["spec_qualified_device"].update(expected_output=[123]),
    ],
)
def test_malformed_or_unqualified_receipt_never_gains_credit_after_projection(mutation):
    row = _qualified_row()
    mutation(row)
    projected = {"status": row["status"], **lane_entry_fields(row)}
    for candidate in (row, projected):
        score = {}
        score_lanes([candidate], score)
        assert score["device_lane_scored_pass"] == 1
        assert score["spec_qualified_device_pass"] == 0
    assert "expected_output" not in projected["spec_qualified_device"]


@pytest.mark.parametrize("field", ["schema", "status", "policy", "scope", "reasons", "bindings", "checks"])
def test_invalid_receipt_projection_cannot_expose_nested_private_values(field):
    row = _qualified_row()
    row["spec_qualified_device"]["status"] = "unqualified"
    row["spec_qualified_device"][field] = {"private_expected_output": [314159]}
    projected = lane_entry_fields(row)
    assert "private_expected_output" not in repr(projected)
    assert "314159" not in repr(projected)
    assert projected["spec_qualified_device"]["status"] == "unqualified"
