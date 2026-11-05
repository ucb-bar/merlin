"""Recurrent profile frames must bind and reconcile with every invocation."""

import pytest

from merlin.model_check.chain_group_profile import reconcile, separate


def frame(program, step, *, dropped=0):
    return (f"CHAIN_PROFILE_BEGIN v1 {program} {step}\nGROUP_ID 0 kernel\n"
            f"GROUP 0 5\nGAP 0 2\nGAP 1 3\nMETRIC group_calls 1\n"
            f"METRIC group_calls_dropped {dropped}\nCHAIN_PROFILE_END v1 {program} {step}\n")


def test_multiple_steps_are_separate_and_reconcile_without_output_transport():
    text = "CHAIN_STEP v1 0 0 10 0\n" + frame(0, 0) + "CHAIN_STEP v1 0 1 10 0\n" + frame(0, 1, dropped=2)
    ordinary, profiles = separate(text)
    assert ordinary.splitlines() == ["CHAIN_STEP v1 0 0 10 0", "CHAIN_STEP v1 0 1 10 0"]
    schedule = [{"program": 0, "step": i} for i in range(2)]
    counters = [{"program": "stage", "cycles": 10} for _ in schedule]
    result = reconcile(profiles, schedule, counters, enabled=True)
    assert result[0]["group_cycles"] == 5 and result[0]["host_gap_cycles"] == 5
    assert result[1]["host_gap_cycles"] is None and not result[1]["complete"]
    with pytest.raises(ValueError, match="roster"):
        reconcile(profiles, schedule, counters, enabled=False)
    counters[1]["cycles"] = 11
    with pytest.raises(ValueError, match="reconcile"):
        reconcile(profiles, schedule, counters, enabled=True)


@pytest.mark.parametrize("text", [
    frame(0, 0) * 2,
    "CHAIN_PROFILE_BEGIN v1 0 0\n" + frame(0, 1),
    "CHAIN_PROFILE_BEGIN v1 0 0\nCHAIN_STEP v1 0 0 1 0\n",
    frame(0, 0).replace("CHAIN_PROFILE_END v1 0 0", "CHAIN_PROFILE_END v1 0 1"),
    frame(0, 0).split("CHAIN_PROFILE_END")[0],
])
def test_duplicate_nested_mismatched_and_truncated_frames_refuse(text):
    with pytest.raises(ValueError):
        separate(text)
