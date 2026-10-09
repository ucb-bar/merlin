from dataclasses import replace

import pytest

from merlin.perf.component_cost import COMPLETE_STAGES
from merlin.perf.component_measurement_stream import BEGIN, END, UNKNOWN, RawMeasurementPlan, parse_measurement_stream


@pytest.fixture
def plan():
    return RawMeasurementPlan(COMPLETE_STAGES, 8192, 4096)


def frame():
    return b"\n".join(
        (
            BEGIN,
            *(
                f"{stage} {edge} {index} 0".encode()
                for index, (stage, edge) in enumerate(
                    (stage, edge) for stage in COMPLETE_STAGES for edge in ("begin", "end")
                )
            ),
            END,
            b"",
        )
    )


def test_complete_raw_roster_preserves_values_without_elapsed_cost(plan):
    observed = parse_measurement_stream(b"original output\n" + frame() + b"DONE\n", plan=plan)
    assert len(observed["events"]) == 22
    assert observed["stage_roster_observed"] == list(COMPLETE_STAGES)
    assert observed["cold"] is observed["warm"] is None
    assert observed["unknown"] == list(UNKNOWN)
    assert "elapsed" not in observed and "cycles" not in observed


def test_raw_wrap_and_changed_control_are_retained_not_normalized(plan):
    data = frame().replace(b"preparation begin 0 0", b"preparation begin 18446744073709551615 7")
    observed = parse_measurement_stream(data, plan=plan)
    assert observed["events"][0]["counter"] == (1 << 64) - 1
    assert observed["events"][0]["control"] == 7 and observed["events"][1]["control"] == 0


@pytest.mark.parametrize(
    "defect", ["missing", "moved", "duplicate", "truncated", "extra_marker", "suffix", "crlf", "unknown_stage"]
)
def test_incomplete_or_contradictory_original_events_refuse(plan, defect):
    lines = frame().splitlines(keepends=True)
    if defect == "missing":
        del lines[5]
    elif defect == "moved":
        lines[3:5], lines[5:7] = lines[5:7], lines[3:5]
    elif defect == "duplicate":
        lines[5] = lines[3]
    elif defect == "truncated":
        lines.pop()
    elif defect == "extra_marker":
        lines.append(BEGIN + b"\n")
    elif defect == "suffix":
        lines[-1] = END + b" ignored\n"
    elif defect == "crlf":
        lines[5] = lines[5].replace(b"\n", b"\r\n")
    else:
        lines[5] = lines[5].replace(b"packing", b"unselected")
    with pytest.raises(ValueError):
        parse_measurement_stream(b"".join(lines), plan=plan)


@pytest.mark.parametrize("token", [b"-1", b"01", b"+1", b"1.0", b"True", b"18446744073709551616"])
def test_closed_unsigned_values_refuse(plan, token):
    with pytest.raises(ValueError):
        parse_measurement_stream(
            frame().replace(b"preparation begin 0 0", b"preparation begin " + token + b" 0"), plan=plan
        )


def test_whole_console_budget_includes_bytes_outside_frame(plan):
    with pytest.raises(ValueError, match="whole console"):
        parse_measurement_stream(frame() + b"x" * 8192, plan=plan)
    with pytest.raises(ValueError, match="exceeded"):
        parse_measurement_stream(frame(), plan=replace(plan, max_frame_bytes=16))


@pytest.mark.parametrize(
    "changed", [{"stage_order": COMPLETE_STAGES[:-1]}, {"max_console_bytes": True}, {"max_frame_bytes": 0}]
)
def test_original_plan_cannot_drop_denominator_or_budget(plan, changed):
    with pytest.raises(ValueError):
        replace(plan, **changed).record()
