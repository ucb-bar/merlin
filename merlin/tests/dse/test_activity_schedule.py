"""Driving examples for movement/compute overlap and exposed accelerator bubbles."""
from merlin.perf.activity_schedule import ActivityEvent, schedule_activity


def _serial_tiles():
    return schedule_activity((
        ActivityEvent("load0", "move", "movement", 40,
                      movement_bytes=4096, movement_commands=1),
        ActivityEvent("compute0", "array", "compute", 80, depends_on=("load0",)),
        ActivityEvent("load1", "move", "movement", 40, depends_on=("compute0",),
                      movement_bytes=4096, movement_commands=1),
        ActivityEvent("compute1", "array", "compute", 80, depends_on=("load1",)),
    ))


def _pipelined_tiles():
    return schedule_activity((
        ActivityEvent("load0", "move", "movement", 40,
                      movement_bytes=4096, movement_commands=1),
        ActivityEvent("compute0", "array", "compute", 80, depends_on=("load0",)),
        # The next buffer becomes available after load0; its transfer can occupy an independent
        # movement engine while compute0 consumes the other buffer.
        ActivityEvent("load1", "move", "movement", 40, depends_on=("load0",),
                      movement_bytes=4096, movement_commands=1),
        ActivityEvent("compute1", "array", "compute", 80,
                      depends_on=("compute0", "load1")),
    ))


def test_double_buffered_schedule_hides_movement_and_raises_compute_utilization() -> None:
    serial = _serial_tiles().occupancy()
    pipelined = _pipelined_tiles().occupancy()

    assert serial.total_cycles == 240
    assert serial.overlap_cycles == 0
    assert serial.compute_utilization == 160 / 240
    assert pipelined.total_cycles == 200
    assert pipelined.overlap_cycles == 40
    # Half the movement is hidden. The first load is a compulsory pipeline fill and remains visible;
    # calling the schedule 100% hidden would quietly erase that fixed term.
    assert pipelined.latency_hiding_efficiency == 0.5
    assert pipelined.compute_utilization == 160 / 200
    assert pipelined.movement_bytes == 8192
    assert pipelined.movement_commands == 2


def test_shared_serial_group_expresses_a_target_that_cannot_overlap_engines() -> None:
    timeline = schedule_activity((
        ActivityEvent("load", "move", "movement", 40, serial_group="shared"),
        ActivityEvent("compute", "array", "compute", 80,
                      depends_on=("load",), serial_group="shared"),
        ActivityEvent("next_load", "move", "movement", 40,
                      depends_on=("load",), serial_group="shared"),
    ))
    occupancy = timeline.occupancy()

    assert timeline.total_cycles == 160
    assert occupancy.overlap_cycles == 0


def test_encoding_conversion_is_visible_as_movement_not_free_metadata() -> None:
    timeline = schedule_activity((
        ActivityEvent("pack", "move", "encoding", 25,
                      movement_bytes=2048, movement_commands=1,
                      encoding_transition=True),
        ActivityEvent("compute", "array", "compute", 50, depends_on=("pack",)),
    ))
    occupancy = timeline.occupancy()

    assert occupancy.encoding_transitions == 1
    assert occupancy.movement_bytes == 2048
    assert occupancy.total_cycles == 75


def test_non_topological_dependency_is_refused_instead_of_assumed_ready() -> None:
    try:
        schedule_activity((ActivityEvent("compute", "array", "compute", 1,
                                         depends_on=("missing_load",)),))
    except ValueError as exc:
        assert "absent dependencies" in str(exc)
    else:
        raise AssertionError("an absent producer must not become a cycle-zero dependency")


def test_multiple_compute_engines_do_not_double_count_hidden_transfer_cycles() -> None:
    occupancy = schedule_activity((
        ActivityEvent("first", "compute0", "compute", 4),
        ActivityEvent("second", "compute1", "compute", 4),
        ActivityEvent("load", "transfer", "movement", 10),
    )).occupancy()
    assert occupancy.overlap_cycles == 4
    assert occupancy.overlap_available_cycles == 4
    assert occupancy.compute_utilization == 0.4


def test_multiple_transfer_engines_count_union_not_sum_of_overlaps() -> None:
    occupancy = schedule_activity((
        ActivityEvent("compute", "compute", "compute", 10),
        ActivityEvent("load", "load", "movement", 4),
        ActivityEvent("store", "store", "movement", 6),
    )).occupancy()
    assert occupancy.overlap_cycles == 6
    assert occupancy.overlap_available_cycles == 6
    assert occupancy.busy == {"compute": 10, "load": 4, "store": 6}
