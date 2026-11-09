"""Worker-only aligned clock reporting refuses malformed and incomplete claims."""

import json
from dataclasses import asdict, replace

import pytest

from merlin.runtime.staged_dispatch import StagedBindings, StagedExecutionContract, StagedRoles, render_zephyr_staged
from merlin.runtime.staged_windows import analyze_worker_windows, main, write_worker_windows


def roles():
    return StagedRoles((0, 1, 2), 2, 2, 2, 2, (0, 1), (0, 1))


def observation():
    return {
        "abi_version": 2,
        "status": 0,
        "quiescent": 1,
        "sequence": 17,
        "global_clock_hart": 0,
        "global_begin": 100,
        "global_end": 180,
        "global_interval_valid": 1,
        "worker_clock_windows_enabled": 1,
        "worker_clock_windows_valid": 1,
        "workers": [
            {
                "hart": hart,
                "status": 0,
                "wait_status": 0,
                "begin_cycle": 1000,
                "end_cycle": 1200,
                "begin_instret": 2000,
                "end_instret": 2100,
                "arm_ready_tick": 100 + hart * 5,
                "compute_begin_tick": 110 + hart * 5,
                "compute_end_tick": 140 + hart * 5,
                "final_ack_tick": 180 + hart * 5,
                "clock_valid_mask": 15,
                "clock_status": 1,
                "clock_reader_status": 0,
            }
            for hart in (0, 1)
        ],
    }


def analyze(record):
    return analyze_worker_windows(
        record,
        roles(),
        expected_sequence=17,
        clock_domain="declared shared test clock",
        counter_semantics="controlled test counters",
    )


def test_aligned_raw_ticks_never_become_physical_busy_or_elapsed_seconds():
    report, trace = analyze(observation())
    assert report["pass"] and report["inclusive_stage_ticks"] == 80
    assert report["workers"][0]["compute_window_fraction"] == 30 / 80
    assert report["workers"][0]["pre_compute_wait_ticks"] == 10
    assert report["workers"][0]["post_compute_wait_ticks"] == 40
    assert report["clock_hz"] is None and report["elapsed_seconds"] is None
    assert report["physical_unit_busy"] is None and report["causal_stall_attribution"] is None
    assert trace["metadata"]["coordinate_is_elapsed_time"] is False
    assert trace["metadata"]["common_raw_tick_origin"] == 100
    first_worker_one = next(event for event in trace["traceEvents"] if event["tid"] == 1)
    assert first_worker_one["ts"] == 5  # one epoch; never rebase each worker separately
    assert {event["tid"] for event in trace["traceEvents"]} == {0, 1}


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update(abi_version=2.0),
        lambda r: r.update(sequence=18),
        lambda r: r.update(global_clock_hart=2),
        lambda r: r.update(global_begin=99),
        lambda r: r.update(worker_clock_windows_valid=0),
        lambda r: r.update(worker_clock_windows_enabled=0),
        lambda r: r.update(quiescent=False),
        lambda r: r["workers"].pop(),
        lambda r: r["workers"][1].update(hart=0),
        lambda r: r["workers"][1].update(hart=2),
        lambda r: r["workers"][0].update(status=True),
        lambda r: r["workers"][0].update(clock_valid_mask=16),
        lambda r: r["workers"][0].update(compute_begin_tick=110.0),
        lambda r: r["workers"][0].update(compute_end_tick=90),
        lambda r: r["workers"][0].update(clock_reader_status=42),
        lambda r: r["workers"][1].update(arm_ready_tick=111),
        lambda r: r["workers"][0].update(final_ack_tick=144),
    ],
)
def test_malformed_or_contradictory_observations_refuse(mutation):
    record = observation()
    mutation(record)
    with pytest.raises(ValueError):
        analyze(record)


def test_declared_clock_regression_preserves_raw_values_and_nulls_durations():
    record = observation()
    record["workers"][1].update(compute_end_tick=90, clock_status=3)
    record["worker_clock_windows_valid"] = 0
    report, trace = analyze(record)
    assert report["pass"] is False
    assert report["workers"][1]["compute_end_tick"] == 90
    assert report["workers"][1]["compute_window_ticks"] is None
    assert {event["tid"] for event in trace["traceEvents"]} == {0}


def test_clock_read_error_is_not_a_fabricated_zero_tick():
    record = observation()
    record["workers"][0].update(compute_begin_tick=None, clock_valid_mask=13, clock_status=2, clock_reader_status=-9)
    record.update(worker_clock_windows_valid=0, global_interval_valid=0)
    report, _ = analyze(record)
    worker = report["workers"][0]
    assert report["pass"] is False and worker["clock_status"] == "read_error"
    assert worker["compute_begin_tick"] is None and worker["compute_window_ticks"] is None
    assert report["inclusive_stage_ticks"] is None


def test_local_counter_regression_keeps_domains_separate():
    record = observation()
    record["workers"][0]["end_cycle"] = 999
    report, trace = analyze(record)
    assert report["pass"] is False
    assert report["workers"][0]["compute_cycles"] is None
    assert report["workers"][0]["compute_window_ticks"] == 30
    assert len(trace["traceEvents"]) == 6


def test_failed_peer_wait_retains_leader_clock_without_claiming_complete_timing():
    record = observation()
    record.update(status=-4, worker_clock_windows_valid=0)
    record["workers"][1].update(wait_status=-4, final_ack_tick=None, clock_status=4, clock_valid_mask=7)
    report, trace = analyze(record)
    assert report["pass"] is False and trace["traceEvents"] == []
    assert report["global_interval_valid"] is True
    assert report["workers"][0]["clock_window_valid"] is True
    assert report["inclusive_stage_ticks"] is None


def test_unquiesced_stage_can_publish_only_stable_error_metadata():
    record = observation()
    record.update(
        status=-5,
        quiescent=0,
        workers=[],
        global_interval_valid=0,
        worker_clock_windows_valid=0,
        global_begin=None,
        global_end=None,
    )
    report, trace = analyze(record)
    assert report["pass"] is False and report["workers"] == [] and trace["traceEvents"] == []
    record["workers"] = observation()["workers"]
    with pytest.raises(ValueError, match="stable failure metadata"):
        analyze(record)


def test_cli_byte_bindings_and_existing_output_preservation(tmp_path):
    source, policy, output = tmp_path / "record.json", tmp_path / "roles.json", tmp_path / "output"
    source.write_text(json.dumps(observation()))
    policy.write_text(json.dumps(asdict(roles())))
    args = [
        "--input",
        str(source),
        "--roles",
        str(policy),
        "--sequence",
        "17",
        "--clock-domain",
        "test clock",
        "--counter-semantics",
        "test counters",
        "--output",
        str(output),
    ]
    assert main(args) == 0
    report = json.loads((output / "report.json").read_text())
    assert str(source.resolve()) in report["input_sha256"]
    original = {file.name: file.read_bytes() for file in output.iterdir()}
    with pytest.raises(FileExistsError):
        write_worker_windows(output, report, {})
    assert {file.name: file.read_bytes() for file in output.iterdir()} == original
    source.write_text(source.read_text().replace('"sequence": 17', '"sequence": 17, "sequence": 17'))
    with pytest.raises(ValueError, match="duplicate JSON key"):
        main([*args[:-1], str(tmp_path / "refused")])
    assert not (tmp_path / "refused").exists()


def test_clock_windows_are_explicitly_selected_in_generated_harness():
    bindings = StagedBindings(
        "context", "initialize", "prepare", "compute", "finish", "hart", "ticks", "cycles", "retired", "completed"
    )
    contract = StagedExecutionContract(*["caller declaration"] * 5)
    kwargs = {"compute_capable_harts": (0, 1), "sequence": 1, "max_wait_polls": 1000}
    with pytest.raises(ValueError, match="checked reader"):
        render_zephyr_staged(roles(), contract, bindings, **kwargs, worker_clock_windows=True)
    source, metadata = render_zephyr_staged(
        roles(),
        contract,
        replace(bindings, shared_window_ticks="checked_ticks"),
        **kwargs,
        worker_clock_windows=True,
        shared_clock_domain="selected clock",
    )
    assert "extern int checked_ticks(void *, uint64_t *);" in source
    assert ".enable_worker_windows = 1" in source and ".shared_window_ticks = checked_ticks" in source
    assert metadata["runtime_abi_version"] == 2 and metadata["clock_hz"] is None
