"""Source request/mask order and finite service do not invent completion prices."""

from dataclasses import replace

import pytest

from merlin.perf.channel_service import CONTRACT_SCHEMA
from merlin.perf.readout_service import (
    compose_readout_service,
    elastic_pipeline_service,
    masked_write_packets,
    pooled_readout_requests,
    rectangular_readout_requests,
)


def contract():
    names = ("accumulator", "convert", "reduce", "a", "d")
    return dict(
        schema=CONTRACT_SCHEMA,
        resources={
            n: dict(
                unit="handshakes", capacity_per_cycle_upper=1, provenance="selected original single-vector/beat port"
            )
            for n in names
        },
        transfer_kinds={
            "write": dict(
                data_beat_bytes=16,
                request={"resource": "a", "units_per_data_beat": 1},
                response={"resource": "d", "units_per_packet": 1},
            )
        },
        unresolved={"partial_write": "no selected downstream partial-write service/ack law"},
        provenance="original source packet and channel contract",
    )


def service(requests):
    return compose_readout_service(
        requests,
        channel_contract=contract(),
        transfer_kind="write",
        beat_bytes=16,
        maximum_bytes=64,
        accumulator_element_bytes=4,
        bindings=dict(accumulator_rows="accumulator", converted_vectors="convert", reduction_updates="reduce"),
        pipeline_latency=8,
        pipeline_initiation_interval=1,
    )


def test_masks_cover_every_payload_byte_once_and_never_skip_zero_mask_beats():
    for address in range(64):
        for size in (1, 3, 16, 17, 63, 64, 65, 129):
            packets = masked_write_packets(address, size, beat_bytes=16, maximum_bytes=64)
            addresses = [
                p.address + b * 16 + lane
                for p in packets
                for b, mask in enumerate(p.beat_masks)
                for lane in range(16)
                if mask >> lane & 1
            ]
            assert addresses == list(range(address, address + size))
            assert sum(p.useful_bytes for p in packets) == size
            assert all(len(p.beat_masks) * 16 == p.bytes for p in packets)
    # A wider packet wins ONLY because it covers more payload. Its first beat
    # has no payload but still traverses the client A channel.
    [p] = masked_write_packets(17, 16, beat_bytes=16, maximum_bytes=64)
    assert p.bytes == 64 and p.beat_masks == (0, 0xFFFE, 1, 0) and not p.full
    # Equal useful payload chooses the smaller transaction.
    [p] = masked_write_packets(0, 1, beat_bytes=16, maximum_bytes=64)
    assert p.bytes == 16 and p.beat_masks == (1,)


def test_scaled_tail_rows_preserve_stride_transactions_and_explicit_unknown_completion():
    requests = rectangular_readout_requests(
        local_row=10, rows=13, columns=16, output_address=0, output_stride=17, output_element_bytes=1
    )
    report = service(requests)
    assert [r.local_row for r in requests] == list(range(10, 23))
    assert [r.output_address for r in requests] == [17 * r for r in range(13)]
    assert report["counts"]["client_payload_bytes"] == 13 * 16
    assert report["counts"]["masked_packets"] > 0 and report["counts"]["zero_mask_beats"] > 0
    assert report["floor"]["resource_demands"]["a"] == report["counts"]["client_beats"]
    assert report["floor"]["resource_demands"]["d"] == report["counts"]["client_packets"]
    assert report["counts"]["local_read_bytes"] == 13 * 16 * 4
    assert report["pipeline"]["latency_cycles"] == 8 and report["pipeline"]["initiation_interval_cycles"] == 1
    assert report["estimated_cycles"] is report["physical_traffic_bytes"] is report["occupancy"] is None
    assert report["completion_chains"] == [] and report["floor"]["status"] == "partial"


def test_pool_reads_are_not_unique_rows_and_only_last_window_element_writes():
    requests = pooled_readout_requests(
        local_row=512,
        input_rows=3,
        input_columns=13,
        output_rows=1,
        output_columns=6,
        vector_columns=1,
        window=(3, 3),
        stride=(2, 2),
        padding=(0, 0),
        output_address=16,
        output_row_stride=102,
        output_pixel_stride=17,
        output_element_bytes=1,
    )
    assert len(requests) == 54
    assert [r.local_row for r in requests[:9]] == [512, 513, 514, 525, 526, 527, 538, 539, 540]
    assert requests[9].local_row == 514  # Same ACC row is actually read again.
    assert all(not requests[i].accumulate for i in range(0, 54, 9))
    assert [i for i, r in enumerate(requests) if r.store] == list(range(8, 54, 9))
    assert [r.output_address for r in requests if r.store] == [16 + 17 * x for x in range(6)]
    report = service(requests)
    assert report["counts"]["accumulator_rows"] == report["counts"]["converted_vectors"] == 54
    assert report["counts"]["reduction_updates"] == 54
    assert report["counts"]["reduction_accumulates"] == 48
    assert report["counts"]["writer_requests"] == 6 and report["counts"]["client_payload_bytes"] == 6
    assert report["last_read_request_by_row"][514] == 9


def test_padding_bypass_has_no_acc_read_or_conversion_but_keeps_reduction_request():
    requests = pooled_readout_requests(
        local_row=0,
        input_rows=1,
        input_columns=1,
        output_rows=1,
        output_columns=1,
        vector_columns=16,
        window=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        output_address=0,
        output_row_stride=16,
        output_pixel_stride=16,
        output_element_bytes=1,
    )
    assert [i for i, r in enumerate(requests) if r.local_row is not None] == [4]
    report = service(requests)
    assert report["counts"]["bypass_requests"] == 8
    assert report["counts"]["accumulator_rows"] == report["counts"]["converted_vectors"] == 1
    assert report["counts"]["reduction_updates"] == 9 and report["counts"]["writer_requests"] == 1
    # No tensor values or inferred numerical padding value enter this observer.
    assert not hasattr(requests[0], "value")


def test_registered_pipeline_latency_and_ii_are_distinct_under_backpressure():
    free = elastic_pipeline_service((0,) * 12, stages=8, ready=lambda t, value: True, max_cycles=100)
    assert free["accepted"] == tuple(range(12))
    assert free["completed"] == tuple(range(8, 20))
    blocked = elastic_pipeline_service((0,) * 12, stages=8, ready=lambda t, value: t >= 20, max_cycles=100)
    assert blocked["accepted"][:8] == tuple(range(8))
    assert blocked["accepted"][8:] == tuple(range(20, 24))
    assert blocked["completed"] == tuple(range(20, 32)) and blocked["peak_slots"] == 8
    unknown = elastic_pipeline_service((0,) * 12, stages=8, ready=lambda t, value: False, max_cycles=30)
    assert unknown["status"] == "UNKNOWN" and all(v is None for v in unknown["completed"])
    with pytest.raises(ValueError, match="unknown downstream"):
        elastic_pipeline_service((0,), stages=8, ready=lambda t, value: None, max_cycles=30)


@pytest.mark.parametrize("kwargs", [dict(beat_bytes=3), dict(maximum_bytes=8), dict(alignment_bytes=128)])
def test_invalid_or_unbounded_packet_contract_refuses(kwargs):
    parameters = dict(beat_bytes=16, maximum_bytes=64)
    parameters.update(kwargs)
    with pytest.raises(ValueError):
        masked_write_packets(0, 16, **parameters)


def test_malformed_requests_and_unknown_resource_do_not_become_zero_cost():
    requests = rectangular_readout_requests(
        local_row=0, rows=1, columns=1, output_address=0, output_stride=1, output_element_bytes=1
    )
    with pytest.raises(ValueError):
        service((replace(requests[0], accumulate=True),))
    bad = contract()
    bad["resources"]["convert"]["capacity_per_cycle_upper"] = None
    r = compose_readout_service(
        requests,
        channel_contract=bad,
        transfer_kind="write",
        beat_bytes=16,
        maximum_bytes=64,
        accumulator_element_bytes=4,
        bindings=dict(accumulator_rows="accumulator", converted_vectors="convert", reduction_updates="reduce"),
    )
    assert r["floor"]["resource_demands"]["convert"] == 1 and "resource:convert" in r["floor"]["unresolved"]
    assert r["pipeline"]["latency_cycles"] is None and r["estimated_cycles"] is None
