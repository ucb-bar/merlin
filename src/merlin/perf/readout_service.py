"""Ordered readout requests, masked client packets and explicit service demand.

Adapters supply decoded geometry, selected widths and resource laws. This pure
consumer preserves repeated local reads and zero-mask beats; it never invents
partial-write coherence, memory response time or a device completion estimate.
"""

from collections.abc import Callable
from dataclasses import dataclass

from merlin.perf.capsule_roofline import compose_service_floor
from merlin.perf.channel_service import compose_transfer_service

_U64 = 1 << 64
_LIMIT = 100_000


def _unsigned(*values, positive=False):
    if any(type(v) is not int or not int(positive) <= v < _U64 for v in values):
        raise ValueError("explicit bounded unsigned geometry required")


@dataclass(frozen=True)
class WritePacket:
    address: int
    bytes: int
    useful_address: int
    useful_bytes: int
    beat_masks: tuple[int, ...]

    @property
    def full(self):
        return self.address == self.useful_address and self.bytes == self.useful_bytes


def masked_write_packets(address, size, *, beat_bytes, maximum_bytes, alignment_bytes=1):
    """Maximize useful bytes, retaining smaller power-of-two size on ties.

    This explicit packet-selection law is supplied by the original adapter.
    Every beat of the selected client envelope remains present, even mask=0.
    Masked bytes describe a client request, not physical read/modify/write work.
    """
    _unsigned(address)
    _unsigned(size, beat_bytes, maximum_bytes, alignment_bytes, positive=True)
    if (
        any(v & (v - 1) for v in (beat_bytes, maximum_bytes, alignment_bytes))
        or maximum_bytes > 4096
        or beat_bytes > 64
        or max(beat_bytes, alignment_bytes) > maximum_bytes
        or address + size > _U64
    ):
        raise ValueError("bounded power-of-two packet/beat/alignment geometry required")
    if size > maximum_bytes * 4096:
        raise ValueError("client packet byte budget exhausted")
    out = []
    while size:
        if len(out) == 4096:
            raise ValueError("client packet budget exhausted")
        chosen, useful = 0, 0
        packet_size = max(beat_bytes, alignment_bytes)
        while packet_size <= maximum_bytes:
            take = min(packet_size - address % packet_size, size)
            if take > useful:
                chosen, useful = packet_size, take
            packet_size *= 2
        base = address // chosen * chosen
        masks = tuple(
            sum(1 << lane for lane in range(beat_bytes) if address <= base + b * beat_bytes + lane < address + useful)
            for b in range(chosen // beat_bytes)
        )
        out.append(WritePacket(base, chosen, address, useful, masks))
        address += useful
        size -= useful
    return tuple(out)


@dataclass(frozen=True)
class ReadoutRequest:
    local_row: int | None
    columns: int
    output_address: int
    output_bytes: int
    accumulate: bool
    store: bool
    reduce: bool = False


def rectangular_readout_requests(*, local_row, rows, columns, output_address, output_stride, output_element_bytes):
    """One local vector and one writer request per original nonpooled row."""
    _unsigned(local_row, output_address)
    _unsigned(rows, columns, output_stride, output_element_bytes, positive=True)
    if (
        rows > _LIMIT
        or local_row + rows >= _U64
        or output_address + (rows - 1) * output_stride + columns * output_element_bytes > _U64
    ):
        raise ValueError("readout request bounds exceeded")
    return tuple(
        ReadoutRequest(
            local_row + r, columns, output_address + r * output_stride, columns * output_element_bytes, False, True
        )
        for r in range(rows)
    )


def pooled_readout_requests(
    *,
    local_row,
    input_rows,
    input_columns,
    output_rows,
    output_columns,
    vector_columns,
    window,
    stride,
    padding,
    output_address,
    output_row_stride,
    output_pixel_stride,
    output_element_bytes,
):
    """Original output-row/output-column/window-row/window-column request order.

    An out-of-range local row is a garbage-address bypass, not an accumulator
    access or an inferred numerical padding value. The first window request
    overwrites the reduction register; later requests accumulate; only the last
    request starts output transactions. The original numerical law is separate.
    """
    _unsigned(local_row, output_address)
    _unsigned(
        input_rows,
        input_columns,
        output_rows,
        output_columns,
        vector_columns,
        output_row_stride,
        output_pixel_stride,
        output_element_bytes,
        positive=True,
    )
    if (
        type(window) is not tuple
        or type(stride) is not tuple
        or type(padding) is not tuple
        or len(window) != 2
        or len(stride) != 2
        or len(padding) != 2
    ):
        raise ValueError("explicit two-axis window, stride and leading padding required")
    _unsigned(*window, *stride, positive=True)
    _unsigned(*padding)
    count = output_rows * output_columns * window[0] * window[1]
    if (
        count > _LIMIT
        or local_row + input_rows * input_columns >= _U64
        or output_address
        + (output_rows - 1) * output_row_stride
        + (output_columns - 1) * output_pixel_stride
        + vector_columns * output_element_bytes
        > _U64
    ):
        raise ValueError("readout request bounds exceeded")
    out = []
    for y in range(output_rows):
        for x in range(output_columns):
            for wy in range(window[0]):
                for wx in range(window[1]):
                    iy, ix = y * stride[0] + wy - padding[0], x * stride[1] + wx - padding[1]
                    row = (
                        local_row + iy * input_columns + ix
                        if 0 <= iy < input_rows and 0 <= ix < input_columns
                        else None
                    )
                    out.append(
                        ReadoutRequest(
                            row,
                            vector_columns,
                            output_address + y * output_row_stride + x * output_pixel_stride,
                            vector_columns * output_element_bytes,
                            wy != 0 or wx != 0,
                            wy == window[0] - 1 and wx == window[1] - 1,
                            True,
                        )
                    )
    return tuple(out)


def compose_readout_service(
    requests,
    *,
    channel_contract,
    transfer_kind,
    beat_bytes,
    maximum_bytes,
    accumulator_element_bytes,
    bindings,
    pipeline_latency=None,
    pipeline_initiation_interval=None,
):
    """Compose exact counts with existing generic channel/resource service floors.

    Bindings name explicit resources for accumulator_rows, converted_vectors,
    and reduction_updates. Their ceilings belong to the selected source owner.
    Pipeline latency is reported independently from initiation interval; neither
    constitutes a complete command latency or a guessed per-command fee.
    """
    if (
        type(requests) is not tuple
        or not requests
        or len(requests) > _LIMIT
        or any(type(r) is not ReadoutRequest for r in requests)
        or type(bindings) is not dict
        or set(bindings) != {"accumulator_rows", "converted_vectors", "reduction_updates"}
    ):
        raise ValueError("bounded immutable original requests and explicit resource bindings required")
    _unsigned(accumulator_element_bytes, positive=True)
    for value in (pipeline_latency, pipeline_initiation_interval):
        if value is not None:
            _unsigned(value, positive=True)
    transfers, packets, reads, last_reads = [], [], 0, {}
    local_bytes = updates = converted = 0
    for i, r in enumerate(requests):
        _unsigned(r.columns, r.output_bytes, positive=True)
        _unsigned(r.output_address)
        if (
            type(r.accumulate) is not bool
            or type(r.store) is not bool
            or type(r.reduce) is not bool
            or r.accumulate
            and not r.reduce
        ):
            raise ValueError("actual reduction/store enable bits required")
        if r.local_row is not None:
            _unsigned(r.local_row)
            reads += 1
            converted += 1
            local_bytes += r.columns * accumulator_element_bytes
            last_reads[r.local_row] = i
        if r.reduce:
            updates += 1
        if r.store:
            row = masked_write_packets(
                r.output_address, r.output_bytes, beat_bytes=beat_bytes, maximum_bytes=maximum_bytes
            )
            if len(packets) + len(row) > _LIMIT:
                raise ValueError("aggregate client packet budget exhausted")
            packets.extend(row)
            transfers.append(
                dict(
                    id=f"output:{i}",
                    kind=transfer_kind,
                    packets=[
                        dict(payload_bytes=p.useful_bytes, packet_bytes=p.bytes, data_beats=len(p.beat_masks))
                        for p in row
                    ],
                    provenance="original ordered readout store-enable request and selected packet law",
                )
            )
    channel = compose_transfer_service(channel_contract, transfers)
    demand = {}
    counts = dict(accumulator_rows=reads, converted_vectors=converted, reduction_updates=updates)
    for name, count in counts.items():
        resource = bindings[name]
        if not isinstance(resource, str) or resource not in channel_contract["resources"]:
            raise ValueError("readout service binding names an unselected resource")
        demand[resource] = demand.get(resource, 0) + count
    event = dict(
        id="readout-local-service",
        demand=demand,
        compulsory=False,
        provenance="original ordered local read/conversion/reduction requests; no occupancy inference",
    )
    unresolved = dict(
        channel["unresolved"],
        readout_completion=(
            "finite elastic backpressure, translation, partial-write downstream service and acknowledgements "
            "require a selected timeline law"
        ),
    )
    floor = compose_service_floor(
        channel_contract["resources"], [*channel["events"], event], scope="emitted_schedule", unresolved=unresolved
    )
    return dict(
        schema="merlin.perf.readout_service.v1",
        counts=dict(
            counts,
            local_read_bytes=local_bytes,
            bypass_requests=sum(r.local_row is None for r in requests),
            reduction_accumulates=sum(r.accumulate for r in requests),
            writer_requests=len(transfers),
            client_packets=len(packets),
            client_payload_bytes=sum(p.useful_bytes for p in packets),
            client_envelope_bytes=sum(p.bytes for p in packets),
            client_beats=sum(len(p.beat_masks) for p in packets),
            masked_packets=sum(not p.full for p in packets),
            zero_mask_beats=sum(mask == 0 for p in packets for mask in p.beat_masks),
        ),
        last_read_request_by_row=last_reads,
        packets=tuple(packets),
        channel_service=channel,
        floor=floor,
        pipeline=dict(
            latency_cycles=pipeline_latency,
            initiation_interval_cycles=pipeline_initiation_interval,
            meaning="latency and issue interval are distinct source laws; neither is command completion",
        ),
        estimated_cycles=None,
        physical_traffic_bytes=None,
        occupancy=None,
        completion_chains=[],
    )


def elastic_pipeline_service(arrivals, *, stages, ready: Callable, max_cycles=1_000_000):
    """Clock an explicitly selected registered elastic pipeline under actual ready.

    Pending input remains stable while full; backpressure propagates through
    occupied stages. This generic service law excludes memory, arithmetic and
    target controller behavior. The caller supplies an explicit ready trace.
    """
    if (
        type(arrivals) is not tuple
        or not arrivals
        or len(arrivals) > _LIMIT
        or type(stages) is not int
        or not 1 <= stages <= 256
        or not callable(ready)
        or type(max_cycles) is not int
        or not 1 <= max_cycles <= 1_000_000
    ):
        raise ValueError("bounded original arrivals, pipeline stages and explicit downstream readiness required")
    _unsigned(*arrivals)
    if tuple(sorted(arrivals)) != arrivals:
        raise ValueError("ordered input arrivals required")
    slots = [None] * stages
    accepted, completed = [None] * len(arrivals), [None] * len(arrivals)
    next_input = 0
    peak = 0
    for t in range(max_cycles):
        allow = ready(t, slots[-1])
        if type(allow) is not bool:
            raise ValueError("unknown downstream ready cannot become free service")
        stalled = [False] * stages
        stalled[-1] = slots[-1] is not None and not allow
        for i in range(stages - 2, -1, -1):
            stalled[i] = slots[i] is not None and stalled[i + 1]
        old = slots[:]
        if old[-1] is not None and allow:
            completed[old[-1]] = t
        for i in range(stages - 1, 0, -1):
            if not stalled[i]:
                slots[i] = old[i - 1]
        if not stalled[0]:
            slots[0] = None
            if next_input < len(arrivals) and arrivals[next_input] <= t:
                slots[0] = next_input
                accepted[next_input] = t
                next_input += 1
        peak = max(peak, sum(v is not None for v in slots))
        if all(v is not None for v in completed):
            return dict(status="complete", accepted=tuple(accepted), completed=tuple(completed), peak_slots=peak)
    return dict(
        status="UNKNOWN",
        reason="bounded service traversal exhausted",
        accepted=tuple(accepted),
        completed=tuple(completed),
        peak_slots=peak,
    )
