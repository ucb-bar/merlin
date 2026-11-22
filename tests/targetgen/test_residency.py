"""Pure byte residency rejects bad declarations before any target execution."""

import json
from dataclasses import replace
from hashlib import sha256

import pytest

from merlin.compiler.schedule.storage.residency import (
    ReservedInterval,
    ResidencyOverflow,
    ResidencyRequest,
    plan_residency,
)


def _request(name="weights", **changes):
    fields = dict(
        name=name, role="immutable_parameter", bytes=16, alignment=8,
        lifetime="application", load_protocol="once_at_load",
    )
    return ResidencyRequest(**(fields | changes))


def _plan(requests=(), **changes):
    options = dict(base=32, capacity=128, dma_base=0, dma_capacity=256)
    return plan_residency(requests, **(options | changes))


def test_table_preserves_distinct_roles_and_explicit_load_protocols():
    requests = [
        _request("weights"),
        _request("state", role="mutable_state"),
        _request("input", role="staging", lifetime="call", load_protocol="per_call"),
        _request("scratch", role="scratch", lifetime="phase:decode", load_protocol="per_phase"),
    ]
    plan = _plan(requests, reserved=[ReservedInterval("executable", 192, 32)])
    table = {row["name"]: row for row in plan.record()["entries"]}
    assert {row["role"] for row in table.values()} == {
        "immutable_parameter", "mutable_state", "staging", "scratch", "image",
    }
    for request in requests:
        row = table[request.name]
        assert set(row) == {"name", "role", "bytes", "alignment", "address", "lifetime", "load_protocol"}
        assert row["lifetime"] == request.lifetime
        assert row["load_protocol"] == request.load_protocol
    assert plan.entry("executable").load_protocol == "once_at_load"
    assert plan.reserved_names == ("executable",)


def test_first_fit_avoids_reserved_holes_and_straddling_reservations_without_aliasing():
    plan = _plan(
        [_request("a", bytes=9), _request("b", bytes=4), _request("c", bytes=8)],
        base=33, capacity=80,
        reserved=[ReservedInterval("image", 48, 8)],
    )
    assert [(plan.entry(name).address, plan.entry(name).bytes) for name in ("a", "b", "c")] == [
        (56, 9), (40, 4), (72, 8),
    ]
    assert plan.required_bytes == 47
    straddling = _plan([_request(bytes=8)], reserved=[ReservedInterval("image", 24, 16)])
    assert straddling.entry("weights").address == 40
    temporal = _plan([
        _request("a", lifetime="phase:one", load_protocol="per_phase"),
        _request("b", lifetime="phase:two", load_protocol="per_phase"),
    ])
    assert temporal.entry("a").end <= temporal.entry("b").address


def test_exact_fit_and_overflow_count_alignment_and_reserved_bytes():
    assert _plan([_request(bytes=128)]).required_bytes == 128
    with pytest.raises(ResidencyOverflow, match="overflow by 1 bytes") as error:
        _plan([_request(bytes=129)])
    assert (error.value.required_bytes, error.value.capacity_bytes, error.value.overflow_bytes) == (129, 128, 1)
    with pytest.raises(ResidencyOverflow) as error:
        _plan([_request(bytes=17, alignment=16)], base=33, capacity=40,
              reserved=[ReservedInterval("image", 48, 16)])
    assert (error.value.required_bytes, error.value.overflow_bytes) == (48, 8)
    plan = _plan(reserved=[ReservedInterval("staging", 32, 128, role="staging")])
    assert plan.required_bytes == plan.capacity == 128
    with pytest.raises(ResidencyOverflow) as error:
        _plan(reserved=[ReservedInterval("staging", 32, 129, role="staging")])
    assert error.value.overflow_bytes == 1
    assert _plan(reserved=[ReservedInterval("image", 160, 1000)]).required_bytes == 0


def test_declared_entire_arena_must_fit_a_valid_dma_aperture():
    for options in (
        dict(base=16, capacity=32, dma_base=32), dict(base=32, capacity=225),
        dict(base=-1), dict(capacity=-1), dict(dma_base=-1), dict(dma_capacity=-1),
        dict(base=True), dict(capacity=1.5), dict(dma_capacity=None),
    ):
        with pytest.raises(ValueError):
            _plan(**options)


def test_reserved_intervals_are_disjoint_and_adjacent_intervals_are_valid():
    with pytest.raises(ValueError, match="overlap"):
        _plan(reserved=[ReservedInterval("a", 0, 12), ReservedInterval("b", 11, 4)])
    plan = _plan(reserved=[ReservedInterval("a", 0, 12), ReservedInterval("b", 12, 4)])
    assert len(plan.entries) == 2


def test_invalid_request_declarations_are_rejected():
    for changes in (
        dict(name=""), dict(role="parameter"), dict(role=[]), dict(bytes=-1), dict(bytes=True),
        dict(alignment=0), dict(alignment=False), dict(lifetime=""), dict(load_protocol="implicit"),
        dict(load_protocol={}),
    ):
        with pytest.raises(ValueError):
            _request(**changes)


def test_reserved_address_alignment_and_duplicate_names_are_validated():
    with pytest.raises(ValueError, match="misaligned"):
        ReservedInterval("image", 3, 8, alignment=4)
    with pytest.raises(ValueError, match="address"):
        ReservedInterval("image", -1, 8)
    for reservations in ((), (ReservedInterval("weights", 192, 8),)):
        requests = [_request()] if reservations else [_request(), _request()]
        with pytest.raises(ValueError, match="distinct names"):
            _plan(requests, reserved=reservations)


def test_empty_and_zero_byte_declarations_do_not_consume_storage():
    assert _plan(capacity=0).required_bytes == 0
    plan = _plan([_request("empty", bytes=0), _request("full", bytes=128)],
                 reserved=[ReservedInterval("empty_image", 40, 0)])
    assert plan.entry("empty").address == plan.entry("full").address == 32
    assert plan.required_bytes == 128


def test_records_and_digest_are_stable_under_declaration_reordering():
    requests = [_request("b"), _request("a", bytes=9)]
    reservations = [ReservedInterval("z_image", 192, 8), ReservedInterval("y_image", 208, 8)]
    first = _plan(requests, reserved=reservations)
    second = _plan(reversed(requests), reserved=reversed(reservations))
    assert first == second
    assert first.to_json() == json.dumps(first.record(), sort_keys=True, separators=(",", ":"))
    assert first.digest() == sha256(first.to_json().encode()).hexdigest() == second.digest()
    changed = _plan([replace(requests[0], load_protocol="per_call"), requests[1]], reserved=reservations)
    assert changed.digest() != first.digest()
    record = first.record()
    record["entries"][0]["address"] = -1
    assert first.entry("a").address == 32


def test_plain_records_are_accepted_and_image_changes_affect_identity():
    request = dict(name="weights", role="immutable_parameter", bytes=16, alignment=8,
                   lifetime="application", load_protocol="once_at_load")
    first = _plan([request], reserved=[dict(name="image", address=192, bytes=8)])
    second = _plan([request], reserved=[dict(name="image", address=200, bytes=8)])
    assert first.entry("weights").address == second.entry("weights").address
    assert first.digest() != second.digest()
    with pytest.raises(KeyError):
        first.entry("missing")
