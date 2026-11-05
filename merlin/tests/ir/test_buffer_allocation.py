"""Finite buffer placement, including aliases and completion lifetimes."""

import random
from dataclasses import replace
from itertools import product

import pytest

from merlin.semantic_compiler.allocate import Reservation, StorageBank
from merlin.semantic_compiler.buffer_allocation import Buffer, allocate_buffers, check_buffer_allocation


def test_completion_lifetimes_and_reservations():
    bank = StorageBank("scratch", "ram", 8, "byte", 2)
    buffers = (Buffer("a", "scratch", 4, 0, 3), Buffer("b", "scratch", 4, 4, 6))
    result = allocate_buffers(buffers, (bank,), reservations=(Reservation("scratch", 0, 2),))
    assert result.status == "sat"
    assert dict(result.addresses) == {"a": 2, "b": 2}
    # The first buffer is still being read at event 4: reuse is now illegal.
    assert (
        allocate_buffers(
            (replace(buffers[0], last_event=4), buffers[1]), (bank,), reservations=(Reservation("scratch", 0, 2),)
        ).status
        == "unsat"
    )


def test_aliases_extend_lifetimes_and_share_backing():
    banks = (StorageBank("fp8", "registers", 8, "byte"), StorageBank("bf16", "registers", 8, "byte"))
    buffers = (
        Buffer("base", "fp8", 4, 0, 1),
        Buffer("view", "bf16", 2, 1, 7, view_of="base", view_offset=2),
        Buffer("other", "fp8", 4, 2, 5),
    )
    result = allocate_buffers(buffers, banks)
    assert dict(result.addresses) == {"base": 0, "view": 2, "other": 4}
    with pytest.raises(ValueError, match="live buffer"):
        check_buffer_allocation(buffers, banks, {"base": 0, "view": 2, "other": 0})
    with pytest.raises(ValueError, match="root"):
        check_buffer_allocation(buffers, banks, {"base": 0, "view": 1, "other": 4})


def test_persistent_and_fixed_buffers():
    bank = StorageBank("dram", "external", 8, "byte")
    a = Buffer("state", "dram", 4, 0, 1, fixed_start=0, persistent=True)
    b = Buffer("temporary", "dram", 4, 10, 20)
    assert dict(allocate_buffers((a, b), (bank,)).addresses) == {"state": 0, "temporary": 4}
    with pytest.raises(ValueError, match="overlap"):
        check_buffer_allocation((a, b), (bank,), {"state": 0, "temporary": 0})


def test_malformed_views_and_units_are_rejected():
    bank = StorageBank("a", "ram", 4, "byte")
    with pytest.raises(ValueError, match="cyclic"):
        allocate_buffers((Buffer("x", "a", 2, 0, 1, view_of="y"), Buffer("y", "a", 2, 0, 1, view_of="x")), (bank,))
    with pytest.raises(ValueError, match="escapes"):
        allocate_buffers((Buffer("x", "a", 2, 0, 1), Buffer("y", "a", 2, 0, 1, view_of="x", view_offset=1)), (bank,))
    with pytest.raises(ValueError, match="address unit"):
        allocate_buffers((), (bank, StorageBank("b", "ram", 4, "word")))


def test_solver_agrees_with_200_independently_enumerated_tiny_instances():
    rng = random.Random(17)
    for _ in range(200):
        capacity = rng.randrange(2, 7)
        buffers = []
        for i in range(3):
            start = rng.randrange(4)
            buffers.append(
                Buffer(
                    str(i),
                    "a",
                    rng.randrange(1, 4),
                    start,
                    start + rng.randrange(3),
                    alignment=rng.choice((1, 2)),
                    persistent=rng.randrange(5) == 0,
                )
            )
        buffers = tuple(buffers)
        # Enumerate all physical starts. Do not call the solver's checker to
        # establish expected feasibility.
        possible = False
        for addresses in product(range(capacity), repeat=3):
            if any(addr % buf.alignment or addr + buf.extent > capacity for addr, buf in zip(addresses, buffers)):
                continue
            valid = True
            for i, a in enumerate(buffers):
                for j, b in enumerate(buffers[:i]):
                    simultaneous = (
                        a.persistent
                        or b.persistent
                        or not (a.last_event < b.first_event or b.last_event < a.first_event)
                    )
                    if simultaneous and not (
                        addresses[i] + a.extent <= addresses[j] or addresses[j] + b.extent <= addresses[i]
                    ):
                        valid = False
            if valid:
                possible = True
                break
        result = allocate_buffers(buffers, (StorageBank("a", "ram", capacity, "byte"),))
        assert result.status == ("sat" if possible else "unsat"), (buffers, capacity, result)


def test_unknown_solver_status_is_not_an_allocation_or_an_unsat_claim(monkeypatch):
    from unittest.mock import Mock

    import z3

    solver = Mock()
    solver.check.return_value = z3.unknown
    solver.reason_unknown.return_value = "injected inconclusive solve"
    monkeypatch.setattr(z3, "Solver", lambda: solver)
    result = allocate_buffers((Buffer("x", "a", 1, 0, 1),), (StorageBank("a", "ram", 2, "byte"),))
    assert result.status == "unknown"
    assert result.addresses == ()
    assert "inconclusive" in result.reason
