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


def test_deterministic_resource_budget_is_fail_closed_and_has_no_wall_deadline(monkeypatch):
    import merlin.semantic_compiler.buffer_allocation as allocation

    def no_clock():
        raise AssertionError("deterministic allocation must not read the wall clock")

    monkeypatch.setattr(allocation, "monotonic", no_clock)
    bank = StorageBank("a", "ram", 256, "byte")
    buffers = (Buffer("inputs", "a", 4, 0, 1), Buffer("outputs", "a", 1, 0, 1),
               Buffer("view", "a", 1, 0, 1, view_of="inputs", view_offset=2))
    result = allocate_buffers(buffers, (bank,), timeout_ms=None, rlimit=1_000_000)
    assert result.status == "sat"
    assert dict(result.addresses) == {"inputs": 0, "outputs": 4, "view": 2}
    exhausted = allocate_buffers(buffers, (bank,), timeout_ms=None, rlimit=1)
    assert exhausted.status == "unknown"
    assert exhausted.addresses == ()
    assert exhausted.reason
    for options in ({"timeout_ms": None}, {"rlimit": 0}, {"rlimit": True}):
        with pytest.raises(ValueError):
            allocate_buffers(buffers, (bank,), **options)


def test_same_lifetime_frames_and_views_have_lexicographic_addresses():
    bank = StorageBank('external', 'test', 129, 'tile', 2)
    buffers = (Buffer('inputs', bank.name, 5, 0, 1), Buffer('outputs', bank.name, 3, 0, 1),
               Buffer('input0', bank.name, 1, 0, 1, view_of='inputs'),
               Buffer('input4', bank.name, 1, 0, 1, view_of='inputs', view_offset=4),
               Buffer('output2', bank.name, 1, 0, 1, view_of='outputs', view_offset=2))
    result = allocate_buffers(buffers, (bank,))
    assert result.status == 'sat'
    assert dict(result.addresses) == {'inputs': 0, 'outputs': 6, 'input0': 0, 'input4': 4, 'output2': 8}
    assert all(item.fixed_start is None for item in buffers)
    check_buffer_allocation(buffers, (bank,), dict(result.addresses))


def test_same_lifetime_proposal_preserves_exact_enumerated_lexicographic_optimum():
    rng = random.Random(107)
    for _ in range(32):
        bank = StorageBank('a', 'test', rng.randrange(3, 8), 'tile')
        buffers = tuple(Buffer(str(i), 'a', rng.randrange(1, 4), 0, 1, alignment=rng.choice((1, 2)))
                        for i in range(3))
        expected = next((positions for positions in product(range(bank.capacity), repeat=3)
            if all(position % item.alignment == 0 and position + item.extent <= bank.capacity
                   for position, item in zip(positions, buffers))
            and all(positions[i] + buffers[i].extent <= positions[j]
                    or positions[j] + buffers[j].extent <= positions[i]
                    for i in range(3) for j in range(i))), None)
        result = allocate_buffers(buffers, (bank,))
        assert result.status == ('unsat' if expected is None else 'sat')
        if expected is not None:
            assert tuple(dict(result.addresses)[item.name] for item in buffers) == expected


def test_valid_proposal_still_requires_satisfiable_solver_query():
    from unittest.mock import MagicMock, patch
    import z3

    bank = StorageBank('a', 'test', 19, 'tile')
    buffers = (Buffer('input', 'a', 3, 0, 1), Buffer('output', 'a', 2, 0, 1))
    solver = MagicMock()
    solver.check.return_value = z3.unknown
    solver.reason_unknown.return_value = 'forced solver exhaustion'
    with patch('z3.Solver', return_value=solver):
        result = allocate_buffers(buffers, (bank,))
    assert result.status == 'unknown' and result.reason == 'forced solver exhaustion'
    solver.check.assert_called_once()


def test_invalid_proposal_fails_independent_checker_before_solver():
    from unittest.mock import patch

    bank = StorageBank('a', 'test', 19, 'tile')
    buffers = (Buffer('input', 'a', 3, 0, 1), Buffer('output', 'a', 2, 0, 1))
    with patch('merlin.semantic_compiler.buffer_allocation.first_fit_buffer_assignment',
               return_value={'input': 0, 'output': 0}), patch('z3.Solver') as solver:
        with pytest.raises(ValueError, match='overlap'):
            allocate_buffers(buffers, (bank,))
    solver.assert_not_called()


def test_ineligible_lifetimes_fixed_persistence_and_interleaved_views_keep_solver_path():
    from unittest.mock import patch

    bank = StorageBank('a', 'test', 9, 'tile')
    variants = (
        (Buffer('a', 'a', 2, 0, 1), Buffer('b', 'a', 2, 2, 3)),
        (Buffer('a', 'a', 2, 0, 1, fixed_start=2), Buffer('b', 'a', 2, 0, 1)),
        (Buffer('a', 'a', 2, 0, 1, persistent=True), Buffer('b', 'a', 2, 0, 1)),
        (Buffer('a', 'a', 2, 0, 1), Buffer('view', 'a', 1, 0, 1, view_of='a'), Buffer('b', 'a', 2, 0, 1)),
    )
    with patch('merlin.semantic_compiler.buffer_allocation.first_fit_buffer_assignment') as propose:
        for buffers in variants:
            result = allocate_buffers(buffers, (bank,))
            assert result.status == 'sat'
            check_buffer_allocation(buffers, (bank,), dict(result.addresses))
    propose.assert_not_called()


def test_allocation_wall_budget_defaults_context_and_explicit_precedence():
    from unittest.mock import MagicMock, patch
    import z3
    from merlin.semantic_compiler.buffer_allocation import allocation_wall_budget, current_allocation_wall_budget_ms

    bank = StorageBank('a', 'test', 19, 'tile')
    buffers = (Buffer('fixed', 'a', 3, 0, 1, fixed_start=0),)
    def observed_timeout(**kwargs):
        solver = MagicMock()
        solver.check.return_value = z3.unknown
        solver.reason_unknown.return_value = 'forced unknown'
        with patch('z3.Solver', return_value=solver), patch('merlin.semantic_compiler.buffer_allocation.monotonic', return_value=0):
            result = allocate_buffers(buffers, (bank,), **kwargs)
        assert result.status == 'unknown' and result.reason == 'forced unknown'
        return next(call.kwargs['timeout'] for call in solver.set.call_args_list if 'timeout' in call.kwargs)
    assert current_allocation_wall_budget_ms() == 1000
    assert observed_timeout() == 1000
    with allocation_wall_budget(30000):
        assert current_allocation_wall_budget_ms() == 30000
        assert observed_timeout() == 30000
        assert observed_timeout(timeout_ms=73) == 73
        assert observed_timeout(timeout_ms=1000) == 1000
        with allocation_wall_budget(83):
            assert observed_timeout() == 83
        assert observed_timeout() == 30000
    assert current_allocation_wall_budget_ms() == 1000


def test_allocation_wall_budget_restores_on_error_and_rejects_invalid_values():
    from merlin.semantic_compiler.buffer_allocation import allocation_wall_budget, current_allocation_wall_budget_ms

    with pytest.raises(RuntimeError):
        with allocation_wall_budget(71):
            raise RuntimeError('test')
    assert current_allocation_wall_budget_ms() == 1000
    for value in (0, -1, True, 1.5, None):
        with pytest.raises(ValueError):
            with allocation_wall_budget(value):
                raise AssertionError('invalid budget entered')


def test_context_budget_does_not_admit_an_exhausted_wall_deadline():
    from unittest.mock import patch
    from merlin.semantic_compiler.buffer_allocation import allocation_wall_budget

    bank = StorageBank('a', 'test', 19, 'tile')
    buffers = (Buffer('input', 'a', 3, 0, 1), Buffer('output', 'a', 2, 0, 1))
    with allocation_wall_budget(30000), patch('merlin.semantic_compiler.buffer_allocation.monotonic', side_effect=(0, 31)):
        result = allocate_buffers(buffers, (bank,))
    assert result.status == 'unknown' and result.reason == 'buffer allocation time budget exhausted'


def test_allocation_wall_budget_is_context_local_across_threads():
    import threading
    from merlin.semantic_compiler.buffer_allocation import allocation_wall_budget, current_allocation_wall_budget_ms

    observed = []
    with allocation_wall_budget(30000):
        thread = threading.Thread(target=lambda: observed.append(current_allocation_wall_budget_ms()))
        thread.start()
        thread.join(2)
        assert not thread.is_alive()
        assert current_allocation_wall_budget_ms() == 30000
    assert observed == [1000]
