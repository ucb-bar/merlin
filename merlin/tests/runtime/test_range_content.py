"""Exact bytes, padding, mutation, session ownership and failed-write guards."""
import pytest
from merlin.runtime.range_content import RangeContentSession


def test_exact_bytes_padding_overlap_and_distinct_sessions():
    device = bytearray(64)
    writes = []
    session = RangeContentSession(max_bytes=64)
    other = RangeContentSession(max_bytes=64)
    def write(address, data):
        writes.append((address, data))
        device[address:address + len(data)] = data
        return 0
    payload = bytearray(b"AB\0\0")
    session.observe_generation(1)
    assert not session.write(0, payload, lambda data: write(0, data))
    payload[0] = ord('Z')
    assert session.entries[(0, 4)] == b"AB\0\0"
    assert not session.write(0, b"AB\0\0", lambda data: write(0, data))
    assert session.hits == 1 and len(writes) == 1
    assert not session.write(0, b"AB\0x", lambda data: write(0, data))
    assert len(writes) == 2
    assert not session.write(2, b"XX", lambda data: write(2, data))
    assert not session.write(0, b"AB\0x", lambda data: write(0, data))
    assert len(writes) == 4
    assert not other.write(0, b"AB\0x", lambda data: write(0, data))
    assert len(writes) == 5
    session.begin_launch([(1, 2)])
    with pytest.raises(ValueError, match="active"):
        session.write(8, b"x", lambda data: write(8, data))
    with pytest.raises(ValueError, match="active"):
        session.close()
    session.end_launch(success=True)
    assert not session.write(0, b"AB\0x", lambda data: write(0, data))
    assert len(writes) == 6
    session.observe_generation(2)
    assert not session.write(0, b"AB\0x", lambda data: write(0, data))
    assert len(writes) == 7
    assert session.close()["hits"] == 1
    with pytest.raises(ValueError, match="closed"):
        session.write(0, b"x", lambda data: 0)


def test_failure_unknown_mutation_capacity_and_malformed_footprints():
    session = RangeContentSession(max_bytes=4, max_entries=1)
    assert not session.write(0, b"abcd", lambda data: 0)
    assert session.write(8, b"x", lambda data: 9) == 9
    assert not session.entries
    assert not session.write(0, b"abcd", lambda data: 0)
    with pytest.raises(RuntimeError):
        session.write(8, b"x", lambda data: (_ for _ in ()).throw(RuntimeError("partial")))
    assert not session.entries
    assert not session.write(0, b"abcd", lambda data: 0)
    session.invalidate_all()  # Explicit external-write/reset hook.
    assert not session.entries
    assert not session.write(0, b"abcd", lambda data: 0)
    session.begin_launch([])
    session.end_launch(success=False)
    assert not session.entries
    assert not session.write(0, b"abcd", lambda data: 0)
    assert not session.write(8, b"y", lambda data: 0)
    assert session.evictions == 1 and (0, 4) not in session.entries
    with pytest.raises(ValueError):
        session.begin_launch([(0, 0)])
    assert not session.entries
