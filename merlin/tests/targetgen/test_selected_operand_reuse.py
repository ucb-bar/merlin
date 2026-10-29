"""Exact input reuse avoids repeated structural work without reusing authority."""
import copy
from concurrent.futures import ThreadPoolExecutor

import pytest

from merlin.targetgen.rocc import selected_operand_protocol as P
from test_selected_operand_protocol import load, selected


def test_concurrent_exact_selection_derives_once_and_returns_detached_observers(tmp_path, monkeypatch):
    original, facts, blob, sidecar, source, isa = selected(tmp_path)
    P._reset_derivations()
    derive = P.apply_selected_layouts
    calls = []

    def counted(*args):
        calls.append(1)
        return derive(*args)

    monkeypatch.setattr(P, "apply_selected_layouts", counted)
    with ThreadPoolExecutor(max_workers=8) as pool:
        observers = list(pool.map(lambda _: load(original, blob, sidecar), range(16)))
    assert len(calls) == 1
    assert len({id(o) for o in observers}) == 16
    assert original._facts == facts and original.operand_observer is None
    assert all(o.isa_constants("array") == observers[0].isa_constants("array") for o in observers)
    # A changed declaration must not hit the old derivation.
    forged = copy.deepcopy(sidecar)
    forged["layouts"]["fields"][0]["width"] = 16
    with pytest.raises(ValueError):
        load(original, blob, forged)
    assert len(calls) == 2


@pytest.mark.parametrize("which", ["firrtl", "isa", "symlink", "owner"])
def test_warm_reuse_still_refuses_mutations(tmp_path, which):
    original, facts, blob, sidecar, source, isa = selected(tmp_path)
    P._reset_derivations()
    load(original, blob, sidecar)
    if which == "owner":
        original._facts["changed"] = True
    elif which == "symlink":
        alternate = tmp_path / "same.fir"
        alternate.write_bytes(source.read_bytes())
        source.unlink()
        source.symlink_to(alternate)
    else:
        path = source if which == "firrtl" else isa
        import os
        before = path.stat()
        path.write_bytes(path.read_bytes().replace(b"4", b"5", 1))
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError):
        load(original, blob, sidecar)


def test_warm_result_mutation_does_not_change_cached_observation(tmp_path):
    original, facts, blob, sidecar, source, isa = selected(tmp_path)
    P._reset_derivations()
    first = load(original, blob, sidecar)
    first._isa.clear()
    second = load(original, blob, sidecar)
    assert second.isa_constants("array")
    assert original._facts == facts


def test_warm_final_source_check_runs_after_new_observer_construction(tmp_path, monkeypatch):
    original, facts, blob, sidecar, source, isa = selected(tmp_path)
    P._reset_derivations()
    load(original, blob, sidecar)
    construct = P.BoundOperandObserver

    def mutate(**kwargs):
        result = construct(**kwargs)
        source.write_text(source.read_text() + "\n")
        return result

    monkeypatch.setattr(P, "BoundOperandObserver", mutate)
    with pytest.raises(ValueError, match="bytes changed"):
        load(original, blob, sidecar)


def test_guarded_source_scope_reuses_hash_not_observer_authority(tmp_path, monkeypatch):
    from merlin.common import guarded_file_digest as G
    monkeypatch.setenv("MERLIN_SNAPSHOT_LEASE_CACHE", "1")
    original, facts, blob, sidecar, source, isa = selected(tmp_path)
    calls = []
    read = G._read
    def counted(path):
        calls.append(path)
        return read(path)
    monkeypatch.setattr(G, "_read", counted)
    owner = G.GuardedDigests()
    try:
        with owner.scope():
            first = load(original, blob, sidecar)
            second = load(original, blob, sidecar)
        if not owner._entries:
            pytest.skip("kernel watches unavailable")
        assert calls.count(source) == 1
        assert first is not second and original._facts == facts
        with pytest.raises(ValueError, match="bytes changed"):
            with owner.scope():
                source.write_bytes(source.read_bytes() + b"\n")
                load(original, blob, sidecar)
    finally:
        owner.close()
