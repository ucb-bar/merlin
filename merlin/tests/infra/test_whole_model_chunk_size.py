"""``chunk_ops="auto"``: a whole-model build derives its forward's chunk size from the forward itself.

A large open model's host forward compiles as one function for hours (SmolVLA: over two hours
unchunked, about eleven minutes cut at 1,000 ops); a small one must stay byte-for-byte unchunked, and
be recorded so, or its reference arm would be rebuilt under a different chunking key."""

from __future__ import annotations

import pytest

from merlin.perf import whole_model_builder as B
from merlin.perf import whole_model_chunks as WC


def test_auto_cuts_a_large_forward_at_the_default_and_leaves_a_small_one_whole():
    assert WC.resolve_chunk_ops(WC.AUTO, forward_ops=WC.DEFAULT_CHUNK_OPS + 1) == WC.DEFAULT_CHUNK_OPS
    assert WC.resolve_chunk_ops("Auto", forward_ops=WC.DEFAULT_CHUNK_OPS) is None
    assert WC.resolve_chunk_ops(WC.AUTO) == WC.DEFAULT_CHUNK_OPS  # an unknown size counts as large


def test_an_explicit_size_is_itself_and_a_misspelled_one_is_refused():
    assert WC.resolve_chunk_ops(None, forward_ops=10**6) is None
    assert WC.resolve_chunk_ops(64) == 64 and WC.resolve_chunk_ops(" 500 ") == 500
    for bad in (0, -3, True, "1k", "chunk", 2.5):
        with pytest.raises(ValueError, match="positive op count"):
            WC.resolve_chunk_ops(bad)


def test_a_closed_model_takes_auto_as_nothing_to_cut_and_refuses_a_size(monkeypatch, tmp_path):
    """One build option serves both: a closed model has no host forward, so ``auto`` asks nothing of it,
    while an explicit size still names something that does not exist and is refused."""
    from merlin.perf import whole_model_build as WMB
    from merlin.perf import whole_model_open as WO

    class Reached(Exception):
        pass

    built = []

    def closed_build(*args, **kwargs):
        built.append(kwargs)
        raise Reached

    monkeypatch.setattr(WO, "is_open_model", lambda capsule, target: False)
    monkeypatch.setattr(WMB, "build", closed_build)
    common = dict(target="t", out_dir=tmp_path, model_capsule="c", machine="m", header="h")
    with pytest.raises(WMB.WholeModelBuildError, match="this model is closed"):
        B.build(tmp_path, chunk_ops=64, **common)
    assert built == []
    with pytest.raises(Reached):
        B.build(tmp_path, chunk_ops=WC.AUTO, **common)
    assert len(built) == 1 and "chunk_ops" not in built[0]
