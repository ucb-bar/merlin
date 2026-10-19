"""Parent admission starts after the original child-capable callback returns."""

import contextlib

import pytest

from merlin.llvmlower.reviewed_pass_capabilities import ReviewedPassCapabilities
from merlin.targetgen.compiler_pass_selection import OriginalPassFactory

_CALL = None


def original_factory(inputs, *, work_root):
    return _CALL(inputs, work_root)


@pytest.mark.parametrize("damage", [None, "callback", "registry"])
def test_original_factory_leases_only_post_callback_qualification(monkeypatch, tmp_path, damage):
    from merlin.common import cpu_pool

    events, held = [], False
    registry = ReviewedPassCapabilities(())

    @contextlib.contextmanager
    def lease(*, timeout_s):
        nonlocal held
        assert not held and timeout_s == 120
        events.append("lease")
        held = True
        try:
            yield
        finally:
            held = False
            events.append("release")

    def callback(inputs, work_root):
        assert not held
        assert inputs == {"neutral": True} and work_root == tmp_path
        events.append("callback_without_parent_lease")
        if damage == "callback":
            raise ValueError("callback refused")
        return registry

    def revalidate(self):
        assert self is registry and held
        events.append("live_postvalidation")
        if damage == "registry":
            raise ValueError("current custody refused")

    monkeypatch.setattr(cpu_pool, "shared_compute_pool", lease)
    monkeypatch.setitem(globals(), "_CALL", callback)
    monkeypatch.setattr(ReviewedPassCapabilities, "revalidate", revalidate)
    factory = OriginalPassFactory(original_factory, {"neutral": True})
    if damage is not None:
        with pytest.raises(ValueError, match="refused"):
            factory.create(work_root=tmp_path)
    else:
        assert factory.create(work_root=tmp_path) is registry
    assert not held
    assert events == (
        ["callback_without_parent_lease"]
        if damage == "callback"
        else ["callback_without_parent_lease", "lease", "live_postvalidation", "release"]
    )
