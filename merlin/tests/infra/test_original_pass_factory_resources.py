"""Parent admission starts after the original child-capable callback returns."""

import contextlib
import importlib.util

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


@pytest.mark.parametrize("damage", [None, "constructor", "record"])
def test_restored_selection_leases_metadata_after_child_capable_factory(monkeypatch, tmp_path, damage):
    from merlin.common import cpu_pool
    from merlin.llvmlower.reviewed_pass_capabilities import PassCapabilityBinding
    from merlin.targetgen import compiler_pass_selection as S

    source = tmp_path / "neutral_factory.py"
    source.write_text(
        "from merlin.llvmlower.reviewed_pass_capabilities import ReviewedPassCapabilities\n"
        "def restore(inputs, *, work_root):\n"
        "    return ReviewedPassCapabilities(())\n"
    )
    spec = importlib.util.spec_from_file_location("neutral_restoration_fixture", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory = OriginalPassFactory(module.restore, {})
    binding = PassCapabilityBinding("hardware", "software", "numerics", "runtime", "library")
    initial = S.CompilerPassSelection(
        ReviewedPassCapabilities(()), binding, mechanisms=("configuration_cleanup",), factory=factory)
    record, held, events = initial.record(), False, []
    original_create = S.OriginalPassFactory.create
    original_initialize = S.CompilerPassSelection.__init__
    original_record = S.CompilerPassSelection.record

    @contextlib.contextmanager
    def leaf(*, timeout_s):
        nonlocal held
        assert not held and timeout_s == 120
        held = True
        events.append("lease")
        try:
            yield
        finally:
            held = False
            events.append("release")

    def create(self, *, work_root):
        assert not held  # The callback may launch an independently leased child.
        events.append("callback_without_parent_lease")
        return original_create(self, work_root=work_root)

    def initialize(self, *args, **kwargs):
        assert held
        events.append("selection_constructor")
        if damage == "constructor":
            raise ValueError("constructor refused")
        return original_initialize(self, *args, **kwargs)

    def selection_record(self):
        assert held
        events.append("selection_record")
        if damage == "record":
            raise ValueError("record refused")
        return original_record(self)

    monkeypatch.setattr(cpu_pool, "shared_compute_pool", leaf)
    monkeypatch.setattr(S.OriginalPassFactory, "create", create)
    monkeypatch.setattr(S.CompilerPassSelection, "__init__", initialize)
    monkeypatch.setattr(S.CompilerPassSelection, "record", selection_record)
    if damage:
        with pytest.raises(ValueError, match="refused"):
            S.restore_original_selection(record, work_root=tmp_path / "restored")
    else:
        restored = S.restore_original_selection(record, work_root=tmp_path / "restored")
        assert restored._record == record
    assert not held
    assert events == ["callback_without_parent_lease", "lease", "release", "lease", "selection_constructor",
                      *([] if damage == "constructor" else ["selection_record"]), "release"]
