"""Independent metadata lifetime controls; no grader or child is launched."""

import contextlib
import contextvars
from types import SimpleNamespace

import pytest
import test_compiler_child_metadata_operation as FIXTURE
from merlin_experiments.phase1.feedback import compiler_profile_child as CHILD
from merlin_experiments.phase2.contracts import StageGateError

from merlin.llvmlower.reviewed_metadata import _metadata_verified
from merlin.targetgen.compiler_pass_selection import scoped_compiler_passes


def selection(tmp_path, name, *, enabled=True):
    private = tmp_path / name
    private.mkdir()
    return FIXTURE.selection(SimpleNamespace(private=private), enabled=enabled)


@pytest.fixture
def preparation(monkeypatch):
    calls = []
    token = object()

    @contextlib.contextmanager
    def prepare(**kwargs):
        calls.append("enter")
        try:
            yield token
        finally:
            calls.append("exit")

    monkeypatch.setattr(CHILD, "_prepare_model_child", prepare)
    return calls, token


@pytest.mark.parametrize("active_selection", ["same", "foreign", "none"])
def test_inherited_metadata_refuses_before_preparing_any_transport(tmp_path, preparation, active_selection):
    passes, _ = selection(tmp_path, "selected")
    foreign, _ = selection(tmp_path, "foreign")
    active = contextlib.nullcontext() if active_selection == "none" else scoped_compiler_passes(passes)
    enclosing = passes if active_selection == "same" else foreign
    with active, enclosing.metadata_operation():
        with pytest.raises((ValueError, StageGateError), match="metadata"):
            with CHILD.prepare_model_child(spec_path="unused", environment={}, package_dir="unused"):
                pytest.fail("inherited metadata reached a launchable transport")
        assert preparation[0] == []


def test_copied_live_and_expired_contexts_cannot_reopen_metadata_transport(tmp_path, preparation):
    passes, _ = selection(tmp_path, "selected")

    def prepare():
        with CHILD.prepare_model_child(spec_path="unused", environment={}, package_dir="unused"):
            pytest.fail("copied metadata context exposed a transport")

    with scoped_compiler_passes(passes):
        with passes.metadata_operation():
            copied = contextvars.copy_context()
            with pytest.raises((ValueError, StageGateError), match="metadata"):
                copied.run(prepare)
        # The copy retains its ContextVar value after the original operation's
        # live owner has expired. It must never recreate launch authority.
        with pytest.raises((ValueError, StageGateError), match="metadata"):
            copied.run(prepare)
    assert preparation[0] == []


def test_caller_exception_releases_transport_after_metadata_closed(tmp_path, monkeypatch):
    passes, capability = selection(tmp_path, "selected")
    observations = []

    @contextlib.contextmanager
    def prepare(**kwargs):
        observations.append(("enter", _metadata_verified(capability)))
        try:
            yield object()
        finally:
            observations.append(("exit", _metadata_verified(capability)))

    monkeypatch.setattr(CHILD, "_prepare_model_child", prepare)
    with scoped_compiler_passes(passes):
        with pytest.raises(RuntimeError, match="caller failed"):
            with CHILD.prepare_model_child(spec_path="unused", environment={}, package_dir="unused"):
                assert not _metadata_verified(capability)
                raise RuntimeError("caller failed")
        assert observations == [("enter", True), ("exit", False)]
        assert not _metadata_verified(capability)


def test_setup_exception_restores_original_metadata_state(tmp_path, monkeypatch):
    passes, capability = selection(tmp_path, "selected")
    observations = []

    @contextlib.contextmanager
    def prepare(**kwargs):
        try:
            observations.append(_metadata_verified(capability))
            raise RuntimeError("setup failed")
        finally:
            observations.append(_metadata_verified(capability))
        yield None

    monkeypatch.setattr(CHILD, "_prepare_model_child", prepare)
    with scoped_compiler_passes(passes):
        with pytest.raises(RuntimeError, match="setup failed"):
            with CHILD.prepare_model_child(spec_path="unused", environment={}, package_dir="unused"):
                pytest.fail("failed setup exposed a transport")
        assert observations == [True, True]
        assert not _metadata_verified(capability)


@pytest.mark.parametrize("selected", [False, True])
def test_no_selection_and_legacy_nonopted_setup_preserve_transport(tmp_path, preparation, selected):
    passes, capability = selection(tmp_path, "legacy", enabled=False)
    scope = scoped_compiler_passes(passes) if selected else contextlib.nullcontext()
    with scope:
        with CHILD.prepare_model_child(spec_path="unused", environment={}, package_dir="unused") as transport:
            assert transport is preparation[1]
            assert not _metadata_verified(capability)
    assert preparation[0] == ["enter", "exit"]
