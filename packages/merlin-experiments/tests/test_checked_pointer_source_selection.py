"""Explicit direct-source routing controls; no target qualification in fixtures."""

import copy
import sys
from types import ModuleType

import pytest
from test_checked_source_model_diagnostic import selected as selected

from merlin.targetgen import checked_source_model_diagnostic as S
from merlin.targetgen import checked_source_model_diagnostic_execution as E


def pointer(selected):
    passes, _, _, facts, _ = selected
    return S.select_checked_source_diagnostic(passes, target="fixture", rtl_facts=facts,
        board_config="config", native_execution_protocol=S.POINTER_PROTOCOL)


def test_legacy_default_never_infers_pointer_mode_from_command_buffer(selected, monkeypatch):
    calls = []
    monkeypatch.setattr(E.NATIVE, "execute_candidate_model", lambda **kwargs: calls.append(kwargs) or "legacy")
    passes, selection, *_ = selected
    for scope in (None, selection):
        from contextlib import nullcontext
        context = nullcontext() if scope is None else S.scoped_checked_source_diagnostic(scope)
        with context:
            assert E.execute_native_model(command_buffer={"compiler_pointer_abi": {"version": 1}}) == "legacy"
    assert len(calls) == 2
    assert "native_execution_protocol" not in selection.record()


def test_only_actual_explicit_host_selection_routes_and_rechecks_owner(selected, monkeypatch):
    selection = pointer(selected)
    seen = []
    module = ModuleType("merlin.targetgen.native_pointer_source_execution")
    module.MAX_TIMEOUT_SECONDS = 600
    module.execute_candidate_pointer_source = lambda **kwargs: seen.append(kwargs) or "complete unchanged result"
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(E.NATIVE, "execute_candidate_model", lambda **_kw: pytest.fail("wrong original route"))
    with S.scoped_checked_source_diagnostic(selection):
        assert E.execute_native_model(command_buffer={"original": True}) == "complete unchanged result"
    assert seen == [{"command_buffer": {"original": True}}]
    assert selection.record()["schema"] == S.POINTER_SCHEMA
    assert selection.record()["formal_admission"] is False


def test_owner_drift_during_pointer_execution_invalidates_result(selected, monkeypatch):
    selection = pointer(selected)
    _, _, issuer, *_ = selected
    module = ModuleType("merlin.targetgen.native_pointer_source_execution")
    module.MAX_TIMEOUT_SECONDS = 600

    def changed(**_kw):
        issuer.refuse_on_check = 0
        return "must not escape"

    module.execute_candidate_pointer_source = changed
    monkeypatch.setitem(sys.modules, module.__name__, module)
    token = S._ACTIVE.set(selection)
    try:
        with pytest.raises(ValueError, match="evidence changed"):
            E.execute_native_model()
    finally:
        S._ACTIVE.reset(token)


@pytest.mark.parametrize("budget, leaf", [(1, 1), (599, 599), (600, 600), (601, 600), (900, 600)])
def test_pointer_leaf_preserves_native_ceiling_with_larger_aggregate_budget(selected, monkeypatch, budget, leaf):
    from merlin.targetgen import native_pointer_source_execution as native

    selection = pointer(selected)
    received = []
    monkeypatch.setattr(native, "execute_candidate_pointer_source", lambda **kw: received.append(kw) or "result")
    arguments = {"timeout": budget, "command_buffer": {"unchanged": True}}
    with S.scoped_checked_source_diagnostic(selection):
        assert E.execute_native_model(**arguments) == "result"
    assert received == [{**arguments, "timeout": leaf}]
    assert arguments["timeout"] == budget


def test_legacy_route_keeps_original_aggregate_timeout(selected, monkeypatch):
    received = []
    monkeypatch.setattr(E.NATIVE, "execute_candidate_model", lambda **kw: received.append(kw) or "legacy")
    _, selection, *_ = selected
    with S.scoped_checked_source_diagnostic(selection):
        assert E.execute_native_model(timeout=900) == "legacy"
    assert received == [{"timeout": 900}]


def test_copied_owner_or_saved_record_cannot_select_pointer_route(selected):
    selection = pointer(selected)
    for forged in (copy.copy(selection), selection.record()):
        token = S._ACTIVE.set(forged)
        try:
            with pytest.raises(ValueError):
                E.execute_native_model()
        finally:
            S._ACTIVE.reset(token)


@pytest.mark.parametrize("protocol", [True, {}, "nearest", ""])
def test_unknown_pointer_protocol_refuses_before_owner_use(selected, protocol):
    passes, _, _, facts, _ = selected
    with pytest.raises(ValueError, match="exact explicit"):
        S.select_checked_source_diagnostic(passes, target="fixture", rtl_facts=facts,
            board_config="config", native_execution_protocol=protocol)
