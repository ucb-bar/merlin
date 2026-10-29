"""Original-owner interface refusals; no positive proof or handler mock."""

import builtins
import threading
from pathlib import Path

import pytest
from test_compiler_configuration_hoist import actual

from merlin.verify import compiler_configuration_hoist as gate
from merlin.verify import gate_worker, pass_validation
from merlin.verify.speculative_gate import SpeculativeGate


@pytest.fixture
def canonical_registry():
    assert pass_validation.CHECKS[gate.SELECTOR] == gate.CHECK
    assert gate.CHECK in gate_worker.BUILTIN_CHECKS


def validation(work):
    tools = object.__new__(pass_validation.CheckerTools)
    object.__setattr__(tools, "sdk_bin", Path("/never-opened"))
    object.__setattr__(tools, "compiler_python", Path("/never-opened"))
    return pass_validation.PassValidation(SpeculativeGate(work, interrupt=False), tools, frozenset({gate.SELECTOR}))


def test_wrong_active_owner_refuses_before_original_producer(tmp_path, canonical_registry):
    first, second = validation(tmp_path / "first"), validation(tmp_path / "second")
    token = pass_validation._ACTIVE.set(first)
    try:
        with pytest.raises(ValueError, match="differs from active"):
            gate.validated_hoist(object(), object(), selected_contracts=None, validation=second)
    finally:
        pass_validation._ACTIVE.reset(token)
        first.gate.close()
        second.gate.close()


def test_guarded_owner_cannot_move_to_an_unbound_thread(tmp_path, canonical_registry):
    owner = validation(tmp_path / "owner")
    failures = []

    def another_thread():
        try:
            gate.validated_hoist(object(), object(), selected_contracts=None, validation=owner)
        except ValueError as error:
            failures.append(str(error))

    with owner.gate.guard():
        thread = threading.Thread(target=another_thread)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert failures == ["hoisting proof must execute on original active owner thread"]
        assert owner.gate.verdicts == []


def test_active_producer_requires_original_selected_contracts(tmp_path, canonical_registry):
    from merlin.compile.scheduling.configuration_loops import hoist_loop_configuration

    owner = validation(tmp_path / "owner")
    with pass_validation.validating(owner):
        with pytest.raises(ValueError, match="original typed selected"):
            hoist_loop_configuration(object(), object())
    assert owner.gate.verdicts == []


def test_custom_handler_alias_cannot_replace_the_canonical_hoist_proof(tmp_path, canonical_registry):
    tools = object.__new__(pass_validation.CheckerTools)
    runner = SpeculativeGate(tmp_path / "owner", interrupt=False, aliases={gate.CHECK: "tensor_refinement"})
    try:
        with pytest.raises(ValueError, match="canonical handler"):
            pass_validation.PassValidation(runner, tools, frozenset({gate.SELECTOR}))
    finally:
        runner.close()


def test_default_producer_executes_without_verifier_import(tmp_path, monkeypatch):
    original_import = builtins.__import__

    def without_verifier(name, *args, **kwargs):
        if name.startswith("merlin.verify"):
            raise ImportError("ordinary producer export contains no verifier")
        return original_import(name, *args, **kwargs)

    # Selected facts are original inputs, and the hook's default execution remains
    # the same production algorithm while verification imports are unavailable.
    with monkeypatch.context() as scoped:
        scoped.setattr(builtins, "__import__", without_verifier)
        _, original, result = actual(tmp_path)
    assert result.report["hoisted_static_calls"] == 1
    assert original.body[1].body[0] == result.kernel.body[1]
