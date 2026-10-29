"""Direct host-interface refusal controls; no mocked positive proof verdicts."""

import builtins
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from configuration_elision_qualification_cases import cases
from test_compiler_configuration import selection

from merlin.verify import compiler_configuration as gate
from merlin.verify import gate_worker, pass_validation
from merlin.verify.speculative_gate import SpeculativeGate


@pytest.fixture
def canonical_registry():
    assert pass_validation.CHECKS[gate.SELECTOR] == gate.CHECK
    assert gate.CHECK in gate_worker.BUILTIN_CHECKS


def validation(work):
    tools = object.__new__(pass_validation.CheckerTools)
    object.__setattr__(tools, "sdk_bin", Path("/NEVER_OPENED"))
    object.__setattr__(tools, "compiler_python", Path("/NEVER_OPENED"))
    return pass_validation.PassValidation(SpeculativeGate(work, interrupt=False), tools, frozenset({gate.SELECTOR}))


def test_wrong_active_owner_refuses_before_running_actual_producer(tmp_path, canonical_registry):
    first, second = validation(tmp_path / "first"), validation(tmp_path / "second")
    token = pass_validation._ACTIVE.set(first)
    try:
        with pytest.raises(ValueError, match="differs from active"):
            gate.validated_apply(SimpleNamespace(apply=lambda _: pytest.fail("producer reached")), object(), second)
    finally:
        pass_validation._ACTIVE.reset(token)
        first.gate.close()
        second.gate.close()


def test_guarded_owner_cannot_move_to_an_unbound_thread(tmp_path, canonical_registry):
    chosen = validation(tmp_path / "owner")
    failures = []

    def another_thread():
        try:
            gate.validated_apply(SimpleNamespace(apply=lambda _: pytest.fail("producer reached")), object(), chosen)
        except ValueError as error:
            failures.append(str(error))

    with chosen.gate.guard():
        worker = threading.Thread(target=another_thread)
        worker.start()
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert failures == ["configuration proof must execute on its original active compiler owner thread"]
        assert chosen.gate.verdicts == []


def test_alias_cannot_replace_the_canonical_configuration_handler(tmp_path, canonical_registry):
    tools = object.__new__(pass_validation.CheckerTools)
    runner = SpeculativeGate(tmp_path / "gate", interrupt=False, aliases={gate.CHECK: "tensor_refinement"})
    try:
        with pytest.raises(ValueError, match="canonical handler"):
            pass_validation.PassValidation(runner, tools, frozenset({gate.SELECTOR}))
    finally:
        runner.close()


def test_kernel_owner_mismatch_refuses_before_projecting_any_candidate(tmp_path, canonical_registry):
    first, second = validation(tmp_path / "first"), validation(tmp_path / "second")
    token = pass_validation._ACTIVE.set(first)
    try:
        with pytest.raises(ValueError, match="differs from active"):
            gate.validated_kernel_result(None, None, object(), object(), second)
    finally:
        pass_validation._ACTIVE.reset(token)
        first.gate.close()
        second.gate.close()


def test_schedule_owner_mismatch_refuses_before_running_the_pipeline(tmp_path, canonical_registry):
    from merlin.compile.scheduling.schedule_pipeline import compile_schedule

    first, second = validation(tmp_path / "first"), validation(tmp_path / "second")
    token = pass_validation._ACTIVE.set(first)
    try:
        with pytest.raises(ValueError, match="differs from active"):
            compile_schedule(None, None, None, None, None, None, pass_validation=second)
    finally:
        pass_validation._ACTIVE.reset(token)
        first.gate.close()
        second.gate.close()


def test_default_apply_executes_without_importing_verifier(tmp_path, monkeypatch):
    chosen = selection(tmp_path)
    original = chosen.parse(next(case.source for case in cases() if case.name == "multifield"))
    original_import = builtins.__import__

    def without_verifier(name, *args, **kwargs):
        if name.startswith("merlin.verify"):
            raise ImportError("ordinary export has no verifier")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(builtins, "__import__", without_verifier)
        result = chosen.apply(original)
        result.revalidate()
    assert result.report["verification"]["removed_operations"] == 2
