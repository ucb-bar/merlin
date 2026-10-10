"""Owned selection controls; building and execution remain diagnostic substitutes."""

from __future__ import annotations

import importlib.util
from functools import partial
from pathlib import Path
from types import MethodType

import pytest

from merlin.runtime.backends.base import parse_console
from merlin.targetgen.contract import compile as compiler
from merlin.targetgen.contract import readback_policy as RB
from merlin.targetgen.contract.build_recipe import HarnessBuildRecipe
from merlin.targetgen.contract.build_service import BuildOnlyService, file_digest
from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService
from merlin.targetgen.contract.execution_service import FunctionalExecutionService

SOURCE = """from pathlib import Path
import json
from merlin.targetgen.contract.build_service import file_digest
def report(elf, evidence_root, status):
    evidence_root.mkdir()
    path = evidence_root / "report.json"
    result = {"status": status, "elf_sha256": file_digest(elf)}
    path.write_text(json.dumps(result))
    return {**result, "report_path": str(path), "report_sha256": file_digest(path)}
def original(*, elf, evidence_root, status="refused", state=None):
    if state is not None:
        state.append(status)
    return report(elf, evidence_root, status)
def replacement(*, elf, evidence_root, status="accepted", state=None):
    return report(elf, evidence_root, "accepted")
def changing(*, elf, evidence_root):
    result = report(elf, evidence_root, "accepted")
    changing.__code__ = replacement.__code__
    return result
class Owner:
    def __init__(self):
        self.calls = 0
    def evaluate(self, *, elf, evidence_root):
        self.calls += 1
        return report(elf, evidence_root, "accepted")
class Spoof:
    __code__ = original.__code__
    def __call__(self, **kwargs):
        return original(**kwargs)
class PartialSubclass(__import__("functools").partial):
    pass
"""


@pytest.fixture
def owned(tmp_path):
    source = tmp_path / "owned_policy.py"
    source.write_text(SOURCE)
    spec = importlib.util.spec_from_file_location("owned_policy", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"owned linked-artifact diagnostic")

    def selected(callback):
        return LinkedElfAdmissionService("fixture", callback, ((str(source), file_digest(source)),))

    return module, artifact, selected


def _evaluate(service, artifact, root):
    return service.evaluate(elf=artifact, target="fixture", evidence_root=root)


@pytest.mark.parametrize("change", ["code", "callback"])
def test_same_source_evaluator_substitution_refuses_before_evaluation(owned, tmp_path, change):
    module, artifact, selected = owned
    service = selected(module.original)
    service.verify("fixture")
    if change == "code":
        module.original.__code__ = module.replacement.__code__
    else:
        object.__setattr__(service, "evaluator", module.replacement)
    with pytest.raises(ValueError, match="selected evaluator"):
        _evaluate(service, artifact, tmp_path / "gate")
    assert not (tmp_path / "gate").exists()


def test_actual_evaluator_code_mutation_refuses_after_return(owned, tmp_path):
    module, artifact, selected = owned
    service = selected(module.changing)
    with pytest.raises(ValueError, match="selected evaluator"):
        _evaluate(service, artifact, tmp_path / "gate")
    assert (tmp_path / "gate/report.json").is_file()  # Retain the actual failed evaluation.


def test_exact_method_owner_stays_selected_while_normal_state_can_change(owned, tmp_path):
    module, artifact, selected = owned
    owner = module.Owner()
    service = selected(owner.evaluate)
    first = _evaluate(service, artifact, tmp_path / "first")
    assert first["status"] == "accepted" and owner.calls == 1
    second = _evaluate(service, artifact, tmp_path / "second")
    assert second["status"] == "accepted" and owner.calls == 2
    object.__setattr__(service, "evaluator", MethodType(service.evaluator.__func__, module.Owner()))
    with pytest.raises(ValueError, match="selected evaluator"):
        service.revalidate(elf=artifact, result=second, target="fixture")


@pytest.mark.parametrize("change", ["replace", "remove", "add"])
def test_exact_partial_keyword_selection_refuses_drift(owned, tmp_path, change):
    module, artifact, selected = owned
    callback = partial(module.original, status="refused")
    service = selected(callback)
    result = _evaluate(service, artifact, tmp_path / "gate")
    assert result["status"] == "refused"
    if change == "replace":
        callback.keywords["status"] = "accepted"
    elif change == "remove":
        del callback.keywords["status"]
    else:
        callback.keywords["state"] = []
    with pytest.raises(ValueError, match="selected evaluator"):
        service.revalidate(elf=artifact, result=result, target="fixture")


def test_nested_partial_keeps_binding_identity_without_claiming_reachable_state(owned, tmp_path):
    module, artifact, selected = owned
    state = []
    callback = partial(partial(module.original, status="refused"), state=state)
    service = selected(callback)
    result = _evaluate(service, artifact, tmp_path / "gate")
    assert result["status"] == "refused" and state == ["refused"]
    state.append("independent mutable state")
    assert service.revalidate(elf=artifact, result=result, target="fixture") == "refused"


@pytest.mark.parametrize("kind", ["callable_object", "partial_subclass"])
def test_dynamic_callable_spoofing_refuses_before_invocation(owned, tmp_path, kind):
    module, artifact, selected = owned
    callback = module.Spoof() if kind == "callable_object" else module.PartialSubclass(module.original)
    service = selected(callback)
    with pytest.raises(ValueError, match="actual Python function or bound method"):
        _evaluate(service, artifact, tmp_path / "gate")
    assert not (tmp_path / "gate").exists()


CONSOLE = "OUT_B64_BEGIN v1 out 1 2 1 s\nOUT_B64_CHUNK 00000000 0002 AQI=\nOUT_B64_END\nDONE\n"


def _render(*args, **kwargs):
    raise AssertionError("the diagnostic substitutes target building")


def _ordinary(tmp_path, monkeypatch, owned, *, mutation=None, status="accepted"):
    module, _artifact, selected = owned
    gate = selected(partial(module.original, status=status))
    dispatched = []

    def mutate():
        module.original.__code__ = module.replacement.__code__

    def build(_cb, _llvm, work, **kwargs):
        work.mkdir()
        elf = work / "artifact"
        elf.write_bytes(b"owned diagnostic artifact; not a guest executable")
        if mutation == "build":
            mutate()
        return elf

    def run(elf, **kwargs):
        dispatched.append(elf)
        if mutation == "execution":
            mutate()
        return CONSOLE

    def parse(console):
        if mutation == "parser":
            mutate()
        elif mutation == "report":
            (tmp_path / "run/elf_admission/report.json").write_text("{}")
        elif mutation == "elf":
            (tmp_path / "run/artifact").write_bytes(b"changed after the actual gate")
        return parse_console(console)

    def receipt(**kwargs):
        return {"elf_sha256": file_digest(Path(kwargs["elf_path"]))}

    def no_backend(_target):
        raise AssertionError("explicit diagnostic must not discover target support")

    from merlin.runtime.backends import base

    monkeypatch.setattr(base, "get_backend", no_backend)
    monkeypatch.setattr(compiler, "compile_lowered_to_elf", build)
    monkeypatch.setattr(RB, "require_current_build_receipt", receipt)
    owner = Path(__file__).resolve()
    pins = ((str(owner), file_digest(owner)),)
    service = FunctionalExecutionService("fixture", "diagnostic", run, parse, pins, '{"scope":"diagnostic"}')
    recipe = HarnessBuildRecipe(tmp_path / "unused", (), (), tmp_path / "unused.ld", 0)
    build_service = BuildOnlyService("fixture", recipe, _render, pins)
    cb = {
        "kernel_abi": {"kind": "whole_program", "outputs": ["out"]},
        "tensors": {"out": {"shape": [1, 2], "dtype": "i8", "role": "output"}},
    }

    def invoke():
        return compiler.run_on_oracle(
            cb,
            "substituted lowering",
            simulator="diagnostic",
            target="fixture",
            workdir=tmp_path / "run",
            timeout=1,
            readback_policy=RB.ReadbackPolicy(RB.FULL_VALUES_B64),
            _build_service=build_service,
            _execution_service=service,
            _elf_admission=gate,
        )

    return invoke, dispatched


@pytest.mark.parametrize("mutation", ["build", "execution", "parser"])
def test_ordinary_gate_reopens_selected_callback_across_actual_call_boundaries(tmp_path, monkeypatch, owned, mutation):
    invoke, dispatched = _ordinary(tmp_path, monkeypatch, owned, mutation=mutation)
    with pytest.raises(ValueError, match="selected evaluator"):
        invoke()
    assert len(dispatched) == (0 if mutation == "build" else 1)


@pytest.mark.parametrize("mutation,reason", [("report", "report has no unchanged"), ("elf", "build identity changed")])
def test_complete_values_cannot_mask_parser_time_artifact_or_report_drift(
    owned, tmp_path, monkeypatch, mutation, reason
):
    invoke, dispatched = _ordinary(tmp_path, monkeypatch, owned, mutation=mutation)
    with pytest.raises(ValueError, match=reason):
        invoke()
    assert len(dispatched) == 1
    assert (tmp_path / "run/oracle_console.log").read_text() == CONSOLE


def test_ordinary_unchanged_selection_keeps_full_values(owned, tmp_path, monkeypatch):
    invoke, dispatched = _ordinary(tmp_path, monkeypatch, owned)
    result = invoke()
    assert result["outputs"] == {"out": [[1, 2]]} and len(dispatched) == 1
    assert result["oracle"]["derived_from_rtl"] is False


def test_original_refusal_never_dispatches_or_returns_numeric_values(owned, tmp_path, monkeypatch):
    invoke, dispatched = _ordinary(tmp_path, monkeypatch, owned, status="refused")
    result = invoke()
    assert result["status"] == "refused_before_execution" and not dispatched
    assert result["execution"] == "not_attempted" and "outputs" not in result
    assert "timing" not in result
