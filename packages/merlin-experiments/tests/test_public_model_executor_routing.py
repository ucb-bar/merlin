"""Exercise public model runner wiring; synthetic records grant no qualification."""

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from merlin.targetgen import capsule_grade, capsule_runner, native_model_execution


@pytest.mark.parametrize("outcome", ["returned", "execution_refused", "capture_changed"])
def test_public_model_calls_native_executor_and_invalidates_changed_capture(tmp_path, monkeypatch, outcome):
    generated = tmp_path / "generated"
    generated.mkdir()
    (tmp_path / "capsule.yaml").write_text("name: fixture\nkind: model\n")
    (tmp_path / "capsule.interface.mlir").write_text("module {}")
    capsule = {"name": "fixture", "kind": "model", "__dir__": str(tmp_path)}
    paths = SimpleNamespace(generated=generated, run_path=tmp_path / "run")
    calls, observed = [], []
    command_buffer = {"kernel_abi": {"kind": "whole_program"}}
    bundle = {"fixture": "complete source inputs"}
    native = {"status": "synthetic executor return; no correctness authority"}

    def emit(*_args, **_kwargs):
        calls.append("emit")
        (generated / "command_buffer.json").write_text(json.dumps(command_buffer))
        (generated / "lowered.llvm.mlir").write_text("module {}")
        return object(), command_buffer, "module {}"

    def execute(**kwargs):
        calls.append("execute")
        observed.append(kwargs)
        if outcome == "execution_refused":
            raise ValueError("original execution refused")
        return native

    @contextmanager
    def capture(*_args, **_kwargs):
        def verify():
            calls.append("verify")
            if outcome == "capture_changed":
                raise ValueError("original captured inputs changed")

        yield bundle, {"scope": "synthetic transport fixture"}, verify

    monkeypatch.setattr(capsule_runner, "make_run_paths", lambda *_a, **_kw: paths)
    monkeypatch.setattr(capsule_runner, "run_entrypoints", emit)
    monkeypatch.setattr(capsule_runner, "_model_runtime_bundle", capture)
    monkeypatch.setattr(
        native_model_execution, "independent_frozen_source_eligibility", lambda *_a, **_kw: {"n_eligible": 1}
    )
    monkeypatch.setattr(native_model_execution, "execute_candidate_model", execute)
    monkeypatch.setattr(capsule_grade, "enforce_model_execution_check", lambda result, *_a, **_kw: result)
    from merlin.perf import work_volume

    monkeypatch.setattr(work_volume, "command_buffer_evidence", lambda *_a, **_kw: ({}, {}))
    for name in ("MERLIN_MODEL_NATIVE_SIMULATOR", "MERLIN_MODEL_NATIVE_RTL_FACTS", "MERLIN_MODEL_NATIVE_BOARD_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    context = dict(
        runs_root=str(tmp_path), run_id="fixture", suite="fixture", dtype="i8", contract={},
        package_prebuilt=True, fourth_output_name="lowered.llvm.mlir",
    )
    result = capsule_runner._grade_public_program_capsule_inline(
        capsule, target="fixture", timeout=10, package_dir=tmp_path, context=context, pkg=object()
    )

    assert calls[:2] == ["emit", "execute"]
    assert len(observed) == 1
    assert observed[0]["command_buffer"] is command_buffer
    assert observed[0]["capture_bundle"] is bundle
    assert observed[0]["out_dir"] == paths.run_path / "candidate_native"
    assert result["status"] == "incomplete"
    if outcome == "returned":
        assert calls == ["emit", "execute", "verify"]
        assert result["candidate_native_execution"] is native
        assert "candidate_emission_failure" not in result
    else:
        assert "candidate_native_execution" not in result
        assert result["candidate_emission_failure"]["type"] == "ValueError"
        assert result["candidate_emission_failure"]["stage"] == (
            "candidate_model_execution" if outcome == "execution_refused" else "verify_runtime_bundle"
        )
