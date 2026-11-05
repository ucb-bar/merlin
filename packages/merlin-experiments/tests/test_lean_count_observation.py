"""Lean count routing preserves exact numerical, ELF, model and scope joins."""

import json
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import lean_count_observation as L


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    elf = tmp_path/"program.elf"
    elf.write_bytes(b"neutral ELF placeholder")
    model = tmp_path/"model.json"
    model.write_text("neutral frozen model")
    monkeypatch.setattr(L, "MODEL_SHA256", L.F.digest(model))
    native = dict(schema="merlin_candidate_native_model_execution_v1", elf=L.NATIVE._digest(elf),
        numeric=dict(status="pass", policy="exact_int", mismatch_count=0,
                     per_output=dict(Y=dict(status="pass", mismatch_count=0, n_elements=8))),
        source=dict(inputs=dict(sha256="a"*64), interface=dict(sha256="b"*64)))
    native_path = tmp_path/"native.json"
    native_path.write_text(json.dumps(native))
    evaluation = dict(native_result=L.NATIVE._digest(native_path), work_root=str(tmp_path), member="neutral")
    runtime = SimpleNamespace(model_path=model, record=lambda: dict(neutral=True), sha256="runtime")
    monkeypatch.setattr(L, "load_target_experiment", lambda _: object())
    monkeypatch.setattr(L.PC, "package_sandbox_policy", lambda *_: object())
    calls = []

    def observe(elf_path, **kwargs):
        assert elf_path == elf and kwargs["numerical_pass"] is True
        calls.append(kwargs)
        kwargs["directory"].mkdir()
        report = kwargs["directory"]/"counts.json"
        report.write_text(json.dumps(dict(model_sha256=L.MODEL_SHA256, scope_sha256="scope",
            estimated_cycles=100, counter_windows=[dict(begin_pc=1, end_pc=2, estimated_cycles=100)],
            timing_complete=True, unresolved=[])))
        return dict(status="observed", elf_sha256=L.NATIVE._digest(elf)["sha256"],
                    report=dict(path=str(report), sha256=L.F.digest(report)), complete_counts=True,
                    scope_binding=dict(begin_pc=1, end_pc=2))

    monkeypatch.setattr(L.OBS, "observe", observe)

    def call():
        return L.observe_evaluation(evaluation, runtime=runtime, target_experiment=tmp_path/"target.yaml",
            package=tmp_path/"package", directory=tmp_path/"counts", floor_cycles=30, remaining_s=100)

    return evaluation, native_path, native, runtime, calls, call


def test_counts_and_structural_ratio_remain_unqualified(fixture):
    _, _, _, _, calls, call = fixture
    result = call()
    assert len(calls) == 1
    assert result["status"] == "observed" and result["estimated_cycles"] == 100
    assert result["attainment_fraction"] == .3
    assert result["native_outputs_complete"] and result["feature_complete"] and result["timing_complete"]
    assert result["qualification"] == "UNQUALIFIED" and result["source_effects"] == "PENDING"
    assert result["formal_admission"] is False and result["roofline_convergence"] is False


@pytest.mark.parametrize("fault", ["numeric", "missing_output", "stale_receipt", "elf", "model"])
def test_inexact_or_changed_native_products_never_start_counts(fixture, fault):
    evaluation, native_path, native, runtime, calls, call = fixture
    if fault == "numeric":
        native["numeric"]["status"] = "fail"
    elif fault == "missing_output":
        native["numeric"]["per_output"] = {}
    elif fault == "elf":
        L.Path(native["elf"]["path"]).write_bytes(b"changed")
    elif fault == "model":
        runtime.model_path.write_text("changed")
    if fault in ("numeric", "missing_output", "stale_receipt"):
        native_path.write_text(json.dumps(dict(native, changed=True)))
        if fault != "stale_receipt":
            evaluation["native_result"] = L.NATIVE._digest(native_path)
    result = call()
    assert result["status"] == "UNKNOWN" and result["estimated_cycles"] is None
    assert not calls


def test_resource_refusal_is_unknown_and_preserved(fixture, monkeypatch):
    *_, call = fixture
    monkeypatch.setattr(L.OBS, "observe", lambda *a, **kw: dict(status="UNKNOWN", reason="900 second timeout"))
    result = call()
    assert result["status"] == "UNKNOWN" and result["reason"] == "900 second timeout"
    assert result["estimated_cycles"] is None and result["attainment_fraction"] is None


def test_count_scope_mismatch_does_not_produce_an_estimate(fixture, monkeypatch):
    *_, call = fixture
    original = L.OBS.observe

    def changed(*args, **kwargs):
        observed = original(*args, **kwargs)
        observed["scope_binding"]["begin_pc"] = 999
        return observed

    monkeypatch.setattr(L.OBS, "observe", changed)
    result = call()
    assert result["status"] == "UNKNOWN" and result["estimated_cycles"] is None


def test_model_estimate_below_structural_floor_cannot_reach_target(fixture, monkeypatch):
    *_, call = fixture
    original = L.OBS.observe

    def below_floor(*args, **kwargs):
        observed = original(*args, **kwargs)
        path = L.Path(observed["report"]["path"])
        report = json.loads(path.read_bytes())
        report["estimated_cycles"] = report["counter_windows"][0]["estimated_cycles"] = 20
        path.write_text(json.dumps(report))
        observed["report"]["sha256"] = L.F.digest(path)
        return observed

    monkeypatch.setattr(L.OBS, "observe", below_floor)
    result = call()
    assert result["status"] == "observed" and result["estimated_cycles"] == 20
    assert result["reason"] == "estimate_below_structural_bound"
    assert result["estimate_below_structural_bound"] is True and result["comparison_consistent"] is False
    assert result["attainment_fraction"] is None and result["estimated_attainment"] is None
    assert result["roofline_convergence"] is False
