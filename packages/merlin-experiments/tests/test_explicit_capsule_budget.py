"""Per-call capsule deadlines do not mutate process-wide environment."""

import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from merlin.targetgen import capsule_runner as C


@pytest.mark.parametrize("inherited,expected", [(None, 17), ("5", 5), ("50", 17), ("invalid", 17)])
def test_explicit_budget_clips_original_environment_without_mutation(monkeypatch, inherited, expected):
    if inherited is None:
        monkeypatch.delenv("MERLIN_MODEL_BUDGET_S", raising=False)
    else:
        monkeypatch.setenv("MERLIN_MODEL_BUDGET_S", inherited)
    before = dict(os.environ)
    assert C.model_budget_seconds(override=17) == expected
    assert dict(os.environ) == before


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf"), "17"])
def test_invalid_explicit_budget_is_not_an_unlimited_child(value):
    with pytest.raises(ValueError, match="finite and positive"):
        C.model_budget_seconds(override=value)


def test_parallel_budget_calls_keep_independent_values(monkeypatch):
    monkeypatch.delenv("MERLIN_MODEL_BUDGET_S", raising=False)
    with ThreadPoolExecutor(max_workers=2) as workers:
        assert list(workers.map(lambda value: C.model_budget_seconds(override=value), (7, 19))) == [7, 19]
    assert "MERLIN_MODEL_BUDGET_S" not in os.environ


def test_actual_model_dispatch_receives_explicit_sealed_deadline(tmp_path, monkeypatch):
    from merlin.targetgen import capsule_candidate_context as CONTEXT
    from merlin.targetgen import checked_source_model_diagnostic as CHECKED
    from merlin.targetgen import owned_host_model_selection as OWNED

    monkeypatch.delenv("MERLIN_MODEL_BUDGET_S", raising=False)
    config = SimpleNamespace(force_match_policy=None, target="fixture", suite="tests", dtype="i8xi8_i32")
    monkeypatch.setattr(C, "make_run_paths", lambda *_a, **_kw: SimpleNamespace(run_path=tmp_path / "run"))
    monkeypatch.setattr(CHECKED, "verify_host_selection", lambda *_a, **_kw: None)
    monkeypatch.setattr(OWNED, "owned_model_required", lambda *_a, **_kw: False)

    class ReachedOriginalSelectedWorker(Exception):
        pass

    def grade(*_a, **kwargs):
        assert kwargs["budget_s"] == 17
        assert kwargs["checked_source_diagnostic"] is selected
        raise ReachedOriginalSelectedWorker

    monkeypatch.setattr(CONTEXT, "grade_selected_model", grade)
    selected = object()  # Neutral routing control, never a source or timing grant.
    with pytest.raises(ReachedOriginalSelectedWorker):
        C.run_capsule(
            {"kind": "model", "name": "public", "numeric_policy": {"mode": "exact"}},
            tmp_path, runs_root=tmp_path / "runs", config=config, oracle_adapters={},
            checked_source_diagnostic=selected, model_budget_s=17,
        )


@pytest.mark.parametrize("inherited,expected", [(None, 30), ("7", 7), ("60", 30)])
def test_connected_source_slice_caps_unset_or_inherited_budget(tmp_path, monkeypatch, inherited, expected):
    from merlin.targetgen import capsule_candidate_context as CONTEXT
    from merlin.targetgen import owned_host_model_selection as OWNED
    from merlin.targetgen import owned_host_slice_execution as SLICE

    if inherited is None:
        monkeypatch.delenv("MERLIN_MODEL_BUDGET_S", raising=False)
    else:
        monkeypatch.setenv("MERLIN_MODEL_BUDGET_S", inherited)
    config = SimpleNamespace(force_match_policy=None, target="fixture", suite="tests", dtype="i8xi8_i32")
    monkeypatch.setattr(C, "make_run_paths", lambda *_a, **_kw: SimpleNamespace(run_path=tmp_path / "run"))
    monkeypatch.setattr(SLICE, "selected_source_slice", lambda _capsule: True)
    monkeypatch.setattr(OWNED, "owned_model_required", lambda *_a, **_kw: True)

    class ReachedOriginalSelectedWorker(Exception):
        pass

    def grade(*_a, **kwargs):
        assert kwargs["budget_s"] == expected
        raise ReachedOriginalSelectedWorker

    monkeypatch.setattr(CONTEXT, "grade_selected_model", grade)
    with pytest.raises(ReachedOriginalSelectedWorker):
        C.run_capsule(
            {"kind": "model_slice", "name": "public", "numeric_policy": {"mode": "exact"}},
            tmp_path, runs_root=tmp_path / "runs", config=config, oracle_adapters={}, timeout=30,
        )
