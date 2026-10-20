"""Actual target resolution checks; no compiler, native owner or target is run."""

import hashlib

import pytest
from merlin_experiments.phase2.contracts import StageGateError
from merlin_experiments.phase2.public_diagnostic_launch import require_current_native_contract

from merlin.targetgen import target_registry


@pytest.fixture
def selection(tmp_path, monkeypatch):
    path = tmp_path / "selected.yaml"
    path.write_text("name: preflight-neutral\n")
    monkeypatch.setenv("MERLIN_TARGET_PATH", "")
    monkeypatch.setenv("MERLIN_TARGETS_DIR", str(tmp_path / "absent-targets"))
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("MERLIN_TARGET_CONTRACT", str(path))
    return path, {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_exact_current_native_selection_returns_detached_pin(selection):
    path, pin = selection
    result = require_current_native_contract("preflight-neutral", pin)
    assert result == pin and result is not pin
    assert target_registry.resolve("preflight-neutral").capability_contract_path == path
    assert not (path.parent / "out").exists()


def test_unselected_preflight_does_not_resolve_or_change_default(monkeypatch):
    def forbidden(_target):
        raise AssertionError("default must not resolve a target")

    monkeypatch.setattr(target_registry, "resolve", forbidden)
    assert require_current_native_contract("preflight-neutral") is None


@pytest.mark.parametrize("same_bytes", [False, True])
def test_wrong_ambient_contract_including_identical_copy_refused(selection, monkeypatch, same_bytes):
    path, pin = selection
    other = path.with_name("other.yaml")
    other.write_bytes(path.read_bytes() if same_bytes else b"name: foreign\n")
    monkeypatch.setenv("MERLIN_TARGET_CONTRACT", str(other))
    with pytest.raises(StageGateError, match="contract path differs"):
        require_current_native_contract("preflight-neutral", pin)


def test_observed_contract_path_takes_precedence_over_ambient_selection(selection, monkeypatch):
    path, pin = selection
    monkeypatch.setenv("MERLIN_TARGET_CONTRACT", str(path.with_name("absent.yaml")))
    with target_registry.observed_contract("preflight-neutral", {"name": "preflight-neutral"}, source_path=path):
        assert require_current_native_contract("preflight-neutral", pin) == pin
    with pytest.raises(StageGateError, match="contract path differs"):
        require_current_native_contract("preflight-neutral", pin)


@pytest.mark.parametrize("fault", ["stale", "missing", "link", "relative", "extra", "bad-sha"])
def test_changed_or_nonordinary_expected_contract_refused(selection, fault):
    path, pin = selection
    if fault == "stale":
        path.write_text("name: changed\n")
    elif fault == "missing":
        path.unlink()
    elif fault == "link":
        saved = path.with_name("saved.yaml")
        path.rename(saved)
        path.symlink_to(saved)
    elif fault == "relative":
        pin["path"] = path.name
    elif fault == "extra":
        pin["qualified"] = True
    else:
        pin["sha256"] = "z" * 64
    with pytest.raises(StageGateError):
        require_current_native_contract("preflight-neutral", pin)


def test_selection_time_source_mutation_is_reopened(selection, monkeypatch):
    path, pin = selection
    original = target_registry.resolve

    def mutate(target):
        selected = original(target)
        path.write_text("name: changed-during-resolution\n")
        return selected

    monkeypatch.setattr(target_registry, "resolve", mutate)
    with pytest.raises(StageGateError, match="differs from its selected bytes"):
        require_current_native_contract("preflight-neutral", pin)
