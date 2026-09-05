"""A whole-model capsule must be captured the way the MODEL declares, and must say what its inputs were.

Two failures this pins, both of which presented as "the compiler cannot build this model":

* the model's own capture declaration (its workload's ``capture.toml`` plus the curated fidelity knobs
  in ``baselines.bundle``) never reached the capture worker, so a loader that declines to invent its
  inputs raised and the capsule was recorded ``not_built`` -- a fact about the invocation, published as
  a fact about the compiler's reach;
* the workload's PINNED interpreter was ignored, so a loader whose dependency lives only in its own
  venv raised ``ModuleNotFoundError`` and was recorded the same way.

And the consequence of fixing the first: a capsule captured on a seeded synthetic input stream grades
COMPILER CORRECTNESS (the compiled program reproduces the reference the same loader produced on the same
operands) and nothing about the model's accuracy on real data. So the capsule must record which it ran
on, tri-state, and withhold the accuracy claim whenever that is not positively established.

Target-agnostic: nothing here names a hardware target; the declarations are keyed by MODEL.
"""
from __future__ import annotations

import json
import subprocess

import pytest
import yaml

from merlin.targetgen import capsule_source as CSrc

_LINALG = (
    "builtin.module {\n"
    "  func.func @forward(%0: tensor<2x2xf32>) -> tensor<2x2xf32> {\n"
    "    return %0 : tensor<2x2xf32>\n  }\n}\n")
_INPUTS = [[[1.0, 2.0], [3.0, 4.0]]]
_GOLDEN = [[1.0, 2.0], [3.0, 4.0]]


def _binding(operand_dtype="f32"):
    from merlin.targetgen import corpus_spec as CS
    return CS.CorpusBinding(
        target="t", tile_dim=16, operand_dtype=operand_dtype, accum_dtype="f32", integer=False,
        tiers=["L0", "L1"], compare="tolerance_float", atol=0.03125, rtol=0.015625,
        classes_for=lambda **_: [])


def _fake_m2m(tmp_path, workload: str, *, capture_toml: str = "") -> tuple:
    """A model2MLIR checkout shaped like the real one: an importable m2m, a workload, an interpreter."""
    root = tmp_path / "model2MLIR"
    (root / "m2m").mkdir(parents=True)
    (root / "m2m" / "__init__.py").write_text("", encoding="utf-8")
    wdir = root / "workloads" / workload
    wdir.mkdir(parents=True)
    (wdir / "loader.py").write_text("def get_model_and_inputs(): ...\n", encoding="utf-8")
    if capture_toml:
        (wdir / "capture.toml").write_text(capture_toml, encoding="utf-8")
    python = root / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o755)
    return root, python


def _stub_worker(monkeypatch, seen: dict, *, meta_extra: dict | None = None):
    """Stand in for the capture worker: record how it was invoked, write a minimal clean capture."""
    def fake_run(cmd, **kwargs):
        seen["cmd"] = list(cmd)
        seen["env"] = dict(kwargs.get("env") or {})
        out = None
        for i, tok in enumerate(cmd):
            if tok == "--out":
                out = cmd[i + 1]
        from pathlib import Path
        d = Path(out)
        d.mkdir(parents=True, exist_ok=True)
        (d / "linalg.mlir").write_text(_LINALG, encoding="utf-8")
        (d / "inputs.json").write_text(json.dumps(_INPUTS), encoding="utf-8")
        (d / "golden.json").write_text(json.dumps(_GOLDEN), encoding="utf-8")
        meta = {"ok": True, "opaque": 0, "opaque_detail": {}, "dtype": "f32",
                "input_abi": [{"shape": [2, 2], "dtype": "f32"}],
                "weights": str(d / "weights.safetensors")}
        meta.update(meta_extra or {})
        (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(CSrc.subprocess, "run", fake_run)
    # the capture cache is exercised on its own below; here every call must reach the (stub) worker
    monkeypatch.setattr(CSrc.PytorchRefSource, "_cache_slot", lambda *a, **k: None)


def _write(tmp_path, monkeypatch, *, workload="acme_net_v2", capture_toml="",
           meta_extra=None, entry_model=None) -> tuple:
    root, python = _fake_m2m(tmp_path, workload, capture_toml=capture_toml)
    monkeypatch.setenv("MERLIN_M2M_DIR", str(root))
    monkeypatch.setenv("MERLIN_MODEL2MLIR", str(root))
    seen: dict = {}
    _stub_worker(monkeypatch, seen, meta_extra=meta_extra)
    src = CSrc.PytorchRefSource(m2m_dir=root, python=python)
    entry = {"name": "SY_model_x", "kind": "model", "cat": "model", "op": "model",
             "model": entry_model or workload, "operand_dtype": "f32", "label": "public"}
    d = CSrc.write_model_capsule(entry, _binding(), tmp_path / "out", source=src)
    return d, seen


# --- the declaration must reach the capture -------------------------------------------------------

def test_declared_loader_env_reaches_the_capture_worker(tmp_path, monkeypatch):
    """A location the workload declares for its loader is in the worker's environment."""
    home = tmp_path / "weight_cache"
    home.mkdir()
    d, seen = _write(tmp_path, monkeypatch,
                     capture_toml=f'[env]\nACME_HOME = "{home}"\nACME_LAYERS = "2"\n')
    assert seen["env"].get("ACME_HOME") == str(home), (
        "the model's own declared loader environment must reach the worker; without it a loader that "
        "declines to invent its inputs raises and the model is recorded as one that cannot be built")
    # a smoke-fidelity knob is NOT replayed (it would capture a smaller model than the golden)
    assert "ACME_LAYERS" not in seen["env"]
    assert d.is_dir()


def test_declared_interpreter_is_used_when_the_workload_pins_one(tmp_path, monkeypatch):
    """A workload whose stack lives in its own venv is captured with THAT python."""
    own = tmp_path / "own_venv"
    (own / "bin").mkdir(parents=True)
    (own / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    (own / "bin" / "python").chmod(0o755)
    _d, seen = _write(tmp_path, monkeypatch, capture_toml=f'venv = "{own}"\n')
    assert seen["cmd"][0] == str(own / "bin" / "python"), (
        "the workload pins the interpreter its dependencies live in; ignoring it turns a missing "
        "module into 'this model cannot be captured'")


def test_a_workload_that_pins_nothing_keeps_the_default_interpreter(tmp_path, monkeypatch):
    _d, seen = _write(tmp_path, monkeypatch)
    assert seen["cmd"][0].endswith("model2MLIR/.venv/bin/python")


def test_capture_cache_is_keyed_on_the_declared_environment(tmp_path, monkeypatch):
    """Two different input declarations are two different captures, so they cannot share a slot."""
    root, python = _fake_m2m(tmp_path, "acme_net_v2")
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out-root"))
    src = CSrc.PytorchRefSource(m2m_dir=root, python=python)
    a = src._cache_slot("model", "f32", "src", None, {"ACME_INPUTS": "real"}, python)
    b = src._cache_slot("model", "f32", "src", None, {"ACME_INPUTS": "synthetic"}, python)
    if a is None or b is None:
        pytest.skip("artifact cache unavailable in this environment")
    assert a != b, "a capture taken under one input declaration must not be served for another"


# --- what the capsule records about its inputs ----------------------------------------------------

def _prov(d):
    """The capsule's input record. It is on ``capsule.yaml`` -- the PUBLIC half, which is what a reader
    has in hand when they quote a pass -- and on the golden, whose provenance is incomplete without
    saying what the reference was computed over."""
    cap = yaml.safe_load((d / "capsule.yaml").read_text(encoding="utf-8"))
    golden = yaml.safe_load((d / "golden.yaml").read_text(encoding="utf-8"))
    block = cap.get("input_provenance")
    assert block, "a whole-model capsule must always record where its inputs came from"
    assert golden["oracle_provenance"].get("input_provenance") == block, (
        "the golden's provenance and the capsule's must be the same record, not two answers")
    return block


def test_synthetic_inputs_are_recorded_and_withhold_the_accuracy_claim(tmp_path, monkeypatch):
    d, _ = _write(tmp_path, monkeypatch, meta_extra={
        "loader_provenance_status": "declared",
        "loader_paper_ready": False,
        "loader_provenance": {"input_source": "synthetic_seed_20260830", "synthetic_inputs": True,
                              "checkpoint": "vendor/acme_net/V2", "full_checkpoint": True}})
    block = _prov(d)
    assert block["synthetic_inputs"] is True
    assert block["accuracy_claim_supported"] is False
    assert "SYNTHETIC" in block["accuracy_claim_withheld_because"]
    assert block["grades"] == "compiler_correctness"
    # the loader's own words are kept, so the capsule is checkable against the capture
    assert block["declared"]["checkpoint"] == "vendor/acme_net/V2"


def test_undeclared_input_provenance_fails_closed_as_unknown(tmp_path, monkeypatch):
    """An undeclared capture is neither real nor synthetic -- and cannot back a claim either way."""
    d, _ = _write(tmp_path, monkeypatch)
    block = _prov(d)
    assert block["synthetic_inputs"] == "unknown", "an unknown must never be recorded as a verdict"
    assert block["accuracy_claim_supported"] is False
    assert "UNKNOWN" in block["accuracy_claim_withheld_because"]


def test_real_attributed_inputs_do_support_the_claim(tmp_path, monkeypatch):
    """The claim field is DERIVED, not hardwired off: a certified real-data capture supports it."""
    d, _ = _write(tmp_path, monkeypatch, meta_extra={
        "loader_provenance_status": "declared",
        "loader_paper_ready": True,
        "loader_provenance": {"input_source": "curated_validation_split", "synthetic_inputs": False,
                              "preprocessing": "VENDOR_V2", "full_checkpoint": True}})
    block = _prov(d)
    assert block["synthetic_inputs"] is False
    assert block["accuracy_claim_supported"] is True
    assert "accuracy_claim_withheld_because" not in block


def test_real_inputs_without_certification_still_withhold_the_claim(tmp_path, monkeypatch):
    d, _ = _write(tmp_path, monkeypatch, meta_extra={
        "loader_provenance_status": "declared", "loader_paper_ready": False,
        "loader_provenance": {"synthetic_inputs": False}})
    block = _prov(d)
    assert block["accuracy_claim_supported"] is False


def test_a_declaration_that_raises_is_recorded_rather_than_swallowed(tmp_path, monkeypatch):
    d, _ = _write(tmp_path, monkeypatch, meta_extra={
        "loader_provenance_status": "error",
        "loader_provenance_error": "RuntimeError: the input stream was not attached"})
    block = _prov(d)
    assert block["status"] == "error" and block["synthetic_inputs"] == "unknown"
    assert "the input stream was not attached" in block["declaration_error"]


# --- the real claim model's declaration -----------------------------------------------------------

_M2M = CSrc.PytorchRefSource()


@pytest.mark.skipif(not (CSrc._m2m_dir() / "workloads").is_dir(),
                    reason="model2MLIR checkout unavailable; set MERLIN_M2M_DIR")
def test_a_roster_name_resolves_to_its_revisioned_workloads_declaration():
    """The roster names a model, the declaration is filed under the WORKLOAD -- and a claim model whose
    loader needs an input declaration must get one, or its capsule can never be built."""
    entry = {"cat": "model", "kind": "model", "op": "model", "name": "SY_model_x", "model": "resnet50"}
    workload = CSrc.resolve_model_workload(entry)
    assert workload and workload.startswith("resnet50"), workload
    assert CSrc.model_capture_env(workload), (
        f"workload {workload!r} declares an input source its loader requires; an empty environment "
        f"here is how the capsule came to be recorded as unbuildable")
