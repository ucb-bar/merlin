"""The whole-model (kind == "model") capsule is graded by compiling the captured model through the
merlin whole-model flow and gating vs its golden. This checks the routing + verdict mapping without a real
compile (compile_rvv is monkeypatched): verified→pass, mismatch→fail, toolchain-absent→incomplete (honest,
never a silent pass)."""
from __future__ import annotations

import merlin.compile_cli as cc
from merlin.targetgen import capsule_runner as R

_CAP = {"name": "M0_small_llama", "kind": "model", "label": "public",
        "operation": {"op": "model", "attributes": {"model": "small_llama", "compile_dtype": "fp32"}}}


def test_model_grade_pass(monkeypatch):
    monkeypatch.setattr(cc, "compile_rvv", lambda *a, **k: {"status": "verified", "verify": {"gate_ok": True}})
    r = R._grade_model_capsule(_CAP, timeout=60)
    assert r["status"] == "pass" and r["kind"] == "model" and r["operation"]["model"] == "small_llama"


def test_model_grade_fail_on_mismatch(monkeypatch):
    monkeypatch.setattr(cc, "compile_rvv",
                        lambda *a, **k: {"status": "run_mismatch", "verify": {"gate_ok": False}})
    r = R._grade_model_capsule(_CAP, timeout=60)
    assert r["status"] == "fail" and r["failure"]["category"] == "FUNCTIONAL_MISMATCH"


def test_model_grade_incomplete_when_toolchain_absent(monkeypatch):
    def _boom(*a, **k):
        raise SystemExit("no clang-23 / no m2m venv")
    monkeypatch.setattr(cc, "compile_rvv", _boom)
    r = R._grade_model_capsule(_CAP, timeout=60)
    assert r["status"] == "incomplete" and r["failure"]["category"] == "NOT_RUN_IS_NOT_PASS"


def test_model_grade_not_run_reported(monkeypatch):
    monkeypatch.setattr(cc, "compile_rvv",
                        lambda *a, **k: {"status": "not_run", "reason": "Zephyr/spike unavailable"})
    r = R._grade_model_capsule(_CAP, timeout=60)
    assert r["status"] == "incomplete"
