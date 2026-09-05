"""Legacy audit/demo surfaces must not escape the experiment-wide RTL-engine policy."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir


HARNESS = merlin_dir() / "experiments/capsule_bench/harness"
TASK = merlin_dir() / "experiments/capsule_bench/targets/gemmini/task/TASK_realistic.md"


def _module(name: str):
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    spec = importlib.util.spec_from_file_location(f"legacy_policy_{name}", HARNESS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_suite_audit_uses_the_central_policy_adapter(monkeypatch):
    audit = _module("full_suite_audit")
    gsim = object()
    calls = []
    monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", "gsim")
    monkeypatch.setattr(audit, "_sim_via", lambda: "chipyard")
    monkeypatch.setattr(
        audit.CR,
        "oracle_adapters",
        lambda target, sim_via: calls.append((target, sim_via)) or {"L2": object(), "L3": gsim},
    )
    monkeypatch.setattr(
        audit.CR,
        "_spike_verilator_adapter",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("the audit bypassed central engine policy")
        ),
    )

    assert audit._adapters_for(["L3"]) == {"L3": gsim}
    assert calls == [(audit.C.TARGET, "chipyard")]


def test_full_suite_gsim_required_path_never_dispatches_verilator(monkeypatch, tmp_path):
    audit = _module("full_suite_audit")
    dispatched = []

    class Backend:
        @staticmethod
        def available(engine):
            return engine == "gsim"

    monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", "gsim")
    monkeypatch.setattr(audit, "_sim_via", lambda: "chipyard")
    from merlin.runtime.backends import base as backends
    monkeypatch.setattr(backends, "get_backend", lambda _target: Backend())
    monkeypatch.setattr(
        audit.CR.oot_compile,
        "run_on_oracle",
        lambda _cb, _llvm, *, simulator, **_kwargs: dispatched.append(simulator) or {"status": "pass"},
    )

    adapter = audit._adapters_for(["L3"])["L3"]
    adapter({}, "module {}", tmp_path, 1)
    assert dispatched == ["gsim"]


def test_verilator_mutation_demo_is_explicitly_qualification_only(monkeypatch):
    demo = _module("demo_prescreen_mutation")
    monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", "gsim")
    with pytest.raises(RuntimeError, match="qualification-only.*GSIM-required"):
        demo._authorize_verilator_qualification(True)

    monkeypatch.delenv("MERLIN_REQUIRED_RTL_ENGINE")
    with pytest.raises(RuntimeError, match="explicit --one-time-verilator-qualification"):
        demo._authorize_verilator_qualification(False)


def test_realistic_task_derives_rtl_engine_instead_of_naming_verilator():
    text = TASK.read_text(encoding="utf-8")
    assert "MERLIN_REQUIRED_RTL_ENGINE" in text
    assert "simjob.py submit --sim verilator" not in text
    assert "verilator (l3) cert" not in text.lower()
