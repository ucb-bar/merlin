"""Normal fixed PME routing controls with synthetic observations, no simulators."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import paired_measurement as PM

from merlin.runtime.backends import base
from merlin.targetgen.contract import artifact_retesting as A


def _adapter_inputs(tmp_path, monkeypatch):
    binary = tmp_path / "synthetic-engine"
    binary.write_text("not an executable or qualified engine")
    certificate = SimpleNamespace(pins={
        "gsim_binary": {"sha256": PM.sha256_file(binary)},
        "gsim_firrtl": {"sha256": "b" * 64}, "gsim_model": {"sha256": "c" * 64},
    })
    monkeypatch.setattr(base, "get_backend", lambda target: SimpleNamespace(gsim_path=lambda: binary))
    monkeypatch.setattr(PM, "_L3_MEMO", {})
    calls, evidence = [], {}
    cb = {"tensors": {"out": {"shape": [1], "dtype": "i8", "role": "output"}}}

    def observe(cb, llvm, *, target, workdir, timeout):
        calls.append((cb, llvm, workdir))
        workdir.mkdir()
        elf = workdir / "fresh.elf"
        elf.write_text("fresh full harness control")
        return {
            "elf": str(elf), "outputs": {"out": [1]}, "cycles": 7,
            "oracle": {"kind": "synthetic_only", "derived_from_rtl": False},
            "artifact_retesting": {"decision": "execute", "execution_reused": False},
        }

    monkeypatch.setattr(PM.DIGEST, "run_gsim", observe)

    class ForbiddenLegacyStore:
        def get(self, *args):
            pytest.fail("ordinary retesting consulted the prebuild LLVM store")

        def put(self, *args):
            pytest.fail("ordinary retesting wrote the prebuild LLVM store")

    return SimpleNamespace(**locals())


@pytest.mark.parametrize("selection", ["queued_public", "fresh_final"])
def test_queue_and_final_bypass_legacy_prebuild_measurements(tmp_path, monkeypatch, selection):
    f = _adapter_inputs(tmp_path, monkeypatch)
    key = PM._l3_memo_key(f.cb, "llvm", f.certificate.pins["gsim_binary"]["sha256"], "observation")
    PM._L3_MEMO[key] = {"result": {"cycles": 9999}, "evidence": {"cycles": 9999}}
    context = A.observe_public_artifact_retests(tmp_path, context={}) if selection == "queued_public" else nullcontext()
    with context:
        run = PM._gsim_l3_adapter(
            "synthetic", f.evidence, f.certificate, reuse_scope="observation",
            store=f.ForbiddenLegacyStore(), reuse_measurement=selection != "fresh_final",
        )
        first = run(f.cb, "llvm", tmp_path / "first", 5)
        second = run(f.cb, "llvm", tmp_path / "second", 5)
    assert len(f.calls) == 2 and first["cycles"] == second["cycles"] == 7
    assert f.evidence["gsim"]["reused_measurement"] is False
    assert f.evidence["gsim"]["artifact_retesting"]["decision"] == "execute"
    assert PM._L3_MEMO[key]["result"]["cycles"] == 9999


def test_legacy_default_measurement_reuse_remains_explicit(tmp_path, monkeypatch):
    f = _adapter_inputs(tmp_path, monkeypatch)
    store = PM.L3MeasurementStore(tmp_path / "legacy-store")
    run = PM._gsim_l3_adapter("synthetic", f.evidence, f.certificate, reuse_scope="observation", store=store)
    run(f.cb, "llvm", tmp_path / "first", 5)
    second = run(f.cb, "llvm", tmp_path / "second", 5)
    assert len(f.calls) == 1 and second["reused_measurement"] is True
    assert second["elf"] is None and f.evidence["gsim"]["reused_measurement"]


@pytest.mark.parametrize("phase, allowed", [("tuning", True), ("held_out", False), ("protected_final", False)])
def test_original_pme_final_phase_dispatch_requires_fresh_measurement(tmp_path, monkeypatch, phase, allowed):
    package, capsule = tmp_path / "package", tmp_path / "capsule"
    package.mkdir()
    capsule.mkdir()
    (package / "source").write_text("candidate")
    (capsule / "source").write_text("input")
    member = SimpleNamespace(source_dir=capsule, source_sha256=PM.hash_tree(capsule)["sha256"])
    decision = SimpleNamespace(workload_sha256="d" * 64, to_dict=lambda: {})
    spec = PM.ExecutionSpec(
        0, 0, "pair", phase, "candidate", "family", "capsule", "r000", package,
        PM.hash_tree(package)["sha256"], member, {}, decision, object(),
    )
    observed = []
    monkeypatch.setattr(PM, "_kernel_record", lambda member: {})
    monkeypatch.setattr(PM.PC, "package_sandbox_policy", lambda *args: None)
    monkeypatch.setattr(PM.PC, "boxed_entrypoints", lambda policy: nullcontext())

    def run(*args, **kwargs):
        observed.append(kwargs["reuse_measurement"])
        return {"scope": "synthetic routing only"}

    monkeypatch.setattr(PM, "_run_el4_compiler_engines", run)
    PM.run_execution(spec, tmp_path / "execution", 5, SimpleNamespace(target="synthetic"), {},
                     contract_root=tmp_path, hardware_counters=False)
    assert observed == [allowed]
