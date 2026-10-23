"""Explicit simulator construction must not select another target or engine."""

import json
from types import SimpleNamespace

import pytest

from merlin.runtime.backends import base
from merlin.targetgen import capsule_runner as CR


def test_explicit_constructor_has_no_target_default_or_legacy_alias():
    assert not hasattr(CR, "default_adapters")
    assert not hasattr(CR, "_spike_verilator_adapter")
    with pytest.raises(TypeError):
        CR.simulator_adapter("spike")


@pytest.mark.parametrize("target", ["synthetic_alpha", "synthetic_beta"])
@pytest.mark.parametrize("sim", ["spike", "verilator"])
def test_explicit_target_engine_and_selection_are_preserved(monkeypatch, tmp_path, target, sim):
    calls = []

    def get_backend(requested):
        assert requested == target
        return SimpleNamespace(available=lambda selected: selected == sim)

    monkeypatch.setattr(base, "get_backend", get_backend)

    def run(cb, llvm, **kwargs):
        calls.append((cb, llvm, kwargs))
        return {"oracle": {"kind": "fixture"}, "outputs": {"result": 7}}

    monkeypatch.setattr(CR.oot_compile, "run_on_oracle", run)
    selection = {"engine": sim, "reason": "explicit experiment policy"}
    result = CR.simulator_adapter(sim, target, selection=selection)({"fixture": True}, "ir", tmp_path, 13)
    assert len(calls) == 1
    assert 0 < calls[0][2].pop("timeout") <= 13
    assert calls == [
        ({"fixture": True}, "ir", {"simulator": sim, "target": target, "workdir": tmp_path})
    ]
    assert result["outputs"] == {"result": 7}
    assert result["oracle"]["selection"] == selection
    assert result["oracle"]["selection"] is not selection


def test_explicit_unavailable_engine_never_substitutes(monkeypatch, tmp_path):
    monkeypatch.setattr(base, "get_backend", lambda target: SimpleNamespace(available=lambda sim: False))
    monkeypatch.setattr(CR.oot_compile, "run_on_oracle", lambda *a, **kw: pytest.fail("unavailable engine executed"))
    with pytest.raises(CR.OracleUnavailable, match="verilator not available"):
        CR.simulator_adapter("verilator", "synthetic")({}, "ir", tmp_path, 1)


def test_gsim_cannot_certify_against_different_selected_firrtl(monkeypatch, tmp_path):
    from merlin.targetgen import gsim_emulator

    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps({"inputs": {"fir_sha256": "a" * 64}}), encoding="utf-8")
    monkeypatch.setenv("MERLIN_RTL_FACTS", str(facts))
    monkeypatch.setattr(
        base,
        "get_backend",
        lambda target: SimpleNamespace(available=lambda sim: True, GSIM_EMU_ENV="TEST_GSIM"),
    )
    monkeypatch.setattr(
        gsim_emulator,
        "resolve",
        lambda target, *, env_var: SimpleNamespace(
            ok=True, receipt_status="bound", receipt={"firrtl_sha256": "b" * 64}
        ),
    )
    monkeypatch.setattr(CR.oot_compile, "run_on_oracle", lambda *a, **kw: pytest.fail("mismatched GSIM ran"))
    with pytest.raises(CR.OracleUnavailable, match="selected FIRRTL"):
        CR.simulator_adapter("gsim", "synthetic")({}, "ir", tmp_path, 1)


def test_gsim_with_matching_selected_firrtl_can_run(monkeypatch, tmp_path):
    from merlin.targetgen import gsim_emulator

    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps({"inputs": {"fir_sha256": "a" * 64}}), encoding="utf-8")
    monkeypatch.setenv("MERLIN_RTL_FACTS", str(facts))
    monkeypatch.setattr(
        base,
        "get_backend",
        lambda target: SimpleNamespace(available=lambda sim: True, GSIM_EMU_ENV="TEST_GSIM"),
    )
    monkeypatch.setattr(
        gsim_emulator,
        "resolve",
        lambda target, *, env_var: SimpleNamespace(
            ok=True, receipt_status="bound", receipt={"firrtl_sha256": "a" * 64}
        ),
    )
    monkeypatch.setattr(CR.oot_compile, "run_on_oracle", lambda *a, **kw: {"outputs": {"result": 7}})
    assert CR.simulator_adapter("gsim", "synthetic")({}, "ir", tmp_path, 1)["outputs"] == {"result": 7}


def test_engine_selection_skips_gsim_with_wrong_firrtl(monkeypatch, tmp_path):
    from merlin.targetgen import gsim_emulator, oracle_policy

    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps({"inputs": {"fir_sha256": "a" * 64}}), encoding="utf-8")
    monkeypatch.setenv("MERLIN_RTL_FACTS", str(facts))
    monkeypatch.delenv("MERLIN_REQUIRED_RTL_ENGINE", raising=False)
    monkeypatch.setattr(
        base,
        "get_backend",
        lambda target: SimpleNamespace(
            gsim_status=lambda: (True, "receipted binary"),
            available=lambda engine: engine == "verilator",
            GSIM_EMU_ENV="TEST_GSIM",
        ),
    )
    monkeypatch.setattr(
        gsim_emulator,
        "resolve",
        lambda target, *, env_var: SimpleNamespace(
            ok=True, receipt_status="bound", receipt={"firrtl_sha256": "b" * 64}
        ),
    )
    selected = oracle_policy.chipyard_l3_selection("synthetic")
    assert selected["engine"] == "verilator"
    assert selected["considered"][1]["available"] is False
    assert "differs from model receipt" in selected["considered"][1]["reason"]
