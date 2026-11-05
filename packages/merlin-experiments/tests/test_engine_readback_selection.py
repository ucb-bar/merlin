"""Policy admission and original per-engine routing, without running simulators."""

from __future__ import annotations

import json

import pytest

from merlin.runtime.backends import base
from merlin.runtime.backends.chipyard_rocc_bound import BoundRoCCBackend
from merlin.targetgen import capsule_runner as CR
from merlin.targetgen import oracle_readback as OR
from merlin.targetgen.contract import readback_policy as RB


@pytest.fixture
def logical_backend(monkeypatch):
    # Substitute deployment/tool validation only. The actual bound backend's
    # engine-policy and memory-transport selection methods remain in use.
    selection = {
        "contract": {
            "runner": {
                "chipyard_rocc": {
                    "engines": {"spike": {}, "gsim": {"readback_transport": RB.COHERENT_DUMP_V1}},
                    "toolchain": {
                        "headers": [
                            {"id": name} for name in ("htif", "out_b64", "out_bin", "out_bin_memory", "out_packet_v2")
                        ]
                    },
                }
            }
        },
    }
    monkeypatch.setattr(BoundRoCCBackend, "_engine", lambda self, engine: self._block()["engines"][engine])
    monkeypatch.setattr(BoundRoCCBackend, "readback_policy", lambda self: RB.ReadbackPolicy(RB.FULL_VALUES_B64))
    backend = BoundRoCCBackend("fixture", json.dumps(selection), None)
    monkeypatch.setattr(base, "get_backend", lambda target: backend)
    return backend


@pytest.mark.parametrize("transport", RB.MEMORY_TRANSPORTS)
def test_global_memory_override_refuses_spike_during_adapter_construction(logical_backend, transport):
    policy = RB.ReadbackPolicy(transport)
    with pytest.raises(ValueError, match="incompatible with selected engine 'spike'"):
        CR.simulator_adapter("spike", "fixture", readback_policy=policy)


def test_explicit_gsim_policy_must_match_original_engine(logical_backend):
    policy = RB.ReadbackPolicy(RB.COHERENT_DUMP_V1)
    adapter = CR.simulator_adapter("gsim", "fixture", readback_policy=policy)
    assert adapter._merlin_readback_policy is policy
    with pytest.raises(ValueError, match="incompatible with selected engine 'gsim'"):
        CR.simulator_adapter("gsim", "fixture", readback_policy=RB.ReadbackPolicy(RB.COHERENT_PACKET_V1))


def test_unset_override_keeps_engine_specific_full_value_readers(logical_backend, monkeypatch, tmp_path):
    from merlin_experiments.phase1.feedback import native_memory_readback as MR

    from merlin.targetgen.contract import compile as C
    from merlin.targetgen.rtl import facts

    calls = []
    monkeypatch.setattr(facts, "rtl_facts_path", lambda target: tmp_path / "facts.json")
    monkeypatch.setattr(MR, "select_memory_engine", lambda **kwargs: ({"fixture": True}, lambda: None))
    values = {"Y": [[3, -7]]}
    monkeypatch.setattr(C, "run_on_oracle", lambda *a, **kw: calls.append(kw) or {"outputs": values})
    cb = {
        "tensors": {"Y": {"shape": [1, 2], "dtype": "i32", "role": "output"}},
        "kernel_abi": {"kind": "whole_program", "args": [{"tensor": "Y", "access": "write"}], "outputs": ["Y"]},
    }
    for engine in ("spike", "gsim"):
        result = OR._run(
            cb,
            "original LLVM",
            sim=engine,
            target="fixture",
            backend=logical_backend,
            workdir=tmp_path,
            timeout=1,
            policy=None,
        )
        assert result["outputs"] == values
    assert calls[0]["readback_policy"] == RB.ReadbackPolicy(RB.FULL_VALUES_B64)
    assert "memory_readback" not in calls[0]
    assert calls[1]["readback_policy"] == RB.ReadbackPolicy(RB.COHERENT_DUMP_V1)
    assert type(calls[1]["memory_readback"]) is MR.NativeLogicalMemoryReadback
    adapters = {engine: CR.simulator_adapter(engine, "fixture") for engine in ("spike", "gsim")}
    assert CR._adapter_readback_policy(adapters) is None


def test_direct_dispatch_rechecks_policy_instead_of_silently_replacing_it(logical_backend, monkeypatch, tmp_path):
    from merlin.targetgen.contract import compile as C

    monkeypatch.setattr(C, "run_on_oracle", lambda *a, **kw: pytest.fail("incompatible policy reached execution"))
    with pytest.raises(ValueError, match="incompatible with selected engine 'spike'"):
        OR._run(
            {},
            "LLVM",
            sim="spike",
            target="fixture",
            backend=logical_backend,
            workdir=tmp_path,
            timeout=1,
            policy=RB.ReadbackPolicy(RB.COHERENT_DUMP_V1),
        )


@pytest.mark.parametrize("transport", (RB.FULL_VALUES_B64, RB.FULL_VALUES_BIN))
def test_explicit_console_policies_preserve_identity_and_legacy_construction(monkeypatch, transport):
    monkeypatch.setattr(base, "get_backend", lambda target: pytest.fail("console adapter eagerly discovered backend"))
    policy = RB.ReadbackPolicy(transport)
    assert CR.simulator_adapter("spike", "fixture", readback_policy=policy)._merlin_readback_policy is policy


def test_legacy_memory_backend_keeps_its_original_reader_admission():
    # Legacy HTIF memory export on Spike is distinct from the logical reader.
    OR.validate_engine_policy(RB.ReadbackPolicy(RB.COHERENT_DUMP_V1), sim="spike", backend=object())
