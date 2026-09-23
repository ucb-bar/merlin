"""Provider deadlines come from the analysis owner, without executing a probe."""

import json
import socket
import subprocess
from types import SimpleNamespace

import pytest

from merlin.common.digest import sha256_bytes
from merlin.perf import host_region_qualifier as HR
from merlin.perf import isolated_probe_provider as IP
from merlin.perf.mechanism_probe import ProbeBinding
from merlin.targetgen import gsim_emulator
from merlin.targetgen.rocc import decode


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    def refused(*args, **kwargs):
        pytest.fail("synthetic provider tests cannot launch processes or listeners")

    monkeypatch.setattr(subprocess, "Popen", refused)
    monkeypatch.setattr(socket.socket, "bind", refused)


def test_isolated_probe_uses_analysis_timeout_for_preparation_and_final_budget(tmp_path, monkeypatch):
    short = tmp_path / "short.mlir"
    short.write_text("synthetic short source")
    elf = tmp_path / "primitive.elf"
    elf.write_bytes(b"synthetic executable bytes, never executed")
    prepared = {
        "workdir": str(tmp_path),
        "elf_sha256": sha256_bytes(elf.read_bytes()),
        "wrapper_sha256": "a" * 64,
        "primitive_mlir_sha256": "b" * 64,
        "domain_digest": "c" * 64,
    }
    engine = {"engine": "synthetic"}
    receipt = tmp_path / "runtime.json"
    receipt.write_text(
        json.dumps(
            {
                **prepared,
                "engine_provenance": engine,
                "correct": True,
                "warmup_runs": 1,
                "measured_runs": 1,
                "total_compute_cycles": 10,
                "elapsed_seconds": 1,
            }
        )
    )
    primitive = {"missing": [], "domain_digest": "c" * 64, "domain": {"kind": "fixture"}, "instruction_indices": [0]}
    monkeypatch.setattr(IP, "monotonic", lambda: 0.0)
    monkeypatch.setattr(IP, "initialized_compute_primitives", lambda *args, **kwargs: [primitive, primitive])
    monkeypatch.setattr(decode, "_parse_module", lambda _: object())
    monkeypatch.setattr(decode, "decode_module", lambda *args, **kwargs: [])
    monkeypatch.setattr(gsim_emulator, "citation", lambda _: engine)
    observed = []

    def prepare(*args, timeout_seconds, **kwargs):
        observed.append(("prepare", timeout_seconds))
        return prepared

    adapter = SimpleNamespace(
        prepare_primitive_probe=prepare,
        isolated_primitive_signature=lambda _: SimpleNamespace(to_dict=lambda: {"fixture": True}),
        runtime_elf_digest=lambda path: sha256_bytes(path.read_bytes()),
    )
    binding = ProbeBinding(*["d" * 64] * 4)
    analysis = SimpleNamespace(timeout_s=30)

    def compile_probe(*args, timeout_s):
        observed.append(("compile", timeout_s))
        return SimpleNamespace(returncode=0, stdout="synthetic lowered probe", stderr="")

    def charge(*args):
        observed.append(("charge", args[-1]))
        analysis.timeout_s = 9

    analysis.session = SimpleNamespace(
        current_probe_binding=lambda _: binding,
        current_artifacts=lambda _: {
            "lowered_text": "model",
            "candidate_lowered_sha256": sha256_bytes(b"model"),
            "decoded_trace": [],
        },
    )
    probes = SimpleNamespace(
        analysis=analysis,
        compile_probe_candidate=compile_probe,
        charge_probe_preparation=charge,
    )
    provider = IP.IsolatedPrimitiveProbeProvider(
        target="synthetic",
        short_interface=short,
        adapter=adapter,
        runtime_receipt=receipt,
        output=tmp_path / "output",
    )
    result = provider(candidate=tmp_path, probes=probes, timeout_s=20)
    assert observed == [("compile", 20.0), ("prepare", 20), ("charge", 0.0)]
    assert result["admission_inputs"]["budget"].timeout_seconds == 9
    assert not hasattr(probes, "timeout_s")
    # The returned execution closure is deliberately never invoked.


@pytest.mark.parametrize("owner_timeout", [0, 5])
def test_host_qualifier_reads_analysis_timeout_before_candidate_access(tmp_path, owner_timeout):
    calls = []

    def selected(*args):
        calls.append(args)
        raise ValueError("synthetic admission boundary reached")

    probes = SimpleNamespace(
        analysis=SimpleNamespace(
            timeout_s=owner_timeout,
            session=SimpleNamespace(selected_changed_portfolio_context=selected),
        ),
    )
    provider = HR.HostChangedRegionQualifier(
        native_layout=lambda _: {},
        expected_symbol="fixture",
        abi_provenance={"fixture": True},
        output=tmp_path,
    )
    if owner_timeout == 0:
        with pytest.raises(ValueError, match="positive remaining budget"):
            provider(candidate=tmp_path, probes=probes, timeout_s=20)
        assert calls == []
    else:
        result = provider(candidate=tmp_path, probes=probes, timeout_s=20)
        assert result["reason"] == "ValueError: synthetic admission boundary reached"
        assert len(calls) == 1
        assert result["full_model_executed"] is False
    assert not hasattr(probes, "timeout_s")
