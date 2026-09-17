"""The target campaign executor covers exact primary and auxiliary requests."""
from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.runtime.backends.base import get_backend


def _modules():
    backend = get_backend("gemmini")
    package = backend.__name__
    return (importlib.import_module(f"{package}.gemmini_calibration_execution"),
            importlib.import_module(f"{package}.gemmini_roofline_auxiliary"),
            importlib.import_module(f"{package}.gemmini_dma_calibration"))


def _cli_module():
    path = (repo_root() / "merlin/experiments/gemmini_perf_bench/scripts/"
            "run_rtl_calibration_campaign.py")
    spec = importlib.util.spec_from_file_location("_run_rtl_calibration_campaign", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _request(execution, identity: dict, mechanism: str | None = None) -> dict:
    row = {"identity": identity, "request_sha256": execution._digest(identity)}
    if mechanism is not None:
        row["mechanism"] = mechanism
        row["required_raw_receipts"] = (
            ["rtl_cycle_measurement", "compiler_command_buffer"]
            if mechanism == "compute" else ["rtl_cycle_measurement", "physical_counter"])
    return row


def test_nondispatchable_campaign_never_reaches_target_runner(monkeypatch, tmp_path) -> None:
    execution, auxiliary, _dma = _modules()
    monkeypatch.setattr(auxiliary, "run_empty_workload", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("nondispatchable campaign reached runner")))

    result = execution.execute(
        {"dispatchable": False}, {}, manifest_sha256="a" * 64,
        rtl_facts_sha256="b" * 64, capabilities_sha256="c" * 64,
        circt_hw_sha256="d" * 64, workdir=tmp_path)

    assert result["status"] == "NO_GO" and result["results"] == []


def test_cli_writes_precise_no_go_before_execution_when_inputs_do_not_link(
        monkeypatch, tmp_path: Path) -> None:
    cli = _cli_module()
    generated = tmp_path / "out"
    generated.mkdir()
    circt = tmp_path / "core.mlir"
    circt.write_text("hw.module @M() {}\n", encoding="utf-8")
    circt_sha = cli.hashlib.sha256(circt.read_bytes()).hexdigest()
    rtl = generated / "rtl.json"
    capabilities = generated / "capabilities.json"
    campaign = generated / "campaign.json"
    rtl.write_text(json.dumps({"inputs": {"core_hw_sha256": circt_sha}}), encoding="utf-8")
    capabilities.write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    campaign.write_text(json.dumps({
        "dispatchable": True,
        "inputs": {"rtl_facts": {"sha256": "0" * 64},
                   "harness_capabilities": {"sha256": "1" * 64}},
    }), encoding="utf-8")
    output = generated / "execution.json"
    monkeypatch.setattr(cli, "out_dir", lambda: generated)
    monkeypatch.setattr(cli.mlc_bridge, "core_hw_mlir", lambda _target: circt)
    monkeypatch.setattr(cli, "_module", lambda: (_ for _ in ()).throw(
        AssertionError("mismatched inputs reached executor")))

    status = cli.main([
        "--rtl-facts", str(rtl), "--harness-capabilities", str(capabilities),
        "--campaign-manifest", str(campaign), "--workdir", str(generated / "work"),
        "--output-json", str(output),
    ])

    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert status == 1 and artifact["status"] == "NO_GO"
    assert artifact["partial_execution_is_admissible"] is False
    assert "not exactly linked" in artifact["issues"][0]


def test_movement_counter_passes_require_the_same_exact_emitted_program() -> None:
    execution, _auxiliary, _dma = _modules()
    layout = {"engines": ["x"], "by_combination": {"x": "busy"}}
    common = {
        "cycles": 9, "correct": True,
        "oracle": {"kind": "rtl", "derived_from_rtl": True},
        "measurement_conditions": {"cache_protocol": "derived", "window": "region"},
        "elf_sha256": "e" * 64,
    }
    occupancy = {**common, "emitter": {"emitted_mlir_sha256": "a" * 64},
                 "counters": {"selection": {"kind": "joint_occupancy", "unit": None},
                              "occupancy": layout, "readings": {"busy": 4}}}
    physical = {**common, "emitter": {"emitted_mlir_sha256": "a" * 64},
                "counters": {"selection": {"kind": "unit", "unit": "derived"},
                             "selected_counters": {"bytes": 1},
                             "readings": {"bytes": 8}}}
    kwargs = {
        "request_sha256": "1" * 64, "rtl_facts_sha256": "2" * 64,
        "capabilities_sha256": "3" * 64, "circt_hw_sha256": "4" * 64,
        "counter_binding": None,
    }

    linked = execution._linked(occupancy, physical, **kwargs)

    assert linked["measurement_identity"]["program"] == {
        "kind": "compiler_emitted_program", "sha256": "a" * 64}
    drifted = dict(physical, emitter={"emitted_mlir_sha256": "b" * 64})
    with pytest.raises(Exception, match="compiler-program identity"):
        execution._linked(occupancy, drifted, **kwargs)


def test_executor_schedules_every_primary_four_empty_runs_and_composition(
        monkeypatch, tmp_path) -> None:
    execution, auxiliary, dma = _modules()
    rtl_sha, capabilities_sha, circt_sha = "a" * 64, "b" * 64, "c" * 64
    protocols = ("tool-derived-protocol-0", "tool-derived-protocol-1")
    layout = {
        "prefix": "opaque", "engines": ["p", "q", "r"], "complete": True,
        "by_combination": {"p": "c0", "q": "c1", "r": "c2", "p+q": "c3",
                           "p+r": "c4", "q+r": "c5", "p+q+r": "c6"},
    }
    codes = {name: index for index, name in enumerate(layout["by_combination"].values())}
    zero = {name: 0 for name in codes}
    command = {
        "tensors": {"w": {"shape": [1, 1]}, "a": {"shape": [1, 1]},
                    "y": {"shape": [1, 1]}},
        "commands": [
            {"opcode": "RES_PACK", "operands": {"src": "w", "dst": "wr"}},
            {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "a", "rhs": "wr", "dst": "x"}},
            {"opcode": "COMMIT", "operands": {"src": "x", "dst": "y"}},
            {"opcode": "EVICT", "operands": {"handle": "wr"}},
        ],
    }

    def raw(*, protocol, readings=None, physical=False, with_command=False):
        counters = ({"selection": {"kind": "unit", "unit": "BYTES"},
                     "selected_counters": {"byte0": 9, "byte1": 10},
                     "readings": {"byte0": 7, "byte1": 0}}
                    if physical else
                    {"selection": {"kind": "joint_occupancy", "unit": None},
                     "occupancy": layout, "readings": readings or dict(zero),
                     "event_codes": codes,
                     "partition": {"module": "M", "counter_module": "C", "source": "circt"}})
        return {
            "cycles": 20, "correct": True,
            "oracle": {"kind": "rtl", "derived_from_rtl": True},
            "measurement_conditions": {"cache_protocol": protocol, "window": "region"},
            "counters": counters, "elf_sha256": "d" * 64,
            "emitter": {"emitted_mlir_sha256": "e" * 64},
            **({"command_buffer": command,
                "command_buffer_sha256": execution._digest(command)} if with_command else {}),
        }

    support = {
        "read": dict(zero, c0=4), "write": dict(zero, c1=4),
        "copy": dict(zero, c0=2, c1=2),
        "compute": dict(zero, c0=2, c1=2, c2=8),
    }

    def run_dma(direction, _payload, _facts, *, protocol, counter_unit, **_kwargs):
        return raw(protocol=protocol, physical=counter_unit is not None,
                   readings=support[direction])

    def run_compute(_facts, _multiple, protocol, *, counter_unit, **_kwargs):
        return raw(protocol=protocol, physical=counter_unit is not None,
                   readings=support["compute"],
                   with_command=True)

    monkeypatch.setattr(dma, "run_dma_calibration", run_dma)
    monkeypatch.setattr(auxiliary, "run_compute_probe", run_compute)
    monkeypatch.setattr(auxiliary, "run_empty_workload", lambda protocol, **_kwargs: {
        "cycles": 3, "correct": True, "oracle": {"kind": "rtl", "derived_from_rtl": True},
        "measurement_conditions": {"cache_protocol": protocol, "window": "region"},
        "elf_sha256": "f" * 64, "command_buffer": {"tensors": {}, "commands": []},
        "command_buffer_sha256": execution._digest({"tensors": {}, "commands": []}),
    })
    monkeypatch.setattr(auxiliary, "run_joint_occupancy_probe", lambda *_args, **kwargs: {
        "status": "measured", "composition_measurement": raw(
            protocol=kwargs["protocol"],
            readings=support["compute"], with_command=True),
        "resource_role_binding": {
            "status": "proved", "artifact_sha256": "1" * 64,
            "counter_layout": layout,
            "kinds": {"p": "movement", "q": "movement", "r": "compute"},
        },
    })

    common = {"rtl_facts_sha256": rtl_sha,
              "harness_capabilities_sha256": capabilities_sha}
    primary = [_request(execution, {
        **common, "sweep_id": "compute", "ordinal": ordinal,
        "coordinates": {"tile_multiple": multiple}, "measurement_protocol": protocols[0],
    }, "compute") for ordinal, multiple in enumerate((1, 2, 3, 4))]
    primary.extend(_request(execution, {
        **common, "sweep_id": f"{direction}-{protocol}", "ordinal": ordinal,
        "coordinates": {"transfer_bytes": payload, "measurement_protocol": protocol},
    }, f"dma_{direction}")
        for direction in ("read", "write", "copy") for protocol in protocols
        for ordinal, payload in enumerate((8, 16, 24, 32)))
    auxiliaries = [_request(execution, {
        **common, "kind": "empty_run", "measurement_protocol": protocol, "replicate": index,
    }) for protocol in protocols for index in range(4)]
    auxiliaries.append(_request(execution, {
        **common, "kind": "composition_probe", "measurement_protocol": protocols[0]}))
    binding = {"status": "proved", "rtl_facts_sha256": rtl_sha, "counter_facts": [{
        "fact_kind": "counter_byte_binding", "artifact_sha256": rtl_sha,
        "counter_field": "byte0", "direction": "read", "unit_bytes": 1,
        "derived_from_rtl": True, "provenance": "fixture RTL proof",
    }, {
        "fact_kind": "counter_byte_binding", "artifact_sha256": rtl_sha,
        "counter_field": "byte1", "direction": "write", "unit_bytes": 1,
        "derived_from_rtl": True, "provenance": "fixture RTL proof",
    }]}

    result = execution.execute(
        {"schema": "rtl_calibration_campaign_v1", "dispatchable": True,
         "execution_contract": {"partial_execution_is_admissible": False},
         "measurement_requests": primary,
         "auxiliary_measurement_requests": auxiliaries}, {}, manifest_sha256="9" * 64,
        rtl_facts_sha256=rtl_sha, capabilities_sha256=capabilities_sha,
        circt_hw_sha256=circt_sha, workdir=tmp_path, counter_binding=binding)

    assert result["status"] == "READY" and result["issues"] == []
    assert len(result["results"]) == 28 and len(result["empty_runs"]) == 8
    assert result["results"][4]["measurement"]["linked_counter_evidence"][
        "physical_byte_counters"]["semantic_resolution"] == "rtl_bound_physical_bytes"
    composition = result["composition_probe"]
    assert composition["occupancy_binding"]["counter_layout"]["kinds"] == {
        "p": "movement", "q": "movement", "r": "compute"}

    monkeypatch.setattr(auxiliary, "run_empty_workload", lambda *_args, **_kwargs: (
        _ for _ in ()).throw(RuntimeError("fixture RTL failure")))
    failed = execution.execute(
        {"schema": "rtl_calibration_campaign_v1", "dispatchable": True,
         "execution_contract": {"partial_execution_is_admissible": False},
         "measurement_requests": primary,
         "auxiliary_measurement_requests": auxiliaries}, {}, manifest_sha256="9" * 64,
        rtl_facts_sha256=rtl_sha, capabilities_sha256=capabilities_sha,
        circt_hw_sha256=circt_sha, workdir=tmp_path, counter_binding=binding)
    assert failed["status"] == "NO_GO"
    assert failed["partial_execution_is_admissible"] is False
    assert len(failed["empty_runs"]) == 0
    assert any("fixture RTL failure" in issue for issue in failed["issues"])
