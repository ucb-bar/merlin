"""Coarse, simulator-free coverage for the saved whole-model bare-metal seam."""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from merlin.compile import baremetal_model as BM
from merlin.compile import model_execution_inputs as MI
from merlin.llvmlower.device_build import DeviceRouting


def _fixture(tmp_path: Path, monkeypatch, *, elements: int = 2):
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(out))
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "model.mlir").write_text("module {}\n")
    (capture / "weights.safetensors").write_bytes(b"weights")
    (capture / "weights.safetensors.manifest.json").write_text("{}")
    (capture / "inputs.npz").write_bytes(b"inputs")
    (capture / "input_order.json").write_text("[]")
    np.save(capture / "golden.npy", np.arange(elements, dtype=np.float32))
    artifacts = {
        p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in capture.iterdir()
    }
    (capture / "capture_receipt.json").write_text(
        json.dumps(
            {
                "schema": "m2m.capture-receipt.v1",
                "materialized_abi": {"complete": True},
                "artifacts": artifacts,
            }
        )
    )
    package = tmp_path / "package"
    package.mkdir()
    (package / "manifest.yaml").write_text("host: scalar\n")
    dts = tmp_path / "selected.dts"
    dts.write_text('cpu@0 { device_type = "cpu"; riscv,isa = "rv64imafdc_zicsr_zifencei"; };\n')
    firrtl = tmp_path / "selected.fir"
    firrtl.write_text("circuit SampleConfig :\n")
    fir_sha = hashlib.sha256(firrtl.read_bytes()).hexdigest()
    facts = tmp_path / "selected-facts.json"
    facts.write_text(
        json.dumps(
            {
                "inputs": {
                    "target": "sample",
                    "fir_sha256": fir_sha,
                    "firrtl_inputs": [{"path": str(firrtl), "sha256": fir_sha}],
                },
                "facts": {"source": {"config": "SampleConfig"}},
            }
        )
    )
    catalog = tmp_path / "boards.yaml"
    catalog.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "boards": {
                    "selected": {
                        "target": "sample",
                        "dram_bytes": 256 << 20,
                        "dram_base": BM.spike_model.DRAM_BASE,
                        "harts": 1,
                        "console": "htif",
                        "flow": "baremetal",
                        "code_reserve": 64 << 20,
                        "host_dts_sha256": hashlib.sha256(dts.read_bytes()).hexdigest(),
                        "rtl_sim_config": "SampleConfig",
                    }
                },
            }
        )
    )
    pkg = SimpleNamespace(
        backend="scalar", cflags=["-march=rv64imafdc_zicsr_zifencei"], is_int8=False, schedule_text=""
    )
    monkeypatch.setattr(BM.registry, "load_rvv_package", lambda path: pkg)
    monkeypatch.setattr(
        BM.spike_model, "arch_extensions", lambda path: ["rv64i", "m", "a", "f", "d", "c", "zicsr", "zifencei"]
    )
    return {
        "capture": capture,
        "package": package,
        "board_catalog": catalog,
        "board": "selected",
        "dts": dts,
        "target": "sample",
        "arena_mb": 1,
        "rtl_facts": facts,
    }, out


def _fake_build(bundle, work, **kwargs):
    work.mkdir(parents=True)
    elf = work / "model.elf"
    elf.write_bytes(b"one linked ELF")
    return {
        "elf": elf,
        "mem_bytes": 256 << 20,
        "build_hash": "abc123",
        "vlen": None,
        # Synthetic build seam only: these bytes test receipt propagation, not a compiler observation.
        "index_lowering": {
            "schema": "merlin.selected-index-lowering.v1",
            "compiler_requested": "mock-clang",
            "compiler_resolved": "/mock/clang",
            "compiler_sha256": "0" * 64,
            "cross_flags": ["--target=riscv64-unknown-elf", *kwargs["cflags_override"]],
            "data_layout": "e-p:64:64",
            "index_bits": 64,
            "effective_pipeline": "synthetic-index-width-pipeline",
            "scope": "synthetic test fixture; no compiler executed",
        },
    }


def _console() -> str:
    bits = [struct.unpack("<I", struct.pack("<f", float(x)))[0] for x in (0, 1)]
    return f"OUT 2 {bits[0]} {bits[1]}\nMETRIC build_hash abc123\nMETRIC memref_rank_mismatch 0\nDONE\n"


def test_math_archive_symbol_request_is_opt_in_and_forwarded_unchanged(tmp_path, monkeypatch):
    inputs, out = _fixture(tmp_path, monkeypatch)
    observed = []

    def build(bundle, work, **kwargs):
        observed.append(dict(kwargs))
        return _fake_build(bundle, work, **kwargs)

    monkeypatch.setattr(BM.spike_model, "build", build)
    default = BM.compile_saved_model(**inputs, output=out / "default", run="none")
    selected = BM.compile_saved_model(**inputs, output=out / "selected", run="none", math_archive_symbols=["sinf"])
    assert default["status"] == selected["status"] == "compiled"
    assert "math_archive_symbols" not in observed[0]
    assert observed[1] == {**observed[0], "math_archive_symbols": ["sinf"]}
    assert default["inputs"] == selected["inputs"]


def test_host_math_policy_is_validated_recorded_and_forwarded_only_when_selected(tmp_path, monkeypatch):
    from merlin.runtime import host_math

    inputs, out = _fixture(tmp_path, monkeypatch)
    observed = []
    original_recipe = host_math.host_math_recipe

    def recipe(policy):
        if policy == "portable_exp_trig":
            return host_math.HostMathRecipe(source="explicit portable runtime")
        return original_recipe(policy)

    def build(bundle, work, **kwargs):
        observed.append(dict(kwargs))
        return _fake_build(bundle, work, **kwargs)

    monkeypatch.setattr(host_math, "host_math_recipe", recipe)
    monkeypatch.setattr(BM.spike_model, "build", build)
    default = BM.compile_saved_model(**inputs, output=out / "default_math", run="none")
    selected = BM.compile_saved_model(
        **inputs, output=out / "selected_math", run="none", host_math_policy="portable_exp_trig",
    )
    assert default["status"] == selected["status"] == "compiled"
    assert "host_math_policy" not in observed[0]
    assert observed[1] == {**observed[0], "host_math_policy": "portable_exp_trig"}
    assert default["inputs"]["host_math_policy"] == "native"
    assert selected["inputs"]["host_math_policy"] == "portable_exp_trig"


@pytest.mark.parametrize("policy", [None, True, "unselected", "portable_trig "])
def test_unknown_host_math_policy_refuses_before_any_input_or_output_io(tmp_path, monkeypatch, policy):
    def forbidden(*args, **kwargs):
        pytest.fail("input/output IO preceded host math policy validation")

    monkeypatch.setattr(Path, "resolve", forbidden)
    monkeypatch.setattr(Path, "is_symlink", forbidden)
    monkeypatch.setattr(BM.spike_model, "build", forbidden)
    output = tmp_path / "absent-output"
    with pytest.raises(BM.BaremetalModelError, match="host_math_policy"):
        BM.compile_saved_model(
            capture=tmp_path / "absent-capture", package=tmp_path / "absent-package",
            board_catalog=tmp_path / "absent-boards", board="board", dts=tmp_path / "absent-dts",
            output=output, target="sample", run="none", arena_mb=1, host_math_policy=policy,
        )
    assert not output.exists()


def test_compile_only_and_native_engine_reuse_one_saved_elf(tmp_path, monkeypatch):
    inputs, out = _fixture(tmp_path, monkeypatch)
    selected_device = DeviceRouting("sample", tmp_path / "device", "i8", "i32")
    selected_device.package_dir.mkdir()
    (selected_device.package_dir / "manifest.yaml").write_text("device: sample\n")
    built_with = []

    def build(bundle, work, **kwargs):
        built_with.append(kwargs["device"])
        result = _fake_build(bundle, work, **kwargs)
        if kwargs["device"] is not None:
            (work / "device_signatures.json").write_text('{"signatures": {}}')
        return result

    monkeypatch.setattr(BM.spike_model, "build", build)
    compiled = BM.compile_saved_model(**inputs, output=out / "compile", run="none", device=selected_device)
    assert compiled["status"] == "compiled"
    assert built_with == [selected_device]
    assert "device_sidecar" in compiled["output"]
    assert "console" not in compiled["output"]
    assert compiled["output"]["index_lowering"]["scope"] == "synthetic test fixture; no compiler executed"

    backend = SimpleNamespace(available=lambda engine: engine == "gsim", run_elf=lambda elf, **kw: _console())
    selection = {"available": True, "engine": "gsim", "fidelity": "elaborated_rtl", "reason": "receipted"}
    command = SimpleNamespace(
        revalidate=lambda: {"command_sha256": "mock"},
        to_evidence=lambda: {"schema": "mock-command", "emulator_argv": ["mock-gsim"], "max_cycles": 1},
    )
    monkeypatch.setattr(
        BM,
        "_native_engine",
        lambda target, run, facts: (
            backend,
            {"selection": selection, "citation": {"binary_sha256": "mock"}},
            lambda: None,
            lambda *a, **kw: command,
        ),
    )
    from contextlib import nullcontext

    from merlin.targetgen import rtl_engine_policy

    monkeypatch.setattr(rtl_engine_policy, "gsim_runtime_slot", lambda **kw: nullcontext())
    verified = BM.compile_saved_model(**inputs, output=out / "native", run="gsim", reference_file="golden.npy")
    assert verified["status"] == "verified_complete_output"
    assert verified["output"]["elements"] == 2
    assert verified["engine_selection"]["selection"] == selection
    assert Path(verified["output"]["console"]).read_text() == _console()
    assert built_with == [selected_device, None]


def test_missing_index_lowering_record_cannot_become_a_compiled_receipt(tmp_path, monkeypatch):
    inputs, out = _fixture(tmp_path, monkeypatch)

    def old_build(bundle, work, **kwargs):
        result = _fake_build(bundle, work, **kwargs)
        del result["index_lowering"]
        return result

    monkeypatch.setattr(BM.spike_model, "build", old_build)
    with pytest.raises(BM.BaremetalModelError, match="index-lowering record"):
        BM.compile_saved_model(**inputs, output=out / "unbound", run="none")
    receipt = json.loads((out / "unbound" / "baremetal_model.json").read_text())
    assert receipt["status"] == "failed"
    assert "index_lowering" not in receipt.get("output", {})


def test_oversize_output_refused_before_build_with_failed_receipt(tmp_path, monkeypatch):
    inputs, out = _fixture(tmp_path, monkeypatch, elements=4097)
    monkeypatch.setattr(BM.spike_model, "build", lambda *a, **kw: pytest.fail("build ran before OUT bound"))
    with pytest.raises(BM.BaremetalModelError, match="4096"):
        BM.compile_saved_model(**inputs, output=out / "too_large", run="spike", reference_file="golden.npy")
    receipt = json.loads((out / "too_large" / "baremetal_model.json").read_text())
    assert receipt["status"] == "failed"
    assert "4096" in receipt["failure"]


def test_native_wrong_board_firrtl_refused_before_build(tmp_path, monkeypatch):
    inputs, out = _fixture(tmp_path, monkeypatch)
    facts = json.loads(inputs["rtl_facts"].read_text())
    facts["facts"]["source"]["config"] = "another-board"
    inputs["rtl_facts"].write_text(json.dumps(facts))
    monkeypatch.setattr(BM.spike_model, "build", lambda *a, **kw: pytest.fail("build ran against foreign FIRRTL"))
    with pytest.raises(MI.ModelExecutionInputError, match="board target/config"):
        BM.compile_saved_model(**inputs, output=out / "foreign", run="gsim", reference_file="golden.npy")
    receipt = json.loads((out / "foreign" / "baremetal_model.json").read_text())
    assert receipt["status"] == "failed"


@pytest.mark.parametrize(
    "failure,expected_stdout,expected_stderr",
    [
        (subprocess.TimeoutExpired("mock-gsim", 1, output=b"OUT 2 0", stderr="timed out"), b"OUT 2 0", b"timed out"),
        (subprocess.CalledProcessError(1, "mock-gsim", output="OUT 2 0", stderr=b"failed"), b"OUT 2 0", b"failed"),
    ],
)
def test_native_failure_keeps_partial_stream_bytes_without_numerical_claim(
    tmp_path,
    monkeypatch,
    failure,
    expected_stdout,
    expected_stderr,
):
    inputs, out = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(BM.spike_model, "build", _fake_build)
    backend = SimpleNamespace(run_elf=lambda elf, **kw: (_ for _ in ()).throw(failure))
    command = SimpleNamespace(
        revalidate=lambda: {"command_sha256": "mock"},
        to_evidence=lambda: {"schema": "mock-command", "emulator_argv": ["mock-gsim"], "max_cycles": 1},
    )
    monkeypatch.setattr(
        BM,
        "_native_engine",
        lambda target, run, facts: (
            backend,
            {"selection": {"engine": "gsim"}, "citation": {"binary_sha256": "mock"}},
            lambda: None,
            lambda *a, **kw: command,
        ),
    )
    from contextlib import nullcontext

    from merlin.targetgen import rtl_engine_policy

    monkeypatch.setattr(rtl_engine_policy, "gsim_runtime_slot", lambda **kw: nullcontext())
    with pytest.raises(type(failure)):
        BM.compile_saved_model(**inputs, output=out / "failed_native", run="gsim", reference_file="golden.npy")
    receipt = json.loads((out / "failed_native" / "baremetal_model.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["failure_artifacts"]["scope"] == "diagnostic_partial_simulator_output_only"
    for stream, expected in (("stdout", expected_stdout), ("stderr", expected_stderr)):
        artifact = receipt["failure_artifacts"][stream]
        assert Path(artifact["path"]).read_bytes() == expected
        assert artifact["sha256"] == hashlib.sha256(expected).hexdigest()
        assert artifact["bytes"] == len(expected)
    assert "console" not in receipt["output"]
    assert "elements" not in receipt["output"]
    assert "metrics" not in receipt["output"]


def test_captured_quantization_scope_is_explicit_and_bound(tmp_path, monkeypatch):
    from merlin.llvmlower.quant_scope import FEATURE

    inputs, out = _fixture(tmp_path, monkeypatch)
    calls = []

    def build(bundle, work, **kwargs):
        calls.append(kwargs)
        return _fake_build(bundle, work, **kwargs)

    monkeypatch.setattr(BM.spike_model, "build", build)
    default = BM.compile_saved_model(**inputs, output=out / "default-scope", run="none")
    selected = BM.compile_saved_model(
        **inputs, output=out / "captured-scope", run="none", respect_captured_quantization_scope=True
    )
    assert "features" not in calls[0]
    assert calls[1]["features"] == frozenset({FEATURE})
    assert "respect_captured_quantization_scope" not in default["inputs"]
    assert selected["inputs"]["respect_captured_quantization_scope"] is True
