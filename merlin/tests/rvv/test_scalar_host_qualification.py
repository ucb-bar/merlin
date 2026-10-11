"""A scalar host receipt passes only a complete, exact saved-capture simulator result."""

from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext

import numpy as np
import pytest
import yaml

from merlin.compile.scalar_host_qualification import ScalarHostQualificationError, qualify
from merlin.runtime.backends import base as backends
from merlin.runtime.backends import spike_model
from merlin.targetgen import gsim_emulator, rtl_engine_policy

ROCKET_ISA = "rv64imafdcbzicsr_zifencei_zihpm_zfh_zba_zbb_zbs_xrocket"


def _inputs(tmp_path, monkeypatch):
    generated = tmp_path / "out"
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(generated))
    # Package minting has its own tests. This fixture must also work from a
    # core-only wheel, with no checkout or optional experiment distribution.
    package = generated / "synthetic-host-package"
    package.mkdir(parents=True)
    (package / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "target": "host",
                "run_id": "scalar_fixture",
                "family": "scalar_linalg",
                "status": "unverified",
                "authoring": {"mode": "test_fixture", "generated_by_agent": False},
                "outputs": {"knobs": "knobs.yaml"},
            }
        )
    )
    (package / "knobs.yaml").write_text(
        yaml.safe_dump(
            {
                "backend": "scalar",
                "dtype_strategy": "int8_w8a8",
                "cflags": ["-march=rv64gc_zba_zbb_zbs_zfh", "-mabi=lp64d", "-O2"],
            }
        )
    )
    capture = tmp_path / "capture"
    capture.mkdir()
    for name in (
        "model.mlir",
        "weights.safetensors",
        "weights.safetensors.manifest.json",
        "inputs.npz",
        "input_order.json",
    ):
        (capture / name).write_bytes(name.encode())
    (capture / "model.mlir").write_text(
        "module { func.func @forward(%arg0: tensor<3xf32>) -> tensor<3xf32> { return %arg0 : tensor<3xf32> } }"
    )
    np.save(capture / "golden.npy", np.array([0.5, -1.0, 2.0], dtype=np.float32))
    names = (
        "model.mlir",
        "weights.safetensors",
        "weights.safetensors.manifest.json",
        "inputs.npz",
        "input_order.json",
        "golden.npy",
    )
    (capture / "capture_receipt.json").write_text(
        json.dumps(
            {
                "schema": "m2m.capture-receipt.v1",
                "materialized_abi": {"complete": True, "inputs": 1},
                "artifacts": {
                    name: {"sha256": hashlib.sha256((capture / name).read_bytes()).hexdigest()} for name in names
                },
            }
        ),
        encoding="utf-8",
    )
    dts = tmp_path / "rocket.dts"
    dts.write_text(f'device_type = "cpu";\nriscv,isa = "{ROCKET_ISA}";\n', encoding="utf-8")
    catalog = tmp_path / "boards.yaml"
    catalog.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "boards": {
                    "rocket": {
                        "target": "example",
                        "rtl_sim_config": "ExampleConfig",
                        "host_dts_sha256": hashlib.sha256(dts.read_bytes()).hexdigest(),
                        "dram_bytes": 256 << 20,
                        "dram_base": 0x80000000,
                        "harts": 1,
                        "vector_harts": 0,
                        "console": "htif",
                        "flow": "baremetal",
                        "code_reserve": 64 << 20,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return capture, package, catalog, dts, generated


def _stub_spike(monkeypatch, *, outputs, console):
    seen = {}

    def build(_capture, work, **kwargs):
        seen["build"] = kwargs
        work.mkdir(parents=True)
        elf = work / "model.elf"
        elf.write_bytes(b"synthetic scalar ELF")
        return {"elf": elf, "mem_bytes": 256 << 20, "build_hash": "test"}

    def run(_elf, **kwargs):
        seen["run"] = kwargs
        return {
            "outputs": np.asarray(outputs)
            if isinstance(outputs, np.ndarray)
            else np.asarray(outputs, dtype=np.float32),
            "console": console,
            "metrics": {"memref_rank_mismatch": 0, "cycles": 10},
        }

    monkeypatch.setattr(spike_model, "build", build)
    monkeypatch.setattr(spike_model, "run", run)
    monkeypatch.setattr(spike_model, "arch_extensions", lambda _elf: ["rv64i", "m", "a", "f", "d", "c"])
    return seen


def _console(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    return "OUT " + str(len(bits)) + " " + " ".join(str(int(value)) for value in bits) + "\nDONE\n"


@pytest.mark.parametrize("mismatch", [False, True])
def test_integer_qualification_compares_raw_bits_not_rounded_floats(tmp_path, monkeypatch, mismatch):
    capture, package, catalog, dts, generated = _inputs(tmp_path, monkeypatch)
    golden = np.array([2**53 + 1, -(2**63), 2**63 - 1], dtype=np.int64)
    np.save(capture / "golden.npy", golden)
    (capture / "model.mlir").write_text(
        "module { func.func @forward(%arg0: tensor<3xi64>) -> tensor<3xi64> { return %arg0 : tensor<3xi64> } }"
    )
    path = capture / "capture_receipt.json"
    document = json.loads(path.read_text())
    document["artifacts"]["golden.npy"]["sha256"] = hashlib.sha256((capture / "golden.npy").read_bytes()).hexdigest()
    document["artifacts"]["model.mlir"]["sha256"] = hashlib.sha256((capture / "model.mlir").read_bytes()).hexdigest()
    path.write_text(json.dumps(document))
    observed = golden.copy()
    if mismatch:
        observed[0] -= 1  # Both values round to the same float64.
    words = [word for value in observed.view(np.uint64) for word in (int(value) & 0xFFFFFFFF, int(value) >> 32)]
    console = "OUT_I64 3 " + " ".join(map(str, words)) + "\nDONE\n"
    _stub_spike(monkeypatch, outputs=observed, console=console)
    output = generated / "qualifications" / "integer"
    if mismatch:
        with pytest.raises(ScalarHostQualificationError, match="exact golden mismatch"):
            qualify(
                capture=capture,
                package=package,
                board_catalog=catalog,
                board="rocket",
                dts=dts,
                output=output,
                arena_mb=32,
            )
    else:
        receipt = qualify(
            capture=capture, package=package, board_catalog=catalog, board="rocket", dts=dts, output=output, arena_mb=32
        )
        assert receipt["output"]["element_dtype"] == "i64"
        assert receipt["output"]["mismatched_elements"] == 0


def test_golden_precision_cannot_override_the_parsed_model_output(tmp_path, monkeypatch):
    capture, package, catalog, dts, generated = _inputs(tmp_path, monkeypatch)
    np.save(capture / "golden.npy", np.array([1, 2, 3], dtype=np.int64))
    path = capture / "capture_receipt.json"
    document = json.loads(path.read_text())
    document["artifacts"]["golden.npy"]["sha256"] = hashlib.sha256((capture / "golden.npy").read_bytes()).hexdigest()
    path.write_text(json.dumps(document))
    monkeypatch.setattr(spike_model, "build", lambda *_a, **_kw: pytest.fail("build before output ABI check"))
    with pytest.raises(ScalarHostQualificationError, match="parsed model output ABI"):
        qualify(
            capture=capture,
            package=package,
            board_catalog=catalog,
            board="rocket",
            dts=dts,
            output=generated / "qualifications" / "wrong-dtype",
            arena_mb=32,
        )


@pytest.mark.parametrize("mismatch", [False, True])
def test_boolean_qualification_checks_all_canonical_output_bytes(tmp_path, monkeypatch, mismatch):
    capture, package, catalog, dts, generated = _inputs(tmp_path, monkeypatch)
    golden = np.array([False, True, False], dtype=np.bool_)
    np.save(capture / "golden.npy", golden)
    (capture / "model.mlir").write_text(
        "module { func.func @forward(%arg0: tensor<3xi1>) -> tensor<3xi1> { return %arg0 : tensor<3xi1> } }"
    )
    path = capture / "capture_receipt.json"
    document = json.loads(path.read_text())
    for name in ("model.mlir", "golden.npy"):
        document["artifacts"][name]["sha256"] = hashlib.sha256((capture / name).read_bytes()).hexdigest()
    path.write_text(json.dumps(document))
    observed = golden.copy()
    if mismatch:
        observed[-1] = True
    console = "OUT_I1 3 " + " ".join(map(str, observed.view(np.uint8))) + "\nDONE\n"
    _stub_spike(monkeypatch, outputs=observed, console=console)
    output = generated / "qualifications" / "boolean"
    if mismatch:
        with pytest.raises(ScalarHostQualificationError, match="exact golden mismatch"):
            qualify(capture=capture, package=package, board_catalog=catalog, board="rocket", dts=dts,
                    output=output, arena_mb=32)
    else:
        receipt = qualify(capture=capture, package=package, board_catalog=catalog, board="rocket", dts=dts,
                          output=output, arena_mb=32)
        assert receipt["output"]["element_dtype"] == "i1"
        assert receipt["output"]["mismatched_elements"] == 0


def _stub_native_gsim(tmp_path, monkeypatch, *, console):
    firrtl = tmp_path / "selected.fir"
    firrtl.write_bytes(b"selected elaborated FIRRTL")
    digest = hashlib.sha256(firrtl.read_bytes()).hexdigest()
    facts = tmp_path / "rtl-facts.json"
    facts.write_text(
        json.dumps(
            {
                "inputs": {
                    "target": "example",
                    "fir_sha256": digest,
                    "firrtl_inputs": [{"path": str(firrtl), "sha256": digest}],
                },
                "facts": {"source": {"config": "ExampleConfig"}},
            }
        ),
        encoding="utf-8",
    )
    emulator = tmp_path / "emulator"
    emulator.write_bytes(b"selected engine")
    citation = {
        "target": "example",
        "engine": "gsim",
        "path": str(emulator),
        "available": True,
        "refused": False,
        "flavour": "binary",
        "receipt_status": "bound",
        "binary_sha256": hashlib.sha256(emulator.read_bytes()).hexdigest(),
        "receipt": {"schema_version": gsim_emulator.STRICT_RECEIPT_SCHEMA, "firrtl_sha256": digest},
    }
    seen = {}

    class Prepared:
        def revalidate(self):
            seen["revalidated"] = seen.get("revalidated", 0) + 1

    class Backend:
        GSIM_EMU_ENV = "TEST_GSIM_EMULATOR"

        def available(self, simulator):
            return simulator == "gsim"

        def gsim_path(self):
            return emulator

        def prepare_gsim_command(self, elf, **kwargs):
            seen["prepared"] = (elf, kwargs)
            return Prepared()

        def run_elf(self, elf, **kwargs):
            seen["run"] = (elf, kwargs)
            return console

    monkeypatch.setattr(backends, "get_backend", lambda _target: Backend())
    monkeypatch.setattr(gsim_emulator, "citation", lambda _target, **_kw: citation)
    monkeypatch.setattr(rtl_engine_policy, "gsim_runtime_slot", lambda **_kw: nullcontext())
    return facts, citation, seen


def test_exact_single_output_receipt_binds_every_input(tmp_path, monkeypatch):
    capture, package, catalog, dts, generated = _inputs(tmp_path, monkeypatch)
    seen = _stub_spike(monkeypatch, outputs=[0.5, -1.0, 2.0], console=_console([0.5, -1.0, 2.0]))
    output = generated / "qualifications" / "one"
    receipt = qualify(
        capture=capture, package=package, board_catalog=catalog, board="rocket", dts=dts, output=output, arena_mb=32
    )
    assert receipt["status"] == "passed_saved_capture_spike"
    assert receipt["inputs"]["host_isa"] == ROCKET_ISA
    assert receipt["inputs"]["simulator_isa"] == "rv64gc_zba_zbb_zbs_zfh"
    assert receipt["inputs"]["package_tree"]["sha256"]
    assert receipt["inputs"]["capture_tree"]["sha256"]
    assert receipt["inputs"]["dts_sha256"] == hashlib.sha256(dts.read_bytes()).hexdigest()
    assert receipt["output"]["elf_sha256"] == hashlib.sha256(b"synthetic scalar ELF").hexdigest()
    assert receipt["output"]["mismatched_elements"] == 0
    assert seen["run"]["isa"] == receipt["inputs"]["simulator_isa"]
    assert seen["build"]["backend"] == "scalar"
    assert json.loads((output / "qualification.json").read_text())["status"] == receipt["status"]


@pytest.mark.parametrize(
    "outputs,multiout,reason",
    [
        ([0.5, -1.0], False, "partial or nonfinite output"),
        ([0.5, -1.0, 3.0], False, "exact golden mismatch"),
        ([0.5, -1.0, 2.0], True, "exactly one OUT"),
    ],
)
def test_partial_mismatch_and_multiout_never_pass(tmp_path, monkeypatch, outputs, multiout, reason):
    capture, package, catalog, dts, generated = _inputs(tmp_path, monkeypatch)
    console = _console(outputs)
    if multiout:
        console += _console([1.0])
    _stub_spike(monkeypatch, outputs=outputs, console=console)
    output = generated / "qualifications" / "failed"
    with pytest.raises(ScalarHostQualificationError, match=reason):
        qualify(
            capture=capture, package=package, board_catalog=catalog, board="rocket", dts=dts, output=output, arena_mb=32
        )
    receipt = json.loads((output / "qualification.json").read_text())
    assert receipt["status"] == "failed"


def test_unpinned_dts_refuses_before_build(tmp_path, monkeypatch):
    capture, package, catalog, dts, generated = _inputs(tmp_path, monkeypatch)
    dts.write_text('riscv,isa = "rv64gcv";\n', encoding="utf-8")
    monkeypatch.setattr(spike_model, "build", lambda *_a, **_kw: pytest.fail("build before DTS pin"))
    with pytest.raises(ValueError, match="pinned SHA256"):
        qualify(
            capture=capture,
            package=package,
            board_catalog=catalog,
            board="rocket",
            dts=dts,
            output=generated / "qualifications" / "bad-dts",
            arena_mb=32,
        )


def test_output_inside_selected_package_refuses_before_writing(tmp_path, monkeypatch):
    capture, package, catalog, dts, _generated = _inputs(tmp_path, monkeypatch)
    output = package / "qualification"
    monkeypatch.setattr(spike_model, "build", lambda *_a, **_kw: pytest.fail("build before output preflight"))
    with pytest.raises(ScalarHostQualificationError, match="nested inside an input"):
        qualify(
            capture=capture, package=package, board_catalog=catalog, board="rocket", dts=dts, output=output, arena_mb=32
        )
    assert not output.exists()


def test_native_gsim_uses_shared_model_parser_and_records_exact_engine(tmp_path, monkeypatch):
    capture, package, catalog, dts, generated = _inputs(tmp_path, monkeypatch)
    _stub_spike(monkeypatch, outputs=[0.5, -1.0, 2.0], console="")
    console = _console([0.5, -1.0, 2.0]).replace("DONE", "METRIC memref_rank_mismatch 0\nDONE")
    facts, citation, seen = _stub_native_gsim(tmp_path, monkeypatch, console=console)
    output = generated / "qualifications" / "native"
    receipt = qualify(
        capture=capture,
        package=package,
        board_catalog=catalog,
        board="rocket",
        dts=dts,
        output=output,
        arena_mb=32,
        simulator="gsim",
        rtl_facts=facts,
    )
    assert receipt["schema"] == "merlin.scalar_host_native_qualification.v1"
    assert receipt["status"] == "passed_saved_capture_native_rtl"
    assert receipt["engine"]["citation"] == citation
    assert receipt["output"]["mismatched_elements"] == 0
    assert receipt["output"]["metrics"]["memref_rank_mismatch"] == 0
    assert receipt["output"]["rtl_console_sha256"] == hashlib.sha256(console.encode()).hexdigest()
    assert seen["run"][1]["simulator"] == "gsim"
    assert seen["revalidated"] == 2


def test_bound_native_engine_revalidates_execution_inputs_without_legacy_command(tmp_path, monkeypatch):
    from pathlib import Path

    from merlin.compile.model_execution_inputs import native_engine
    from merlin.runtime.backends.chipyard_rocc_bound import BoundRoCCBackend

    _, citation, seen = _stub_native_gsim(tmp_path, monkeypatch, console="")

    class BoundBackend(BoundRoCCBackend):
        def available(self, simulator):
            return simulator == "gsim"

        def gsim_path(self):
            return Path(citation["path"])

        def verify_execution_inputs(self):
            if seen.get("drift"):
                raise ValueError("selected bound execution inputs changed")
            seen["bound_checks"] = seen.get("bound_checks", 0) + 1

    backend = BoundBackend("example", "{}", None)
    monkeypatch.setattr(backends, "get_backend", lambda _target: backend)
    selected, identity, revalidate, prepare = native_engine(
        "example", "gsim", {"firrtl_sha256": citation["receipt"]["firrtl_sha256"]}
    )
    assert selected is backend and identity == citation and prepare is None
    revalidate()
    assert seen["bound_checks"] == 2
    seen["drift"] = True
    with pytest.raises(ValueError, match="bound execution inputs changed"):
        revalidate()


@pytest.mark.parametrize(
    "console,reason",
    [
        ("OUT 3 1056964608 3212836864 1073741824\nMETRIC memref_rank_mismatch 0\n", "DONE"),
        ("OUT 2 1056964608 3212836864\nMETRIC memref_rank_mismatch 0\nDONE\n", "partial"),
    ],
)
def test_native_incomplete_result_never_passes(tmp_path, monkeypatch, console, reason):
    capture, package, catalog, dts, generated = _inputs(tmp_path, monkeypatch)
    _stub_spike(monkeypatch, outputs=[0.5, -1.0, 2.0], console="")
    facts, _citation, seen = _stub_native_gsim(tmp_path, monkeypatch, console=console)
    output = generated / "qualifications" / "native-incomplete"
    with pytest.raises((ScalarHostQualificationError, spike_model.SpikeModelError), match=reason):
        qualify(
            capture=capture,
            package=package,
            board_catalog=catalog,
            board="rocket",
            dts=dts,
            output=output,
            arena_mb=32,
            simulator="gsim",
            rtl_facts=facts,
        )
    assert "run" in seen
    assert json.loads((output / "qualification.json").read_text())["status"] == "failed"


def test_native_facts_target_mismatch_refuses_before_build(tmp_path, monkeypatch):
    capture, package, catalog, dts, generated = _inputs(tmp_path, monkeypatch)
    facts, _citation, _seen = _stub_native_gsim(tmp_path, monkeypatch, console="")
    document = json.loads(facts.read_text())
    document["inputs"]["target"] = "different-target"
    facts.write_text(json.dumps(document))
    monkeypatch.setattr(spike_model, "build", lambda *_a, **_kw: pytest.fail("build before RTL facts pin"))
    with pytest.raises(ScalarHostQualificationError, match="board target/config"):
        qualify(
            capture=capture,
            package=package,
            board_catalog=catalog,
            board="rocket",
            dts=dts,
            output=generated / "qualifications" / "wrong-target",
            arena_mb=32,
            simulator="gsim",
            rtl_facts=facts,
        )


@pytest.mark.parametrize(
    "arch",
    [
        ["rv64i", "m", "a", "f", "d", "c", "v"],
        ["rv64i", "m", "a", "f", "d", "c", "zicond"],
    ],
)
def test_elf_with_vector_or_undeclared_extension_refuses(tmp_path, monkeypatch, arch):
    capture, package, catalog, dts, generated = _inputs(tmp_path, monkeypatch)
    _stub_spike(monkeypatch, outputs=[0.5, -1.0, 2.0], console=_console([0.5, -1.0, 2.0]))
    monkeypatch.setattr(spike_model, "arch_extensions", lambda _elf: arch)
    output = generated / "qualifications" / "bad-elf"
    with pytest.raises(ScalarHostQualificationError, match="ELF ISA is not scalar-compatible"):
        qualify(
            capture=capture, package=package, board_catalog=catalog, board="rocket", dts=dts, output=output, arena_mb=32
        )
    assert json.loads((output / "qualification.json").read_text())["status"] == "failed"
