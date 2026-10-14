"""Runtime intervals must be checked even when every ELF segment fits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from merlin.runtime import elf_audit
from merlin.runtime.execution_memory import (
    ExecutionMemoryError,
    MemoryMapBinding,
    MemoryReservation,
    admit_execution_memory,
)


def _map(tmp_path, *, regions=None, bits=64):
    evidence = tmp_path / "decoded.txt"
    evidence.write_text("synthetic independent decoded-region evidence\n")
    record = {
        "schema": "merlin.execution-memory-map.v1",
        "status": "complete",
        "execution_identity": "independent-test-engine",
        "identity_mapped": True,
        "address_bits": bits,
        "regions": regions or [{"name": "ram", "begin": 0x1000, "bytes": 0x9000, "permissions": "RWX"}],
        "file_pins": [{"path": str(evidence), "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest()}],
    }
    path = tmp_path / "map.json"
    path.write_text(json.dumps(record))
    return MemoryMapBinding(path, hashlib.sha256(path.read_bytes()).hexdigest(), "independent-test-engine")


def _image(tmp_path, monkeypatch):
    path = tmp_path / "image.elf"
    path.write_bytes(b"\x7fELF\x02\x01\x01" + b"\x00" * 9 + b"\x02\x00\x00\x00")
    segments = [elf_audit.Segment("LOAD", 0x1000, 0x100, 0x100, "RX", paddr=0x1000)]
    monkeypatch.setattr(elf_audit, "read_elf", lambda *_a, **_k: (0x1000, segments, {}))
    return path, segments


def test_hidden_allocator_outside_map_refuses_despite_fitting_segments(tmp_path, monkeypatch):
    image, _ = _image(tmp_path, monkeypatch)
    with pytest.raises(ExecutionMemoryError, match="allocator.*outside"):
        admit_execution_memory(image, _map(tmp_path), (MemoryReservation("allocator", 0xC000, 0x1000),))


def test_valid_stack_heap_external_buffers_are_all_in_receipt(tmp_path, monkeypatch):
    image, _ = _image(tmp_path, monkeypatch)
    uses = (
        MemoryReservation("stack", 0x2000, 0x1000),
        MemoryReservation("heap", 0x4000, 0x1000),
        MemoryReservation("external", 0x8000, 0x1000, permissions="R", alignment=64),
    )
    receipt = admit_execution_memory(image, _map(tmp_path), uses)
    assert receipt["status"] == "pass"
    assert [use["name"] for use in receipt["runtime_reservations"]] == ["stack", "heap", "external"]
    assert receipt["elf"]["sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "use",
    [
        MemoryReservation("overlap-load", 0x1080, 0x100),
        MemoryReservation("overflow", (1 << 64) - 8, 16),
        MemoryReservation("misaligned", 0x2001, 0x100, alignment=64),
    ],
)
def test_unsafe_runtime_extents_refuse(tmp_path, monkeypatch, use):
    image, _ = _image(tmp_path, monkeypatch)
    with pytest.raises(ExecutionMemoryError):
        admit_execution_memory(image, _map(tmp_path), (use,))


def test_runtime_aliases_need_separate_ownership_proof(tmp_path, monkeypatch):
    image, _ = _image(tmp_path, monkeypatch)
    with pytest.raises(ExecutionMemoryError, match="overlap"):
        admit_execution_memory(
            image,
            _map(tmp_path),
            (MemoryReservation("heap", 0x3000, 0x1000), MemoryReservation("external", 0x3800, 0x1000)),
        )


def test_permission_and_fragment_gaps_refuse(tmp_path, monkeypatch):
    image, _ = _image(tmp_path, monkeypatch)
    binding = _map(
        tmp_path,
        regions=[
            {"name": "text", "begin": 0x1000, "bytes": 0x1000, "permissions": "RX"},
            {"name": "data-left", "begin": 0x3000, "bytes": 0x1000, "permissions": "RW"},
            {"name": "data-right", "begin": 0x5000, "bytes": 0x1000, "permissions": "RW"},
        ],
    )
    for use in (MemoryReservation("readonly-heap", 0x1800, 0x100), MemoryReservation("gap", 0x3800, 0x2000)):
        with pytest.raises(ExecutionMemoryError, match="outside"):
            admit_execution_memory(image, binding, (use,))


def test_adjacent_regions_cover_extent_without_losing_permissions(tmp_path, monkeypatch):
    image, _ = _image(tmp_path, monkeypatch)
    binding = _map(
        tmp_path,
        regions=[
            {"name": "text", "begin": 0x1000, "bytes": 0x1000, "permissions": "RX"},
            {"name": "left", "begin": 0x3000, "bytes": 0x1000, "permissions": "RW"},
            {"name": "right", "begin": 0x4000, "bytes": 0x1000, "permissions": "RW"},
        ],
    )
    assert admit_execution_memory(image, binding, (MemoryReservation("heap", 0x3800, 0x1000),))["status"] == "pass"


def test_map_and_evidence_drift_refuse(tmp_path, monkeypatch):
    image, _ = _image(tmp_path, monkeypatch)
    binding = _map(tmp_path)
    (tmp_path / "decoded.txt").write_text("changed")
    with pytest.raises(ExecutionMemoryError, match="evidence"):
        admit_execution_memory(image, binding, ())
    binding = _map(tmp_path)
    binding.path.write_text("{}")
    with pytest.raises(ExecutionMemoryError, match="map.*changed"):
        admit_execution_memory(image, binding, ())


def test_unknown_map_or_nonidentity_physical_load_refuses(tmp_path, monkeypatch):
    image, segments = _image(tmp_path, monkeypatch)
    with pytest.raises(ExecutionMemoryError, match="map"):
        admit_execution_memory(image, None, ())
    segments[0].paddr = 0x4000
    with pytest.raises(ExecutionMemoryError, match="identity"):
        admit_execution_memory(image, _map(tmp_path), ())


def test_selected_engine_disagreement_refuses(tmp_path, monkeypatch):
    image, _ = _image(tmp_path, monkeypatch)
    binding = _map(tmp_path)
    mismatched = MemoryMapBinding(binding.path, binding.sha256, "another-selected-engine")
    with pytest.raises(ExecutionMemoryError, match="execution identity"):
        admit_execution_memory(image, mismatched, ())


def test_malformed_load_rows_do_not_disappear_under_strict_audit(monkeypatch):
    monkeypatch.setattr(elf_audit, "_tool", lambda *_: "readelf")
    monkeypatch.setattr(
        elf_audit,
        "_run",
        lambda *_: "Entry point address: 0x1000\nLOAD broken\nLOAD 0x0 0x1000 0x1000 0x10 0x10 R E 0x1000\n",
    )
    with pytest.raises(elf_audit.ElfAuditError, match="LOAD"):
        elf_audit.read_elf(Path("ignored"), strict=True)


def test_strict_load_preserves_physical_address_and_normalizes_execution_flag(monkeypatch):
    monkeypatch.setattr(elf_audit, "_tool", lambda *_: "readelf")
    monkeypatch.setattr(
        elf_audit, "_run", lambda *_: "Entry point address: 0x1000\nLOAD 0x0 0x1000 0x1000 0x10 0x10 R E 0x1000\n"
    )
    _, segments, _ = elf_audit.read_elf(Path("ignored"), strict=True)
    assert segments[0].paddr == 0x1000 and segments[0].flags == "RX"
    assert elf_audit.read_elf(Path("ignored"))[1][0].flags == "RE"


@pytest.mark.parametrize("selection,uses", [(None, (MemoryReservation("external", 0x2000, 0x100),)), ("unknown", ())])
def test_normal_model_build_refuses_unknown_admission_before_lowering(tmp_path, monkeypatch, selection, uses):
    from merlin.runtime.backends import spike_model

    monkeypatch.setattr(spike_model, "lower_model_file", lambda *_a, **_k: pytest.fail("lowering must not run"))
    work = tmp_path / "build"
    work.mkdir()
    admission = work / "execution_memory_admission.json"
    admission.write_text('{"status":"pass","stale":true}')
    with pytest.raises(ExecutionMemoryError):
        spike_model.build(
            tmp_path / "capture", work, execution_memory_map=selection, execution_memory_reservations=uses
        )
    assert not admission.exists()
    assert not (work / "compilation_recipe.json").exists()
    assert not list(work.glob("*.elf"))


@pytest.mark.parametrize("packed", [True, False])
def test_actual_normal_build_closes_heap_and_stack_before_completion(tmp_path, packed):
    import numpy as np

    from merlin.llvmlower import toolchain
    from merlin.runtime.backends import spike, spike_model

    if not spike.available() or not toolchain.m2m_python().is_file() or not toolchain.clang().is_file():
        pytest.skip("bare-metal and upstream toolchain unavailable")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model.mlir").write_text("""module {
      func.func @forward(%arg0: tensor<5xf32>) -> tensor<5xf32> {
        %init = tensor.empty() : tensor<5xf32>
        %out = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"]
        } ins(%arg0 : tensor<5xf32>) outs(%init : tensor<5xf32>) {
          ^bb0(%value: f32, %unused: f32):
            %two = arith.constant 2.0 : f32
            %sum = arith.addf %value, %two : f32
            linalg.yield %sum : f32
        } -> tensor<5xf32>
        return %out : tensor<5xf32>
      }
    }""")
    (bundle / "weights.safetensors.manifest.json").write_text('{"0":{"kind":"input","name":"values"}}')
    values = np.array([-5, -0.75, 0, 1.25, 7], dtype=np.float32)
    np.savez(bundle / "inputs.npz", in0=values)
    binding = _map(
        tmp_path, regions=[{"name": "declared-span", "begin": 0x80000000, "bytes": 4 << 20, "permissions": "RWX"}]
    )
    work = tmp_path / "selected"
    options = {
        "arena_mb": 1,
        "backend": "scalar",
        "host_vectorize": False,
        "cflags_override": ["-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-builtin"],
    }
    if packed:
        options.update(dram_bytes=4 << 20, code_reserve=1 << 20)
        baseline = spike_model.build(bundle, tmp_path / "unselected", **options)
        built = spike_model.build(bundle, work, execution_memory_map=binding, **options)
        reserved = built["weights_base"] - 0x80000000
        assert reserved >= options["code_reserve"]
        assert reserved >= (
            built["static_io_bytes"] + (work / "model.o").stat().st_size
            + options["code_reserve"] // 4
        )
        assert built["execution_memory_admission"]["status"] == "pass"
        uses = built["execution_memory_admission"]["runtime_reservations"]
        assert {use["name"] for use in uses} == {"runtime-allocator", "runtime-stack"}
        assert (work / "model.o").read_bytes() == (tmp_path / "unselected/model.o").read_bytes()
        assert (work / "model_main.o").read_bytes() == (tmp_path / "unselected/model_main.o").read_bytes()
        assert built["elf"].read_bytes() == baseline["elf"].read_bytes()
        ran = spike_model.run(built["elf"], mem_bytes=built["mem_bytes"], isa="rv64gc", timeout=60)
        np.testing.assert_array_equal(ran["outputs"], values + np.float32(2))
        assert json.loads((work / "compilation_recipe.json").read_text())["status"] == "completed"
    else:
        with pytest.raises(ExecutionMemoryError, match="runtime-allocator.*outside"):
            spike_model.build(bundle, work, execution_memory_map=binding, **options)
        assert not (work / "execution_memory_admission.json").exists()
        assert json.loads((work / "compilation_recipe.json").read_text())["status"] != "completed"
