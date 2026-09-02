"""Target auxiliary calibration paths stay evidence-derived and fail closed."""
from __future__ import annotations

import importlib
from pathlib import Path

from merlin.runtime.backends.base import get_backend


def _module():
    backend = get_backend("gemmini")
    return importlib.import_module(f"{backend.__name__}.gemmini_roofline_auxiliary")


def _probe(layout: dict, readings: dict[str, int], *, cycles: int = 20) -> dict:
    return {
        "cycles": cycles, "elf_sha256": "e" * 64,
        "command_buffer_sha256": "c" * 64, "emitter": {"status": "accepted"},
        "counters": {"occupancy": layout, "readings": readings},
    }


def test_differential_activity_derives_roles_without_counter_name_semantics() -> None:
    module = _module()
    layout = {
        "prefix": "opaque", "engines": ["alpha", "beta", "gamma"], "complete": True,
        "by_combination": {
            "alpha": "counter-0", "beta": "counter-1", "gamma": "counter-2",
            "alpha+beta": "counter-3", "alpha+gamma": "counter-4",
            "beta+gamma": "counter-5", "alpha+beta+gamma": "counter-6",
        },
    }
    zero = {name: 0 for name in layout["by_combination"].values()}
    probes = {
        "read": _probe(layout, dict(zero, **{"counter-0": 4})),
        "write": _probe(layout, dict(zero, **{"counter-1": 5})),
        "copy": _probe(layout, dict(zero, **{"counter-0": 2, "counter-1": 3})),
        "compute": _probe(layout, dict(
            zero, **{"counter-0": 2, "counter-1": 2, "counter-2": 9})),
    }

    result = module.derive_resource_kinds(
        probes, rtl_facts_sha256="a" * 64, circt_hw_sha256="b" * 64)

    assert result["status"] == "proved"
    assert result["kinds"] == {
        "alpha": "movement", "beta": "movement", "gamma": "compute"}
    assert result["method"] == "direction_pure_dma_plus_native_compute_differential_v1"


def test_differential_activity_refuses_ambiguous_engine_roles() -> None:
    module = _module()
    layout = {"prefix": "opaque", "engines": ["x", "y"], "complete": True,
              "by_combination": {"x": "c0", "y": "c1", "x+y": "c2"}}
    zero = {"c0": 0, "c1": 0, "c2": 0}
    probes = {
        "read": _probe(layout, dict(zero, c0=4)),
        "write": _probe(layout, dict(zero, c0=5)),
        "copy": _probe(layout, dict(zero, c0=6)),
        "compute": _probe(layout, {"c0": 2, "c1": 9, "c2": 0}),
    }

    result = module.derive_resource_kinds(
        probes, rtl_facts_sha256="a" * 64, circt_hw_sha256="b" * 64)

    assert result["status"] == "unknown"
    assert "uniquely identify" in result["why"]


def test_empty_capability_uses_actual_native_compiler_product(
        monkeypatch, tmp_path: Path) -> None:
    module = _module()
    backend = get_backend("gemmini")
    codegen = importlib.import_module(f"{backend.__name__}.gemmini_codegen_mlir")
    circt = tmp_path / "core.hw.mlir"
    circt.write_text("hw.module @opaque() {}\n", encoding="utf-8")
    monkeypatch.setattr(module, "_exact_inputs", lambda *_args: (
        "a" * 64, circt, "b" * 64))
    observed: list[dict] = []

    def emit(command):
        observed.append(command)
        return "module { llvm.func @empty() { llvm.return } }", []

    monkeypatch.setattr(codegen, "emit_kernel_mlir", emit)

    result = module.probe_empty_workload(
        {"inputs": {}}, rtl_facts_sha256="a" * 64, stage="emission")

    assert result["status"] == "accepted"
    assert observed == [{"tensors": {}, "commands": []}]
    assert result["command_buffer"] == observed[0]
    assert result["rtl_facts_sha256"] == "a" * 64


def test_empty_capability_never_promotes_compiler_failure(monkeypatch, tmp_path: Path) -> None:
    module = _module()
    backend = get_backend("gemmini")
    codegen = importlib.import_module(f"{backend.__name__}.gemmini_codegen_mlir")
    circt = tmp_path / "core.hw.mlir"
    circt.write_text("hw.module @opaque() {}\n", encoding="utf-8")
    monkeypatch.setattr(module, "_exact_inputs", lambda *_args: (
        "a" * 64, circt, "b" * 64))
    monkeypatch.setattr(codegen, "emit_kernel_mlir", lambda _command: (_ for _ in ()).throw(
        RuntimeError("fixture rejection")))

    result = module.probe_empty_workload(
        {"inputs": {}}, rtl_facts_sha256="a" * 64, stage="emission")

    assert result["status"] == "unknown"
    assert "fixture rejection" in result["why"]
