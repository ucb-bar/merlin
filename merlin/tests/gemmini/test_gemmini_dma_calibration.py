"""The target-owned DMA calibrator derives encodings and keeps payload distinct from traffic."""
from __future__ import annotations

import importlib
from types import SimpleNamespace

from merlin.runtime.backends.base import get_backend


def _module():
    backend = get_backend("gemmini")
    return importlib.import_module(f"{backend.__name__}.gemmini_dma_calibration")


def _facts(dtype: str = "i8") -> dict:
    return {"facts": {"datapaths": [{"name": "input", "dtype": dtype}]}}


def test_target_source_uses_separate_header_operations_without_copied_encodings() -> None:
    dma = _module()

    read = dma._source("read", elements=3, element_bytes=2)
    write = dma._source("write", elements=3, element_bytes=2)
    copy = dma._source("copy", elements=3, element_bytes=2)

    assert "gemmini_extended_mvin" in read and "gemmini_extended_mvout" not in read
    assert "gemmini_extended_mvout" in write and "gemmini_extended_mvin" not in write
    assert "gemmini_extended_mvin" in copy and "gemmini_extended_mvout" in copy
    assert "k_MVIN" not in read + write + copy
    assert ".insn" not in read + write + copy


def test_emitter_chunks_by_header_capability_and_never_calls_payload_physical_traffic(
        monkeypatch) -> None:
    dma = _module()
    monkeypatch.setattr(dma, "max_command_payload_bytes", lambda _facts: (
        4, {"path": "generated_params.h", "sha256": "a" * 64, "macro": "LIMIT"}))

    def recipe(direction, *, elements, element_bytes):
        calls = []
        if direction in {"read", "copy"}:
            calls.append(dma._Asm("derived-read", "r,r", ("src", elements)))
        if direction in {"write", "copy"}:
            calls.append(dma._Asm("derived-write", "r,r", ("dst", elements)))
        return tuple(calls), {"direction": direction, "elements": elements,
                              "element_bytes": element_bytes}

    monkeypatch.setattr(dma, "_chunk_recipe", recipe)

    text, arguments, receipt = dma.emit_dma_kernel_mlir("copy", 10, _facts())

    assert arguments == ["src", "dst"]
    assert receipt["chunks_payload_bytes"] == [4, 4, 2]
    assert receipt["requested_payload_bytes"] == 10
    assert receipt["physical_traffic_bytes"] == {
        "status": "unmeasured", "value": None,
        "why": "only a validated RTL counter binding can establish physical traffic",
    }
    assert text.count("derived-read") == 3
    assert text.count("derived-write") == 3
    # Non-zero chunks advance both pointers; re-reading one cache line is not substituted for a larger
    # payload extent.
    assert "constant(4 : i64)" in text
    assert "constant(8 : i64)" in text


def test_header_llvm_parser_rejects_nonliteral_hidden_computation() -> None:
    dma = _module()
    llvm = """define void @probe(ptr %src, ptr %dst) {
  %x = call i64 @hidden()
  call void asm sideeffect \"fence\", \"\"()
  ret void
}
"""

    try:
        dma._probe_asm(llvm)
    except Exception as exc:
        assert "self-contained asm recipe" in str(exc)
    else:
        raise AssertionError("a target-header recipe with hidden computation was accepted")


def test_nonintegral_payload_is_refused_before_header_compilation(monkeypatch) -> None:
    dma = _module()
    monkeypatch.setattr(dma, "_target_headers", lambda: (_ for _ in ()).throw(
        AssertionError("invalid payload reached the target header")))

    try:
        dma.emit_dma_kernel_mlir("read", 3, _facts("i16"))
    except Exception as exc:
        assert "exact multiple" in str(exc)
    else:
        raise AssertionError("nonintegral RTL elements were silently rounded")


def test_performance_runner_rejects_functional_oracle_before_building(monkeypatch) -> None:
    dma = _module()
    backend = get_backend("gemmini")
    gemmini = importlib.import_module(f"{backend.__name__}.gemmini")
    monkeypatch.setattr(dma, "build_dma_object", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("functional oracle reached compiler")))

    try:
        dma.run_dma_calibration("read", 4, _facts(), protocol="fresh_elf_process",
                                simulator=next(name for name, receipt in gemmini.ORACLE.items()
                                               if receipt.get("derived_from_rtl") is False))
    except Exception as exc:
        assert "RTL-derived simulator" in str(exc)
    else:
        raise AssertionError("functional simulator was accepted for performance calibration")


def test_byte_named_counter_readings_do_not_become_physical_bytes(monkeypatch, tmp_path) -> None:
    dma = _module()
    backend = get_backend("gemmini")
    gemmini = importlib.import_module(f"{backend.__name__}.gemmini")
    from merlin.perf import hw_counters

    obj = tmp_path / "kernel.o"
    obj.write_bytes(b"object")
    monkeypatch.setattr(dma, "build_dma_object", lambda *_args, **_kwargs: (
        obj, {"status": "accepted"}))
    monkeypatch.setattr(dma, "_harness", lambda *_args, **_kwargs: (
        "int main(void) { return 0; }", {"cache_protocol": "derived"}))
    monkeypatch.setattr(gemmini, "harness_build_recipe", lambda: SimpleNamespace(
        compiler="compiler", cflags=(), include_roots=(), link_script="script",
        support_sources=()))

    def link(command, **_kwargs):
        output = command[command.index("-o") + 1]
        dma.Path(output).write_bytes(b"elf")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(dma.subprocess, "run", link)
    monkeypatch.setattr(gemmini, "run_elf", lambda *_args, **_kwargs: "raw console")
    monkeypatch.setattr(gemmini, "parse_output", lambda _text: ({}, {"cycles": 9}))
    header = tmp_path / "counter.h"
    header.write_text("#define READ_BYTES 1\n#define WRITE_BYTES 2\n", encoding="utf-8")
    monkeypatch.setattr(hw_counters, "counters_for_target", lambda _target: {
        "status": "derived", "header_sha256": "schema", "header": str(header)})
    monkeypatch.setattr(hw_counters, "parse_counter_schema", lambda _text: "schema")
    monkeypatch.setattr(hw_counters, "parse_counter_output", lambda _text: {
        "READ_BYTES": 256, "WRITE_BYTES": 0})

    result = dma.run_dma_calibration(
        "read", 64, _facts(), protocol="derived", simulator="verilator", workdir=tmp_path)

    assert result["raw_counter_readings"]["readings"] == {
        "READ_BYTES": 256, "WRITE_BYTES": 0}
    assert result["physical_traffic_bytes"]["status"] == "unknown"
    assert result["physical_traffic_bytes"]["value"] is None
