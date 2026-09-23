"""Compile/run evidence for the synthetic whole-session production tracer."""

from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path

import pytest

from merlin.compare.paper_session_abi import (
    InputEndpoint,
    InputFrame,
    decode_response,
    encode_request,
    encode_response,
)
from merlin.compare.paper_session_tracer import (
    ENTRYPOINT,
    build_relocatable_object,
    build_tracer_package,
    inspect_relocatable_object,
    render_runner_source,
    run_tracer_package,
)


def _compiler() -> str:
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("host C compiler is unavailable")
    return compiler


def _request(descriptor, values=(2, 1, 4, 5)) -> bytes:
    frames = [
        InputFrame(InputEndpoint(program, input_index), step, struct.pack(">Q", value))
        for (program, input_index, step), value in zip(descriptor.required_input_keys, values, strict=True)
    ]
    return encode_request(descriptor, frames)


def _records(packet: bytes, *, calls: int = 0) -> tuple[int, list[bytes]]:
    descriptor_size = int.from_bytes(packet[9:13], "big")
    offset = 13 + descriptor_size
    count = int.from_bytes(packet[offset : offset + 4], "big")
    offset += 4 + calls * 8
    if calls:
        count = int.from_bytes(packet[offset : offset + 4], "big")
        offset += 4
    rows: list[bytes] = []
    for _ in range(count):
        size = int.from_bytes(packet[offset + 12 : offset + 20], "big")
        end = offset + 20 + size
        rows.append(packet[offset:end])
        offset = end
    return offset - sum(map(len, rows)), rows


def test_runner_source_can_bind_exact_response_capacity():
    generic = render_runner_source()
    bound = render_runner_source(response_capacity=123456)

    assert "malloc(used + 4096)" in generic
    assert "response_capacity = 123456ULL" in bound
    assert "malloc(response_capacity)" in bound
    assert "used + 4096" not in bound
    with pytest.raises(ValueError, match="positive"):
        render_runner_source(response_capacity=0)


def test_compiled_package_executes_prefill_and_three_carried_state_steps(tmp_path):
    package = build_tracer_package(tmp_path / "package", compiler=_compiler())
    response_bytes = run_tracer_package(package, _request(package.descriptor))
    response = decode_response(response_bytes, expected_descriptor=package.descriptor)

    # state = state * 3 + delta, seeded only by the prefill input.
    assert [struct.unpack(">Q", frame.payload)[0] for frame in response.outputs] == [7, 25, 80]
    assert [(call.program, call.step) for call in response.executed_calls] == [(0, 0), (1, 0), (1, 1), (1, 2)]

    receipt = json.loads(package.receipt.read_text(encoding="utf-8"))
    assert receipt["entrypoint"] == ENTRYPOINT
    assert receipt["descriptor_sha256"] == package.descriptor.sha256
    assert receipt["object_sha256"]
    assert "golden" not in package.source.read_text(encoding="utf-8").lower()

    readelf, nm = shutil.which("readelf"), shutil.which("nm")
    if readelf and nm:
        evidence = inspect_relocatable_object(
            package.object, readelf=readelf, nm=nm, expected_machine="Advanced Micro Devices X86-64"
        )
        assert evidence.elf_type.startswith("REL")
        assert ENTRYPOINT in evidence.symbols


def test_compiled_runner_rejects_missing_or_reordered_input_frames(tmp_path):
    package = build_tracer_package(tmp_path / "package", compiler=_compiler())
    request = _request(package.descriptor)
    start, frames = _records(request)
    assert len(frames) == 4

    reordered = request[:start] + frames[1] + frames[0] + b"".join(frames[2:])
    with pytest.raises(RuntimeError, match="rejected session packet"):
        run_tracer_package(package, reordered)

    omitted = request[:start] + b"".join(frames[:-1])
    with pytest.raises(RuntimeError, match="rejected session packet"):
        run_tracer_package(package, omitted)


def test_output_and_stage_contract_rejects_omissions_and_reordering(tmp_path):
    package = build_tracer_package(tmp_path / "package", compiler=_compiler())
    response = run_tracer_package(package, _request(package.descriptor))
    decoded = decode_response(response, expected_descriptor=package.descriptor)

    with pytest.raises(ValueError, match="stage or recurrent step was omitted"):
        encode_response(package.descriptor, package.descriptor.calls[1:], decoded.outputs)
    with pytest.raises(ValueError, match="missing="):
        encode_response(package.descriptor, package.descriptor.calls, decoded.outputs[:-1])

    descriptor_size = int.from_bytes(response[9:13], "big")
    calls_at = 13 + descriptor_size
    call_count = int.from_bytes(response[calls_at : calls_at + 4], "big")
    trace_start = calls_at + 4
    trace = [response[trace_start + i * 8 : trace_start + (i + 1) * 8] for i in range(call_count)]
    output_count_at = trace_start + call_count * 8
    trace_reordered = response[:trace_start] + trace[1] + trace[0] + b"".join(trace[2:]) + response[output_count_at:]
    with pytest.raises(ValueError, match="exact whole-session schedule"):
        decode_response(trace_reordered, expected_descriptor=package.descriptor)

    outputs_start, outputs = _records(response, calls=call_count)
    assert len(outputs) == 3
    outputs_reordered = response[:outputs_start] + outputs[1] + outputs[0] + outputs[2]
    with pytest.raises(ValueError, match="missing, extra, or non-canonical"):
        decode_response(outputs_reordered, expected_descriptor=package.descriptor)


def test_riscv_relocatable_object_exports_the_common_entrypoint_when_clang_supports_it(tmp_path: Path):
    clang, readelf, nm = shutil.which("clang"), shutil.which("readelf"), shutil.which("nm")
    if not clang or not readelf or not nm:
        pytest.skip("cross-object inspection tools are unavailable")
    package = build_tracer_package(tmp_path / "host", compiler=_compiler())
    output = tmp_path / "model_session_riscv.o"
    try:
        build_relocatable_object(
            package.source, output, compiler=clang, flags=("--target=riscv64-unknown-linux-gnu", "-ffreestanding")
        )
    except RuntimeError as exc:
        pytest.skip(f"installed clang has no RISC-V object support: {exc}")
    evidence = inspect_relocatable_object(output, readelf=readelf, nm=nm, expected_machine="RISC-V")
    assert evidence.elf_class == "ELF64"
    assert evidence.elf_type.startswith("REL")
    assert evidence.machine == "RISC-V"
    assert ENTRYPOINT in evidence.symbols
