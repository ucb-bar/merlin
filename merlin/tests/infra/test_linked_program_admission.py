"""Actual linked section checks; neutral encodings do not qualify a hardware target."""

import inspect
import json
import shutil
import struct
import subprocess
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest
from test_selected_program_contracts import inputs
from test_selected_program_contracts import native as native

from merlin.model_check.program_admission import service
from merlin.targetgen.contract.build_service import file_digest
from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService
from merlin.targetgen.elf_lanes import ElfUnreadable, executable_sections, read_program_section


def linked_bytes(payload, *, bits=64, endian="<", flags=2, kind=1, mapped=True, duplicate=False):
    """A complete neutral final ELF with a file-backed declared section."""
    names = b"\0.shstrtab\0.program\0"
    data = bytearray(1024 + len(payload))
    data[:7] = b"\x7fELF" + bytes([2 if bits == 64 else 1, 1 if endian == "<" else 2, 1])
    if bits == 64:
        header, program, section = endian + "HHIQQQIHHHHHH", endian + "IIQQQQQQ", endian + "IIQQQQIIQQ"
        struct.pack_into(header, data, 16, 2, 0, 1, 0, 64, 256, 0, 64, 56, 1, 64, 4 if duplicate else 3, 1)
        struct.pack_into(program, data, 64, 1, 4, 1024, 4096 if mapped else 4097, 4096, len(payload), len(payload), 1)
    else:
        header, program, section = endian + "HHIIIIIHHHHHH", endian + "IIIIIIII", endian + "IIIIIIIIII"
        struct.pack_into(header, data, 16, 2, 0, 1, 0, 64, 256, 0, 52, 32, 1, 40, 4 if duplicate else 3, 1)
        struct.pack_into(program, data, 64, 1, 1024, 4096 if mapped else 4097, 4096, len(payload), len(payload), 4, 1)
    stride = struct.calcsize(section)
    struct.pack_into(section, data, 256 + stride, 1, 3, 0, 0, 768, len(names), 0, 0, 1, 0)
    row = (11, kind, flags, 4096, 1024, len(payload), 0, 0, 1, 0)
    struct.pack_into(section, data, 256 + 2 * stride, *row)
    if duplicate:
        struct.pack_into(section, data, 256 + 3 * stride, *row)
    data[768 : 768 + len(names)] = names
    data[1024:] = payload
    return bytes(data)


@pytest.mark.parametrize("bits,endian", [(32, "<"), (32, ">"), (64, "<"), (64, ">")])
def test_section_reader_preserves_exact_final_bytes_and_executable_reader(tmp_path, bits, endian):
    elf = tmp_path / "model.elf"
    elf.write_bytes(linked_bytes(b"\0\xffprogram", bits=bits, endian=endian, flags=6))
    image, location = read_program_section(elf, ".program", max_bytes=9)
    assert image == b"\0\xffprogram"
    assert location == dict(section=".program", file_offset=1024, virtual_address=4096, bytes=9, writable=False)
    assert executable_sections(elf.read_bytes()) == [(".program", 1024, 9, 4096)]


@pytest.mark.parametrize(
    "change,match",
    [
        ({"flags": 0}, "allocated"),
        ({"flags": 0x802}, "uncompressed"),
        ({"kind": 8}, "PROGBITS"),
        ({"mapped": False}, "load segment"),
        ({"duplicate": True}, "ambiguous"),
    ],
)
def test_section_reader_refuses_unrealized_or_ambiguous_programs(tmp_path, change, match):
    elf = tmp_path / "model.elf"
    elf.write_bytes(linked_bytes(b"code", **change))
    with pytest.raises(ElfUnreadable, match=match):
        read_program_section(elf, ".program", max_bytes=4)


@pytest.mark.parametrize("mutation", ["budget", "missing", "truncated", "object", "symlink", "overflow"])
def test_section_reader_refuses_unsupported_images(tmp_path, mutation):
    elf = tmp_path / "model.elf"
    data = bytearray(linked_bytes(b"code"))
    if mutation == "truncated":
        data = data[:-1]
    elif mutation == "object":
        struct.pack_into("<H", data, 16, 1)
    elif mutation == "overflow":
        struct.pack_into("<Q", data, 64 + 16, (1 << 64) - 1)
    elf.write_bytes(data)
    if mutation == "symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(elf)
        elf = alias
    with pytest.raises((ValueError, ElfUnreadable)):
        read_program_section(
            elf, ".missing" if mutation == "missing" else ".program", max_bytes=3 if mutation == "budget" else 4
        )


def admit(*, elf, evidence_root, status="accepted"):
    evidence_root.mkdir(parents=True, exist_ok=True)
    path = evidence_root / "original.json"
    report = dict(status=status, elf_sha256=file_digest(elf), protected_findings="not copied into shared reports")
    path.write_text(json.dumps(report))
    return {**report, "report_path": str(path), "report_sha256": file_digest(path)}


def checked(model, *, status="accepted", **kwargs):
    selected, _, _, _, codec, _, _, _ = model
    owner = Path(__file__).resolve()
    prerequisite = LinkedElfAdmissionService(
        selected.target, partial(admit, status=status), ((str(owner), file_digest(owner)),)
    )
    decoder_owner = Path(inspect.getsourcefile(codec.program_decoder.decode)).resolve()
    return service(
        prerequisite=prerequisite,
        selected_contracts=selected,
        decoder=codec.program_decoder,
        decoder_sources=((str(decoder_owner), file_digest(decoder_owner)),),
        section=kwargs.pop("section", ".program"),
        reader_max_bytes=1 << 20,
        **kwargs,
    )


def actual_elf(tmp_path, payload):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("native compiler unavailable")
    source = tmp_path / "image.c"
    # Volatile prevents constant folding away the loaded program.
    source.write_text(
        '__attribute__((section(".program"),used,aligned(1)))\n'
        "const volatile unsigned char program[]={" + ",".join(map(str, payload)) + "};\n"
        "int main(void) {return program[0];}\n"
    )
    elf = tmp_path / "model.elf"
    subprocess.run([cc, str(source), "-o", str(elf)], check=True, capture_output=True)
    return elf


@pytest.mark.parametrize("word_format,separate", [("<BBBQQQ", False), (">BBBIII", True)])
def test_actual_linked_bytes_checked_and_independently_executed_with_state(
    tmp_path, native, word_format, separate, record_property
):
    model = inputs(tmp_path, width=13, word_format=word_format, separate=separate, count=6)
    selected, _, _, _, _, _, word, length = model
    payload = b"".join(
        word.pack(*row)
        for row in (
            (0, 2, 3, 0, 0, length),
            (1, 3, 3, 0, 0, length),
            (2, 3, 3, 0, length, length),
            (3, 0, 0, 0, 0, length),
            (1, 3, 3, length, length, length),
            (0, 3, 3, length, 0, length),
        )
    )
    elf = actual_elf(tmp_path, payload)
    policy = checked(model)
    result = policy.evaluate(elf=elf, target=selected.target, evidence_root=tmp_path / "admission")
    assert result["status"] == "accepted", result
    assert result["program"]["instruction_count"] == 6
    assert result["program"]["encoded_bytes"] == len(payload)
    assert "protected_findings" not in json.dumps(result)
    image, _ = read_program_section(elf, ".program", max_bytes=len(payload))
    assert image == payload
    state = bytearray([253] * (2 * length + 8))
    expected = state[:]
    data = bytearray((i * 17) % 256 for i in range(len(state)))
    counts = []
    for position in range(3):
        observed, _ = native(
            (image,),
            data,
            state,
            width=8 if word_format.endswith("Q") else 4,
            big=word_format.startswith(">"),
            separate=separate,
        )
        values = bytes((x + 2) % 256 for x in data[:length])
        expected[: 2 * length] = values + values
        assert state == expected
        assert policy.revalidate(elf=elf, result=result, target=selected.target) == "accepted"
        counts.append(observed)
        data[:length] = state[:length]
    record_property(
        "linked_program",
        json.dumps(
            dict(bytes=len(payload), instructions=6, calls=3, commands_and_requested_bytes=counts, timing="UNKNOWN")
        ),
    )


@pytest.mark.parametrize("mutation", ["size", "address", "incomplete", "unknown", "parent", "facts", "software"])
def test_linked_program_refuses_actual_invalid_bytes_or_stale_inputs(tmp_path, mutation):
    model = inputs(tmp_path, count=2)
    selected, _, _, _, codec, _, word, length = model
    payload = word.pack(0, 2, 3, 0, 0, length)
    if mutation == "size":
        payload *= 3
    elif mutation == "address":
        payload = word.pack(0, 2, 3, 999, 0, length)
    elif mutation == "incomplete":
        payload = word.pack(2, 2, 3, 0, 0, length)
    elif mutation == "unknown":
        payload = bytes([255]) + payload[1:]
    elf = actual_elf(tmp_path, payload)
    policy = checked(model, status="refused" if mutation == "parent" else "accepted")
    if mutation in ("facts", "software"):
        path = selected.facts_path if mutation == "facts" else selected.software_path
        path.write_bytes(path.read_bytes() + b"\n")
        with pytest.raises(ValueError, match="source pin changed|specification changed"):
            policy.evaluate(elf=elf, target=selected.target, evidence_root=tmp_path / "admission")
    else:
        result = policy.evaluate(elf=elf, target=selected.target, evidence_root=tmp_path / "admission")
        assert result["status"] == "refused", result
        assert result["reason"]
        assert "program" not in result


def test_reader_policy_can_only_narrow_selected_limits(tmp_path):
    model = inputs(tmp_path, count=2)
    payload = model[6].pack(0, 2, 3, 0, 0, model[7]) * 2
    elf = actual_elf(tmp_path, payload)
    selected = model[0]
    # Resolve the same original selected limits through the existing binder.
    bounds, _, _, _ = selected.program.bind(selected, None, model[4].program_decoder, None, None)
    policy = checked(model, limits=replace(bounds, max_instructions=1))
    result = policy.evaluate(elf=elf, target=selected.target, evidence_root=tmp_path / "narrow")
    assert result["status"] == "refused"
    assert "segment" in result["reason"] or "budget" in result["reason"]


def test_section_reader_does_not_copy_a_large_sparse_model_image(tmp_path):
    import tracemalloc

    elf = tmp_path / "sparse.elf"
    elf.write_bytes(linked_bytes(b"bounded"))
    with elf.open("r+b") as stream:
        stream.truncate(1 << 30)
    tracemalloc.start()
    try:
        image, _ = read_program_section(elf, ".program", max_bytes=7)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert image == b"bounded"
    assert peak < 2 << 20


@pytest.mark.parametrize("mutation", ["facts", "software", "decoder"])
def test_linked_admission_rechecks_selection_at_consumption(tmp_path, mutation):
    model = inputs(tmp_path)
    elf = actual_elf(tmp_path, model[6].pack(0, 2, 3, 0, 0, model[7]))
    policy = checked(model)
    result = policy.evaluate(elf=elf, target=model[0].target, evidence_root=tmp_path / "admission")
    assert result["status"] == "accepted"
    if mutation in ("facts", "software"):
        path = model[0].facts_path if mutation == "facts" else model[0].software_path
        path.write_bytes(path.read_bytes() + b"\n")
        with pytest.raises(ValueError, match="source pin changed"):
            policy.revalidate(elf=elf, result=result, target=model[0].target)
    else:
        object.__setattr__(model[4].program_decoder, "decode", lambda _: ())
        with pytest.raises(ValueError, match="decoder changed"):
            policy.evaluate(elf=elf, target=model[0].target, evidence_root=tmp_path / "changed")
