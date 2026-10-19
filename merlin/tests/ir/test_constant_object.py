"""Immutable object reuse reopens complete data and rejects changed inputs."""

import os
import shutil
import subprocess

import pytest

from merlin.llvmlower.constant_object import ConstantBlob, reuse_constant_object


def assemble(directory, name, source):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host assembler unavailable")
    path = directory / (name + ".S")
    path.write_text(source)
    output = path.with_suffix(".o")
    subprocess.run([cc, "-c", str(path), "-o", str(output)], check=True, capture_output=True)
    return output


@pytest.fixture
def objects(tmp_path):
    data = tmp_path / "data.bin"
    data.write_bytes(bytes(range(137)))
    tail = tmp_path / "tail.bin"
    tail.write_bytes(b"private tail")
    body = (f'.section .rodata\n.balign 64\n.global a\na:\n.incbin "{data}"\n'
            f'.balign 64\n.global b\nb:\n.incbin "{tail}"\n.balign 64\n')
    source = assemble(tmp_path, "source", body)
    reference = assemble(tmp_path, "reference", ".section .rodata\n.balign 64\n"
                         ".global probe\nprobe:\n.byte 0\n.balign 64\n")
    blobs = (ConstantBlob("a", data, 64), ConstantBlob("b", tail, 64))
    return source, reference, blobs, body


def reuse(tmp_path, objects, *, source=None, blobs=None):
    original, reference, operands, _ = objects
    return reuse_constant_object(source or original, tmp_path / "reused.o",
                                 operands if blobs is None else blobs,
                                 reference_object=reference, section=".rodata")


def test_complete_reuse_is_same_inode_with_independent_receipt(tmp_path, objects):
    source, _, _, _ = objects
    before = source.read_bytes()
    record = reuse(tmp_path, objects)
    assert os.path.samefile(source, tmp_path / "reused.o")
    assert source.read_bytes() == before
    assert record["shared_inode"] and record["bytes"] == len(before)
    assert [(r["symbol"], r["offset"], r["bytes"]) for r in record["operands"]] == [
        ("a", 0, 137), ("b", 192, 12)]


@pytest.mark.parametrize("mutation", ["bytes", "symbol", "alignment", "order"])
def test_changed_fresh_operand_contract_refuses(tmp_path, objects, mutation):
    _, _, blobs, _ = objects
    if mutation == "bytes":
        blobs[0].path.write_bytes(b"changed" + blobs[0].path.read_bytes()[7:])
    elif mutation == "symbol":
        blobs = (ConstantBlob("different", blobs[0].path, 64), blobs[1])
    elif mutation == "alignment":
        blobs = (blobs[0], ConstantBlob("b", blobs[1].path, 128))
    else:
        blobs = tuple(reversed(blobs))
    with pytest.raises(ValueError):
        reuse(tmp_path, objects, blobs=blobs)
    assert not (tmp_path / "reused.o").exists()


@pytest.mark.parametrize("extra", [
    '.section .data\n.byte 1\n',
    '.text\n.byte 1\n',
    '.section .rodata\n.quad unknown\n',
    '.section .rodata\n.global extra\nextra:\n',
    '.section .rodata\n.byte 1\n',
    '.section .other,"a"\n.byte 1\n',
])
def test_code_other_storage_relocations_extra_symbols_or_extent_refuse(tmp_path, objects, extra):
    altered = assemble(tmp_path, "altered", objects[3] + extra)
    with pytest.raises(ValueError):
        reuse(tmp_path, objects, source=altered)
    assert not (tmp_path / "reused.o").exists()


def test_nonzero_padding_refuses(tmp_path, objects):
    body = objects[3].replace('.balign 64\n.global b', '.balign 64, 255\n.global b')
    altered = assemble(tmp_path, "padding", body)
    with pytest.raises(ValueError, match="padding"):
        reuse(tmp_path, objects, source=altered)


def test_target_header_difference_refuses(tmp_path, objects):
    altered = tmp_path / "foreign.o"
    raw = bytearray(objects[0].read_bytes())
    raw[18:20] = bytes((0xff, 0xff))
    altered.write_bytes(raw)
    with pytest.raises(ValueError, match="target reference"):
        reuse(tmp_path, objects, source=altered)


def test_symlink_or_existing_destination_refuses(tmp_path, objects):
    link = tmp_path / "link.o"
    link.symlink_to(objects[0])
    with pytest.raises(ValueError, match="regular"):
        reuse(tmp_path, objects, source=link)
    (tmp_path / "reused.o").write_bytes(b"keep")
    with pytest.raises(ValueError, match="fresh destination"):
        reuse(tmp_path, objects)
    assert (tmp_path / "reused.o").read_bytes() == b"keep"
