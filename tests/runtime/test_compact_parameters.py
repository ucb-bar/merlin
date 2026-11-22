"""Compact host backing retains port identities and authenticated byte semantics."""
import json
import shutil
import struct
import subprocess
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from merlin.compiler.lower.c.parameters import ApplicationParameters
from merlin.compiler.lower.c.compact_parameters import compact_parameters
from merlin.compiler.lower.c.weights import WeightEntry
from merlin.compiler.semantic.model import KernelRequest, SemanticNode, TensorType


def fixture(tmp_path):
    payload = bytes(range(16))
    header = json.dumps({
        "shared": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        "resident": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]},
        "host": {"dtype": "F32", "shape": [1], "data_offsets": [12, 16]},
    }).encode()
    archive = struct.pack("<Q", len(header)) + header + payload
    path = tmp_path / "weights.safetensors"
    path.write_bytes(archive)
    entries = (WeightEntry(0, "shared", "F32", [1], 0, 4),
               WeightEntry(1, "shared", "F32", [1], 0, 4),
               WeightEntry(2, "resident", "F32", [2], 4, 8),
               WeightEntry(3, "host", "F32", [1], 12, 4))
    nodes = tuple(SemanticNode(f"arg{i}", "input", (), TensorType(tuple(e.shape), "f32", "raw-bit-pattern"),
                               effect="input") for i, e in enumerate(entries))
    source = KernelRequest(nodes, ("arg3",), ("external",),
                           tuple((n.id, "external") for n in nodes), "synthetic")
    original = ApplicationParameters(source.digest(), "a" * 64, sha256(archive).hexdigest(),
                                     sha256(payload).hexdigest(), len(payload), entries, path)
    original.check(source)
    return source, original


def test_compact_bytes_and_emitted_c_binding(tmp_path):
    for index, (removed, expected, offsets) in enumerate([
        (("arg2",), bytes(range(4)) + bytes(range(12, 16)), (0, 0, None, 4)),
        ((), bytes(range(16)), (0, 0, 4, 12)),
        (("arg0", "arg1", "arg2", "arg3"), b"\0", (None,) * 4),
    ]):
        case = tmp_path / f"case_{index}"
        case.mkdir()
        _compact_bytes_and_emitted_c_binding(case, removed, expected, offsets)


def _compact_bytes_and_emitted_c_binding(tmp_path, removed, expected, offsets):
    source, original = fixture(tmp_path)
    compact = compact_parameters(original, source, resident_only=removed)
    compact.check(source, resident_only=removed)
    assert compact.source_request_digest == original.source_request_digest
    assert compact.node_ids == original.node_ids
    assert compact.compact_offsets == offsets
    assert compact.payload_bytes == len(expected)
    assert compact.payload_sha256 == sha256(expected).hexdigest()
    record = compact.record()
    assert record["source_parameters"] == original.record()
    assert record["payload_sha256"] != original.payload_sha256 or not removed
    destination = tmp_path / "compact.bin"
    compact.write_payload(destination)
    assert destination.read_bytes() == expected
    with pytest.raises(FileExistsError):
        compact.write_payload(destination)
    assert destination.read_bytes() == expected
    for node, offset in zip(original.node_ids, offsets):
        assert compact.pointer_expression(node) == ("0" if offset is None else f"parameters->data+{offset}u")
    declarations, implementation, headers = compact.emit_binding("check")
    for name, text in headers.items():
        (tmp_path / name).write_text(text)
    actual = ",".join(map(str, expected))
    expressions = "\n".join(
        f"assert(({compact.pointer_expression(node)})=="
        + ("0);" if offset is None else f"bytes+{offset}u);")
        for node, offset in zip(original.node_ids, offsets))
    c = """
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <string.h>
#include "parameter_sha256.h"
static int check_valid(const void *p,size_t n) { return p && (uintptr_t)p<=UINTPTR_MAX-n; }
static int check_overlap(const void *p,size_t n,const void *q,size_t m) {
  return (uintptr_t)p<(uintptr_t)q+m && (uintptr_t)q<(uintptr_t)p+n;
}
""" + declarations + implementation + f"""
int main(void) {{
  _Alignas(8) unsigned char bytes[]={{{actual}}};
  check_parameters descriptor={{0}},before;
  check_parameters *parameters=&descriptor;
  assert(check_parameters_bind(parameters,bytes,sizeof(bytes))==0);
  assert(check_parameters_ready(parameters));
  {expressions}
  before=descriptor;bytes[0]^=1;
  assert(check_parameters_bind(parameters,bytes,sizeof(bytes))==-2);
  assert(memcmp(&before,parameters,sizeof(before))==0);
  assert(check_parameters_bind(parameters,bytes,sizeof(bytes)+1)==-1);
  return 0;
}}
"""
    (tmp_path / "check.c").write_text(c)
    compiler = shutil.which("cc")
    assert compiler, "C compiler is required for binding execution"
    built = subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "check.c", "-o", "check"],
                   cwd=tmp_path, capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    subprocess.run([str(tmp_path / "check")], check=True, capture_output=True)


def test_changed_archive_refuses_and_cleans_output(tmp_path):
    for index, damage in enumerate(["changed", "truncated"]):
        case = tmp_path / f"case_{index}"
        case.mkdir()
        _changed_archive_refuses_and_cleans_output(case, damage)


def _changed_archive_refuses_and_cleans_output(tmp_path, damage):
    source, original = fixture(tmp_path)
    compact = compact_parameters(original, source, resident_only=("arg2",))
    raw = original.archive_path.read_bytes()
    original.archive_path.write_bytes(raw[:-1] if damage == "truncated" else raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError):
        compact_parameters(original, source, resident_only=("arg2",))
    with pytest.raises(ValueError):
        compact.check(source)
    output = tmp_path / "bad.bin"
    with pytest.raises(ValueError):
        compact.write_payload(output)
    assert not output.exists()


def test_projection_and_source_proof_refuse(tmp_path):
    source, original = fixture(tmp_path)
    compact = compact_parameters(original, source, resident_only=("arg2",))
    for removed in (("arg2", "arg2"), ("unknown",), (1,)):
        with pytest.raises(ValueError):
            compact_parameters(original, source, resident_only=removed)
    for changed in (replace(compact, resident_only=("arg0",)),
                    replace(compact, compact_offsets=(0, 0, None, 0)),
                    replace(compact, payload_sha256="0" * 64)):
        with pytest.raises(ValueError):
            changed.check(source)
    with pytest.raises(ValueError):
        compact.check(source, resident_only=())
    with pytest.raises(ValueError):
        compact.check(replace(source, source_identity="changed"))
    with pytest.raises(ValueError):
        compact.pointer_expression("unknown")
    with pytest.raises(ValueError):
        compact.pointer_expression("arg0", "bad->name")


def test_aligned_projection_and_bounded_streaming(tmp_path, monkeypatch):
    payload = bytes(range(24))
    header = json.dumps({
        "bytes": {"dtype": "I8", "shape": [3], "data_offsets": [0, 3]},
        "removed": {"dtype": "I8", "shape": [13], "data_offsets": [3, 16]},
        "wide": {"dtype": "I64", "shape": [1], "data_offsets": [16, 24]},
    }).encode()
    archive = struct.pack("<Q", len(header)) + header + payload
    path = tmp_path / "weights.safetensors"
    path.write_bytes(archive)
    entries = (WeightEntry(0, "bytes", "I8", [3], 0, 3),
               WeightEntry(1, "removed", "I8", [13], 3, 13),
               WeightEntry(2, "wide", "I64", [1], 16, 8))
    nodes = tuple(SemanticNode(f"arg{i}", "input", (), TensorType(tuple(e.shape), dtype, "raw-bit-pattern"),
                               effect="input") for i, (e, dtype) in enumerate(zip(entries, ("i8", "i8", "i64"))))
    source = KernelRequest(nodes, ("arg2",), ("external",),
                           tuple((n.id, "external") for n in nodes), "synthetic")
    original = ApplicationParameters(source.digest(), "a" * 64, sha256(archive).hexdigest(),
                                     sha256(payload).hexdigest(), len(payload), entries, path)
    # Full-file byte reads are forbidden even for this tiny archive: construction,
    # checking and publication must use the verified streaming snapshot route.
    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", lambda _: pytest.fail("unbounded payload read"))
        compact = compact_parameters(original, source, resident_only=("arg1",))
        compact.check(source)
        compact.write_payload(tmp_path / "compact.bin")
    assert compact.compact_offsets == (0, None, 8)
    assert (tmp_path / "compact.bin").read_bytes() == payload[:3] + b"\0" * 5 + payload[16:]
