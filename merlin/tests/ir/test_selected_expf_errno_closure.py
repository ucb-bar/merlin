"""Explicit all-input errno closure and actual linked pure-getter refusal tests."""

import hashlib
import struct
from dataclasses import FrozenInstanceError

import pytest

from merlin.llvmlower import selected_expf_equivalence as E
from merlin.llvmlower import selected_expf_errno_closure as C


@pytest.fixture
def closure(tmp_path, monkeypatch):
    library = tmp_path / "libm.a"
    library.write_bytes(b"selected archive fixture")
    monkeypatch.setattr(E, "SELECTED_LIBM_SHA256", hashlib.sha256(library.read_bytes()).hexdigest())
    return C.select_expf_errno_closure(E.select_expf_underflow_equivalence(library))


def test_explicit_immutable_and_live(closure):
    assert closure.preserved_errno == 34
    with pytest.raises(FrozenInstanceError):
        closure.underflow = None
    closure.underflow.library_path.write_bytes(b"changed")
    with pytest.raises(E.ExpfEquivalenceError, match="bytes changed"):
        closure.revalidate()


def test_explicit_nested_compiler_scope(closure):
    assert C.current_expf_errno_closure() is None
    with pytest.raises(E.ExpfEquivalenceError, match="live underflow scope"):
        with C.selected_expf_errno_closure_scope(closure):
            pass
    with E.selected_expf_underflow_scope(closure.underflow):
        with C.selected_expf_errno_closure_scope(closure):
            assert C.current_expf_errno_closure() is closure
        assert C.current_expf_errno_closure() is None


def test_scope_reopens_archive_and_cleans_up(closure):
    with pytest.raises(E.ExpfEquivalenceError, match="bytes changed"):
        with E.selected_expf_underflow_scope(closure.underflow):
            with C.selected_expf_errno_closure_scope(closure):
                closure.underflow.library_path.write_bytes(b"changed")
                C.current_expf_errno_closure()
    assert C.current_expf_errno_closure() is None


def test_other_underflow_scope_refuses(closure):
    other = E.select_expf_underflow_equivalence(closure.underflow.library_path)
    with E.selected_expf_underflow_scope(other):
        with pytest.raises(E.ExpfEquivalenceError, match="exact live"):
            with C.selected_expf_errno_closure_scope(closure):
                pass


@pytest.mark.parametrize("foreign", [None, {}, object()])
def test_foreign_selection_refuses(foreign):
    with pytest.raises(E.ExpfEquivalenceError):
        C.select_expf_errno_closure(foreign)


def _evidence(tmp_path, closure, *, code=None, overflow="original", getter_count=1):
    code = struct.pack("<IIH", 0x00115517, 0x4A050513, 0x8082) if code is None else code
    data = bytearray(0x100 + len(code))
    data[:6] = b"\x7fELF\x02\x01"
    data[18:20] = b"\xf3\x00"
    struct.pack_into("<Q", data, 32, 64)
    struct.pack_into("<HH", data, 54, 56, 1)
    struct.pack_into("<IIQQQQQQ", data, 64, 1, 5, 0x100, 0x80000000, 0, len(code), len(code), 2)
    data[0x100:] = code
    elf = tmp_path / "output.elf"
    elf.write_bytes(data)
    rows = [f" {s} 0x80001000 0x10 {closure.underflow.library_path}({m})" for s, m in E._SECTIONS.items()]
    owner = f"{closure.underflow.library_path}(libm_a-math_errf.o)" if overflow == "original" else overflow
    rows.append(f" .text.__math_oflowf\n 0x80002000 0x10 {owner}")
    rows.extend([f" .text.__errno 0x80000000 {len(code):#x} runtime.o"] * getter_count)
    path = tmp_path / "output.map"
    path.write_text("\n".join(rows))
    return elf, path


def test_complete_live_link(closure, tmp_path):
    elf, path = _evidence(tmp_path, closure)
    receipt = C.verify_expf_errno_closure_link(closure, elf=elf, map_path=path)
    assert receipt["all_input_errno_store_values"] == [34]
    assert receipt["errno_preserved_from"] == 34
    assert receipt["qualification"] == "UNQUALIFIED"


@pytest.mark.parametrize("index", [0, 1, 4, 5, 6, 8, 9])
def test_changed_getter_instruction_refuses(closure, tmp_path, index):
    code = bytearray(struct.pack("<IIH", 0x00115517, 0x4A050513, 0x8082))
    # Change operand/opcode bits, not an unrestricted address immediate.
    bit = 0 if index in {1, 6} else 4
    code[index] ^= 1 << bit
    elf, path = _evidence(tmp_path, closure, code=code)
    with pytest.raises(E.ExpfEquivalenceError, match="pure fixed-pointer"):
        C.verify_expf_errno_closure_link(closure, elf=elf, map_path=path)


@pytest.mark.parametrize("count", [0, 2])
def test_missing_duplicate_getter_refuses(closure, tmp_path, count):
    elf, path = _evidence(tmp_path, closure, getter_count=count)
    with pytest.raises(E.ExpfEquivalenceError, match="one linked getter"):
        C.verify_expf_errno_closure_link(closure, elf=elf, map_path=path)


def test_overflow_override_refuses(closure, tmp_path):
    elf, path = _evidence(tmp_path, closure, overflow="foreign.o")
    with pytest.raises(E.ExpfEquivalenceError, match="overflow path"):
        C.verify_expf_errno_closure_link(closure, elf=elf, map_path=path)


@pytest.mark.parametrize("mutation", ["wrong_segment", "truncated", "header_size", "missing_headers"])
def test_malformed_elf_refuses(closure, tmp_path, mutation):
    elf, path = _evidence(tmp_path, closure)
    data = bytearray(elf.read_bytes())
    if mutation == "wrong_segment":
        struct.pack_into("<I", data, 64, 2)
    elif mutation == "truncated":
        del data[-1]
    elif mutation == "header_size":
        struct.pack_into("<H", data, 54, 48)
    else:
        struct.pack_into("<H", data, 56, 0)
    elf.write_bytes(data)
    with pytest.raises(E.ExpfEquivalenceError):
        C.verify_expf_errno_closure_link(closure, elf=elf, map_path=path)
