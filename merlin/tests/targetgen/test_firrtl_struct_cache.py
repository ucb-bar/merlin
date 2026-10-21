"""Module-line reuse is content-only; source checks and caller ownership stay fresh."""

import os

import pytest

from merlin.targetgen.rtl import firrtl_struct as F

SOURCE = b"""FIRRTL version 4.2.0
circuit Design:
  module First:
    input x : UInt<8>
    node result = add(x, UInt<8>(1))
  module Second:
    input y : UInt<8>
    node other = add(y, UInt<8>(2))
"""


@pytest.fixture(autouse=True)
def fresh_cache():
    F._reset_body_cache()
    yield
    F._reset_body_cache()


def count_parses(monkeypatch):
    original = F._read_module_lines
    calls = []

    def read(lines, wanted):
        calls.append(wanted)
        return original(lines, wanted)

    monkeypatch.setattr(F, "_read_module_lines", read)
    return calls


def test_identical_content_across_paths_reuses_parse_but_fully_reads_each_file(tmp_path, monkeypatch):
    first, second = tmp_path / "first.fir", tmp_path / "second.fir"
    first.write_bytes(SOURCE)
    second.write_bytes(SOURCE)
    calls = count_parses(monkeypatch)
    original_digest = F.hashlib.file_digest
    reads = []

    def digest(stream, algorithm):
        value = original_digest(stream, algorithm)
        assert stream.read() == b""
        reads.append(stream.name)
        return value

    monkeypatch.setattr(F.hashlib, "file_digest", digest)
    expected = F.read_module_bodies(first, ["First", "Second"])
    assert F.read_module_bodies(second, ["Second", "First", "First"]) == expected
    assert F.read_module_bodies(first, ["First", "Second"]) == expected
    assert len(calls) == 1
    assert reads == [str(first), str(second), str(first)]


@pytest.mark.parametrize("selected", [True, False])
def test_same_size_restored_mtime_mutation_reparses_complete_content(tmp_path, monkeypatch, selected):
    path = tmp_path / "selected.fir"
    path.write_bytes(SOURCE)
    calls = count_parses(monkeypatch)
    before = F.read_module_bodies(path, ["First"])
    stamp = path.stat()
    changed = SOURCE.replace(b"UInt<8>(1)" if selected else b"UInt<8>(2)", b"UInt<8>(3)")
    assert len(changed) == len(SOURCE)
    path.write_bytes(changed)
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    after = F.read_module_bodies(path, ["First"])
    assert (after != before) is selected
    assert len(calls) == 2  # Even an unrequested module changes the complete source identity.


def test_returned_lines_and_parsed_module_have_no_mutable_cache_alias(tmp_path):
    path = tmp_path / "selected.fir"
    path.write_bytes(SOURCE)
    before = F.read_module_bodies(path, ["First"])
    expected = {name: list(lines) for name, lines in before.items()}
    before["First"].clear()
    before["foreign"] = [(1, "foreign")]
    assert F.read_module_bodies(path, ["First"]) == expected
    module = F.load_modules(path, ["First"])["First"]
    module.nodes.clear()
    module.decls.clear()
    assert "result" in F.load_modules(path, ["First"])["First"].nodes
    assert "x" in F.load_modules(path, ["First"])["First"].decls


def test_requested_names_and_newly_present_modules_are_fresh(tmp_path):
    path = tmp_path / "selected.fir"
    path.write_bytes(SOURCE)
    assert set(F.read_module_bodies(path, ["First"])) == {"First"}
    assert set(F.read_module_bodies(path, ["Second"])) == {"Second"}
    assert F.read_module_bodies(path, ["Later"]) == {}
    path.write_bytes(SOURCE + b"  module Later:\n    input z : UInt<8>\n")
    assert set(F.read_module_bodies(path, ["Later"])) == {"Later"}


def test_removed_source_never_reuses_warm_result(tmp_path):
    path = tmp_path / "selected.fir"
    path.write_bytes(SOURCE)
    assert "First" in F.read_module_bodies(path, ["First"])
    path.unlink()
    with pytest.raises(FileNotFoundError):
        F.read_module_bodies(path, ["First"])


def test_miss_uses_digest_of_actual_parsed_bytes_not_earlier_lookup(tmp_path, monkeypatch):
    path = tmp_path / "selected.fir"
    path.write_bytes(SOURCE)
    changed = SOURCE.replace(b"UInt<8>(1)", b"UInt<8>(9)")
    original = F._read_module_lines

    def replace_before_read(lines, wanted):
        path.write_bytes(changed)
        return original(lines, wanted)

    monkeypatch.setattr(F, "_read_module_lines", replace_before_read)
    mutated = F.read_module_bodies(path, ["First"])
    monkeypatch.setattr(F, "_read_module_lines", original)
    path.write_bytes(SOURCE)
    assert F.read_module_bodies(path, ["First"]) != mutated
    path.write_bytes(changed)
    assert F.read_module_bodies(path, ["First"]) == mutated


@pytest.mark.parametrize("newline", [b"\n", b"\r\n", b"\r"])
def test_text_decoding_and_universal_newlines_keep_original_meaning(tmp_path, newline):
    path = tmp_path / "selected.fir"
    raw = SOURCE.replace(b"node result", b"node result\xff").replace(b"\n", newline)
    path.write_bytes(raw)
    with path.open(encoding="utf-8", errors="replace") as stream:
        expected = F._read_module_lines(stream, frozenset(["First"]))
    assert F.read_module_bodies(path, ["First"]) == expected
    assert F.read_module_bodies(path, ["First"]) == expected


def test_cache_entry_bound_and_oversized_results_bypass_reuse(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "_BODY_CACHE_ENTRIES", 2)
    path = tmp_path / "selected.fir"
    for digit in b"1234":
        path.write_bytes(SOURCE.replace(b"UInt<8>(1)", b"UInt<8>(" + bytes([digit]) + b")"))
        assert "First" in F.read_module_bodies(path, ["First"])
    assert len(F._body_cache) == 2
    assert F._body_cache_bytes <= F._BODY_CACHE_BYTES
    F._reset_body_cache()
    monkeypatch.setattr(F, "_BODY_CACHE_BYTES", 1)
    calls = count_parses(monkeypatch)
    expected = F.read_module_bodies(path, ["First"])
    assert F.read_module_bodies(path, ["First"]) == expected
    assert len(calls) == 2
    assert F._body_cache_bytes == 0
