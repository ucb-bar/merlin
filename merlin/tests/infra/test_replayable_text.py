"""Complete raw UTF-8 transport, including boundaries and source drift."""

import hashlib
import tracemalloc

import pytest

from merlin.common.replayable_text import ReplayableText


@pytest.mark.parametrize("chunk", [1, 2, 3, 4, 7, 16, 65536])
@pytest.mark.parametrize(
    "text", ["", "\n", "a\r\nb\rc\nd\ve\ff\x1cg\x1dh\x1ei\x85j\u2028k\u2029", "α\r\nβ\nlast", "unterminated"]
)
def test_every_line_matches_complete_python_splitlines(tmp_path, text, chunk):
    path = tmp_path / "trace"
    raw = text.encode("utf-8")
    path.write_bytes(raw)
    stream = ReplayableText(path, max_bytes=len(raw), chunk_bytes=chunk)
    assert list(stream.splitlines()) == text.splitlines()
    assert list(stream.splitlines()) == text.splitlines()
    assert stream.record()["sha256"] == hashlib.sha256(raw).hexdigest()
    assert stream.record()["bytes"] == len(raw)


def test_file_ceiling_is_raw_bytes_not_characters_or_compressed_size(tmp_path):
    path = tmp_path / "trace"
    path.write_bytes("α".encode())
    with pytest.raises(ValueError, match="byte budget"):
        ReplayableText(path, max_bytes=1)


def test_changed_source_before_or_during_complete_pass_refuses(tmp_path):
    path = tmp_path / "trace"
    path.write_text("a\nb\nc\n")
    stream = ReplayableText(path, max_bytes=100, chunk_bytes=1)
    lines = stream.splitlines()
    assert next(lines) == "a"
    path.write_text("a\nb\nd\n")
    with pytest.raises(ValueError, match="changed"):
        list(lines)
    with pytest.raises(ValueError, match="changed"):
        list(stream.splitlines())


def test_byte_identical_replacement_does_not_inherit_original_custody(tmp_path):
    path = tmp_path / "trace"
    path.write_bytes(b"a\nb\n")
    stream = ReplayableText(path, max_bytes=100)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(path.read_bytes())
    replacement.replace(path)
    with pytest.raises(ValueError, match="changed"):
        stream.verify()


@pytest.mark.parametrize("raw", [b"\xff", b"good\n\xe2\x82"])
def test_invalid_or_truncated_utf8_cannot_produce_complete_lines(tmp_path, raw):
    path = tmp_path / "trace"
    path.write_bytes(raw)
    stream = ReplayableText(path, max_bytes=100, chunk_bytes=1)
    with pytest.raises(UnicodeDecodeError):
        list(stream.splitlines())


def test_leaf_symlink_is_not_an_ordinary_selected_trace(tmp_path):
    path = tmp_path / "trace"
    path.write_text("line")
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="ordinary"):
        ReplayableText(link, max_bytes=100)


def test_replaying_many_lines_does_not_allocate_the_complete_text(tmp_path):
    path = tmp_path / "trace"
    line = b"complete original line\n"
    count = 200_000
    path.write_bytes(line * count)
    stream = ReplayableText(path, max_bytes=len(line) * count)
    tracemalloc.start()
    try:
        assert sum(1 for _ in stream.splitlines()) == count
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 2 << 20
