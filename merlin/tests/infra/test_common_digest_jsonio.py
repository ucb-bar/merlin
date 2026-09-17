"""merlin.common.digest / merlin.common.jsonio reproduce, byte for byte, every private helper they replace.

Each ORIGINAL spelling found in the tree is reproduced verbatim below and compared with the shared helper
over a corpus that includes non-ASCII text, floats, nesting and booleans -- so migrating a call site onto
the shared helper cannot change a persisted digest. Pinned vectors guard the helpers themselves.
"""

from __future__ import annotations

import hashlib
import json
import math

import pytest

from merlin.common.digest import is_sha256, sha256_bytes, sha256_file, sha256_text
from merlin.common.jsonio import canonical_json, canonical_sha256, write_canonical_json, write_pretty_json

CORPUS = [
    {"b": 1, "a": [1, 2.5, None, True]},
    {"unicode": "é → ✓", "nested": {"z": {"y": [0.1, -3, 1e300]}}},
    [],
    "plain",
    {"k": ""},
]

# The original spellings, verbatim (compare/paper_*, perf/*, runtime/*, baselines/*).
ORIGINAL_ASCII_EQUIVALENT = {
    "lax .encode('utf-8')": lambda v: json.dumps(v, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    "lax .encode()": lambda v: json.dumps(v, sort_keys=True, separators=(",", ":")).encode(),
    "strict ascii": lambda v: json.dumps(
        v, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii"),
    "strict .encode()": lambda v: json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(),
}


@pytest.mark.parametrize("name", sorted(ORIGINAL_ASCII_EQUIVALENT))
@pytest.mark.parametrize("value", CORPUS, ids=range(len(CORPUS)))
def test_every_ascii_spelling_is_byte_identical(name, value):
    assert canonical_json(value) == ORIGINAL_ASCII_EQUIVALENT[name](value)
    assert canonical_sha256(value) == hashlib.sha256(ORIGINAL_ASCII_EQUIVALENT[name](value)).hexdigest()


@pytest.mark.parametrize("value", CORPUS, ids=range(len(CORPUS)))
def test_raw_utf8_and_newline_contracts_keep_their_own_bytes(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert canonical_json(value, ensure_ascii=False) == raw.encode("utf-8")  # targetgen/gsim_emulator
    nl = (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")
    assert canonical_json(value, trailing_newline=True) == nl  # compare/frozen_environment
    nl_raw = (raw + "\n").encode("utf-8")
    assert canonical_json(value, ensure_ascii=False, trailing_newline=True) == nl_raw  # perf/deployment_admissibility


def test_nan_is_refused_unless_the_lax_contract_is_asked_for():
    with pytest.raises(ValueError):
        canonical_json({"x": math.nan})
    lax = json.dumps({"x": math.nan}, sort_keys=True, separators=(",", ":")).encode()
    assert canonical_json({"x": math.nan}, allow_nan=True) == lax


def test_pinned_vectors():
    v = {"b": 1, "a": [1, 2.5, None, True], "u": "é"}
    assert canonical_json(v) == b'{"a":[1,2.5,null,true],"b":1,"u":"\\u00e9"}'
    assert canonical_sha256(v) == hashlib.sha256(b'{"a":[1,2.5,null,true],"b":1,"u":"\\u00e9"}').hexdigest()
    assert sha256_bytes(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert sha256_text("é") == hashlib.sha256("é".encode("utf-8")).hexdigest()


def test_sha256_file_streams_and_matches_the_whole_read(tmp_path):
    p = tmp_path / "big.bin"
    p.write_bytes(bytes(range(256)) * 9000)  # > one 1 MiB chunk
    assert sha256_file(p) == hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(OSError):
        sha256_file(tmp_path / "absent")


def test_is_sha256_is_strict_unless_told_otherwise():
    good = "a" * 64
    assert is_sha256(good)
    assert not is_sha256(good.upper()) and is_sha256(good.upper(), allow_upper=True)
    assert not is_sha256("a" * 63) and not is_sha256("g" * 64) and not is_sha256(None)
    assert not is_sha256(b"a" * 64)


def test_writers(tmp_path):
    p = tmp_path / "sub" / "c.json"
    write_canonical_json(p, {"b": 1, "a": 2})
    assert p.read_bytes() == b'{"a":2,"b":1}\n'
    q = tmp_path / "d.json"
    write_pretty_json(q, {"b": 1, "a": 2})
    assert q.read_text() == '{\n  "a": 2,\n  "b": 1\n}\n'
    with pytest.raises(FileNotFoundError):
        write_pretty_json(tmp_path / "missing" / "e.json", {})
    write_pretty_json(tmp_path / "made" / "e.json", {}, mkdir=True)
