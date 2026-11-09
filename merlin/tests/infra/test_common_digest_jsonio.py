"""Record and file content identity: exact bytes, explicit readers and live reuse scopes.

Legacy serializer spellings/pinned vectors retain their original byte contracts.
Real kernel watches remain separate obligations from serialization or numerical
authority; shared files exercise both ordinary streamed and guarded SHA readers.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from merlin.common import guarded_file_digest as G
from merlin.common.digest import is_sha256, sha256_bytes, sha256_file, sha256_text
from merlin.common.jsonio import (
    canonical_json,
    canonical_sha256,
    readable_json,
    strict_json_equal,
    write_canonical_json,
    write_pretty_json,
)
from merlin.common.strict_json import loads

CORPUS = [
    {"b": 1, "a": [1, 2.5, None, True]},
    {"unicode": "é → ✓", "nested": {"z": {"y": [0.1, -3, 1e300]}}},
    [],
    "plain",
    {"k": ""},
]


ORIGINAL_ASCII_EQUIVALENT = {
    "lax .encode('utf-8')": lambda v: json.dumps(v, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    "lax .encode()": lambda v: json.dumps(v, sort_keys=True, separators=(",", ":")).encode(),
    "strict ascii": lambda v: json.dumps(
        v, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii"),
    "strict .encode()": lambda v: json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(),
}


@pytest.fixture
def source_file(tmp_path, request):
    path = tmp_path / "selected.fir"
    path.write_bytes(getattr(request, "param", b"original source\n"))
    return path


@pytest.mark.parametrize("value", CORPUS, ids=range(len(CORPUS)))
@pytest.mark.parametrize(
    "name,options",
    [
        *[(name, {}) for name in sorted(ORIGINAL_ASCII_EQUIVALENT)],
        ("raw UTF-8", {"ensure_ascii": False}),
        ("newline", {"trailing_newline": True}),
        ("raw UTF-8 newline", {"ensure_ascii": False, "trailing_newline": True}),
    ],
)
def test_serialization_preserves_each_original_byte_contract(value, name, options):
    if name in ORIGINAL_ASCII_EQUIVALENT:
        expected = ORIGINAL_ASCII_EQUIVALENT[name](value)
    else:
        text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=options.get("ensure_ascii", True))
        expected = (text + ("\n" if options.get("trailing_newline") else "")).encode("utf-8")
    assert canonical_json(value, **options) == expected
    assert canonical_sha256(value, **options) == hashlib.sha256(expected).hexdigest()


def test_nan_is_refused_unless_the_lax_contract_is_asked_for():
    with pytest.raises(ValueError):
        canonical_json({"x": math.nan})
    lax = json.dumps({"x": math.nan}, sort_keys=True, separators=(",", ":")).encode()
    assert canonical_json({"x": math.nan}, allow_nan=True) == lax


_TYPED_RECORD = {"kind": "index", "ordinals": [0, 1], "checked": True, "none": None}


@pytest.mark.parametrize(
    "update",
    [
        {"ordinals": [False, 1]},
        {"ordinals": [0.0, 1]},
        {"ordinals": (0, 1)},
        {"extra": 1},
        {"ordinals": [0, math.nan]},
    ],
)
def test_strict_json_equality_refuses_numeric_container_and_field_aliases(update):
    assert not strict_json_equal({**_TYPED_RECORD, **update}, _TYPED_RECORD)


def test_strict_json_equality_accepts_the_same_typed_record():
    assert strict_json_equal(_TYPED_RECORD, json.loads(canonical_json(_TYPED_RECORD)))


def test_pinned_vectors():
    v = {"b": 1, "a": [1, 2.5, None, True], "u": "é"}
    assert canonical_json(v) == b'{"a":[1,2.5,null,true],"b":1,"u":"\\u00e9"}'
    assert canonical_sha256(v) == hashlib.sha256(b'{"a":[1,2.5,null,true],"b":1,"u":"\\u00e9"}').hexdigest()
    assert sha256_bytes(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert sha256_text("é") == hashlib.sha256("é".encode()).hexdigest()


@pytest.mark.parametrize("source_file", [bytes(range(256)) * 9000], indirect=True, ids=["multiple-1MiB-chunks"])
def test_file_digest_streams_and_matches_an_independent_whole_read(source_file):
    assert sha256_file(source_file) == hashlib.sha256(source_file.read_bytes()).hexdigest()
    with pytest.raises(OSError):
        sha256_file(source_file.parent / "absent")


def test_is_sha256_is_strict_unless_told_otherwise():
    good = "a" * 64
    assert is_sha256(good)
    assert not is_sha256(good.upper()) and is_sha256(good.upper(), allow_upper=True)
    assert not is_sha256("a" * 63) and not is_sha256("g" * 64) and not is_sha256(None)
    assert not is_sha256(b"a" * 64)


@pytest.mark.parametrize(
    "writer,options,creates_parent,expected",
    [
        (write_canonical_json, {}, True, b'{"a":2,"b":1}\n'),
        (write_pretty_json, {}, False, b'{\n  "a": 2,\n  "b": 1\n}\n'),
        (write_pretty_json, {"mkdir": True}, True, b'{\n  "a": 2,\n  "b": 1\n}\n'),
    ],
)
def test_json_writers_preserve_bytes_and_explicit_parent_creation(tmp_path, writer, options, creates_parent, expected):
    path = tmp_path / "sub" / "record.json"
    if not creates_parent:
        with pytest.raises(FileNotFoundError):
            writer(path, {"b": 1, "a": 2}, **options)
        path.parent.mkdir()
    writer(path, {"b": 1, "a": 2}, **options)
    assert path.read_bytes() == expected


@pytest.mark.parametrize(
    "value",
    [
        {"tensor": [[1, 2], [3, 4]], "rows": [{"a": 1}, 2], "empty": [], "none": {}, 7: None, True: "t"},
        [1.5, "x", None, [True, False]],
        {"nested": [[{"deep": [1, 2]}]], "text": 'café "quoted"\n'},
        "scalar",
    ],
)
def test_parses_to_what_json_dumps_parses_to(value):
    assert json.loads(readable_json(value)) == json.loads(json.dumps(value))


def test_numeric_arrays_take_one_line_and_objects_stay_indented():
    text = readable_json({"W": [[1, 2, 3], [4, 5, 6]], "meta": {"shape": [2, 3]}})
    assert '"W": [[1,2,3],[4,5,6]]' in text
    assert '\n  "meta": {\n    "shape": [2,3]\n  }' in text
    assert len(readable_json({"W": list(range(1000))})) < len(json.dumps({"W": list(range(1000))}, indent=2)) / 2


def test_unsupported_keys_are_refused_like_json_dumps():
    with pytest.raises(TypeError):
        readable_json({(1, 2): "tuple key"})


@pytest.mark.parametrize(
    "raw,options,error,message",
    [
        (b'{"claim":false,"claim":true}', {}, ValueError, "duplicate JSON key"),
        (b'{"nested":{"x":1,"x":2}}', {}, ValueError, "duplicate JSON key"),
        *[
            ('{"ignored":[{"number":' + token + "}]}", {}, ValueError, "non-finite JSON value")
            for token in ("NaN", "Infinity", "-Infinity", "1e999", "-1e999")
        ],
        ('"é"', {"max_bytes": 3}, ValueError, "bounded reader"),
        (b"[] ", {"max_bytes": 2}, ValueError, "bounded reader"),
        *[(b"0", {"max_bytes": limit}, ValueError, "reader limit") for limit in (-1, True, 1.5)],
        ({}, {}, TypeError, "bytes or text"),
        *[(raw, {}, ValueError, None) for raw in (b'"\xff"', b'{"x":', b"[1,]")],
    ],
)
def test_strict_record_reader_refuses_ambiguous_unbounded_or_invalid_input(raw, options, error, message):
    with pytest.raises(error, match=message):
        loads(raw, **options)


def test_valid_records_preserve_json_types_and_values():
    raw = '{"text":"NaN", "integer":123456789123456789, "negative":-17, "finite":1e-300, "array":[null,true]}'
    assert loads(raw) == json.loads(raw)
    assert loads(raw.encode()) == json.loads(raw)
    assert loads("0", max_bytes=1) == 0


@pytest.fixture
def watched(source_file, monkeypatch):
    monkeypatch.setenv("MERLIN_SNAPSHOT_LEASE_CACHE", "1")
    path = source_file
    owner = G.GuardedDigests()
    calls = []
    read = G._read

    def counted(path):
        calls.append(path)
        return read(path)

    monkeypatch.setattr(G, "_read", counted)
    yield path, owner, calls
    owner.close()


def test_concurrent_scopes_hash_once_and_unscoped_reads_remain_fresh(watched):
    path, owner, calls = watched
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    with owner.scope():
        assert G.sha256(path) == expected
    if not owner._entries:
        pytest.skip("kernel watches unavailable")

    def verify(_):
        with owner.scope():
            return G.sha256(path)

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(verify, range(12))) == [expected] * 12
    assert len(calls) == 1
    G.sha256(path)
    G.sha256(path)
    assert len(calls) == 3


@pytest.mark.parametrize("mutation", ["write", "hardlink", "replace", "symlink", "after_last_consumer"])
def test_mutation_never_reuses_old_bytes_even_with_restored_mtime(watched, mutation):
    path, owner, _ = watched
    original = path.stat()
    alias = path.parent / "alias"
    final_check = mutation == "after_last_consumer"
    if mutation == "hardlink":
        os.link(path, alias)
    with pytest.raises(ValueError, match="bytes changed" if final_check else None):
        with owner.scope():
            G.sha256(path)
            if not owner._entries:
                pytest.skip("kernel watches unavailable")
            if final_check:
                path.write_bytes(b"changed after observation")
            elif mutation == "replace":
                alias.write_bytes(b"different bytes\n")
                alias.replace(path)
            elif mutation == "symlink":
                path.rename(alias)
                path.symlink_to(alias)
            else:
                (alias if mutation == "hardlink" else path).write_bytes(b"modified source\n")
                os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
            if not final_check:
                G.sha256(path)


def test_watch_unavailable_uses_full_reads(watched, monkeypatch):
    path, owner, calls = watched
    monkeypatch.delenv("MERLIN_SNAPSHOT_LEASE_CACHE")
    with owner.scope():
        G.sha256(path)
        G.sha256(path)
    assert len(calls) == 2 and not owner._entries


def test_closed_guard_falls_back_and_never_rearms(watched):
    path, owner, calls = watched
    with owner.scope():
        G.sha256(path)
        if not owner._entries:
            pytest.skip("kernel watches unavailable")
        owner._entries[path][1].close()  # also the overflow/error transition
        G.sha256(path)
        G.sha256(path)
    assert len(calls) == 4  # entry, two reads, publication recheck
    assert owner._entries[path][1].fd is None


def test_inherited_and_closed_owners_cannot_reuse(watched, monkeypatch):
    path, owner, calls = watched
    with owner.scope():
        G.sha256(path)
    monkeypatch.setattr(owner, "pid", -1)
    with pytest.raises(ValueError, match="original live process"):
        with owner.scope():
            pass
    before = len(calls)
    owner._digest(path)
    assert len(calls) == before + 1
    monkeypatch.setattr(owner, "pid", os.getpid())
    owner.close()
    with pytest.raises(ValueError, match="original live process"):
        with owner.scope():
            pass
