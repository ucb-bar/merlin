"""Pure syntax reuse never reuses admission, filesystem or closure verdicts."""

from dataclasses import replace

import pytest
from test_compiler_library import _library

from merlin.targetgen import compiler_library as L


@pytest.fixture(autouse=True)
def empty_syntax_cache():
    L._cached_syntax_facts.cache_clear()
    yield
    L._cached_syntax_facts.cache_clear()


def test_reuse_parsing_but_reopen_and_hash_every_current_member(tmp_path, monkeypatch):
    root, contract = _library(tmp_path)
    L._cached_syntax_facts.cache_clear()
    reads, parses = [], []
    read, parse = L._read, L._parse_syntax_facts

    def counted_read(root, relative):
        reads.append(relative)
        return read(root, relative)

    def counted_parse(payload, filename):
        parses.append(filename)
        return parse(payload, filename)

    monkeypatch.setattr(L, "_read", counted_read)
    monkeypatch.setattr(L, "_parse_syntax_facts", counted_parse)
    contract.verify(root)
    assert parses == ["merlin/__init__.py", "merlin/portable.py"]
    first_reads = tuple(reads)
    contract.verify(root)
    assert tuple(reads) == first_reads + first_reads
    assert len(parses) == 2
    (root / "merlin/portable.py").write_text("def lower(value): return 0\n")
    with pytest.raises(L.CompilerLibraryError, match="bytes changed"):
        contract.verify(root)


def test_same_syntax_does_not_cache_allowed_membership_or_import_policy(tmp_path, monkeypatch):
    root, _ = _library(tmp_path)
    (root / "merlin/portable.py").write_text("from . import helper\ndef lower(x): return helper.value\n")
    (root / "merlin/helper.py").write_text("value = 1\n")
    contract = L.freeze_compiler_library(
        root,
        review_id="syntax-test",
        public_modules=("merlin.portable",),
        sources=(
            ("merlin/__init__.py", "merlin"),
            ("merlin/portable.py", "merlin.portable"),
            ("merlin/helper.py", "merlin.helper"),
        ),
    )
    contract.verify(root)
    without_helper = replace(contract, members=contract.members[:-1])
    with pytest.raises(L.CompilerLibraryError, match="unreviewed namespace import"):
        without_helper.verify(root)
    forbidden = L._forbidden
    monkeypatch.setattr(L, "_forbidden", lambda module: module == "merlin.helper" or forbidden(module))
    with pytest.raises(L.CompilerLibraryError, match="unreviewed library dependency"):
        contract.verify(root)


def test_current_namespace_exports_are_rederived_under_same_leaf_syntax(tmp_path):
    root, _ = _library(tmp_path)
    (root / "merlin/__init__.py").write_text("value = 1\n")
    (root / "merlin/portable.py").write_text("from . import value\n")
    contract = L.freeze_compiler_library(
        root,
        review_id="namespace-test",
        public_modules=("merlin.portable",),
        sources=(("merlin/__init__.py", "merlin"), ("merlin/portable.py", "merlin.portable")),
    )
    contract.verify(root)
    (root / "merlin/__init__.py").write_text("different = 2\n")
    with pytest.raises(L.CompilerLibraryError, match="unreviewed namespace import"):
        L.freeze_compiler_library(
            root,
            review_id="namespace-test",
            public_modules=("merlin.portable",),
            sources=(("merlin/__init__.py", "merlin"), ("merlin/portable.py", "merlin.portable")),
        )


def test_retained_facts_are_immutable_and_large_sources_are_not_cached(monkeypatch):
    calls = []
    parse = L._parse_syntax_facts

    def counted(payload, filename):
        calls.append(filename)
        return parse(payload, filename)

    monkeypatch.setattr(L, "_parse_syntax_facts", counted)
    payload = b"#" + b"a" * (256 << 10) + b"\nvalue = 1\n"
    assert L._syntax_facts(payload, "large.py") == (("value",), ())
    assert L._syntax_facts(payload, "large.py") == (("value",), ())
    assert calls == ["large.py", "large.py"]
    assert L._cached_syntax_facts.cache_info().currsize == 0
    facts = L._syntax_facts(b"from .helper import value\n", "small.py")
    assert facts == (("value",), (("from", 1, "helper", ("value",)),))
    with pytest.raises(TypeError):
        facts[0][0] = "changed"
    for index in range(129):
        L._syntax_facts(f"value = {index}\n".encode(), "small.py")
    assert L._cached_syntax_facts.cache_info().currsize == 128


@pytest.mark.parametrize("payload", [b"def unfinished(", b"\xff"])
def test_invalid_syntax_never_becomes_a_cached_success(payload):
    for _ in range(2):
        with pytest.raises((SyntaxError, UnicodeError)):
            L._syntax_facts(payload, "invalid.py")
    assert L._cached_syntax_facts.cache_info().currsize == 0


def test_replaced_syntax_parser_cannot_inherit_previous_parse_facts(monkeypatch):
    payload = b"value = 1\n"
    assert L._syntax_facts(payload, "source.py") == (("value",), ())
    parse = L.ast.parse
    calls = []

    def replacement(*args, **kwargs):
        calls.append(True)
        return parse(*args, **kwargs)

    monkeypatch.setattr(L.ast, "parse", replacement)
    assert L._syntax_facts(payload, "source.py") == (("value",), ())
    assert calls == [True]
