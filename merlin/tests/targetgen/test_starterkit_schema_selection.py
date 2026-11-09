"""Source-only starter-kit components select a public schema explicitly and fail closed."""

import builtins
import json

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen.oot_starterkit import cmdbuf
from merlin.targetgen.oot_starterkit.verify import legal_functs, structural_checks, validate


@pytest.fixture
def public_schema(tmp_path):
    schema = tmp_path / "public-command-buffer.schema.json"
    source = repo_root() / "merlin/contract/schemas/command_buffer.schema.json"
    schema.write_bytes(source.read_bytes())
    return schema


def test_source_only_builder_and_verifier_use_selected_public_schema(public_schema, monkeypatch, tmp_path):
    monkeypatch.setattr(cmdbuf, "_SCHEMA", tmp_path / "not-packaged.json")
    builder = cmdbuf.CommandBufferBuilder("unrelated_target", schema_path=public_schema).command("EVICT")
    buffer = builder.to_dict()
    assert validate(cb=buffer, schema_path=public_schema) == {"ok": True, "findings": []}
    bad = dict(buffer, commands=[{"opcode": "not_a_declared_command"}])
    verdict = validate(cb=bad, schema_path=public_schema)
    assert not verdict["ok"] and any("schema:" in finding for finding in verdict["findings"])
    builder._cb = bad
    with pytest.raises(ValueError, match="schema-valid"):
        builder.to_dict()


def test_explicit_missing_schema_is_not_replaced_by_packaged_schema(public_schema, tmp_path):
    builder = cmdbuf.CommandBufferBuilder("unrelated_target", schema_path=tmp_path / "missing.json").command("EVICT")
    with pytest.raises(FileNotFoundError):
        builder.validate()


def test_explicit_broken_schema_fails_closed(tmp_path):
    schema = tmp_path / "broken.json"
    schema.write_text("not JSON")
    with pytest.raises(json.JSONDecodeError):
        cmdbuf.CommandBufferBuilder("unrelated_target", schema_path=schema).command("EVICT").validate()


def test_default_packaged_schema_selection_is_preserved(public_schema, monkeypatch):
    monkeypatch.setattr(cmdbuf, "_SCHEMA", public_schema)
    assert cmdbuf.CommandBufferBuilder("unrelated_target").command("EVICT").validate() == []


def test_missing_default_schema_still_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(cmdbuf, "_SCHEMA", tmp_path / "not-packaged.json")
    with pytest.raises(FileNotFoundError):
        cmdbuf.CommandBufferBuilder("unrelated_target").command("EVICT").validate()


def test_explicit_schema_requires_schema_validator(public_schema, monkeypatch):
    original_import = builtins.__import__

    def without_jsonschema(name, *args, **kwargs):
        if name == "jsonschema":
            raise ImportError("jsonschema deliberately unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_jsonschema)
    with pytest.raises(ImportError, match="deliberately unavailable"):
        cmdbuf.CommandBufferBuilder("unrelated_target", schema_path=public_schema).command("EVICT").validate()


def test_two_unrelated_public_isa_vocabularies_share_the_same_verifier(tmp_path):
    systolic = tmp_path / "matrix.h"
    simt = tmp_path / "threads.h"
    systolic.write_text("#define MATRIX_PREPARE 0\n#define MATRIX_RUN 4\n")
    simt.write_text("#define THREAD_BARRIER 0x1\n#define THREAD_LAUNCH 0x8\n")

    for header, prefix, names, codes in (
        (systolic, "MATRIX_", ("PREPARE", "RUN"), (0, 4)),
        (simt, "THREAD_", ("BARRIER", "LAUNCH"), (1, 8)),
    ):
        legal = legal_functs(header, define_prefix=prefix)
        assert tuple(legal) == names
        trace = {"instructions": [{"name": names[1], "funct": codes[1]}, {"name": names[0], "funct": codes[0]}]}
        assert structural_checks(trace, legal) == []  # ordering is target-owned
        assert validate(trace=trace, isa_header=header, define_prefix=prefix)["ok"]
        trace["instructions"].append({"name": "OUTSIDE", "funct": 99})
        assert not validate(trace=trace, isa_header=header, define_prefix=prefix)["ok"]

    assert structural_checks({"instructions": "malformed"})
