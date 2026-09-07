"""This target brings its own compiler, so it needs its own key — and its toolchain stamp is sacred.

The shared operator build path is cached by `build_cache.build_identity`, keyed on a declared harness
recipe. A target with its own compiler has no such recipe, and so was left rebuilding every grade:
measured across its runs on disk, 1,757 s of screen-tier build against 3,160 s of screen-tier
simulation, re-derived for artifacts that had not changed.

The stamp is the delicate part. `compile_for_oracle` records WHICH toolchain produced the graded ELF
precisely so the vendor fork can never be a silent fallback. A cached build that restored the files
without the stamp would be the single case where that record went missing.
"""
from __future__ import annotations

import pytest

from merlin.targetgen import build_cache as BC


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_ELF_BUILD_CACHE", str(tmp_path / "store"))
    return tmp_path / "store"


@pytest.fixture
def muon(monkeypatch):
    """The muon MODULE, not the backend package that re-exports it: the cache lives in the module,
    and the package is only a re-export. Reached through the loaded package so its relative imports
    resolve -- loading the file standalone cannot work."""
    from merlin.runtime.backends import base as B
    m = B.get_backend("muon").muon
    monkeypatch.setattr(m, "_build_cache_key",
                        lambda kind, target, inputs: BC.artifact_identity(
                            kind=kind, target=target, inputs=inputs,
                            source_files=[__file__], toolchain="stub-toolchain"))
    return m


def _elf(work, name="kernel.radiance.elf", body=b"bytes"):
    work.mkdir(parents=True, exist_ok=True)
    p = work / name
    p.write_bytes(b"\x7fELF" + body)
    return p


# --------------------------------------------------------------------------------------------
# the key
# --------------------------------------------------------------------------------------------

def test_a_generic_identity_needs_every_component():
    ok = BC.artifact_identity(kind="k", target="t", inputs={"a": 1},
                              source_files=[__file__], toolchain="tc")
    assert ok
    assert BC.artifact_identity(kind="", target="t", inputs={}, source_files=[__file__],
                                toolchain="tc") is None
    assert BC.artifact_identity(kind="k", target="", inputs={}, source_files=[__file__],
                                toolchain="tc") is None
    assert BC.artifact_identity(kind="k", target="t", inputs={}, source_files=[],
                                toolchain="tc") is None
    assert BC.artifact_identity(kind="k", target="t", inputs={}, source_files=[__file__],
                                toolchain=None) is None
    assert BC.artifact_identity(kind="k", target="t", inputs={}, source_files=["/nope/absent.py"],
                                toolchain="tc") is None


def test_inputs_reach_the_identity():
    def k(**inputs):
        return BC.artifact_identity(kind="k", target="t", inputs=inputs,
                                    source_files=[__file__], toolchain="tc")
    base = k(src="module {}")
    assert k(src="module {}") == base
    assert k(src="module { // edited }") != base
    assert k(src="module {}", warps=2) != base


def test_two_builders_cannot_collide_on_one_key():
    """`kind` separates the namespaces: two different compilers keyed the same would hand one
    builder's output to the other."""
    a = BC.artifact_identity(kind="one", target="t", inputs={"x": 1},
                             source_files=[__file__], toolchain="tc")
    b = BC.artifact_identity(kind="two", target="t", inputs={"x": 1},
                             source_files=[__file__], toolchain="tc")
    assert a and b and a != b


def test_a_different_toolchain_is_a_different_key():
    a = BC.artifact_identity(kind="k", target="t", inputs={}, source_files=[__file__], toolchain="A")
    b = BC.artifact_identity(kind="k", target="t", inputs={}, source_files=[__file__], toolchain="B")
    assert a != b


# --------------------------------------------------------------------------------------------
# the stamp
# --------------------------------------------------------------------------------------------

def test_the_stamp_is_stored_and_restored(tmp_path):
    key = BC.artifact_identity(kind="k", target="t", inputs={"x": 1},
                               source_files=[__file__], toolchain="tc")
    work = tmp_path / "w"
    _elf(work)
    BC.store(key, work, "kernel.radiance.elf", metadata={"toolchain": "fork-free"})
    assert BC.metadata_for(key) == {"toolchain": "fork-free"}
    second = tmp_path / "w2"
    got = BC.reuse(second, key)
    assert got is not None and got.name == "kernel.radiance.elf"


def test_the_primary_output_name_can_come_from_the_record(tmp_path):
    """This builder's output name is not the operator path's, and the caller does not always know it
    -- so a restore must be able to ask the record rather than be told."""
    key = BC.artifact_identity(kind="k", target="t", inputs={"x": 2},
                               source_files=[__file__], toolchain="tc")
    work = tmp_path / "w"
    _elf(work, "kernel.soc.elf")
    BC.store(key, work, "kernel.soc.elf")
    got = BC.reuse(tmp_path / "w2", key)
    assert got is not None and got.name == "kernel.soc.elf"


def test_a_record_for_a_different_primary_output_is_not_used(tmp_path):
    key = BC.artifact_identity(kind="k", target="t", inputs={"x": 3},
                               source_files=[__file__], toolchain="tc")
    work = tmp_path / "w"
    _elf(work, "kernel.soc.elf")
    BC.store(key, work, "kernel.soc.elf")
    assert BC.reuse(tmp_path / "w2", key, "kernel.radiance.elf") is None


def test_metadata_is_none_when_it_cannot_be_established(tmp_path):
    assert BC.metadata_for(None) is None
    assert BC.metadata_for("0" * 64) is None


def test_metadata_from_another_record_version_is_unavailable(tmp_path):
    """UNAVAILABLE, not empty. The caller rebuilds on unavailable and proceeds on empty, so a record
    this reader does not understand must not be flattened into "there was no stamp"."""
    import json
    key = BC.artifact_identity(kind="k", target="t", inputs={"x": 9},
                               source_files=[__file__], toolchain="tc")
    work = tmp_path / "w"
    _elf(work)
    BC.store(key, work, "kernel.radiance.elf", metadata={"toolchain": "fork-free"})
    assert BC.metadata_for(key) == {"toolchain": "fork-free"}
    rec = next(BC.cache_root().rglob("record.json"))
    doc = json.loads(rec.read_text())
    doc["version"] = BC.RECORD_VERSION + 1
    rec.write_text(json.dumps(doc))
    assert BC.metadata_for(key) is None, "a record from another version was read anyway"


def test_metadata_for_a_mismatched_key_is_unavailable(tmp_path):
    import json
    key = BC.artifact_identity(kind="k", target="t", inputs={"x": 10},
                               source_files=[__file__], toolchain="tc")
    work = tmp_path / "w"
    _elf(work)
    BC.store(key, work, "kernel.radiance.elf", metadata={"toolchain": "fork-free"})
    rec = next(BC.cache_root().rglob("record.json"))
    doc = json.loads(rec.read_text())
    doc["key"] = "f" * 64
    rec.write_text(json.dumps(doc))
    assert BC.metadata_for(key) is None


def test_a_build_stored_without_metadata_reports_empty_not_missing(tmp_path):
    """Empty and unavailable are different: the caller rebuilds on unavailable, and must be able to
    tell that from a build that genuinely carried no metadata."""
    key = BC.artifact_identity(kind="k", target="t", inputs={"x": 4},
                               source_files=[__file__], toolchain="tc")
    work = tmp_path / "w"
    _elf(work)
    BC.store(key, work, "kernel.radiance.elf")
    assert BC.metadata_for(key) == {}


def test_the_oracle_build_refuses_an_unstamped_hit():
    """THE FAIL-CLOSED DIRECTION. Restoring files without the toolchain stamp would make a cached
    build the one case where a vendor-fork fallback went unrecorded -- which is exactly the
    measurement `compile_for_oracle` exists to make."""
    from merlin.common.paths import repo_root
    src = (repo_root() / "merlin" / "targets" / "muon" / "backend" / "muon.py").read_text()
    i = src.index("def compile_for_oracle(")
    j = src.find("\ndef ", i + 1)
    body = src[i:j if j != -1 else len(src)]
    assert "if _hit is not None and _stamp:" in body, (
        "a hit must be used only when it carries a toolchain stamp")
    assert 'metadata={"toolchain": "fork-free"}' in body
    assert 'metadata={"toolchain": "clang-muon-fork"}' in body


def test_both_muon_build_paths_consult_the_cache():
    """Keying a build and never LOOKING UP the key is a cache that cannot hit -- the exact defect this
    whole series has been chasing -- so both the key and the lookup are pinned."""
    from merlin.common.paths import repo_root
    src = (repo_root() / "merlin" / "targets" / "muon" / "backend" / "muon.py").read_text()
    def _fn_body(text, name):
        i = text.index(f"def {name}(")
        j = text.find("\ndef ", i + 1)
        return text[i:j if j != -1 else len(text)]
    for fn, kind in (("compile_mlir_forkfree", "muon-mlir-forkfree"),
                     ("compile_for_oracle", "muon-oracle")):
        body = _fn_body(src, fn)
        assert kind in body, f"{fn} does not key its build"
        assert "_bc.reuse(" in body, f"{fn} computes a key but never consults the cache"
        assert "_bc.store(" in body, f"{fn} never stores what it built"


def test_the_mlir_path_returns_a_hit_rather_than_rebuilding(muon, tmp_path, monkeypatch):
    """Behavioural, not textual: with a stored build in place, the compile must return it without
    reaching the toolchain at all."""
    from merlin.targetgen.contract import toolchain as _tc
    key = muon._build_cache_key("muon-mlir-forkfree", "radiance",
                                {"mlir": "module {}", "cb": {}, "num_warps": 1})
    assert key, "the fixture must be able to form a key"
    work = tmp_path / "gen"
    _elf(work, "kernel.radiance.elf", b"prebuilt")
    BC.store(key, work, "kernel.radiance.elf")

    monkeypatch.setattr(_tc, "mlir_bin",
                        lambda *a, **k: pytest.fail("the toolchain was invoked despite a cache hit"))
    got = muon.compile_mlir_forkfree("module {}", {}, tmp_path / "fresh", target="radiance",
                                     num_warps=1)
    assert got is not None and got.read_bytes().endswith(b"prebuilt")
