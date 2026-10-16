"""A provider's source roster is data; shared build machinery contains no ISA."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from merlin_timing import provider_bundle as B


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "bundle"
    B.export_bundle(
        {"observer.cc": b"// independent observer\n", "facts.json": b"{}\n"}, root, entrypoint="observer.cc"
    )
    return root


def test_distinct_provider_rosters_and_independent_builder(tmp_path, bundle):
    second = tmp_path / "second"
    receipt = B.export_bundle({"other.cpp": b"// other ISA\n"}, second, entrypoint="other.cpp")
    assert receipt["entrypoint"] == "other.cpp"
    assert {row["path"] for row in receipt["members"]} == {"other.cpp", B.BUILDER}
    assert (second / B.BUILDER).read_bytes() == Path(B.__file__).read_bytes()
    for directory in (bundle, second):
        result = subprocess.run(
            [sys.executable, "-I", "-S", directory / B.BUILDER, "--verify-only"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert json.loads(result.stdout) == B.verify_bundle(directory)


@pytest.mark.parametrize("mutation", ["changed", "missing", "extra", "directory", "link", "large"])
def test_modified_bundle_refuses(bundle, tmp_path, mutation):
    source = bundle / "observer.cc"
    if mutation == "changed":
        source.write_bytes(b"changed\n")
    elif mutation == "missing":
        source.unlink()
    elif mutation == "extra":
        (bundle / "unlisted.cc").write_bytes(b"extra")
    elif mutation == "directory":
        source.unlink()
        source.mkdir()
    elif mutation == "link":
        source.unlink()
        source.symlink_to(tmp_path / "absent")
    else:
        source.write_bytes(b"x" * (B.MAX_BYTES + 1))
    with pytest.raises((ValueError, OSError)):
        B.verify_bundle(bundle)


@pytest.mark.parametrize("mutation", ["escape", "duplicate", "entrypoint", "boolean_size", "duplicate_json"])
def test_manifest_cannot_redirect_inputs(bundle, mutation):
    path = bundle / B.MANIFEST
    document = json.loads(path.read_bytes())
    if mutation == "escape":
        document["members"][0]["path"] = "../outside"
    elif mutation == "duplicate":
        document["members"].append(document["members"][0])
    elif mutation == "entrypoint":
        document["entrypoint"] = "unlisted.cc"
    elif mutation == "boolean_size":
        document["members"][0]["bytes"] = True
    else:
        path.write_text('{"schema":"one","schema":"two"}')
    if mutation != "duplicate_json":
        path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        B.verify_bundle(bundle)


def test_export_and_build_cannot_write_into_sources(bundle, tmp_path):
    with pytest.raises(ValueError, match="already exists"):
        B.export_bundle({"main.cc": b""}, bundle, entrypoint="main.cc")
    linked = tmp_path / "linked"
    linked.symlink_to(bundle, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        B.export_bundle({"main.cc": b""}, linked, entrypoint="main.cc")
    for destination in (bundle / "provider.so", tmp_path / "absent" / ".." / "bundle" / "provider.so"):
        with pytest.raises(ValueError, match="outside the closed"):
            B.build(bundle, tmp_path, destination, "unavailable-compiler")
    for payloads, entrypoint in (({"../outside.cc": b""}, "../outside.cc"), ({B.BUILDER: b""}, B.BUILDER)):
        with pytest.raises(ValueError):
            B.export_bundle(payloads, tmp_path / "new", entrypoint=entrypoint)


@pytest.mark.parametrize("change", [None, "source", "runtime"])
def test_build_pins_inputs_and_publishes_a_new_inode(bundle, tmp_path, monkeypatch, change):
    runtime = tmp_path / "runtime"
    for name in B.RUNTIME_HEADERS:
        header = runtime / name
        header.parent.mkdir(parents=True, exist_ok=True)
        header.write_bytes(b"generic timing ABI")
    output = tmp_path / "provider.so"
    output.write_bytes(b"old mapped library")
    with output.open("rb") as mapped:

        def compile_provider(argv, *, check, timeout, env):
            assert check and timeout == 60
            assert argv[-4] == str(runtime)
            assert str(bundle / "observer.cc") in argv
            assert Path(env["TMPDIR"]).parent == output.parent
            Path(argv[-1]).write_bytes(b"new provider")
            if change == "source":
                (bundle / "observer.cc").write_bytes(b"changed")
            if change == "runtime":
                (runtime / B.RUNTIME_HEADERS[0]).write_bytes(b"changed")

        monkeypatch.setattr(B.subprocess, "run", compile_provider)
        if change:
            with pytest.raises(ValueError):
                B.build(bundle, runtime, output, "selected-cxx")
            assert output.read_bytes() == b"old mapped library"
        else:
            receipt = B.build(bundle, runtime, output, "selected-cxx")
            assert receipt["jobs"] == 1 and len(receipt["generic_header_sha256"]) == 2
            assert output.read_bytes() == b"new provider"
        assert mapped.read() == b"old mapped library"
