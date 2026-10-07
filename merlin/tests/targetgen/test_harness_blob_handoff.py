"""A renderer's exact bytes survive the target-neutral harness sidecar seam."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess

import pytest

from merlin.targetgen.contract.harness_blobs import render_blob_asm, stage_harness_blobs


def test_blob_sidecar_assembles_to_exact_aligned_bytes(tmp_path):
    if not shutil.which("cc") or not shutil.which("objcopy"):
        pytest.skip("requires a host C toolchain")
    payload = bytes(range(32))
    (source,) = stage_harness_blobs(tmp_path, {"T_W": {"bytes": payload, "align": 16, "elems": 32}})
    assert (tmp_path / "harness_blob_T_W.bin").read_bytes() == payload
    receipt = json.loads((tmp_path / "harness_blobs.json").read_text())
    assert receipt["schema"] == "merlin.harness_blobs.v1"
    assert receipt["blobs"] == [
        {
            "symbol": "T_W",
            "file": "harness_blob_T_W.bin",
            "assembly": "harness_blob_T_W.S",
            "object": "harness_blob_T_W.o",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": 32,
            "alignment": 16,
            "elements": 32,
        }
    ]
    assert '.incbin "harness_blob_T_W.bin"' in source.read_text()
    result = subprocess.run(["cc", "-c", source.name, "-o", "blob.o"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    objcopy = subprocess.run(
        ["objcopy", "--dump-section", ".rodata=blob.raw", "blob.o"], cwd=tmp_path, capture_output=True, text=True
    )
    assert objcopy.returncode == 0, objcopy.stderr
    assert (tmp_path / "blob.raw").read_bytes() == payload


@pytest.mark.parametrize(
    "declaration",
    [
        {"T/escape": {"bytes": b"x", "align": 1, "elems": 1}},
        {"T_W": {"bytes": b"x", "align": 3, "elems": 1}},
        {"T_W": {"bytes": b"x", "align": 1, "elems": 2}},
        {"T_W": {"bytes": b"x", "align": 1, "elems": 0}},
    ],
)
def test_invalid_blob_declarations_fail_closed(tmp_path, declaration):
    with pytest.raises(ValueError):
        stage_harness_blobs(tmp_path, declaration)


def test_blob_assembler_rejects_unsafe_payload_names():
    for name in ('weights".bin', "weights\n.global injected.bin", "../weights.bin", "weights\\escape.bin"):
        with pytest.raises(ValueError, match="invalid harness blob payload name"):
            render_blob_asm("T_W", name, align=16, elems=32)


def test_generic_linker_passes_renderer_blobs_to_the_executable(tmp_path, monkeypatch):
    if not shutil.which("cc"):
        pytest.skip("requires a host C compiler")
    from merlin.runtime.backends import base
    from merlin.targetgen import runtime_build
    from merlin.targetgen.contract import compile as compiler

    class Recipe:
        load_address = 0
        link_script = tmp_path / "unused.ld"
        support_sources = ()
        error_cls = RuntimeError

        def with_effective_abi(self):
            # The host compiler's default ABI is the one both halves of this ELF are built for.
            return self

        @staticmethod
        def compile_command(*, source, output):
            return ["cc", "-c", str(source), "-o", str(output)]

        @staticmethod
        def link_command(*, objects, output, link_script):
            return ["cc", *(str(item) for item in objects), "-o", str(output)]

    def render(_cb, *, target, blobs):
        assert target == "synthetic"
        blobs["T_W"] = {"bytes": b"\x07\x08\x09\x0a", "align": 4, "elems": 4}
        return "extern const unsigned char T_W[4]; int main(void) { return T_W[0] != 7 || T_W[3] != 10; }\n"

    monkeypatch.setattr(base, "harness_build_recipe", lambda target: Recipe())
    monkeypatch.setattr(base, "harness_renderer", lambda target: render)
    monkeypatch.setattr(runtime_build, "derived_link_script", lambda *args: tmp_path / "unused.ld")
    kernel = tmp_path / "kernel.o"
    subprocess.run(["cc", "-c", "-x", "c", "-", "-o", str(kernel)], input="\n", text=True, check=True)
    elf = compiler.link_elf({}, kernel, tmp_path, target="synthetic")
    assert subprocess.run([str(elf)], check=False).returncode == 0
    assert (tmp_path / "harness_blob_T_W.bin").read_bytes() == b"\x07\x08\x09\x0a"


def test_build_only_requires_an_explicit_sidecar_capability(monkeypatch):
    from merlin.targetgen.contract.build_service import BuildOnlyService

    monkeypatch.setattr(BuildOnlyService, "verify", lambda self, target: None)
    seen = []

    def legacy_wrapper(_cb, **kwargs):
        seen.append(kwargs)
        return "legacy"

    legacy = BuildOnlyService("synthetic", object(), legacy_wrapper, ())
    assert legacy.render({}, target="synthetic", inputs={"W": [1]}, blobs={}) == "legacy"
    assert seen == [{"inputs": {"W": [1]}}]

    def blob_renderer(_cb, *, inputs, blobs):
        blobs["T_W"] = {"bytes": b"\x07", "align": 1, "elems": 1}
        return "sidecar"

    offered = {}
    opt_in = BuildOnlyService("synthetic", object(), blob_renderer, ())
    assert opt_in.render({}, target="synthetic", inputs={"W": [1]}, blobs=offered) == "sidecar"
    assert offered["T_W"]["bytes"] == b"\x07"


@pytest.mark.parametrize("support_basename", ["harness", "kernel"])
def test_contract_link_keeps_support_and_imported_objects_distinct(tmp_path, monkeypatch, support_basename):
    """A provider cannot clobber the generated harness or an imported device object."""
    if not shutil.which("cc"):
        pytest.skip("requires a host C compiler")
    from merlin.runtime.backends import base
    from merlin.targetgen import runtime_build
    from merlin.targetgen.contract import compile as compiler

    support = tmp_path / "support"
    support.mkdir()
    source = support / f"{support_basename}.c"
    source.write_text("int right(void) { return 25; }\n")

    class Recipe:
        load_address = 0
        link_script = tmp_path / "unused.ld"
        support_sources = (source,)
        error_cls = RuntimeError

        def with_effective_abi(self):
            # The synthetic native harness has no target ABI overrides.
            return self

        @staticmethod
        def compile_command(*, source, output):
            return ["cc", "-c", str(source), "-o", str(output)]

        @staticmethod
        def link_command(*, objects, output, link_script):
            return ["cc", *(str(item) for item in objects), "-o", str(output)]

    def render(_cb, *, target, blobs):
        return "extern int left(void), right(void); int main(void) { return left()+right()==42 ? 0 : 1; }\n"

    monkeypatch.setattr(base, "harness_build_recipe", lambda _: Recipe())
    monkeypatch.setattr(base, "harness_renderer", lambda _: render)
    monkeypatch.setattr(runtime_build, "derived_link_script", lambda *_: Recipe.link_script)
    kernel = tmp_path / "kernel.o"
    subprocess.run(
        ["cc", "-c", "-x", "c", "-", "-o", str(kernel)], input="int left(void) { return 17; }\n", text=True, check=True
    )
    original = kernel.read_bytes()
    elf = compiler.link_elf({}, kernel, tmp_path, target="synthetic")
    assert subprocess.run([str(elf)], check=False).returncode == 0
    assert kernel.read_bytes() == original
