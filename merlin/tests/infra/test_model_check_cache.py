"""Kernel objects and replay consoles are reused only for identical inputs, and any package byte invalidates."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.model_check import cache as KC
from merlin.model_check import replay as RP

_IFACE = "module {}\n"


def _catalog(interfaces: dict[str, str]):
    return SimpleNamespace(
        target="t",
        interfaces=interfaces,
        sidecar={"entries": {s: {"op": "matmul"} for s in interfaces}},
        routing={"operand_dtype": "int8", "accum_dtype": "int32", "numeric_policy": None},
    )


class _Build:
    """Stands in for the device build: one tiny host object per symbol, or a refusal."""

    def __init__(self, refuse: dict[str, str] | None = None, interface: str | None = None):
        self.calls: list[list[str]] = []
        self.refuse = refuse or {}
        self.interface = interface

    def __call__(self, target, sidecar, symbols, *, package_dir, workdir, toolchain, **_kw):
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        self.calls.append(list(symbols))
        kernels, skipped, objects = {}, [], []
        for index, symbol in enumerate(symbols):
            if symbol in self.refuse:
                skipped.append((symbol, self.refuse[symbol]))
                continue
            source = workdir / f"{symbol}.c"
            source.write_text(f"int kern_{index}(void) {{ return {index + 1}; }}\n")
            subprocess.run(["cc", "-c", str(source), "-o", str(workdir / f"{symbol}.o")], check=True)
            (workdir / f"{symbol}.iface.mlir").write_text(self.interface or _IFACE)
            kernels[symbol] = f"kern_{index}"
            objects.append(workdir / f"{symbol}.o")
        return SimpleNamespace(kernels=kernels, skipped=tuple(skipped), objects=tuple(objects))


def _defined(obj: Path) -> set[str]:
    out = subprocess.run(["nm", "--defined-only", str(obj)], capture_output=True, text=True, check=True).stdout
    return {line.split()[-1] for line in out.splitlines() if line.strip()}


needs_cc = pytest.mark.skipif(shutil.which("cc") is None or shutil.which("nm") is None, reason="needs cc and nm")


def test_package_digest_sees_every_byte_and_follows_links(tmp_path):
    pkg = tmp_path / "pkg"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "manifest.yaml").write_text("a: 1\n")
    outside = tmp_path / "tool.py"
    outside.write_text("print(1)\n")
    (pkg / "sub" / "tool.py").symlink_to(outside)
    before = KC.package_digest(pkg)
    assert KC.package_digest(pkg) == before
    outside.write_text("print(2)\n")  # the linked file's bytes are the package's
    after = KC.package_digest(pkg)
    assert after != before
    (pkg / "manifest.yaml").write_text("a: 2\n")
    assert KC.package_digest(pkg) != after


@needs_cc
def test_objects_are_reused_until_the_package_changes(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "manifest.yaml").write_text("v: 1\n")
    catalog = _catalog({"a": _IFACE, "b": _IFACE + "// b\n"})
    toolchain = SimpleNamespace(device_cflags=("-O2",))
    build = _Build(interface=None)

    def run(work):
        return KC.build_kernels(
            catalog,
            ["a"],
            package_dir=pkg,
            workdir=tmp_path / work,
            toolchain=toolchain,
            build=build,
            root=tmp_path / "c",
        )

    first = run("w1")
    assert build.calls == [["a"]] and first.stats["stored"] == 1
    (obj,) = first.objects
    assert first.kernels["a"] in _defined(obj) and str(obj).startswith(str(tmp_path / "c"))
    second = run("w2")
    assert build.calls == [["a"]] and second.stats["hits"] == 1  # nothing rebuilt
    assert second.kernels == first.kernels and second.objects == first.objects
    (pkg / "manifest.yaml").write_text("v: 2\n")
    third = run("w3")
    assert build.calls == [["a"], ["a"]] and third.stats["misses"] == 1


@needs_cc
def test_refusals_are_stored_but_timeouts_and_foreign_interfaces_are_not(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "manifest.yaml").write_text("v: 1\n")
    catalog = _catalog({"r": _IFACE, "t": _IFACE + "// t\n"})
    toolchain = SimpleNamespace(device_cflags=())
    build = _Build(refuse={"r": "package lowering: BackendDeclined: no", "t": "package lowering: timed out after 9 s"})
    kwargs = {"package_dir": pkg, "toolchain": toolchain, "build": build, "root": tmp_path / "c"}
    first = KC.build_kernels(catalog, ["r", "t"], workdir=tmp_path / "w1", **kwargs)
    assert first.refused == {"r": "package lowering: BackendDeclined: no", "t": "package lowering: timed out after 9 s"}
    second = KC.build_kernels(catalog, ["r", "t"], workdir=tmp_path / "w2", **kwargs)
    assert build.calls[-1] == ["t"] and second.refused == first.refused  # only the timeout is retried
    foreign = _Build(interface="module { /* another interface */ }\n")
    other = _catalog({"f": _IFACE + "// f\n"})
    built = KC.build_kernels(other, ["f"], workdir=tmp_path / "w3", **{**kwargs, "build": foreign})
    assert built.stats["stored"] == 0 and built.objects == [tmp_path / "w3" / "f.o"]  # used, never stored


@needs_cc
def test_without_the_cache_the_build_is_returned_as_built(tmp_path):
    build = _Build()
    built = KC.build_kernels(
        _catalog({"a": _IFACE}),
        ["a"],
        package_dir=tmp_path,
        workdir=tmp_path / "w",
        toolchain=SimpleNamespace(device_cflags=()),
        build=build,
        enabled=False,
    )
    assert built.kernels == {"a": "kern_0"} and built.objects == [tmp_path / "w" / "a.o"]


def test_a_replay_console_is_reused_only_for_the_same_image(tmp_path, monkeypatch):
    harness = tmp_path / "harness"
    harness.mkdir()
    for name in ("crt.S", "htif.c", "htif.h", "model_link.ld"):
        (harness / name).write_text(name)
    toolchain = SimpleNamespace(gcc="cc", gcc_cflags=("-O2",), device_cflags=(), harness_dir=harness)
    monkeypatch.setattr(KC, "engine_identity", lambda *a, **k: {"binary": "spike", "flags": []})
    runs: list[Path] = []
    monkeypatch.setattr(RP, "link_image", lambda work, *a, **k: work / "replay.elf")

    def fake_run(target, elf, **_kw):
        runs.append(elf)
        return f"MCCASE 0 0 0x1 0x2\nMCDONE run{len(runs)}\n", 1.5

    monkeypatch.setattr(RP, "run_image", fake_run)
    blob = tmp_path / "blob_0.bin"
    blob.write_bytes(b"\x01" * 64)
    main_c, data_s = tmp_path / "main.c", tmp_path / "data.S"
    main_c.write_text("int main(void) { return 0; }\n")
    data_s.write_text(f'.section .weights, "a"\n.incbin "{blob}"\n')
    obj = tmp_path / "k.o"
    obj.write_bytes(b"object")
    kwargs = {"target": "t", "dram_base": 0, "dram_bytes": 1, "timeout": 5, "root": tmp_path / "c"}
    first = KC.replay_image(tmp_path, main_c, data_s, [obj], toolchain, **kwargs)
    second = KC.replay_image(tmp_path, main_c, data_s, [obj], toolchain, **kwargs)
    assert len(runs) == 1 and second[0] == first[0] and second[2]["cached"] and not first[2]["cached"]
    moved = tmp_path / "moved.bin"  # the same operand bytes at another path: the same image
    moved.write_bytes(blob.read_bytes())
    data_s.write_text(f'.section .weights, "a"\n.incbin "{moved}"\n')
    assert KC.replay_image(tmp_path, main_c, data_s, [obj], toolchain, **kwargs)[2]["cached"]
    moved.write_bytes(b"\x02" * 64)  # one operand byte: another image
    assert not KC.replay_image(tmp_path, main_c, data_s, [obj], toolchain, **kwargs)[2]["cached"]
    obj.write_bytes(b"object2")  # one object byte: another image
    assert not KC.replay_image(tmp_path, main_c, data_s, [obj], toolchain, **kwargs)[2]["cached"]
    assert not KC.replay_image(tmp_path, main_c, data_s, [obj], toolchain, **kwargs, enabled=False)[2]["cached"]
    assert len(runs) == 4
