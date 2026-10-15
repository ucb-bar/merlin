"""Building the device side of an offloaded model into linkable objects.

The pieces either side already existed: the rewrite mints a symbol per contraction signature, the
shim adapts the MLIR calling convention, and a target's own package emits a device kernel from an
interface capsule. Missing was the step that runs the package once per distinct extent and gives each
result a distinct symbol.

The failure this prevents is not a link error. A package emits its entry under the single name the
backend contract declares, so several kernels linked together without renaming resolve every call to
whichever object came first -- a model quietly runs one layer's kernel for every layer, computes
numbers, and is wrong. `test_each_signature_gets_a_distinct_kernel_symbol` is that check.

Requires a usable backend package and toolchain; skips (never fails) without them.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from merlin.llvmlower.device_build import build_device_objects, kernel_symbol

pytestmark = pytest.mark.target("gemmini", "atlas", "radiance", "saturn_opu_mxv256d128")

_PKG = os.environ.get("MERLIN_TEST_DEVICE_PACKAGE")
_SIGS = {"d0": (16, 16, 32), "d1": (8, 64, 128)}
_DTS = {"d0": ("i8", "i8", "i32"), "d1": ("i8", "i8", "i32")}


def _nm():
    from pathlib import Path

    from merlin.llvmlower.toolchain import DEFAULT_LLVM_INSTALL

    local = Path(DEFAULT_LLVM_INSTALL) / "bin" / "llvm-nm"
    return str(local) if local.exists() else (shutil.which("llvm-nm") or shutil.which("nm"))


def _built(tmp_path):
    if not _PKG:
        pytest.skip("set MERLIN_TEST_DEVICE_PACKAGE to a backend package to exercise the build")
    b = build_device_objects(
        "gemmini", _SIGS, _DTS, package_dir=_PKG, workdir=tmp_path, operand_dtype="int8", accum_dtype="i32", timeout=900
    )
    if not b.ok:
        pytest.skip(f"device build unavailable here: {b.skipped}")
    return b


def _syms(path, kind):
    nm = _nm()
    if nm is None:
        pytest.skip("no nm available")
    out = subprocess.run([nm, f"--{kind}-only", str(path)], capture_output=True, text=True)
    return {ln.split()[-1] for ln in out.stdout.splitlines() if ln.strip() and not ln.endswith(":")}


# ------------------------------------------------------------------ symbol distinctness


def test_kernel_symbols_are_distinct_by_construction():
    """Pure, so it holds without a package: the naming itself must not collide."""
    assert kernel_symbol("k", 0) != kernel_symbol("k", 1)
    assert len({kernel_symbol("k", i) for i in range(8)}) == 8


def test_each_signature_gets_a_distinct_kernel_symbol(tmp_path):
    b = _built(tmp_path)
    assert len(set(b.kernels.values())) == len(b.kernels) > 1, (
        "two signatures sharing a kernel symbol would silently run one layer's kernel for both"
    )


def test_repeated_artifacts_compile_once_but_keep_distinct_symbols(tmp_path, monkeypatch):
    """A reduced repeated-group proxy: three signatures, two identical emitted kernels."""
    from pathlib import Path
    from types import SimpleNamespace

    from merlin.llvmlower import device_build, device_shim, toolchain
    from merlin.targetgen import corpus_spec, package_runtime

    monkeypatch.setattr(device_build, "objects_buildable", lambda device: None)
    monkeypatch.setattr(device_build, "_objcopy", lambda: "objcopy")
    monkeypatch.setattr("merlin.compile.mesh._mesh_tile_binding", lambda *args, **kwargs: None)
    monkeypatch.setattr(device_shim, "kernel_abi_for", lambda device: SimpleNamespace(symbol="kernel"))
    monkeypatch.setattr(
        device_shim,
        "emit_translation_unit",
        lambda device, signatures, *args, **kwargs: SimpleNamespace(
            symbols=tuple(signatures), text="void entry(void) {}", skipped=()
        ),
    )
    monkeypatch.setattr(corpus_spec, "build", lambda entry, binding: (None, entry["name"]))
    monkeypatch.setattr(package_runtime, "load_package", lambda path: SimpleNamespace(target="synthetic"))
    emitted = {"a": "same artifact", "b": "same artifact", "c": "different artifact"}
    from merlin.targetgen import capsule_common

    def lower(pkg, path, generated, **kwargs):
        generated.mkdir(parents=True, exist_ok=True)
        artifact = emitted[path.read_text()]
        (generated / "lowered.llvm.mlir").write_text(artifact)
        return {}, artifact

    monkeypatch.setattr(capsule_common, "lower_interface", lower)
    monkeypatch.setattr(
        package_runtime,
        "run_entrypoint",
        lambda pkg, name, path, **kwargs: SimpleNamespace(returncode=0, stdout=emitted[path.read_text()], stderr=""),
    )
    monkeypatch.setattr(toolchain, "mlir_translate", lambda: "translate")
    monkeypatch.setattr(toolchain, "clang", lambda: "clang")
    calls = []

    def fake_run(argv, *, timeout):
        calls.append(argv[0])
        if argv[0] == "objcopy":
            Path(argv[-1]).write_bytes(Path(argv[-2]).read_bytes() + argv[1].encode())
        else:
            Path(argv[argv.index("-o") + 1]).write_bytes(str(argv).encode())
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(device_build, "_run_build_tool", fake_run)
    built = device_build.build_device_objects(
        "synthetic",
        {s: (2, 2, 2) for s in emitted},
        {s: ("i8", "i8", "i32") for s in emitted},
        package_dir=tmp_path,
        workdir=tmp_path / "build",
        operand_dtype="i8",
        accum_dtype="i32",
    )
    assert built.ok
    assert built.object_dedup == {"unique_artifacts": 2, "emitted_symbols": 3}
    assert calls.count("translate") == 2
    assert calls.count("clang") == 3  # two kernels and the shared shim
    assert calls.count("objcopy") == 3
    assert len(set(built.kernels.values())) == 3
    assert len({path.read_bytes() for path in built.objects[:3]}) == 3


def test_every_kernel_object_defines_exactly_its_renamed_symbol(tmp_path):
    b = _built(tmp_path)
    for sym, kernel in b.kernels.items():
        obj = next(o for o in b.objects if o.stem == sym)
        assert kernel in _syms(obj, "defined")


def test_the_shim_defines_the_entries_and_needs_the_kernels(tmp_path):
    b = _built(tmp_path)
    defined = _syms(b.shim_object, "defined")
    undefined = _syms(b.shim_object, "undefined")
    assert set(b.kernels) <= defined, "the shim must define one entry per signature"
    assert set(b.kernels.values()) <= undefined, "and call the kernels by their renamed symbols"


# ------------------------------------------------------------------ the archive


def test_the_archive_carries_everything_the_host_link_needs(tmp_path):
    b = _built(tmp_path)
    a = b.archive(tmp_path / "libdevice.a")
    assert a is not None and a.is_file() and a.stat().st_size > 0
    exported = _syms(a, "defined")
    assert set(b.kernels) <= exported and set(b.kernels.values()) <= exported


def test_archiving_nothing_yields_nothing_rather_than_an_empty_archive(tmp_path):
    from merlin.llvmlower.device_build import DeviceBuild

    assert DeviceBuild(device="d").archive(tmp_path / "x.a") is None


# ------------------------------------------------------------------ partial failure is reported


def test_an_unusable_package_is_reported_not_raised(tmp_path):
    b = build_device_objects(
        "gemmini", _SIGS, _DTS, package_dir=tmp_path / "nope", workdir=tmp_path, operand_dtype="int8", accum_dtype="i32"
    )
    assert not b.ok and b.skipped and any("package" in why for _, why in b.skipped)


@pytest.mark.parametrize("stop_on_first_failure, expected", [(False, ["a", "b"]), (True, ["a"])])
def test_required_whole_model_kernel_stops_after_first_tool_timeout(
    tmp_path, monkeypatch, stop_on_first_failure, expected
):
    """A required kernel's compile timeout must not start the next expensive kernel."""
    from pathlib import Path
    from types import SimpleNamespace

    from merlin.compile import mesh
    from merlin.llvmlower import device_build as DB
    from merlin.llvmlower import device_shim, toolchain
    from merlin.targetgen import corpus_spec, package_runtime

    monkeypatch.setattr(DB, "objects_buildable", lambda _device: None)
    monkeypatch.setattr(device_shim, "kernel_abi_for", lambda _device: SimpleNamespace(symbol="kernel"))
    monkeypatch.setattr(DB, "kernel_entry", lambda *_args: ({"op": "matmul"}, "stated_group", ""))
    monkeypatch.setattr(DB, "_objcopy", lambda: "objcopy")
    monkeypatch.setattr(mesh, "_mesh_tile_binding", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(corpus_spec, "build", lambda *_args: (None, "module {}"))
    monkeypatch.setattr(package_runtime, "load_package", lambda _path: object())
    monkeypatch.setattr(toolchain, "mlir_translate", lambda: "mlir-translate")
    monkeypatch.setattr(toolchain, "clang", lambda: "clang")
    seen = []

    def emit(_pkg, _entrypoint, path, **_kwargs):
        seen.append(Path(path).name.removesuffix(".iface.mlir"))
        return SimpleNamespace(returncode=0, stdout="module {}", stderr="")

    def tool(argv, *, timeout):
        if argv[0] == "clang":
            return subprocess.CompletedProcess(argv, 124, "", f"timed out after {timeout} seconds")
        if argv[0] == "mlir-translate":
            Path(argv[-1]).write_text("define void @kernel() { ret void }\n")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(package_runtime, "run_entrypoint", emit)
    from merlin.targetgen import capsule_common

    def lower(pkg, path, generated, **kwargs):
        generated.mkdir(parents=True, exist_ok=True)
        artifact = emit(pkg, "emit_target_artifact", path, **kwargs).stdout
        (generated / "lowered.llvm.mlir").write_text(artifact)
        return {}, artifact

    monkeypatch.setattr(capsule_common, "lower_interface", lower)
    monkeypatch.setattr(DB, "_run_build_tool", tool)
    kwargs = {"stop_on_first_failure": True} if stop_on_first_failure else {}
    result = build_device_objects(
        "neutral",
        {"a": (2, 2, 2), "b": (3, 3, 3)},
        {"a": ("i8", "i8", "i32"), "b": ("i8", "i8", "i32")},
        package_dir=tmp_path / "package",
        workdir=tmp_path / "work",
        operand_dtype="int8",
        accum_dtype="i32",
        timeout=7,
        **kwargs,
    )
    assert seen == expected
    assert [symbol for symbol, _reason in result.skipped] == expected
    assert all("clang: timed out after 7 seconds" in reason for _symbol, reason in result.skipped)
    assert not result.ok and not result.objects and result.shim_object is None


def test_fail_fast_selection_requires_a_bool(tmp_path):
    with pytest.raises(ValueError, match="stop_on_first_failure must be a bool"):
        build_device_objects(
            "neutral",
            {},
            {},
            package_dir=tmp_path,
            workdir=tmp_path / "work",
            operand_dtype="int8",
            accum_dtype="i32",
            stop_on_first_failure=1,
        )


def test_exact_model_build_hands_the_selected_interface_to_the_oot_package(tmp_path, monkeypatch):
    """The build must not regenerate a same-shaped but different capsule."""
    from types import SimpleNamespace

    from merlin.common.digest import sha256_text
    from merlin.common.tree_hash import hash_tree
    from merlin.targetgen import oot_runner
    from merlin.targetgen.contract.interface_emit import emit_interface_mlir

    cb = {
        "abi_version": "0.1",
        "target": "gemmini",
        "tensors": {
            "B": {"shape": [19, 8], "dtype": "i8", "role": "input"},
            "A": {"shape": [4, 19], "dtype": "i8", "role": "input"},
        },
        "commands": [
            {"opcode": "RES_PACK", "operands": {"src": "B", "dst": "B_res"}, "attributes": {"layout": "packed_rhs"}},
            {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A", "rhs": "B_res", "dst": "acc"}},
            {
                "opcode": "COMMIT",
                "operands": {"src": "acc", "dst": "Y"},
                "attributes": {"output_dtype": "i32", "epilogue": []},
            },
        ],
    }
    interface = emit_interface_mlir(cb)
    package = tmp_path / "package"
    package.mkdir()
    (package / "manifest.yaml").write_text("target: gemmini\n")
    monkeypatch.setattr("merlin.llvmlower.device_build.objects_buildable", lambda _device: None)
    monkeypatch.setattr(oot_runner, "load_package", lambda _path: SimpleNamespace(target="gemmini"))
    monkeypatch.setattr(
        "merlin.targetgen.corpus_spec.build",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("shape-based capsule regeneration is forbidden")
        ),
    )
    seen = []

    def inspect_package_call(_pkg, _entrypoint, path, **_kwargs):
        seen.append(path.read_text())
        return SimpleNamespace(returncode=1, stderr="synthetic stop after interface handoff", stdout="")

    monkeypatch.setattr(oot_runner, "run_entrypoint", inspect_package_call)
    result = build_device_objects(
        "gemmini",
        {"selected": (4, 8, 19)},
        {"selected": ("i8", "i8", "i32")},
        package_dir=package,
        workdir=tmp_path / "build",
        operand_dtype="int8",
        accum_dtype="i32",
        expected_interfaces={"selected": {"mlir": interface, "sha256": sha256_text(interface)}},
        package_sha256=hash_tree(package)["sha256"],
    )
    assert seen == [interface]
    assert not result.ok and "synthetic stop" in result.skipped[0][1]

    cb["tensors"] = {"A": cb["tensors"]["A"], "B": cb["tensors"]["B"]}
    swapped = emit_interface_mlir(cb)
    with pytest.raises(ValueError, match="pointer ABI"):
        build_device_objects(
            "gemmini",
            {"selected": (4, 8, 19)},
            {"selected": ("i8", "i8", "i32")},
            package_dir=package,
            workdir=tmp_path / "bad",
            operand_dtype="int8",
            accum_dtype="i32",
            expected_interfaces={"selected": {"mlir": swapped, "sha256": sha256_text(swapped)}},
            package_sha256=hash_tree(package)["sha256"],
        )
    assert seen == [interface], "wrong pointer order must be refused before package invocation"


def test_a_batched_signature_builds_the_same_kernel_as_its_unbatched_form(tmp_path):
    """The batch is a loop in the shim over disjoint slices, not a third axis the device sees.
    Building a separate kernel per batch size would mint one per B for identical work."""
    if not _PKG:
        pytest.skip("set MERLIN_TEST_DEVICE_PACKAGE to exercise the build")
    b = build_device_objects(
        "gemmini",
        {"b0": (4, 16, 16, 32)},
        {"b0": ("i8", "i8", "i32")},
        package_dir=_PKG,
        workdir=tmp_path,
        operand_dtype="int8",
        accum_dtype="i32",
        timeout=900,
    )
    if not b.ok:
        pytest.skip(f"device build unavailable here: {b.skipped}")
    assert set(b.kernels) == {"b0"}, "a batched signature still needs exactly one kernel"


def test_a_shape_this_path_cannot_build_is_skipped_with_its_shape(tmp_path):
    """A model whose third extent the path cannot express should still build the other two."""
    b = build_device_objects(
        "gemmini",
        {"b0": (2, 3, 16, 16, 32)},
        {"b0": ("i8", "i8", "i32")},
        package_dir=_PKG or (tmp_path / "nope"),
        workdir=tmp_path,
        operand_dtype="int8",
        accum_dtype="i32",
    )
    assert not b.ok
    assert any("extents" in why or "package" in why for _, why in b.skipped)


# ------------------------------------------------------------------ which devices this path can build


def test_a_device_this_path_cannot_compile_is_declined_with_its_transport():
    """The pipeline runs a package's artifact through mlir-translate and clang, so it works exactly
    for a device whose artifact IS LLVM-dialect MLIR. A command-buffer device emits JSON and a
    self-hosted one emits its own source; handing either to mlir-translate fails obscurely, and the
    shim would then declare an extern kernel nothing in the archive defines.

    This is the first consumer of the transport axis the Link derives -- before it, `endpoint_kind`
    answered four questions at once and none of them was 'can this be compiled here'."""
    from merlin.llvmlower.device_build import boundary_buildable, objects_buildable

    roster = ("gemmini", "radiance", "atlas", "saturn_opu_mxv256d128")
    verdicts = {t: objects_buildable(t) for t in roster}
    declined = {t: why for t, why in verdicts.items() if why}
    if not declined:
        pytest.skip("no non-compilable device resolvable in this checkout")
    for t, why in declined.items():
        assert "reached by" in why and t in why, f"{t}: the decline must name the device and transport"

    # THE BOUNDARY QUESTION IS NOT THE OBJECT QUESTION, and its decline is not phrased like one. A
    # transport with its own emitter answers with the FACT it is missing (an underivable DRAM window),
    # not with "reached by X" -- that reason belongs to this pipeline, which is not the one refusing.
    for t, why in ((t, boundary_buildable(t)) for t in roster):
        if why:
            assert t in why, f"{t}: a decline must name the device it is about"
            assert len(why) > 40, f"{t}: a decline must say what is missing, not just that it is"


def test_the_decline_happens_before_any_work(tmp_path):
    """Named early so the reason is the transport, not a confusing failure three tools later."""
    from merlin.llvmlower.device_build import objects_buildable

    target = next((t for t in ("saturn_opu_mxv256d128", "radiance", "atlas") if objects_buildable(t)), None)
    if target is None:
        pytest.skip("no non-compilable device resolvable here")
    b = build_device_objects(
        target,
        _SIGS,
        _DTS,
        package_dir=_PKG or (tmp_path / "nope"),
        workdir=tmp_path,
        operand_dtype="int8",
        accum_dtype="i32",
    )
    assert not b.ok
    assert any("reached by" in why for _s, why in b.skipped)
    assert not list(tmp_path.glob("*.o")), "nothing should have been compiled before declining"
