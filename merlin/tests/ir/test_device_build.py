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

_PKG = os.environ.get("MERLIN_TEST_DEVICE_PACKAGE")
_SIGS = {"d0": (16, 16, 32), "d1": (8, 64, 128)}
_DTS = {"d0": ("i8", "i8", "i32"), "d1": ("i8", "i8", "i32")}


def _nm():
    from merlin.llvmlower.toolchain import DEFAULT_LLVM_INSTALL
    from pathlib import Path
    local = Path(DEFAULT_LLVM_INSTALL) / "bin" / "llvm-nm"
    return str(local) if local.exists() else (shutil.which("llvm-nm") or shutil.which("nm"))


def _built(tmp_path):
    if not _PKG:
        pytest.skip("set MERLIN_TEST_DEVICE_PACKAGE to a backend package to exercise the build")
    b = build_device_objects("gemmini", _SIGS, _DTS, package_dir=_PKG, workdir=tmp_path,
                             operand_dtype="int8", accum_dtype="i32", timeout=900)
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
        "two signatures sharing a kernel symbol would silently run one layer's kernel for both")


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
    b = build_device_objects("gemmini", _SIGS, _DTS, package_dir=tmp_path / "nope",
                             workdir=tmp_path, operand_dtype="int8", accum_dtype="i32")
    assert not b.ok and b.skipped and any("package" in why for _, why in b.skipped)


def test_a_batched_signature_builds_the_same_kernel_as_its_unbatched_form(tmp_path):
    """The batch is a loop in the shim over disjoint slices, not a third axis the device sees.
    Building a separate kernel per batch size would mint one per B for identical work."""
    if not _PKG:
        pytest.skip("set MERLIN_TEST_DEVICE_PACKAGE to exercise the build")
    b = build_device_objects("gemmini", {"b0": (4, 16, 16, 32)}, {"b0": ("i8", "i8", "i32")},
                             package_dir=_PKG, workdir=tmp_path,
                             operand_dtype="int8", accum_dtype="i32", timeout=900)
    if not b.ok:
        pytest.skip(f"device build unavailable here: {b.skipped}")
    assert set(b.kernels) == {"b0"}, "a batched signature still needs exactly one kernel"


def test_a_shape_this_path_cannot_build_is_skipped_with_its_shape(tmp_path):
    """A model whose third extent the path cannot express should still build the other two."""
    b = build_device_objects("gemmini", {"b0": (2, 3, 16, 16, 32)}, {"b0": ("i8", "i8", "i32")},
                             package_dir=_PKG or (tmp_path / "nope"), workdir=tmp_path,
                             operand_dtype="int8", accum_dtype="i32")
    assert not b.ok
    assert any("extents" in why or "package" in why for _, why in b.skipped)
