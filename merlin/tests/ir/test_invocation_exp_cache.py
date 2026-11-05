"""Cross-loop reuse requires complete environment equality and live storage."""

from __future__ import annotations

import ctypes
import hashlib
import subprocess

import numpy as np
import pytest
import test_adjacent_exp as shared_tests

from merlin.llvmlower import adjacent_exp as A
from merlin.llvmlower import selected_expf_equivalence as E
from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
from merlin.llvmlower.codegen import build_host_shared
from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.llvmlower.toolchain import clang

SOURCE = shared_tests.SOURCE
environment = pytest.fixture(shared_tests.environment.__wrapped__)


def two_loops():
    body = SOURCE.split(" %e=", 1)[1].split(" return %r", 1)[0]
    first = " %e=" + body
    for name in ("out", "e", "r", "a", "b", "d", "v"):
        first = first.replace("%" + name, "%first_" + name)
    return (
        "module {func.func private @cache_environment_gap()\n"
        "func.func @forward(%x:tensor<4x8xf32>,%s:tensor<4x8xf32>,%y:tensor<4x8xf32>)"
        "->(tensor<4x8xf32>,tensor<4x8xf32>) attributes {llvm.emit_c_interface} {"
        + first
        + "func.call @cache_environment_gap():()->()\n %e="
        + body.replace("ins(%x,%s:", "ins(%y,%s:")
        + "return %first_r,%r:tensor<4x8xf32>,tensor<4x8xf32> }}"
    )


@pytest.fixture
def binding(tmp_path, monkeypatch):
    library = tmp_path / "binding-fixture.a"
    library.write_bytes(b"mechanical binding fixture only; not an executable selected libm")
    monkeypatch.setattr(E, "SELECTED_LIBM_SHA256", hashlib.sha256(library.read_bytes()).hexdigest())
    grant = E.select_expf_underflow_equivalence(library)
    with E.selected_expf_underflow_scope(grant):
        yield grant


GAP_RUNTIME = r"""
#include <fenv.h>
static int selected_gap;
void select_gap(int n) { selected_gap = n; }
void cache_environment_gap(void) {
  if (selected_gap == 1) feclearexcept(FE_ALL_EXCEPT);
  if (selected_gap == 2) fesetround(FE_UPWARD);
  if (selected_gap == 3) { feclearexcept(FE_ALL_EXCEPT); errno = 0; }
  if (selected_gap == 4) errno = 17;
}
"""


def build(directory, feature):
    directory.mkdir()
    llvm = lower_to_llvm_ir(two_loops(), workdir=directory, features={feature})
    path = directory / "model.ll"
    path.write_text(llvm)
    library = build_host_shared(path, directory / "model.so")
    model = HostModel.load(str(library), image_policy=PrivateHostImagePolicy(directory))
    return model, model.lib, llvm


def execute(model, x):
    s = np.zeros_like(x)
    outputs = [np.empty_like(x), np.empty_like(x)]
    y = x.copy()
    model([(v.ctypes.data, v.shape) for v in [x, s, y, *outputs]])
    return [v.view("u4").copy() for v in outputs]


@pytest.mark.parametrize("gap", range(5))
def test_complete_environment_changes_invalidate_private_cache(tmp_path, monkeypatch, binding, environment, gap):
    # Inputs stay outside the target-specific class: this executes the generic
    # snapshot/cache mechanism on the host libm, not the selected-callee theorem.
    monkeypatch.setattr(A, "RUNTIME_C", A.RUNTIME_C + GAP_RUNTIME)
    control, control_lib, _ = build(tmp_path / "control", A.ERRNO_FEATURE)
    candidate, candidate_lib, llvm = build(tmp_path / "candidate", A.INVOCATION_FEATURE)
    assert llvm.count("alloca i8, i64 16384,") == 1
    assert "alloca float, i64 1024," not in llvm
    assert "@__merlin_exp_environment_matches(ptr" in llvm
    control_lib.select_gap(gap)
    candidate_lib.select_gap(gap)
    for mode in range(4):
        for dirty_errno in (0, 34, 42):
            for repeat in range(2):
                x = np.tile(np.array([0.125, 0.25, 0.5, 1, 2, 3, 4, -2], dtype="f4"), (4, 1))
                x[:, repeat] = -0.0 if repeat else np.float32(0.125)
                assert environment.mode(mode) == 0
                environment.reset(dirty_errno)
                expected = execute(control, x)
                expected_state = environment.flags(), environment.err()
                assert environment.mode(mode) == 0
                environment.reset(dirty_errno)
                actual = execute(candidate, x)
                actual_state = environment.flags(), environment.err()
                for a, b in zip(actual, expected):
                    np.testing.assert_array_equal(a, b)
                assert actual_state == expected_state


def test_invocation_cache_requires_live_binding(tmp_path):
    with pytest.raises(ValueError, match="explicit live callee grant"):
        lower_to_llvm_ir(two_loops(), workdir=tmp_path, features={A.INVOCATION_FEATURE})


def test_total_storage_budget_and_function_boundaries(tmp_path, binding):
    function = two_loops().partition("\n")[2][:-1]
    text = "module {func.func private @cache_environment_gap()\n" + "\n".join(
        function.replace("@forward", f"@f{i}") for i in range(9)
    ) + "}"
    llvm = lower_to_llvm_ir(text, workdir=tmp_path, features={A.INVOCATION_FEATURE})
    assert llvm.count("alloca i8, i64 16384,") == 8
    assert "alloca float, i64 1024," not in llvm
    assert llvm.count("alloca i8, i64 4096,") == 8
    refused = llvm.split("define void @f8(", 1)[1].split("\n}", 1)[0]
    assert "alloca" not in refused
    assert "@expf(" in refused


def test_nested_regions_keep_original_lowering(tmp_path, binding):
    body = " %e=" + SOURCE.split(" %e=", 1)[1].split(" return %r", 1)[0]
    text = (
        "module {func.func @forward(%x:tensor<4x8xf32>,%s:tensor<4x8xf32>,%condition:i1)"
        "->tensor<4x8xf32> attributes {llvm.emit_c_interface} {"
        "%result=scf.if %condition ->tensor<4x8xf32>{"
        + body
        + "scf.yield %r:tensor<4x8xf32> } else {scf.yield %x:tensor<4x8xf32>}"
        "return %result:tensor<4x8xf32> }}"
    )
    original = lower_to_llvm_ir(text, workdir=tmp_path / "original")
    selected = lower_to_llvm_ir(text, workdir=tmp_path / "selected", features={A.INVOCATION_FEATURE})
    assert selected == original
    assert "@__merlin_exp_environment_capture(" not in selected


@pytest.mark.parametrize("failed_capture", [False, True])
def test_environment_helper_fails_closed_and_preserves_errno(tmp_path, environment, failed_capture):
    from merlin.llvmlower.exp_cache_environment import RUNTIME_C

    source = RUNTIME_C.replace("int status = fegetenv(&current);", "int status = 1;") if failed_capture else RUNTIME_C
    path = tmp_path / "helper.c"
    path.write_text(source)
    library = tmp_path / f"helper-{failed_capture}.so"
    subprocess.run([str(clang()), "-shared", "-fPIC", str(path), "-lm", "-o", str(library)], check=True)
    helper = ctypes.CDLL(str(library))
    capture = helper.__merlin_exp_environment_capture
    matches = helper.__merlin_exp_environment_matches
    for function in (capture, matches):
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ssize_t, ctypes.c_ssize_t, ctypes.c_ssize_t]
    storage = (ctypes.c_ubyte * 4096)(*[0xA5] * 4096)
    for offset, size, stride in [(1, 4096, 1), (0, 0, 1), (0, 4096, 2)]:
        environment.reset(42)
        assert capture(storage, storage, offset, size, stride) == 0
        assert bytes(storage) == bytes([0xA5]) * 4096
        assert environment.err() == 42
    environment.reset(17)
    assert capture(storage, storage, 0, 4096, 1) == (0 if failed_capture else 1)
    assert environment.err() == 17
    assert matches(storage, storage, 0, 4096, 1) == (0 if failed_capture else 1)
    if failed_capture:
        assert bytes(storage) == bytes([0xA5]) * 4096
