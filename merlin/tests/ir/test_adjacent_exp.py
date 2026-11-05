"""Adjacent exact-exp reuse preserves complete words, rounding, flags and errno."""

from __future__ import annotations

import ctypes
import re
import subprocess

import numpy as np
import pytest

from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
from merlin.llvmlower.adjacent_exp import BOUNDED_FEATURE, ERRNO_FEATURE, FEATURE, LARGE_FEATURE, UNDERFLOW_FEATURE
from merlin.llvmlower.codegen import build_host_shared
from merlin.llvmlower.pipeline import PipelineError, lower_to_llvm_ir
from merlin.llvmlower.toolchain import clang

SOURCE = """module {func.func @forward(%x:tensor<4x8xf32>,%s:tensor<4x8xf32>)
 ->tensor<4x8xf32> attributes {llvm.emit_c_interface STRICT} {
 %e=tensor.empty():tensor<4x8xf32>
 %r=linalg.generic {indexing_maps=[affine_map<(i,j)->(i,j)>,
 affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]}
 ins(%x,%s:tensor<4x8xf32>,tensor<4x8xf32>) outs(%e:tensor<4x8xf32>){
 ^bb0(%a:f32,%b:f32,%out:f32):
 %d=arith.subf %a,%b:f32
 %v=math.exp %d:f32
 linalg.yield %v:f32 }->tensor<4x8xf32>
 return %r:tensor<4x8xf32> }}"""


def build(directory, selected, *, strict=False, text=SOURCE, feature=FEATURE):
    directory.mkdir()
    llvm = lower_to_llvm_ir(
        text.replace("STRICT", ", strictfp" if strict else ""),
        workdir=directory,
        features={feature} if selected else set(),
    )
    path = directory / "model.ll"
    path.write_text(llvm)
    return HostModel.load(
        str(build_host_shared(path, directory / "model.so")), image_policy=PrivateHostImagePolicy(directory)
    ), llvm


def execute(model, x, s):
    output = np.empty_like(x)
    model([(x.ctypes.data, x.shape), (s.ctypes.data, s.shape), (output.ctypes.data, output.shape)])
    return output


@pytest.fixture
def environment(tmp_path):
    source = tmp_path / "environment.c"
    source.write_text("""#include <fenv.h>
 #include <errno.h>
 int mode(int n){const int modes[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};return fesetround(modes[n]);}
 int flags(void){return fetestexcept(FE_ALL_EXCEPT);}
 int reset(int n){errno=n;return feclearexcept(FE_ALL_EXCEPT);}
 int err(void){return errno;}
 """)
    path = tmp_path / "environment.so"
    subprocess.run(
        [str(clang()), "-shared", "-fPIC", str(source), "-lm", "-o", str(path)], capture_output=True, check=True
    )
    env = ctypes.CDLL(str(path))
    yield env
    assert env.mode(0) == 0


@pytest.mark.parametrize("feature", [FEATURE, BOUNDED_FEATURE, LARGE_FEATURE, ERRNO_FEATURE])
@pytest.mark.parametrize(
    "case", ["finite", "underflow", "overflow", "signed_zero", "signaling_nan", "infinity", "interleaved"]
)
def test_all_values_modes_flags_and_errno(tmp_path, environment, case, feature):
    control, _ = build(tmp_path / "control", False)
    candidate, _ = build(tmp_path / "candidate", True, feature=feature)
    x = np.repeat(np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype="f4"), 4).reshape(4, 8)
    s = np.zeros_like(x)
    if case == "underflow":
        x[:] = np.repeat(np.array([-104.0, -1000.0, -np.finfo("f4").max, -90.0], dtype="f4"), 8).reshape(4, 8)
    if case == "overflow":
        x[:] = np.repeat(np.array([90.0, 1000.0, np.finfo("f4").max, 88.0], dtype="f4"), 8).reshape(4, 8)
    if case == "signed_zero":
        x.view("u4")[:] = np.repeat(np.array([0, 0x80000000, 0, 0x80000000], dtype="u4"), 8).reshape(4, 8)
    if case == "signaling_nan":
        x.view("u4")[:] = np.repeat(np.array([0x7F800001, 0x7FC00123, 0xFF800001, 0xFFC00123], dtype="u4"), 8).reshape(
            4, 8
        )
    if case == "infinity":
        x[:] = np.repeat(np.array([np.inf, -np.inf, np.inf, -np.inf], dtype="f4"), 8).reshape(4, 8)
    if case == "interleaved":
        x[:] = np.array([-2.0, -104.0, -2.0, 90.0, -2.0, 0.0, -2.0, -64.0], dtype="f4")
        x.view("u4")[:, 5] = 0x7F800001
    for mode in range(4):
        assert environment.mode(mode) == 0
        environment.reset(42)
        expected = execute(control, x, s)
        flags, err = environment.flags(), environment.err()
        environment.reset(42)
        actual = execute(candidate, x, s)
        actual_flags, actual_err = environment.flags(), environment.err()
        np.testing.assert_array_equal(actual.view("u4"), expected.view("u4"))
        assert (actual_flags, actual_err) == (flags, err)
    assert environment.mode(0) == 0


@pytest.mark.parametrize("feature", [FEATURE, BOUNDED_FEATURE, LARGE_FEATURE, ERRNO_FEATURE])
def test_strict_context_uses_original_path(tmp_path, feature):
    _, control = build(tmp_path / "control", False, strict=True)
    _, candidate = build(tmp_path / "candidate", True, strict=True, feature=feature)
    assert candidate == control


@pytest.mark.parametrize("feature", [FEATURE, BOUNDED_FEATURE, LARGE_FEATURE, ERRNO_FEATURE])
def test_transposed_in_place_tensor_keeps_upstream_alias_authority(tmp_path, feature):
    text = SOURCE.replace("4x8", "4x4").replace("outs(%e:", "outs(%x:")
    text = text.replace("affine_map<(i,j)->(i,j)>],iterator_types", "affine_map<(i,j)->(j,i)>],iterator_types")
    control, _ = build(tmp_path / "control", False, text=text)
    candidate, _ = build(tmp_path / "candidate", True, text=text, feature=feature)
    x = np.repeat(np.array([-2.0, -1.0, 0.0, 1.0], dtype="f4"), 4).reshape(4, 4)
    s = np.zeros_like(x)
    expected = execute(control, x.copy(), s)
    actual = execute(candidate, x.copy(), s)
    np.testing.assert_array_equal(actual.view("u4"), expected.view("u4"))


@pytest.mark.parametrize("feature", [FEATURE, BOUNDED_FEATURE, LARGE_FEATURE, ERRNO_FEATURE])
def test_unknown_intervening_effect_retains_original_path(tmp_path, feature):
    text = SOURCE.replace("module {", "module {func.func private @observe_environment(f32)->f32\n")
    text = text.replace(
        "%v=math.exp %d:f32", "%q=func.call @observe_environment(%d):(f32)->f32\n %v=math.exp %q:f32"
    ).replace(" STRICT", "")
    control = lower_to_llvm_ir(text, workdir=tmp_path / "control")
    candidate = lower_to_llvm_ir(text, workdir=tmp_path / "candidate", features={feature})
    assert candidate == control


@pytest.mark.parametrize("feature", [BOUNDED_FEATURE, LARGE_FEATURE, ERRNO_FEATURE])
def test_bounded_cache_collision_and_invocation_storage_are_exact(tmp_path, environment, feature):
    control, _ = build(tmp_path / "control", False)
    candidate, _ = build(tmp_path / "candidate", True, feature=feature)
    for mode in range(4):
        assert environment.mode(mode) == 0
        for start in (-64, -32, 0, 32):
            x = np.tile(
                np.array(
                    [start, start + 0.125, start, start + 0.25, start, start + 0.5, start, start + 0.125], dtype="f4"
                ),
                (4, 1),
            )
            s = np.zeros_like(x)
            environment.reset(17)
            expected = execute(control, x, s)
            flags, err = environment.flags(), environment.err()
            environment.reset(17)
            actual = execute(candidate, x, s)
            actual_flags, actual_err = environment.flags(), environment.err()
            np.testing.assert_array_equal(actual.view("u4"), expected.view("u4"))
            assert (actual_flags, actual_err) == (flags, err)
    assert environment.mode(0) == 0


@pytest.mark.parametrize("feature,entries", [(BOUNDED_FEATURE, 64), (LARGE_FEATURE, 128), (ERRNO_FEATURE, 128)])
def test_bounded_cache_total_storage_budget_keeps_adjacent_fallback(tmp_path, feature, entries):
    count = 8192 // entries
    functions = [
        SOURCE.replace("@forward", f"@f{i}").replace(" STRICT", "")[len("module {") : -1] for i in range(count + 1)
    ]
    llvm = lower_to_llvm_ir("module {" + "\n".join(functions) + "}", workdir=tmp_path, features={feature})
    assert llvm.count(f"alloca i32, i64 {entries},") == count * (2 if feature == ERRNO_FEATURE else 1)
    assert llvm.count(f"alloca float, i64 {entries},") == count
    last = llvm.split(f"define void @f{count}(", 1)[1].split("\n}", 1)[0]
    assert "alloca" not in last
    assert "@expf" in last


def test_errno_cache_runtime_symbol_collision_refuses(tmp_path):
    text = SOURCE.replace(" STRICT", "").replace(
        "module {", "module {func.func @__merlin_exp_errno_location()->i32 {%z=arith.constant 0:i32 return %z:i32}\n"
    )
    with pytest.raises(PipelineError, match="exp errno helper symbol collision"):
        lower_to_llvm_ir(text, workdir=tmp_path, features={ERRNO_FEATURE})


def test_errno_cache_exp_call_retains_memory_effects(tmp_path):
    _, llvm = build(tmp_path / "candidate", True, feature=ERRNO_FEATURE)
    declaration = next(
        line for line in llvm.splitlines() if line.startswith("declare float @__merlin_exp_observe_errno(")
    )
    effects = declaration
    group = re.search(r"#(\d+)", declaration)
    if group:
        effects += next(line for line in llvm.splitlines() if line.startswith("attributes #" + group[1] + " ="))
    assert "memory(none)" not in effects
    assert "@expf(" not in llvm


def test_selected_callee_class_requires_live_explicit_grant(tmp_path):
    with pytest.raises(ValueError, match="explicit live callee grant"):
        lower_to_llvm_ir(SOURCE.replace(" STRICT", ""), workdir=tmp_path, features={UNDERFLOW_FEATURE})


def test_selected_callee_class_binds_only_live_facts(tmp_path, monkeypatch):
    import hashlib

    from merlin.llvmlower import selected_expf_equivalence as E
    from merlin.llvmlower.adjacent_exp import bind_runner

    library = tmp_path / "fixture.a"
    library.write_bytes(b"structural compiler binding fixture; not executable libm")
    monkeypatch.setattr(E, "SELECTED_LIBM_SHA256", hashlib.sha256(library.read_bytes()).hexdigest())
    grant = E.select_expf_underflow_equivalence(library)
    source = "_AE_ORIGINAL_RUN_STAGES = _run_stages"
    with E.selected_expf_underflow_scope(grant):
        bound = bind_runner(source, features={UNDERFLOW_FEATURE})
        assert "_AE_SELECTED_EXPF_CLASS = (3268407734, 4286578687)" in bound
        llvm = lower_to_llvm_ir(
            SOURCE.replace(" STRICT", ""), workdir=tmp_path / "compile", features={UNDERFLOW_FEATURE}
        )
        assert "@__merlin_exp_observe_errno" in llvm
    assert bind_runner(source, features={ERRNO_FEATURE}) == source
    with pytest.raises(ValueError, match="explicit live callee grant"):
        bind_runner(source, features={UNDERFLOW_FEATURE})
