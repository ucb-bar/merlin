"""Stock host O3 is explicit and preserves complete words and floating state."""

import ctypes
import subprocess

import numpy as np
import pytest

from merlin.llvmlower.impr_features import apply_cflags
from merlin.llvmlower.stock_host_optimization import FEATURE, edit_cflags


def test_lazy_feature_only_replaces_one_level_and_preserves_other_flags():
    flags = ["--target=ordinary", "-O2", "-fno-fast-math", "-ffp-contract=off", "-fPIC"]
    assert apply_cflags(flags, frozenset()) == flags
    assert apply_cflags(flags, frozenset({FEATURE})) == [flags[0], "-O3", *flags[2:]]
    assert flags[1] == "-O2"
    assert edit_cflags(edit_cflags(flags)) == edit_cflags(flags)
    assert edit_cflags(["-fPIC"]) == ["-fPIC", "-O3"]


@pytest.mark.parametrize("flags", [["-O2", "-O3"], ["-Ofast"], ["-O4"]])
def test_ambiguous_or_nonordinary_level_refuses(flags):
    with pytest.raises(ValueError):
        edit_cflags(flags)


def test_ordinary_model_plan_uses_selected_o3_without_changing_harness(monkeypatch):
    from merlin.runtime.backends import spike_model

    # Preserve the actual common plan; only the independent tool query is bounded
    # here so the test checks flag propagation without a cross compiler dependency.
    def observe(compiler, flags):
        return dict(schema="merlin.selected-index-lowering.v1", compiler_requested=str(compiler),
                    compiler_resolved="/compiler", compiler_sha256="a" * 64, cross_flags=flags,
                    index_bits=64, data_layout="e-p:64:64")

    monkeypatch.setattr(spike_model.target_data_layout, "observe_index_width", observe)
    control = spike_model.selected_model_compiler_plan(
        backend="scalar", cflags_override=None, vlen=None, features=frozenset())
    candidate = spike_model.selected_model_compiler_plan(
        backend="scalar", cflags_override=None, vlen=None, features=frozenset({FEATURE}))
    assert candidate["gcc_cflags"] == control["gcc_cflags"]
    assert candidate["clang_cflags"] == control["clang_cflags"]
    assert candidate["model_cflags"] == edit_cflags(control["model_cflags"])


def test_complete_native_recurrence_words_flags_modes_and_canaries(tmp_path):
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    from merlin.llvmlower.toolchain import clang

    source = """module {func.func @forward(%a:tensor<3x7xf32>,%b:tensor<7x5xf32>)
      ->tensor<3x5xf32> attributes {llvm.emit_c_interface} {
      %zero=arith.constant 0.0:f32
      %initial=tensor.splat %zero:tensor<3x5xf32>
      %r=linalg.generic {indexing_maps=[affine_map<(i,j,k)->(i,k)>,
        affine_map<(i,j,k)->(k,j)>,affine_map<(i,j,k)->(i,j)>],
        iterator_types=["parallel","parallel","reduction"]}
        ins(%a,%b:tensor<3x7xf32>,tensor<7x5xf32>) outs(%initial:tensor<3x5xf32>){
        ^bb0(%x:f32,%y:f32,%acc:f32): %p=arith.mulf %x,%y:f32
        %sum=arith.addf %p,%acc:f32
        linalg.yield %sum:f32 }->tensor<3x5xf32>
      return %r:tensor<3x5xf32> }}"""
    llvm = lower_to_llvm_ir(source, workdir=tmp_path / "ir")
    ll = tmp_path / "source.ll"
    ll.write_text(llvm)
    helper = tmp_path / "environment.c"
    helper.write_text("""#include <fenv.h>
      int mode(int n){const int modes[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};return fesetround(modes[n]);}
      int flags(void){return fetestexcept(FE_ALL_EXCEPT);}
      int clear(void){return feclearexcept(FE_ALL_EXCEPT);}
    """)
    models = []
    for name, flags in [("control", ["-O2", "-fPIC"]),
                        ("candidate", edit_cflags(["-O2", "-fPIC"]))]:
        obj, so = tmp_path / f"{name}.o", tmp_path / f"{name}.so"
        subprocess.run([str(clang()), *flags, "-c", str(ll), "-o", str(obj)],
                       capture_output=True, check=True)
        subprocess.run([str(clang()), "-shared", "-fPIC", str(obj), str(mlir_runtime_c()),
                        str(helper), "-lm", "-o", str(so)], capture_output=True, check=True)
        models.append((HostModel.load(str(so)), ctypes.CDLL(str(so))))
    values = np.array([0, 0x80000000, 1, 0x80000001, 0x3F800000, 0xBF800000,
                       0x7F7FFFFF, 0x7F800000, 0x7FC01234, 0x7F800001, 0x3F000001,
                       0x00800000, 0x7F000000, 0xFF000000, 0x3F7FFFFF], dtype="u4")
    try:
        for mode in range(4):
            for shift in range(7):
                a = np.resize(np.roll(values, shift), 21).view("f4").reshape(3, 7)
                b = np.resize(np.roll(values, shift + 3), 35).view("f4").reshape(7, 5)
                outputs, flags = [], []
                for model, environment in models:
                    assert environment.mode(mode) == 0
                    store = np.full(31, 0xA5A5A5A5, dtype="u4")
                    out = store[8:23].view("f4").reshape(3, 5)
                    for _ in range(2):
                        environment.clear()
                        model([(x.ctypes.data, x.shape) for x in (a, b, out)])
                        flags.append(environment.flags())
                    outputs.append(out.view("u4").copy())
                    assert np.all(store[:8] == 0xA5A5A5A5) and np.all(store[23:] == 0xA5A5A5A5)
                np.testing.assert_array_equal(outputs[1], outputs[0])
                assert len(set(flags)) == 1
    finally:
        assert models[0][1].mode(0) == 0
