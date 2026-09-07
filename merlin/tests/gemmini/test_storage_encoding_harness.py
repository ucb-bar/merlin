"""Explicit caller formats, without full-model compilation or execution."""
from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from merlin.perf.storage_encoding import GroupedAxesStorage
from merlin.runtime.backends import base as bk
from merlin.runtime.storage_binding import resolve_storage_bindings, StoragePrepackRequired
from merlin.runtime.tensor import Tensor

gem = bk.get_backend("gemmini")
CodegenError = gem.gemmini_codegen.CodegenError


def fixture():
    encodings = {
        "A": GroupedAxesStorage((2, 3), "i8", ((0,), (1,)), (2, 3), (5, 1), 10),
        "scale": GroupedAxesStorage((), "f32", ((),), (1,), (1,), 1),
        "Y": GroupedAxesStorage((2, 3), "i16", ((1,), (0,)), (3, 2), (4, 1), 12),
    }
    cb = {"abi_version": "0.1", "target": "gemmini", "commands": [],
          "tensors": {name: {"shape": list(enc.physical_shape), "dtype": enc.dtype,
                             "role": "output" if name == "Y" else "input"}
                      for name, enc in encodings.items()},
          "params": {"storage_encodings": {name: enc.to_dict() for name, enc in encodings.items()}},
          "kernel_abi": {"kind": "whole_program", "args": [
              {"tensor": name, "access": "write" if name == "Y" else "read"} for name in encodings],
              "outputs": ["Y"]}}
    return cb, {"A": [[-3, -2, -1], [0, 1, 2]], "scale": -0.0}


def initializer(source, name):
    return [int(value) for value in re.search(rf"T_{name}\[\d+\].*?= \{{([^}}]*)\}};", source).group(1).split(",")]


def test_reference_transpose_pack_and_production_prepack_refusal():
    cb, inputs = fixture()
    encoding = GroupedAxesStorage((2, 3), "i8", ((1,), (0,)), (3, 2), (4, 1), 12)
    cb["params"]["storage_encodings"]["A"] = encoding.to_dict()
    cb["tensors"]["A"].update(shape=[3, 2], role="weight")
    reference = resolve_storage_bindings(cb, inputs, max_storage_bytes=100,
                                        reference_only_allow_prepack=True)
    assert reference["A"].pack_words([-3, -2, -1, 0, 1, 2]) == [-3, 0, 0, 0, -2, 1, 0, 0, -1, 2, 0, 0]
    assert reference["A"].setup_evidence()["prepack_authorization"] == "UNPROVEN"
    with pytest.raises(CodegenError, match="requires_prepack") as caught:
        gem.render_harness(cb, target="gemmini", inputs=inputs)
    assert caught.value.storage_obligations["A"]["requires_prepack"] is True


def test_stride_only_transpose_cannot_hide_prepacking():
    cb, inputs = fixture()
    cb["params"]["storage_encodings"]["A"] = GroupedAxesStorage(
        (2, 3), "i8", ((0,), (1,)), (2, 3), (1, 2), 6).to_dict()
    with pytest.raises(StoragePrepackRequired):
        resolve_storage_bindings(cb, inputs, max_storage_bytes=100)


def test_unit_axis_reordering_is_not_a_nontrivial_transpose():
    cb, inputs = fixture()
    cb["params"]["storage_encodings"]["A"] = GroupedAxesStorage(
        (1, 3), "i8", ((1,), (0,)), (3, 1), (4, 1), 12).to_dict()
    cb["tensors"]["A"]["shape"] = [3, 1]
    inputs["A"] = [[1, 2, 3]]
    binding = resolve_storage_bindings(cb, inputs, max_storage_bytes=100)["A"]
    assert binding.requires_prepack is False
    assert binding.pack_words([1, 2, 3]) == [1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]


def test_input_storage_offset_and_nontrivial_float_words_are_preserved():
    cb, inputs = fixture()
    cb["params"]["storage_encodings"]["scale"] = GroupedAxesStorage(
        (), "f32", ((),), (1,), (1,), 3, 1).to_dict()
    binding = resolve_storage_bindings(cb, inputs, max_storage_bytes=100)["scale"]
    assert binding.pack_words([0x7FC12345]) == [0, 0x7FC12345, 0]
    assert initializer(gem.render_harness(cb, target="gemmini", inputs=inputs), "scale") == [0, 0x80000000, 0]


def test_scalar_and_padding_pack_preserve_words_and_unchanged_roi(monkeypatch):
    monkeypatch.setenv("MERLIN_CACHE_STATE", "warm")
    cb, inputs = fixture()
    source = gem.render_harness(cb, target="gemmini", inputs=inputs)
    assert initializer(source, "A") == [-3, -2, -1, 0, 0, 0, 1, 2, 0, 0]
    assert initializer(source, "scale") == [0x80000000]
    assert "T_Y[12]" in source
    measured = source.split("uint64_t c0 = read_cycles();", 1)[1].split("uint64_t c1 = read_cycles();", 1)[0]
    assert measured.strip() == "gemmini_kernel((void*)T_A, (void*)T_scale, (void*)T_Y);\n  gemmini_fence();"
    assert source.count("gemmini_kernel((void*)T_A, (void*)T_scale, (void*)T_Y);") == 2


@pytest.mark.parametrize("change", ["missing", "empty", "extra", "dtype", "shape", "bool_shape", "metadata",
                                    "input_shape", "typed_dtype", "typed_shape", "recipe", "budget"])
def test_contradictory_storage_refuses_before_materializing(tmp_path, change):
    cb, inputs = fixture()
    if change == "missing": del cb["params"]["storage_encodings"]["scale"]
    elif change == "empty": cb["params"]["storage_encodings"] = {}
    elif change == "extra": cb["params"]["storage_encodings"]["extra"] = cb["params"]["storage_encodings"]["A"]
    elif change == "dtype": cb["tensors"]["A"]["dtype"] = "i32"
    elif change == "shape": cb["tensors"]["A"]["shape"] = [3, 2]
    elif change == "bool_shape": cb["tensors"]["scale"]["shape"] = [True]
    elif change == "metadata": del cb["params"]["storage_encodings"]["A"]["offset_elements"]
    elif change == "input_shape": inputs["A"] = [[-3, -2], [-1, 0], [1, 2]]
    elif change == "typed_dtype": inputs["A"] = Tensor((2, 3), list(range(6)), "i16")
    elif change == "typed_shape": inputs["A"] = Tensor((3, 2), list(range(6)), "i8")
    elif change == "recipe": cb["params"]["im2col_recipes"] = [{"source": "A", "target": "Y"}]
    with pytest.raises(ValueError):
        resolve_storage_bindings(cb, inputs, max_storage_bytes=1 if change == "budget" else 100)


def test_tiny_native_caller_reads_output_in_declared_logical_order(tmp_path, monkeypatch):
    compiler = shutil.which("cc")
    if not compiler:
        pytest.skip("native C compiler unavailable")
    monkeypatch.setenv("MERLIN_CACHE_STATE", "warm")
    cb, inputs = fixture()
    source = gem.render_harness(cb, target="gemmini", inputs=inputs)
    include = tmp_path / "include"
    include.mkdir()
    (include / "gemmini_testutils.h").write_text("#include <stdint.h>\ntypedef int8_t elem_t;\n#define row_align(x)\n#define row_align_acc(x)\nstatic uint64_t read_cycles(void){return 0;}\nstatic void gemmini_fence(void){}\n")
    (tmp_path / "caller.c").write_text(source + '''
void gemmini_kernel(void *a, void *scale, void *out) {
  const int8_t expected[10] = {-3,-2,-1,0,0,0,1,2,0,0};
  for (int k=0;k<10;k++) if (((int8_t*)a)[k]!=expected[k]) __builtin_trap();
  if (((uint32_t*)scale)[0] != 0x80000000u) __builtin_trap();
  for (int r=0;r<2;r++) for (int c=0;c<3;c++) ((int16_t*)out)[c*4+r]=((int8_t*)a)[r*5+c];
}
''')
    built = subprocess.run([compiler, "-std=c11", "-I", str(tmp_path), str(tmp_path / "caller.c"), "-o", str(tmp_path / "caller")],
                           capture_output=True, text=True, timeout=10)
    assert built.returncode == 0, built.stderr
    run = subprocess.run([str(tmp_path / "caller")], capture_output=True, text=True, timeout=5)
    assert run.returncode == 0, run.stderr
    assert "OUT Y 2 3 -3 -2 -1 0 1 2" in run.stdout


def test_rank_zero_output_uses_exact_offset():
    cb, inputs = fixture()
    cb["params"]["storage_encodings"]["Y"] = GroupedAxesStorage(
        (), "i16", ((),), (1,), (1,), 3, 2).to_dict()
    cb["tensors"]["Y"]["shape"] = [1]
    source = gem.render_harness(cb, target="gemmini", inputs=inputs)
    assert 'printf("OUT Y 1 1")' in source and "T_Y[2]" in source


def test_absent_storage_contract_selects_legacy_path():
    cb, _ = fixture()
    del cb["params"]["storage_encodings"]
    assert resolve_storage_bindings(cb, max_storage_bytes=1) is None
