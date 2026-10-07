"""Normal lowering keeps the target layout independent of optional copy gates."""

import json

import pytest

from merlin.llvmlower import int_softmax_table, pipeline, toolchain

SOURCE = """module {
  func.func @add(%a: f32, %b: f32) -> f32 {
    %c = arith.addf %a, %b : f32
    return %c : f32
  }
}"""
LAYOUT = "e-p:64:64-i64:64-f64:64-n8:16:32:64-S128"


@pytest.mark.parametrize("uniform,contiguous", [(False, False), (True, False), (False, True), (True, True)])
def test_actual_runner_layout_slot_and_independent_copy_gates(tmp_path, uniform, contiguous):
    if not toolchain.m2m_python().is_file():
        pytest.skip("selected upstream lowering toolchain unavailable")
    features = frozenset(
        name
        for name, enabled in [
            ("fold_uniform_fill_copy", uniform),
            ("specialize_contiguous_copy", contiguous),
        ]
        if enabled
    )
    llvm = pipeline.lower_to_llvm_ir(SOURCE, workdir=tmp_path, vectorize=False, data_layout=LAYOUT, features=features)
    record = json.loads((tmp_path / "lowering_recipe.json").read_text())
    # argv[0] is the Python executable; the runner's sys.argv starts at argv[1].
    runner_argv = record["commands"][0]["argv"][1:]
    assert runner_argv[17] == LAYOUT
    assert runner_argv[18:20] == ["1" if uniform else "0", "1" if contiguous else "0"]
    # The integer-softmax gate is appended after the copy gates and stays off unless selected.
    assert len(runner_argv) == int_softmax_table.ARGV_INDEX + 1 and runner_argv[int_softmax_table.ARGV_INDEX] == "0"
    assert f'target datalayout = "{LAYOUT}"' in llvm
    assert (tmp_path / "uniform_fill_copy.json").exists() == uniform
    assert (tmp_path / "contiguous_suffix_copy.json").exists() == contiguous
    assert record["status"] == "returned"


def test_empty_copy_selection_keeps_default_lowering_bytes(tmp_path):
    if not toolchain.m2m_python().is_file():
        pytest.skip("selected upstream lowering toolchain unavailable")
    ordinary = pipeline.lower_to_llvm_ir(SOURCE, workdir=tmp_path / "ordinary", vectorize=False)
    empty = pipeline.lower_to_llvm_ir(SOURCE, workdir=tmp_path / "empty", vectorize=False, features=frozenset())
    assert ordinary == empty
    record = json.loads((tmp_path / "empty/lowering_recipe.json").read_text())
    assert record["commands"][0]["argv"][1:][17:] == ["", "0", "0", "0"]
