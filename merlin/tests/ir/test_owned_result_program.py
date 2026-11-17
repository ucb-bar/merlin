"""Reopen a real native result program without invoking its compiler."""

import json

import numpy as np
import pytest
from test_owned_result_execution import COMPACT

from merlin.llvmlower import toolchain
from merlin.llvmlower.owned_result_program import load, save
from merlin.llvmlower.owned_results import compile_owned_results

pytestmark = pytest.mark.skipif(not toolchain.available(), reason="native MLIR unavailable")


@pytest.fixture
def program(tmp_path):
    root = tmp_path / "build"
    root.mkdir()
    program = compile_owned_results(COMPACT, workdir=root, timeout=30, allow_existing_workdir=True)
    record = save(program)
    return program, record


def test_loaded_program_executes_changed_data_without_recompilation(program, monkeypatch):
    original, record = program
    import merlin.llvmlower.pipeline

    monkeypatch.setattr(
        merlin.llvmlower.pipeline, "lower_to_llvm_ir", lambda *a, **k: pytest.fail("runtime called compiler")
    )
    restored = load(record)
    for values in ([], [0], [-1, 7, 3, 0], [9, -2, 0, 4, 5]):
        x = np.array(values, dtype=np.int32)
        outputs, observation = restored.execute((x,))
        for out, wanted in zip(outputs, (x[x > 0], x[x > 0], x), strict=True):
            np.testing.assert_array_equal(out, wanted)
        assert observation["source_calls"] == 1
    assert original.library.read_bytes() == restored.library.read_bytes()


@pytest.mark.parametrize("name", ("source.mlir", "model.so", "owned_result_bridge.c"))
def test_changed_product_refuses_before_loading(program, name):
    original, record = program
    path = record.parent / name
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        load(record)


def test_serialized_paths_cannot_redirect_native_products(program):
    original, record = program
    data = json.loads(record.read_text())
    data["artifacts"]["/foreign/model.so"] = data["artifacts"].pop("model.so")
    record.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="member roster"):
        load(record)


def test_nonempty_build_directory_cannot_be_overwritten(tmp_path):
    marker = tmp_path / "source.mlir"
    marker.write_text("incoming work")
    with pytest.raises(ValueError, match="must be empty"):
        compile_owned_results(COMPACT, workdir=tmp_path, allow_existing_workdir=True)
    assert marker.read_text() == "incoming work"
