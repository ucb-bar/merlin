"""The explicit stock transport compiles connected host IR and rejects lost policy."""

from __future__ import annotations

import ctypes
import json

import numpy as np
import pytest

from merlin.llvmlower import pipeline, toolchain


@pytest.fixture
def stock(monkeypatch):
    monkeypatch.setenv("MERLIN_COMPILER_DRIVER", "stock-cli")
    monkeypatch.setenv("MERLIN_FUSION_GUARD", "0")
    # The stock transport explicitly refuses the Python runner's checked
    # deallocation rewrite. Select the plain scalar pipeline for these tests.
    monkeypatch.setenv("MERLIN_SPLIT_FRESH_DEALLOCS", "0")
    monkeypatch.setenv("MERLIN_SLICE_UNUSED_ELEMENTWISE_OUTPUT", "0")
    monkeypatch.delenv("MERLIN_PASSES", raising=False)
    monkeypatch.delenv("MERLIN_SINK_DEALLOCS", raising=False)


@pytest.mark.parametrize("options", [{"vectorize": True}, {"retain_llvm_dialect": True}, {"parallel": True}])
def test_runner_only_options_refuse_before_native_tools(stock, tmp_path, monkeypatch, options):
    from merlin.llvmlower import stock_cli

    monkeypatch.setattr(stock_cli, "llvm_install", lambda: pytest.fail("native tools reached"))
    stale = tmp_path / "lowering_recipe.json"
    stale.write_text('{"status":"returned"}')
    with pytest.raises(pipeline.PipelineError, match="Python-runner options"):
        pipeline.lower_to_llvm_ir("module {}", workdir=tmp_path, features=frozenset(), **options)
    assert not stale.exists()


def test_guarded_fusion_cannot_be_silently_dropped(stock, tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_FUSION_GUARD", "1")
    with pytest.raises(pipeline.PipelineError, match="fusion-guard deselection"):
        pipeline.lower_to_llvm_ir("module {}", workdir=tmp_path, features=frozenset())


def test_unknown_driver_refuses(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_COMPILER_DRIVER", "typo")
    with pytest.raises(pipeline.PipelineError, match="unknown upstream compiler driver"):
        pipeline.lower_to_llvm_ir("module {}", workdir=tmp_path, features=frozenset())


@pytest.mark.parametrize("selected,guard,error", [
    ("func.func(linalg-fuse-elementwise-ops,canonicalize)", "1", "fusion-guard"),
    ("builtin.module(func.func(canonicalize,linalg-fuse-elementwise-ops))", "1", "fusion-guard"),
    ("func.func(canonicalize,optimize-allocation-liveness)", "0", "deallocation placement"),
])
def test_nested_pass_grouping_cannot_bypass_required_checks(stock, tmp_path, monkeypatch, selected, guard, error):
    from merlin.llvmlower import stock_cli

    monkeypatch.setenv("MERLIN_FUSION_GUARD", guard)
    monkeypatch.setattr(stock_cli, "llvm_install", lambda: pytest.fail("native tools reached"))
    with pytest.raises(pipeline.PipelineError, match=error):
        pipeline.lower_to_llvm_ir("module {}", workdir=tmp_path, pipeline=selected, features=frozenset())


def test_expired_final_translation_cannot_record_a_success(stock, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from merlin.llvmlower import stock_cli

    clock = [0.0]
    monkeypatch.setattr(stock_cli.time, "monotonic", lambda: clock[0])
    binary = tmp_path / "bin"
    binary.mkdir()
    for name in ("mlir-opt", "mlir-translate"):
        (binary / name).write_text("selected test tool")
    monkeypatch.setattr(stock_cli, "llvm_install", lambda: tmp_path)
    monkeypatch.setattr(stock_cli, "mlir_translate", lambda: binary / "mlir-translate")

    def finish(command, **kwargs):
        kwargs["outputs"][0].write_text("module {}")
        if kwargs["stage"] == "stock_llvm_translation":
            clock[0] = 2.0
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(stock_cli, "run_in_build_view", finish)
    with pytest.raises(TimeoutError, match="deadline expired"):
        pipeline.lower_to_llvm_ir("module {}", workdir=tmp_path, features=frozenset(), timeout=1)
    receipt = json.loads((tmp_path / "lowering_recipe.json").read_text())
    assert receipt["status"] != "returned"


_SOURCE = """module {
  func.func @forward(%a: tensor<3x5xi32>, %b: tensor<3x5xi32>) -> (tensor<3x5xi32>, tensor<3x5xi32>) {
    %e = tensor.empty() : tensor<3x5xi32>
    %p = linalg.generic {indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>,
      affine_map<(i,j)->(i,j)>], iterator_types = ["parallel", "parallel"]}
      ins(%a, %b : tensor<3x5xi32>, tensor<3x5xi32>) outs(%e : tensor<3x5xi32>) {
      ^bb0(%x: i32, %y: i32, %z: i32):
        %v = arith.addi %x, %y : i32
        linalg.yield %v : i32
    } -> tensor<3x5xi32>
    %e2 = tensor.empty() : tensor<3x5xi32>
    %q = linalg.generic {indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>,
      affine_map<(i,j)->(i,j)>], iterator_types = ["parallel", "parallel"]}
      ins(%p, %b : tensor<3x5xi32>, tensor<3x5xi32>) outs(%e2 : tensor<3x5xi32>) {
      ^bb0(%x: i32, %y: i32, %z: i32):
        %v = arith.muli %x, %y : i32
        linalg.yield %v : i32
    } -> tensor<3x5xi32>
    func.return %p, %q : tensor<3x5xi32>, tensor<3x5xi32>
  }
} """


def test_connected_multioutput_host_compiles_links_and_executes(stock, tmp_path):
    from merlin.llvmlower.abi import descriptor
    from merlin.llvmlower.lower import lower_model

    tools = (toolchain.llvm_install() / "bin" / "mlir-opt", toolchain.mlir_translate(), toolchain.clang())
    if not all(path.is_file() for path in tools):
        pytest.skip("requires selected stock MLIR and clang tools; no simulator")
    built = lower_model(_SOURCE, tmp_path, targets=("host",), features=frozenset())
    image = ctypes.CDLL(str(built.host_so))
    call = image._mlir_ciface_forward
    call.argtypes, call.restype = [ctypes.c_void_p] * 4, None
    a = np.resize(np.array([-(2**31), 2**31 - 1, -1, 0, 1], dtype=np.int32), (3, 5))
    b = np.arange(-7, 8, dtype=np.int32).reshape(3, 5)
    p, q = [np.full((3, 5), 0x12345678, dtype=np.int32) for _ in range(2)]
    args = [descriptor(value.ctypes.data, value.shape) for value in (a, b, p, q)]
    call(*(ctypes.byref(value) for value in args))
    np.testing.assert_array_equal(p, a + b)
    np.testing.assert_array_equal(q, (a + b) * b)
    recipe = json.loads((tmp_path / "lowering_recipe.json").read_text())
    assert recipe["status"] == "returned"
    assert [entry["argv"][0] for entry in recipe["commands"]] == [str(tools[0]), str(tools[1])]


def test_automatic_destination_selection_retains_the_legal_stock_route(stock, tmp_path):
    from merlin.common.paths import repo_root
    from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
    from merlin.llvmlower.lower import lower_model

    tools = (toolchain.llvm_install() / "bin" / "mlir-opt", toolchain.mlir_translate(), toolchain.clang())
    if not all(path.is_file() for path in tools):
        pytest.skip("requires selected stock MLIR and clang tools")
    fixture = repo_root() / "merlin/tests/ir/insert_destination_fixture.mlir"
    source = fixture.read_text().replace("@padded", "@forward")
    built = lower_model(source, tmp_path, targets=("host",), features=frozenset())
    assert built.stats["automatic_host_destinations"]["rewrites"] == 0
    invoke = HostModel.load(str(built.host_so), image_policy=PrivateHostImagePolicy(tmp_path.resolve()))
    values = np.arange(-3, 3, dtype=np.float32).reshape(2, 3)
    out = np.full((4, 5), 99, dtype=np.int8)
    invoke([(array.ctypes.data, array.shape) for array in (values, out)])
    expected = np.zeros((4, 5), dtype=np.int8)
    expected[1:3, 1:4] = values.astype(np.int8)
    np.testing.assert_array_equal(out, expected)
