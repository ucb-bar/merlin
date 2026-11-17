"""Execution discovers extents; invocation-local ownership validates payloads."""

import os
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
from owned_result_fixture import COMPACT, RESIZE

from merlin.llvmlower import toolchain
from merlin.llvmlower.abi import ScalarArg
from merlin.llvmlower.owned_result_abi import TensorSlot, pipeline, prepare
from merlin.llvmlower.owned_results import _view, compile_owned_results

pytestmark = pytest.mark.skipif(not toolchain.available(), reason="native MLIR unavailable")


@pytest.fixture(scope="module")
def compact(tmp_path_factory):
    return compile_owned_results(COMPACT, workdir=tmp_path_factory.mktemp("compact") / "build", timeout=30)


@pytest.mark.parametrize("words", ([], [0], [-1], [7], [0, -1, 4, 0, 9, -3], [1, 2, 3, 4], [-(1 << 31), (1 << 31) - 1]))
def test_original_computation_discovers_all_result_extents(compact, words):
    x = np.array(words, dtype=np.int32)
    snapshot = x.copy()
    outputs, report = compact.execute((x,))
    for actual, expected in zip(outputs, (x[x > 0], x[x > 0], x), strict=True):
        np.testing.assert_array_equal(actual, expected)
        assert not np.shares_memory(actual, x)
    assert not np.shares_memory(outputs[0], outputs[1])
    assert report["source_calls"] == 1
    assert report["owned_allocations"] in (2, 3)
    assert 0 < report["peak_allocation_bytes"] <= compact.max_bytes
    np.testing.assert_array_equal(x, snapshot)


def test_runtime_resizing_preserves_zero_trip_empty_and_repeated_invocations(tmp_path):
    program = compile_owned_results(RESIZE, workdir=tmp_path / "build", timeout=30)
    for words in ([], [7], [-(1 << 31), (1 << 31)-1, 4, -9, 0, 11, -1]):
        x = np.asarray(words, np.int32)
        before = x.tobytes()
        for steps in (0, 1, 2, 7, 19):
            (actual,), report = program.execute((x, ScalarArg(steps, "i32")))
            wanted = x[:max(0, x.size-steps)]
            assert (actual.shape, actual.dtype, actual.tobytes()) == (wanted.shape, wanted.dtype, wanted.tobytes())
            assert x.tobytes() == before and not np.shares_memory(actual, x)
            assert report["source_calls"] == 1 and report["owned_allocations"] == 1


def test_rank_zero_and_native_scalar_arguments_preserve_exact_words(tmp_path):
    source = """module {func.func @forward(%x:tensor<f64>,%flag:i1,%word:i64)
      -> (tensor<f64>,tensor<i64>) {
      %selected=arith.select %flag,%word,%word:i64
      %result=tensor.from_elements %selected:tensor<i64>
      return %x,%result:tensor<f64>,tensor<i64>
    }}"""
    program = compile_owned_results(source, workdir=tmp_path / "build", timeout=30)
    for word in (0, 1 << 60, -(1 << 63), (1 << 63) - 1):
        x = np.array(0xFFF0000000000001, dtype=np.uint64).view(np.float64)
        outputs, _ = program.execute((x, ScalarArg(True, "i1"), ScalarArg(word, "i64")))
        assert outputs[0].shape == ()
        assert outputs[0].tobytes() == x.tobytes()
        assert outputs[1].item() == word


def test_mixed_result_ranks_and_zero_argument_entry(tmp_path):
    source = """module {func.func @forward() -> (tensor<2x3xi32>,tensor<i32>) {
      %value=arith.constant 19:i32
      %empty=tensor.empty():tensor<2x3xi32>
      %matrix=linalg.fill ins(%value:i32) outs(%empty:tensor<2x3xi32>)->tensor<2x3xi32>
      %scalar=tensor.from_elements %value:tensor<i32>
      return %matrix,%scalar:tensor<2x3xi32>,tensor<i32>
    }}"""
    program = compile_owned_results(source, workdir=tmp_path / "build", timeout=30)
    outputs, _ = program.execute(())
    np.testing.assert_array_equal(outputs[0], np.full((2, 3), 19, dtype=np.int32))
    assert outputs[1].shape == () and outputs[1].item() == 19


def test_source_location_and_provenance_survive_result_transport(tmp_path):
    from merlin.frontends.linalg_mlir import parse_mlir_text

    source = COMPACT.replace(
        "%empty=tensor.empty(%n):tensor<?xi32>",
        '%empty="tensor.empty"(%n) {prov.op="compact",prov.region_id="selected"}'
        ' : (index)->tensor<?xi32> loc("compact allocation")',
    )
    program = compile_owned_results(source, workdir=tmp_path / "build", timeout=30)
    prepared = parse_mlir_text((program.source.parent / "prepared.mlir").read_text())
    allocation = next(op for op in prepared.walk() if op.name == "tensor.empty")
    assert allocation.attributes["prov.op"].data == "compact"
    assert allocation.attributes["prov.region_id"].data == "selected"
    assert "compact allocation" in str(allocation.location)
    outputs, _ = program.execute((np.array([0, 7, -2, 9], dtype=np.int32),))
    np.testing.assert_array_equal(outputs[0], [7, 9])


@pytest.fixture(scope="module")
def allocation_program(tmp_path_factory):
    source = """module {func.func @forward(%n:i64) -> tensor<?xi32> {
      %size=arith.index_cast %n:i64 to index
      %value=arith.constant 7:i32
      %empty=tensor.empty(%size):tensor<?xi32>
      %out=linalg.fill ins(%value:i32) outs(%empty:tensor<?xi32>)->tensor<?xi32>
      return %out:tensor<?xi32>
    }}"""
    return compile_owned_results(source, workdir=tmp_path_factory.mktemp("allocation-products") / "build", timeout=30)


@pytest.mark.parametrize(
    "size,message",
    (
        (-1, "negative dimension"),
        (1 << 62, "byte product exceeds budget"),
        ((1 << 63) - 1, "byte product exceeds budget"),
    ),
)
def test_allocation_products_refuse_before_native_byte_arithmetic(allocation_program, tmp_path, size, message):
    driver = tmp_path / "size.py"
    driver.write_text("""
import ctypes,resource,sys
from merlin.llvmlower.abi import make_descriptor
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
lib=ctypes.CDLL(sys.argv[1])
libc=ctypes.CDLL(None)
libc.setbuf.argtypes=[ctypes.c_void_p,ctypes.c_void_p]
libc.setbuf(ctypes.c_void_p.in_dll(libc,"stdout"),None)
lib.merlin_owned_results_begin.argtypes=[ctypes.c_size_t,ctypes.c_size_t]
assert lib.merlin_owned_results_begin(64<<20,4096)
n=ctypes.c_int64(int(sys.argv[2]))
result=make_descriptor(1)()
args=(ctypes.c_void_p*1)(ctypes.addressof(n))
outs=(ctypes.c_void_p*1)(ctypes.addressof(result))
lib.merlin_owned_result_invoke(args,outs)
raise AssertionError('invalid allocation product was executed')
""")
    done = subprocess.run(
        [sys.executable, str(driver), str(allocation_program.library), str(size)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert done.returncode < 0
    assert message in done.stdout + done.stderr


def test_checked_allocation_products_preserve_valid_and_empty_invocations(allocation_program):
    for n in (0, 1, 7, 31):
        (out,), _ = allocation_program.execute((ScalarArg(n, "i64"),))
        np.testing.assert_array_equal(out, np.full(n, 7, dtype=np.int32))


def test_allocation_guard_is_consumed_once_across_destination_planning_stages(tmp_path):
    program = compile_owned_results(
        COMPACT, workdir=tmp_path / "build", timeout=30, features=frozenset({"reuse_loop_tensor_destinations"})
    )
    x = np.array([0, -1, 7, 3], dtype=np.int32)
    outputs, report = program.execute((x,))
    for out, expected in zip(outputs, (x[x > 0], x[x > 0], x), strict=True):
        np.testing.assert_array_equal(out, expected)
    assert report["source_calls"] == 1
    assert "owned allocation: byte product exceeds budget" in (program.source.parent / "model.ll").read_text()


def test_repeated_calls_keep_copied_results_alive_and_release_native_owners(compact):
    saved = []
    for i in range(12):
        x = np.arange(i, dtype=np.int32) - 3
        outputs, _ = compact.execute((x,))
        saved.append((outputs, x.copy()))
    for outputs, x in saved:
        for actual, expected in zip(outputs, (x[x > 0], x[x > 0], x), strict=True):
            np.testing.assert_array_equal(actual, expected)


def test_signature_storage_and_output_budgets_cannot_be_replaced(compact):
    with pytest.raises(ValueError, match="storage/rank"):
        compact.execute((np.array([1], dtype=np.int64),))
    with pytest.raises(ValueError, match="input roster"):
        compact.execute(())
    with pytest.raises(ValueError, match="output byte budget"):
        replace(compact, max_output_bytes=3).execute((np.array([1, 2], dtype=np.int32),))
    # The failed observation still releases the invocation. Another call works.
    outputs, _ = compact.execute((np.array([3], dtype=np.int32),))
    np.testing.assert_array_equal(outputs[0], [3])
    with pytest.raises(ValueError, match="signature changed"):
        replace(compact, outputs=(TensorSlot("i32", (999,)),)).execute((np.array([1], dtype=np.int32),))


def test_changed_loaded_library_refuses_before_native_call(compact, tmp_path):
    # A foreign image must not borrow the source/build correspondence.
    image = tmp_path / "foreign.so"
    image.write_bytes(compact.library.read_bytes())
    with pytest.raises(ValueError, match="loaded image differs"):
        replace(compact, library=image).execute((np.array([1], dtype=np.int32),))


def test_allocation_budget_fails_in_the_actual_native_worker(compact, tmp_path):
    driver = tmp_path / "budget.py"
    driver.write_text(
        """
import ctypes, resource, sys
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
lib=ctypes.CDLL(sys.argv[1])
lib.merlin_owned_results_begin.argtypes=[ctypes.c_size_t,ctypes.c_size_t]
assert lib.merlin_owned_results_begin(1,1)
"""
        + """
from merlin.llvmlower.abi import descriptor, make_descriptor
import numpy as np
x=np.array([1],dtype=np.int32)
d=descriptor(x.ctypes.data,x.shape)
results=[make_descriptor(1)() for _ in range(3)]
args=(ctypes.c_void_p*1)(ctypes.addressof(d))
outs=(ctypes.c_void_p*3)(*(ctypes.addressof(r) for r in results))
lib.merlin_owned_result_invoke(args,outs)
raise AssertionError('allocation budget was bypassed')
"""
    )
    result = subprocess.run(
        [sys.executable, str(driver), str(compact.library)],
        capture_output=True,
        text=True,
        timeout=10,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"),
    )
    assert result.returncode < 0
    assert "invocation allocation budget exceeded" in result.stderr


@pytest.mark.parametrize("mutation", ("owner", "aligned", "offset", "negative", "oversize", "stride"))
def test_invalid_executed_descriptors_refuse_before_payload_reads(mutation):
    import ctypes

    from merlin.llvmlower.abi import descriptor

    data = (ctypes.c_int32 * 4)(1, 2, 3, 4)
    base = ctypes.addressof(data)
    result = descriptor(base, (4,))
    if mutation == "owner":
        result.allocated = base + 1
    elif mutation == "aligned":
        result.aligned = base + 64
    elif mutation == "offset":
        result.offset = -1
    elif mutation == "negative":
        result.sizes[0] = -1
    elif mutation == "oversize":
        result.sizes[0] = 5
    else:
        result.strides[0] = 2
    with pytest.raises(ValueError, match="extent|allocation|footprint"):
        _view(result, TensorSlot("i32", (-1,)), lambda owner: 16 if owner == base else 0)


def test_signed_and_broadcast_strides_use_the_complete_owned_footprint():
    import ctypes

    from merlin.llvmlower.abi import descriptor

    data = (ctypes.c_int32 * 4)(1, 2, 3, 4)
    base = ctypes.addressof(data)
    result = descriptor(base, (4,))
    result.offset, result.strides[0] = 3, -1
    actual, owner, count = _view(result, TensorSlot("i32", (-1,)), lambda p: 16 if p == base else 0)
    np.testing.assert_array_equal(actual, [4, 3, 2, 1])
    assert owner == base and count == 16
    result.offset, result.strides[0] = 2, 0
    actual, _, _ = _view(result, TensorSlot("i32", (-1,)), lambda p: 16 if p == base else 0)
    np.testing.assert_array_equal(actual, [3, 3, 3, 3])
    result.sizes[0], result.offset, result.aligned = 0, -(1 << 63), None
    actual, _, _ = _view(result, TensorSlot("i32", (-1,)), lambda p: 16 if p == base else 0)
    assert actual.shape == (0,)


def test_source_writable_inputs_and_runtime_symbol_collisions_refuse():
    for source in (
        COMPACT.replace("writable=false", "writable=true"),
        COMPACT.replace("module {", "module {func.func private @merlin_owned_results_end()"),
    ):
        with pytest.raises(ValueError, match="immutable|collides"):
            prepare(source)


def test_selected_pipeline_cannot_disable_ownership(monkeypatch):
    monkeypatch.setenv("MERLIN_NO_DEALLOC", "1")
    with pytest.raises(ValueError, match="ownership/deallocation"):
        pipeline()
