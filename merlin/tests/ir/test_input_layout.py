"""Strided input ABI exercises real arithmetic, views, mutation and caller storage."""

import ctypes
import json
import subprocess

import numpy as np
import pytest

from merlin.common.paths import repo_root, runtime_dir
from merlin.llvmlower.c_runtime import generate
from merlin.llvmlower.input_layout import has_overlapping_elements, storage_array
from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.llvmlower.toolchain import clang, llvm_install


def _input(torch, layout):
    values = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4) / 10
    if layout == "padded":
        x = torch.empty(1, 2, 3, 8)[..., ::2]
        x.copy_(values)
        return x
    if layout == "offset":
        x = torch.empty(40).as_strided(values.shape, (24, 12, 4, 1), 7)
        x.copy_(values)
        return x
    if layout == "zero_stride":
        return values[:, :1].expand(1, 2, 3, 4)
    return values.contiguous(memory_format=torch.channels_last)


@pytest.mark.parametrize("layout", ["padded", "offset", "zero_stride", "channels_last"])
@pytest.mark.parametrize("operation", ["elementwise", "reduction", "view", "mutation", "partial_mutation"])
def test_native_strided_input_contract(tmp_path, layout, operation):
    tmp_path = tmp_path / f"{operation}-{layout}"
    tmp_path.mkdir()
    torch = pytest.importorskip("torch")
    m2m = pytest.importorskip("m2m")

    def fn(x):
        if operation == "elementwise":
            return x.sin()
        if operation == "reduction":
            return x.sum(dim=2)
        if operation == "view":
            return x.transpose(1, 3)
        if operation == "partial_mutation":
            return x[:, 0].add_(1)
        return x.fill_(3.25)

    class Model(torch.nn.Module):
        def forward(self, x):
            return fn(x), x

    x = _input(torch, layout)
    logical = x.numpy().copy()
    converted = m2m.convert(Model(), (x,), backend="fx_importer")
    assert converted.ok, converted.diagnostics
    from xdsl.dialects.builtin import UnitAttr
    from xdsl.dialects.func import FuncOp

    for op in converted.module.walk():
        if isinstance(op, FuncOp) and op.sym_name.data == "forward":
            op.attributes["llvm.emit_c_interface"] = UnitAttr()
    source = str(converted.module)
    contract = json.loads(converted.module.attributes["m2m.result_layout"].data)
    assert contract["inputs"][0]["stride"] == list(x.stride())
    assert contract["inputs"][0]["storage_offset"] == x.storage_offset()
    expected, post = Model()(_input(torch, layout))
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.mlir").write_text(source)
    (model / "weights.safetensors.manifest.json").write_text(json.dumps({"0": {"kind": "input", "name": "x"}}))
    np.savez(model / "inputs.npz", in0=logical)
    generated = tmp_path / "cgen"
    generate(model, generated, model / "inputs.npz", dump_all_outputs=True)
    if layout == "zero_stride" and operation in {"mutation", "partial_mutation"}:
        with pytest.raises(RuntimeError, match="input 0 has overlapping logical elements"):
            lower_to_llvm_ir(source, workdir=tmp_path / "lower", preserve_result_descriptors=True)
        return
    lowered = lower_to_llvm_ir(source, workdir=tmp_path / "lower", preserve_result_descriptors=True)
    (tmp_path / "model.ll").write_text(lowered)
    driver = generated / "driver.c"
    driver.write_text("""
#include "model_gen.h"
#include "model_io.h"
void test_call(void) {
  merlin_descriptor_t descriptors[MERLIN_N_ARGS];
  merlin_run_multi(MERLIN_ARGS, MERLIN_N_ARGS, 0, MERLIN_INPUT_PTR, MERLIN_OUTPUT_PTR, descriptors);
}
void *test_input(void) { return MERLIN_INPUT_PTR[0]; }
""")
    library = tmp_path / "model.so"
    libdir = llvm_install() / "lib"
    subprocess.run(
        [
            str(clang()),
            "-shared",
            "-fPIC",
            "-O1",
            "-Wl,-Bsymbolic",
            str(tmp_path / "model.ll"),
            str(generated / "model_call.c"),
            str(driver),
            str(runtime_dir() / "c/merlin_model.c"),
            "-I" + str(generated),
            "-I" + str(runtime_dir() / "c"),
            "-L" + str(libdir),
            "-lmlir_c_runner_utils",
            "-Wl,-rpath," + str(libdir),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
    )
    native = ctypes.CDLL(str(library))
    native.test_input.restype = ctypes.c_void_p
    address = native.test_input()
    initial = storage_array(logical, contract["inputs"][0])
    physical = np.ctypeslib.as_array(ctypes.cast(address, ctypes.POINTER(ctypes.c_float)), shape=initial.shape)
    np.testing.assert_array_equal(physical, initial)
    native.test_call()
    native.merlin_result_descriptor.restype = ctypes.c_void_p
    for ordinal, record in enumerate(contract["results"]):
        rank = len(record["shape"])
        words = (ctypes.c_int64 * (3 + 2 * rank)).from_address(native.merlin_result_descriptor(ordinal))
        assert words[2] == record["storage_offset"]
        assert list(words[3 : 3 + rank]) == record["shape"]
        assert list(words[3 + rank :]) == record["stride"]
        value = post if record["role"] == "user_input_mutation" or ordinal == len(contract["results"]) - 1 else expected
        pointer = ctypes.cast(words[1], ctypes.POINTER(ctypes.c_float))
        actual = np.array(
            [
                pointer[words[2] + sum(i * s for i, s in zip(coord, record["stride"]))]
                for coord in np.ndindex(*record["shape"])
            ],
            dtype=np.float32,
        ).reshape(record["shape"])
        np.testing.assert_allclose(actual, value.numpy(), rtol=1e-6, atol=1e-6)
        if record["alias_inputs"]:
            assert words[0] == address
            # Returned descriptors write through to the physical caller allocation.
            saved = pointer[words[2]]
            pointer[words[2]] = -17
            assert physical[words[2]] == -17
            pointer[words[2]] = saved
        else:
            assert words[0] != address
            ctypes.CDLL(None).free(ctypes.c_void_p(words[0]))
    np.testing.assert_array_equal(physical, storage_array(post.numpy(), contract["inputs"][0]))


def test_overlapping_input_values_refuse():
    with pytest.raises(ValueError, match="overlapping captured storage addresses"):
        storage_array(np.arange(6, dtype=np.float32).reshape(2, 3), dict(shape=[2, 3], stride=[0, 1], storage_offset=0))


@pytest.mark.parametrize(
    "shape,strides,overlap",
    [
        ([2, 3], [0, 1], True),
        ([2, 3], [2, 1], True),
        ([2, 3], [3, 2], False),
        ([1, 3], [0, 1], False),
        ([0, 3], [0, 0], False),
        ([3, 2], [1, 3], False),
    ],
)
def test_input_address_injectivity(shape, strides, overlap):
    assert has_overlapping_elements(dict(shape=shape, stride=strides, storage_offset=7)) is overlap


def test_probe_partial_mutation_capsule_refuses(tmp_path):
    torch = pytest.importorskip("torch", reason="Torch layout checks run under MERLIN_M2M_PYTHON")
    m2m = pytest.importorskip("m2m", reason="Model2MLIR layout checks run under MERLIN_M2M_PYTHON")
    import yaml

    spec = yaml.safe_load(
        (repo_root() / "experiments/reference-data/phase0/compiler-regression-overlap.yaml").read_text()
    )
    call = spec["calls"][0]
    namespace = {"torch": torch}
    exec(call["program"], namespace)
    x = namespace["input_tensor"]()
    expected, post = namespace["Model"]()(namespace["input_tensor"]())
    torch.testing.assert_close(expected, post[:, 0])
    converted = m2m.convert(namespace["Model"](), (x,), backend="fx_importer")
    assert converted.ok, converted.diagnostics
    with pytest.raises(RuntimeError, match="caller-storage write-back requires an injective"):
        lower_to_llvm_ir(str(converted.module), workdir=tmp_path, preserve_result_descriptors=True)


def test_nonzero_overlapping_strides_allow_readonly_borrowed_return(tmp_path):
    torch = pytest.importorskip("torch", reason="Torch layout checks run under MERLIN_M2M_PYTHON")
    m2m = pytest.importorskip("m2m", reason="Model2MLIR layout checks run under MERLIN_M2M_PYTHON")

    class Model(torch.nn.Module):
        def forward(self, x):
            return x.transpose(0, 1)

    x = torch.arange(5, dtype=torch.float32).as_strided((2, 3), (2, 1))
    converted = m2m.convert(Model(), (x,), backend="fx_importer")
    assert converted.ok, converted.diagnostics
    lowered = lower_to_llvm_ir(str(converted.module), workdir=tmp_path, preserve_result_descriptors=True)
    assert "@forward" in lowered


def test_overlapping_borrowed_predecessor_refuses_result_copy():
    from merlin.llvmlower.input_layout import refuse_overlapping_input_aliases

    input_layout = dict(shape=[2, 3], stride=[0, 1], storage_offset=0)
    borrowed = dict(input_layout, alias_inputs=[0])
    contract = dict(inputs=[input_layout], results=[borrowed])
    refuse_overlapping_input_aliases(contract)
    contract["results"].append(dict(input_layout, alias_inputs=[], alias_results=[0]))
    with pytest.raises(ValueError, match="caller-storage write-back requires an injective"):
        refuse_overlapping_input_aliases(contract)


def test_expanded_mutation_growth_owns_storage_for_its_views_and_copies():
    from merlin.llvmlower.input_layout import refuse_overlapping_input_aliases

    original = dict(shape=[2, 3], stride=[0, 1], storage_offset=0)
    grown = dict(shape=[3, 2], stride=[2, 1], storage_offset=0)
    mutation = dict(grown, alias_inputs=[0], role="user_input_mutation", input_index=0)
    view = dict(grown, alias_inputs=[0], alias_results=[0], role="user_output")
    copy = dict(grown, alias_inputs=[], alias_results=[1], role="user_output")
    refuse_overlapping_input_aliases(dict(inputs=[original], results=[mutation, view, copy]))


@pytest.mark.parametrize("overlap", [False, True])
def test_duplicate_mutation_results_refuse_after_growth(overlap):
    from merlin.llvmlower.input_layout import refuse_overlapping_input_aliases

    original = dict(shape=[2, 3], stride=[0, 1] if overlap else [3, 1], storage_offset=0)
    grown = dict(shape=[4, 3], stride=[3, 1], storage_offset=0)
    mutation = dict(grown, alias_inputs=[0], role="user_input_mutation", input_index=0)
    shrink = dict(
        shape=[1, 3], stride=[3, 1], storage_offset=0, alias_inputs=[0], role="user_input_mutation", input_index=0
    )
    with pytest.raises(ValueError, match="duplicate mutation result"):
        refuse_overlapping_input_aliases(dict(inputs=[original], results=[mutation, shrink]))


@pytest.mark.parametrize("shape,stride", [([1, 3], [3, 1]), ([-1, 3], [3, 1]), ([3, 2], [{"shape_axes": [1]}, 1])])
def test_expanded_mutation_without_fresh_storage_proof_still_refuses(shape, stride):
    from merlin.llvmlower.input_layout import refuse_overlapping_input_aliases

    original = dict(shape=[2, 3], stride=[0, 1], storage_offset=0)
    mutation = dict(
        shape=shape, stride=stride, storage_offset=0, alias_inputs=[0], role="user_input_mutation", input_index=0
    )
    with pytest.raises(ValueError, match="caller-storage write-back requires an injective"):
        refuse_overlapping_input_aliases(dict(inputs=[original], results=[mutation]))


def test_later_growth_does_not_hide_an_earlier_caller_write():
    from merlin.llvmlower.input_layout import refuse_overlapping_input_aliases

    original = dict(shape=[2, 3], stride=[0, 1], storage_offset=0)
    borrowed = dict(original, alias_inputs=[0], role="user_output")
    copy = dict(original, alias_inputs=[], alias_results=[0], role="user_output")
    grown = dict(
        shape=[3, 2], stride=[2, 1], storage_offset=0, alias_inputs=[0], role="user_input_mutation", input_index=0
    )
    with pytest.raises(ValueError, match="caller-storage write-back requires an injective"):
        refuse_overlapping_input_aliases(dict(inputs=[original], results=[borrowed, copy, grown]))


@pytest.mark.parametrize("stride", [[3, 1], [6, 2]])
def test_caller_cannot_skip_declared_input_layout(tmp_path, stride):
    from merlin.llvmlower.result_layout import module_attribute

    model = tmp_path / "model"
    model.mkdir()
    contract = dict(inputs=[dict(shape=[2, 3], stride=stride, storage_offset=0)], results=[])
    source = "module" + module_attribute(contract) + " { func.func @forward(%a: tensor<2x3xf32>) { return } }"
    (model / "model.mlir").write_text(source)
    with pytest.raises(ValueError, match="input-layout contract requires dump_all_outputs=True"):
        generate(model, tmp_path / "cgen", tmp_path / "inputs.npz")


@pytest.mark.parametrize("stride,offset", [([-1, 1], 0), ([3, 1], -1), ([{"shape_axes": [1]}, 1], 0)])
def test_unrepresentable_input_layout_refuses(stride, offset):
    with pytest.raises(ValueError, match="static nonnegative layout"):
        storage_array(np.zeros((2, 3), dtype=np.float32), dict(shape=[2, 3], stride=stride, storage_offset=offset))
