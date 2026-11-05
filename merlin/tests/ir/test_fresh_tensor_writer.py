"""Explicit full writers keep live tensor inputs and fresh allocation ownership."""

import subprocess

import numpy as np
import pytest
from xdsl.dialects import tensor
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr
from xdsl.parser import Parser

from merlin.llvmlower.fresh_tensor_writer import FreshTensorWriterContract, rewrite_fresh_tensor_writers
from merlin.xdsl_dialects._common import make_context, text

SOURCE = """module {
func.func @forward(%a:tensor<4xi8>) -> (tensor<4xi8>,tensor<4xi8>) attributes {llvm.emit_c_interface} {
 %scratch=tensor.empty():tensor<7xi32>
 %out=tensor.empty():tensor<4xi8>
 %r=func.call @writer(%a,%scratch,%out):(tensor<4xi8>,tensor<7xi32>,tensor<4xi8>)->tensor<4xi8>
 return %a,%r:tensor<4xi8>,tensor<4xi8>
}
func.func private @writer(tensor<4xi8>,tensor<7xi32>,tensor<4xi8>) -> tensor<4xi8> attributes {llvm.emit_c_interface}
}"""


def fixture(source=SOURCE):
    module = Parser(make_context(tensor.Tensor), source).parse_module()
    module.body.block.last_op.properties["arg_attrs"] = ArrayAttr(
        [DictionaryAttr({"bufferization.access": StringAttr(a)}) for a in ("read", "write", "write")]
    )
    return module


def contract(symbol="writer", borrowed="borrowed", alignment=32):
    return FreshTensorWriterContract(symbol, 2, (1, 2), borrowed, alignment)


@pytest.mark.parametrize(
    "bad",
    [
        contract("absent", "other"),
        contract(borrowed="writer"),
        contract("second", "other", alignment=0),
        contract("second", "other", alignment=3),
        contract("second", "other", alignment=1 << 63),
    ],
)
def test_invalid_later_selection_does_not_partially_mutate(bad):
    module = fixture()
    before = text(module, generic=True)
    with pytest.raises(ValueError):
        rewrite_fresh_tensor_writers(module, [contract(), bad])
    assert text(module, generic=True) == before


def test_borrowed_names_cannot_collide_across_planned_calls():
    module = fixture()
    decl = module.body.block.last_op
    second = decl.clone()
    second.sym_name = StringAttr("second")
    module.body.block.add_op(second)
    # A second call is unnecessary: symbol collision must be diagnosed first.
    before = text(module, generic=True)
    with pytest.raises(ValueError, match="collision"):
        rewrite_fresh_tensor_writers(module, [contract(), contract("second")])
    assert text(module, generic=True) == before


def test_empty_selection_is_unchanged():
    module = fixture()
    before = text(module, generic=True)
    assert rewrite_fresh_tensor_writers(module, []) == []
    assert text(module, generic=True) == before


@pytest.mark.parametrize("optimization", ["-O0", "-O2"])
@pytest.mark.parametrize("declaration_abi", ["ranked_c", "expanded_memref"])
@pytest.mark.parametrize("live_destination", [False, True])
def test_multiple_full_writers_native_default_deallocation(
    tmp_path, optimization, declaration_abi, live_destination, upstream_host_tools
):
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    from merlin.llvmlower.toolchain import clang

    source = SOURCE
    if live_destination:
        source = source.replace("(%a:tensor<4xi8>)", "(%a:tensor<4xi8>,%old:tensor<4xi8>)")
        source = source.replace(" %out=tensor.empty():tensor<4xi8>", "")
        source = source.replace("%out)", "%old)").replace("return %a,%r", "return %old,%r")
    module = fixture(source)
    from dataclasses import replace

    if declaration_abi == "expanded_memref":
        del module.body.block.last_op.attributes["llvm.emit_c_interface"]
    report = rewrite_fresh_tensor_writers(
        module, [replace(contract(), declaration_abi=declaration_abi, allow_initialized_writers=live_destination)]
    )
    assert report[0]["fresh_writers"] == [1, 2]
    assert report[0]["borrowed_symbol"] == "borrowed"
    assert text(module, generic=True).count("alignment = 32") == 2
    llvm = lower_to_llvm_ir(text(module, generic=True), workdir=tmp_path / "lower", vectorize=False)
    assert "call void @free(" in llvm  # Default upstream buffer deallocation remains enabled.
    source = tmp_path / "model.ll"
    source.write_text(llvm)
    adapter = tmp_path / "adapter.c"
    adapter.write_text("""#include <stdint.h>
#include <stdlib.h>
typedef struct {void *allocated,*aligned;int64_t offset,size[1],stride[1];} M;
void _mlir_ciface_borrowed(M*a,M*scratch,M*out) {
 if(a->aligned==scratch->aligned || a->aligned==out->aligned || scratch->aligned==out->aligned) abort();

 for(int i=0;i<7;i++)((int32_t*)scratch->aligned)[i]=i+3;
 for(int i=0;i<4;i++)((int8_t*)out->aligned)[i]=(int8_t)(
   ((int8_t*)a->aligned)[a->offset+i*a->stride[0]]+((int32_t*)scratch->aligned)[i]);
}
""")
    # Pytest may delete passed temp dirs and reuse their names while dlopen
    # retains a library handle. Distinct variants require distinct library paths.
    obj = tmp_path / "model.o"
    lib = tmp_path / f"model_{live_destination}_{declaration_abi}_{optimization}.so"
    subprocess.run([str(clang()), optimization, "-fPIC", "-c", str(source), "-o", str(obj)], check=True)
    subprocess.run(
        [str(clang()), optimization, "-fPIC", "-shared", str(obj), str(adapter), str(mlir_runtime_c()), "-o", str(lib)],
        check=True,
    )
    inp = np.array([-128, -1, 0, 120], dtype=np.int8)
    first, second = np.full(4, 77, dtype=np.int8), np.full(4, 77, dtype=np.int8)
    invoke = HostModel.load(str(lib))
    old = np.array([51, 52, 53, 54], dtype=np.int8)
    for _ in range(20):
        arrays = (inp, old, first, second) if live_destination else (inp, first, second)
        invoke([(v.ctypes.data, v.shape) for v in arrays])
        np.testing.assert_array_equal(old, [51, 52, 53, 54])
        np.testing.assert_array_equal(inp, [-128, -1, 0, 120])
        np.testing.assert_array_equal(first, old if live_destination else inp)
        np.testing.assert_array_equal(second, [-125, 3, 5, 126])


def test_expanded_declaration_requires_explicit_contract():
    module = fixture()
    del module.body.block.last_op.attributes["llvm.emit_c_interface"]
    before = text(module, generic=True)
    with pytest.raises(ValueError, match="ABI contract"):
        rewrite_fresh_tensor_writers(module, [contract()])
    assert text(module, generic=True) == before


def test_explicit_wrapper_name_retargets_tensor_calls_without_abi_collision():
    from dataclasses import replace

    module = fixture()
    del module.body.block.last_op.attributes["llvm.emit_c_interface"]
    rewrite_fresh_tensor_writers(
        module, [replace(contract(), declaration_abi="expanded_memref", wrapper_symbol="tensor_wrapper")]
    )
    declarations = {op.sym_name.data for op in module.body.block.ops}
    assert "writer" not in declarations and "tensor_wrapper" in declarations
    callees = [op.callee.root_reference.data for op in module.walk() if op.name == "func.call"]
    assert callees == ["tensor_wrapper", "borrowed"]


@pytest.mark.parametrize("name", ["forward", "borrowed", ""])
def test_wrapper_collision_refuses_before_mutation(name):
    from dataclasses import replace

    module = fixture()
    before = text(module, generic=True)
    with pytest.raises(ValueError):
        rewrite_fresh_tensor_writers(module, [replace(contract(), wrapper_symbol=name)])
    assert text(module, generic=True) == before
