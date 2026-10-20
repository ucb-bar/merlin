"""One borrowed invocation preserves every independently owned tensor result.

This exercises the ordinary upstream lowering and native ABI, not hardware
retention, numerical qualification or a horizontal matmul implementation.
"""

import ctypes
import subprocess
from dataclasses import replace

import numpy as np
import pytest
from xdsl.dialects import bufferization, func, memref, tensor
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr
from xdsl.parser import Parser

from merlin.llvmlower.borrowed_buffer_calls import BorrowedBufferEffects
from merlin.llvmlower.fresh_tensor_writer import FreshTensorWriterContract, rewrite_fresh_tensor_writers
from merlin.xdsl_dialects._common import make_context, text

SOURCE = """module {
func.func @forward(%a:tensor<2x3xi8>,%state:tensor<3xi16>) ->
 (tensor<2x3xi8>,tensor<2x3xi32>,tensor<3xi16>,tensor<2x3xi32>) attributes {llvm.emit_c_interface} {
 %out=tensor.empty():tensor<2x3xi32>
 %next=tensor.empty():tensor<3xi16>
 %r:2=func.call @writer(%a,%state,%out,%next):
 (tensor<2x3xi8>,tensor<3xi16>,tensor<2x3xi32>,tensor<3xi16>)->(tensor<3xi16>,tensor<2x3xi32>)
 return %a,%r#1,%r#0,%r#1:tensor<2x3xi8>,tensor<2x3xi32>,tensor<3xi16>,tensor<2x3xi32>
}
func.func private @writer(tensor<2x3xi8>,tensor<3xi16>,tensor<2x3xi32>,tensor<3xi16>)->
 (tensor<3xi16>,tensor<2x3xi32>) attributes {llvm.emit_c_interface}
}"""

BASELINE = """module {
func.func @forward(%a:tensor<2x3xi8>,%state:tensor<3xi16>) ->
 (tensor<2x3xi8>,tensor<2x3xi32>,tensor<3xi16>,tensor<2x3xi32>) attributes {llvm.emit_c_interface} {
 %out=tensor.empty():tensor<2x3xi32>
 %next=tensor.empty():tensor<3xi16>
 %r=func.call @first(%a,%state,%out):(tensor<2x3xi8>,tensor<3xi16>,tensor<2x3xi32>)->tensor<2x3xi32>
 %s=func.call @second(%a,%state,%next):(tensor<2x3xi8>,tensor<3xi16>,tensor<3xi16>)->tensor<3xi16>
 return %a,%r,%s,%r:tensor<2x3xi8>,tensor<2x3xi32>,tensor<3xi16>,tensor<2x3xi32>
}
func.func private @first(tensor<2x3xi8>,tensor<3xi16>,tensor<2x3xi32>)->tensor<2x3xi32>
 attributes {llvm.emit_c_interface}
func.func private @second(tensor<2x3xi8>,tensor<3xi16>,tensor<3xi16>)->tensor<3xi16> attributes {llvm.emit_c_interface}
}"""


def contract(**changes):
    return replace(
        FreshTensorWriterContract(
            "writer",
            3,
            (2, 3),
            "borrowed",
            32,
            borrowed_effects=BorrowedBufferEffects("neutral-plural-transport-premises", True, True, True),
            result_arguments=(3, 2),
        ),
        **changes,
    )


def fixture(source=SOURCE):
    module = Parser(make_context(tensor.Tensor), source).parse_module()
    for declaration in module.body.block.ops:
        if isinstance(declaration, func.FuncOp) and not declaration.body.blocks:
            declaration.properties["arg_attrs"] = ArrayAttr(
                [
                    DictionaryAttr({"bufferization.access": StringAttr("read" if i < 2 else "write")})
                    for i, _ in enumerate(declaration.function_type.inputs)
                ]
            )
    module.verify()
    return module


@pytest.mark.parametrize(
    "indices",
    [(3,), (3, 3), (3, 0), (3, 4), (2, 3), (True, 2), (3, -1), [3, 2], tuple(range(129)), ()],
)
def test_invalid_result_mapping_refuses_without_mutation(indices):
    module = fixture()
    before = text(module, generic=True)
    with pytest.raises(ValueError, match="result"):
        rewrite_fresh_tensor_writers(module, [contract(result_arguments=indices)])
    assert text(module, generic=True) == before


def test_reordered_same_arity_types_refuse_before_mutation():
    module = fixture()
    before = text(module, generic=True)
    with pytest.raises(ValueError, match="identity"):
        rewrite_fresh_tensor_writers(module, [contract(result_argument=2, result_arguments=(2, 3))])
    assert text(module, generic=True) == before


def test_changed_actual_result_type_refuses_before_mutation():
    module = fixture()
    call = next(op for op in module.walk() if isinstance(op, func.CallOp))
    call.results[0]._type = call.results[1].type
    before = text(module, generic=True)
    with pytest.raises(ValueError, match="complete ordered results"):
        rewrite_fresh_tensor_writers(module, [contract()])
    assert text(module, generic=True) == before


def test_one_void_invocation_and_distinct_ordered_output_owners():
    module = fixture()
    reports = rewrite_fresh_tensor_writers(module, [contract()])
    wrapper = next(op for op in module.body.block.ops if op.sym_name.data == "writer")
    operations = tuple(wrapper.body.block.ops)
    calls = [op for op in operations if isinstance(op, func.CallOp)]
    returned = [op for op in operations if isinstance(op, bufferization.ToTensorOp)]
    assert len(calls) == 1 and not calls[0].results
    assert len(returned) == 2
    assert tuple(operations[-1].arguments) == tuple(op.tensor for op in returned)
    assert tuple(op.memref for op in returned) == (calls[0].arguments[3], calls[0].arguments[2])
    owners = [argument.owner for argument in calls[0].arguments[2:]]
    assert all(isinstance(owner, memref.AllocOp) for owner in owners) and owners[0] is not owners[1]
    assert reports[0]["result_arguments"] == [3, 2]
    assert reports[0]["result_ranks"] == [1, 2]


def test_unreturned_writable_tensor_is_not_silently_discarded():
    module = fixture(
        SOURCE.replace("%next):", "%next,%scratch):")
        .replace(" %r:2=", " %scratch=tensor.empty():tensor<5xi8>\n %r:2=")
        .replace("tensor<2x3xi32>,tensor<3xi16>)", "tensor<2x3xi32>,tensor<3xi16>,tensor<5xi8>)")
    )
    before = text(module, generic=True)
    with pytest.raises(ValueError, match="every writable"):
        rewrite_fresh_tensor_writers(module, [contract(fully_written_arguments=(2, 3, 4))])
    assert text(module, generic=True) == before


def test_shared_writable_actual_argument_refuses_before_mutation():
    source = SOURCE.replace("tensor<3xi16>", "tensor<2x3xi32>").replace("%out,%next)", "%out,%out)")
    module = fixture(source)
    before = text(module, generic=True)
    with pytest.raises(ValueError, match="sole-use tensor.empty"):
        rewrite_fresh_tensor_writers(module, [contract()])
    assert text(module, generic=True) == before


def test_singleton_native_effect_checker_cannot_admit_plural_mapping(tmp_path):
    from merlin.llvmlower.borrowed_buffer_calls import from_writers

    writer = contract(borrowed_effects=BorrowedBufferEffects("neutral-plural-transport-premises", True, True, True))
    module = fixture()
    rewrite_fresh_tensor_writers(module, [writer])
    selected = tmp_path / "plural.mlir"
    selected.write_text(text(module, generic=True))
    with pytest.raises(ValueError, match="one fresh ranked result"):
        from_writers(selected, (writer,), (), lambda: None)


def test_singleton_lowered_witness_does_not_claim_plural_result_ownership():
    from test_lowered_fresh_writer_result import CONTRACT, prove
    from test_lowered_fresh_writer_result import fixture as lowered_fixture

    _, wrapper, callsite = lowered_fixture()
    with pytest.raises(ValueError, match="unsupported lowered fresh writer"):
        prove(wrapper, callsite, replace(CONTRACT, result_arguments=(2,)))


def test_plural_effects_are_not_inferred_from_the_signature():
    module = fixture()
    before = text(module, generic=True)
    with pytest.raises(ValueError, match="original borrowed ABI effects"):
        rewrite_fresh_tensor_writers(module, [contract(borrowed_effects=None)])
    assert text(module, generic=True) == before


@pytest.mark.parametrize("declaration_abi", ["ranked_c", "expanded_memref"])
@pytest.mark.parametrize("optimization", ["-O0", "-O2"])
def test_plural_results_native_against_separate_calls_and_reference(
    tmp_path, declaration_abi, optimization, upstream_host_tools
):
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    from merlin.llvmlower.toolchain import clang

    adapter = tmp_path / "adapter.c"
    adapter.write_text("""#include <stdint.h>
#include <stdlib.h>
typedef struct{void*allocated,*aligned;int64_t offset,size[1],stride[1];}M1;
typedef struct{void*allocated,*aligned;int64_t offset,size[2],stride[2];}M2;
static int calls;
int actual_calls(void){return calls;}
static void check(M2*a,M1*s,void*out){
 if(a->size[0]!=2||a->size[1]!=3||s->size[0]!=3||out==a->aligned||out==s->aligned)abort();
}
static void first(M2*a,M1*s,M2*out){
 check(a,s,out->aligned);
 for(int r=0;r<2;r++)for(int c=0;c<3;c++)
 ((int32_t*)out->aligned)[out->offset+r*out->stride[0]+c*out->stride[1]]=
  (int32_t)((int8_t*)a->aligned)[a->offset+r*a->stride[0]+c*a->stride[1]]*7+
  ((int16_t*)s->aligned)[s->offset+c*s->stride[0]];
}
static void second(M2*a,M1*s,M1*out){
 check(a,s,out->aligned);
 for(int c=0;c<3;c++)
 ((uint16_t*)out->aligned)[out->offset+c*out->stride[0]]=(uint16_t)(
  (int32_t)((int16_t*)s->aligned)[s->offset+c*s->stride[0]]+
  ((int8_t*)a->aligned)[a->offset+c*a->stride[1]]-
  ((int8_t*)a->aligned)[a->offset+a->stride[0]+c*a->stride[1]]);
}
void _mlir_ciface_first_borrowed(M2*a,M1*s,M2*out){calls++;first(a,s,out);}
void _mlir_ciface_second_borrowed(M2*a,M1*s,M1*out){calls++;second(a,s,out);}
void _mlir_ciface_borrowed(M2*a,M1*s,M2*out,M1*next){
 calls++;if(out->allocated==next->allocated)abort();first(a,s,out);second(a,s,next);
}
""")
    libraries = []
    for name, source in (("baseline", BASELINE), ("plural", SOURCE)):
        module = fixture(source)
        contracts = (
            [contract(declaration_abi=declaration_abi)]
            if name == "plural"
            else [
                FreshTensorWriterContract(symbol, 2, (2,), symbol + "_borrowed", 32, declaration_abi)
                for symbol in ("first", "second")
            ]
        )
        if declaration_abi == "expanded_memref":
            for operation in module.body.block.ops:
                if isinstance(operation, func.FuncOp) and not operation.body.blocks:
                    del operation.attributes["llvm.emit_c_interface"]
        rewrite_fresh_tensor_writers(module, contracts)
        selected = tmp_path / (name + ".mlir")
        selected.write_text(text(module, generic=True))
        llvm = lower_to_llvm_ir(selected.read_text(), workdir=tmp_path / (name + "-lower"), vectorize=False)
        assert "call void @free(" in llvm
        translated, obj, library = [tmp_path / (name + suffix) for suffix in (".ll", ".o", ".so")]
        translated.write_text(llvm)
        subprocess.run([str(clang()), optimization, "-fPIC", "-c", str(translated), "-o", str(obj)], check=True)
        subprocess.run(
            ["cc", optimization, "-fPIC", "-shared", str(obj), str(adapter), str(mlir_runtime_c()), "-o", str(library)],
            check=True,
        )
        libraries.append(HostModel.load(str(library)))

    original = np.array([[-128, 127, -7], [127, -128, 11]], np.int8)
    initial = np.array([32760, -32760, 17], np.int16)
    states = [initial.copy(), initial.copy()]
    reference = [int(value) for value in initial]
    for iteration in range(3):
        expected = np.array([[int(original[r, c]) * 7 + reference[c] for c in range(3)] for r in range(2)], np.int32)
        reference = [
            ((reference[c] + int(original[0, c]) - int(original[1, c]) + 32768) % 65536) - 32768 for c in range(3)
        ]
        arms = []
        for index, invoke in enumerate(libraries):
            # Every caller buffer has outer canaries, including returned copies
            # of an input and an output observed twice by the original source.
            arrays, backing = [], []
            for value in (
                original,
                states[index],
                np.empty((2, 3), np.int8),
                np.empty((2, 3), np.int32),
                np.empty(3, np.int16),
                np.empty((2, 3), np.int32),
            ):
                owner = np.full(value.size + 4, 53, value.dtype)
                view = owner[2:-2].reshape(value.shape)
                if len(arrays) < 2:
                    view[...] = value
                arrays.append(view)
                backing.append(owner)
            preserved = [array.copy() for array in arrays[:2]]
            invoke([(array.ctypes.data, array.shape) for array in arrays])
            for owner in backing:
                np.testing.assert_array_equal(owner[:2], 53)
                np.testing.assert_array_equal(owner[-2:], 53)
            for value, before in zip(arrays[:2], preserved, strict=True):
                np.testing.assert_array_equal(value, before)
            np.testing.assert_array_equal(arrays[2], original)
            np.testing.assert_array_equal(arrays[3], expected)
            np.testing.assert_array_equal(arrays[4], reference)
            np.testing.assert_array_equal(arrays[5], expected)
            states[index] = arrays[4].copy()
            invoke.lib.actual_calls.restype = ctypes.c_int
            assert invoke.lib.actual_calls() == (iteration + 1) * (2 if index == 0 else 1)
            arms.append([array.copy() for array in arrays[2:]])
        for left, right in zip(*arms, strict=True):
            np.testing.assert_array_equal(left, right)
