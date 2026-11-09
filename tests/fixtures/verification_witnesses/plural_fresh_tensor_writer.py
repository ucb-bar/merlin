import ctypes
import subprocess
import uuid
from dataclasses import replace

import numpy as np
import pytest
from xdsl.dialects import bufferization, func, memref, tensor
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr
from xdsl.parser import Parser

from merlin.llvmlower.borrowed_buffer_calls import BorrowedBufferEffects
from merlin.llvmlower.fresh_tensor_writer import FreshTensorWriterContract, rewrite_fresh_tensor_writers
from merlin.xdsl_dialects._common import make_context, text

SOURCE = "module {\nfunc.func @forward(%a:tensor<2x3xi8>,%state:tensor<3xi16>) ->\n (tensor<2x3xi8>,tensor<2x3xi32>,tensor<3xi16>,tensor<2x3xi32>) attributes {llvm.emit_c_interface} {\n %out=tensor.empty():tensor<2x3xi32>\n %next=tensor.empty():tensor<3xi16>\n %r:2=func.call @writer(%a,%state,%out,%next):\n (tensor<2x3xi8>,tensor<3xi16>,tensor<2x3xi32>,tensor<3xi16>)->(tensor<3xi16>,tensor<2x3xi32>)\n return %a,%r#1,%r#0,%r#1:tensor<2x3xi8>,tensor<2x3xi32>,tensor<3xi16>,tensor<2x3xi32>\n}\nfunc.func private @writer(tensor<2x3xi8>,tensor<3xi16>,tensor<2x3xi32>,tensor<3xi16>)->\n (tensor<3xi16>,tensor<2x3xi32>) attributes {llvm.emit_c_interface}\n}"
BASELINE = "module {\nfunc.func @forward(%a:tensor<2x3xi8>,%state:tensor<3xi16>) ->\n (tensor<2x3xi8>,tensor<2x3xi32>,tensor<3xi16>,tensor<2x3xi32>) attributes {llvm.emit_c_interface} {\n %out=tensor.empty():tensor<2x3xi32>\n %next=tensor.empty():tensor<3xi16>\n %r=func.call @first(%a,%state,%out):(tensor<2x3xi8>,tensor<3xi16>,tensor<2x3xi32>)->tensor<2x3xi32>\n %s=func.call @second(%a,%state,%next):(tensor<2x3xi8>,tensor<3xi16>,tensor<3xi16>)->tensor<3xi16>\n return %a,%r,%s,%r:tensor<2x3xi8>,tensor<2x3xi32>,tensor<3xi16>,tensor<2x3xi32>\n}\nfunc.func private @first(tensor<2x3xi8>,tensor<3xi16>,tensor<2x3xi32>)->tensor<2x3xi32>\n attributes {llvm.emit_c_interface}\nfunc.func private @second(tensor<2x3xi8>,tensor<3xi16>,tensor<3xi16>)->tensor<3xi16> attributes {llvm.emit_c_interface}\n}"


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
        if isinstance(declaration, func.FuncOp) and (not declaration.body.blocks):
            declaration.properties["arg_attrs"] = ArrayAttr(
                [
                    DictionaryAttr({"bufferization.access": StringAttr("read" if i < 2 else "write")})
                    for (i, _) in enumerate(declaration.function_type.inputs)
                ]
            )
    module.verify()
    return module


def run_plural_results_native_against_separate_calls_and_reference(
    tmp_path, declaration_abi, optimization, upstream_host_tools
):
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    from merlin.llvmlower.toolchain import clang

    adapter = tmp_path / "adapter.c"
    adapter.write_text(
        "#include <stdint.h>\n#include <stdlib.h>\ntypedef struct{void*allocated,*aligned;int64_t offset,size[1],stride[1];}M1;\ntypedef struct{void*allocated,*aligned;int64_t offset,size[2],stride[2];}M2;\nstatic int calls;\nint actual_calls(void){return calls;}\nstatic void check(M2*a,M1*s,void*out){\n if(a->size[0]!=2||a->size[1]!=3||s->size[0]!=3||out==a->aligned||out==s->aligned)abort();\n}\nstatic void first(M2*a,M1*s,M2*out){\n check(a,s,out->aligned);\n for(int r=0;r<2;r++)for(int c=0;c<3;c++)\n ((int32_t*)out->aligned)[out->offset+r*out->stride[0]+c*out->stride[1]]=\n  (int32_t)((int8_t*)a->aligned)[a->offset+r*a->stride[0]+c*a->stride[1]]*7+\n  ((int16_t*)s->aligned)[s->offset+c*s->stride[0]];\n}\nstatic void second(M2*a,M1*s,M1*out){\n check(a,s,out->aligned);\n for(int c=0;c<3;c++)\n ((uint16_t*)out->aligned)[out->offset+c*out->stride[0]]=(uint16_t)(\n  (int32_t)((int16_t*)s->aligned)[s->offset+c*s->stride[0]]+\n  ((int8_t*)a->aligned)[a->offset+c*a->stride[1]]-\n  ((int8_t*)a->aligned)[a->offset+a->stride[0]+c*a->stride[1]]);\n}\nvoid _mlir_ciface_first_borrowed(M2*a,M1*s,M2*out){calls++;first(a,s,out);}\nvoid _mlir_ciface_second_borrowed(M2*a,M1*s,M1*out){calls++;second(a,s,out);}\nvoid _mlir_ciface_borrowed(M2*a,M1*s,M2*out,M1*next){\n calls++;if(out->allocated==next->allocated)abort();first(a,s,out);second(a,s,next);\n}\n"
    )
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
                if isinstance(operation, func.FuncOp) and (not operation.body.blocks):
                    del operation.attributes["llvm.emit_c_interface"]
        rewrite_fresh_tensor_writers(module, contracts)
        selected = tmp_path / (name + ".mlir")
        selected.write_text(text(module, generic=True))
        llvm = lower_to_llvm_ir(selected.read_text(), workdir=tmp_path / (name + "-lower"), vectorize=False)
        assert "call void @free(" in llvm
        (translated, obj) = [tmp_path / (name + suffix) for suffix in (".ll", ".o")]
        library = tmp_path / f"{name}-{uuid.uuid4().hex}.so"
        translated.write_text(llvm)
        subprocess.run([str(clang()), optimization, "-fPIC", "-c", str(translated), "-o", str(obj)], check=True)
        subprocess.run(
            [
                str(clang()),
                optimization,
                "-fPIC",
                "-shared",
                str(obj),
                str(adapter),
                str(mlir_runtime_c()),
                "-o",
                str(library),
            ],
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
            (reference[c] + int(original[0, c]) - int(original[1, c]) + 32768) % 65536 - 32768 for c in range(3)
        ]
        arms = []
        for index, invoke in enumerate(libraries):
            (arrays, backing) = ([], [])
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


# Explicit fixture/reference reexports consumed by retained direct tests.
__all__ = [
    "contract",
    "fixture",
    "run_plural_results_native_against_separate_calls_and_reference",
]
