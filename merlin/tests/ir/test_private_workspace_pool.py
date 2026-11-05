"""Storage commoning uses explicit noescape/completion, never cached contents."""

import subprocess
from dataclasses import replace

import numpy as np
import pytest
from test_fresh_tensor_writer import contract
from test_private_writer_workspace import workspace
from xdsl.dialects import func, memref, tensor
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr, SymbolRefAttr
from xdsl.parser import Parser

from merlin.llvmlower.fresh_tensor_writer import rewrite_fresh_tensor_writers
from merlin.llvmlower.private_workspace_pool import pool_private_writer_workspaces
from merlin.xdsl_dialects._common import make_context, text

SOURCE = """module {
 func.func @forward(%a:tensor<4xi8>)->(tensor<4xi8>,tensor<4xi8>) attributes {llvm.emit_c_interface} {
  %x=func.call @helper(%a):(tensor<4xi8>)->tensor<4xi8>
  %y=func.call @helper(%x):(tensor<4xi8>)->tensor<4xi8>
  return %x,%y:tensor<4xi8>,tensor<4xi8>
 }
 func.func private @helper(%a:tensor<4xi8>)->tensor<4xi8> {
  %scratch=tensor.empty():tensor<7xi32>
  %out=tensor.empty():tensor<4xi8>
  %r=func.call @writer(%a,%scratch,%out):(tensor<4xi8>,tensor<7xi32>,tensor<4xi8>)->tensor<4xi8>
  return %r:tensor<4xi8>
 }
 func.func private @writer(tensor<4xi8>,tensor<7xi32>,tensor<4xi8>)->tensor<4xi8> attributes {llvm.emit_c_interface}
}"""


def fixture():
    module = Parser(make_context(tensor.Tensor), SOURCE).parse_module()
    declaration = module.body.block.last_op
    declaration.properties["arg_attrs"] = ArrayAttr(
        [DictionaryAttr({"bufferization.access": StringAttr(mode)}) for mode in ["read", "write", "write"]]
    )
    report = rewrite_fresh_tensor_writers(module, [replace(contract(), private_workspaces=(workspace(),))])
    return module, report


def test_pool_dominates_sequential_helper_calls_and_keeps_public_abi():
    module, reports = fixture()
    public = module.body.block.first_op
    before_type = public.function_type
    result = pool_private_writer_workspaces(module, reports)
    assert public.function_type == before_type
    assert result["required_bytes"] == [37] and result["required_alignments"] == [64]
    assert result["root_calls"] == 2 and result["private_wrapper_calls"] == 1
    assert not result["data_cache"] and result["release"] == "normal_upstream_owner_deallocation"
    allocations = [op for op in public.body.block.ops if isinstance(op, memref.AllocOp)]
    assert len(allocations) == 1
    helper = next(op for op in module.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "helper")
    writer = next(op for op in module.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "writer")
    assert len(helper.function_type.inputs) == 2 and len(writer.function_type.inputs) == 4
    assert not any(
        isinstance(op, memref.AllocOp)
        and op.memref.type.element_type == allocations[0].memref.type.element_type
        and op.memref.type.get_shape() == (37,)
        for op in writer.body.block.ops
    )
    calls = [op for op in public.body.block.ops if isinstance(op, func.CallOp)]
    assert calls[0].arguments[-1] is calls[1].arguments[-1]
    module.verify()


@pytest.mark.parametrize(
    "mutation", ["multiple_public_owners", "symbolic_escape", "extra_workspace_use", "false_completion"]
)
def test_unsupported_pooling_refuses_before_mutation(mutation):
    module, reports = fixture()
    if mutation == "multiple_public_owners":
        second = module.body.block.first_op.clone()
        second.sym_name = StringAttr("other_public")
        module.body.block.add_op(second)
    elif mutation == "symbolic_escape":
        module.attributes["address"] = ArrayAttr([DictionaryAttr({"nested": SymbolRefAttr("writer")})])
    elif mutation == "extra_workspace_use":
        writer = next(
            op for op in module.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "writer"
        )
        borrowed = next(op for op in writer.body.block.ops if isinstance(op, func.CallOp))
        # Any second consumer, even a pure descriptor cast, invalidates the
        # generated allocation's sole-use/noescape ownership witness.
        cast = memref.CastOp.get(borrowed.arguments[-1], borrowed.arguments[-1].type)
        writer.body.block.insert_op_before(cast, borrowed)
    else:
        reports[0]["private_workspaces"][0]["synchronous_before_return"] = False
    before = text(module)
    with pytest.raises(ValueError):
        pool_private_writer_workspaces(module, reports)
    assert text(module) == before


@pytest.mark.parametrize("optimization", ["-O0", "-O2"])
def test_actual_upstream_pooled_workspace_native(tmp_path, optimization, upstream_host_tools):
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    from merlin.llvmlower.toolchain import clang

    module, reports = fixture()
    pool_private_writer_workspaces(module, reports)
    llvm = lower_to_llvm_ir(text(module, generic=True), workdir=tmp_path / "lower", vectorize=False)
    source = tmp_path / "model.ll"
    source.write_text(llvm)
    adapter = tmp_path / "adapter.c"
    adapter.write_text("""#include <stdint.h>
#include <stdlib.h>
typedef struct{void*allocated,*aligned;int64_t offset,size[1],stride[1];}M;
static void*current;static int calls,releases;
void __real_free(void*);
void __wrap_free(void*p){if(current &&p==current){if(calls%2)abort();releases++;current=0;}__real_free(p);}
int workspace_calls(void){return calls;}int workspace_releases(void){return releases;}
void _mlir_ciface_borrowed(M*a,M*s,M*out,M*w){
 if(w->size[0]<37 || w->offset || w->stride[0]!=1 || (uintptr_t)w->aligned%64)abort();
 if(w->aligned==a->aligned || w->aligned==s->aligned || w->aligned==out->aligned)abort();
 if(calls%2){if(w->allocated!=current)abort();}else{if(current)abort();current=w->allocated;}
 // Initialize the private state each call; repeated use shares only bytes.
 for(int i=0;i<37;i++)((uint8_t*)w->aligned)[i]=(uint8_t)(i+5);
 for(int i=0;i<7;i++)((int32_t*)s->aligned)[i]=i;
 for(int i=0;i<4;i++){
  ((int8_t*)out->aligned)[i]=(int8_t)(((int8_t*)a->aligned)[a->offset+i*a->stride[0]]+((uint8_t*)w->aligned)[i]);
 }
 calls++;
}
""")
    obj = tmp_path / "model.o"
    library = tmp_path / ("pooled_" + optimization + ".so")
    subprocess.run([str(clang()), optimization, "-fPIC", "-c", str(source), "-o", str(obj)], check=True)
    subprocess.run(
        [
            "cc",
            optimization,
            "-fPIC",
            "-shared",
            str(obj),
            str(adapter),
            str(mlir_runtime_c()),
            "-Wl,--wrap=free",
            "-o",
            str(library),
        ],
        check=True,
    )
    invoke = HostModel.load(str(library))
    original = np.array([-128, -1, 0, 100], np.int8)
    for iteration in range(20):
        first = np.full(4, 77, np.int8)
        second = np.full(4, 77, np.int8)
        invoke([(v.ctypes.data, v.shape) for v in [original, first, second]])
        np.testing.assert_array_equal(original, [-128, -1, 0, 100])
        np.testing.assert_array_equal(first, [-123, 5, 7, 108])
        np.testing.assert_array_equal(second, [-118, 11, 14, 116])
        assert invoke.lib.workspace_calls() == 2 * (iteration + 1)
        assert invoke.lib.workspace_releases() == iteration + 1


def two_provider_fixture():
    from xdsl.dialects.builtin import UnitAttr

    module, reports = fixture()
    helper = next(op for op in module.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "helper")
    second = helper.clone()
    second.sym_name = StringAttr("helper_second")
    call = next(op for op in second.body.block.ops if isinstance(op, func.CallOp))
    call.properties["callee"] = SymbolRefAttr("writer_second")
    module.body.block.add_op(second)
    owner = module.body.block.first_op
    owner_calls = [op for op in owner.body.block.ops if isinstance(op, func.CallOp)]
    owner_calls[1].properties["callee"] = SymbolRefAttr("helper_second")
    writer = next(op for op in module.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "writer")
    declaration = func.FuncOp.external("writer_second", writer.function_type.inputs, writer.function_type.outputs)
    declaration.attributes["llvm.emit_c_interface"] = UnitAttr()
    declaration.properties["arg_attrs"] = writer.properties["arg_attrs"]
    module.body.block.add_op(declaration)
    reports += rewrite_fresh_tensor_writers(
        module, [replace(contract("writer_second", "borrowed_second"), private_workspaces=(workspace(53, 128),))]
    )
    return module, reports


def test_different_workspace_capacities_share_maximum_owned_extent():
    module, reports = two_provider_fixture()
    result = pool_private_writer_workspaces(module, reports)
    assert result["required_bytes"] == [53] and result["required_alignments"] == [128]
    assert result["root_calls"] == 2 and result["private_wrapper_calls"] == 2
    assert result["allocations"] == 1
    owner = module.body.block.first_op
    calls = [op for op in owner.body.block.ops if isinstance(op, func.CallOp)]
    assert calls[0].arguments[-1] is calls[1].arguments[-1]
    module.verify()


def test_explicit_stack_pool_keeps_owner_abi_and_checks_padding_before_mutation():
    module, reports = fixture()
    public = module.body.block.first_op
    before_type, before = public.function_type, text(module)
    # 37 data bytes and up to 63 alignment bytes must both be budgeted.
    with pytest.raises(ValueError, match="exceeds"):
        pool_private_writer_workspaces(module, reports, stack_budget_bytes=99)
    assert text(module) == before
    result = pool_private_writer_workspaces(module, reports, stack_budget_bytes=100)
    assert public.function_type == before_type
    assert result["storage"] == "invocation_stack"
    assert result["maximum_storage_bytes_including_alignment"] == 100
    assert result["release"] == "owner_return_stack_lifetime"
    assert result["final_frame_admission"] == "not_established"
    assert len([op for op in public.body.block.ops if isinstance(op, memref.AllocaOp)]) == 1
    assert not any(isinstance(op, memref.AllocOp) for op in public.body.block.ops)
    calls = [op for op in public.body.block.ops if isinstance(op, func.CallOp)]
    assert calls[0].arguments[-1] is calls[1].arguments[-1]
    module.verify()
