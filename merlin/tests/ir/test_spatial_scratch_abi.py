"""Caller-owned spatial scratch stays private and checks transport before use."""

import hashlib
import shutil
import subprocess
from dataclasses import replace

import pytest
from xdsl.dialects import func, memref, tensor
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr
from xdsl.parser import Parser

from merlin.compile.scheduling.block_schedule import SpatialMaxReadout
from merlin.compile.scheduling.program_limits import AddressWindow
from merlin.compile.scheduling.spatial_producer import PackedSpatialProducerPlan, SpatialProducerPlan
from merlin.compile.scheduling.spatial_readout import RoundedReadout, SpatialReadoutChain
from merlin.compile.scheduling.window_transfer import CountedConvolutionPlan, Window2D
from merlin.llvmlower.borrowed_buffer_calls import BorrowedBufferEffects
from merlin.llvmlower.fresh_tensor_writer import FreshTensorWriterContract, PrivateWorkspaceContract
from merlin.llvmlower.spatial_scratch_abi import emit_spatial_raw_entry, prepare_spatial_scratch_abi
from merlin.xdsl_dialects._common import make_context, text

SOURCE = """module {
 func.func @forward(%x:tensor<2x5x5x3xi8>,%w:tensor<27x17xi8>,%b:tensor<17xi32>)
   -> tensor<18x17xi8> attributes {llvm.emit_c_interface} {
  %out=tensor.empty():tensor<18x17xi8>
  %r=func.call @pair(%x,%w,%b,%out):
    (tensor<2x5x5x3xi8>,tensor<27x17xi8>,tensor<17xi32>,tensor<18x17xi8>)->tensor<18x17xi8>
  return %r:tensor<18x17xi8>
 }
 func.func private @pair(tensor<2x5x5x3xi8>,tensor<27x17xi8>,tensor<17xi32>,tensor<18x17xi8>)
  ->tensor<18x17xi8> attributes {llvm.emit_c_interface}
}"""


def fixture():
    module = Parser(make_context(tensor.Tensor), SOURCE).parse_module()
    module.body.block.last_op.properties["arg_attrs"] = ArrayAttr(
        [DictionaryAttr({"bufferization.access": StringAttr(mode)}) for mode in ("read", "read", "read", "write")]
    )
    p = CountedConvolutionPlan(
        Window2D((2, 5, 5, 3), (3, 3), padding=(1, 1, 1, 1)), 17, 16, 5, 16, 3, 128, 256, 0, 16, 0, True
    )
    q = SpatialReadoutChain(RoundedReadout((0.5,), True), RoundedReadout((0.9900199770927429,)), None)
    plan = SpatialProducerPlan(
        p,
        SpatialMaxReadout((5, 5), (3, 3), (2, 2), (1, 1, 1, 1), -128),
        (q, q),
        2,
        readout_mode="staged_readouts",
        intermediate=AddressWindow("private_stripe", 0, 240),
    )
    effects = BorrowedBufferEffects("synthetic ABI test only; no device qualification", True, True, True)
    contract = FreshTensorWriterContract("pair", 3, (3,), "borrowed_pair", 64, borrowed_effects=effects)
    workspace = PrivateWorkspaceContract(240, 64, "a" * 64, True, True, True)
    return module, plan, contract, workspace


def prepare(module, plan, contract, workspace):
    return prepare_spatial_scratch_abi(
        module,
        plan,
        contract,
        workspace,
        kernel_symbol="raw_pair",
        pointer_alignments=(1, 1, 4, 1),
        source_binding_sha256=hashlib.sha256(text(module).encode()).hexdigest(),
    )


def test_normal_private_pool_preserves_original_logical_signature():
    module, plan, contract, workspace = fixture()
    before = text(module)
    artifact = prepare(module, plan, contract, workspace)
    assert text(module) == before
    original = module.body.block.first_op
    public = artifact.module.body.block.first_op
    assert public.function_type == original.function_type
    assert len(public.function_type.inputs) == 3 and len(public.function_type.outputs) == 1
    allocations = [op for op in public.body.block.ops if isinstance(op, memref.AllocOp)]
    assert len(allocations) == 1 and allocations[0].memref.type.get_shape() == (240,)
    assert allocations[0].alignment.value.data == 64
    borrowed = next(
        op
        for op in artifact.module.body.block.ops
        if isinstance(op, func.FuncOp) and op.sym_name.data == "borrowed_pair"
    )
    assert len(borrowed.function_type.inputs) == 5 and not borrowed.function_type.outputs
    assert artifact.receipt["pool"]["required_bytes"] == [240]
    assert artifact.receipt["pool"]["release"] == "normal_upstream_owner_deallocation"
    assert artifact.receipt["target_execution"] == "not_established"
    artifact.module.verify()


def test_packed_stripe_and_input_panel_share_normal_private_owner_not_logical_arguments():
    module, source, contract, workspace = fixture()
    plan = PackedSpatialProducerPlan(source, 7, AddressWindow("private_packed", 0, 512), 64)
    assert plan.stripe_bytes == 255 and plan.panel_offset == 256 and plan.required_bytes == 445
    with pytest.raises(ValueError, match="complete staged stripe"):
        prepare(module, plan, contract, workspace)
    artifact = prepare(module, plan, contract, replace(workspace, bytes=512))
    original, current = module.body.block.first_op, artifact.module.body.block.first_op
    assert current.function_type == original.function_type
    assert len(current.function_type.inputs) == 3
    assert artifact.receipt["required_stripe_bytes"] == 445
    assert artifact.receipt["plan_sha256"] == plan.identity != source.identity
    with pytest.raises(ValueError, match="alignment"):
        prepare(module, plan, contract, replace(workspace, bytes=512, alignment=32))


@pytest.mark.parametrize("mutation", ["capacity", "completion", "noescape", "wrong_shape", "second_owner"])
def test_missing_original_ownership_refuses_without_mutating_source(mutation):
    module, plan, contract, workspace = fixture()
    if mutation == "capacity":
        workspace = replace(workspace, bytes=239)
    elif mutation == "completion":
        workspace = replace(workspace, synchronous_before_return=False)
    elif mutation == "noescape":
        object.__setattr__(contract.borrowed_effects, "no_retained_aliases", False)
    elif mutation == "wrong_shape":
        plan = replace(plan, producer=replace(plan.producer, bias=False))
    else:
        second = module.body.block.first_op.clone()
        second.sym_name = StringAttr("second_public")
        module.body.block.add_op(second)
    before = text(module)
    with pytest.raises(ValueError):
        prepare(module, plan, contract, workspace)
    assert text(module) == before


def test_actual_descriptor_bridge_accepts_capacity_and_rejects_aliases_before_call(tmp_path):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("native C compiler unavailable")
    artifact = prepare(*fixture())
    harness = r"""
#include <stdlib.h>
#include <signal.h>
static _Alignas(64) unsigned char x[150],w[459],b[68],y[306],storage[304];
static volatile sig_atomic_t calls;
void raw_pair(void*a,void*k,void*bias,void*out,void*scratch){
 if(a!=x || k!=w || bias!=b || out!=y || scratch!=storage)abort();
 calls++;for(unsigned i=0;i<240;i++)((unsigned char*)scratch)[i]=(unsigned char)(i+calls);
 ((unsigned char*)out)[0]=((unsigned char*)scratch)[0];
}
static void trapped(int signal){(void)signal;_Exit(calls==0?0:99);}
int main(int argc,char**argv){
 int test=argc>1?atoi(argv[1]):0;
 M4 a={x,x,0,{2,5,5,3},{75,15,3,1}};
 M2 k={w,w,0,{27,17},{17,1}},out={y,y,0,{18,17},{17,1}};
 M1 bias={b,b,0,{17},{1}},s={storage,storage,0,{304},{1}};
 signal(SIGILL,trapped);signal(SIGTRAP,trapped);
 switch(test){
 case 1:s.size[0]=239;break;
 case 2:s.aligned=storage+1;break;
 case 3:s.aligned=y;break;
 case 4:out.aligned=x;break;
 case 5:k.stride[0]=16;break;
 case 6:s.offset=-1;break;
 case 7:out.aligned=&s;break;
 case 8:a.size[0]=1;break;
 case 9:s.offset=INT64_MAX;break;
 case 10:s.stride[0]=2;break;
 case 11:bias.aligned=b+1;break;
 case 12:k.aligned=x;break;
 }
 _mlir_ciface_borrowed_pair(&a,&k,&bias,&out,&s);
 if(test)return 7;
 if(calls!=1 || y[0]!=1)return 8;
 for(unsigned i=0;i<240;i++)storage[i]=255;
 _mlir_ciface_borrowed_pair(&a,&k,&bias,&out,&s);
 return calls==2 && y[0]==2 ?0:9;
}
"""
    source, binary = tmp_path / "bridge.c", tmp_path / "bridge"
    source.write_text(artifact.bridge_source + harness)
    subprocess.run(
        [cc, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    for case in range(13):
        run = subprocess.run([str(binary), str(case)], capture_output=True, text=True, timeout=10)
        assert run.returncode == 0, (case, run.returncode, run.stderr)


def test_upstream_owner_raw_entry_keeps_logical_pointers_and_releases_scratch(tmp_path, upstream_host_tools):
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    from merlin.llvmlower.tensor_helper_inlining import composition_features
    from merlin.llvmlower.toolchain import clang

    original, plan, contract, workspace = fixture()
    artifact = prepare(original, plan, contract, workspace)
    llvm = lower_to_llvm_ir(
        text(artifact.module, generic=True),
        workdir=tmp_path / "lower",
        vectorize=False,
        features=composition_features(None, has_writers=True),
        timeout=120,
    )
    # Normal helper inlining/result forwarding removes full-result copies;
    # the only allocation remains invocation-private aligned byte scratch.
    assert llvm.count("call ptr @malloc(") == llvm.count("call void @free(") == 1
    assert "llvm.memcpy" not in llvm
    model = tmp_path / "owner.ll"
    model.write_text(llvm)
    owner = tmp_path / "owner.o"
    subprocess.run(
        [str(clang()), "-O2", "-c", str(model), "-o", str(owner)], check=True, capture_output=True, text=True
    )
    source, binary = tmp_path / "linked.c", tmp_path / "linked"
    source.write_text(
        artifact.bridge_source
        + emit_spatial_raw_entry(artifact, plan)
        + r"""
#include <stdlib.h>
static _Alignas(64) unsigned char x[150],w[459],bias[68],out[306];
static void*live;static size_t capacity;static int allocations,releases,inside,calls;
void*__real_malloc(size_t);void __real_free(void*);
void*__wrap_malloc(size_t n){if(live)abort();capacity=n;live=__real_malloc(n);allocations++;return live;}
void __wrap_free(void*p){if(inside || p!=live)abort();live=0;releases++;__real_free(p);}
void raw_pair(void*a,void*k,void*b,void*y,void*scratch){
 if(a!=x || k!=w || b!=bias || y!=out || !live || (uintptr_t)scratch%64)abort();
 if((uintptr_t)scratch<(uintptr_t)live || (uintptr_t)scratch+240>(uintptr_t)live+capacity)abort();
 inside=1;calls++;
 for(unsigned i=0;i<240;i++)((unsigned char*)scratch)[i]=(unsigned char)(i+calls);
 for(unsigned i=0;i<306;i++)((unsigned char*)y)[i]=((unsigned char*)scratch)[i%240];
 inside=0;
}
int main(void){
 kernel_entry(x,w,bias,out);
 if(allocations!=1 || releases!=1 || calls!=1 || out[0]!=1 || live)return 1;
 kernel_entry(x,w,bias,out);
 return allocations==2 && releases==2 && calls==2 && out[0]==2 && !live ?0:2;
}
"""
    )
    subprocess.run(
        [
            str(clang()),
            "-std=c11",
            "-O2",
            str(source),
            str(owner),
            "-Wl,--wrap=malloc",
            "-Wl,--wrap=free",
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert run.returncode == 0, run.stderr


def test_public_entry_compiles_separately_from_the_private_device_descriptor_bridge(tmp_path, upstream_host_tools):
    from merlin.llvmlower.toolchain import clang

    original, plan, contract, workspace = fixture()
    artifact = prepare(original, plan, contract, workspace)
    source = tmp_path / "public-entry.c"
    source.write_text(emit_spatial_raw_entry(artifact, plan, standalone=True))
    assert "raw_pair" not in source.read_text()
    assert "malloc" not in source.read_text() and "free(" not in source.read_text()
    subprocess.run(
        [str(clang()), "-std=c11", "-O2", "-Werror", "-c", str(source), "-o", str(tmp_path / "public-entry.o")],
        check=True,
        capture_output=True,
        text=True,
    )


def test_explicit_stack_owner_lowers_without_heap_and_measures_static_frame(tmp_path, upstream_host_tools):
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    from merlin.llvmlower.tensor_helper_inlining import composition_features
    from merlin.llvmlower.toolchain import clang
    from merlin.targetgen.contract.stack_usage import measure_entrypoint

    module, plan, contract, workspace = fixture()
    artifact = prepare_spatial_scratch_abi(
        module,
        plan,
        contract,
        workspace,
        kernel_symbol="raw_pair",
        pointer_alignments=(1, 1, 4, 1),
        source_binding_sha256=hashlib.sha256(text(module).encode()).hexdigest(),
        stack_budget_bytes=workspace.bytes + workspace.alignment - 1,
    )
    llvm = lower_to_llvm_ir(
        text(artifact.module, generic=True),
        workdir=tmp_path / "lower",
        vectorize=False,
        features=composition_features(None, has_writers=True),
        timeout=120,
    )
    assert "@malloc" not in llvm and "@free" not in llvm and "llvm.memcpy" not in llvm
    source = tmp_path / "owner.ll"
    source.write_text(llvm)
    subprocess.run(
        [
            str(clang()),
            "--target=riscv64-unknown-elf",
            "-march=rv64gc",
            "-mabi=lp64d",
            "-mcmodel=medany",
            "-O2",
            "-fstack-usage",
            "-c",
            str(source),
            "-o",
            str(tmp_path / "owner.o"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    # This test's declared frame budget is not a target runtime admission.
    measurement = measure_entrypoint(
        tmp_path / "owner.su",
        llvm_path=source,
        entry_symbol="forward",
        max_static_bytes=4096,
    )
    assert workspace.bytes <= measurement.frame_bytes <= 4096
    assert artifact.receipt["pool"]["release"] == "owner_return_stack_lifetime"
