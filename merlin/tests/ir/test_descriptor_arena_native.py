"""Original descriptor effects survive actual rewriting and native ownership.

The neutral C implementation supplies the independent blocking/noescape premise;
neither its output descriptor equality nor the numerical checks grants it.
"""

import ctypes
import subprocess
from dataclasses import replace

import numpy as np
import pytest
from xdsl.dialects import func
from xdsl.dialects.builtin import MemRefType

from merlin.common.digest import sha256_file
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.abi import HostModel
from merlin.llvmlower.arena_bind import bind_arena
from merlin.llvmlower.borrowed_buffer_calls import BorrowedBufferCall, BorrowedBufferCalls, BorrowedBufferEffects
from merlin.llvmlower.descriptor_result_identity import DescriptorResultIdentity
from merlin.llvmlower.descriptor_result_preparation import DescriptorResultPreparation
from merlin.llvmlower.device_shim import _borrowed_result_wrapper
from merlin.llvmlower.lower import lower_model
from merlin.llvmlower.target_data_layout import _query
from merlin.llvmlower.toolchain import clang


def source(m=2, k=3, n=4, *, observer=False):
    a, w, q, out = (f"tensor<{m}x{k}xi8>", f"tensor<{k}x{n}xi8>",
                    f"tensor<{m}x{n}xi8>", f"tensor<{m}x{n}xi32>")
    ww = f"tensor<{n}x{n}xi8>"
    return f'''module {{
 func.func private @first(%a:{a} {{bufferization.access="read"}},
  %b:{w} {{bufferization.access="read"}},%c:{out} {{bufferization.access="write"}})->{out}
  attributes {{llvm.emit_c_interface}}
 func.func private @second(%a:{q} {{bufferization.access="read"}},
  %b:{ww} {{bufferization.access="read"}},%c:{out} {{bufferization.access="write"}})->{out}
  attributes {{llvm.emit_c_interface}}
 func.func @forward(%a:{a},%w:{w},%w1:{ww})->({out}{','+out if observer else ''}) {{
  %empty=tensor.empty():{out}
  %p=func.call @first(%a,%w,%empty):({a},{w},{out})->{out}
  %qe=tensor.empty():{q}
  %q=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],
   iterator_types=["parallel","parallel"]}} ins(%p:{out}) outs(%qe:{q}) {{
   ^bb0(%x:i32,%unused:i8):
    %v=arith.trunci %x:i32 to i8
    linalg.yield %v:i8
  }}->{q}
  %ye=tensor.empty():{out}
  %y=func.call @second(%q,%w1,%ye):({q},{ww},{out})->{out}
  %oe=tensor.empty():{out}
  %o=linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],
   iterator_types=["parallel","parallel"]}} ins(%y:{out}) outs(%oe:{out}) {{
   ^bb0(%x:i32,%unused:i32):
    %one=arith.constant 1:i32
    %v=arith.addi %x,%one:i32
    linalg.yield %v:i32
  }}->{out}
  return %o{',%p' if observer else ''}:{out}{','+out if observer else ''}
 }}
}}'''


def implementation():
    code = r'''
#include <stdint.h>
#include <stdlib.h>
typedef struct {void*allocated,*aligned;intptr_t offset,sizes[2],strides[2];} merlin_memref_2d;
static uintptr_t expected_first,expected_middle,expected_second;
static int checked;
void expect_arena(uintptr_t first,uintptr_t middle,uintptr_t second) {
 expected_first=first;expected_middle=middle;expected_second=second;checked=0;
}
int checked_addresses(void){return checked;}
static void product(const merlin_memref_2d*a,const merlin_memref_2d*w,const merlin_memref_2d*y){
 for(intptr_t i=0;i<y->sizes[0];i++)for(intptr_t j=0;j<y->sizes[1];j++){
  int32_t sum=0;
  for(intptr_t k=0;k<a->sizes[1];k++)sum+=
   (int32_t)((int8_t*)a->aligned)[a->offset+i*a->strides[0]+k*a->strides[1]]*
   (int32_t)((int8_t*)w->aligned)[w->offset+k*w->strides[0]+j*w->strides[1]];
  ((int32_t*)y->aligned)[y->offset+i*y->strides[0]+j*y->strides[1]]=sum;
 }
}
void _mlir_ciface_first(merlin_memref_2d*r,const merlin_memref_2d*a,const merlin_memref_2d*w,const merlin_memref_2d*y){
 if(expected_first){if((uintptr_t)y->aligned!=expected_first)abort();checked++;}
 product(a,w,y);*r=*y;
}
void _mlir_ciface_second(merlin_memref_2d*r,const merlin_memref_2d*a,const merlin_memref_2d*w,const merlin_memref_2d*y){
 if(expected_second){if((uintptr_t)y->aligned!=expected_second||(uintptr_t)a->aligned!=expected_middle||a->aligned==y->aligned)abort();checked++;}
 product(a,w,y);*r=*y;
}
'''
    parameters = ",".join(f"void*{x}a,void*{x}p,intptr_t {x}o,intptr_t {x}r,intptr_t {x}c,intptr_t {x}s,intptr_t {x}t"
                          for x in "abc")
    descriptors = "".join(f"merlin_memref_2d {x}={{{x}a,{x}p,{x}o,{{{x}r,{x}c}},{{{x}s,{x}t}}}};" for x in "abc")
    for name in ("first", "second"):
        code += (f"merlin_memref_2d {name}({parameters}){{{descriptors}merlin_memref_2d r;"
                 f"_mlir_ciface_{name}(&r,&a,&b,&c);return r;}}\n")
        code += _borrowed_result_wrapper(name, 2, name + "_borrowed")
    return code


def selection(tmp_path, program):
    original, c = tmp_path / "source.mlir", tmp_path / "implementation.c"
    original.write_text(program)
    c.write_text(implementation())
    module = parse_mlir_text(program)
    effects = BorrowedBufferEffects("source-bound synchronous neutral contractions; no retention or release", True, True, True)
    identities, calls = [], []
    for op in module.body.block.ops:
        if type(op) is not func.FuncOp or op.body.blocks:
            continue
        name = op.sym_name.data
        types = tuple(op.function_type.inputs)
        identities.append(DescriptorResultIdentity(name, name + "_borrowed", name + "_identity", 2,
                          types, ("read", "read", "write"), effects))
        calls.append(BorrowedBufferCall(name + "_identity", name + "_borrowed",
                     tuple(str(MemRefType(t.get_element_type(), t.get_shape())) for t in types), 2, effects))
    state = {"valid": True}

    def validate():
        if not state["valid"]:
            raise ValueError("original neutral implementation is no longer qualified")

    digest = sha256_file(original)
    owner = BorrowedBufferCalls(original, digest, digest, tuple(calls), ((c, sha256_file(c)),), validate)
    return tuple(identities), owner, state


@pytest.mark.parametrize("shape,observer", [((2, 3, 4), False), ((3, 3, 5), False), ((2, 3, 4), True)])
def test_descriptor_chain_reaches_current_native_lifetime_and_real_arena(tmp_path, upstream_host_tools, shape, observer):
    program = source(*shape, observer=observer)
    identities, owner, state = selection(tmp_path, program)
    result = lower_model(program, tmp_path / "lower", targets=(), descriptor_result_identities=identities,
        descriptor_result_effects=owner, index_bits=64, data_layout=_query(clang(), ("-O2",)),
        static_arena=False, hoist_static_allocs=False)
    assert result.stats["descriptor_result_identity"]["rewrites"] == 2
    assert result.borrowed_arena_lifetime is not None
    assert result.stats["borrowed_buffer_lifetime"]["status"] == "checked"
    llvm = result.ll_path.read_text()
    unmodified, refused = bind_arena(llvm)
    assert unmodified == llvm and refused.bound == 0
    selected, report = bind_arena(llvm, borrowed_lifetime=result.borrowed_arena_lifetime)
    assert report.bound >= 2 and report.refused == 0
    placements = report.to_dict()["placements"]
    if observer:
        assert len(placements) == 2  # Original observed result remains caller-owned.
        first, (middle, second) = None, placements
    else:
        assert len(placements) == 3
        first, middle, second = placements
    assert middle["offset"] != second["offset"]
    if not observer:
        assert first["offset"] == second["offset"]
        assert first["release_line"] < second["allocation_line"]

    for label, text in (("heap", llvm), ("arena", selected)):
        ll, library = tmp_path / (label + ".ll"), tmp_path / (label + "_" + "_".join(map(str, (*shape, int(observer)))) + ".so")
        address = "getelementptr (i8, ptr @merlin_arena, i64 0)" if label == "arena" else "null"
        ll.write_text(text + "\ndefine ptr @arena_address(){ret ptr " + address + "}\n")
        subprocess.run([str(clang()), "-O2", "-shared", "-fPIC", str(ll), str(tmp_path / "implementation.c"),
                        "-Wl,-z,defs", "-o", str(library)], check=True, capture_output=True)
        loaded, invoke = ctypes.CDLL(str(library)), HostModel.load(str(library))
        loaded.arena_address.restype = ctypes.c_void_p
        loaded.expect_arena.argtypes = [ctypes.c_size_t] * 3
        base = loaded.arena_address() or 0
        loaded.expect_arena(*(base + item["offset"] if base and item is not None else 0
                              for item in (first, middle, second)))
        m, k, n = shape
        for seed in (1, 7):
            a = ((np.arange(m*k).reshape(m,k)*19+seed)%251-125).astype(np.int8)
            w = ((np.arange(k*n).reshape(k,n)*7+seed)%31-15).astype(np.int8)
            w1 = ((np.arange(n*n).reshape(n,n)*11+seed)%23-11).astype(np.int8)
            expected_p = a.astype(np.int32) @ w.astype(np.int32)
            expected = expected_p.astype(np.int8).astype(np.int32) @ w1.astype(np.int32) + 1
            guarded = np.full(expected.size+8, -1234567, np.int32)
            out = guarded[4:-4].reshape(expected.shape)
            observed = np.full_like(expected_p, -98765)
            if observer and base:
                loaded.expect_arena(observed.ctypes.data, base + middle["offset"], base + second["offset"])
            arguments = (a, w, w1, out, observed) if observer else (a, w, w1, out)
            copies = tuple(x.copy() for x in (a,w,w1))
            invoke([(x.ctypes.data, x.shape) for x in arguments])
            np.testing.assert_array_equal(out, expected)
            if observer:
                np.testing.assert_array_equal(observed, expected_p)
            assert np.all(guarded[:4] == -1234567) and np.all(guarded[-4:] == -1234567)
            for x, before in zip((a,w,w1), copies, strict=True):
                np.testing.assert_array_equal(x, before)
        assert loaded.checked_addresses() == ((2 if observer else 4) if base else 0)
    state["valid"] = False
    with pytest.raises(ValueError, match="no longer qualified"):
        bind_arena(llvm, borrowed_lifetime=result.borrowed_arena_lifetime)
    state["valid"] = True
    changed, stale = bind_arena(llvm + "\n", borrowed_lifetime=result.borrowed_arena_lifetime)
    assert changed == llvm + "\n" and stale.bound == 0
    assert stale.borrowed_lifetime["status"] == "llvm_changed_conservative"


@pytest.mark.parametrize("mutation", ["record", "roster", "type", "effect_identity", "result"])
def test_only_original_matching_live_effects_can_join_descriptor_rewrite(tmp_path, mutation):
    identities, owner, _ = selection(tmp_path, source())
    if mutation == "record":
        owner = owner.record()
    elif mutation == "roster":
        owner = replace(owner, calls=owner.calls[:1])
    else:
        call = owner.calls[0]
        if mutation == "type":
            call = replace(call, argument_types=("memref<1xi8>",)*3)
        elif mutation == "effect_identity":
            call = replace(call, effects=replace(call.effects))
        else:
            call = replace(call, result_argument=0)
        owner = replace(owner, calls=(call, *owner.calls[1:]))
    with pytest.raises(ValueError, match="original|descriptor"):
        DescriptorResultPreparation(identities, effects=owner)


def test_late_original_callback_replacement_or_source_change_cannot_replay(tmp_path):
    identities, owner, _ = selection(tmp_path, source())
    prepared = DescriptorResultPreparation(identities, effects=owner)
    object.__setattr__(owner, "validate_original", lambda: None)
    with pytest.raises(ValueError, match="original descriptor"):
        prepared.revalidate()


def test_observed_initializer_does_not_gain_descriptor_result_identity(tmp_path):
    program = source().replace("return %o:tensor<2x4xi32>",
        "return %o,%empty:tensor<2x4xi32>,tensor<2x4xi32>")
    program = program.replace("->(tensor<2x4xi32>) {", "->(tensor<2x4xi32>,tensor<2x4xi32>) {")
    identities, owner, _ = selection(tmp_path, program)
    prepared = DescriptorResultPreparation(identities, effects=owner)
    selected = prepared.prepare(program, tmp_path)
    assert prepared.rewrites == 1  # The unobserved second output alone is eligible.
    current = prepared.native_calls()
    assert tuple(call.borrowed_symbol for call in current.calls) == ("second_borrowed",)
    module = parse_mlir_text(selected)
    assert any(type(op) is func.CallOp and op.callee.root_reference.data == "first" for op in module.walk())
    owner.source_path.write_text(program + "\n")
    with pytest.raises(ValueError, match="source or contracts changed"):
        prepared.revalidate()


@pytest.mark.parametrize("change", ["good", "swap", "alias", "release", "result"])
def test_native_result_drop_forwarding_never_authorizes_changed_aliases_or_releases(change, upstream_host_tools):
    from merlin.llvmlower.pipeline import dealloc_placement_violations

    typ = "memref<2x2xi32>"
    args = "%a,%b,%out" if change not in ("swap", "alias") else "%b,%a,%out" if change == "swap" else "%a,%b,%a"
    release = f"memref.dealloc %out:{typ}" if change == "release" else ""
    result = f"->{typ}" if change == "result" else ""
    returned = f"%out:{typ}" if change == "result" else ""
    program = f'''module {{
     func.func private @borrowed({typ},{typ},{typ}) attributes {{llvm.emit_c_interface}}
     func.func private @wrapper(%a:{typ},%b:{typ},%out:{typ}){result} {{
      func.call @borrowed({args}):({typ},{typ},{typ})->()
      {release}
      return {returned}
     }}
    }}'''
    effect = BorrowedBufferEffects("independent original neutral blocking premise", True, True, True)
    call = BorrowedBufferCall("wrapper", "borrowed", (typ,)*3, 2, effect)
    violations = dealloc_placement_violations(program, borrowed_calls=(call,))
    assert bool(violations) is (change != "good"), violations
    if change != "good":
        assert any("borrowed ABI" in problem for problem in violations)
