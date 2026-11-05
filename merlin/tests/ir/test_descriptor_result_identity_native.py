"""Real ordinary host lowering with checked C bridge, live results and state.

The synthetic writer reads the initialized output and increments a visible
counter. This proves no full-write permission or invocation elimination is used.
No target completion/qualification or performance result is inferred.
The shared selection helper's effects are the original neutral fixture's
declaration for the C body below, independently of its numerical test results.
"""

import ctypes
import hashlib
import json
import subprocess

import numpy as np
import pytest
from test_descriptor_result_identity import selection, source

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.codegen import mlir_runtime_c
from merlin.llvmlower.descriptor_result_identity import ensure_registered, rewrite_descriptor_result_identities
from merlin.llvmlower.device_shim import _borrowed_result_wrapper
from merlin.llvmlower.lower import lower_model
from merlin.llvmlower.toolchain import clang
from merlin.xdsl_dialects._common import text


def _runtime():
    names = [
        (
            f"void *{n}_alloc",
            f"void *{n}_aligned",
            f"intptr_t {n}_off",
            f"intptr_t {n}_s0",
            f"intptr_t {n}_s1",
            f"intptr_t {n}_st0",
            f"intptr_t {n}_st1",
        )
        for n in "abc"
    ]
    fields = [
        f"{n}->allocated,{n}->aligned,{n}->offset,{n}->sizes[0],{n}->sizes[1],{n}->strides[0],{n}->strides[1]"
        for n in "abc"
    ]
    return (
        """#include <stdint.h>
#include <stdlib.h>
typedef struct {void *allocated,*aligned; intptr_t offset,sizes[2],strides[2];} merlin_memref_2d;
static uint32_t calls;
uint32_t get_calls(void){return calls;}
"""
        + f"""merlin_memref_2d writer({",".join(x for group in names for x in group)}){{
 ++calls;
 for(intptr_t i=0;i<c_s0;i++)for(intptr_t j=0;j<c_s1;j++){{
  intptr_t ai=a_off+i*a_st0+j*a_st1,bi=b_off+i*b_st0+j*b_st1,ci=c_off+i*c_st0+j*c_st1;
  ((uint32_t*)c_aligned)[ci]+=((uint32_t*)a_aligned)[ai]+((uint32_t*)b_aligned)[bi]+calls;
 }}
 return (merlin_memref_2d){{c_alloc,c_aligned,c_off,{{c_s0,c_s1}},{{c_st0,c_st1}}}};
}}
void _mlir_ciface_writer(merlin_memref_2d *r,const merlin_memref_2d *a,
 const merlin_memref_2d *b,const merlin_memref_2d *c){{*r=writer({",".join(fields)});}}
"""
        + _borrowed_result_wrapper("writer", 2, "borrowed")
    )


def _source(rows, cols):
    t = f"tensor<{rows}x{cols}xi32>"
    original = source(rows, cols).replace('bufferization.access="write"', 'bufferization.access="read-write"')
    # Both next and the original writer result escape, detecting CSE reuse of
    # the old initializer across the call; state is a real invocation argument.
    return original.replace(f"%b:{t})->({t},{t})", f"%b:{t},%state:{t})->({t},{t},{t})").replace(
        f"return %r,%a:{t},{t}",
        f"""%out=tensor.empty():{t}
%next=linalg.generic {{indexing_maps=[affine_map<(d0,d1)->(d0,d1)>,
affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>],
iterator_types=["parallel","parallel"]}} ins(%r,%state:{t},{t}) outs(%out:{t}) {{
^bb0(%x:i32,%s:i32,%old:i32):
%sum=arith.addi %x,%s:i32
linalg.yield %sum:i32
}}->{t}
return %next,%r,%a:{t},{t},{t}""",
    )


def _descriptor_type():
    class D(ctypes.Structure):
        _fields_ = [
            ("allocated", ctypes.c_void_p),
            ("aligned", ctypes.c_void_p),
            ("offset", ctypes.c_int64),
            ("sizes", ctypes.c_int64 * 2),
            ("strides", ctypes.c_int64 * 2),
        ]

    return D


@pytest.mark.parametrize("rows,cols", [(2, 7), (3, 5)])
def test_real_native_result_identity_preserves_all_outputs_and_carried_state(tmp_path, rows, cols):
    selected_tool = clang()
    runtime = tmp_path / "writer.c"
    runtime.write_text(_runtime())
    D = _descriptor_type()

    def desc(array):
        return D(
            array.ctypes.data,
            array.ctypes.data,
            0,
            (ctypes.c_int64 * 2)(*array.shape),
            (ctypes.c_int64 * 2)(*(s // 4 for s in array.strides)),
        )

    records = {}
    for arm in ("original", "identity"):
        directory = tmp_path / arm
        directory.mkdir()
        module = parse_mlir_text(_source(rows, cols))
        feature = frozenset()
        if arm == "identity":
            assert rewrite_descriptor_result_identities(module, (selection(module),))[0]["calls"] == 1
            feature = frozenset((ensure_registered(),))
        module.verify()
        prepared = text(module)
        result = lower_model(prepared, directory / "lower", targets=(), features=feature)
        llvm = directory / "model.ll"
        llvm_text = result.ll_path.read_text()
        llvm.write_text(llvm_text)
        library = directory / "native.so"
        argv = [
            str(selected_tool),
            "-shared",
            "-O2",
            "-fPIC",
            str(llvm),
            str(runtime),
            str(mlir_runtime_c()),
            "-lm",
            "-Wl,-z,defs",
            "-o",
            str(library),
        ]
        built = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        assert built.returncode == 0, built.stderr
        loaded = ctypes.CDLL(str(library), mode=ctypes.RTLD_LOCAL)
        call = loaded._mlir_ciface_forward
        call.argtypes = [ctypes.POINTER(D)] * 6
        loaded.get_calls.restype = ctypes.c_uint32
        state = np.arange(rows * cols, dtype=np.uint32).reshape(rows, cols)
        calls = []
        for epoch in range(3):
            a = np.asarray([((i * 117 + epoch) % 1001) - 500 for i in range(rows * cols)], dtype=np.int32).reshape(
                rows, cols
            )
            # Equal read-only input pointers are legal; output remains fresh.
            b = a if epoch == 1 else (a * 3).copy()
            before = a.copy(), b.copy(), state.copy()
            storage = [np.full(rows * cols + 16, 0x12345678, dtype=np.uint32) for _ in range(3)]
            outputs = [value[8:-8].reshape(rows, cols) for value in storage]
            values = [desc(x) for x in (a, b, state, *outputs)]
            call(*(ctypes.byref(value) for value in values))
            expected = np.empty((rows, cols), dtype=np.uint32)
            next_state = expected.copy()
            for i in range(rows):
                for j in range(cols):
                    expected[i, j] = (5 + int(a[i, j]) + int(b[i, j]) + epoch + 1) % (1 << 32)
                    next_state[i, j] = (int(expected[i, j]) + int(state[i, j])) % (1 << 32)
            np.savez(
                directory / f"call-{epoch}.npz",
                a=a,
                b=b,
                old_state=state,
                next=outputs[0],
                wide=outputs[1],
                copy=outputs[2],
                guards=np.stack(storage),
            )
            assert outputs[0].tobytes() == next_state.tobytes()
            assert outputs[1].tobytes() == expected.tobytes()
            assert outputs[2].tobytes() == a.tobytes()
            assert all(actual.tobytes() == old.tobytes() for actual, old in zip((a, b, state), before, strict=True))
            assert all(np.all(x[:8] == 0x12345678) and np.all(x[-8:] == 0x12345678) for x in storage)
            assert loaded.get_calls() == epoch + 1
            state = outputs[0].copy()
            calls.append([hashlib.sha256(x.tobytes()).hexdigest() for x in outputs])
        records[arm] = {
            "calls": calls,
            "native": str(library),
            "native_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            "argv": argv,
            "llvm": str(llvm),
            "malloc_sites": llvm_text.count("call ptr @malloc("),
        }
    assert records["original"]["calls"] == records["identity"]["calls"]
    (tmp_path / "pair.json").write_text(json.dumps(records, indent=2) + "\n")
