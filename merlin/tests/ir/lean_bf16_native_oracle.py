"""Test-only adapter: execute a 16-bit float tensor function with the compiler's own MLIR JIT.

Run in the compiler interpreter. Only upstream MLIR passes run here (the shared fidelity pipeline,
which keeps the source's loop order); no Merlin parser, rewrite or interpreter is involved.
Operands and the result cross the boundary as raw 16-bit patterns in ranked ``i16`` memrefs, so
every observed bit comes from the executed code. ``opt_level`` selects LLVM's code generation
level, which decides whether a ``contract``-flagged multiply-add may be fused.
"""

import ctypes
import ctypes.util
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
from torch_mlir import ir, passmanager
from torch_mlir.execution_engine import ExecutionEngine

sys.path.insert(0, sys.argv[2])
from mlir_execute import PASSES  # noqa: E402  (the shared fidelity pipeline, imported by path)


def _tensor(shape, element):
    return f"tensor<{''.join(f'{d}x' for d in shape)}{element}>"


def _memref(shape):
    return f"memref<{''.join(f'{d}x' for d in shape)}i16>"


def probe(function, inputs, output):
    lines, args = [], []
    params = [f"%in{i}: {_memref(spec['shape'])}" for i, spec in enumerate(inputs)]
    for i, spec in enumerate(inputs):
        raw = _tensor(spec["shape"], "i16")
        lines.append(f"%raw{i} = bufferization.to_tensor %in{i} restrict : {_memref(spec['shape'])} to {raw}")
        lines.append(f"%arg{i} = arith.bitcast %raw{i} : {raw} to {_tensor(spec['shape'], spec['format'])}")
        args.append(f"%arg{i}")
    signature = ", ".join(_tensor(spec["shape"], spec["format"]) for spec in inputs)
    result = _tensor(output["shape"], output["format"])
    lines.append(f"%result = func.call @{function}({', '.join(args)}) : ({signature}) -> {result}")
    lines.append(f"%bits = arith.bitcast %result : {result} to {_tensor(output['shape'], 'i16')}")
    lines.append(
        f"bufferization.materialize_in_destination %bits in restrict writable %out : "
        f"({_tensor(output['shape'], 'i16')}, {_memref(output['shape'])}) -> ()"
    )
    lines.append("return")
    return (
        f"func.func private @{function}({signature}) -> {result}\n"
        f"func.func @__bf16_probe({', '.join(params)}, %out: {_memref(output['shape'])}) "
        "attributes {llvm.emit_c_interface} {\n" + "\n".join(lines) + "\n}"
    )


def descriptor(array):
    rank = array.ndim

    class Memref(ctypes.Structure):
        _fields_ = [
            ("allocated", ctypes.c_void_p),
            ("aligned", ctypes.c_void_p),
            ("offset", ctypes.c_int64),
            ("sizes", ctypes.c_int64 * rank),
            ("strides", ctypes.c_int64 * rank),
        ]

    strides = [s // array.itemsize for s in array.strides]
    pointer = array.ctypes.data_as(ctypes.c_void_p)
    return Memref(pointer, pointer, 0, (ctypes.c_int64 * rank)(*array.shape), (ctypes.c_int64 * rank)(*strides))


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    source = Path(request["source"]).read_bytes()
    with ir.Context():
        module = ir.Module.parse(source.decode())
        wrapper = ir.Module.parse(probe(request["function"], request["inputs"], request["output"]))
        with ir.InsertionPoint(module.body):
            for op in wrapper.body.operations:
                if ir.StringAttr(op.attributes["sym_name"]).value == "__bf16_probe":
                    op.operation.clone()
        module.operation.verify()
        pipeline = "builtin.module(" + ",".join(PASSES) + ")"
        passmanager.PassManager.parse(pipeline).run(module.operation)
        Path(request["lowered"]).write_text(str(module))
        libm = ctypes.CDLL(ctypes.util.find_library("m"), mode=ctypes.RTLD_GLOBAL)
        engine = ExecutionEngine(module, opt_level=request["opt_level"], shared_libs=[])
        operands = [np.ascontiguousarray(np.load(spec["path"]).astype(np.uint16)) for spec in request["inputs"]]
        result = np.full(request["output"]["shape"], 0xA5A5, dtype=np.uint16)
        descriptors = [descriptor(array) for array in [*operands, result]]
        engine.invoke("__bf16_probe", *[ctypes.pointer(ctypes.pointer(d)) for d in descriptors])
        np.save(request["output"]["path"], result)
        del engine, libm
    print(
        json.dumps(
            {
                "source_sha256": hashlib.sha256(source).hexdigest(),
                "mlir_revision": importlib.metadata.version("torch-mlir"),
                "pipeline": pipeline,
                "opt_level": request["opt_level"],
            }
        )
    )


if __name__ == "__main__":
    main()
