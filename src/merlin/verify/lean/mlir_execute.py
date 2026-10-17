"""Independent native oracle for Lean fidelity tests, run in the compiler Python environment.

Only upstream MLIR passes run here: no Merlin parser, preprocessing, rewrite, or interpreter.
Inputs arrive at run time through i64 memrefs; a generated wrapper packs tensors and observes
result bits. Native execution is isolated in this subprocess, including JIT errors or crashes.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

from mlir_map_contract import probe as probe_map_contract
from torch_mlir import ir, passmanager
from torch_mlir.execution_engine import ExecutionEngine

# Deliberately no canonicalize/CSE/fusion before linalg-to-loops: preserve the source's reduction
# order and arithmetic. This also exercises runtime operations instead of folding constant probes.
PASSES = [
    "func.func(convert-elementwise-to-linalg)",
    "func.func(linalg-generalize-named-ops)",
    "one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map}",
    "func.func(convert-linalg-to-loops)",
    "convert-scf-to-cf",
    "expand-strided-metadata",
    "lower-affine",
    "convert-math-to-llvm",
    "convert-math-to-libm",
    "convert-index-to-llvm",
    "convert-arith-to-llvm",
    "finalize-memref-to-llvm",
    "convert-func-to-llvm",
    "convert-cf-to-llvm",
    "reconcile-unrealized-casts",
]


def _descriptor_type(index_bits):
    index = ctypes.c_int32 if index_bits == 32 else ctypes.c_int64

    class Memref(ctypes.Structure):
        _fields_ = [
            ("allocated", ctypes.POINTER(ctypes.c_uint64)),
            ("aligned", ctypes.POINTER(ctypes.c_uint64)),
            ("offset", index),
            ("sizes", index * 1),
            ("strides", index * 1),
        ]

    return Memref


def run(request: dict) -> dict:
    map_contract = probe_map_contract()
    if request.get("map_contract"):
        return map_contract
    source = Path(request["source"]).read_bytes()
    with ir.Context():
        module = ir.Module.parse(source.decode())
        if request.get("bufferize_only"):
            end = next(i for i, stage in enumerate(PASSES) if stage.startswith("one-shot-bufferize"))
            pipeline = "builtin.module(" + ",".join(PASSES[: end + 1]) + ")"
            passmanager.PassManager.parse(pipeline).run(module.operation)
            return {
                "mlir": module.operation.get_asm(print_generic_op_form=True),
                "pipeline": pipeline,
                "map_contract": map_contract,
            }
        if request.get("generalize"):
            passmanager.PassManager.parse("builtin.module(func.func(linalg-generalize-named-ops))").run(
                module.operation
            )
            return {"mlir": module.operation.get_asm(print_generic_op_form=True), "map_contract": map_contract}
        wrapper = ir.Module.parse(request["wrapper"])
        with ir.InsertionPoint(module.body):
            for op in wrapper.body.operations:
                if ir.StringAttr(op.attributes["sym_name"]).value == "__merlin_fidelity_probe":
                    op.operation.clone()
        module.operation.verify()
        # Bind index width to every conversion that consumes it, independently of Lean's Ctx.
        width = request["index_bits"]
        passes = [
            p + f"{{index-bitwidth={width}}}"
            if p
            in {
                "convert-index-to-llvm",
                "convert-arith-to-llvm",
                "finalize-memref-to-llvm",
                "convert-func-to-llvm",
                "convert-cf-to-llvm",
            }
            else p
            for p in PASSES
        ]
        pipeline = "builtin.module(" + ",".join(passes) + ")"
        passmanager.PassManager.parse(pipeline).run(module.operation)
        Path(request["lowered"]).write_text(str(module))
        if width != 64 and "@memrefCopy" in str(module):
            raise ValueError("upstream CRunnerUtils memrefCopy requires 64-bit descriptor indices")
        # Make libm's symbols visible to the JIT (sqrt, rounding, fma).
        libm = ctypes.CDLL(ctypes.util.find_library("m"), mode=ctypes.RTLD_GLOBAL)
        runner = request.get("runner_utils")
        engine = ExecutionEngine(module, opt_level=0, shared_libs=[runner] if runner else [])
        descriptor = _descriptor_type(width)
        results = []
        for sample in request["samples"]:
            words = [int(x) for arg in sample for x in arg]
            n_in, n_out = len(words), request["output_words"]
            inputs = (ctypes.c_uint64 * max(1, n_in))(*words)
            # Sentinels make an unwritten observation visible as a mismatch.
            outputs = (ctypes.c_uint64 * max(1, n_out))(*([0xA5A5A5A5A5A5A5A5] * max(1, n_out)))
            ins = descriptor(inputs, inputs, 0, (n_in,), (1,))
            outs = descriptor(outputs, outputs, 0, (n_out,), (1,))
            engine.invoke(
                "__merlin_fidelity_probe", ctypes.pointer(ctypes.pointer(ins)), ctypes.pointer(ctypes.pointer(outs))
            )
            results.append([str(outputs[i]) for i in range(n_out)])
        del engine, libm
        return {
            "results": results,
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "mlir_revision": importlib.metadata.version("torch-mlir"),
            "pipeline": pipeline,
            "map_contract": map_contract,
            "runner_utils_sha256": hashlib.sha256(Path(runner).read_bytes()).hexdigest() if runner else None,
        }


if __name__ == "__main__":
    print(json.dumps(run(json.loads(Path(sys.argv[1]).read_text()))))
