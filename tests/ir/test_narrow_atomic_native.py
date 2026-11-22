"""Real byte/halfword atomics retain overflow and compare/exchange old words."""

import ctypes
import os
import subprocess

import pytest

from merlin.compiler.lower.model.codegen import build_host_shared
from merlin.compiler.lower.native.pipeline import lower_to_llvm_ir


@pytest.mark.skipif(not (os.environ.get("MERLIN_COMPILER_PYTHON") and os.environ.get("MERLIN_CLANG")),
                   reason="explicit native MLIR Python and host compiler required")
@pytest.mark.parametrize("bits", [8, 16])
def test_native_atomic_updates_and_both_cas_outcomes_keep_source_words(tmp_path, bits):
    word = f"i{bits}"
    entry = f"words_{bits}"
    source = f'''module {{
      func.func @{entry}(%raw:i64,%increment:i64,%replacement:i64)->i64 attributes {{llvm.emit_c_interface}} {{
        %seed=arith.trunci %raw:i64 to {word}
        %add=arith.trunci %increment:i64 to {word}
        %desired=arith.trunci %replacement:i64 to {word}
        %zero=arith.constant 0:index
        %one=arith.constant 1:{word}
        %slot=memref.alloca():memref<1x{word}>
        memref.store %seed,%slot[%zero]:memref<1x{word}>
        %memory=memref.cast %slot:memref<1x{word}> to memref<?x{word}>
        %old=memref.atomic_rmw addi %add,%memory[%zero]:({word},memref<?x{word}>)->{word}
        %expected=arith.addi %seed,%add:{word}
        %before=memref.generic_atomic_rmw %memory[%zero]:memref<?x{word}> {{
          ^bb0(%current:{word}):
            %equal=arith.cmpi eq,%current,%expected:{word}
            %updated=arith.select %equal,%desired,%current:{word}
            memref.atomic_yield %updated:{word}
        }}
        %wrong=arith.xori %desired,%one:{word}
        %failed=memref.generic_atomic_rmw %memory[%zero]:memref<?x{word}> {{
          ^bb0(%current:{word}):
            %equal=arith.cmpi eq,%current,%wrong:{word}
            %updated=arith.select %equal,%seed,%current:{word}
            memref.atomic_yield %updated:{word}
        }}
        %final=memref.load %slot[%zero]:memref<1x{word}>
        %w0=arith.extui %old:{word} to i64
        %w1=arith.extui %before:{word} to i64
        %w2=arith.extui %failed:{word} to i64
        %w3=arith.extui %final:{word} to i64
        %s1=arith.constant {bits}:i64
        %s2=arith.constant {2 * bits}:i64
        %s3=arith.constant {3 * bits}:i64
        %a=arith.shli %w1,%s1:i64
        %b=arith.shli %w2,%s2:i64
        %c=arith.shli %w3,%s3:i64
        %r0=arith.ori %w0,%a:i64
        %r1=arith.ori %r0,%b:i64
        %r2=arith.ori %r1,%c:i64
        return %r2:i64
      }}}}'''
    lowered = lower_to_llvm_ir(source, workdir=tmp_path / "lowering",
                               features=frozenset({"lower_integer_compare_exchange"}))
    path = tmp_path / f"{entry}.ll"
    path.write_text(lowered)
    library = ctypes.CDLL(str(build_host_shared(path, tmp_path / f"{entry}.so")))
    function = getattr(library, "_mlir_ciface_" + entry)
    function.argtypes, function.restype = [ctypes.c_uint64] * 3, ctypes.c_uint64
    mask, sign = (1 << bits) - 1, 1 << (bits - 1)
    for seed, increment, replacement in ((sign - 1, 1, mask), (sign, mask, 0),
                                         (0, mask, sign - 1), (mask, 1, sign)):
        result = function(seed, increment, replacement)
        actual = [(result >> (bits * i)) & mask for i in range(4)]
        assert actual == [seed, (seed + increment) & mask, replacement, replacement]


@pytest.mark.skipif(not (os.environ.get("MERLIN_COMPILER_PYTHON") and os.environ.get("MERLIN_CLANG")),
                   reason="explicit native MLIR Python and host compiler required")
@pytest.mark.parametrize("dtype,bits", [("f16", 16), ("bf16", 16), ("f8E4M3FN", 8)])
def test_float_exchange_transports_original_bits_without_float_calls(tmp_path, dtype, bits):
    word, entry = f"i{bits}", f"exchange_{dtype}"
    payload = {"f16": 0x7C11, "bf16": 0x7F81, "f8E4M3FN": 0x7F}[dtype]
    source = f'''module {{
      func.func @{entry}(%raw:i64,%next:i64)->i64 attributes {{llvm.emit_c_interface}} {{
        %a=arith.trunci %raw:i64 to {word}
        %b=arith.trunci %next:i64 to {word}
        %seed=arith.bitcast %a:{word} to {dtype}
        %value=arith.bitcast %b:{word} to {dtype}
        %zero=arith.constant 0:index
        %slot=memref.alloca():memref<1x{dtype}>
        memref.store %seed,%slot[%zero]:memref<1x{dtype}>
        %memory=memref.cast %slot:memref<1x{dtype}> to memref<?x{dtype}>
        %old=memref.atomic_rmw assign %value,%memory[%zero]:({dtype},memref<?x{dtype}>)->{dtype}
        %before=memref.generic_atomic_rmw %memory[%zero]:memref<?x{dtype}> {{
          ^bb0(%current:{dtype}):
            memref.atomic_yield %seed:{dtype}
        }}
        %constant_old=memref.generic_atomic_rmw %memory[%zero]:memref<?x{dtype}> {{
          ^bb0(%current:{dtype}):
            %constant=arith.constant 0x{payload:X}:{dtype}
            memref.atomic_yield %constant:{dtype}
        }}
        %final=memref.load %slot[%zero]:memref<1x{dtype}>
        %w0=arith.bitcast %old:{dtype} to {word}
        %w1=arith.bitcast %before:{dtype} to {word}
        %w2=arith.bitcast %constant_old:{dtype} to {word}
        %w3=arith.bitcast %final:{dtype} to {word}
        %r0=arith.extui %w0:{word} to i64
        %r1=arith.extui %w1:{word} to i64
        %r2=arith.extui %w2:{word} to i64
        %r3=arith.extui %w3:{word} to i64
        %s1=arith.constant {bits}:i64
        %s2=arith.constant {2 * bits}:i64
        %s3=arith.constant {3 * bits}:i64
        %b1=arith.shli %r1,%s1:i64
        %b2=arith.shli %r2,%s2:i64
        %b3=arith.shli %r3,%s3:i64
        %v0=arith.ori %r0,%b1:i64
        %v1=arith.ori %v0,%b2:i64
        %v2=arith.ori %v1,%b3:i64
        return %v2:i64
      }}}}'''
    features = {"lower_atomic_word_views"}
    if dtype == "bf16":
        features.add("preserve_bfloat_transport")
    source_path = tmp_path / f"{entry}.mlir"
    source_path.write_text(source)
    lowered = lower_to_llvm_ir(source, workdir=tmp_path / "lowering", features=frozenset(features))
    path = tmp_path / f"{entry}.ll"
    path.write_text(lowered)
    library = ctypes.CDLL(str(build_host_shared(path, tmp_path / f"{entry}.so")))
    function = getattr(library, "_mlir_ciface_" + entry)
    function.argtypes, function.restype = [ctypes.c_uint64] * 2, ctypes.c_uint64
    mask, sign = (1 << bits) - 1, 1 << (bits - 1)
    for seed, value in ((mask, sign), (1, 0), (sign, mask - 1), (0, mask)):
        result = function(seed, value)
        assert [(result >> (bits * i)) & mask for i in range(4)] == [seed, value, seed, payload]
    # A valid typed signature cannot grant borrowing to a different address calculation.
    check = '''import sys
from pathlib import Path
from torch_mlir import ir, passmanager
from merlin.compiler.lower.options.atomic_compare_exchange import SUPPORT
with ir.Context() as ctx, ir.Location.unknown():
    scope = {"ir": ir}
    exec(SUPPORT, scope)
    module = ir.Module.parse(scope["_bitwise_exchange_source"]("update", sys.argv[1]))
    pm = passmanager.PassManager.parse("builtin.module(canonicalize,cse)")
    pm.run(module.operation)
    function = tuple(module.body.operations)[0].operation
    assert scope["_word_helper_equal"](function, "update", module)
    addition = next(op for op in scope["_cas_walk"](function) if op.name == "arith.addi")
    addition.operands[0] = function.regions[0].blocks[0].arguments[1]
    module.operation.verify()
    pm.run(module.operation)
    assert not scope["_word_helper_equal"](function, "update", module)
    program = ir.Module.parse(Path(sys.argv[2]).read_text())
    atomic = next(op for op in scope["_cas_walk"](program.operation) if op.name == "memref.atomic_rmw")
    # Project the actual atomic for the xDSL reader, without unrelated native
    # printer properties on integer truncation in the executable test program.
    reader = ir.Module.create()
    inputs, results = [v.type for v in atomic.operands], [v.type for v in atomic.results]
    with ir.InsertionPoint(reader.body):
        function = ir.Operation.create("func.func", regions=1, attributes={
            "sym_name": ir.StringAttr.get("reader"),
            "function_type": ir.TypeAttr.get(ir.FunctionType.get(inputs, results))})
    block = ir.Block.create_at_start(function.regions[0], inputs)
    with ir.InsertionPoint(block):
        clone = atomic.clone()
        for i, argument in enumerate(block.arguments):
            clone.operands[i] = argument
        ir.Operation.create("func.return", operands=clone.results)
    reader.operation.verify()
    print(reader.operation.get_asm(print_generic_op_form=True))
'''
    checked = subprocess.run([os.environ["MERLIN_COMPILER_PYTHON"], "-c", check, dtype, str(source_path)],
                             capture_output=True, text=True, timeout=30)
    assert checked.returncode == 0, checked.stdout + checked.stderr
    from merlin.compiler.frontend.linalg_mlir import parse_mlir_text

    parse_mlir_text(checked.stdout).verify()
