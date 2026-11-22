"""Real byte/halfword atomics retain overflow and compare/exchange old words."""

import ctypes
import os

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
