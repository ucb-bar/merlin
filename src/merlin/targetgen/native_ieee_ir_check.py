"""Execute the actual integer conversion IR in a small native C checker.

This is proof tooling, never a compiler admission-time execution or a grader.
All floating SSA values carry raw bits; C has no floating operations here.
"""

import ctypes
import os
import subprocess
import tempfile
from pathlib import Path


def conversion_function(source, target, policy):
    from xdsl.dialects.builtin import bf16, f16, f32, f64
    from xdsl.ir import Block

    from merlin.llvmlower._ir import emitter
    from merlin.llvmlower.ieee_dyadic_ir import decode, encode
    from merlin.llvmlower.ieee_pointwise import special_conversion

    types = {"f16": f16, "bf16": bf16, "f32": f32, "f64": f64}
    block = Block(arg_types=[types[source]])

    emit = emitter(block)
    integer, exponent, negative = decode(block.args[0], emit)
    finite = encode(integer, exponent, types[target], emit, zero_negative=negative)
    result = special_conversion(
        "arith.extf" if types[source].bitwidth < types[target].bitwidth else "arith.truncf",
        block.args[0],
        types[target],
        emit,
        finite_fallback=finite,
    )
    values = {block.args[0]: "input[i]"}
    lines = [
        "#include <stdint.h>",
        "#include <stddef.h>",
        "typedef unsigned __int128 U;",
        "void check(const uint64_t *input, uint64_t *output, size_t n) {",
        "for (size_t i=0;i<n;i++) {",
    ]
    counter = [0]

    def width(typ):
        return typ.width.data if hasattr(typ, "width") else typ.bitwidth

    def literal(number):
        number &= (1 << 128) - 1
        return f"(((U)0x{number >> 64:x}ULL<<64)|0x{number & ((1 << 64) - 1):x}ULL)"

    def signed(name, bits):
        sign = literal(1 << (bits - 1))
        return f"((__int128)(({name}^{sign})-{sign}))"

    def region(body, destination=None):
        for op in body.ops:
            name = op.name
            args = [values[v] for v in op.operands]
            if name == "scf.yield":
                lines.append(f"{destination}={args[0]};")
                continue
            symbol = f"v{counter[0]}"
            counter[0] += 1
            typ = op.results[0].type
            bits = width(typ)
            if name == "scf.if":
                lines.append(f"U {symbol}; if ({args[0]}) {{")
                region(op.true_region.block, symbol)
                lines.append("} else {")
                region(op.false_region.block, symbol)
                lines.append("}")
                values[op.results[0]] = symbol
                continue
            if name == "arith.constant":
                expression = literal(int(op.value.value.data))
            elif name in {"arith.bitcast", "arith.extui", "arith.trunci"}:
                expression = args[0]
            elif name == "arith.extsi":
                expression = signed(args[0], width(op.operands[0].type))
            elif name == "arith.select":
                expression = f"({args[0]} ? {args[1]} : {args[2]})"
            elif name == "arith.cmpi":
                p = op.predicate.value.data
                a, b = args
                if p in (2, 3, 4, 5):
                    a, b = [signed(v, width(x.type)) for v, x in zip(args, op.operands)]
                operator = ("==", "!=", "<", "<=", ">", ">=", "<", "<=", ">", ">=")[p]
                expression = f"({a} {operator} {b})"
            else:
                operator = {
                    "arith.addi": "+",
                    "arith.subi": "-",
                    "arith.muli": "*",
                    "arith.andi": "&",
                    "arith.ori": "|",
                    "arith.xori": "^",
                    "arith.shli": "<<",
                    "arith.shrui": ">>",
                }.get(name)
                if operator is None:
                    raise ValueError("unproved checker IR operation " + name)
                expression = f"({args[0]} {operator} {args[1]})"
            lines.append(f"U {symbol}=((U)({expression})) & {literal((1 << bits) - 1)};")
            values[op.results[0]] = symbol

    region(block)
    from merlin.targetgen.native_ieee_certificate import FORMATS

    fraction, exponent, _ = FORMATS[source]
    mask = (1 << (fraction + exponent)) - 1
    infinity = ((1 << exponent) - 1) << fraction
    tf, te, _ = FORMATS[target]
    final = values[result]
    if policy in {"positive_all_payload", "positive_quiet", "all_ones"}:
        override = ((1 << te) - 1) << tf | ((1 << tf) - 1 if policy == "positive_all_payload" else 1 << (tf - 1))
        if policy == "all_ones":
            override = (1 << (tf + te + 1)) - 1
        final = f"((input[i]&{mask}ULL)>{infinity}ULL ? {override}ULL : {final})"
    lines.extend([f"output[i]=(uint64_t){final};", "}", "}"])
    temporary = tempfile.TemporaryDirectory(prefix="ieee-conversion-check-")
    root = Path(temporary.name)
    src, library = root / "check.c", root / "check.so"
    src.write_text("\n".join(lines) + "\n")
    subprocess.run([os.environ.get("CC", "cc"), "-O2", "-shared", "-fPIC", str(src), "-o", str(library)], check=True)
    handle = ctypes.CDLL(str(library))
    function = handle.check
    function.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]

    def run(words):
        import numpy as np

        inputs = np.asarray(words, dtype=np.uint64)
        outputs = np.empty(len(inputs), dtype=np.uint64)
        function(
            inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            len(inputs),
        )
        return outputs

    run.resources = temporary, handle
    run.source = src.read_text()
    return run
