"""MLIR text builders shared by the IR and targetgen tests.

Most fixtures are one ``@forward`` function around identity-map ``linalg.generic`` ops. These
helpers write that boilerplate, so a test states only its types and scalar body.
"""

import operator
import re
import struct
from fractions import Fraction

import numpy as np

from merlin.frontends.linalg_mlir import parse_mlir_text


def element(typ):
    """Element type of ``tensor<...xT>`` (a scalar type is its own element)."""
    match = re.fullmatch(r"tensor<(?:[\d?]+x)*(.+)>", typ)
    return match.group(1) if match else typ


def generic(
    ins,
    result,
    body,
    *,
    yields="%v",
    name="%r",
    iterators=None,
    args="abcdefgh",
    maps=None,
    out=None,
    acc="%unused",
    attrs="",
):
    """``name = linalg.generic`` over ``ins`` ([(value, type)]) into ``out`` (default a fresh empty).

    Maps default to identities. Block arguments are ``%a, %b, ...`` for the inputs and ``acc``
    for the output; ``body`` is the scalar computation and ``yields`` its result (None: ``body``
    ends with its own yield).
    """
    rank = len(re.findall(r"[\d?]+x", result))
    dims = ",".join(f"d{i}" for i in range(rank))
    maps = ",".join(maps or [f"affine_map<({dims})->({dims})>"] * (len(ins) + 1))
    kinds = ",".join(f'"{kind}"' for kind in iterators or ["parallel"] * rank)
    init = "" if out else f"{name}_init=tensor.empty() : {result}\n"
    out, typ = out or f"{name}_init", element(result)
    params = "".join(f"%{arg}:{element(t)}," for arg, (_, t) in zip(args, ins))
    attrs = f" attrs={attrs}" if attrs else ""
    sources = f"ins({','.join(v for v, _ in ins)}:{','.join(t for _, t in ins)})" if ins else ""
    return f"""{init}{name}=linalg.generic {{indexing_maps=[{maps}],iterator_types=[{kinds}]}}
  {sources} outs({out}:{result}){attrs} {{
^bb0({params}{acc}:{typ}):
{body}
{f"linalg.yield {yields} : {typ}" if yields else ""}
}} -> {result}"""


def forward(params, result, body, *, ret="%r", attrs="", decls=""):
    """Parse ``module { decls func.func @forward(params) -> result attrs { body; return ret } }``."""
    types = result.removeprefix("(").removesuffix(")")
    attrs = f" attributes {attrs}" if attrs else ""
    return parse_mlir_text(
        f"module {{{decls}\nfunc.func @forward({params}) -> {result}{attrs} {{\n{body}\nreturn {ret} : {types}\n}} }}"
    )


def pointwise(inputs, result, body, *, prelude="", **kwargs):
    """``@forward(%x, %y, ...)`` over ``inputs`` types returning one identity-map generic of them."""
    ins = list(zip(("%x", "%y", "%z", "%w"), inputs))
    params = ",".join(f"{value}:{typ}" for value, typ in ins)
    return forward(params, result, prelude + "\n" + generic(ins, result, body, **kwargs))


def returned(module):
    """The first value the module's ``func.return`` returns."""
    return next(op for op in module.walk() if op.name == "func.return").operands[0]


def assert_unchanged(module, transform, expected=0):
    """A refusing ``transform`` reports ``expected`` and leaves the module text intact."""
    before = str(module)
    outcome = transform(module)
    assert outcome == expected, outcome
    assert str(module) == before, "refused transform changed the module"


_BINARY = {
    "arith.addi": operator.add,
    "arith.subi": operator.sub,
    "arith.muli": operator.mul,
    "arith.andi": operator.and_,
    "arith.ori": operator.or_,
    "arith.xori": operator.xor,
    "arith.shli": operator.lshift,
    "arith.shrui": operator.rshift,
    "arith.divui": operator.floordiv,
    "arith.remui": operator.mod,
}


def signed(word, typ):
    """Two's-complement value of an unsigned ``word`` of integer type ``typ``."""
    width = typ.width.data
    return word - (1 << width) if word >> (width - 1) & 1 else word


def float_value(word, typ):
    """The IEEE value stored in ``word`` of floating type ``typ``."""
    if str(typ) == "bf16":
        return struct.unpack("<f", (word << 16).to_bytes(4, "little"))[0]
    fmt = {"f16": "<e", "f32": "<f", "f64": "<d"}[str(typ)]
    return struct.unpack(fmt, word.to_bytes(struct.calcsize(fmt), "little"))[0]


def run_ops(ops, values):
    """Interpret scalar ``ops`` on unsigned storage words in ``values``; return what they yield.

    Integer results are reduced to their width, floating values are raw IEEE words.
    """
    from merlin.targetgen.exact_arithmetic import round_binary

    for op in ops:
        a = [values[v] for v in op.operands]
        name = op.op_name.data if op.name == "builtin.unregistered" else op.name
        typ = op.results[0].type if op.results else None
        if name in {"linalg.yield", "scf.yield"}:
            return a[0]
        if name in {"arith.shli", "arith.shrui", "arith.shrsi"}:
            assert 0 <= a[1] < op.operands[0].type.width.data, (name, a[1])
        if name == "cf.assert":
            assert a[0]
            continue
        if name == "arith.constant" and hasattr(op.value, "get_values"):
            value = np.array(list(op.value.get_values()), dtype=object).reshape(typ.get_shape())
        elif name == "arith.constant":
            value = op.value.value.data
            if str(typ) in {"f16", "bf16", "f32", "f64"}:
                value = int.from_bytes(typ.pack([value]), "little")
        elif name == "tensor.extract":
            value = a[0][tuple(a[1:])]
        elif name == "scf.if":
            value = run_ops(op.regions[0 if a[0] else 1].block.ops, values)
        elif name in {"arith.bitcast", "arith.trunci", "arith.extui", "arith.index_cast"}:
            value = a[0]
        elif name == "arith.extsi":
            value = signed(a[0], op.operands[0].type)
        elif name == "arith.select":
            value = a[1] if a[0] else a[2]
        elif name == "arith.cmpi":
            x, y = a
            predicate = op.predicate.value.data
            if 2 <= predicate <= 5:
                x, y = [signed(v, o.type) for v, o in zip(a, op.operands)]
            value = [x == y, x != y, x < y, x <= y, x > y, x >= y, x < y, x <= y, x > y, x >= y][predicate]
        elif name == "arith.cmpf":
            x, y = [float_value(v, o.type) for v, o in zip(a, op.operands)]
            value = {1: x == y, 2: x > y, 3: x >= y, 4: x < y, 5: x <= y}[op.predicate.value.data]
        elif name in {"arith.shrsi", "arith.maxsi", "arith.minsi"}:
            x = signed(a[0], op.operands[0].type)
            if name == "arith.shrsi":
                value = x >> a[1]
            else:
                value = (max if name == "arith.maxsi" else min)(x, signed(a[1], op.operands[1].type))
        elif name in {"llvm.call_intrinsic", "llvm.intr.ctlz"}:
            value = op.operands[0].type.width.data - a[0].bit_length()
        elif name == "arith.uitofp":
            value = int.from_bytes(typ.pack([round_binary(Fraction(a[0]))]), "little")
        elif name == "arith.mulf":
            x, y = [Fraction(float_value(v, o.type)) for v, o in zip(a, op.operands)]
            value = int.from_bytes(typ.pack([round_binary(x * y)]), "little")
        else:
            value = _BINARY[name](*a)
        values[op.results[0]] = int(value) & ((1 << typ.width.data) - 1) if hasattr(typ, "width") else value


def run_block(block, arguments):
    """Run a scalar region on its argument words; return its yielded word."""
    return run_ops(block.ops, dict(zip(block.args, arguments, strict=True)))


def limbs(module, input_bounds, bounds=None):
    """Lower ``module`` into ``test.limb`` integer components; there must be a plan and a valid module."""
    from merlin.llvmlower.integer_components import rewrite_integer_components

    plans = rewrite_integer_components(
        module, limb_attribute="test.limb", input_bounds=input_bounds, known_bounds=bounds
    )
    assert plans
    module.verify()
    return plans


def emitter(block):
    """An ``emit`` callback that appends each op to ``block`` and returns its result."""
    return lambda op: (block.add_op(op), op.results[0])[1]


def native_library(module, tmp_path):
    """Lower an xDSL module with the normal host pipeline, build it, and load its entry point."""
    import hashlib
    import subprocess

    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    from merlin.llvmlower.toolchain import clang
    from merlin.xdsl_dialects._common import text

    llvm = lower_to_llvm_ir(text(module, generic=True), workdir=tmp_path / "lower", vectorize=False)
    src = tmp_path / "model.ll"
    src.write_text(llvm)
    obj, lib = src.with_suffix(".o"), tmp_path / f"model_{hashlib.sha256(llvm.encode()).hexdigest()}.so"
    subprocess.run([str(clang()), "-O2", "-fPIC", "-c", str(src), "-o", str(obj)], check=True, capture_output=True)
    runtime = ["cc", "-fPIC", "-shared", str(obj), str(mlir_runtime_c()), "-lm", "-o", str(lib)]
    subprocess.run(runtime, check=True, capture_output=True)
    model = HostModel.load(str(lib))
    return lambda *arrays: model([(a.ctypes.data, a.shape) for a in arrays])
